"""Experiments behind notebooks/model_comparison/model_comparison_run_noise.ipynb.

    nice -n 10 python scripts/run_noise_experiment.py determinism
    nice -n 10 python scripts/run_noise_experiment.py time
    nice -n 10 python scripts/run_noise_experiment.py a
    nice -n 10 python scripts/run_noise_experiment.py b
    python scripts/run_noise_experiment.py collect

Every task writes one parquet under RAW and is skipped when that file exists, so an
interrupted run resumes where it stopped. `collect` merges the task files into the few
small files the notebook reads. Every fit uses one thread; WORKERS fits run side by side.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"  # set before numpy loads, and inherited by the worker processes

import io
import json
import platform
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.io import arff
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # convergence and dtype chatter; workers re-import this module

OUT = Path(__file__).resolve().parents[1] / "results" / "model_comparison" / "run_noise"
RAW = OUT / "raw"
WORKERS = 7  # half of the 14 cores

# SmallDataBenchmarks, pinned to the commit the decision notebook reads
SDB_COMMIT = "0227b621e4cfbd4eed32797b9054484defed794f"
SDB_URL = f"https://raw.githubusercontent.com/firefly1248/SmallDataBenchmarks/{SDB_COMMIT}/"
SDB_MAX_ROWS = 10_000  # the benchmark's own cap, drawn with default_rng(0)

# binary OpenML datasets with at least 30 000 rows, pinned by ID; nomao (1486) is out, its
# pilot accuracy of 0.959 is too close to 1
LARGE = {"adult": 1590, "electricity": 151, "bank-marketing": 1461, "credit-default": 42477,
         "higgs": 23512, "covertype": 44121, "jannis": 44131, "miniboone": 41150, "road-safety": 45038}
# experiment B, slowest first; miniboone is left out of B for the time budget
B_DATASETS = ("jannis", "higgs", "credit-default", "road-safety", "bank-marketing", "covertype",
              "electricity", "adult")

K_SEEDS = 40
A_TEST_SDB = 1000          # or half the rows, whichever is smaller
A_TEST_POOL = 5000         # large datasets: 25 blocks of 200, 5 of 1000, 1 of 5000
N_TRAIN = 5000             # large datasets, experiments A and B

B_H = 20_000
B_V = 1000                 # V, V2 and V3 each
B_TRIALS = 200
B_ITERATIONS = 300
B_PREFIXES = (1, 3, 7, 15, 30, 50, 100, 200)
B_TOP, B_RERUN_SEEDS = 5, 5
B_STUDIES = 10             # R, per dataset and sampler


CB = {"thread_count": 1, "verbose": 0, "allow_writing_files": False}


def model(family, seed):
    if family == "catboost":
        return CatBoostClassifier(random_seed=seed, **CB)
    if family == "catboost_sub":
        return CatBoostClassifier(random_seed=seed, **CB, bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8)
    if family == "lightgbm_bag":
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, n_jobs=1, random_state=seed, deterministic=True,
                              force_row_wise=True, verbose=-1)
    if family == "mlp":  # exactly 100 epochs: tol=0 and a patience longer than the run
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPClassifier((64, 64), max_iter=100, tol=0, n_iter_no_change=1000, random_state=seed))
    raise ValueError(family)


FAMILIES = ("catboost", "catboost_sub", "lightgbm_bag", "mlp")


def search_model(params, seed):
    return CatBoostClassifier(**params, iterations=B_ITERATIONS, bootstrap_type="Bayesian",
                              random_seed=seed, **CB)


def predict(m, X):
    return m.classes_[m.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------- data

def sdb_names():
    path = RAW / "data" / "sdb_names.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        names = pd.read_csv(SDB_URL + "results/catboost_retune_calibration.csv").dataset
        path.write_text("\n".join(names))
    return path.read_text().split()


def load_sdb(name):
    path = RAW / "data" / "sdb" / f"{name}.arff"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_bytes(urllib.request.urlopen(SDB_URL + f"datasets/{name}.arff").read())
        path.with_suffix(".tmp").rename(path)
    df = pd.DataFrame(arff.loadarff(io.StringIO(path.read_text()))[0])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    if len(df) > SDB_MAX_ROWS:
        df = df.iloc[np.random.default_rng(0).choice(len(df), SDB_MAX_ROWS, replace=False)]
    y = df["Class"].map({v: i for i, v in enumerate(sorted(df["Class"].unique()))}).to_numpy()
    return pd.get_dummies(df.drop(columns="Class")).to_numpy(float), y


def load_large(name):
    b = fetch_openml(data_id=LARGE[name], as_frame=True, data_home=str(RAW / "data" / "openml"), parser="auto")
    y = pd.factorize(b.target.astype(str), sort=True)[0]
    return pd.get_dummies(b.data).to_numpy(float), y


def prefetch():
    """Download everything once, in this process, before the workers start reading."""
    for name in sdb_names():
        load_sdb(name)
    for name in LARGE:
        load_large(name)


def load(source, name):
    return load_sdb(name) if source == "sdb" else load_large(name)


def a_split(source, name, n):
    """Experiment A: test rows first, in random order, then the training rows."""
    perm = np.random.default_rng(hash_name(name)).permutation(n)
    if source == "sdb":
        m = min(A_TEST_SDB, n // 2)
        return perm[:m], perm[m:]
    return perm[:A_TEST_POOL], perm[A_TEST_POOL:A_TEST_POOL + N_TRAIN]


def hash_name(name):
    return int.from_bytes(name.encode(), "little") % (2 ** 32)


# ---------------------------------------------------------------- experiment A

def a_task(source, name, family, seeds=range(K_SEEDS)):
    X, y = load(source, name)
    test, train = a_split(source, name, len(y))
    cols, secs = {}, []
    for s in seeds:
        t = time.perf_counter()
        m = model(family, s).fit(X[train], y[train])
        cols[f"c{s:02d}"] = predict(m, X[test]) == y[test]
        secs.append(time.perf_counter() - t)
    frame = pd.DataFrame({"row": np.arange(len(test)), **cols})
    return frame, {"n": len(y), "n_train": len(train), "n_test": len(test), "p": X.shape[1],
                   "classes": int(y.max() + 1), "fit_seconds": secs}


def run_a_task(source, name, family):
    path = RAW / "a" / f"{source}__{name}__{family}.parquet"
    if path.exists():
        return path
    frame, info = a_task(source, name, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".tmp"), compression="zstd")
    path.with_suffix(".json").write_text(json.dumps(info))
    path.with_suffix(".tmp").rename(path)
    return path


# ---------------------------------------------------------------- experiment B

def b_split(name, n, study):
    """H is fixed per dataset; train, V, V2 and V3 are redrawn from the pool for every study.
    Both samplers get the same draw for the same study number."""
    perm = np.random.default_rng(hash_name(name) + 1).permutation(n)
    H, pool = perm[:B_H], perm[B_H:]
    pool = np.random.default_rng([hash_name(name), study]).permutation(pool)
    train, V, V2, V3, _ = np.split(pool, np.cumsum([N_TRAIN, B_V, B_V, B_V]))
    return train, V, V2, V3, H


def suggest(trial):
    return {"depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5)}


SAMPLERS = {"tpe": optuna.samplers.TPESampler, "random": optuna.samplers.RandomSampler}


def top_by_v(trials, T):
    """Trial numbers of the B_TOP best by V among the first T, earlier trial first on ties."""
    head = trials.iloc[:T]
    return head.sort_values(["acc_v", "number"], ascending=[False, True]).number.iloc[:B_TOP].tolist()


def b_task(name, sampler, study):
    X, y = load_large(name)
    train, V, V2, V3, H = b_split(name, len(y), study)
    k = list(SAMPLERS).index(sampler)
    rows, configs = [], {}

    def objective(trial):
        params = configs[trial.number] = suggest(trial)
        seed = 1_000_000 * k + 1000 * study + trial.number
        t = time.perf_counter()
        m = search_model(params, seed).fit(X[train], y[train])
        acc_v, acc_h = (np.mean(predict(m, X[idx]) == y[idx]) for idx in (V, H))
        rows.append({"number": trial.number, **params, "seed": seed, "acc_v": acc_v, "acc_h": acc_h,
                     "fit_seconds": time.perf_counter() - t})
        return acc_v

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.create_study(direction="maximize", sampler=SAMPLERS[sampler](seed=study)).optimize(objective, n_trials=B_TRIALS)
    trials = pd.DataFrame(rows)

    # re-run the top five of every prefix that has five trials, five fresh seeds each
    rerun_numbers = sorted({n for T in B_PREFIXES if T >= B_TOP for n in top_by_v(trials, T)})
    reruns = []
    for n in rerun_numbers:
        for j in range(B_RERUN_SEEDS):
            seed = 5_000_000 + 1_000_000 * k + 10_000 * study + 10 * n + j
            m = search_model(configs[n], seed).fit(X[train], y[train])
            reruns.append({"number": n, "seed": seed, **{f"acc_{k}": np.mean(predict(m, X[idx]) == y[idx])
                                                         for k, idx in (("v2", V2), ("v3", V3), ("h", H))}})
    return trials, pd.DataFrame(reruns)


def run_b_task(name, sampler, study):
    path = RAW / "b" / f"{name}__{sampler}__{study:02d}.parquet"
    if path.exists():
        return path
    trials, reruns = b_task(name, sampler, study)
    path.parent.mkdir(parents=True, exist_ok=True)
    reruns.to_parquet(path.with_name(path.stem + "__reruns.parquet"), compression="zstd")
    trials.to_parquet(path.with_suffix(".tmp"), compression="zstd")
    path.with_suffix(".tmp").rename(path)
    return path


# ---------------------------------------------------------------- driver

def run_pool(fn, tasks):
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(fn, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            print(f"[{i}/{len(tasks)}] {futures[f]} {time.time() - t0:.0f}s", flush=True)


def versions():
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__, "lightgbm": lightgbm.__version__, "optuna": optuna.__version__}


def determinism_task(source, name, family, seed):
    """Two fits with one seed, inside one process: bit-identical probabilities expected."""
    X, y = load(source, name)
    test, train = a_split(source, name, len(y))
    return [model(family, s).fit(X[train], y[train]).predict_proba(X[test]) for s in (seed, seed, seed + 1)]


def determinism():
    """Same seed in two separate worker processes and twice within one: identical bytes?"""
    cases = [(s, n, f) for s, n in (("sdb", "spambase"), ("sdb", "wine-quality-5class"), ("openml", "adult"))
             for f in FAMILIES]
    out = []
    with ProcessPoolExecutor(WORKERS) as ex:
        runs = {c: [ex.submit(determinism_task, *c, 0) for _ in range(2)] for c in cases}
        for c, fs in runs.items():
            (a, a2, other), (b, _, _) = (f.result() for f in fs)
            out.append({"source": c[0], "dataset": c[1], "family": c[2],
                        "same_process_identical": bool(np.array_equal(a, a2)),
                        "two_processes_identical": bool(np.array_equal(a, b)),
                        "max_abs_diff_two_processes": float(np.abs(a - b).max()),
                        "other_seed_rows_changed": float(np.mean(a.argmax(1) != other.argmax(1)))})
            print(out[-1], flush=True)
    (OUT / "determinism.json").write_text(json.dumps({"versions": versions(), "cases": out}, indent=1))


def a_tasks():
    return ([("sdb", n, f) for n in sdb_names() for f in FAMILIES] +
            [("openml", n, f) for n in LARGE for f in FAMILIES])


def timing():
    """One seed of every A task and one search fit per depth: the budget for the design cell."""
    rows = []
    with ProcessPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(_one_seed, *t) for t in a_tasks()]
        futures += [ex.submit(_b_timing, n, d) for n in LARGE for d in range(4, 11)]
        for f in as_completed(futures):
            rows.append(f.result())
    pd.DataFrame(rows).to_csv(RAW / "timing.csv", index=False)


def _one_seed(source, name, family):
    frame, info = a_task(source, name, family, seeds=[0])
    return {"source": source, "dataset": name, "family": family, "acc": frame.c00.mean(),
            **{k: v for k, v in info.items() if k != "fit_seconds"}, "seconds": info["fit_seconds"][0]}


def _b_timing(name, depth):
    X, y = load_large(name)
    train, V, V2, V3, H = b_split(name, len(y), 0)
    t = time.perf_counter()
    m = search_model({"depth": depth, "learning_rate": 0.05}, 0).fit(X[train], y[train])
    predict(m, X[H])
    return {"source": "b", "dataset": name, "family": f"depth{depth}", "seconds": time.perf_counter() - t}


def collect():
    a, a_info = [], []
    for path in sorted((RAW / "a").glob("*.parquet")):
        source, name, family = path.stem.split("__")
        a.append(pd.read_parquet(path).assign(source=source, dataset=name, family=family))
        info = json.loads(path.with_suffix(".json").read_text())
        secs = info.pop("fit_seconds")
        a_info.append({"source": source, "dataset": name, "family": family, **info,
                       "fits": len(secs), "fit_seconds": sum(secs)})
    a = pd.concat(a, ignore_index=True)
    a[["source", "dataset", "family"]] = a[["source", "dataset", "family"]].astype("category")
    a.to_parquet(OUT / "a_correct.parquet", compression="zstd", index=False)

    trials, reruns = [], []
    for path in sorted((RAW / "b").glob("*.parquet")):
        if path.stem.endswith("__reruns"):
            continue
        name, sampler, study = path.stem.split("__")
        key = {"dataset": name, "sampler": sampler, "study": int(study)}
        trials.append(pd.read_parquet(path).assign(**key))
        reruns.append(pd.read_parquet(path.with_name(path.stem + "__reruns.parquet")).assign(**key))
    if trials:
        pd.concat(trials, ignore_index=True).to_parquet(OUT / "b_trials.parquet", compression="zstd", index=False)
        pd.concat(reruns, ignore_index=True).to_parquet(OUT / "b_reruns.parquet", compression="zstd", index=False)

    pd.DataFrame(a_info).to_csv(OUT / "a_info.csv", index=False)
    meta = {"versions": versions(), "sdb_commit": SDB_COMMIT, "openml_ids": LARGE, "b_datasets": B_DATASETS,
            "constants": {k: globals()[k] for k in ("K_SEEDS", "A_TEST_SDB", "A_TEST_POOL", "N_TRAIN", "B_H", "B_V",
                                                    "B_TRIALS", "B_ITERATIONS", "B_PREFIXES", "B_TOP",
                                                    "B_RERUN_SEEDS", "B_STUDIES", "WORKERS")}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode in ("determinism", "time", "a", "b"):
        prefetch()
    if mode == "determinism":
        determinism()
    elif mode == "time":
        timing()
    elif mode == "a":
        # slowest first, from the pilot timings, so the long tasks do not finish last
        pilot = pd.read_csv(RAW / "timing.csv").set_index(["source", "dataset", "family"]).seconds
        run_pool(run_a_task, sorted(a_tasks(), key=lambda t: -pilot.get(t, 0)))
    elif mode == "b":
        run_pool(run_b_task, [(n, s, r) for n in B_DATASETS for r in range(B_STUDIES) for s in SAMPLERS])
    elif mode == "collect":
        collect()
