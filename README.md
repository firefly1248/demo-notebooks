# demo-notebooks

Self-contained notebooks backing posts and articles. One notebook per topic, each pulling its data
from public URLs and runnable end to end in Colab.

| Notebook | Topic | Colab |
|---|---|---|
| [catboost_rmsewithuncertainty_conformal](notebooks/catboost_rmsewithuncertainty_conformal.ipynb) | Turning the CatBoost `RMSEWithUncertainty` sigma into a calibrated 90% interval, and which held-out diagnostic predicts a lopsided one | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_rmsewithuncertainty_conformal.ipynb) |
| [catboost_sglb_knowledge_uncertainty](notebooks/catboost_sglb_knowledge_uncertainty.ipynb) | Checking CatBoost's SGLB knowledge uncertainty against the actual error on a known function | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_sglb_knowledge_uncertainty.ipynb) |
| [panel_merf_gpboost_variance](notebooks/panel_merf_gpboost_variance.ipynb) | Where MERF and GPBoost get their noise variance from, and what their 90% interval actually covers | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/panel_merf_gpboost_variance.ipynb) |
| [catboost_quantile_multiquantile_cqr](notebooks/catboost_quantile_multiquantile_cqr.ipynb) | What a CatBoost quantile band actually covers, when `MultiQuantile` is enough and when it needs conformalizing | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_quantile_multiquantile_cqr.ipynb) |

## catboost_rmsewithuncertainty_conformal

Builds a 90% interval from the `RMSEWithUncertainty` head three ways (raw, sigma-normalized
conformal, CQR at nominal 5/95 and 10/90) on one synthetic and two public datasets, measuring
coverage, width and where the misses land.

It also tests which held-out diagnostic predicts a lopsided interval. Skewness of the normalized
residuals does not: on diamonds it ranges 0.38 to 5.58 across four hyperparameter settings while
the miss imbalance stays between 0.4 and 0.8 points. Bowley skew and the ratio of the 5th and 95th
percentiles do.

![coverage split](figures/catboost_rmsewithuncertainty_conformal.png)

About five minutes on a free Colab CPU.

## catboost_sglb_knowledge_uncertainty

`posterior_sampling=True` plus `virtual_ensembles_predict` reports knowledge uncertainty from a
single model. This notebook measures that signal against the truth: the data comes from a known and
bounded 2D function, so every test point has both a reported uncertainty and a real error, and no
region is intrinsically harder to predict than another.

Both uncertainty columns are variances, so the notebook square-roots them before comparing them
with an error in the units of the target. In a gap cut out of the training data the model is 8.2x
more wrong than on covered data while the reported uncertainty is 1.1x. Past the edge of the range
the error is 20x, the signal is one constant across all 800 test points, and the total uncertainty
comes to 0.66x, below the covered level on all ten runs. A nearest-neighbour distance to the
training set ranks the error at 0.78 against 0.00 for knowledge uncertainty.

![knowledge uncertainty against error](figures/catboost_sglb_knowledge_uncertainty.png)

About a minute on a free Colab CPU.

## panel_merf_gpboost_variance

MERF and GPBoost both fit a nonlinear fixed part with trees and a random intercept per unit, and
both return a variance decomposition. Both also estimate the noise variance from residuals on rows
the fixed part was trained on, so the number tracks how hard the trees are fitted. On a synthetic
panel where the group variance and the noise variance are both 1.0, the group variance comes back
at 0.92 while the noise variance reads 0.58 for GPBoost at the setting a validation split picks,
and 0.71, 0.38 and 0.054 for MERF at 50, 100 and 400 base learner trees.

GPBoost's own 90% interval inherits that, covering 73% of held-out rows on seen units. Across the
20 settings in the tuning grid its coverage runs from 7% to 87%, and the 87% comes from underfitted
settings that the validation split rejects. Algorithm 2 of Sigrist (2022), which estimates the covariance parameters
from held-out residuals, brings coverage to 90% at the same accuracy.

![reported variance and actual coverage](figures/panel_merf_gpboost_variance.png)

About half an hour on a laptop CPU.

## catboost_quantile_multiquantile_cqr

The same 90% band built three ways — two separate `Quantile` models, `MultiQuantile` on the outer
pair, and `MultiQuantile` on ten levels — across the three datasets from the conformalized quantile
regression paper, at three hyperparameter settings and eight seeds.

None of the 27 seed-averaged raw bands reached the level it claimed; they run from 55.6% to 89.8%,
and how hard the trees are fitted moves coverage more than which of the three arms produced the
band. Conformalizing centres every dataset on 90% and holds there across a training sweep where the
raw band climbs from 61% at a 500-row pool to 86% at 36,584.

Asking one model for ten levels ran about 3x faster than fitting ten and crossed far less often, on
4% to 74% of rows against 24% to 97%. It is still not the best answer to crossing: sorting each row
of predictions removes crossings outright and lands the individual levels closer in all nine
settings.

![raw and conformalized coverage against training size](figures/catboost_quantile_multiquantile_cqr.png)

About fifteen minutes on a laptop CPU.
