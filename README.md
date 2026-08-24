# demo-notebooks

Self-contained notebooks backing posts and articles. One notebook per topic, each pulling its data
from public URLs and runnable end to end in Colab.

| Notebook | Topic | Colab |
|---|---|---|
| [catboost_rmsewithuncertainty_conformal](notebooks/catboost_rmsewithuncertainty_conformal.ipynb) | Turning the CatBoost `RMSEWithUncertainty` sigma into a calibrated 90% interval, and which held-out diagnostic predicts a lopsided one | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_rmsewithuncertainty_conformal.ipynb) |
| [catboost_sglb_knowledge_uncertainty](notebooks/catboost_sglb_knowledge_uncertainty.ipynb) | Checking CatBoost's SGLB knowledge uncertainty against the actual error on a known function | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_sglb_knowledge_uncertainty.ipynb) |

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
