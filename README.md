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
single model. This notebook measures that signal against the truth: the data comes from a known 2D
function, so every test point has both a reported uncertainty and a real error.

In a gap cut out of the training data the model is 21x more wrong than on covered data while the
signal rises 1.8x. Eight units past the edge of the range the model is 95x more wrong, and on 4 of
10 seeds the signal stays under 1.5x. Ten independently trained models track the gap closely and
are no more dependable past the edge.

![knowledge uncertainty against error](figures/catboost_sglb_knowledge_uncertainty.png)

About two minutes on a free Colab CPU.
