# demo-notebooks

Self-contained notebooks backing posts and articles. One notebook per topic, each pulling its data
from public URLs and runnable end to end in Colab.

| Notebook | Topic | Colab |
|---|---|---|
| [catboost_rmsewithuncertainty_conformal](notebooks/catboost_rmsewithuncertainty_conformal.ipynb) | Turning the CatBoost `RMSEWithUncertainty` sigma into a calibrated 90% interval, and which held-out diagnostic predicts a lopsided one | [open](https://colab.research.google.com/github/firefly1248/demo-notebooks/blob/main/notebooks/catboost_rmsewithuncertainty_conformal.ipynb) |

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
