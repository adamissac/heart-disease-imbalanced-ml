# Beyond Accuracy: Heart Disease Classification on Imbalanced Survey Data

Predicting self-reported heart disease from behavioral, medical, and demographic
features in the CDC BRFSS 2020 survey — and a case study in why **accuracy is the
wrong headline metric** when the classes are imbalanced.

## TL;DR

Four standard classifiers (Logistic Regression, KNN, Random Forest, SVM) all reach
~89% accuracy on this data. But:

- **None of them beats a model that predicts "no disease" for everyone** (the test
  set is 89.3% negative, so that trivial baseline also scores 89.3%).
- At default settings they detect only **1.5%–15.2%** of actual heart disease cases.

Two standard imbalance fixes recover most of the lost detection:

| Fix | Model | Sensitivity (before → after) |
|---|---|---|
| Class weighting | Logistic Regression | 13.6% → **77.3%** |
| Class weighting | SVM (RBF) | 1.5% → **62.1%** |
| Threshold tuning (0.50 → 0.228) | Random Forest | 9.1% → **65.2%** |

The Random Forest's ROC-AUC was **0.79** the whole time — the predictive signal was
there; the default 0.5 decision threshold was hiding it.

