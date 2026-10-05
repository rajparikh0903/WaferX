# Final Root-Cause Model Evaluation

## Final model
LightGBM V2-style with leak-free train-only feature filtering and median imputation.

- Test ROC-AUC: **0.7449**
- Test PR-AUC: **0.2563**
- Test accuracy @ classification threshold 0.50: **0.9322**
- Test recall @ screening threshold 0.10: **0.6250**
- Test precision @ screening threshold 0.10: **0.2326**
- Test F1 @ screening threshold 0.10: **0.3390**

## Why two thresholds?
The 0.10 threshold is the high-sensitivity investigation/screening operating point. The 0.50 threshold is the conventional classification operating point used for the accuracy comparison. These are intentionally separated because accuracy is dominated by PASS samples in highly imbalanced SECOM data.

## Original repository benchmark
The original repository used a chronological split with XGBoost: test accuracy at its reported threshold was about 0.914, ROC-AUC 0.644, and PR-AUC 0.087.
The final package preserves the root-cause/counterfactual architecture while upgrading the scorer and making train-only preprocessing explicit.

## Causality note
SECOM does not contain ground-truth causal labels. Root-cause outputs are candidate hypotheses supported by SHAP, deviation, PASS-vs-FAIL association, historical similarity, and optional external anomaly evidence. What-if results are model-based counterfactuals, not physically validated process corrections.