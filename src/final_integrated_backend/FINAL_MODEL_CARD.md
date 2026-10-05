# Final Model Card

## Primary model
LightGBM V2-style binary classifier with train-only feature filtering and median imputation.

## Inputs
UCI SECOM 590 process measurements + labels/timestamps for context.

## Outputs
- Failure probability
- High-sensitivity screening decision (threshold 0.10)
- Conventional classification decision (threshold 0.50)
- SHAP local explanation
- Root-cause candidate ranking
- Constrained counterfactual / What-If scenarios
- Optional external anomaly evidence

## Operating points
### Screening
Threshold 0.10:
- Test recall: 62.50%
- Test precision: 23.26%
- Test F1: 33.90%
- Test accuracy: 83.47%

### Conventional classification
Threshold 0.50:
- Test accuracy: 93.22%
- Test recall: 6.25%
- Test precision: 50.00%
- Test F1: 11.11%

## Discrimination
- Test ROC-AUC: 0.7449
- Test PR-AUC: 0.2563

## Root-cause caveat
SECOM provides process measurements and PASS/FAIL labels, not ground-truth physical causal labels. Root-cause outputs are hypotheses/candidates supported by SHAP, deviation, statistical association, historical similarity, and optional anomaly evidence.

## Counterfactual caveat
What-If recommendations are model-based predictions and must not be treated as physically validated process changes. Real equipment changes require engineering approval and safety constraints.
