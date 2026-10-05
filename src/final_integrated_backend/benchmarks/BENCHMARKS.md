# Model Benchmark Ledger

## Original repository
- Model: XGBoost
- Split: temporal
- Test ROC-AUC: 0.6439
- Test PR-AUC: 0.0870
- Test accuracy at reported threshold 0.1597: 0.9140
- FAIL recall: 0.0588
- FAIL precision: 0.0833

## Supplied Claude V2
- Model: LightGBM V2
- Split: random stratified
- Test ROC-AUC: 0.7466
- Test PR-AUC: 0.1913
- At threshold 0.10: accuracy 0.5847, FAIL recall 0.8125, precision 0.1204, F1 0.2097
- The exact Claude artifact and scripts are preserved here for traceability.

## Final hybrid
- Model: LightGBM V2-style
- Split: random stratified
- Feature filtering: training split only
- Imputation: training split only
- Test ROC-AUC: 0.7449
- Test PR-AUC: 0.2563
- At threshold 0.10: accuracy 0.8347, FAIL recall 0.6250, precision 0.2326, F1 0.3390
- At threshold 0.50: accuracy 0.9322, FAIL recall 0.0625, precision 0.5000, F1 0.1111

The final build keeps the original root-cause candidate, evidence, similarity, and constrained counterfactual layers, while making the LightGBM integration native and leak-free.
