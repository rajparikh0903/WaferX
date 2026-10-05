# Split sensitivity (REAL SECOM)

SECOM's fail rate drifts over time, so a random split is optimistic. Same pipeline, two splits (test metrics):

## Temporal split (default, honest)

| Model               |   Precision |   Recall |    F1 |   ROC-AUC |   PR-AUC |
|:--------------------|------------:|---------:|------:|----------:|---------:|
| logistic_regression |       0.069 |    0.471 | 0.12  |     0.597 |    0.072 |
| random_forest       |       0.036 |    0.059 | 0.044 |     0.558 |    0.071 |
| xgboost             |       0.083 |    0.059 | 0.069 |     0.644 |    0.087 |
| lightgbm            |       0     |    0     | 0     |     0.66  |    0.081 |

Test FAIL prevalence: 0.054

## Stratified random split (optimistic)

| Model               |   Precision |   Recall |    F1 |   ROC-AUC |   PR-AUC |
|:--------------------|------------:|---------:|------:|----------:|---------:|
| logistic_regression |       0.16  |    0.381 | 0.225 |     0.743 |    0.167 |
| random_forest       |       0.357 |    0.238 | 0.286 |     0.777 |    0.236 |
| xgboost             |       0.2   |    0.476 | 0.282 |     0.741 |    0.247 |
| lightgbm            |       0.429 |    0.143 | 0.214 |     0.763 |    0.257 |

Test FAIL prevalence: 0.067