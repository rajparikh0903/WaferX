# SYNTHETIC ROOT-CAUSE VALIDATION (generated data - not SECOM)

8000 samples, 40 features, injected causes ['Feature 4', 'Feature 12', 'Feature 28']; 85 test failures evaluated; failure-model test PR-AUC 0.479.

| ranking method | Top-1 | Top-3 | MRR |
|---|---|---|---|
| full_root_cause_score | 0.953 | 1.000 | 0.976 |
| shap_only | 0.953 | 1.000 | 0.976 |
| deviation_only | 0.553 | 0.894 | 0.730 |

Random-guess Top-1 would be about 0.075.

Anomaly evidence was not supplied here (works without it). Results validate the ranking logic on a problem where the answer is known; they say nothing about SECOM root-cause accuracy.