---
name: yield-investigation
description: Investigate semiconductor yield issues using live YieldTwin process, anomaly, root-cause, and counterfactual APIs.
---

When investigating a YieldTwin manufacturing event:

1. Identify the sample ID.
2. Call analyze_sample.
3. Call get_anomaly.
4. Call get_root_causes.
5. Explain the highest-ranked candidates using the returned evidence.
6. If a parameter change is requested, call run_what_if.
7. Use get_feature_importance when a global feature ranking is needed.
8. Never invent sensor values, SHAP values, anomaly scores, or model outputs.
9. Clearly distinguish model-based root-cause hypotheses from physically proven causes.
10. State that physical process changes require engineering validation.