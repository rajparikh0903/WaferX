# Root-Cause Intelligence Engine — Final Hybrid Build

This package is the merged/best-of build for the semiconductor root-cause module.

## What was kept / improved

- Kept the original repository's root-cause candidate engine: SHAP + deviation + PASS/FAIL statistical association + historical similarity + external anomaly evidence.
- Kept the constrained counterfactual / What-If engine.
- Replaced the weaker original XGBoost scorer with a LightGBM V2-style model.
- Fixed a weakness in the supplied V2 preprocessing: feature filtering is now learned from the training split only, not the whole dataset.
- Added native LightGBM SHAP contribution support.
- Added two explicit operating points:
  - `0.10` = high-sensitivity investigation/screening
  - `0.50` = conventional classification / accuracy benchmark
- The supplied V2 result (ROC-AUC 0.7466, PR-AUC 0.1913, recall 0.8125 at threshold 0.10) was independently reproduced from its archive. The final retrained, leak-free LightGBM improved the same random-stratified workflow further: ROC-AUC ≈ 0.7449 and PR-AUC ≈ 0.2563, with 0.9322 accuracy at threshold 0.50 and 0.625 recall at threshold 0.10.

## Important evaluation note

The original repository used a chronological split; the supplied V2 package used a random stratified split. These are different evaluation protocols. Do not compare the raw numbers as if they were the same test set.

The final package uses a **leak-free random stratified split for the hackathon model** and separately documents the original temporal benchmark. Accuracy is not used as the primary model-selection metric because SECOM is highly imbalanced; PR-AUC and failure recall matter for screening.

## Run

```bash
pip install -r requirements.txt
python train_final.py
```

Then start the API:

```bash
uvicorn src.inference.api:app --reload
```

## Main API capabilities

- `POST /predict`
- `POST /root-cause`
- `POST /what-if`
- `POST /analyze`
- `GET /model-info`
- `GET /feature-importance`

The root-cause API accepts an optional `anomaly_score` and `anomaly_label`, so your teammate's anomaly detector can be plugged in without changing the root-cause engine.

## Model output philosophy

Never call a SHAP-ranked feature a proven physical cause. The UI/API should call these **Root-Cause Candidates** or **Root-Cause Hypotheses**. SECOM's anonymous measurements do not provide physical sensor identities or causal ground truth.

What-If results are **model-based counterfactual recommendations** and require engineering validation before any real process change.
