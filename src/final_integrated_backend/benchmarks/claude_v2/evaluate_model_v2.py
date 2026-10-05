from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_secom_v2.joblib"


# Load model
model = joblib.load(MODEL_PATH)

# Load unseen test data
X_test = pd.read_parquet(DATA_DIR / "X_test_imputed.parquet")
y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["target"]


print(f"Test data: {X_test.shape}")
print(f"Test FAIL cases: {y_test.sum()}")
print(f"Test PASS cases: {(y_test == 0).sum()}")


# Predict probability of FAIL
y_probability = model.predict_proba(X_test)[:, 1]

# Default classification threshold
threshold = 0.10

y_prediction = (y_probability >= threshold).astype(int)


# Metrics
roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_prediction
)

cm = confusion_matrix(y_test, y_prediction)


print("\nMODEL EVALUATION")
print("=" * 40)

print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"PR-AUC:            {pr_auc:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

print("\nConfusion Matrix")
print("--------------------------------")
print(cm)

print("\nClassification Report")
print("--------------------------------")
print(
    classification_report(
        y_test,
        y_prediction,
        target_names=["PASS", "FAIL"],
        zero_division=0
    )
)


# Save predictions for later analysis
results = pd.DataFrame({
    "actual": y_test.values,
    "failure_probability": y_probability,
    "predicted": y_prediction
})

results.to_csv(
    DATA_DIR / "test_predictions.csv",
    index=False
)

print("\nSaved:")
print(DATA_DIR / "test_predictions.csv")