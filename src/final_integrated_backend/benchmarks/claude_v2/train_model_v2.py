from pathlib import Path

import joblib
import pandas as pd
import lightgbm as lgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load data
X_train = pd.read_parquet(DATA_DIR / "X_train_imputed.parquet")
X_val = pd.read_parquet(DATA_DIR / "X_val_imputed.parquet")

y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["target"]
y_val = pd.read_parquet(DATA_DIR / "y_val.parquet")["target"]


print(f"Training data:   {X_train.shape}")
print(f"Validation data: {X_val.shape}")

print(f"\nTraining FAIL cases:   {y_train.sum()}")
print(f"Training PASS cases:   {(y_train == 0).sum()}")

print(f"Validation FAIL cases: {y_val.sum()}")
print(f"Validation PASS cases: {(y_val == 0).sum()}")


# Handle class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"\nScale positive weight: {scale_pos_weight:.2f}")


# More conservative LightGBM model
model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=1000,
    learning_rate=0.02,

    num_leaves=15,
    max_depth=5,

    min_child_samples=30,
    min_split_gain=0.01,

    reg_alpha=0.5,
    reg_lambda=1.0,

    subsample=0.8,
    subsample_freq=1,

    colsample_bytree=0.7,

    scale_pos_weight=scale_pos_weight,

    random_state=42,
    n_jobs=-1
)


print("\nTraining LightGBM V2...")
print("--------------------------------")


model.fit(
    X_train,
    y_train,
    eval_X=X_val,
    eval_y=y_val,
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=75,
            verbose=True
        )
    ]
)


# Save V2 separately
model_path = MODEL_DIR / "lightgbm_secom_v2.joblib"

joblib.dump(model, model_path)


print("\nModel V2 training complete!")
print("--------------------------------")
print(f"Best iteration: {model.best_iteration_}")
print(f"Model saved to:")
print(model_path)