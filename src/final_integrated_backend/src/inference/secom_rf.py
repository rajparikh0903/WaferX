from pathlib import Path
import joblib
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "secom_rf"
    / "secom_rf_k100_exact.joblib"
)


class SECOMRFModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.artifact = joblib.load(model_path)

        self.dropbad = self.artifact["dropbad"]
        self.imputer = self.artifact["imputer"]
        self.robust_scaler = self.artifact["robust_scaler"]
        self.top100_indices = self.artifact["top100_indices"]
        self.standard_scaler = self.artifact["standard_scaler"]
        self.model = self.artifact["model"]

    def preprocess(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # DropBad: preserve the exact selected raw columns
        X = X[:, self.dropbad.keep_]

        # Median imputation
        X = self.imputer.transform(X)

        # Robust scaling
        X = self.robust_scaler.transform(X)

        # Exact training-time clipping
        X = np.clip(X, -10, 10)

        # Top 100 ANOVA features
        X = X[:, self.top100_indices]

        # Final StandardScaler
        X = self.standard_scaler.transform(X)

        return X

    def predict(self, X):
        Xp = self.preprocess(X)
        return self.model.predict(Xp)

    def predict_proba(self, X):
        Xp = self.preprocess(X)
        return self.model.predict_proba(Xp)

    def analyze(self, X):
        probs = self.predict_proba(X)
        pred = self.predict(X)

        classes = self.model.classes_

        # Locate class 1 instead of assuming column 1
        fail_idx = int(np.where(classes == 1)[0][0])

        failure_probability = float(probs[0, fail_idx])

        return {
            "predicted_class": int(pred[0]),
            "failure_probability": failure_probability,
            "classes": classes.tolist(),
        }


rf_model = SECOMRFModel()