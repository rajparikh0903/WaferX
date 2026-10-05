import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.synthetic_validation import make_synthetic  # noqa: E402
from src.explainability.shap_engine import ShapEngine  # noqa: E402
from src.inference.pipeline import AnalysisEngine, build_reference_bundle  # noqa: E402
from src.utils import load_config  # noqa: E402

ARTEFACTS = ROOT / "models" / "metadata" / "metadata.json"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth(cfg):
    """Small synthetic problem + trained model + full analysis engine (no SECOM artefacts needed)."""
    X, y, Z, names = make_synthetic(4000, 20, [2, 7, 13], seed=3)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[:3000], idx[3000:]
    model = xgb.XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, scale_pos_weight=4,
                              tree_method="hist", random_state=0, n_jobs=2).fit(X.iloc[tr], y[tr])
    se = ShapEngine(model, names)
    gi = se.global_importance(X.iloc[tr])
    oof = model.predict_proba(X.iloc[tr])[:, 1]
    bundle = build_reference_bundle(X.iloc[tr], X.iloc[tr], y[tr], names, gi, oof, 0.4, None, cfg)
    eng = AnalysisEngine(model, names, 0.4, se, bundle, cfg)
    fails = [i for i in te if y[i] == 1]
    return {"X": X, "y": y, "tr": tr, "te": te, "model": model, "names": names, "shap": se, "engine": eng,
            "bundle": bundle, "fail_rows": fails}


@pytest.fixture(scope="session")
def real_pipeline():
    if not ARTEFACTS.exists():
        pytest.skip("train first: python run_pipeline.py train")
    from src.inference.pipeline import RootCausePipeline
    return RootCausePipeline()
