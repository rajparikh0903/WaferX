import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import SecomPreprocessor, make_splits


def _frame(n=200, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 6)), columns=[f"Sensor {i}" for i in range(1, 7)])
    X["Sensor 3"] = 5.0                                  # constant
    X.loc[r.choice(n, 40, replace=False), "Sensor 1"] = np.nan   # 20% missing
    X.loc[r.choice(n, 150, replace=False), "Sensor 6"] = np.nan  # 75% missing -> dropped
    return X


def test_constant_and_high_missing_columns_dropped():
    p = SecomPreprocessor().fit(_frame())
    assert "Sensor 3" not in p.kept_columns_ and "Sensor 6" not in p.kept_columns_
    assert "Sensor 1" in p.kept_columns_


def test_missing_values_imputed_with_train_median_only():
    X = _frame()
    tr, te = X.iloc[:150], X.iloc[150:].copy()
    p = SecomPreprocessor().fit(tr)
    te.loc[te.index[0], "Sensor 1"] = np.nan
    out = p.transform(te)
    assert not out.isna().any().any()
    assert out.loc[te.index[0], "Sensor 1"] == pytest.approx(tr["Sensor 1"].median())   # train median, not test


def test_scaler_fitted_on_train_only():
    X = _frame()
    tr = X.iloc[:100]
    p = SecomPreprocessor().fit(tr)
    assert p.scaler_.mean_[p.kept_columns_.index("Sensor 2")] == pytest.approx(tr["Sensor 2"].mean())


def test_indicators_added_only_when_enabled():
    X = _frame()
    p = SecomPreprocessor(use_indicators=False).fit(X)
    assert not any(c.endswith("__missing") for c in p.transform(X).columns)
    assert "Sensor 1__missing" in p.transform(X, with_indicators=True).columns


@pytest.mark.parametrize("bad", ["abc", np.inf])
def test_invalid_values_rejected(bad):
    X = _frame()
    p = SecomPreprocessor().fit(X)
    X2 = X.copy().astype(object)
    X2.iloc[0, 0] = bad
    with pytest.raises(ValueError):
        p.transform(X2)


def test_feature_mismatch_rejected():
    X = _frame()
    p = SecomPreprocessor().fit(X)
    with pytest.raises(ValueError, match="mismatch"):
        p.transform(X.drop(columns=["Sensor 2"]))
    with pytest.raises(ValueError, match="mismatch"):
        p.transform(X.assign(**{"Sensor 99": 1.0}))


def test_temporal_split_is_chronological_and_disjoint(cfg):
    ts = pd.Series(pd.date_range("2008-01-01", periods=100, freq="h"))
    y = pd.Series(np.r_[np.zeros(90), np.ones(10)].astype(int))
    local_cfg = dict(cfg)
    local_cfg["split"] = dict(cfg["split"])
    local_cfg["split"]["method"] = "temporal"
    s = make_splits(y, ts, local_cfg)
    assert ts.iloc[s["train"]].max() <= ts.iloc[s["val"]].min() <= ts.iloc[s["val"]].max() <= ts.iloc[s["test"]].min()
    assert len(set(s["train"]) & set(s["val"])) == 0 and len(set(s["val"]) & set(s["test"])) == 0
    assert sum(len(v) for v in s.values()) == 100
