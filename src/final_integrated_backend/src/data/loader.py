"""Load the official UCI SECOM files (secom.data + secom_labels.data)."""
from __future__ import annotations

import pandas as pd

from src.utils import resolve


def feature_names(n: int, cfg: dict) -> list[str]:
    tpl = cfg["data"]["feature_name_template"]
    return [tpl.format(i=i + 1) for i in range(n)]


def load_secom(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, timestamps).

    X: 590 measurement columns named "Sensor 1".."Sensor 590" (NaN preserved)
    y: 1 = FAIL, 0 = PASS (SECOM encodes -1 = pass, 1 = fail)
    index = sample_id = zero-based row position in the original file.
    """
    raw = resolve(cfg, "raw_dir")
    X = pd.read_csv(raw / cfg["data"]["data_file"], sep=r"\s+", header=None, na_values=["NaN", "nan"])
    lab = pd.read_csv(raw / cfg["data"]["labels_file"], sep=" ", header=None,
                      names=["label", "timestamp"], quotechar='"')
    if len(X) != len(lab):
        raise ValueError(f"Row mismatch: data has {len(X)} rows, labels have {len(lab)}")
    X.columns = feature_names(X.shape[1], cfg)
    X.index.name = "sample_id"
    y = (lab["label"].astype(int) == 1).astype(int)
    y.index = X.index
    y.name = "fail"
    ts = pd.to_datetime(lab["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    ts.index = X.index
    ts.name = "timestamp"
    return X, y, ts
