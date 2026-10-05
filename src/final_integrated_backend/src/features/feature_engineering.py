"""Feature engineering.

Deliberately minimal: SECOM sensors are anonymous, and the what-if engine must be able to change a
single *raw* process measurement. Derived/aggregate features would break that mapping, so the only
engineered features are missing-value indicators (one 0/1 column per sensor with notable missingness).
"""
from __future__ import annotations

import pandas as pd

MISSING_SUFFIX = "__missing"


def indicator_name(col: str) -> str:
    return f"{col}{MISSING_SUFFIX}"


def is_indicator(name: str) -> bool:
    return name.endswith(MISSING_SUFFIX)


def base_name(name: str) -> str:
    return name[: -len(MISSING_SUFFIX)] if is_indicator(name) else name


def missing_indicator_frame(X_raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = X_raw[cols].isna().astype(float)
    out.columns = [indicator_name(c) for c in cols]
    return out
