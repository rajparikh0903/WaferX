"""
Compatibility implementation for the custom ``supervised_process.DropBad``
class referenced by ``secom_rf_k100_exact.joblib``.

Purpose
-------
The saved SECOM Random Forest artifact stores an instance of
``supervised_process.DropBad``.  Python/joblib must be able to import this
class before the artifact can be loaded.

For inference, the important state is ``keep_`` (the fitted column indices),
which is restored automatically by pickle/joblib.  ``transform()`` then
selects exactly those columns.

IMPORTANT
---------
This file is an inference-compatibility implementation.  The original
training-time DropBad selection rule is not encoded in the JSON metadata, so
``fit()`` below provides a conservative reusable rule for new training data,
but you should NOT retrain the existing RF model with it unless you have the
original training code.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DropBad:
    """Drop unusable sensor columns and preserve the selected column indices.

    The loaded model already contains ``keep_``.  During model inference,
    ``transform()`` uses that stored list exactly, so the saved RF artifact
    sees the same 442 columns it saw during training before imputation.

    Parameters
    ----------
    max_missing_fraction:
        Maximum fraction of NaN/inf values allowed when fitting a new
        DropBad instance.  This is only a fallback fitting policy and is not
        claimed to be the original training rule.
    min_unique_values:
        Minimum number of finite unique values required when fitting a new
        DropBad instance.
    """

    def __init__(
        self,
        max_missing_fraction: float = 0.5,
        min_unique_values: int = 2,
    ) -> None:
        if not 0.0 <= max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be between 0 and 1")
        if min_unique_values < 1:
            raise ValueError("min_unique_values must be >= 1")

        self.max_missing_fraction = float(max_missing_fraction)
        self.min_unique_values = int(min_unique_values)

    def fit(self, X: Any, y: Any = None) -> "DropBad":
        """Fit a reusable column selector on a numeric 2-D array.

        This fallback implementation removes columns that are mostly missing
        or have too few finite unique values.  It is intended for convenience
        only; the existing serialized RF model does not need to call fit().
        """
        X = self._validate_input(X)

        finite = np.isfinite(X)
        missing_fraction = 1.0 - finite.mean(axis=0)

        keep: list[int] = []
        for j in range(X.shape[1]):
            if missing_fraction[j] > self.max_missing_fraction:
                continue

            vals = X[finite[:, j], j]
            if np.unique(vals).size < self.min_unique_values:
                continue

            keep.append(j)

        if not keep:
            raise ValueError("DropBad removed every feature; check input data or thresholds")

        self.keep_ = keep
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(keep)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Select the fitted/serialized columns."""
        X = self._validate_input(X)

        if not hasattr(self, "keep_"):
            raise RuntimeError("DropBad has not been fitted and has no serialized keep_ state")

        keep = np.asarray(self.keep_, dtype=int)

        if keep.ndim != 1 or keep.size == 0:
            raise ValueError("Invalid DropBad.keep_ state")

        if np.any(keep < 0) or np.any(keep >= X.shape[1]):
            raise ValueError(
                f"DropBad.keep_ contains indices outside the input width "
                f"({X.shape[1]} features)"
            )

        return X[:, keep]

    def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> np.ndarray:
        """Fit then transform."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected-feature mask or selected indices."""
        if not hasattr(self, "keep_"):
            raise RuntimeError("DropBad has not been fitted")

        keep = np.asarray(self.keep_, dtype=int)

        if indices:
            return keep

        n_features = getattr(self, "n_features_in_", int(keep.max()) + 1)
        mask = np.zeros(n_features, dtype=bool)
        mask[keep] = True
        return mask

    def _validate_input(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D feature matrix, got shape {X.shape}")
        return X

    def __getstate__(self) -> dict[str, Any]:
        """Use normal instance state so joblib/pickle can serialize it."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the fitted state embedded in the joblib artifact."""
        self.__dict__.update(state)
