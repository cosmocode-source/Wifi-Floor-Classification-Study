"""Leakage-safe sklearn preprocessing transformers and pipeline constructors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from shared.config import (
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_UNDETECTED_REPLACEMENT,
    NUM_WAP_FEATURES,
    UNDETECTED_SENTINEL,
    WAP_COLUMNS,
)
from shared.data_validation import validate_predictor_columns


class SentinelReplacementTransformer(BaseEstimator, TransformerMixin):
    'Replace UJIIndoorLoc undetected-WAP sentinel (100) with a low RSSI value (-110.0).'

    def __init__(
        self,
        sentinel_value: int | float = UNDETECTED_SENTINEL,
        replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
        enforce_wap_schema: bool = False,
    ) -> None:
        self.sentinel_value = sentinel_value
        self.replacement_value = replacement_value
        self.enforce_wap_schema = enforce_wap_schema

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> SentinelReplacementTransformer:
        """Validate input dimensions and column names (if DataFrame) during fit."""
        if not np.isfinite(float(self.sentinel_value)) or not np.isfinite(float(self.replacement_value)):
            raise ValueError("Sentinel and replacement values must be finite.")
        self._validate_input(X, is_fit=True)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Replace exact occurrences of `sentinel_value` with `replacement_value` as float64."""
        check_is_fitted(self, "n_features_in_")
        arr = self._validate_input(X, is_fit=False)
        out = arr.astype(np.float64, copy=True)
        sentinel_mask = out == float(self.sentinel_value)
        out[sentinel_mask] = float(self.replacement_value)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return feature names unchanged."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(getattr(self, "n_features_in_", NUM_WAP_FEATURES))], dtype=object)

    def _validate_input(self, X: pd.DataFrame | np.ndarray, is_fit: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.enforce_wap_schema:
                validate_predictor_columns(list(X.columns))
            if is_fit:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
                self.n_features_in_ = X.shape[1]
            else:
                if hasattr(self, "feature_names_in_") and not np.array_equal(
                    np.asarray(X.columns, dtype=object), self.feature_names_in_
                ):
                    raise ValueError("Feature names/order differ from those used during fit.")
                if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"Feature count mismatch: expected {self.n_features_in_}, got {X.shape[1]}."
                    )
            arr = X.to_numpy(dtype=np.float64, copy=False)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array for WAP features, got {arr.ndim}D.")
            if self.enforce_wap_schema and arr.shape[1] != NUM_WAP_FEATURES:
                raise ValueError(f"Expected {NUM_WAP_FEATURES} WAP features, got {arr.shape[1]}.")
            if is_fit:
                self.n_features_in_ = arr.shape[1]
            elif hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features_in_}, got {arr.shape[1]}."
                )

        if not np.isfinite(arr).all():
            raise ValueError("Input features contain NaN or infinite values.")
        return arr


def build_knn_preprocessing_pipeline(
    n_components: int,
    scale: bool = True,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = False,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Construct a leakage-safe preprocessing pipeline (Sentinel -> optional StandardScaler -> PCA(whiten=False))."""
    if not isinstance(n_components, (int, np.integer)) or int(n_components) < 1:
        raise ValueError(f"n_components must be a positive integer, got {n_components!r}.")

    steps: list[tuple[str, Any]] = [
        (
            "sentinel_replacement",
            SentinelReplacementTransformer(
                sentinel_value=sentinel_value,
                replacement_value=replacement_value,
                enforce_wap_schema=enforce_wap_schema,
            ),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    steps.append(
        (
            "pca",
            PCA(
                n_components=int(n_components),
                whiten=False,
                svd_solver="full",
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps=steps)


def build_pca_knn_pipeline(
    n_components: int,
    n_neighbors: int = 1,
    weights: str = "uniform",
    metric: str = "euclidean",
    scale: bool = True,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = False,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Pipeline:
    """Construct a full leakage-safe PCA + kNN classification pipeline.

    Supports both uniform-weight kNN (`weights="uniform"`) and distance-weighted kNN (`weights="distance"`).
    All learned steps (`StandardScaler`, `PCA`, `KNeighborsClassifier`) fit exclusively on the training split.
    """
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}.")
    if not isinstance(n_neighbors, (int, np.integer)) or int(n_neighbors) < 1:
        raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors!r}.")

    prep = build_knn_preprocessing_pipeline(
        n_components=int(n_components),
        scale=scale,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
    )
    steps = list(prep.steps)
    steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=int(n_neighbors),
                weights=weights,
                metric=metric,
                n_jobs=n_jobs,
            ),
        )
    )
    return Pipeline(steps=steps)


def build_gradient_boosting_pipeline(
    learning_rate: float = 0.1,
    n_estimators: int = 189,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    max_features: str | int | float | None = None,
    scale: bool = False,
    use_pca: bool = False,
    n_components: int | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = False,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Construct a leakage-safe classical GradientBoostingClassifier pipeline.

    Uses SentinelReplacementTransformer (100 -> -110.0) without StandardScaler or PCA by default,
    while allowing optional `scale=True` and/or `use_pca=True` (with `n_components`) when requested.
    """
    steps: list[tuple[str, Any]] = [
        (
            "sentinel_replacement",
            SentinelReplacementTransformer(
                sentinel_value=sentinel_value,
                replacement_value=replacement_value,
                enforce_wap_schema=enforce_wap_schema,
            ),
        ),
    ]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    if use_pca:
        if n_components is None or not isinstance(n_components, (int, np.integer)) or int(n_components) < 1:
            raise ValueError(
                f"When use_pca=True in build_gradient_boosting_pipeline, n_components must be a positive integer; "
                f"got {n_components!r}."
            )
        steps.append(
            (
                "pca",
                PCA(
                    n_components=int(n_components),
                    whiten=False,
                    svd_solver="full",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "classifier",
            GradientBoostingClassifier(
                learning_rate=float(learning_rate),
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                subsample=float(subsample),
                max_features=max_features,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps=steps)


def describe_preprocessing_config(
    scale: bool,
    use_pca: bool,
    n_components: int | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
) -> dict[str, Any]:
    """Generate a standardized preprocessing metadata dictionary for result_schema."""
    return {
        "predictor_allowlist": f"{WAP_COLUMNS[0]}..{WAP_COLUMNS[-1]} ({NUM_WAP_FEATURES} features)",
        "sentinel_value": sentinel_value,
        "replacement_value": replacement_value,
        "replacement_is_assumption": True,
        "standard_scaler": bool(scale),
        "pca_enabled": bool(use_pca),
        "pca_n_components": int(n_components) if (use_pca and n_components is not None) else None,
        "pca_whiten": False if use_pca else None,
        "fit_scope": "training_partition_only",
    }
