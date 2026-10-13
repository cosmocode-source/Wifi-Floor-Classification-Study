"""Unit tests for shared.preprocessing (sentinel replacement, PCA whiten=False, fold-local fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.config import WAP_COLUMNS
from shared.data_validation import DataValidationError
from shared.preprocessing import (
    SentinelReplacementTransformer,
    build_gradient_boosting_pipeline,
    build_knn_preprocessing_pipeline,
    build_pca_knn_pipeline,
)


def test_sentinel_replacement_converts_100_to_minus_110_and_preserves_zero() -> None:
    """Sentinel 100 must become -110.0 while genuine 0 dBm and negative RSSI remain unchanged."""
    raw = np.array(
        [
            [100, 0, -55, -104],
            [-90, 100, 0, 100],
        ],
        dtype=float,
    )
    transformer = SentinelReplacementTransformer(sentinel_value=100, replacement_value=-110.0)
    transformed = transformer.fit_transform(raw)

    expected = np.array(
        [
            [-110.0, 0.0, -55.0, -104.0],
            [-90.0, -110.0, 0.0, -110.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(transformed, expected)
    # Input array must not be mutated in place
    assert raw[0, 0] == 100.0


def test_sentinel_transformer_enforces_wap_schema_when_enabled(synthetic_uji_df: pd.DataFrame) -> None:
    """When enforce_wap_schema=True, passing BUILDINGID or FLOOR in DataFrame must raise DataValidationError."""
    transformer = SentinelReplacementTransformer(enforce_wap_schema=True)
    valid_X = synthetic_uji_df.loc[:, list(WAP_COLUMNS)]
    out = transformer.fit_transform(valid_X)
    assert out.shape == valid_X.shape

    leaked_X = synthetic_uji_df.loc[:, [*WAP_COLUMNS[:-1], "BUILDINGID"]]
    with pytest.raises(DataValidationError):
        transformer.fit(leaked_X)


def test_knn_pipeline_pca_whiten_is_false_and_scaler_is_optional() -> None:
    """PCA must always use whiten=False, and StandardScaler must be toggleable."""
    pipe_scaled = build_pca_knn_pipeline(n_components=10, n_neighbors=3, weights="uniform", scale=True)
    assert "scaler" in pipe_scaled.named_steps
    assert pipe_scaled.named_steps["pca"].whiten is False
    assert pipe_scaled.named_steps["classifier"].weights == "uniform"

    pipe_unscaled = build_pca_knn_pipeline(n_components=10, n_neighbors=3, weights="distance", scale=False)
    assert "scaler" not in pipe_unscaled.named_steps
    assert pipe_unscaled.named_steps["pca"].whiten is False
    assert pipe_unscaled.named_steps["classifier"].weights == "distance"


def test_fold_local_preprocessing_prevents_validation_leakage() -> None:
    """StandardScaler and PCA inside the pipeline must fit strictly on X_train, never on X_val."""
    rng = np.random.default_rng(99)
    X_train = rng.normal(loc=-80.0, scale=5.0, size=(40, 20))
    X_train[X_train > 0] = -10.0
    y_train = np.array([0, 1] * 20, dtype=int)

    # Validation fold has a drastically shifted distribution that would alter mean/std if leaked
    X_val = np.full((15, 20), 100.0, dtype=float)

    pipe = build_knn_preprocessing_pipeline(n_components=5, scale=True, random_state=42)
    pipe.fit(X_train, y_train)

    expected_train_mean = X_train.mean(axis=0)
    scaler_mean_before = pipe.named_steps["scaler"].mean_.copy()
    pca_components_before = pipe.named_steps["pca"].components_.copy()

    np.testing.assert_allclose(scaler_mean_before, expected_train_mean, atol=1e-10)

    # Transforming X_val must not refit or mutate learned parameters
    _ = pipe.transform(X_val)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, scaler_mean_before, atol=1e-12)
    np.testing.assert_allclose(pipe.named_steps["pca"].components_, pca_components_before, atol=1e-12)


def test_gradient_boosting_pipeline_excludes_pca_and_scaler_by_default_and_supports_optional_toggle() -> None:
    """Classical gradient boosting excludes scaler/PCA by default, but allows optional scale and PCA."""
    gb_default = build_gradient_boosting_pipeline(learning_rate=0.1, n_estimators=20, max_depth=4)
    assert list(gb_default.named_steps.keys()) == ["sentinel_replacement", "classifier"]
    assert gb_default.named_steps["classifier"].max_depth == 4
    assert gb_default.named_steps["classifier"].n_estimators == 20

    gb_scaled_pca = build_gradient_boosting_pipeline(
        learning_rate=0.05,
        n_estimators=15,
        max_depth=3,
        scale=True,
        use_pca=True,
        n_components=8,
    )
    assert list(gb_scaled_pca.named_steps.keys()) == ["sentinel_replacement", "scaler", "pca", "classifier"]
    assert gb_scaled_pca.named_steps["pca"].n_components == 8
    assert gb_scaled_pca.named_steps["pca"].whiten is False

    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        build_gradient_boosting_pipeline(use_pca=True, n_components=None)
