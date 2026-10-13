"""Regression tests for defects found during the uploaded-archive review."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from shared.config import DEFAULT_TRAINING_CSV, DEFAULT_VALIDATION_CSV, PROJECT_ROOT
from shared.create_splits import SplitCreationError, create_development_splits, validate_split_manifest
from shared.data_validation import (
    DataValidationError,
    audit_dataframe,
    compute_file_sha256,
    extract_predictors_and_targets,
    diagnose_duplicates_and_undetected,
    verify_training_csv_provenance,
)
from shared.metrics import select_best_candidate
from shared.preprocessing import SentinelReplacementTransformer
from shared.result_schema import ResultSchemaError, validate_predictions_dataframe


def test_default_paths_use_dataset_even_when_files_are_missing():
    assert DEFAULT_TRAINING_CSV == PROJECT_ROOT / "Dataset" / "trainingData.csv"
    assert DEFAULT_VALIDATION_CSV == PROJECT_ROOT / "Dataset" / "validationData.csv"


@pytest.mark.parametrize("column", ["FLOOR", "BUILDINGID"])
def test_fractional_labels_are_not_silently_truncated(synthetic_uji_df, column):
    frame = synthetic_uji_df.copy()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = 0.5
    with pytest.raises(DataValidationError, match="integer"):
        extract_predictors_and_targets(frame)
    with pytest.raises(DataValidationError):
        audit_dataframe(frame, strict=True)


def test_transform_requires_fit_and_preserves_feature_order():
    transformer = SentinelReplacementTransformer()
    frame = pd.DataFrame({"a": [-50, 100], "b": [0, -70]})
    with pytest.raises(NotFittedError):
        transformer.transform(frame)
    transformer.fit(frame)
    with pytest.raises(ValueError, match="names/order"):
        transformer.transform(frame[["b", "a"]])
    np.testing.assert_array_equal(transformer.transform(frame), [[-50, 0], [-110, -70]])


def test_transformer_rejects_nonfinite_replacement():
    with pytest.raises(ValueError, match="finite"):
        SentinelReplacementTransformer(replacement_value=np.nan).fit(np.zeros((2, 2)))


def test_manifest_cannot_remap_source_indices(synthetic_uji_df):
    manifest = create_development_splits(synthetic_uji_df)
    manifest.loc[[0, 1], "source_row_index"] = [1, 0]
    with pytest.raises(SplitCreationError, match="source_row_index"):
        validate_split_manifest(manifest)
    assert not validate_split_manifest(manifest, strict=False).valid


def test_changed_csv_content_invalidates_frozen_provenance(tmp_path, synthetic_uji_df):
    csv = tmp_path / "trainingData.csv"
    synthetic_uji_df.to_csv(csv, index=False)
    original_hash = compute_file_sha256(csv)
    verify_training_csv_provenance(csv, pd.read_csv(csv), original_hash)
    changed = synthetic_uji_df.copy()
    changed.loc[0, "WAP001"] = -50
    changed.to_csv(csv, index=False)
    with pytest.raises(DataValidationError, match="provenance"):
        verify_training_csv_provenance(csv, pd.read_csv(csv), original_hash)


def test_nan_cannot_win_candidate_selection():
    with pytest.raises(ValueError, match="finite"):
        select_best_candidate([{"mean_cv_accuracy": np.nan, "mean_cv_macro_f1": 0.9}])


def test_fractional_prediction_labels_rejected():
    predictions = pd.DataFrame({
        "row_id": ["r0"], "building_id": [0], "true_floor": [0],
        "predicted_floor": [0.5], "evaluation_role": ["internal_holdout"],
        "model_id": ["pca_knn"],
    })
    with pytest.raises(ResultSchemaError, match="integer"):
        validate_predictions_dataframe(predictions)


def test_cross_building_duplicates_do_not_hide_within_building_conflicts(synthetic_uji_df):
    frame = synthetic_uji_df.iloc[:4].copy()
    frame.loc[:, [f"WAP{i:03d}" for i in range(1, 521)]] = 100
    frame["BUILDINGID"] = [0, 0, 1, 1]
    frame["FLOOR"] = [0, 1, 2, 2]
    _, _, diagnostics = diagnose_duplicates_and_undetected(frame)
    assert diagnostics["cross_building_duplicate_groups"] == 1
    assert diagnostics["conflicting_floor_duplicate_groups"] == 1
    assert diagnostics["conflicting_floor_row_count"] == 2
    assert diagnostics["same_building_same_floor_duplicate_groups"] == 1


def test_shared_public_exports_still_resolve():
    import shared

    for name in shared.__all__:
        assert getattr(shared, name) is not None