"""Unit tests for shared.metrics, shared.result_schema, and shared.artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.artifacts import (
    UntrustedArtifactError,
    create_unique_run_dir,
    load_trusted_pipeline_artifact,
    save_pipeline_artifact,
)
from shared.config import WAP_COLUMNS
from shared.metrics import (
    compute_classification_metrics,
    compute_pooled_metrics,
    evaluate_predictions_by_building,
    select_best_candidate,
)
from shared.preprocessing import (
    build_gradient_boosting_pipeline,
    build_pca_knn_pipeline,
    describe_preprocessing_config,
)
from shared.result_schema import (
    ResultSchemaError,
    build_completed_result,
    build_non_completed_result,
    load_predictions_csv,
    load_result_json,
    save_predictions_csv,
    save_result_json,
    validate_predictions_dataframe,
    validate_result_payload,
)


def test_pooled_accuracy_uses_pooled_counts_not_unweighted_building_average() -> None:
    """Pooled accuracy MUST equal total_correct / total_rows, NOT mean(building_accuracies)."""
    # Building 0: 100 rows, 90 correct (accuracy = 0.90)
    # Building 1: 10 rows, 5 correct (accuracy = 0.50)
    # Unweighted building mean = (0.90 + 0.50) / 2 = 0.70
    # True pooled accuracy = (90 + 5) / (100 + 10) = 95 / 110 = 0.86363636...
    b0_true = [0] * 50 + [1] * 50
    b0_pred = [0] * 45 + [1] * 5 + [1] * 45 + [0] * 5  # 90 correct

    b1_true = [0] * 5 + [1] * 5
    b1_pred = [0] * 3 + [1] * 2 + [1] * 2 + [0] * 3  # 5 correct

    preds_df = pd.DataFrame(
        {
            "row_id": [f"syn_{i:04d}" for i in range(110)],
            "building_id": [0] * 100 + [1] * 10,
            "true_floor": b0_true + b1_true,
            "predicted_floor": b0_pred + b1_pred,
            "evaluation_role": ["internal_holdout"] * 110,
            "model_id": ["pca_knn"] * 110,
        }
    )

    per_bldg, pooled, _, _ = evaluate_predictions_by_building(preds_df)
    assert np.isclose(per_bldg["0"]["accuracy"], 0.90)
    assert np.isclose(per_bldg["1"]["accuracy"], 0.50)

    expected_pooled_acc = 95.0 / 110.0
    assert np.isclose(pooled["accuracy"], expected_pooled_acc, atol=1e-12)
    assert not np.isclose(pooled["accuracy"], 0.70)
    assert pooled["correct_count"] == 95
    assert pooled["incorrect_count"] == 15
    assert pooled["evaluated_row_count"] == 110


def test_pooled_metrics_fixed_label_policy_across_buildings_with_different_floor_counts() -> None:
    """Pooled metrics must handle Buildings 0 & 1 (floors 0..3) and Building 2 (floors 0..4) cleanly."""
    preds_df = pd.DataFrame(
        {
            "row_id": [f"syn_row_{i:03d}" for i in range(13)],
            "building_id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "true_floor": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4],
            "predicted_floor": [0, 1, 2, 3, 0, 1, 2, 2, 0, 1, 2, 3, 4],
            "evaluation_role": ["internal_holdout"] * 13,
            "model_id": ["gradient_boosting"] * 13,
        }
    )
    per_bldg, pooled, supports, cms = evaluate_predictions_by_building(
        preds_df, fixed_pooled_labels=[0, 1, 2, 3, 4]
    )
    assert list(per_bldg["0"]["per_class"].keys()) == ["0", "1", "2", "3"]
    assert list(per_bldg["2"]["per_class"].keys()) == ["0", "1", "2", "3", "4"]
    assert list(pooled["per_class"].keys()) == ["0", "1", "2", "3", "4"]
    assert pooled["label_policy"] == "fixed_global_floor_union_with_zero_division_0"
    assert "support_weighted_building_macro_f1" in pooled
    assert "unweighted_building_mean_macro_f1" in pooled
    assert cms["pooled"]["labels"] == [0, 1, 2, 3, 4]
    assert supports["pooled"]["evaluated_row_count"] == 13


def test_candidate_selection_accuracy_with_macro_f1_tie_breaker_and_error_handling() -> None:
    """select_best_candidate must break ties in mean_cv_accuracy using mean_cv_macro_f1 and reject empty sets."""
    candidates = [
        {"id": "cand_a", "feasible": True, "mean_cv_accuracy": 0.9100, "mean_cv_macro_f1": 0.8800},
        {"id": "cand_b", "feasible": True, "mean_cv_accuracy": 0.9100, "mean_cv_macro_f1": 0.8950},
        {"id": "cand_c", "feasible": False, "mean_cv_accuracy": 0.9900, "mean_cv_macro_f1": 0.9900},
        {"id": "cand_d", "feasible": True, "status": "skipped", "mean_cv_accuracy": 0.9800, "mean_cv_macro_f1": 0.9800},
    ]
    winner = select_best_candidate(candidates)
    assert winner["id"] == "cand_b"

    # When both accuracy and macro_f1 are tied within tolerance, earlier candidate wins deterministically
    exact_tie = [
        {"id": "first_simpler", "feasible": True, "mean_cv_accuracy": 0.9000, "mean_cv_macro_f1": 0.8900},
        {"id": "second_complex", "feasible": True, "mean_cv_accuracy": 0.9000, "mean_cv_macro_f1": 0.8900},
    ]
    assert select_best_candidate(exact_tie)["id"] == "first_simpler"

    with pytest.raises(ValueError, match="No feasible completed candidates"):
        select_best_candidate([{"id": "bad", "feasible": False, "mean_cv_accuracy": 0.5, "mean_cv_macro_f1": 0.5}])


def test_result_schema_v1_roundtrip_and_rejection_of_fabricated_failed_metrics(
    tmp_path: Path,
) -> None:
    """Completed results and predictions must round-trip cleanly; failed/skipped runs must reject fabricated metrics."""
    preds_df = pd.DataFrame(
        {
            "row_id": [f"train_{i:06d}" for i in range(12)],
            "building_id": [0] * 4 + [1] * 4 + [2] * 4,
            "true_floor": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 4],
            "predicted_floor": [0, 1, 2, 2, 0, 1, 2, 3, 0, 1, 4, 4],
            "evaluation_role": ["internal_holdout"] * 12,
            "model_id": ["pca_knn"] * 12,
        }
    )
    pred_path = save_predictions_csv(preds_df, tmp_path / "predictions.csv", evaluation_role="internal_holdout")
    loaded_preds = load_predictions_csv(pred_path, evaluation_role="internal_holdout", validate=True)
    assert len(loaded_preds) == 12

    per_bldg, pooled, supports, cms = evaluate_predictions_by_building(loaded_preds)

    completed = build_completed_result(
        run_id="20261012T000000Z_pca_knn_internal_holdout_12345678",
        model_id="pca_knn",
        dataset_hash="a" * 64,
        split_manifest_hash="b" * 64,
        evaluation_role="internal_holdout",
        preprocessing=describe_preprocessing_config(scale=True, use_pca=True, n_components=10),
        selected_parameters={"0": {"n_components": 10, "n_neighbors": 3, "weights": "uniform"}},
        selection_metric={"primary": "cv_accuracy", "tie_breaker": "cv_macro_f1"},
        per_building=per_bldg,
        pooled=pooled,
        supports=supports,
        confusion_matrices=cms,
        timing={"fit_seconds": 0.12, "predict_seconds": 0.03, "total_seconds": 0.15},
        prediction_file_path=pred_path,
        model_artifact_paths={"0": "models/pca_knn/artifacts/b0.joblib"},
        assumptions=["Sentinel 100 replaced with -110.0 dBm (project assumption)."],
        warnings=[],
    )
    res_json_path = save_result_json(completed, tmp_path / "result.json")
    loaded_result = load_result_json(res_json_path, validate=True)
    assert loaded_result["run_id"] == completed["run_id"]
    assert np.isclose(loaded_result["pooled"]["accuracy"], completed["pooled"]["accuracy"])

    # Non-completed result works cleanly when metrics are None
    skipped = build_non_completed_result(
        run_id="run_skipped_01",
        model_id="pca_knn",
        status="skipped",
        evaluation_role="cv_val",
        reason="PCA components exceeded training partition size.",
    )
    validate_result_payload(skipped)

    # Fabricating metrics in a failed/skipped result must raise ResultSchemaError
    fabricated = dict(skipped)
    fabricated["pooled"] = pooled
    with pytest.raises(ResultSchemaError, match="must NOT contain fabricated numerical metrics"):
        validate_result_payload(fabricated)


def test_oof_predictions_and_final_holdout_role_isolation() -> None:
    """Out-of-fold predictions must contain fold_id/repeat_id, and evaluation roles must never be mixed."""
    df_missing_fold = pd.DataFrame(
        {
            "row_id": ["train_000001", "train_000002"],
            "building_id": [0, 0],
            "true_floor": [0, 1],
            "predicted_floor": [0, 1],
            "evaluation_role": ["cv_val", "cv_val"],
            "model_id": ["pca_weighted_knn", "pca_weighted_knn"],
        }
    )
    with pytest.raises(ResultSchemaError, match="fold_id"):
        validate_predictions_dataframe(df_missing_fold, evaluation_role="cv_val")

    df_valid_oof = df_missing_fold.copy()
    df_valid_oof["fold_id"] = [0, 1]
    df_valid_oof["repeat_id"] = [0, 0]
    validate_predictions_dataframe(df_valid_oof, evaluation_role="cv_val")

    # Mixing internal_holdout and official_validation rows in a single predictions table is prohibited
    df_mixed_roles = pd.DataFrame(
        {
            "row_id": ["train_000001", "val_000001"],
            "building_id": [0, 0],
            "true_floor": [0, 1],
            "predicted_floor": [0, 1],
            "evaluation_role": ["internal_holdout", "official_validation"],
            "model_id": ["pca_knn", "pca_knn"],
        }
    )
    with pytest.raises(ResultSchemaError, match="mixes multiple evaluation roles"):
        validate_predictions_dataframe(df_mixed_roles)


def test_artifact_roundtrip_all_three_models_and_sha256_trust_verification(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Run directories must never overwrite, and all 3 model families must round-trip with SHA-256 verification."""
    run_id, run_dir = create_unique_run_dir(
        tmp_path, model_id="pca_knn", evaluation_role="internal_holdout", run_id="fixed_run_01"
    )
    assert run_dir.is_dir()

    # Attempting to create the same run directory again must fail
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_unique_run_dir(
            tmp_path, model_id="pca_knn", evaluation_role="internal_holdout", run_id="fixed_run_01"
        )

    X = synthetic_uji_df.loc[:45, list(WAP_COLUMNS)]
    y = synthetic_uji_df.loc[:45, "FLOOR"].astype(int)

    pipelines = {
        "pca_knn": build_pca_knn_pipeline(n_components=5, n_neighbors=3, weights="uniform", scale=True),
        "pca_weighted_knn": build_pca_knn_pipeline(n_components=5, n_neighbors=3, weights="distance", scale=True),
        "gradient_boosting": build_gradient_boosting_pipeline(learning_rate=0.1, n_estimators=10, max_depth=3),
    }

    for idx, (model_id, pipe) in enumerate(pipelines.items()):
        pipe.fit(X, y)
        meta = save_pipeline_artifact(
            pipeline=pipe,
            output_dir=run_dir,
            building_id=idx,
            model_id=model_id,
            feature_names=WAP_COLUMNS,
            class_labels=[0, 1, 2, 3],
            dataset_hash="a" * 64,
            split_manifest_hash="b" * 64,
            run_id=run_id,
        )
        artifact_file = Path(meta["artifact_path"])
        loaded_pipe, loaded_meta = load_trusted_pipeline_artifact(
            artifact_file, allowed_root=tmp_path
        )
        assert loaded_meta["artifact_sha256"] == meta["artifact_sha256"]
        assert loaded_meta["model_id"] == model_id
        np.testing.assert_array_equal(loaded_pipe.predict(X), pipe.predict(X))

    # Test security and integrity checks on building 0 artifact
    b0_artifact = run_dir / "pca_knn_building_0_pipeline.joblib"
    other_root = tmp_path / "other_allowed_root"
    other_root.mkdir()
    with pytest.raises(UntrustedArtifactError, match="outside allowed root"):
        load_trusted_pipeline_artifact(b0_artifact, allowed_root=other_root)

    # Corrupt a byte in the serialized file -> SHA-256 verification must block loading
    raw_bytes = bytearray(b0_artifact.read_bytes())
    raw_bytes[-1] ^= 0xFF
    b0_artifact.write_bytes(bytes(raw_bytes))

    with pytest.raises(UntrustedArtifactError, match="SHA-256 integrity check failed"):
        load_trusted_pipeline_artifact(b0_artifact, allowed_root=tmp_path)
