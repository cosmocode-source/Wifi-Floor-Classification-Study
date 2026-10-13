"""Synthetic unit and integration tests for Stage 2: PCA + uniform-weight kNN (`models.pca_knn`).

NOTE: All tests in this file operate exclusively on synthetic fixtures (`synthetic_uji_df`)
and temporary directories (`tmp_path`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.pca_knn.clean_data import (
    clean_pca_knn_dataframe,
    extract_validated_wap_features_and_targets,
    load_pca_knn_config,
    run_clean_data,
)
from models.pca_knn.test import evaluate_pca_knn, load_frozen_selection_config
from models.pca_knn.train import (
    FIXED_BASELINE_PARAMS,
    build_fixed_baseline_pipeline,
    build_uniform_pca_knn_pipeline,
    evaluate_single_candidate_cv,
    generate_candidate_grid,
    resolve_parallelism,
    select_best_pca_knn_candidate,
    train_pca_knn,
)
from shared.artifacts import UntrustedArtifactError, load_trusted_pipeline_artifact
from shared.config import (
    BUILDING_COLUMN,
    EVALUATION_ROLE_CV_VAL,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
    MODEL_ID_PCA_KNN,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    TARGET_COLUMN,
    WAP_COLUMNS,
)
from shared.create_splits import (
    align_dataset_with_manifest,
    create_development_splits,
    iter_building_cv_folds,
    save_split_manifest,
)
from shared.data_validation import DataValidationError, compute_file_sha256
from shared.result_schema import (
    load_predictions_csv,
    load_result_json,
    validate_predictions_dataframe,
    validate_result_payload,
)


def _write_synthetic_workspace(
    tmp_path: Path,
    synthetic_df: pd.DataFrame,
) -> tuple[Path, Path, str, str]:
    """Write synthetic training CSV and shared split manifest to `tmp_path`."""
    train_csv = tmp_path / "trainingData.csv"
    manifest_csv = tmp_path / "development_splits.csv"
    manifest_meta = tmp_path / "development_splits_metadata.json"

    synthetic_df.to_csv(train_csv, index=False)
    train_sha = compute_file_sha256(train_csv)

    manifest_df = create_development_splits(
        train_df=synthetic_df,
        source_sha256=train_sha,
        holdout_fraction=0.20,
        primary_cv_folds=5,
        lightweight_cv_folds=3,
        seed=42,
        source_name="trainingData.csv",
    )
    manifest_sha, _ = save_split_manifest(
        manifest_df=manifest_df,
        manifest_csv_path=manifest_csv,
        metadata_json_path=manifest_meta,
    )
    return train_csv, manifest_csv, train_sha, manifest_sha


def test_uniform_weights_and_euclidean_metric_enforced() -> None:
    """1. Verify uniform weights and Euclidean distance are selected across config and pipeline builders."""
    cfg = load_pca_knn_config()
    assert cfg["model_id"] == MODEL_ID_PCA_KNN
    assert cfg["estimator"]["weights"] == "uniform"
    assert cfg["estimator"]["metric"] == "euclidean"

    for scaling in ("none", "standard"):
        pipe = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling=scaling)
        clf = pipe.named_steps["classifier"]
        assert clf.weights == "uniform"
        assert clf.metric == "euclidean"
        assert clf.n_neighbors == 3

    grid = generate_candidate_grid(pca_components=[5, 50], k_values=[1, 3], scaling_modes=["none", "standard"])
    assert len(grid) == 8
    assert all(c["weights"] == "uniform" and c["metric"] == "euclidean" for c in grid)


def test_paper_inspired_baseline_preserved(synthetic_uji_df: pd.DataFrame) -> None:
    """2. Verify paper-inspired fixed baseline (k=1, PCA=50, unscaled centered inputs, whiten=False) is preserved."""
    cfg = load_pca_knn_config()
    fb = cfg["fixed_baseline"]
    assert fb["n_neighbors"] == 1
    assert fb["n_components"] == 50
    assert fb["scaling"] == "none"
    assert fb["whiten"] is False
    assert fb["weights"] == "uniform"
    assert fb["metric"] == "euclidean"
    assert FIXED_BASELINE_PARAMS["n_neighbors"] == 1
    assert FIXED_BASELINE_PARAMS["n_components"] == 50
    assert FIXED_BASELINE_PARAMS["scale"] is False
    assert FIXED_BASELINE_PARAMS["whiten"] is False

    pipe = build_fixed_baseline_pipeline()
    assert "scaler" not in pipe.named_steps
    pca_step = pipe.named_steps["pca"]
    clf_step = pipe.named_steps["classifier"]
    assert pca_step.n_components == 50
    assert pca_step.whiten is False
    assert clf_step.n_neighbors == 1
    assert clf_step.weights == "uniform"
    assert clf_step.metric == "euclidean"

    # Fit on synthetic data (>= 51 rows so PCA=50 is feasible) and verify PCA centers inputs internally
    X, y, _, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    pipe.fit(X.iloc[:60], y.iloc[:60])
    assert hasattr(pca_step, "mean_")
    assert pca_step.components_.shape == (50, 520)


def test_pca_fitted_strictly_inside_cv_training_folds(synthetic_uji_df: pd.DataFrame) -> None:
    """3. Verify PCA and StandardScaler are fitted inside each CV training fold, never globally or on validation/holdout."""
    manifest_df = create_development_splits(synthetic_uji_df, seed=42)
    aligned_df = align_dataset_with_manifest(synthetic_uji_df, manifest_df)
    X, y, _, _ = extract_validated_wap_features_and_targets(aligned_df, require_row_id=True)

    folds = list(iter_building_cv_folds(manifest_df, building_id=0, n_splits=3))
    fold0_id, train_idx_0, val_idx_0 = folds[0]

    # Perturb validation and internal_holdout rows in a copy of X (keeping values within valid RSSI [-104, 0])
    X_perturbed = X.copy()
    holdout_idx = np.flatnonzero((aligned_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT).to_numpy())
    X_perturbed.iloc[val_idx_0, :20] = -1.0
    X_perturbed.iloc[holdout_idx, :20] = 0.0

    pipe_clean = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")
    pipe_pert = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")

    pipe_clean.fit(X.iloc[train_idx_0], y.iloc[train_idx_0])
    pipe_pert.fit(X_perturbed.iloc[train_idx_0], y.iloc[train_idx_0])

    # Because fitting occurs strictly on train_idx_0, changes to val_idx_0 and holdout_idx have zero effect
    np.testing.assert_allclose(
        pipe_clean.named_steps["scaler"].mean_,
        pipe_pert.named_steps["scaler"].mean_,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        pipe_clean.named_steps["pca"].components_,
        pipe_pert.named_steps["pca"].components_,
        rtol=0.0,
        atol=0.0,
    )

    # Also verify that distinct folds fit distinct scaler means and PCA subspaces
    _, train_idx_1, _ = folds[1]
    pipe_fold1 = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")
    pipe_fold1.fit(X.iloc[train_idx_1], y.iloc[train_idx_1])
    assert not np.allclose(
        pipe_clean.named_steps["scaler"].mean_,
        pipe_fold1.named_steps["scaler"].mean_,
    )
    assert not np.allclose(
        pipe_clean.named_steps["pca"].components_,
        pipe_fold1.named_steps["pca"].components_,
    )


def test_test_only_input_changes_cannot_alter_fitted_transforms(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """4. Verify test-only input changes cannot mutate training-fitted transforms or alter predictions on unchanged rows."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    train_out = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        pca_components=[5],
        k_values=[1, 3],
        scaling_modes=["standard"],
        run_id="run_immutability_check",
    )

    art_path_b0 = Path(train_out["selection_config"]["variants"]["tuned"]["artifact_paths"]["0"])
    pipe_before, _ = load_trusted_pipeline_artifact(art_path_b0, allowed_root=tmp_path)

    pca_mean_before = pipe_before.named_steps["pca"].mean_.copy()
    pca_comp_before = pipe_before.named_steps["pca"].components_.copy()
    scaler_mean_before = pipe_before.named_steps["scaler"].mean_.copy()
    scaler_scale_before = pipe_before.named_steps["scaler"].scale_.copy()
    knn_fit_x_before = pipe_before.named_steps["classifier"]._fit_X.copy()

    X, _, bldg, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    X_b0_test = X.loc[bldg == 0].iloc[:10].copy()
    preds_initial = pipe_before.predict(X_b0_test)

    # Create a heavily modified test batch where rows 1..9 have extreme valid RSSI values (-104 or 0),
    # while row 0 is kept identical.
    X_b0_modified = X_b0_test.copy()
    X_b0_modified.iloc[1:, :100] = 0
    X_b0_modified.iloc[1:, 100:200] = -104

    preds_modified = pipe_before.predict(X_b0_modified)

    # Row 0 prediction must be identical, and all fitted parameters must be untouched
    assert int(preds_initial[0]) == int(preds_modified[0])
    np.testing.assert_array_equal(pca_mean_before, pipe_before.named_steps["pca"].mean_)
    np.testing.assert_array_equal(pca_comp_before, pipe_before.named_steps["pca"].components_)
    np.testing.assert_array_equal(scaler_mean_before, pipe_before.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(scaler_scale_before, pipe_before.named_steps["scaler"].scale_)
    np.testing.assert_array_equal(knn_fit_x_before, pipe_before.named_steps["classifier"]._fit_X)


def test_infeasible_pca_and_k_candidates_rejected_with_skip_reasons(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """5. Verify infeasible PCA dimensions and neighbor counts k are rejected and recorded in search_results.csv."""
    # Building 0 has 60 rows total -> 48 dev_train rows -> 3-fold CV has 32 train rows per fold -> max PCA = 31, max k = 32
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    summary = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        cv_folds=3,
        pca_components=[5, 50, 200],  # 50 and 200 exceed min_fold_train_rows - 1 (31) for Buildings 0 & 1
        k_values=[1, 3, 100],         # k=100 exceeds min_fold_train_rows (32)
        scaling_modes=["none"],
        run_id="run_feasibility_test",
    )

    search_df = pd.read_csv(summary["search_results_csv"])
    b0_df = search_df[search_df["building_id"] == 0]

    # PCA=50 and PCA=200 must be skipped with PCA_COMPONENTS_EXCEED_MAX on Building 0
    pca50_rows = b0_df[b0_df["n_components"] == 50]
    assert not pca50_rows.empty
    assert (pca50_rows["status"] == "skipped").all()
    assert (~pca50_rows["feasible"]).all()
    assert (pca50_rows["reason_code"] == "PCA_COMPONENTS_EXCEED_MAX").all()

    # PCA=5, k=100 must be skipped with KNN_NEIGHBORS_EXCEED_TRAIN_ROWS on Building 0
    k100_row = b0_df[(b0_df["n_components"] == 5) & (b0_df["n_neighbors"] == 100)]
    assert len(k100_row) == 1
    assert k100_row.iloc[0]["status"] == "skipped"
    assert not bool(k100_row.iloc[0]["feasible"])
    assert k100_row.iloc[0]["reason_code"] == "KNN_NEIGHBORS_EXCEED_TRAIN_ROWS"

    # Feasible candidates (PCA=5, k=1 and k=3) must complete and one must be selected
    feasible_b0 = b0_df[b0_df["status"] == "completed"]
    assert len(feasible_b0) == 2
    assert int(b0_df["selected_for_building"].sum()) == 1


def test_missing_and_extra_predictor_leakage_caught(synthetic_uji_df: pd.DataFrame) -> None:
    """6. Verify missing WAP columns or metadata/target leakage in predictor columns are rejected."""
    # Missing WAP001
    df_missing = synthetic_uji_df.drop(columns=["WAP001"])
    with pytest.raises(DataValidationError, match="Missing 1 required WAP"):
        extract_validated_wap_features_and_targets(df_missing)

    # Extra metadata column passed in candidate_predictor_columns
    leaky_cols = [*WAP_COLUMNS[:-1], "BUILDINGID"]
    with pytest.raises(DataValidationError, match="Predictor leakage violation"):
        extract_validated_wap_features_and_targets(
            synthetic_uji_df,
            candidate_predictor_columns=leaky_cols,
        )

    # Misordered WAP columns inside pipeline with enforce_wap_schema=True
    X, y, _, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    X_swapped = X.loc[:, ["WAP002", "WAP001", *WAP_COLUMNS[2:]]]
    pipe = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=1, enforce_wap_schema=True)
    with pytest.raises(DataValidationError, match="canonical order"):
        pipe.fit(X_swapped, y)


def test_saved_and_reloaded_pipeline_inference_equivalence(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """7. Verify saved and reloaded pipelines produce bit-for-bit identical predictions and reject tampering."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    summary = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        pca_components=[5],
        k_values=[3],
        scaling_modes=["standard"],
        run_id="run_reload_equiv",
    )

    X, y, bldg, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    for b_id in (0, 1, 2):
        art_path = Path(summary["selection_config"]["variants"]["tuned"]["artifact_paths"][str(b_id)])
        loaded_pipe, meta = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)
        loaded_pipe_2, _ = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)

        X_b = X.loc[bldg == b_id]
        preds_1 = loaded_pipe.predict(X_b)
        preds_2 = loaded_pipe_2.predict(X_b)
        np.testing.assert_array_equal(preds_1, preds_2)

        # Also compare directly against a freshly fitted pipeline on the exact same dev_train rows
        manifest_df = pd.read_csv(manifest_csv)
        dev_mask = (manifest_df["building_id"] == b_id) & (manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN)
        fresh_pipe = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard", random_state=42)
        fresh_pipe.fit(X.loc[dev_mask], y.loc[dev_mask])
        np.testing.assert_array_equal(preds_1, fresh_pipe.predict(X_b))

    # Corrupting the artifact file must trigger UntrustedArtifactError on reload
    art_0 = Path(summary["selection_config"]["variants"]["tuned"]["artifact_paths"]["0"])
    art_0.write_bytes(art_0.read_bytes() + b"corrupted")
    with pytest.raises(UntrustedArtifactError, match="SHA-256 integrity check failed"):
        load_trusted_pipeline_artifact(art_0, allowed_root=tmp_path)


def test_end_to_end_train_and_holdout_metrics_schema_compatibility(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """8. Verify CV and internal_holdout outputs satisfy result_schema v1.0 and keep variants distinguishable."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    train_summary = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        cv_folds=3,
        pca_components=[5],
        k_values=[1, 3],
        scaling_modes=["none", "standard"],
        run_id="train_schema_test",
    )

    # Validate CV result JSONs and OOF prediction CSVs
    cv_tuned = load_result_json(train_summary["cv_tuned_metrics_json"], validate=True)
    assert cv_tuned["evaluation_role"] == EVALUATION_ROLE_CV_VAL
    assert cv_tuned["status"] == "completed"
    cv_oof_df = load_predictions_csv(cv_tuned["prediction_file_path"], evaluation_role=EVALUATION_ROLE_CV_VAL)
    assert len(cv_oof_df) == int(cv_tuned["pooled"]["evaluated_row_count"])

    cv_fixed = load_result_json(train_summary["cv_fixed_baseline_metrics_json"], validate=True)
    assert cv_fixed["evaluation_role"] == EVALUATION_ROLE_CV_VAL

    # Evaluate on internal_holdout via test.py
    eval_summary = evaluate_pca_knn(
        eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        variant="both",
        train_run_dir=train_summary["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        allowed_root=tmp_path,
        eval_run_id="eval_holdout_schema_test",
    )

    tuned_info = eval_summary["variants"]["tuned"]
    fixed_info = eval_summary["variants"]["fixed_baseline"]
    assert tuned_info["metrics_json"] != fixed_info["metrics_json"]

    holdout_tuned = load_result_json(tuned_info["metrics_json"], validate=True)
    validate_result_payload(holdout_tuned)
    assert holdout_tuned["evaluation_role"] == EVALUATION_ROLE_INTERNAL_HOLDOUT
    assert holdout_tuned["status"] == "completed"

    holdout_preds = load_predictions_csv(
        tuned_info["predictions_csv"],
        evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT,
    )
    validate_predictions_dataframe(holdout_preds, evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT)
    assert len(holdout_preds) == int(holdout_tuned["pooled"]["evaluated_row_count"])


def test_development_mode_blocks_official_holdout_and_requires_confirmation(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """9. Verify development mode cannot implicitly open official holdout and official mode requires confirmation."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    val_csv = tmp_path / "validationData.csv"
    synthetic_uji_df.to_csv(val_csv, index=False)

    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    # train.py must reject validationData.csv
    with pytest.raises(PermissionError, match="validationData.csv"):
        train_pca_knn(
            train_csv_path=val_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            profile="smoke",
        )

    train_summary = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        run_id="run_guard_test",
    )

    # Default internal_holdout mode must reject --val-csv
    with pytest.raises(PermissionError, match="Development mode"):
        evaluate_pca_knn(
            eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
            train_run_dir=train_summary["artifacts_dir"],
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            val_csv_path=val_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            allowed_root=tmp_path,
        )

    # official_validation mode without confirm_final_evaluation=True must raise PermissionError
    with pytest.raises(PermissionError, match="confirm-final-evaluation"):
        evaluate_pca_knn(
            eval_mode=EVALUATION_ROLE_OFFICIAL_VALIDATION,
            confirm_final_evaluation=False,
            train_run_dir=train_summary["artifacts_dir"],
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            val_csv_path=val_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            allowed_root=tmp_path,
        )

    # Explicit confirmation + explicit refit on all training rows must succeed and record refit provenance
    official_summary = evaluate_pca_knn(
        eval_mode=EVALUATION_ROLE_OFFICIAL_VALIDATION,
        variant="tuned",
        confirm_final_evaluation=True,
        refit_all_training=True,
        train_run_dir=train_summary["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        val_csv_path=val_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        allowed_root=tmp_path,
        eval_run_id="eval_official_confirmed",
    )
    assert official_summary["confirm_final_evaluation"] is True
    assert official_summary["explicit_final_refit_all_training_rows"] is True
    official_res = load_result_json(official_summary["variants"]["tuned"]["metrics_json"], validate=True)
    assert official_res["evaluation_role"] == EVALUATION_ROLE_OFFICIAL_VALIDATION
    assert official_res["preprocessing"]["fit_scope"] == "all_official_training_rows_explicit_final_refit"


def test_non_contiguous_original_floor_labels_preserved(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """10. Verify non-contiguous original floor labels (e.g. [0, 2, 4]) are preserved without zero-indexing."""
    df_non_contig = synthetic_uji_df.copy()
    # Map Building 0 floors {0, 1, 2, 3} -> {0, 2, 4, 4} so floors 1 and 3 are absent in Building 0
    b0_mask = df_non_contig[BUILDING_COLUMN] == 0
    floor_map = {0: 0, 1: 2, 2: 4, 3: 4}
    df_non_contig.loc[b0_mask, TARGET_COLUMN] = df_non_contig.loc[b0_mask, TARGET_COLUMN].map(floor_map)

    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, df_non_contig)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    train_summary = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        pca_components=[5],
        k_values=[1],
        scaling_modes=["none"],
        run_id="run_non_contig_floors",
    )

    eval_summary = evaluate_pca_knn(
        eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        variant="tuned",
        train_run_dir=train_summary["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        allowed_root=tmp_path,
        eval_run_id="eval_non_contig_floors",
    )

    metrics = load_result_json(eval_summary["variants"]["tuned"]["metrics_json"])
    b0_per_class = metrics["per_building"]["0"]["per_class"]
    b0_cm_labels = metrics["confusion_matrices"]["0"]["labels"]

    assert sorted(b0_per_class.keys()) == ["0", "2", "4"]
    assert b0_cm_labels == [0, 2, 4]

    preds_df = load_predictions_csv(eval_summary["variants"]["tuned"]["predictions_csv"])
    b0_preds = preds_df[preds_df["building_id"] == 0]
    assert set(b0_preds["predicted_floor"].unique()).issubset({0, 2, 4})
    assert set(b0_preds["true_floor"].unique()) == {0, 2, 4}


def test_parallelism_prevents_nested_workers_and_all_cores() -> None:
    """Verify parallelism configuration rejects -1 and forces estimator_n_jobs=1 when search_workers > 1."""
    with pytest.raises(ValueError, match="prohibited"):
        resolve_parallelism(search_workers=-1, estimator_n_jobs=1)
    with pytest.raises(ValueError, match="prohibited"):
        resolve_parallelism(search_workers=1, estimator_n_jobs=0)

    s_workers, est_jobs, warnings = resolve_parallelism(search_workers=2, estimator_n_jobs=4)
    assert s_workers == 2
    assert est_jobs == 1
    assert len(warnings) == 1


def test_clean_data_stateless_and_requires_justification_for_exclusions(
    tmp_path: Path,
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """Verify clean_data preserves row IDs and split roles, converts 100 -> -110.0, and requires exclusion justification."""
    train_csv, manifest_csv, train_sha, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    manifest_df = pd.read_csv(manifest_csv)

    cleaned_df, report = clean_pca_knn_dataframe(
        df=synthetic_uji_df,
        manifest_df=manifest_df,
        dataset_role="train",
        source_sha256=train_sha,
    )
    assert len(cleaned_df) == len(synthetic_uji_df)
    assert report["leakage_safeguards"]["global_scaling_fitted"] is False
    assert report["leakage_safeguards"]["global_pca_fitted"] is False
    assert not (cleaned_df.loc[:, list(WAP_COLUMNS)].to_numpy() == 100).any()
    assert (cleaned_df.loc[:, list(WAP_COLUMNS)].to_numpy() == -110.0).any()
    # Genuine 0 dBm preserved
    assert (cleaned_df.loc[:, list(WAP_COLUMNS)].to_numpy() == 0.0).any()

    # Exclusion without justification must raise ValueError
    first_rid = str(cleaned_df["row_id"].iloc[0])
    with pytest.raises(ValueError, match="exclusion_justification"):
        clean_pca_knn_dataframe(
            df=synthetic_uji_df,
            manifest_df=manifest_df,
            dataset_role="train",
            source_sha256=train_sha,
            exclude_row_ids=[first_rid],
            exclusion_justification="",
        )

    # End-to-end run_clean_data CLI runner
    out_dir = tmp_path / "processed"
    run_report = run_clean_data(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        output_dir=out_dir,
    )
    assert Path(run_report["output_csv_path"]).is_file()
    assert (out_dir / "cleaning_manifest.json").is_file()


def test_deterministic_candidate_tie_breaking() -> None:
    """Verify select_best_pca_knn_candidate enforces accuracy -> macro_f1 -> smaller PCA -> smaller k -> scaling='none'."""
    tied_candidates = [
        {
            "candidate_index": 3,
            "candidate_id": "pca50_k5_scale_standard",
            "n_components": 50,
            "n_neighbors": 5,
            "scaling": "standard",
            "feasible": True,
            "status": "completed",
            "mean_cv_accuracy": 0.9000,
            "mean_cv_macro_f1": 0.8800,
        },
        {
            "candidate_index": 2,
            "candidate_id": "pca5_k3_scale_standard",
            "n_components": 5,
            "n_neighbors": 3,
            "scaling": "standard",
            "feasible": True,
            "status": "completed",
            "mean_cv_accuracy": 0.9000,
            "mean_cv_macro_f1": 0.8800,
        },
        {
            "candidate_index": 1,
            "candidate_id": "pca5_k3_scale_none",
            "n_components": 5,
            "n_neighbors": 3,
            "scaling": "none",
            "feasible": True,
            "status": "completed",
            "mean_cv_accuracy": 0.9000,
            "mean_cv_macro_f1": 0.8800,
        },
    ]
    winner = select_best_pca_knn_candidate(tied_candidates)
    assert winner["candidate_id"] == "pca5_k3_scale_none"
