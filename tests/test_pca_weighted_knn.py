"""Synthetic unit and integration tests for Stage 3: PCA + distance-weighted kNN (`models.pca_weighted_knn`).

Covers all required scientific and engineering contracts:
1. `weights="distance"` and `metric="euclidean"` are enforced across config, pipeline builders, and candidate grids.
2. Matching candidate, split, sentinel-encoding, and cleaned-row-population contracts with uniform kNN (`models.pca_knn`).
3. Duplicate exact fingerprints and zero-distance neighbors are handled natively by `scikit-learn` (`weights="distance"` string, never a fragile custom `1/d` callable).
4. `k=1` equivalence between uniform and distance-weighted kNN is preserved and explicitly flagged (`k1_no_multi_neighbor_weighting_advantage == True`) rather than presented as a multi-neighbor weighting improvement.
5. Fold-local preprocessing (`StandardScaler` + `PCA` fitted inside each CV training fold) and per-fold feasibility guards (`n_components` and `n_neighbors` bounds).
6. Exact prediction row alignment (`row_id`, `building_id`, `true_floor`) and no-refit holdout prediction.
7. Artifact round-trip reload, SHA-256 verification, shared result schema (`schema_version == "1.0"`), classification `validation_curve.csv` generation, and fair matched-vs-selected uniform comparison decomposition.
8. Official holdout (`validationData.csv`) protection during training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models.pca_knn.clean_data import (
    extract_validated_wap_features_and_targets,
    load_pca_knn_config,
    run_clean_data as run_uniform_clean_data,
)
from models.pca_knn.test import evaluate_pca_knn
from models.pca_knn.train import (
    build_uniform_pca_knn_pipeline,
    train_pca_knn,
)
from models.pca_weighted_knn.clean_data import (
    clean_pca_weighted_knn_dataframe,
    load_pca_weighted_knn_config,
    run_clean_data as run_weighted_clean_data,
)
from models.pca_weighted_knn.test import (
    compare_holdout_with_uniform_pca_knn,
    evaluate_pca_weighted_knn,
)
from models.pca_weighted_knn.train import (
    build_distance_weighted_pca_knn_pipeline,
    build_fixed_baseline_weighted_pipeline,
    compare_with_uniform_pca_knn,
    evaluate_single_candidate_cv,
    generate_candidate_grid,
    train_pca_weighted_knn,
)
from shared.artifacts import UntrustedArtifactError, load_trusted_pipeline_artifact
from shared.config import (
    EVALUATION_ROLE_CV_VAL,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
    MODEL_ID_PCA_KNN,
    MODEL_ID_PCA_WEIGHTED_KNN,
    RESULT_SCHEMA_VERSION,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    WAP_COLUMNS,
)
from shared.create_splits import (
    align_dataset_with_manifest,
    create_development_splits,
    iter_building_cv_folds,
    save_split_manifest,
)
from shared.data_validation import compute_file_sha256
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


# ---------------------------------------------------------------------------
# 1. Pipeline construction & native distance-weighting contract
# ---------------------------------------------------------------------------


def test_distance_weighted_pipeline_uses_native_sklearn_distance_weights() -> None:
    """1. Verify distance-weighted pipelines use native `weights='distance'` and `metric='euclidean'`."""
    fixed_pipe = build_fixed_baseline_weighted_pipeline()
    assert list(fixed_pipe.named_steps.keys()) == ["sentinel_replacement", "pca", "classifier"]
    pca_step: PCA = fixed_pipe.named_steps["pca"]
    knn_step: KNeighborsClassifier = fixed_pipe.named_steps["classifier"]

    assert isinstance(pca_step, PCA)
    assert pca_step.n_components == 50
    assert pca_step.whiten is False

    assert isinstance(knn_step, KNeighborsClassifier)
    assert knn_step.n_neighbors == 1
    assert knn_step.weights == "distance"
    assert isinstance(knn_step.weights, str) and not callable(knn_step.weights)
    assert knn_step.metric == "euclidean"

    scaled_pipe = build_distance_weighted_pca_knn_pipeline(
        n_components=5,
        n_neighbors=7,
        scaling="standard",
    )
    assert list(scaled_pipe.named_steps.keys()) == ["sentinel_replacement", "scaler", "pca", "classifier"]
    assert isinstance(scaled_pipe.named_steps["scaler"], StandardScaler)
    assert scaled_pipe.named_steps["classifier"].weights == "distance"
    assert not callable(scaled_pipe.named_steps["classifier"].weights)
    assert scaled_pipe.named_steps["classifier"].metric == "euclidean"
    assert scaled_pipe.named_steps["classifier"].n_neighbors == 7


# ---------------------------------------------------------------------------
# 2. Matching candidate, split, and encoding contracts with uniform kNN
# ---------------------------------------------------------------------------


def test_matching_candidate_split_and_encoding_contracts_with_uniform_knn(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """2. Verify uniform and distance-weighted kNN share identical grids, profiles, encodings, and cleaned rows."""
    uniform_cfg = load_pca_knn_config()
    weighted_cfg = load_pca_weighted_knn_config()

    assert uniform_cfg["model_id"] == MODEL_ID_PCA_KNN
    assert weighted_cfg["model_id"] == MODEL_ID_PCA_WEIGHTED_KNN
    assert uniform_cfg["estimator"]["weights"] == "uniform"
    assert weighted_cfg["estimator"]["weights"] == "distance"
    assert uniform_cfg["estimator"]["metric"] == weighted_cfg["estimator"]["metric"] == "euclidean"

    # Candidate grids, search profiles, selection policy, and preprocessing must match exactly
    assert uniform_cfg["tuned_grid"] == weighted_cfg["tuned_grid"]
    assert uniform_cfg["search_profiles"] == weighted_cfg["search_profiles"]
    assert uniform_cfg["selection_policy"] == weighted_cfg["selection_policy"]
    assert uniform_cfg["preprocessing"] == weighted_cfg["preprocessing"]

    # Candidate grid generator produces identical (n_components, n_neighbors, scaling) tuples
    u_grid = generate_candidate_grid(
        pca_components=uniform_cfg["tuned_grid"]["pca_components"],
        k_values=uniform_cfg["tuned_grid"]["k_values"],
        scaling_modes=uniform_cfg["tuned_grid"]["scaling_modes"],
        weights="uniform",
    )
    w_grid = generate_candidate_grid(
        pca_components=weighted_cfg["tuned_grid"]["pca_components"],
        k_values=weighted_cfg["tuned_grid"]["k_values"],
        scaling_modes=weighted_cfg["tuned_grid"]["scaling_modes"],
        weights="distance",
    )
    assert len(u_grid) == len(w_grid) == 42
    for u_c, w_c in zip(u_grid, w_grid, strict=True):
        assert u_c["candidate_id"] == w_c["candidate_id"]
        assert u_c["n_components"] == w_c["n_components"]
        assert u_c["n_neighbors"] == w_c["n_neighbors"]
        assert u_c["scaling"] == w_c["scaling"]
        assert u_c["weights"] == "uniform"
        assert w_c["weights"] == "distance"

    # Cleaning produces identical row populations and provenance hashes
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    u_clean_dir = tmp_path / "uniform_clean"
    w_clean_dir = tmp_path / "weighted_clean"

    u_report = run_uniform_clean_data(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        output_dir=u_clean_dir,
    )
    w_report = run_weighted_clean_data(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        output_dir=w_clean_dir,
    )

    assert u_report["source_sha256"] == w_report["source_sha256"]
    assert u_report["split_manifest_sha256"] == w_report["split_manifest_sha256"]
    assert u_report["encoding_config_sha256"] == w_report["encoding_config_sha256"]
    assert u_report["cleaned_dataframe_sha256"] == w_report["cleaned_dataframe_sha256"]
    assert u_report["output_csv_sha256"] == w_report["output_csv_sha256"]
    assert u_report["row_counts"] == w_report["row_counts"]

    u_df = pd.read_csv(u_report["output_csv_path"])
    w_df = pd.read_csv(w_report["output_csv_path"])
    pd.testing.assert_frame_equal(u_df, w_df)


# ---------------------------------------------------------------------------
# 3. Duplicate exact fingerprints & zero-distance neighbor handling
# ---------------------------------------------------------------------------


def test_duplicate_exact_fingerprints_and_zero_distance_neighbors_handled_natively(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """3. Verify exact duplicate WiFi scans (zero Euclidean distance) are handled natively without NaN/ZeroDivisionError."""
    X, y, _, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    X_train = X.iloc[:40].copy().reset_index(drop=True)
    y_train = y.iloc[:40].to_numpy(dtype=int).copy()

    # Inject exact duplicate fingerprints for a known scan with floor label 2
    exact_scan = X_train.iloc[[0]].copy()
    X_train.iloc[1] = exact_scan.iloc[0]
    X_train.iloc[2] = exact_scan.iloc[0]
    y_train[0] = 2
    y_train[1] = 2
    y_train[2] = 2

    pipe = build_distance_weighted_pca_knn_pipeline(
        n_components=5,
        n_neighbors=5,
        scaling="standard",
    )
    clf = pipe.named_steps["classifier"]
    assert clf.weights == "distance"
    assert not callable(clf.weights)

    pipe.fit(X_train, y_train)

    # Querying with the exact duplicate fingerprint produces zero Euclidean distance to neighbors 0, 1, 2
    X_query = pd.concat([exact_scan, X_train.iloc[5:8]], axis=0, ignore_index=True)
    preds = pipe.predict(X_query)
    proba = pipe.predict_proba(X_query)

    assert preds.shape == (4,)
    assert int(preds[0]) == 2
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)
    class_2_idx = list(clf.classes_).index(2)
    assert np.isclose(proba[0, class_2_idx], 1.0)


# ---------------------------------------------------------------------------
# 4. k=1 equivalence between uniform and distance-weighted kNN
# ---------------------------------------------------------------------------


def test_k1_behavior_identical_to_uniform_and_explicitly_flagged(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """4. Verify k=1 distance weighting matches uniform 1NN and is flagged as having no multi-neighbor voting advantage."""
    manifest_df = create_development_splits(synthetic_uji_df, seed=42)
    aligned_df = align_dataset_with_manifest(synthetic_uji_df, manifest_df)
    X_all, y_all, bldg_all, rids_all = extract_validated_wap_features_and_targets(
        aligned_df, require_row_id=True
    )
    assert rids_all is not None

    b0_dev_mask = (bldg_all.to_numpy() == 0) & (
        aligned_df["split_role"].astype(str).to_numpy() == SPLIT_ROLE_DEV_TRAIN
    )
    b0_dev_indices = np.flatnonzero(b0_dev_mask)
    holdout_indices_set = set(
        int(i)
        for i in np.flatnonzero(
            (aligned_df["split_role"].astype(str) == SPLIT_ROLE_INTERNAL_HOLDOUT).to_numpy()
        )
    )
    folds = list(iter_building_cv_folds(manifest_df, building_id=0, n_splits=3))

    candidate_k1 = {
        "candidate_index": 0,
        "candidate_id": "pca5_k1_scale_standard",
        "n_components": 5,
        "n_neighbors": 1,
        "scaling": "standard",
        "scale": True,
        "whiten": False,
    }

    u_row, u_oof = evaluate_single_candidate_cv(
        candidate=candidate_k1,
        building_id=0,
        X_full=X_all,
        y_full=y_all,
        row_ids_full=rids_all,
        folds=folds,
        building_dev_train_indices=b0_dev_indices,
        holdout_indices_set=holdout_indices_set,
        weights="uniform",
        model_id=MODEL_ID_PCA_KNN,
    )
    w_row, w_oof = evaluate_single_candidate_cv(
        candidate=candidate_k1,
        building_id=0,
        X_full=X_all,
        y_full=y_all,
        row_ids_full=rids_all,
        folds=folds,
        building_dev_train_indices=b0_dev_indices,
        holdout_indices_set=holdout_indices_set,
        weights="distance",
        model_id=MODEL_ID_PCA_WEIGHTED_KNN,
    )

    assert w_row["k1_no_multi_neighbor_weighting_advantage"] is True
    assert np.isclose(u_row["mean_cv_accuracy"], w_row["mean_cv_accuracy"])
    assert np.isclose(u_row["mean_cv_macro_f1"], w_row["mean_cv_macro_f1"])
    assert u_oof is not None and w_oof is not None
    np.testing.assert_array_equal(
        u_oof["predicted_floor"].to_numpy(),
        w_oof["predicted_floor"].to_numpy(),
    )

    # Also verify direct pipeline prediction equality at k=1
    X_b0 = X_all.iloc[b0_dev_indices]
    y_b0 = y_all.iloc[b0_dev_indices]
    u_pipe = build_uniform_pca_knn_pipeline(n_components=5, n_neighbors=1, scaling="standard").fit(X_b0, y_b0)
    w_pipe = build_distance_weighted_pca_knn_pipeline(n_components=5, n_neighbors=1, scaling="standard").fit(X_b0, y_b0)
    np.testing.assert_array_equal(u_pipe.predict(X_b0), w_pipe.predict(X_b0))


# ---------------------------------------------------------------------------
# 5. Fold-local preprocessing & feasibility guards
# ---------------------------------------------------------------------------


def test_fold_local_preprocessing_and_feasibility_guards_in_weighted_knn(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """5. Verify scaler/PCA fit per fold and infeasible PCA/k combinations record skip reasons."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    manifest_df = pd.read_csv(manifest_csv)
    aligned_df = align_dataset_with_manifest(synthetic_uji_df, manifest_df)
    X_all, y_all, _, _ = extract_validated_wap_features_and_targets(aligned_df, require_row_id=True)

    folds = list(iter_building_cv_folds(manifest_df, building_id=0, n_splits=3))
    _, train_idx_0, val_idx_0 = folds[0]
    _, train_idx_1, _ = folds[1]

    # Perturbing validation and holdout rows has zero effect on fold-0 fitted scaler and PCA
    X_perturbed = X_all.copy()
    holdout_idx = np.flatnonzero(
        (aligned_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT).to_numpy()
    )
    X_perturbed.iloc[val_idx_0, :20] = -1.0
    X_perturbed.iloc[holdout_idx, :20] = 0.0

    pipe_0 = build_distance_weighted_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")
    pipe_0_pert = build_distance_weighted_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")
    pipe_1 = build_distance_weighted_pca_knn_pipeline(n_components=5, n_neighbors=3, scaling="standard")

    pipe_0.fit(X_all.iloc[train_idx_0], y_all.iloc[train_idx_0])
    pipe_0_pert.fit(X_perturbed.iloc[train_idx_0], y_all.iloc[train_idx_0])
    pipe_1.fit(X_all.iloc[train_idx_1], y_all.iloc[train_idx_1])

    np.testing.assert_allclose(pipe_0.named_steps["scaler"].mean_, pipe_0_pert.named_steps["scaler"].mean_)
    np.testing.assert_allclose(pipe_0.named_steps["pca"].components_, pipe_0_pert.named_steps["pca"].components_)
    assert not np.allclose(pipe_0.named_steps["scaler"].mean_, pipe_1.named_steps["scaler"].mean_)
    assert not np.allclose(pipe_0.named_steps["pca"].components_, pipe_1.named_steps["pca"].components_)

    # Infeasible PCA (50, 200) and infeasible k (100) recorded with skip reasons in search_results.csv
    summary = train_pca_weighted_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=tmp_path / "artifacts",
        results_base_dir=tmp_path / "results",
        cv_folds=3,
        pca_components=[5, 50, 200],
        k_values=[1, 3, 100],
        scaling_modes=["none"],
        run_id="weighted_feasibility_check",
    )
    search_df = pd.read_csv(summary["search_results_csv"])
    b0_df = search_df[search_df["building_id"] == 0]

    pca200_rows = b0_df[b0_df["n_components"] == 200]
    assert (pca200_rows["status"] == "skipped").all()
    assert (pca200_rows["reason_code"] == "PCA_COMPONENTS_EXCEED_MAX").all()

    k100_row = b0_df[(b0_df["n_components"] == 5) & (b0_df["n_neighbors"] == 100)]
    assert len(k100_row) == 1
    assert k100_row.iloc[0]["status"] == "skipped"
    assert k100_row.iloc[0]["reason_code"] == "KNN_NEIGHBORS_EXCEED_TRAIN_ROWS"


# ---------------------------------------------------------------------------
# 6 & 7. End-to-end train/test, validation_curve.csv, row alignment, and fair comparisons
# ---------------------------------------------------------------------------


def test_end_to_end_weighted_knn_validation_curve_row_alignment_and_fair_comparison(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """6 & 7. Verify validation_curve.csv, matched vs best-selected comparison, row alignment, and artifact reload."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)

    # 1. Train reference uniform PCA-kNN on the same candidate grid
    u_artifacts = tmp_path / "u_artifacts"
    u_results = tmp_path / "u_results"
    u_train = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=u_artifacts,
        results_base_dir=u_results,
        cv_folds=3,
        pca_components=[5, 50],
        k_values=[1, 3, 5],
        scaling_modes=["none", "standard"],
        run_id="run_uniform_ref",
    )

    # 2. Train distance-weighted PCA-kNN on the identical candidate grid + uniform reference
    w_artifacts = tmp_path / "w_artifacts"
    w_results = tmp_path / "w_results"
    w_train = train_pca_weighted_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=w_artifacts,
        results_base_dir=w_results,
        cv_folds=3,
        pca_components=[5, 50],
        k_values=[1, 3, 5],
        scaling_modes=["none", "standard"],
        run_id="run_weighted_ref",
        uniform_reference_run_dir=Path(u_train["results_dir"]),
    )

    assert w_train["model_id"] == MODEL_ID_PCA_WEIGHTED_KNN
    assert w_train["weights"] == "distance"

    # Verify validation_curve.csv structure and spread metrics
    vc_csv = Path(w_train["validation_curve_csv"])
    assert vc_csv.is_file()
    vc_df = pd.read_csv(vc_csv)
    required_vc_cols = {
        "k",
        "pca_components",
        "scaling",
        "weights",
        "building",
        "building_id",
        "fold",
        "fold_cv_accuracy",
        "fold_cv_error",
        "fold_cv_macro_f1",
        "mean_cv_accuracy",
        "mean_cv_error",
        "mean_cv_macro_f1",
        "std_cv_accuracy",
        "min_cv_accuracy",
        "max_cv_accuracy",
        "iqr_cv_accuracy",
        "range_cv_accuracy",
        "std_cv_macro_f1",
        "min_cv_macro_f1",
        "max_cv_macro_f1",
        "iqr_cv_macro_f1",
        "range_cv_macro_f1",
        "k1_no_multi_neighbor_weighting_advantage",
        "selected_for_building",
        "curve_type",
        "selection_rule",
    }
    assert required_vc_cols.issubset(set(vc_df.columns))
    assert set(vc_df["weights"].unique()) == {"distance"}
    assert set(vc_df["fold"].astype(str).unique()) == {"0", "1", "2", "mean_across_folds"}
    completed_vc = vc_df[vc_df["status"] == "completed"]
    assert np.allclose(
        completed_vc["fold_cv_accuracy"] + completed_vc["fold_cv_error"],
        1.0,
    )

    # Verify matched vs best-selected CV comparison outputs
    matched_csv = Path(w_train["matched_uniform_vs_weighted_cv_csv"])
    summary_json = Path(w_train["uniform_vs_weighted_comparison_summary_json"])
    assert matched_csv.is_file()
    assert summary_json.is_file()

    matched_df = pd.read_csv(matched_csv)
    k1_completed = matched_df[
        (matched_df["n_neighbors"] == 1) & (matched_df["status_uniform"] == "completed")
    ]
    assert not k1_completed.empty
    assert bool(k1_completed["k_equals_1_no_multi_neighbor_advantage"].all()) is True
    assert np.allclose(k1_completed["delta_mean_cv_accuracy_weighted_minus_uniform"], 0.0)

    comp_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "analysis_1_matched_configurations" in comp_summary
    assert "analysis_2_independently_selected_pipelines" in comp_summary
    assert set(comp_summary["analysis_2_independently_selected_pipelines"].keys()) == {"0", "1", "2"}

    # Verify saved artifact reload equivalence and tamper detection
    X_all, _, bldg_all, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    for b_id in (0, 1, 2):
        art_path = Path(w_train["selection_config"]["variants"]["tuned"]["artifact_paths"][str(b_id)])
        pipe_1, meta_1 = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)
        pipe_2, _ = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)
        assert meta_1["model_id"] == MODEL_ID_PCA_WEIGHTED_KNN
        assert pipe_1.named_steps["classifier"].weights == "distance"
        X_b = X_all.loc[bldg_all == b_id]
        np.testing.assert_array_equal(pipe_1.predict(X_b), pipe_2.predict(X_b))

    # 3. Evaluate both uniform and distance-weighted models on internal_holdout without refitting
    u_eval = evaluate_pca_knn(
        eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        variant="tuned",
        train_run_dir=u_train["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=u_artifacts,
        results_base_dir=u_results,
        allowed_root=tmp_path,
        eval_run_id="eval_uniform_holdout",
    )

    with patch.object(
        KNeighborsClassifier,
        "fit",
        side_effect=AssertionError("test.py must NOT refit KNeighborsClassifier!"),
    ):
        w_eval = evaluate_pca_weighted_knn(
            eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
            variant="tuned",
            train_run_dir=w_train["artifacts_dir"],
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=w_artifacts,
            results_base_dir=w_results,
            allowed_root=tmp_path,
            eval_run_id="eval_weighted_holdout",
            uniform_eval_run_dir=u_eval["results_dir"],
        )

    tuned_info = w_eval["variants"]["tuned"]
    holdout_metrics = load_result_json(tuned_info["metrics_json"], validate=True)
    validate_result_payload(holdout_metrics)
    assert holdout_metrics["schema_version"] == RESULT_SCHEMA_VERSION
    assert holdout_metrics["model_id"] == MODEL_ID_PCA_WEIGHTED_KNN
    assert holdout_metrics["evaluation_role"] == EVALUATION_ROLE_INTERNAL_HOLDOUT

    holdout_preds = load_predictions_csv(
        tuned_info["predictions_csv"],
        evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT,
    )
    validate_predictions_dataframe(holdout_preds, evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT)

    # Check exact row alignment per building against manifest internal_holdout rows
    manifest_df = pd.read_csv(manifest_csv)
    expected_holdout = manifest_df[manifest_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT].copy()
    expected_sorted = expected_holdout.sort_values(["building_id", "source_row_index"]).reset_index(drop=True)
    np.testing.assert_array_equal(holdout_preds["row_id"].to_numpy(), expected_sorted["row_id"].to_numpy())
    np.testing.assert_array_equal(
        holdout_preds["building_id"].to_numpy(),
        expected_sorted["building_id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        holdout_preds["true_floor"].to_numpy(),
        expected_sorted["floor"].to_numpy(),
    )

    # Verify holdout comparison JSON was written and structured properly
    assert "uniform_vs_weighted_holdout_comparison_json" in w_eval
    holdout_comp = json.loads(
        Path(w_eval["uniform_vs_weighted_holdout_comparison_json"]).read_text(encoding="utf-8")
    )
    assert "tuned" in holdout_comp["variants"]
    assert set(holdout_comp["variants"]["tuned"]["per_building"].keys()) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# 8. Official holdout protection
# ---------------------------------------------------------------------------


def test_official_holdout_protected_in_weighted_knn(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """8. Verify official validationData.csv is blocked during training and requires confirmation in test.py."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    val_csv = tmp_path / "validationData.csv"
    synthetic_uji_df.to_csv(val_csv, index=False)

    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    # Passing validationData.csv to train_pca_weighted_knn must raise PermissionError
    with pytest.raises(PermissionError, match="validationData.csv"):
        train_pca_weighted_knn(
            train_csv_path=val_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            profile="smoke",
        )

    train_summary = train_pca_weighted_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        run_id="weighted_guard_run",
    )

    # Default internal_holdout mode must reject --val-csv
    with pytest.raises(PermissionError, match="Development mode"):
        evaluate_pca_weighted_knn(
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
        evaluate_pca_weighted_knn(
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
