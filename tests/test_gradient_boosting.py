"""Synthetic unit and integration tests for Stage 4: Classical Gradient Boosting (`models.gradient_boosting`).

Covers all required scientific and engineering contracts:
1. Correct estimator family: exact `sklearn.ensemble.GradientBoostingClassifier` (no `XGBoost` or `HistGradientBoostingClassifier`), with paper-inspired starting configuration (`max_depth=4, n_estimators=189, learning_rate=0.1, subsample=1.0`).
2. No implicit PCA or standard scaling (`scale=False, use_pca=False` by default; pipeline contains only `sentinel_replacement` and `classifier`).
3. Finite input validation (`100 -> -110.0`, genuine `0` dBm preserved as `0.0`, `NaN` and `inf` rejected).
4. Shared row/split identity and encoding hash equivalence with `pca_knn` and `pca_weighted_knn`.
5. Original `FLOOR` label preservation (non-contiguous floor labels `[0, 2, 4]` never zero-indexed).
6. No early-stopping holdout leakage (`n_iter_no_change is None`) and bounded deterministic randomized search (`8` trials in `lightweight`, `16` in `full` out of `24`, recorded in `sampled_candidates.json`).
7. Save/load pipeline inference equivalence and SHA-256 tamper rejection.
8. Common metrics/predictions schema (`schema_version == "1.0"`), separate starting vs tuned variants, and separate search vs refit timing logs.
9. Official holdout (`validationData.csv`) protection during training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from models.gradient_boosting.clean_data import (
    clean_gradient_boosting_dataframe,
    extract_validated_wap_features_and_targets,
    load_gradient_boosting_config,
    run_clean_data as run_gb_clean_data,
)
from models.gradient_boosting.test import (
    evaluate_gradient_boosting,
    load_frozen_selection_config,
)
from models.gradient_boosting.train import (
    STARTING_GB_PARAMS,
    build_configured_gradient_boosting_pipeline,
    build_starting_gradient_boosting_pipeline,
    enumerate_full_candidate_space,
    resolve_gb_parallelism,
    sample_gradient_boosting_candidates,
    select_best_gb_candidate,
    train_gradient_boosting,
)
from models.pca_knn.clean_data import run_clean_data as run_uniform_clean_data
from models.pca_knn.test import evaluate_pca_knn
from models.pca_knn.train import train_pca_knn
from shared.artifacts import UntrustedArtifactError, load_trusted_pipeline_artifact
from shared.config import (
    BUILDING_COLUMN,
    EVALUATION_ROLE_CV_VAL,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
    MODEL_ID_GRADIENT_BOOSTING,
    RESULT_SCHEMA_VERSION,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    TARGET_COLUMN,
    WAP_COLUMNS,
)
from shared.create_splits import (
    create_development_splits,
    save_split_manifest,
)
from shared.data_validation import DataValidationError, compute_file_sha256
from shared.preprocessing import SentinelReplacementTransformer
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
# 1 & 2. Correct estimator family & no implicit PCA/scaling
# ---------------------------------------------------------------------------


def test_correct_estimator_family_and_no_implicit_pca_or_scaling() -> None:
    """1 & 2. Verify exact `GradientBoostingClassifier`, starting params, and no implicit PCA or StandardScaler."""
    cfg = load_gradient_boosting_config()
    assert cfg["model_id"] == MODEL_ID_GRADIENT_BOOSTING
    assert cfg["estimator"]["class_name"] == "sklearn.ensemble.GradientBoostingClassifier"
    assert cfg["estimator"]["n_iter_no_change"] is None
    assert cfg["estimator"]["early_stopping_enabled"] is False
    assert cfg["preprocessing"]["scale"] is False
    assert cfg["preprocessing"]["use_pca"] is False
    assert cfg["preprocessing"]["n_components"] is None

    fb = cfg["fixed_baseline"]
    assert fb["max_depth"] == 4
    assert fb["n_estimators"] == 189
    assert np.isclose(fb["learning_rate"], 0.1)
    assert fb["min_samples_leaf"] == 1
    assert np.isclose(fb["subsample"], 1.0)
    assert fb["max_features"] is None
    assert fb["scale"] is False
    assert fb["use_pca"] is False

    starting_pipe = build_starting_gradient_boosting_pipeline()
    assert list(starting_pipe.named_steps.keys()) == ["sentinel_replacement", "classifier"]
    assert "pca" not in starting_pipe.named_steps
    assert "scaler" not in starting_pipe.named_steps

    clf = starting_pipe.named_steps["classifier"]
    assert type(clf) is GradientBoostingClassifier
    assert not isinstance(clf, HistGradientBoostingClassifier)
    assert "xgboost" not in type(clf).__module__.lower()
    assert clf.max_depth == STARTING_GB_PARAMS["max_depth"] == 4
    assert clf.n_estimators == STARTING_GB_PARAMS["n_estimators"] == 189
    assert np.isclose(clf.learning_rate, STARTING_GB_PARAMS["learning_rate"])
    assert clf.min_samples_leaf == STARTING_GB_PARAMS["min_samples_leaf"] == 1
    assert np.isclose(clf.subsample, STARTING_GB_PARAMS["subsample"])
    assert clf.max_features is None
    assert clf.n_iter_no_change is None


# ---------------------------------------------------------------------------
# 3. Finite input validation
# ---------------------------------------------------------------------------


def test_finite_input_validation_and_sentinel_replacement(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """3. Verify 100 -> -110.0 finite encoding, genuine 0 dBm preservation, and rejection of NaN/inf."""
    manifest_df = create_development_splits(synthetic_uji_df, seed=42)
    cleaned_df, report = clean_gradient_boosting_dataframe(
        df=synthetic_uji_df,
        manifest_df=manifest_df,
        dataset_role="train",
    )
    wap_vals = cleaned_df.loc[:, list(WAP_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(wap_vals).all()
    assert not (wap_vals == 100.0).any()
    assert (wap_vals == -110.0).any()
    assert (wap_vals == 0.0).any()
    assert report["leakage_safeguards"]["global_scaling_fitted"] is False
    assert report["leakage_safeguards"]["global_pca_fitted"] is False
    assert report["leakage_safeguards"]["native_nan_handling_claimed"] is False

    # Inject NaN into a WAP column -> must raise DataValidationError / ValueError
    df_with_nan = synthetic_uji_df.copy()
    df_with_nan["WAP001"] = df_with_nan["WAP001"].astype(float)
    df_with_nan.loc[0, "WAP001"] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        extract_validated_wap_features_and_targets(df_with_nan)

    transformer = SentinelReplacementTransformer(enforce_wap_schema=True)
    with pytest.raises(ValueError, match="NaN or infinite"):
        transformer.fit_transform(df_with_nan.loc[:, list(WAP_COLUMNS)])

    # Inject +inf into a WAP column -> must raise DataValidationError / ValueError
    df_with_inf = synthetic_uji_df.copy()
    df_with_inf["WAP002"] = df_with_inf["WAP002"].astype(float)
    df_with_inf.loc[0, "WAP002"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        extract_validated_wap_features_and_targets(df_with_inf)


# ---------------------------------------------------------------------------
# 4. Shared row/split identity with PCA-kNN
# ---------------------------------------------------------------------------


def test_shared_row_and_split_identity_with_knn(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """4. Verify gradient_boosting and pca_knn produce identical cleaned row populations and split hashes."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    knn_clean_dir = tmp_path / "knn_clean"
    gb_clean_dir = tmp_path / "gb_clean"

    knn_report = run_uniform_clean_data(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        output_dir=knn_clean_dir,
    )
    gb_report = run_gb_clean_data(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        output_dir=gb_clean_dir,
    )

    assert knn_report["source_sha256"] == gb_report["source_sha256"]
    assert knn_report["split_manifest_sha256"] == gb_report["split_manifest_sha256"]
    assert knn_report["encoding_config_sha256"] == gb_report["encoding_config_sha256"]
    assert knn_report["cleaned_dataframe_sha256"] == gb_report["cleaned_dataframe_sha256"]
    assert knn_report["output_csv_sha256"] == gb_report["output_csv_sha256"]
    assert knn_report["row_counts"] == gb_report["row_counts"]

    knn_df = pd.read_csv(knn_report["output_csv_path"])
    gb_df = pd.read_csv(gb_report["output_csv_path"])
    pd.testing.assert_frame_equal(knn_df, gb_df)


# ---------------------------------------------------------------------------
# 5. Original label preservation
# ---------------------------------------------------------------------------


def test_original_non_contiguous_floor_labels_preserved(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """5. Verify non-contiguous floor labels (e.g. [0, 2, 4]) are preserved in classes_, predictions, and metrics."""
    df_non_contig = synthetic_uji_df.copy()
    b0_mask = df_non_contig[BUILDING_COLUMN] == 0
    floor_map = {0: 0, 1: 2, 2: 4, 3: 4}
    df_non_contig.loc[b0_mask, TARGET_COLUMN] = df_non_contig.loc[b0_mask, TARGET_COLUMN].map(floor_map)

    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, df_non_contig)
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    train_summary = train_gradient_boosting(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        profile="smoke",
        n_trials=1,
        n_estimators=[10],
        learning_rates=[0.1],
        max_depths=[3],
        min_samples_leaf=[1],
        run_id="gb_non_contig_floors",
    )

    art_b0 = Path(train_summary["selection_config"]["variants"]["tuned"]["artifact_paths"]["0"])
    loaded_pipe, meta = load_trusted_pipeline_artifact(art_b0, allowed_root=tmp_path)
    assert meta["class_labels"] == [0, 2, 4]
    assert [int(c) for c in loaded_pipe.named_steps["classifier"].classes_.tolist()] == [0, 2, 4]

    eval_summary = evaluate_gradient_boosting(
        eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        variant="tuned",
        train_run_dir=train_summary["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=artifacts_dir,
        results_base_dir=results_dir,
        allowed_root=tmp_path,
        eval_run_id="gb_eval_non_contig_floors",
    )

    metrics = load_result_json(eval_summary["variants"]["tuned"]["metrics_json"])
    assert sorted(metrics["per_building"]["0"]["per_class"].keys()) == ["0", "2", "4"]
    assert metrics["confusion_matrices"]["0"]["labels"] == [0, 2, 4]

    preds_df = load_predictions_csv(eval_summary["variants"]["tuned"]["predictions_csv"])
    b0_preds = preds_df[preds_df["building_id"] == 0]
    assert set(b0_preds["predicted_floor"].unique()).issubset({0, 2, 4})
    assert set(b0_preds["true_floor"].unique()) == {0, 2, 4}


# ---------------------------------------------------------------------------
# 6. Bounded deterministic randomized search & no early-stopping leakage
# ---------------------------------------------------------------------------


def test_bounded_deterministic_sampling_and_no_early_stopping() -> None:
    """6. Verify bounded deterministic randomized search (8 lightweight, 16 full out of 24) and no early stopping."""
    full_pool = enumerate_full_candidate_space()
    assert len(full_pool) == 24  # 3 * 2 * 2 * 2 = 24 combinations
    assert sum(1 for c in full_pool if c["is_starting_baseline"]) == 1

    # Lightweight mode: 8 trials (< 24 full product)
    lw_1, lw_meta_1 = sample_gradient_boosting_candidates(n_trials=8, seed=42)
    lw_2, _ = sample_gradient_boosting_candidates(n_trials=8, seed=42)
    assert len(lw_1) == 8
    assert lw_meta_1["total_cartesian_product_size"] == 24
    assert lw_meta_1["full_product_evaluated"] is False
    assert [c["candidate_id"] for c in lw_1] == [c["candidate_id"] for c in lw_2]
    assert any(c["is_starting_baseline"] for c in lw_1)
    assert all(c["n_iter_no_change"] is None for c in lw_1)

    # Full mode: 16 trials (< 24 full product)
    full_1, full_meta_1 = sample_gradient_boosting_candidates(n_trials=16, seed=42)
    assert len(full_1) == 16
    assert full_meta_1["total_cartesian_product_size"] == 24
    assert full_meta_1["full_product_evaluated"] is False
    assert any(c["is_starting_baseline"] for c in full_1)

    # Deterministic tie-breaking: shallower depth -> fewer estimators -> larger min_samples_leaf -> smaller lr
    tied = [
        {
            "candidate_index": 2,
            "candidate_id": "gb_est300_lr0.1_d4_leaf1",
            "max_depth": 4,
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "learning_rate": 0.1,
            "feasible": True,
            "status": "completed",
            "mean_cv_accuracy": 0.91,
            "mean_cv_macro_f1": 0.89,
        },
        {
            "candidate_index": 1,
            "candidate_id": "gb_est100_lr0.05_d3_leaf5",
            "max_depth": 3,
            "n_estimators": 100,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "feasible": True,
            "status": "completed",
            "mean_cv_accuracy": 0.91,
            "mean_cv_macro_f1": 0.89,
        },
    ]
    winner = select_best_gb_candidate(tied)
    assert winner["candidate_id"] == "gb_est100_lr0.05_d3_leaf5"

    # Parallelism guard: -1 is prohibited
    with pytest.raises(ValueError, match="prohibited"):
        resolve_gb_parallelism(search_workers=-1, blas_threads=1)
    w, t, warnings = resolve_gb_parallelism(search_workers=2, blas_threads=4)
    assert w == 2 and t == 1 and len(warnings) == 1


# ---------------------------------------------------------------------------
# 7 & 8. Save/load equivalence, schema v1.0, separate timing, and row alignment with kNN
# ---------------------------------------------------------------------------


def test_end_to_end_gb_save_load_schema_timing_and_knn_holdout_alignment(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """7 & 8. Verify save/load equivalence, schema v1.0, separate search vs refit timing, and exact holdout alignment."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    gb_artifacts = tmp_path / "gb_artifacts"
    gb_results = tmp_path / "gb_results"

    # Patch STARTING_GB_PARAMS n_estimators to 12 in this synthetic test only so 3 buildings x (3 folds + refit) runs fast
    fast_starting = {**STARTING_GB_PARAMS, "n_estimators": 12}
    with patch.dict("models.gradient_boosting.train.STARTING_GB_PARAMS", fast_starting):
        gb_train = train_gradient_boosting(
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=gb_artifacts,
            results_base_dir=gb_results,
            profile="smoke",
            cv_folds=3,
            n_trials=2,
            n_estimators=[8, 12],
            learning_rates=[0.1],
            max_depths=[3, 4],
            min_samples_leaf=[1],
            run_id="gb_smoke_run",
        )

    assert Path(gb_train["sampled_candidates_json"]).is_file()
    assert Path(gb_train["search_results_csv"]).is_file()
    sampled_meta = json.loads(Path(gb_train["sampled_candidates_json"]).read_text(encoding="utf-8"))
    assert sampled_meta["sampled_candidate_count"] == 2

    # Check CV result JSONs and separate search vs refit timing
    cv_tuned = load_result_json(gb_train["cv_tuned_metrics_json"], validate=True)
    cv_fixed = load_result_json(gb_train["cv_fixed_baseline_metrics_json"], validate=True)
    assert cv_tuned["evaluation_role"] == EVALUATION_ROLE_CV_VAL
    assert cv_fixed["evaluation_role"] == EVALUATION_ROLE_CV_VAL
    assert cv_tuned["preprocessing"]["pca_enabled"] is False
    assert cv_tuned["preprocessing"]["standard_scaler"] is False
    assert "cv_search_fit_seconds" in cv_tuned["timing"]
    assert "cv_search_total_seconds" in cv_tuned["timing"]
    assert "dev_train_refit_seconds" in cv_tuned["timing"]
    assert cv_tuned["timing"]["cv_search_fit_seconds"] >= 0.0
    assert cv_tuned["timing"]["dev_train_refit_seconds"] >= 0.0

    sel_cfg = load_frozen_selection_config(gb_train["selection_config_json"])
    assert sel_cfg["early_stopping_enabled"] is False
    assert "cv_search_fit_seconds" in sel_cfg["timing_summary"]
    assert "tuned_dev_train_refit_seconds" in sel_cfg["timing_summary"]

    # Verify save/load equivalence across all 3 buildings
    X_all, _, bldg_all, _ = extract_validated_wap_features_and_targets(synthetic_uji_df)
    for b_id in (0, 1, 2):
        art_path = Path(sel_cfg["variants"]["tuned"]["artifact_paths"][str(b_id)])
        pipe_1, _ = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)
        pipe_2, _ = load_trusted_pipeline_artifact(art_path, allowed_root=tmp_path)
        X_b = X_all.loc[bldg_all == b_id]
        np.testing.assert_array_equal(pipe_1.predict(X_b), pipe_2.predict(X_b))
        np.testing.assert_allclose(pipe_1.predict_proba(X_b), pipe_2.predict_proba(X_b))

    # Evaluate on internal_holdout without refitting
    with patch.object(
        GradientBoostingClassifier,
        "fit",
        side_effect=AssertionError("test.py must NOT refit GradientBoostingClassifier!"),
    ):
        gb_eval = evaluate_gradient_boosting(
            eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
            variant="both",
            train_run_dir=gb_train["artifacts_dir"],
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=gb_artifacts,
            results_base_dir=gb_results,
            allowed_root=tmp_path,
            eval_run_id="gb_holdout_eval",
        )

    assert "tuned" in gb_eval["variants"]
    assert "fixed_baseline" in gb_eval["variants"]
    assert gb_eval["variants"]["tuned"]["metrics_json"] != gb_eval["variants"]["fixed_baseline"]["metrics_json"]

    gb_holdout_metrics = load_result_json(gb_eval["variants"]["tuned"]["metrics_json"], validate=True)
    validate_result_payload(gb_holdout_metrics)
    assert gb_holdout_metrics["schema_version"] == RESULT_SCHEMA_VERSION
    assert gb_holdout_metrics["model_id"] == MODEL_ID_GRADIENT_BOOSTING

    gb_holdout_preds = load_predictions_csv(
        gb_eval["variants"]["tuned"]["predictions_csv"],
        evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT,
    )
    validate_predictions_dataframe(gb_holdout_preds, evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT)

    # Compare internal_holdout evaluation rows against PCA-kNN to verify identical row_id/building_id/true_floor
    knn_train = train_pca_knn(
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=tmp_path / "knn_artifacts",
        results_base_dir=tmp_path / "knn_results",
        profile="smoke",
        run_id="knn_ref_run",
    )
    knn_eval = evaluate_pca_knn(
        eval_mode=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        variant="tuned",
        train_run_dir=knn_train["artifacts_dir"],
        train_csv_path=train_csv,
        split_manifest_path=manifest_csv,
        artifacts_base_dir=tmp_path / "knn_artifacts",
        results_base_dir=tmp_path / "knn_results",
        allowed_root=tmp_path,
        eval_run_id="knn_holdout_eval",
    )
    knn_holdout_preds = load_predictions_csv(knn_eval["variants"]["tuned"]["predictions_csv"])

    np.testing.assert_array_equal(gb_holdout_preds["row_id"].to_numpy(), knn_holdout_preds["row_id"].to_numpy())
    np.testing.assert_array_equal(
        gb_holdout_preds["building_id"].to_numpy(),
        knn_holdout_preds["building_id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        gb_holdout_preds["true_floor"].to_numpy(),
        knn_holdout_preds["true_floor"].to_numpy(),
    )

    # Corrupting an artifact file must trigger UntrustedArtifactError
    art_0 = Path(sel_cfg["variants"]["tuned"]["artifact_paths"]["0"])
    art_0.write_bytes(art_0.read_bytes() + b"tampered")
    with pytest.raises(UntrustedArtifactError, match="SHA-256 integrity check failed"):
        load_trusted_pipeline_artifact(art_0, allowed_root=tmp_path)


# ---------------------------------------------------------------------------
# 9. Official holdout protection
# ---------------------------------------------------------------------------


def test_development_commands_cannot_implicitly_access_official_holdout(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """9. Verify validationData.csv is blocked in train.py and internal_holdout mode, and requires confirmation."""
    train_csv, manifest_csv, _, _ = _write_synthetic_workspace(tmp_path, synthetic_uji_df)
    val_csv = tmp_path / "validationData.csv"
    synthetic_uji_df.to_csv(val_csv, index=False)

    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"

    # Passing validationData.csv to train_gradient_boosting must raise PermissionError
    with pytest.raises(PermissionError, match="validationData.csv"):
        train_gradient_boosting(
            train_csv_path=val_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            profile="smoke",
        )

    fast_starting = {**STARTING_GB_PARAMS, "n_estimators": 8}
    with patch.dict("models.gradient_boosting.train.STARTING_GB_PARAMS", fast_starting):
        train_summary = train_gradient_boosting(
            train_csv_path=train_csv,
            split_manifest_path=manifest_csv,
            artifacts_base_dir=artifacts_dir,
            results_base_dir=results_dir,
            profile="smoke",
            n_trials=1,
            n_estimators=[8],
            learning_rates=[0.1],
            max_depths=[3],
            min_samples_leaf=[1],
            run_id="gb_guard_run",
        )

    # Default internal_holdout mode must reject --val-csv
    with pytest.raises(PermissionError, match="Development mode"):
        evaluate_gradient_boosting(
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
        evaluate_gradient_boosting(
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
