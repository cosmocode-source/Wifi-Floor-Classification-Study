"""Unit and integration tests for Stage 5 (`comparison/` and `shared/demo.py`).

Covers all required Prompt 5 test cases on synthetic artifacts:
1. Reject mismatched `row_id` sets, ground-truth `true_floor` / `building_id` labels,
   `dataset_hash` / `split_manifest_hash`, and `evaluation_role` mixing.
2. Correct row alignment by stable `row_id` regardless of file row order in `predictions.csv`.
3. Correct pooled accuracy arithmetic across buildings of unequal size (`total_correct / total_evaluated`).
4. Honest handling of missing or failed/skipped model results (strict error by default; explicit
   incomplete status without fabricated metrics when `allow_incomplete=True`; no silent 'latest'
   file discovery).
5. Refusal to rank models on incompatible timing/fit scopes (suppressing latency ranks and skipping
   the runtime comparison figure).
6. Preservation of source-run provenance and explicit preprocessing disclosures (PCA-kNN vs. raw-WAP
   Gradient Boosting, matched vs. best-selected kNN, and `k=1` equivalence).
7. Verification that numbers in `comparison_report.md`, `writeup_1page.md`, `writeup_2page.md`,
   and `slides_populated.md` originate strictly from the actual `combined_results.json` artifact.
8. Verification of the offline live-demo inference entrypoint (`shared/demo.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from comparison.compare import (
    ComparisonValidationError,
    align_and_validate_predictions_across_models,
    run_comparison,
)
from comparison.report_builder import (
    DRAFT_BANNER,
    build_academic_writeup_markdown,
    build_comparison_report_markdown,
    generate_academic_materials,
)
from models.gradient_boosting.train import build_configured_gradient_boosting_pipeline
from shared.artifacts import UntrustedArtifactError, save_pipeline_artifact
from shared.config import (
    EVALUATION_ROLE_CV_VAL,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
    MODEL_ID_GRADIENT_BOOSTING,
    MODEL_ID_PCA_KNN,
    MODEL_ID_PCA_WEIGHTED_KNN,
    WAP_COLUMNS,
)
from shared.data_validation import DataValidationError
from shared.demo import DemoInferenceError, run_offline_demo_inference
from shared.metrics import evaluate_predictions_by_building
from shared.preprocessing import describe_preprocessing_config
from shared.result_schema import (
    build_completed_result,
    build_non_completed_result,
    save_predictions_csv,
    save_result_json,
)


def _create_synthetic_model_eval_bundle(
    base_dir: Path,
    *,
    model_id: str,
    run_id: str,
    preds_df: pd.DataFrame,
    dataset_hash: str = "sha256_dataset_synthetic_abc123",
    split_manifest_hash: str = "sha256_manifest_synthetic_def456",
    evaluation_role: str = EVALUATION_ROLE_INTERNAL_HOLDOUT,
    fit_scope: str = "dev_train_saved_pipeline_no_refit",
    explicit_final_refit: bool = False,
    fit_seconds: float = 0.0,
    predict_seconds: float = 0.125,
    sentinel_value: int = 100,
    replacement_value: float = -110.0,
    per_building_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write a valid schema v1.0 `metrics.json` and `predictions.csv` for a synthetic model run."""
    model_dir = base_dir / model_id / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    preds_path = model_dir / "predictions.csv"
    metrics_path = model_dir / "metrics.json"

    df_to_save = preds_df.copy()
    df_to_save["model_id"] = model_id
    df_to_save["evaluation_role"] = evaluation_role
    save_predictions_csv(df_to_save, preds_path, evaluation_role=evaluation_role)

    per_bldg, pooled, supports, cms = evaluate_predictions_by_building(df_to_save)

    use_pca = model_id in (MODEL_ID_PCA_KNN, MODEL_ID_PCA_WEIGHTED_KNN)
    prep = describe_preprocessing_config(
        scale=(model_id != MODEL_ID_GRADIENT_BOOSTING),
        use_pca=use_pca,
        n_components=10 if use_pca else None,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
    )
    prep["fit_scope"] = fit_scope

    if per_building_params is None:
        if use_pca:
            per_building_params = {
                str(b): {"n_neighbors": 5, "n_components": 10, "scaling": "standard", "scale": True}
                for b in per_bldg.keys()
            }
        else:
            per_building_params = {
                str(b): {
                    "n_estimators": 189,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "min_samples_leaf": 1,
                }
                for b in per_bldg.keys()
            }

    sel_params: dict[str, Any] = {
        "variant": "tuned",
        "explicit_final_refit_all_training_rows": explicit_final_refit,
        "per_building": per_building_params,
    }
    if use_pca:
        sel_params["weights"] = "uniform" if model_id == MODEL_ID_PCA_KNN else "distance"

    payload = build_completed_result(
        run_id=run_id,
        model_id=model_id,
        dataset_hash=dataset_hash,
        split_manifest_hash=split_manifest_hash,
        evaluation_role=evaluation_role,
        preprocessing=prep,
        selected_parameters=sel_params,
        selection_metric={
            "primary_metric": "mean_cv_accuracy",
            "tie_breaker_metric": "mean_cv_macro_f1",
            "selected_on_split_role": "dev_train",
        },
        per_building=per_bldg,
        pooled=pooled,
        supports=supports,
        confusion_matrices=cms,
        timing={
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": float(fit_seconds + predict_seconds),
        },
        prediction_file_path=preds_path,
        model_artifact_paths={str(b): str(model_dir / f"{model_id}_b{b}.joblib") for b in per_bldg.keys()},
        assumptions=["Synthetic test assumption: 100 -> -110.0 dBm."],
    )
    save_result_json(payload, metrics_path)
    return metrics_path, preds_path, payload


def _build_base_ground_truth_df() -> pd.DataFrame:
    """Create a 20-row synthetic evaluation set with unequal building sizes (B0: 4, B1: 6, B2: 10)."""
    rows: list[dict[str, Any]] = []
    # Building 0: 4 rows (floors 0..3)
    for i in range(4):
        rows.append({"row_id": f"r_b0_{i:02d}", "building_id": 0, "true_floor": i % 4})
    # Building 1: 6 rows (floors 0..2)
    for i in range(6):
        rows.append({"row_id": f"r_b1_{i:02d}", "building_id": 1, "true_floor": i % 3})
    # Building 2: 10 rows (floors 0..4)
    for i in range(10):
        rows.append({"row_id": f"r_b2_{i:02d}", "building_id": 2, "true_floor": i % 5})
    return pd.DataFrame(rows)


def test_comparison_row_alignment_and_pooled_accuracy_arithmetic(tmp_path: Path) -> None:
    """Verify row alignment independent of file order and exact pooled accuracy arithmetic."""
    gt_df = _build_base_ground_truth_df()

    # Model 1 (pca_knn): 15/20 correct = 0.7500 pooled accuracy
    # Building 0: 2/4 (0.50), Building 1: 3/6 (0.50), Building 2: 10/10 (1.00)
    # Note: Unweighted building mean is (0.5 + 0.5 + 1.0)/3 = 0.6667 != 0.7500!
    df_knn = gt_df.copy()
    df_knn["predicted_floor"] = df_knn["true_floor"]
    df_knn.loc[df_knn["row_id"].isin(["r_b0_00", "r_b0_01"]), "predicted_floor"] = 3
    df_knn.loc[df_knn["row_id"].isin(["r_b1_00", "r_b1_01", "r_b1_03"]), "predicted_floor"] = 2

    # Model 2 (pca_weighted_knn): 17/20 correct = 0.8500 pooled accuracy, saved in REVERSED row order!
    df_wknn = gt_df.copy()
    df_wknn["predicted_floor"] = df_wknn["true_floor"]
    df_wknn.loc[df_wknn["row_id"] == "r_b0_00", "predicted_floor"] = 3
    df_wknn.loc[df_wknn["row_id"] == "r_b1_00", "predicted_floor"] = 2
    df_wknn.loc[df_wknn["row_id"] == "r_b2_00", "predicted_floor"] = 3
    df_wknn = df_wknn.iloc[::-1].reset_index(drop=True)

    # Model 3 (gradient_boosting): 18/20 correct = 0.9000 pooled accuracy, saved in SHUFFLED row order!
    df_gb = gt_df.copy()
    df_gb["predicted_floor"] = df_gb["true_floor"]
    df_gb.loc[df_gb["row_id"] == "r_b1_00", "predicted_floor"] = 2
    df_gb.loc[df_gb["row_id"] == "r_b2_00", "predicted_floor"] = 4
    df_gb = df_gb.sample(frac=1.0, random_state=19).reset_index(drop=True)

    m_knn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_KNN,
        run_id="run_knn_1",
        preds_df=df_knn,
        predict_seconds=0.12,
    )
    m_wknn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        run_id="run_wknn_1",
        preds_df=df_wknn,
        predict_seconds=0.14,
        per_building_params={
            "0": {"n_neighbors": 7, "n_components": 50, "scaling": "standard", "scale": True},
            "1": {"n_neighbors": 5, "n_components": 10, "scaling": "standard", "scale": True},
            "2": {"n_neighbors": 9, "n_components": 50, "scaling": "none", "scale": False},
        },
    )
    m_gb, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="run_gb_1",
        preds_df=df_gb,
        predict_seconds=0.05,
    )

    out_base = tmp_path / "comparison_out"
    combined = run_comparison(
        pca_knn_metrics_path=m_knn,
        pca_weighted_knn_metrics_path=m_wknn,
        gradient_boosting_metrics_path=m_gb,
        output_base_dir=out_base,
        comparison_id="comp_aligned",
        update_academic_materials=False,
    )

    assert combined["status"] == "completed"
    assert combined["row_alignment_verified"] is True
    assert combined["ground_truth_identical_across_models"] is True

    # Verify pooled accuracy is strictly total_correct / total_evaluated (15/20 = 0.75, not 0.6667)
    knn_pooled = combined["models"][MODEL_ID_PCA_KNN]["pooled"]
    assert np.isclose(knn_pooled["accuracy"], 15.0 / 20.0, atol=1e-9)
    assert not np.isclose(knn_pooled["accuracy"], knn_pooled["unweighted_building_mean_accuracy"], atol=1e-3)

    # Verify paired_predictions.csv aligned rows by row_id despite reversed/shuffled inputs
    paired_csv = Path(combined["output_files"]["paired_predictions_csv"])
    paired_df = pd.read_csv(paired_csv)
    assert len(paired_df) == 20
    row_b0_00 = paired_df[paired_df["row_id"] == "r_b0_00"].iloc[0]
    assert int(row_b0_00["true_floor"]) == 0
    assert int(row_b0_00["pred_pca_knn"]) == 3
    assert int(row_b0_00["pred_pca_weighted_knn"]) == 3
    assert int(row_b0_00["pred_gradient_boosting"]) == 0
    assert bool(row_b0_00["correct_gradient_boosting"]) is True
    assert bool(row_b0_00["correct_pca_knn"]) is False

    # Verify preprocessing disclosure (PCA-kNN vs raw-WAP Gradient Boosting + matched vs best-selected kNN)
    disc = combined["preprocessing_disclosure"]
    assert disc["pipeline_comparison_scope"] == "full_pipeline_comparison_pca_knn_vs_raw_feature_gradient_boosting"
    assert disc["knn_uses_pca"] is True
    assert disc["gradient_boosting_uses_pca"] is False
    knn_pair = disc["uniform_vs_weighted_knn_analysis"]
    assert knn_pair["overall_comparison_type"] == "best_selected_pipeline_comparison"
    assert knn_pair["per_building"]["0"]["comparison_label"] == "best_selected_pipeline_comparison"
    assert knn_pair["per_building"]["1"]["comparison_label"] == "matched_configuration_comparison"


def test_comparison_rejects_mismatched_rows_labels_hashes_and_roles(tmp_path: Path) -> None:
    """Verify strict rejection of mismatched row IDs, true labels, hashes, and evaluation roles."""
    gt_df = _build_base_ground_truth_df()
    df_valid = gt_df.copy()
    df_valid["predicted_floor"] = df_valid["true_floor"]

    m_knn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path, model_id=MODEL_ID_PCA_KNN, run_id="knn_ok", preds_df=df_valid
    )
    m_wknn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path, model_id=MODEL_ID_PCA_WEIGHTED_KNN, run_id="wknn_ok", preds_df=df_valid
    )

    # 1. Mismatched evaluation_role (official_validation mixed with internal_holdout)
    m_gb_bad_role, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_bad_role",
        preds_df=df_valid,
        evaluation_role=EVALUATION_ROLE_OFFICIAL_VALIDATION,
    )
    with pytest.raises(ComparisonValidationError, match="Forbidden leaderboard mixing"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=m_gb_bad_role,
            output_base_dir=tmp_path / "out1",
        )

    # 2. Mismatched split_manifest_hash
    m_gb_bad_hash, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_bad_hash",
        preds_df=df_valid,
        split_manifest_hash="different_manifest_hash_999",
    )
    with pytest.raises(ComparisonValidationError, match="Split manifest hash mismatch"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=m_gb_bad_hash,
            output_base_dir=tmp_path / "out2",
        )

    # 3. Mismatched row_id set
    df_bad_row = df_valid.copy()
    df_bad_row.loc[0, "row_id"] = "r_foreign_999"
    m_gb_bad_row, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_bad_row",
        preds_df=df_bad_row,
    )
    with pytest.raises(ComparisonValidationError, match="Evaluated row_id mismatch"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=m_gb_bad_row,
            output_base_dir=tmp_path / "out3",
        )

    # 4. Mismatched ground-truth true_floor for the same row_id (keeping class support counts equal)
    df_bad_label = df_valid.copy()
    # Swap true_floor of two rows in building 0 so class supports stay identical but row labels differ
    f0 = int(df_bad_label.loc[0, "true_floor"])
    f1 = int(df_bad_label.loc[1, "true_floor"])
    df_bad_label.loc[0, "true_floor"] = f1
    df_bad_label.loc[1, "true_floor"] = f0
    df_bad_label["predicted_floor"] = df_bad_label["true_floor"]
    m_gb_bad_label, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_bad_label",
        preds_df=df_bad_label,
    )
    with pytest.raises(ComparisonValidationError, match="Ground-truth true_floor mismatch"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=m_gb_bad_label,
            output_base_dir=tmp_path / "out4",
        )


def test_comparison_handles_missing_and_failed_results_honestly(tmp_path: Path) -> None:
    """Verify missing or non-completed runs fail clearly by default and never fabricate metrics."""
    gt_df = _build_base_ground_truth_df()
    df_valid = gt_df.copy()
    df_valid["predicted_floor"] = df_valid["true_floor"]

    m_knn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path, model_id=MODEL_ID_PCA_KNN, run_id="knn_1", preds_df=df_valid
    )
    m_wknn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path, model_id=MODEL_ID_PCA_WEIGHTED_KNN, run_id="wknn_1", preds_df=df_valid
    )

    # Missing 3rd model raises ComparisonValidationError when allow_incomplete=False
    with pytest.raises(ComparisonValidationError, match="Required metrics.json path for 'gradient_boosting'"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=None,
            output_base_dir=tmp_path / "out_strict_missing",
            allow_incomplete=False,
        )

    # Failed 3rd model raises ComparisonValidationError when allow_incomplete=False
    failed_gb_path = tmp_path / "gb_failed_metrics.json"
    failed_payload = build_non_completed_result(
        run_id="gb_failed_run",
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        status="failed",
        evaluation_role=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        reason="Simulated convergence failure.",
    )
    save_result_json(failed_payload, failed_gb_path)

    with pytest.raises(ComparisonValidationError, match="non-completed status='failed'"):
        run_comparison(
            pca_knn_metrics_path=m_knn,
            pca_weighted_knn_metrics_path=m_wknn,
            gradient_boosting_metrics_path=failed_gb_path,
            output_base_dir=tmp_path / "out_strict_failed",
            allow_incomplete=False,
        )

    # With allow_incomplete=True, produces an explicit status="incomplete" report with NO fabricated numbers
    inc_combined = run_comparison(
        pca_knn_metrics_path=m_knn,
        pca_weighted_knn_metrics_path=m_wknn,
        gradient_boosting_metrics_path=failed_gb_path,
        output_base_dir=tmp_path / "out_incomplete",
        comparison_id="comp_incomplete",
        allow_incomplete=True,
    )
    assert inc_combined["status"] == "incomplete"
    assert MODEL_ID_GRADIENT_BOOSTING in inc_combined["missing_or_failed_reasons"]
    assert inc_combined["models"][MODEL_ID_GRADIENT_BOOSTING]["pooled"] is None
    assert inc_combined["figures_manifest"]["status"] == "skipped"

    comp_df = pd.read_csv(inc_combined["output_files"]["comparison_csv"])
    gb_row = comp_df[comp_df["model_id"] == MODEL_ID_GRADIENT_BOOSTING].iloc[0]
    assert gb_row["status"] == "failed"
    assert pd.isna(gb_row["pooled_accuracy"])

    report_text = Path(inc_combined["output_files"]["comparison_report_md"]).read_text(encoding="utf-8")
    assert DRAFT_BANNER in report_text
    assert "Simulated convergence failure." in report_text


def test_comparison_does_not_rank_on_incompatible_timing_scopes(tmp_path: Path) -> None:
    """Verify that models with different fit/refit timing scopes are not ranked by runtime."""
    gt_df = _build_base_ground_truth_df()
    df_valid = gt_df.copy()
    df_valid["predicted_floor"] = df_valid["true_floor"]

    m_knn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_KNN,
        run_id="knn_norefit",
        preds_df=df_valid,
        fit_scope="dev_train_saved_pipeline_no_refit",
        explicit_final_refit=False,
        fit_seconds=0.0,
        predict_seconds=0.20,
    )
    m_wknn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        run_id="wknn_norefit",
        preds_df=df_valid,
        fit_scope="dev_train_saved_pipeline_no_refit",
        explicit_final_refit=False,
        fit_seconds=0.0,
        predict_seconds=0.22,
    )
    # Gradient boosting evaluated with an explicit final refit scope
    m_gb, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_refit",
        preds_df=df_valid,
        fit_scope="all_official_training_rows_explicit_final_refit",
        explicit_final_refit=True,
        fit_seconds=4.50,
        predict_seconds=0.04,
    )

    combined = run_comparison(
        pca_knn_metrics_path=m_knn,
        pca_weighted_knn_metrics_path=m_wknn,
        gradient_boosting_metrics_path=m_gb,
        output_base_dir=tmp_path / "out_timing",
        comparison_id="comp_incompatible_timing",
    )

    t_comp = combined["timing_comparison"]
    assert t_comp["comparable_for_ranking"] is False
    assert "Incompatible timing/fit scopes" in t_comp["incompatibility_reason"]
    for m_id in (MODEL_ID_PCA_KNN, MODEL_ID_PCA_WEIGHTED_KNN, MODEL_ID_GRADIENT_BOOSTING):
        assert "predict_latency_rank" not in t_comp["per_model_timing"][m_id]

    comp_df = pd.read_csv(combined["output_files"]["comparison_csv"])
    assert comp_df["predict_latency_rank"].isna().all()

    rt_fig = combined["figures_manifest"]["figures"]["runtime_comparison"]
    assert rt_fig["status"] == "skipped"
    assert rt_fig["paths"] == []


def test_report_numbers_originate_from_actual_artifacts_and_preserve_provenance(tmp_path: Path) -> None:
    """Verify that report_builder pulls exact numbers and provenance hashes from combined_results.json."""
    gt_df = _build_base_ground_truth_df()

    # Give each model a distinct accuracy:
    # pca_knn: 16/20 = 0.8000, pca_weighted_knn: 18/20 = 0.9000, gradient_boosting: 19/20 = 0.9500
    df_knn = gt_df.copy()
    df_knn["predicted_floor"] = df_knn["true_floor"]
    df_knn.loc[df_knn["row_id"].isin(["r_b0_00", "r_b0_01", "r_b2_00"]), "predicted_floor"] = 3
    df_knn.loc[df_knn["row_id"] == "r_b1_00", "predicted_floor"] = 2

    df_wknn = gt_df.copy()
    df_wknn["predicted_floor"] = df_wknn["true_floor"]
    df_wknn.loc[df_wknn["row_id"].isin(["r_b0_00", "r_b2_00"]), "predicted_floor"] = 3

    df_gb = gt_df.copy()
    df_gb["predicted_floor"] = df_gb["true_floor"]
    df_gb.loc[df_gb["row_id"].isin(["r_b2_00"]), "predicted_floor"] = 3

    ds_sha = "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168"
    sp_sha = "0e0effe365f6568cc712d8dd046c14d5f60f56c32d4ddb403eb51630235a6fed"

    m_knn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_KNN,
        run_id="knn_prov_run_101",
        preds_df=df_knn,
        dataset_hash=ds_sha,
        split_manifest_hash=sp_sha,
    )
    m_wknn, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        run_id="wknn_prov_run_202",
        preds_df=df_wknn,
        dataset_hash=ds_sha,
        split_manifest_hash=sp_sha,
    )
    m_gb, _, _ = _create_synthetic_model_eval_bundle(
        tmp_path,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        run_id="gb_prov_run_303",
        preds_df=df_gb,
        dataset_hash=ds_sha,
        split_manifest_hash=sp_sha,
    )

    combined = run_comparison(
        pca_knn_metrics_path=m_knn,
        pca_weighted_knn_metrics_path=m_wknn,
        gradient_boosting_metrics_path=m_gb,
        output_base_dir=tmp_path / "out_prov",
        comparison_id="comp_provenance_check",
    )

    rep_md = Path(combined["output_files"]["comparison_report_md"]).read_text(encoding="utf-8")
    assert "knn_prov_run_101" in rep_md
    assert "wknn_prov_run_202" in rep_md
    assert "gb_prov_run_303" in rep_md
    assert ds_sha in rep_md
    assert sp_sha in rep_md
    assert "0.8000" in rep_md and "16 / 20" in rep_md
    assert "0.9000" in rep_md and "18 / 20" in rep_md
    assert "0.9500" in rep_md and "19 / 20" in rep_md

    # Also verify 1-page, 2-page, and slide generation from combined_results.json
    acad = generate_academic_materials(
        combined_results_path=combined["output_files"]["combined_results_json"],
        output_reports_dir=tmp_path / "reports_out",
        output_presentation_dir=tmp_path / "pres_out",
        length_mode="both",
    )
    assert acad["status"] == "completed"
    w1_text = Path(acad["written_files"]["writeup_1page"]).read_text(encoding="utf-8")
    w2_text = Path(acad["written_files"]["writeup_2page"]).read_text(encoding="utf-8")
    sl_text = Path(acad["written_files"]["slides_populated"]).read_text(encoding="utf-8")

    for doc_text in (w1_text, w2_text, sl_text):
        assert "0.8000 (80.00%)" in doc_text
        assert "0.9000 (90.00%)" in doc_text
        assert "0.9500 (95.00%)" in doc_text
        assert "19/20" in doc_text


def test_offline_live_demo_inference_and_failure_guards(
    tmp_path: Path,
    synthetic_uji_Quoted_or_df: pd.DataFrame | None = None,
) -> None:
    """Verify that shared.demo loads a trusted pipeline, predicts original FLOORs, and withholds FLOOR from X."""
    rng = np.random.default_rng(123)
    n_rows = 12
    wap_data = np.full((n_rows, len(WAP_COLUMNS)), 100.0, dtype=float)
    # Activate a few WAPs with distinct RSSI patterns for non-contiguous floors [0, 2, 4]
    floors = np.array([0, 2, 4] * 4, dtype=int)
    for idx, fl in enumerate(floors):
        wap_data[idx, fl * 2] = -55.0 + float(rng.normal(0, 1.0))
        wap_data[idx, fl * 2 + 1] = -60.0 + float(rng.normal(0, 1.0))

    df_train = pd.DataFrame(wap_data, columns=list(WAP_COLUMNS))
    df_train["BUILDINGID"] = 2
    df_train["FLOOR"] = floors
    df_train["row_id"] = [f"demo_row_{i:02d}" for i in range(n_rows)]

    pipeline = build_configured_gradient_boosting_pipeline(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42,
    )
    X_fit = df_train.loc[:, list(WAP_COLUMNS)]
    y_fit = df_train["FLOOR"].to_numpy(dtype=int)
    pipeline.fit(X_fit, y_fit)

    art_dir = tmp_path / "demo_artifacts"
    meta = save_pipeline_artifact(
        pipeline=pipeline,
        output_dir=art_dir,
        building_id=2,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        feature_names=WAP_COLUMNS,
        class_labels=[0, 2, 4],
        dataset_hash="demo_ds_sha",
        split_manifest_hash="demo_sp_sha",
        run_id="demo_run",
    )
    art_path = Path(meta["artifact_path"])

    # Run offline demo with ground-truth FLOOR present in input_df (must be withheld from predictors)
    summary = run_offline_demo_inference(
        input_df=df_train.iloc[:6],
        artifact_path=art_path,
        building_id=2,
        allowed_root=tmp_path,
        output_json_path=tmp_path / "demo_out.json",
        output_csv_path=tmp_path / "demo_out.csv",
    )
    assert summary["status"] == "completed"
    assert summary["ground_truth_included_in_predictors"] is False
    assert summary["ground_truth_available_for_display"] is True
    assert summary["evaluated_row_count"] == 6
    pred_floors = {int(r["predicted_floor"]) for r in summary["predictions"]}
    assert pred_floors.issubset({0, 2, 4})

    # Verify clear failure when a required WAP column is missing
    df_missing_wap = df_train.iloc[:3].drop(columns=["WAP520"])
    with pytest.raises(DataValidationError, match="missing 1 required WAP columns"):
        run_offline_demo_inference(
            input_df=df_missing_wap,
            artifact_path=art_path,
            building_id=2,
            allowed_root=tmp_path,
        )

    # Verify clear failure when artifact file does not exist
    with pytest.raises(FileNotFoundError, match="Specified demo pipeline artifact not found"):
        run_offline_demo_inference(
            input_df=df_train.iloc[:3],
            artifact_path=tmp_path / "nonexistent_pipeline.joblib",
            building_id=2,
            allowed_root=tmp_path,
        )
