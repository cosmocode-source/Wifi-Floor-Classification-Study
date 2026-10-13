'Multi-model comparison builder, cross-artifact validator, and canonical result generator.'

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import yaml

from comparison.plots import generate_all_comparison_figures
from comparison.report_builder import (
    build_comparison_report_markdown,
    generate_academic_materials,
)
from models.pca_knn.train import build_matched_knn_comparison_table
from shared.config import (
    COMPARISON_DIR,
    COMPARISON_RESULTS_DIR,
    EVALUATION_ROLE_CV_VAL,
    MODEL_ID_GRADIENT_BOOSTING,
    MODEL_ID_PCA_KNN,
    MODEL_ID_PCA_WEIGHTED_KNN,
    POOLED_MACRO_F1_LABEL_POLICY,
    RESULT_SCHEMA_VERSION,
    VALID_EVALUATION_ROLES,
    VALID_MODEL_IDS,
)
from shared.metrics import evaluate_predictions_by_building
from shared.result_schema import (
    ResultSchemaError,
    collect_environment_info,
    load_predictions_csv,
    load_result_json,
)


DEFAULT_COMPARISON_CONFIG_YAML: Path = COMPARISON_DIR / "config.yaml"


class ComparisonValidationError(ValueError):
    """Raised when comparison inputs are missing, incompatible, or violate fairness contracts."""


def load_comparison_config(
    config_path: Path | str = DEFAULT_COMPARISON_CONFIG_YAML,
) -> dict[str, Any]:
    """Load `comparison/config.yaml`."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Comparison configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)!r}.")
    return data


def _resolve_predictions_path(
    metrics_payload: Mapping[str, Any],
    metrics_json_path: Path,
    explicit_predictions_path: Path | str | None = None,
) -> Path:
    """Resolve and verify the `predictions.csv` path associated with a model's `metrics.json`."""
    if explicit_predictions_path is not None:
        p = Path(explicit_predictions_path)
        if not p.is_file():
            raise ComparisonValidationError(
                f"Explicit predictions CSV for model '{metrics_payload.get('model_id')}' not found: {p}"
            )
        return p.resolve()

    recorded = metrics_payload.get("prediction_file_path")
    if not recorded:
        raise ComparisonValidationError(
            f"Metrics file '{metrics_json_path}' does not record a 'prediction_file_path'."
        )
    cand = Path(str(recorded))
    if cand.is_file():
        return cand.resolve()

    # Check relative to metrics_json_path directory
    rel_cand = metrics_json_path.parent / cand.name
    if rel_cand.is_file():
        return rel_cand.resolve()

    extra_cand = metrics_payload.get("extra_metadata", {}).get("canonical_variant_predictions_csv")
    if extra_cand and Path(str(extra_cand)).is_file():
        return Path(str(extra_cand)).resolve()

    raise ComparisonValidationError(
        f"Predictions CSV for '{metrics_json_path}' not found at '{cand}' or '{rel_cand}'."
    )


def _validate_and_recompute_single_model(
    expected_model_id: str,
    metrics_payload: Mapping[str, Any],
    preds_df: pd.DataFrame,
    tolerance: float = 1e-7,
) -> None:
    """Verify internal consistency between a model's `metrics.json` and its `predictions.csv`."""
    actual_model_id = str(metrics_payload.get("model_id", ""))
    if actual_model_id != expected_model_id:
        raise ComparisonValidationError(
            f"Slot for '{expected_model_id}' received metrics with model_id={actual_model_id!r}."
        )

    eval_role = str(metrics_payload["evaluation_role"])
    if set(preds_df["model_id"].astype(str).unique()) != {expected_model_id}:
        raise ComparisonValidationError(
            f"Predictions CSV for '{expected_model_id}' contains mismatched model_id values: "
            f"{sorted(preds_df['model_id'].astype(str).unique())}."
        )
    if set(preds_df["evaluation_role"].astype(str).unique()) != {eval_role}:
        raise ComparisonValidationError(
            f"Predictions CSV evaluation_role {sorted(preds_df['evaluation_role'].astype(str).unique())} "
            f"does not match metrics.json evaluation_role '{eval_role}'."
        )

    recomp_pb, recomp_pooled, _, _ = evaluate_predictions_by_building(preds_df)
    rec_pooled = metrics_payload["pooled"]

    if int(recomp_pooled["evaluated_row_count"]) != int(rec_pooled["evaluated_row_count"]):
        raise ComparisonValidationError(
            f"Model '{expected_model_id}' pooled evaluated_row_count ({rec_pooled['evaluated_row_count']}) "
            f"does not match predictions.csv row count ({recomp_pooled['evaluated_row_count']})."
        )
    if int(recomp_pooled["incorrect_count"]) != int(rec_pooled["incorrect_count"]):
        raise ComparisonValidationError(
            f"Model '{expected_model_id}' pooled incorrect_count ({rec_pooled['incorrect_count']}) "
            f"does not match recomputed predictions.csv incorrect_count ({recomp_pooled['incorrect_count']})."
        )

    for metric_key in ("accuracy", "error", "macro_f1", "balanced_accuracy"):
        if not np.isclose(float(recomp_pooled[metric_key]), float(rec_pooled[metric_key]), atol=tolerance, rtol=0):
            raise ComparisonValidationError(
                f"Model '{expected_model_id}' pooled.{metric_key} ({rec_pooled[metric_key]}) "
                f"differs from recomputed predictions value ({recomp_pooled[metric_key]})."
            )

    rec_pb = metrics_payload["per_building"]
    if set(recomp_pb.keys()) != set(rec_pb.keys()):
        raise ComparisonValidationError(
            f"Model '{expected_model_id}' per_building keys {sorted(rec_pb.keys())} "
            f"do not match predictions.csv buildings {sorted(recomp_pb.keys())}."
        )
    for b_key, b_recomp in recomp_pb.items():
        b_rec = rec_pb[b_key]
        if int(b_recomp["evaluated_row_count"]) != int(b_rec["evaluated_row_count"]):
            raise ComparisonValidationError(
                f"Model '{expected_model_id}' building {b_key} evaluated_row_count mismatch."
            )
        for metric_key in ("accuracy", "error", "macro_f1", "balanced_accuracy"):
            if not np.isclose(float(b_recomp[metric_key]), float(b_rec[metric_key]), atol=tolerance, rtol=0):
                raise ComparisonValidationError(
                    f"Model '{expected_model_id}' building {b_key} {metric_key} mismatch: "
                    f"{b_rec[metric_key]} vs {b_recomp[metric_key]}."
                )


def align_and_validate_predictions_across_models(
    predictions_by_model: Mapping[str, pd.DataFrame],
    expected_evaluation_role: str,
) -> pd.DataFrame:
    'Align predictions across models strictly by stable `row_id` (not file row order).'
    if not predictions_by_model:
        raise ComparisonValidationError("No model prediction DataFrames supplied for alignment.")

    reference_model_id: str | None = None
    reference_indexed: pd.DataFrame | None = None
    aligned_model_Series: dict[str, pd.DataFrame] = {}

    for model_id, df in predictions_by_model.items():
        if df["row_id"].duplicated().any():
            dup_cnt = int(df["row_id"].duplicated().sum())
            raise ComparisonValidationError(
                f"Predictions for '{model_id}' contain {dup_cnt} duplicate row_id keys."
            )
        work = df.copy()
        work["row_id"] = work["row_id"].astype(str)
        work["building_id"] = work["building_id"].astype(int)
        work["true_floor"] = work["true_floor"].astype(int)
        work["predicted_floor"] = work["predicted_floor"].astype(int)

        # Sort by building_id then row_id for deterministic canonical ordering independent of file order
        work_sorted = work.sort_values(by=["building_id", "row_id"])
        work_indexed = work_sorted.set_index("row_id", drop=False)
        work_indexed.index.name = "_row_id_idx"

        if reference_indexed is None:
            reference_model_id = model_id
            reference_indexed = work_indexed
        else:
            ref_ids = set(reference_indexed.index)
            cur_ids = set(work_indexed.index)
            if ref_ids != cur_ids:
                missing = sorted(ref_ids - cur_ids)[:5]
                extra = sorted(cur_ids - ref_ids)[:5]
                raise ComparisonValidationError(
                    f"Evaluated row_id mismatch between '{reference_model_id}' and '{model_id}': "
                    f"missing={missing}, extra={extra}, symmetric_diff_count={len(ref_ids ^ cur_ids)}."
                )
            # Align strictly by row_id index of reference_indexed
            work_aligned = work_indexed.loc[reference_indexed.index]
            if not (work_aligned["building_id"].to_numpy() == reference_indexed["building_id"].to_numpy()).all():
                bad_rows = reference_indexed.index[
                    work_aligned["building_id"].to_numpy() != reference_indexed["building_id"].to_numpy()
                ].tolist()[:5]
                raise ComparisonValidationError(
                    f"building_id mismatch after row_id alignment between '{reference_model_id}' "
                    f"and '{model_id}' at row_ids {bad_rows}."
                )
            if not (work_aligned["true_floor"].to_numpy() == reference_indexed["true_floor"].to_numpy()).all():
                bad_rows = reference_indexed.index[
                    work_aligned["true_floor"].to_numpy() != reference_indexed["true_floor"].to_numpy()
                ].tolist()[:5]
                raise ComparisonValidationError(
                    f"Ground-truth true_floor mismatch after row_id alignment between "
                    f"'{reference_model_id}' and '{model_id}' at row_ids {bad_rows}."
                )
        aligned_model_Series[model_id] = work_indexed.loc[reference_indexed.index]

    assert reference_indexed is not None
    paired = pd.DataFrame(
        {
            "row_id": reference_indexed["row_id"].to_numpy(dtype=object),
            "building_id": reference_indexed["building_id"].to_numpy(dtype=int),
            "true_floor": reference_indexed["true_floor"].to_numpy(dtype=int),
            "evaluation_role": np.full(len(reference_indexed), str(expected_evaluation_role), dtype=object),
        }
    )

    for model_id in VALID_MODEL_IDS:
        if model_id in aligned_model_Series:
            m_df = aligned_model_Series[model_id]
            preds = m_df["predicted_floor"].to_numpy(dtype=int)
            trues = paired["true_floor"].to_numpy(dtype=int)
            paired[f"pred_{model_id}"] = preds
            paired[f"correct_{model_id}"] = (preds == trues)

    present_pred_cols = [f"pred_{m}" for m in VALID_MODEL_IDS if f"pred_{m}" in paired.columns]
    if len(present_pred_cols) == 3:
        paired["all_three_models_agree"] = (
            (paired["pred_pca_knn"] == paired["pred_pca_weighted_knn"])
            & (paired["pred_pca_weighted_knn"] == paired["pred_gradient_boosting"])
        )
        paired["knn_uniform_and_weighted_agree"] = (
            paired["pred_pca_knn"] == paired["pred_pca_weighted_knn"]
        )

    return paired.reset_index(drop=True)


def _build_preprocessing_and_fairness_disclosure(
    models_data: Mapping[str, Mapping[str, Any]],
    uniform_search_df: pd.DataFrame | None = None,
    weighted_search_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Verify kNN encoding parity and construct explicit preprocessing and fairness disclosures."""
    u_data = models_data.get(MODEL_ID_PCA_KNN)
    w_data = models_data.get(MODEL_ID_PCA_WEIGHTED_KNN)
    gb_data = models_data.get(MODEL_ID_GRADIENT_BOOSTING)

    knn_encoding_matched = True
    if (
        u_data
        and w_data
        and u_data.get("status") == "completed"
        and w_data.get("status") == "completed"
    ):
        u_prep = u_data.get("preprocessing", {})
        w_prep = w_data.get("preprocessing", {})
        if (
            u_prep.get("sentinel_value") != w_prep.get("sentinel_value")
            or float(u_prep.get("replacement_value", -110.0)) != float(w_prep.get("replacement_value", -110.0))
        ):
            raise ComparisonValidationError(
                f"Incompatible signal encoding between pca_knn ({u_prep}) and pca_weighted_knn ({w_prep})."
            )
        u_enc_sha = u_prep.get("encoding_config_sha256")
        w_enc_sha = w_prep.get("encoding_config_sha256")
        if u_enc_sha and w_enc_sha and u_enc_sha != w_enc_sha:
            raise ComparisonValidationError(
                f"encoding_config_sha256 mismatch between pca_knn ({u_enc_sha}) and pca_weighted_knn ({w_enc_sha})."
            )
        if not bool(u_prep.get("use_pca", True)) or not bool(w_prep.get("use_pca", True)):
            raise ComparisonValidationError(
                "Both pca_knn and pca_weighted_knn must use PCA feature representation (use_pca=True)."
            )

    # Analyze per-building selected parameters between uniform and weighted kNN
    knn_analysis: dict[str, Any] = {"per_building": {}}
    if (
        u_data
        and w_data
        and u_data.get("status") == "completed"
        and w_data.get("status") == "completed"
    ):
        u_pb_params = u_data.get("selected_parameters", {}).get("per_building", {})
        w_pb_params = w_data.get("selected_parameters", {}).get("per_building", {})
        u_pb_metrics = u_data.get("per_building", {})
        w_pb_metrics = w_data.get("per_building", {})

        all_matched = True
        for b_key in sorted(set(u_pb_metrics.keys()) & set(w_pb_metrics.keys()), key=int):
            up = u_pb_params.get(b_key, {})
            wp = w_pb_params.get(b_key, {})
            same_setting = (
                up.get("n_neighbors") == wp.get("n_neighbors")
                and up.get("n_components") == wp.get("n_components")
                and up.get("scaling") == wp.get("scaling")
            )
            if not same_setting:
                all_matched = False
            both_k1 = (up.get("n_neighbors") == 1 and wp.get("n_neighbors") == 1)
            comp_label = (
                "matched_k1_equivalence_check"
                if both_k1 and same_setting
                else (
                    "matched_configuration_comparison"
                    if same_setting
                    else "best_selected_pipeline_comparison"
                )
            )
            knn_analysis["per_building"][b_key] = {
                "building_id": int(b_key),
                "uniform_params": up,
                "weighted_params": wp,
                "matched_k_pca_scaling": bool(same_setting),
                "both_k_equals_1": bool(both_k1),
                "comparison_label": comp_label,
                "accuracy_delta_weighted_minus_uniform": float(
                    w_pb_metrics[b_key]["accuracy"] - u_pb_metrics[b_key]["accuracy"]
                ),
                "macro_f1_delta_weighted_minus_uniform": float(
                    w_pb_metrics[b_key]["macro_f1"] - u_pb_metrics[b_key]["macro_f1"]
                ),
            }
        knn_analysis["overall_comparison_type"] = (
            "matched_configuration_comparison"
            if all_matched
            else "best_selected_pipeline_comparison"
        )

    if uniform_search_df is not None and weighted_search_df is not None:
        matched_table = build_matched_knn_comparison_table(uniform_search_df, weighted_search_df)
        k_gt1 = matched_table[
            (matched_table["status_uniform"] == "completed")
            & (matched_table["status_weighted"] == "completed")
            & (matched_table["n_neighbors"].astype(int) > 1)
        ]
        k_eq1 = matched_table[
            (matched_table["status_uniform"] == "completed")
            & (matched_table["status_weighted"] == "completed")
            & (matched_table["n_neighbors"].astype(int) == 1)
        ]
        knn_analysis["matched_cv_grid_summary"] = {
            "total_matched_pairs": int(len(matched_table)),
            "completed_k_gt_1_pairs": int(len(k_gt1)),
            "mean_cv_accuracy_delta_k_gt_1": (
                float(k_gt1["delta_mean_cv_accuracy_weighted_minus_uniform"].mean())
                if not k_gt1.empty
                else None
            ),
            "max_abs_cv_accuracy_delta_at_k1": (
                float(k_eq1["delta_mean_cv_accuracy_weighted_minus_uniform"].abs().max())
                if not k_eq1.empty
                else 0.0
            ),
        }

    gb_uses_pca = (
        bool(gb_data.get("preprocessing", {}).get("use_pca", False))
        if gb_data and gb_data.get("status") == "completed"
        else False
    )
    gb_uses_scale = (
        bool(gb_data.get("preprocessing", {}).get("scale", False))
        if gb_data and gb_data.get("status") == "completed"
        else False
    )

    return {
        "pipeline_comparison_scope": (
            "full_pipeline_comparison_pca_knn_vs_raw_feature_gradient_boosting"
            if not gb_uses_pca
            else "matched_pca_feature_comparison"
        ),
        "knn_encoding_matched": knn_encoding_matched,
        "knn_uses_pca": True,
        "gradient_boosting_uses_pca": gb_uses_pca,
        "gradient_boosting_uses_scaling": gb_uses_scale,
        "gradient_boosting_feature_representation_note": (
            "Classical GradientBoostingClassifier operates on finite sentinel-replaced (100 -> -110.0 dBm) "
            "raw WAP001..WAP520 features (use_pca=False, scale=False), while pca_knn and pca_weighted_knn "
            "operate on fold-local PCA projections. This difference is explicitly disclosed as a full-pipeline "
            "comparison rather than an isolated estimator-only comparison."
        ),
        "k1_equivalence_note": (
            "At k=1, inverse-distance weighting assigns 100% weight to the single nearest neighbor and is "
            "mathematically equivalent to uniform 1NN. Any k=1 comparison is reported strictly as a parity check."
        ),
        "uniform_vs_weighted_knn_analysis": knn_analysis,
    }


def _evaluate_timing_compatibility(
    models_data: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify whether model timing metrics share a comparable scope and batch size before ranking."""
    completed = {
        m_id: data
        for m_id, data in models_data.items()
        if data and data.get("status") == "completed"
    }
    if len(completed) < 2:
        return {
            "comparable_for_ranking": False,
            "incompatibility_reason": "Fewer than 2 completed models available for runtime comparison.",
            "note": "Runtime ranking omitted because fewer than 2 models completed.",
            "per_model_timing": {},
        }

    fit_scopes: set[str] = set()
    row_counts: set[int] = set()
    per_model_timing: dict[str, Any] = {}
    missing_timing = False

    for m_id, data in completed.items():
        t_block = data.get("timing", {})
        p_sec = t_block.get("predict_seconds")
        f_sec = t_block.get("fit_seconds")
        if p_sec is None or not np.isfinite(float(p_sec)) or float(p_sec) < 0.0:
            missing_timing = True
        scope = str(data.get("preprocessing", {}).get("fit_scope", "unspecified_scope"))
        refit_flag = bool(
            data.get("selected_parameters", {}).get("explicit_final_refit_all_training_rows", False)
        )
        effective_scope = f"{scope}|refit={refit_flag}"
        fit_scopes.add(effective_scope)
        n_rows = int(data["pooled"]["evaluated_row_count"])
        row_counts.add(n_rows)

        per_model_timing[m_id] = {
            "fit_scope": scope,
            "explicit_final_refit_all_training_rows": refit_flag,
            "evaluated_row_count": n_rows,
            "fit_seconds": float(f_sec) if f_sec is not None else None,
            "predict_seconds": float(p_sec) if p_sec is not None else None,
            "total_seconds": (
                float(t_block["total_seconds"]) if t_block.get("total_seconds") is not None else None
            ),
            "predict_ms_per_row": (
                float(p_sec) * 1000.0 / max(1, n_rows) if p_sec is not None else None
            ),
        }

    if missing_timing:
        return {
            "comparable_for_ranking": False,
            "incompatibility_reason": "One or more models have null or invalid predict_seconds.",
            "note": "Runtime ranking disabled because timing measurements are incomplete.",
            "per_model_timing": per_model_timing,
        }

    if len(row_counts) > 1:
        return {
            "comparable_for_ranking": False,
            "incompatibility_reason": f"Models evaluated different batch sizes: {sorted(row_counts)}.",
            "note": "Runtime ranking disabled due to mismatched evaluation batch sizes.",
            "per_model_timing": per_model_timing,
        }

    if len(fit_scopes) > 1:
        return {
            "comparable_for_ranking": False,
            "incompatibility_reason": (
                f"Incompatible timing/fit scopes across models: {sorted(fit_scopes)}. "
                "Refusing to rank models with different fit/refit scopes."
            ),
            "note": (
                "Models were executed under different timing/refit scopes "
                f"({sorted(fit_scopes)}); runtime ranking is suppressed to prevent unfair comparison."
            ),
            "per_model_timing": per_model_timing,
        }

    # Comparable! Compute descriptive predict_seconds ordering without claiming universal hardware speed
    ordered_by_pred = sorted(
        per_model_timing.keys(),
        key=lambda m: float(per_model_timing[m]["predict_seconds"]),
    )
    for rank_idx, m_id in enumerate(ordered_by_pred, start=1):
        per_model_timing[m_id]["predict_latency_rank"] = rank_idx

    shared_scope = next(iter(fit_scopes)).split("|")[0]
    return {
        "comparable_for_ranking": True,
        "shared_fit_scope": shared_scope,
        "shared_batch_size": next(iter(row_counts)),
        "fastest_inference_model_id": ordered_by_pred[0],
        "note": (
            f"All compared models share fit_scope='{shared_scope}' and batch_size={next(iter(row_counts))} "
            "under sequential CPU execution (n_jobs=1)."
        ),
        "per_model_timing": per_model_timing,
    }


def _build_summary_tables(
    combined_results: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive `comparison.csv` and `per_building_comparison.csv` strictly from `combined_results.json`."""
    models_map: Mapping[str, Any] = combined_results.get("models", {})
    timing_map: Mapping[str, Any] = combined_results.get("timing_comparison", {}).get("per_model_timing", {})
    comparable_timing = bool(combined_results.get("timing_comparison", {}).get("comparable_for_ranking", False))

    pooled_rows: list[dict[str, Any]] = []
    building_rows: list[dict[str, Any]] = []

    for m_id in VALID_MODEL_IDS:
        m_entry = models_map.get(m_id)
        if not m_entry or m_entry.get("status") != "completed":
            pooled_rows.append(
                {
                    "comparison_id": combined_results.get("comparison_id"),
                    "model_id": m_id,
                    "variant": "tuned",
                    "run_id": m_entry.get("run_id") if m_entry else None,
                    "status": m_entry.get("status", "missing") if m_entry else "missing",
                    "evaluation_role": combined_results.get("evaluation_role"),
                    "dataset_hash": combined_results.get("dataset_hash"),
                    "split_manifest_hash": combined_results.get("split_manifest_hash"),
                    "pooled_accuracy": np.nan,
                    "pooled_error": np.nan,
                    "pooled_macro_f1": np.nan,
                    "pooled_balanced_accuracy": np.nan,
                    "correct_count": np.nan,
                    "incorrect_count": np.nan,
                    "evaluated_row_count": np.nan,
                    "use_pca": None,
                    "scale": None,
                    "fit_scope": None,
                    "fit_seconds": np.nan,
                    "predict_seconds": np.nan,
                    "predict_ms_per_row": np.nan,
                    "predict_latency_rank": np.nan,
                }
            )
            continue

        pooled = m_entry["pooled"]
        prep = m_entry.get("preprocessing", {})
        t_info = timing_map.get(m_id, {})
        eval_cnt = int(pooled["evaluated_row_count"])
        inc_cnt = int(pooled["incorrect_count"])
        corr_cnt = eval_cnt - inc_cnt

        pooled_rows.append(
            {
                "comparison_id": combined_results.get("comparison_id"),
                "model_id": m_id,
                "variant": m_entry.get("selected_parameters", {}).get("variant", "tuned"),
                "run_id": m_entry["run_id"],
                "status": "completed",
                "evaluation_role": m_entry["evaluation_role"],
                "dataset_hash": m_entry["dataset_hash"],
                "split_manifest_hash": m_entry["split_manifest_hash"],
                "pooled_accuracy": float(pooled["accuracy"]),
                "pooled_error": float(pooled["error"]),
                "pooled_macro_f1": float(pooled["macro_f1"]),
                "pooled_balanced_accuracy": float(pooled["balanced_accuracy"]),
                "unweighted_building_mean_accuracy": pooled.get("unweighted_building_mean_accuracy", np.nan),
                "support_weighted_building_macro_f1": pooled.get("support_weighted_building_macro_f1", np.nan),
                "correct_count": corr_cnt,
                "incorrect_count": inc_cnt,
                "evaluated_row_count": eval_cnt,
                "use_pca": bool(prep.get("use_pca", m_id != MODEL_ID_GRADIENT_BOOSTING)),
                "scale": bool(prep.get("scale", False)),
                "sentinel_replacement": float(prep.get("replacement_value", -110.0)),
                "fit_scope": prep.get("fit_scope"),
                "fit_seconds": t_info.get("fit_seconds", np.nan),
                "predict_seconds": t_info.get("predict_seconds", np.nan),
                "predict_ms_per_row": t_info.get("predict_ms_per_row", np.nan),
                "predict_latency_rank": (
                    t_info.get("predict_latency_rank", np.nan) if comparable_timing else np.nan
                ),
                "source_metrics_json_path": m_entry.get("source_metrics_json_path"),
            }
        )

        pb_params = m_entry.get("selected_parameters", {}).get("per_building", {})
        for b_key, b_metrics in sorted(m_entry.get("per_building", {}).items(), key=lambda kv: int(kv[0])):
            b_eval = int(b_metrics["evaluated_row_count"])
            b_inc = int(b_metrics["incorrect_count"])
            b_corr = b_eval - b_inc
            b_p = pb_params.get(b_key, {})
            building_rows.append(
                {
                    "comparison_id": combined_results.get("comparison_id"),
                    "building_id": int(b_key),
                    "model_id": m_id,
                    "variant": m_entry.get("selected_parameters", {}).get("variant", "tuned"),
                    "run_id": m_entry["run_id"],
                    "evaluation_role": m_entry["evaluation_role"],
                    "accuracy": float(b_metrics["accuracy"]),
                    "error": float(b_metrics["error"]),
                    "macro_f1": float(b_metrics["macro_f1"]),
                    "balanced_accuracy": float(b_metrics["balanced_accuracy"]),
                    "correct_count": b_corr,
                    "incorrect_count": b_inc,
                    "evaluated_row_count": b_eval,
                    "n_neighbors": b_p.get("n_neighbors", np.nan),
                    "n_components": b_p.get("n_components", np.nan),
                    "scaling": b_p.get("scaling", None),
                    "weights": m_entry.get("selected_parameters", {}).get("weights", None),
                    "n_estimators": b_p.get("n_estimators", np.nan),
                    "learning_rate": b_p.get("learning_rate", np.nan),
                    "max_depth": b_p.get("max_depth", np.nan),
                    "min_samples_leaf": b_p.get("min_samples_leaf", np.nan),
                }
            )

    return pd.DataFrame(pooled_rows), pd.DataFrame(building_rows)


def run_comparison(
    *,
    pca_knn_metrics_path: Path | str | None,
    pca_weighted_knn_metrics_path: Path | str | None,
    gradient_boosting_metrics_path: Path | str | None,
    pca_knn_predictions_path: Path | str | None = None,
    pca_weighted_knn_predictions_path: Path | str | None = None,
    gradient_boosting_predictions_path: Path | str | None = None,
    pca_knn_baseline_metrics_path: Path | str | None = None,
    pca_weighted_knn_baseline_metrics_path: Path | str | None = None,
    gradient_boosting_baseline_metrics_path: Path | str | None = None,
    pca_knn_cv_metrics_path: Path | str | None = None,
    pca_weighted_knn_cv_metrics_path: Path | str | None = None,
    gradient_boosting_cv_metrics_path: Path | str | None = None,
    pca_knn_search_csv_path: Path | str | None = None,
    pca_weighted_knn_search_csv_path: Path | str | None = None,
    gradient_boosting_search_csv_path: Path | str | None = None,
    pca_knn_validation_curve_csv_path: Path | str | None = None,
    pca_weighted_knn_validation_curve_csv_path: Path | str | None = None,
    output_base_dir: Path | str = COMPARISON_RESULTS_DIR,
    comparison_id: str | None = None,
    allow_incomplete: bool = False,
    update_academic_materials: bool = False,
    config_path: Path | str = DEFAULT_COMPARISON_CONFIG_YAML,
) -> dict[str, Any]:
    """Validate three model evaluation artifacts and build the canonical comparison directory."""
    cfg = load_comparison_config(config_path)
    tolerance = float(cfg.get("comparison", {}).get("metric_tolerance", 1e-7))

    input_specs: dict[str, tuple[Path | str | None, Path | str | None]] = {
        MODEL_ID_PCA_KNN: (pca_knn_metrics_path, pca_knn_predictions_path),
        MODEL_ID_PCA_WEIGHTED_KNN: (pca_weighted_knn_metrics_path, pca_weighted_knn_predictions_path),
        MODEL_ID_GRADIENT_BOOSTING: (gradient_boosting_metrics_path, gradient_boosting_predictions_path),
    }

    loaded_models: dict[str, dict[str, Any]] = {}
    loaded_predictions: dict[str, pd.DataFrame] = {}
    missing_or_failed_reasons: dict[str, str] = {}
    warnings_list: list[str] = []

    for model_id, (m_path_raw, p_path_raw) in input_specs.items():
        if m_path_raw is None:
            reason = (
                f"Required metrics.json path for '{model_id}' was not provided. "
                "Comparison builder refuses to guess 'latest' files automatically."
            )
            if not allow_incomplete:
                raise ComparisonValidationError(reason)
            missing_or_failed_reasons[model_id] = reason
            loaded_models[model_id] = {
                "model_id": model_id,
                "status": "missing",
                "status_reason": reason,
            }
            continue

        m_path = Path(m_path_raw)
        if not m_path.is_file():
            reason = f"Metrics JSON file for '{model_id}' does not exist at '{m_path}'."
            if not allow_incomplete:
                raise ComparisonValidationError(reason)
            missing_or_failed_reasons[model_id] = reason
            loaded_models[model_id] = {
                "model_id": model_id,
                "status": "missing",
                "status_reason": reason,
                "source_metrics_json_path": str(m_path),
            }
            continue

        try:
            payload = load_result_json(m_path, validate=True)
        except ResultSchemaError as exc:
            raise ComparisonValidationError(
                f"Result schema validation failed for '{model_id}' at '{m_path}': {exc}"
            ) from exc

        if payload.get("model_id") != model_id:
            raise ComparisonValidationError(
                f"Expected model_id='{model_id}' in '{m_path}', got {payload.get('model_id')!r}."
            )

        status = str(payload.get("status", ""))
        if status != "completed":
            reason = (
                f"Model '{model_id}' at '{m_path}' has non-completed status={status!r} "
                f"(reason: {payload.get('status_reason', 'unspecified')})."
            )
            if not allow_incomplete:
                raise ComparisonValidationError(reason)
            missing_or_failed_reasons[model_id] = reason
            loaded_models[model_id] = {
                **payload,
                "source_metrics_json_path": str(m_path.resolve()),
            }
            continue

        pred_path = _resolve_predictions_path(payload, m_path, explicit_predictions_path=p_path_raw)
        try:
            preds_df = load_predictions_csv(
                pred_path,
                evaluation_role=str(payload["evaluation_role"]),
                validate=True,
            )
        except ResultSchemaError as exc:
            raise ComparisonValidationError(
                f"Predictions CSV schema validation failed for '{model_id}' at '{pred_path}': {exc}"
            ) from exc

        _validate_and_recompute_single_model(
            expected_model_id=model_id,
            metrics_payload=payload,
            preds_df=preds_df,
            tolerance=tolerance,
        )

        payload_copy = dict(payload)
        payload_copy["source_metrics_json_path"] = str(m_path.resolve())
        payload_copy["prediction_file_path"] = str(pred_path.resolve())
        loaded_models[model_id] = payload_copy
        loaded_predictions[model_id] = preds_df

    completed_ids = [
        m_id for m_id in VALID_MODEL_IDS
        if m_id in loaded_models and loaded_models[m_id].get("status") == "completed"
    ]

    if not completed_ids and not allow_incomplete:
        raise ComparisonValidationError("No completed model evaluation runs available to compare.")

    # Cross-model provenance, evaluation_role, and target consistency checks across all completed models
    eval_roles = {str(loaded_models[m]["evaluation_role"]) for m in completed_ids}
    if len(eval_roles) > 1:
        raise ComparisonValidationError(
            f"Forbidden leaderboard mixing: cannot compare models evaluated on different evaluation_role "
            f"splits in one comparison: { {m: loaded_models[m]['evaluation_role'] for m in completed_ids} }."
        )
    shared_eval_role = next(iter(eval_roles)) if eval_roles else "incomplete"

    dataset_hashes = {str(loaded_models[m]["dataset_hash"]) for m in completed_ids}
    if len(dataset_hashes) > 1:
        raise ComparisonValidationError(
            f"Dataset hash mismatch across models: { {m: loaded_models[m]['dataset_hash'] for m in completed_ids} }."
        )
    shared_dataset_hash = next(iter(dataset_hashes)) if dataset_hashes else ""

    manifest_hashes = {str(loaded_models[m]["split_manifest_hash"]) for m in completed_ids}
    if len(manifest_hashes) > 1:
        raise ComparisonValidationError(
            f"Split manifest hash mismatch across models: "
            f"{ {m: loaded_models[m]['split_manifest_hash'] for m in completed_ids} }."
        )
    shared_manifest_hash = next(iter(manifest_hashes)) if manifest_hashes else ""

    # Verify per-building sample counts and ground-truth class supports match across all completed models
    if len(completed_ids) >= 2:
        def _normalize_gt_supports(sup_dict: Mapping[str, Any]) -> dict[str, Any]:
            norm_bldg: dict[str, Any] = {}
            for b_k, b_v in sup_dict.get("per_building", {}).items():
                norm_bldg[str(b_k)] = {
                    "evaluated_row_count": int(b_v.get("evaluated_row_count", 0)),
                    "class_supports": {
                        str(ck): int(cv)
                        for ck, cv in b_v.get("class_supports", {}).items()
                        if int(cv) > 0
                    },
                }
            pooled_v = sup_dict.get("pooled", {})
            return {
                "per_building": norm_bldg,
                "pooled": {
                    "evaluated_row_count": int(pooled_v.get("evaluated_row_count", 0)),
                    "class_supports": {
                        str(ck): int(cv)
                        for ck, cv in pooled_v.get("class_supports", {}).items()
                        if int(cv) > 0
                    },
                },
            }

        ref_m = completed_ids[0]
        ref_supports = _normalize_gt_supports(loaded_models[ref_m]["supports"])
        for other_m in completed_ids[1:]:
            other_supports = _normalize_gt_supports(loaded_models[other_m]["supports"])
            if other_supports != ref_supports:
                raise ComparisonValidationError(
                    f"Per-building or pooled class support mismatch between '{ref_m}' and '{other_m}': "
                    f"{ref_supports} vs {other_supports}."
                )

    paired_df: pd.DataFrame | None = None
    if loaded_predictions:
        paired_df = align_and_validate_predictions_across_models(
            loaded_predictions,
            expected_evaluation_role=shared_eval_role,
        )

    # Load optional search / validation curve CSVs if provided
    u_search_df = pd.read_csv(pca_knn_search_csv_path) if pca_knn_search_csv_path else None
    w_search_df = pd.read_csv(pca_weighted_knn_search_csv_path) if pca_weighted_knn_search_csv_path else None
    u_vc_df = (
        pd.read_csv(pca_knn_validation_curve_csv_path)
        if pca_knn_validation_curve_csv_path
        else u_search_df
    )
    w_vc_df = (
        pd.read_csv(pca_weighted_knn_validation_curve_csv_path)
        if pca_weighted_knn_validation_curve_csv_path
        else w_search_df
    )

    prep_disclosure = _build_preprocessing_and_fairness_disclosure(
        loaded_models,
        uniform_search_df=u_search_df,
        weighted_search_df=w_search_df,
    )
    timing_comparison = _evaluate_timing_compatibility(loaded_models)
    if not timing_comparison.get("comparable_for_ranking", False) and len(completed_ids) >= 2:
        warnings_list.append(
            str(timing_comparison.get("incompatibility_reason", "Timing scopes are not comparable."))
        )

    # Load optional supplementary fixed/starting baselines (checking they match dataset/manifest/role)
    supplementary_baselines: dict[str, Any] = {}
    baseline_paths = {
        MODEL_ID_PCA_KNN: pca_knn_baseline_metrics_path,
        MODEL_ID_PCA_WEIGHTED_KNN: pca_weighted_knn_baseline_metrics_path,
        MODEL_ID_GRADIENT_BOOSTING: gradient_boosting_baseline_metrics_path,
    }
    for m_id, b_path_raw in baseline_paths.items():
        if b_path_raw is None:
            continue
        b_path = Path(b_path_raw)
        if not b_path.is_file():
            raise ComparisonValidationError(f"Specified baseline metrics JSON not found: {b_path}")
        b_payload = load_result_json(b_path, validate=True)
        if b_payload.get("model_id") != m_id:
            raise ComparisonValidationError(
                f"Baseline metrics for '{m_id}' has mismatched model_id={b_payload.get('model_id')!r}."
            )
        if shared_eval_role != "incomplete" and b_payload.get("evaluation_role") != shared_eval_role:
            raise ComparisonValidationError(
                f"Baseline metrics for '{m_id}' has evaluation_role={b_payload.get('evaluation_role')!r}, "
                f"expected '{shared_eval_role}'."
            )
        if shared_dataset_hash and b_payload.get("dataset_hash") != shared_dataset_hash:
            raise ComparisonValidationError(
                f"Baseline metrics for '{m_id}' has mismatched dataset_hash."
            )
        supplementary_baselines[m_id] = b_payload

    # Load optional supplementary training-side CV summaries (strictly kept in a separate cv_val section!)
    supplementary_cv: dict[str, Any] = {}
    cv_paths = {
        MODEL_ID_PCA_KNN: pca_knn_cv_metrics_path,
        MODEL_ID_PCA_WEIGHTED_KNN: pca_weighted_knn_cv_metrics_path,
        MODEL_ID_GRADIENT_BOOSTING: gradient_boosting_cv_metrics_path,
    }
    for m_id, cv_path_raw in cv_paths.items():
        if cv_path_raw is None:
            continue
        cv_path = Path(cv_path_raw)
        if not cv_path.is_file():
            raise ComparisonValidationError(f"Specified CV metrics JSON not found: {cv_path}")
        cv_payload = load_result_json(cv_path, validate=True)
        if cv_payload.get("model_id") != m_id:
            raise ComparisonValidationError(
                f"CV metrics for '{m_id}' has mismatched model_id={cv_payload.get('model_id')!r}."
            )
        if cv_payload.get("evaluation_role") != EVALUATION_ROLE_CV_VAL:
            raise ComparisonValidationError(
                f"Supplementary CV metrics for '{m_id}' must have evaluation_role='cv_val', "
                f"got {cv_payload.get('evaluation_role')!r}."
            )
        if shared_manifest_hash and cv_payload.get("split_manifest_hash") != shared_manifest_hash:
            raise ComparisonValidationError(
                f"Supplementary CV metrics for '{m_id}' has mismatched split_manifest_hash."
            )
        supplementary_cv[m_id] = cv_payload

    overall_status = "completed" if len(completed_ids) == 3 else "incomplete"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_uid = uuid.uuid4().hex[:8]
    resolved_comp_id = comparison_id or f"{ts}_comparison_{shared_eval_role}_{short_uid}"

    out_dir = Path(output_base_dir) / resolved_comp_id
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = out_dir / "figures"

    # Descriptive ranking on the evaluated split (with strict non-retuning discipline note)
    descriptive_ordering: list[dict[str, Any]] = []
    for m_id in completed_ids:
        p = loaded_models[m_id]["pooled"]
        descriptive_ordering.append(
            {
                "model_id": m_id,
                "run_id": loaded_models[m_id]["run_id"],
                "pooled_accuracy": float(p["accuracy"]),
                "pooled_macro_f1": float(p["macro_f1"]),
                "pooled_balanced_accuracy": float(p["balanced_accuracy"]),
                "correct_count": int(p["evaluated_row_count"]) - int(p["incorrect_count"]),
                "incorrect_count": int(p["incorrect_count"]),
                "evaluated_row_count": int(p["evaluated_row_count"]),
            }
        )
    descriptive_ordering.sort(
        key=lambda x: (x["pooled_accuracy"], x["pooled_macro_f1"]),
        reverse=True,
    )

    combined_results: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "comparison_id": resolved_comp_id,
        "status": overall_status,
        "evaluation_role": shared_eval_role,
        "dataset_hash": shared_dataset_hash,
        "split_manifest_hash": shared_manifest_hash,
        "target_definition": {
            "target_column": "FLOOR",
            "building_selector_column": "BUILDINGID",
            "predictor_columns": "WAP001..WAP520 (520 columns)",
            "known_buildings": sorted(
                int(b) for b in loaded_models[completed_ids[0]]["per_building"].keys()
            ) if completed_ids else [0, 1, 2],
        },
        "evaluated_row_count": (
            int(loaded_models[completed_ids[0]]["pooled"]["evaluated_row_count"])
            if completed_ids
            else None
        ),
        "pooled_macro_f1_policy": POOLED_MACRO_F1_LABEL_POLICY,
        "row_alignment_verified": bool(paired_df is not None),
        "ground_truth_identical_across_models": bool(paired_df is not None),
        "descriptive_split_ordering": descriptive_ordering,
        "model_selection_discipline": {
            "selection_split_role": "dev_train",
            "primary_selection_metric": "mean_cv_accuracy",
            "tie_breaker_selection_metric": "mean_cv_macro_f1",
            "holdout_retuning_permitted": False,
            "selection_rule_statement": (
                "Choice between uniform and distance-weighted kNN and all model hyperparameters "
                "must be determined using training-side cross-validation (dev_train folds), "
                "never from internal_holdout or official_validation scores."
            ),
            "frozen_holdout_statement": (
                "Holdout splits compare frozen pipelines only. Observing the highest descriptive "
                "score on a holdout split does not permit post-hoc retuning or a universal "
                "statistical superiority claim."
            ),
            "statistical_reporting_statement": (
                "No parametric confidence intervals or hypothesis test p-values are fabricated "
                "from correlated cross-validation folds or single holdout splits."
            ),
        },
        "preprocessing_disclosure": prep_disclosure,
        "timing_comparison": timing_comparison,
        "models": loaded_models,
        "supplementary_baselines": supplementary_baselines,
        "supplementary_training_cv": supplementary_cv,
        "missing_or_failed_reasons": missing_or_failed_reasons,
        "warnings": warnings_list,
        "environment": collect_environment_info(),
    }

    # Generate figures from combined_results and optional validation curve DataFrames
    figures_manifest = generate_all_comparison_figures(
        combined_results=combined_results,
        output_dir=figures_dir,
        uniform_curve_df=u_vc_df,
        weighted_curve_df=w_vc_df,
    )
    combined_results["figures_manifest"] = figures_manifest

    combined_json_path = out_dir / "combined_results.json"
    comparison_csv_path = out_dir / "comparison.csv"
    per_bldg_csv_path = out_dir / "per_building_comparison.csv"
    paired_preds_csv_path = out_dir / "paired_predictions.csv"
    report_md_path = out_dir / "comparison_report.md"

    pooled_table_df, per_bldg_table_df = _build_summary_tables(combined_results)
    pooled_table_df.to_csv(comparison_csv_path, index=False)
    per_bldg_table_df.to_csv(per_bldg_csv_path, index=False)

    if paired_df is not None:
        paired_df.to_csv(paired_preds_csv_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "row_id",
                "building_id",
                "true_floor",
                "evaluation_role",
                "pred_pca_knn",
                "correct_pca_knn",
                "pred_pca_weighted_knn",
                "correct_pca_weighted_knn",
                "pred_gradient_boosting",
                "correct_gradient_boosting",
            ]
        ).to_csv(paired_preds_csv_path, index=False)

    combined_results["output_files"] = {
        "combined_results_json": str(combined_json_path.resolve()),
        "comparison_csv": str(comparison_csv_path.resolve()),
        "per_building_comparison_csv": str(per_bldg_csv_path.resolve()),
        "paired_predictions_csv": str(paired_preds_csv_path.resolve()),
        "comparison_report_md": str(report_md_path.resolve()),
        "figures_dir": str(figures_dir.resolve()),
    }

    combined_json_path.write_text(
        json.dumps(combined_results, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    report_md = build_comparison_report_markdown(combined_results)
    report_md_path.write_text(report_md, encoding="utf-8")

    if update_academic_materials:
        academic_out = generate_academic_materials(
            combined_results_path=combined_json_path,
            length_mode="both",
        )
        combined_results["academic_materials"] = academic_out
        combined_json_path.write_text(
            json.dumps(combined_results, indent=2, sort_keys=False),
            encoding="utf-8",
        )

    return combined_results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m comparison.compare`."""
    parser = argparse.ArgumentParser(
        prog="python -m comparison.compare",
        description=(
            "Validate and combine evaluation results across pca_knn, pca_weighted_knn, "
            "and gradient_boosting into canonical comparison tables, reports, and figures."
        ),
    )
    parser.add_argument(
        "--pca-knn-metrics",
        type=Path,
        default=None,
        help="Explicit path to pca_knn metrics.json file.",
    )
    parser.add_argument(
        "--pca-weighted-knn-metrics",
        type=Path,
        default=None,
        help="Explicit path to pca_weighted_knn metrics.json file.",
    )
    parser.add_argument(
        "--gradient-boosting-metrics",
        type=Path,
        default=None,
        help="Explicit path to gradient_boosting metrics.json file.",
    )
    parser.add_argument(
        "--pca-knn-predictions",
        type=Path,
        default=None,
        help="Optional explicit path to pca_knn predictions.csv (defaults to prediction_file_path in metrics.json).",
    )
    parser.add_argument(
        "--pca-weighted-knn-predictions",
        type=Path,
        default=None,
        help="Optional explicit path to pca_weighted_knn predictions.csv.",
    )
    parser.add_argument(
        "--gradient-boosting-predictions",
        type=Path,
        default=None,
        help="Optional explicit path to gradient_boosting predictions.csv.",
    )
    parser.add_argument(
        "--pca-knn-baseline-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to pca_knn fixed_baseline metrics.json.",
    )
    parser.add_argument(
        "--pca-weighted-knn-baseline-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to pca_weighted_knn fixed_baseline metrics.json.",
    )
    parser.add_argument(
        "--gradient-boosting-baseline-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to gradient_boosting fixed_baseline (starting) metrics.json.",
    )
    parser.add_argument(
        "--pca-knn-cv-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to pca_knn training-side cv_tuned_metrics.json.",
    )
    parser.add_argument(
        "--pca-weighted-knn-cv-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to pca_weighted_knn training-side cv_tuned_metrics.json.",
    )
    parser.add_argument(
        "--gradient-boosting-cv-metrics",
        type=Path,
        default=None,
        help="Optional explicit path to gradient_boosting training-side cv_tuned_metrics.json.",
    )
    parser.add_argument(
        "--pca-knn-search-csv",
        type=Path,
        default=None,
        help="Optional explicit path to pca_knn search_results.csv.",
    )
    parser.add_argument(
        "--pca-weighted-knn-search-csv",
        type=Path,
        default=None,
        help="Optional explicit path to pca_weighted_knn search_results.csv.",
    )
    parser.add_argument(
        "--gradient-boosting-search-csv",
        type=Path,
        default=None,
        help="Optional explicit path to gradient_boosting search_results.csv.",
    )
    parser.add_argument(
        "--pca-knn-validation-curve-csv",
        type=Path,
        default=None,
        help="Optional explicit path to pca_knn validation_curve.csv.",
    )
    parser.add_argument(
        "--pca-weighted-knn-validation-curve-csv",
        type=Path,
        default=None,
        help="Optional explicit path to pca_weighted_knn validation_curve.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARISON_RESULTS_DIR,
        help="Base output directory for comparison/results/<comparison_id>.",
    )
    parser.add_argument(
        "--comparison-id",
        type=str,
        default=None,
        help="Explicit comparison identifier (creates comparison/results/<comparison_id>).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate an explicitly incomplete report when one or more model runs are missing or failed.",
    )
    parser.add_argument(
        "--update-academic-materials",
        action="store_true",
        help="Also populate reports/writeup_1page.md, reports/writeup_2page.md, and presentation/slides_populated.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_COMPARISON_CONFIG_YAML,
        help="Path to comparison/config.yaml.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m comparison.compare`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    combined = run_comparison(
        pca_knn_metrics_path=args.pca_knn_metrics,
        pca_weighted_knn_metrics_path=args.pca_weighted_knn_metrics,
        gradient_boosting_metrics_path=args.gradient_boosting_metrics,
        pca_knn_predictions_path=args.pca_knn_predictions,
        pca_weighted_knn_predictions_path=args.pca_weighted_knn_predictions,
        gradient_boosting_predictions_path=args.gradient_boosting_predictions,
        pca_knn_baseline_metrics_path=args.pca_knn_baseline_metrics,
        pca_weighted_knn_baseline_metrics_path=args.pca_weighted_knn_baseline_metrics,
        gradient_boosting_baseline_metrics_path=args.gradient_boosting_baseline_metrics,
        pca_knn_cv_metrics_path=args.pca_knn_cv_metrics,
        pca_weighted_knn_cv_metrics_path=args.pca_weighted_knn_cv_metrics,
        gradient_boosting_cv_metrics_path=args.gradient_boosting_cv_metrics,
        pca_knn_search_csv_path=args.pca_knn_search_csv,
        pca_weighted_knn_search_csv_path=args.pca_weighted_knn_search_csv,
        gradient_boosting_search_csv_path=args.gradient_boosting_search_csv,
        pca_knn_validation_curve_csv_path=args.pca_knn_validation_curve_csv,
        pca_weighted_knn_validation_curve_csv_path=args.pca_weighted_knn_validation_curve_csv,
        output_base_dir=args.output_dir,
        comparison_id=args.comparison_id,
        allow_incomplete=bool(args.allow_incomplete),
        update_academic_materials=bool(args.update_academic_materials),
        config_path=args.config,
    )
    print(json.dumps(combined["output_files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
