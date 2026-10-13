'Training, fold-local cross-validation search, and pipeline serialization for PCA-kNN models.'

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models.pca_knn.clean_data import (
    build_configured_pca_knn_pipeline,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_YAML,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    compute_encoding_config_sha256,
    extract_validated_wap_features_and_targets,
    load_pca_knn_config,
)
from shared.artifacts import create_unique_run_dir, save_pipeline_artifact
from shared.config import (
    BUILDING_COLUMN,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    EVALUATION_ROLE_CV_VAL,
    MODEL_ID_PCA_KNN,
    ROW_ID_COLUMN,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    SPLIT_ROLE_OFFICIAL_VALIDATION,
    TARGET_COLUMN,
    UNDETECTED_SENTINEL,
    WAP_COLUMNS,
)
from shared.create_splits import (
    SplitCreationError,
    align_dataset_with_manifest,
    iter_building_cv_folds,
    load_split_manifest,
)
from shared.data_validation import compute_dataframe_sha256, compute_file_sha256
from shared.feasibility import FeasibilityResult, check_candidate_feasibility
from shared.metrics import (
    compute_classification_metrics,
    evaluate_predictions_by_building,
    select_best_candidate,
)
from shared.preprocessing import describe_preprocessing_config
from shared.result_schema import (
    build_completed_result,
    build_non_completed_result,
    collect_environment_info,
    save_predictions_csv,
    save_result_json,
)

FIXED_BASELINE_PARAMS: dict[str, Any] = {
    "n_neighbors": 1,
    "n_components": 50,
    "scaling": "none",
    "scale": False,
    "whiten": False,
    "weights": "uniform",
    "metric": "euclidean",
}

DECLARED_TIE_BREAKER_POLICY: tuple[str, ...] = (
    "1. Higher mean_cv_accuracy (tolerance 1e-9)",
    "2. Higher mean_cv_macro_f1 (tolerance 1e-9)",
    "3. Smaller n_components (simpler PCA subspace)",
    "4. Smaller n_neighbors (simpler kNN neighborhood)",
    "5. scaling == 'none' preferred over 'standard' (fewer learned parameters)",
    "6. Lower candidate_index in canonical grid order",
)

PCA_KNN_ASSUMPTIONS: tuple[str, ...] = (
    "Undetected WAP sentinel 100 is replaced with -110.0 dBm (documented project assumption, not a verified paper setting).",
    "Genuine 0 dBm signals are preserved as 0.0 dBm.",
    "Floor classification is trained and evaluated separately within each known BUILDINGID (0, 1, 2) using only WAP001..WAP520.",
    "StandardScaler (when enabled) and PCA (whiten=False) are fitted strictly inside each training partition.",
    "Uniform neighbor voting (weights='uniform') with Euclidean distance is enforced for all pca_knn candidates.",
)

PCA_WEIGHTED_KNN_ASSUMPTIONS: tuple[str, ...] = (
    "Undetected WAP sentinel 100 is replaced with -110.0 dBm (documented project assumption, not a verified paper setting).",
    "Genuine 0 dBm signals are preserved as 0.0 dBm.",
    "Floor classification is trained and evaluated separately within each known BUILDINGID (0, 1, 2) using only WAP001..WAP520.",
    "StandardScaler (when enabled) and PCA (whiten=False) are fitted strictly inside each training partition.",
    "Inverse-distance neighbor voting (weights='distance') with Euclidean distance is enforced via scikit-learn KNeighborsClassifier.",
    "At k=1, distance weighting offers no multi-neighbor voting advantage over uniform 1NN; substantive weighting effects occur only for k > 1.",
)


def get_model_assumptions(weights: str = "uniform") -> tuple[str, ...]:
    """Return the documented scientific assumptions tuple for uniform or distance-weighted PCA-kNN."""
    return PCA_WEIGHTED_KNN_ASSUMPTIONS if weights == "distance" else PCA_KNN_ASSUMPTIONS


def resolve_parallelism(
    search_workers: int = 1,
    estimator_n_jobs: int = 1,
) -> tuple[int, int, list[str]]:
    """Enforce bounded CPU parallelism without nested parallel workers.

    Raises ValueError if negative or zero worker counts (such as -1 for all cores) are passed.
    Forces `estimator_n_jobs = 1` whenever `search_workers > 1`.
    """
    if (
        not isinstance(search_workers, (int, np.integer))
        or isinstance(search_workers, bool)
        or int(search_workers) < 1
    ):
        raise ValueError(
            f"search_workers must be a positive integer >= 1 (all-core -1 is prohibited); got {search_workers!r}."
        )
    if (
        not isinstance(estimator_n_jobs, (int, np.integer))
        or isinstance(estimator_n_jobs, bool)
        or int(estimator_n_jobs) < 1
    ):
        raise ValueError(
            f"estimator_n_jobs must be a positive integer >= 1 (all-core -1 is prohibited); got {estimator_n_jobs!r}."
        )

    resolved_search = int(search_workers)
    resolved_estimator = int(estimator_n_jobs)
    warnings: list[str] = []

    if resolved_search > 1 and resolved_estimator > 1:
        warnings.append(
            f"Parallelism safeguard: search_workers={resolved_search} > 1; forcing estimator_n_jobs=1 "
            f"(was {resolved_estimator}) to prevent nested parallelism."
        )
        resolved_estimator = 1

    return resolved_search, resolved_estimator, warnings




def build_uniform_pca_knn_pipeline(
    n_components: int,
    n_neighbors: int,
    scaling: str = "none",
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = 1,
) -> Pipeline:
    """Construct a leakage-safe uniform-weight Euclidean PCA-kNN pipeline."""
    return build_configured_pca_knn_pipeline(
        n_components=n_components,
        n_neighbors=n_neighbors,
        scaling=scaling,
        weights="uniform",
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_fixed_baseline_pipeline(
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = 1,
    weights: str = "uniform",
) -> Pipeline:
    """Construct the fixed paper-inspired anchor pipeline: k=1, PCA=50, unscaled centered inputs, whiten=False."""
    pipeline = build_configured_pca_knn_pipeline(
        n_components=int(FIXED_BASELINE_PARAMS["n_components"]),
        n_neighbors=int(FIXED_BASELINE_PARAMS["n_neighbors"]),
        scaling=str(FIXED_BASELINE_PARAMS["scaling"]),
        weights=weights,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    assert "scaler" not in pipeline.named_steps, "Paper-inspired fixed baseline must not include StandardScaler."
    return pipeline


def generate_candidate_grid(
    pca_components: Sequence[int],
    k_values: Sequence[int],
    scaling_modes: Sequence[str],
    weights: str = "uniform",
) -> list[dict[str, Any]]:
    'Generate the candidate hyperparameter grid in deterministic canonical tie-breaking order.'
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}.")
    invalid_scaling = [s for s in scaling_modes if s not in ("none", "standard")]
    if invalid_scaling:
        raise ValueError(f"Unsupported scaling modes {invalid_scaling}; allowed: ['none', 'standard'].")

    ordered_pca = sorted(dict.fromkeys(int(c) for c in pca_components))
    ordered_k = sorted(dict.fromkeys(int(k) for k in k_values))
    scaling_priority = {"none": 0, "standard": 1}
    ordered_scaling = sorted(dict.fromkeys(str(s) for s in scaling_modes), key=lambda s: scaling_priority[s])

    candidates: list[dict[str, Any]] = []
    idx = 0
    for n_comp in ordered_pca:
        for k_val in ordered_k:
            for sc_mode in ordered_scaling:
                is_fixed = (
                    n_comp == FIXED_BASELINE_PARAMS["n_components"]
                    and k_val == FIXED_BASELINE_PARAMS["n_neighbors"]
                    and sc_mode == FIXED_BASELINE_PARAMS["scaling"]
                )
                candidates.append(
                    {
                        "candidate_index": idx,
                        "candidate_id": f"pca{n_comp}_k{k_val}_scale_{sc_mode}",
                        "n_components": int(n_comp),
                        "n_neighbors": int(k_val),
                        "scaling": sc_mode,
                        "scale": sc_mode == "standard",
                        "whiten": False,
                        "weights": str(weights),
                        "metric": "euclidean",
                        "is_fixed_baseline": bool(is_fixed),
                        "k1_no_multi_neighbor_weighting_advantage": bool(int(k_val) == 1),
                    }
                )
                idx += 1
    return candidates


def select_best_pca_knn_candidate(
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    'Select the winning feasible candidate for a building using predeclared deterministic tie-breakers.'
    scaling_rank = {"none": 0, "standard": 1}
    sorted_by_tie_breakers = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            int(c.get("n_components", 999999)),
            int(c.get("n_neighbors", 999999)),
            scaling_rank.get(str(c.get("scaling", "none")), 99),
            int(c.get("candidate_index", 999999)),
        ),
    )
    return select_best_candidate(
        sorted_by_tie_breakers,
        primary_key="mean_cv_accuracy",
        tie_breaker_key="mean_cv_macro_f1",
        tolerance=tolerance,
    )


def _compute_spread_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    """Compute descriptive score spread statistics across CV folds (`mean`, `std`, `min`, `max`, `iqr`, `range`)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return {
            f"mean_{prefix}": np.nan,
            f"std_{prefix}": np.nan,
            f"min_{prefix}": np.nan,
            f"max_{prefix}": np.nan,
            f"iqr_{prefix}": np.nan,
            f"range_{prefix}": np.nan,
        }
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        f"mean_{prefix}": float(np.mean(arr)),
        f"std_{prefix}": float(np.std(arr, ddof=0)),
        f"min_{prefix}": float(np.min(arr)),
        f"max_{prefix}": float(np.max(arr)),
        f"iqr_{prefix}": float(q75 - q25),
        f"range_{prefix}": float(np.max(arr) - np.min(arr)),
    }


def evaluate_single_candidate_cv(
    candidate: Mapping[str, Any],
    building_id: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    row_ids_full: pd.Series,
    folds: Sequence[tuple[int, np.ndarray, np.ndarray]],
    building_dev_train_indices: np.ndarray,
    holdout_indices_set: set[int],
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    seed: int = DEFAULT_RANDOM_SEED,
    estimator_n_jobs: int = 1,
    candidate_role_label: str = "grid_candidate",
    weights: str | None = None,
    model_id: str = MODEL_ID_PCA_KNN,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Evaluate a single PCA-kNN candidate on one building across precomputed stratified CV folds.

    Returns `(search_row_dict, oof_predictions_df_or_none)`.
    Never touches `holdout_indices_set`. Fits scaling and PCA strictly inside each fold's `train_idx`.
    """
    n_comp = int(candidate["n_components"])
    n_neigh = int(candidate["n_neighbors"])
    scaling = str(candidate["scaling"])
    scale_bool = scaling == "standard"
    resolved_weights = str(weights if weights is not None else candidate.get("weights", "uniform"))
    cand_id = str(candidate.get("candidate_id", f"pca{n_comp}_k{n_neigh}_scale_{scaling}"))
    cand_idx = int(candidate.get("candidate_index", 0))
    n_splits = len(folds)

    y_dev_b = y_full.iloc[building_dev_train_indices].to_numpy(dtype=int)
    building_floors = sorted(int(f) for f in np.unique(y_dev_b))
    fold_train_sizes = [int(len(tr_idx)) for _, tr_idx, _ in folds]

    # Check both fold-level feasibility and full dev_train refit feasibility
    feas_cv: FeasibilityResult = check_candidate_feasibility(
        y_train=y_dev_b,
        n_features=int(X_full.shape[1]),
        n_splits=n_splits,
        n_components=n_comp,
        n_neighbors=n_neigh,
        fold_train_sizes=fold_train_sizes,
        strict=False,
    )

    base_row: dict[str, Any] = {
        "model_id": str(model_id),
        "building_id": int(building_id),
        "candidate_index": cand_idx,
        "candidate_id": cand_id,
        "candidate_role": candidate_role_label,
        "is_fixed_baseline": bool(candidate.get("is_fixed_baseline", False)),
        "n_components": n_comp,
        "n_neighbors": n_neigh,
        "scaling": scaling,
        "scale": scale_bool,
        "whiten": False,
        "weights": resolved_weights,
        "metric": "euclidean",
        "k1_no_multi_neighbor_weighting_advantage": bool(n_neigh == 1),
        "cv_folds": int(n_splits),
        "min_fold_train_rows": int(min(fold_train_sizes)) if fold_train_sizes else 0,
        "dev_train_rows": int(len(building_dev_train_indices)),
    }

    if not feas_cv.feasible:
        base_row.update(
            {
                "feasible": False,
                "status": "skipped",
                "reason_code": feas_cv.reason_code,
                "message": feas_cv.message,
                **_compute_spread_stats([], "cv_accuracy"),
                **_compute_spread_stats([], "cv_error"),
                **_compute_spread_stats([], "cv_macro_f1"),
                "mean_cv_balanced_accuracy": np.nan,
                "std_cv_balanced_accuracy": np.nan,
                "oof_accuracy": np.nan,
                "oof_macro_f1": np.nan,
                "total_fit_seconds": 0.0,
                "total_predict_seconds": 0.0,
                "selected_for_building": False,
            }
        )
        for fold_id, _, _ in folds:
            base_row[f"fold_{fold_id}_accuracy"] = np.nan
            base_row[f"fold_{fold_id}_error"] = np.nan
            base_row[f"fold_{fold_id}_macro_f1"] = np.nan
            base_row[f"fold_{fold_id}_balanced_accuracy"] = np.nan
        return base_row, None

    fold_accuracies: list[float] = []
    fold_errors: list[float] = []
    fold_macro_f1s: list[float] = []
    fold_bal_accs: list[float] = []
    oof_frames: list[pd.DataFrame] = []
    total_fit_sec = 0.0
    total_pred_sec = 0.0

    try:
        for fold_id, train_idx, val_idx in folds:
            train_set = set(int(i) for i in train_idx)
            val_set = set(int(i) for i in val_idx)
            if not train_set.isdisjoint(val_set):
                raise SplitCreationError(
                    f"Fold {fold_id} in building {building_id} has overlapping train and val indices."
                )
            if not train_set.isdisjoint(holdout_indices_set) or not val_set.isdisjoint(holdout_indices_set):
                raise SplitCreationError(
                    f"Leakage detected: fold {fold_id} in building {building_id} intersects internal_holdout."
                )

            X_tr = X_full.iloc[train_idx]
            y_tr = y_full.iloc[train_idx].to_numpy(dtype=int)
            X_va = X_full.iloc[val_idx]
            y_va = y_full.iloc[val_idx].to_numpy(dtype=int)
            rids_va = row_ids_full.iloc[val_idx].astype(str).to_numpy()

            pipeline = build_configured_pca_knn_pipeline(
                n_components=n_comp,
                n_neighbors=n_neigh,
                scaling=scaling,
                weights=resolved_weights,
                sentinel_value=sentinel_value,
                replacement_value=replacement_value,
                enforce_wap_schema=True,
                random_state=int(seed),
                n_jobs=estimator_n_jobs,
            )

            t0 = time.perf_counter()
            pipeline.fit(X_tr, y_tr)
            fit_elapsed = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = pipeline.predict(X_va)
            pred_elapsed = time.perf_counter() - t1

            total_fit_sec += fit_elapsed
            total_pred_sec += pred_elapsed

            f_metrics = compute_classification_metrics(y_va, y_pred, labels=building_floors)
            fold_accuracies.append(float(f_metrics["accuracy"]))
            fold_errors.append(float(f_metrics["error"]))
            fold_macro_f1s.append(float(f_metrics["macro_f1"]))
            fold_bal_accs.append(float(f_metrics["balanced_accuracy"]))

            base_row[f"fold_{fold_id}_accuracy"] = float(f_metrics["accuracy"])
            base_row[f"fold_{fold_id}_error"] = float(f_metrics["error"])
            base_row[f"fold_{fold_id}_macro_f1"] = float(f_metrics["macro_f1"])
            base_row[f"fold_{fold_id}_balanced_accuracy"] = float(f_metrics["balanced_accuracy"])

            oof_frames.append(
                pd.DataFrame(
                    {
                        "row_id": rids_va,
                        "building_id": np.full(len(val_idx), int(building_id), dtype=int),
                        "true_floor": y_va.astype(int),
                        "predicted_floor": np.asarray(y_pred, dtype=int),
                        "evaluation_role": np.full(len(val_idx), EVALUATION_ROLE_CV_VAL, dtype=object),
                        "model_id": np.full(len(val_idx), str(model_id), dtype=object),
                        "fold_id": np.full(len(val_idx), int(fold_id), dtype=int),
                        "repeat_id": np.zeros(len(val_idx), dtype=int),
                    }
                )
            )
    except Exception as exc:
        base_row.update(
            {
                "feasible": True,
                "status": "failed",
                "reason_code": "FIT_PREDICT_EXCEPTION",
                "message": f"{type(exc).__name__}: {exc}",
                **_compute_spread_stats([], "cv_accuracy"),
                **_compute_spread_stats([], "cv_error"),
                **_compute_spread_stats([], "cv_macro_f1"),
                "mean_cv_balanced_accuracy": np.nan,
                "std_cv_balanced_accuracy": np.nan,
                "oof_accuracy": np.nan,
                "oof_macro_f1": np.nan,
                "total_fit_seconds": float(total_fit_sec),
                "total_predict_seconds": float(total_pred_sec),
                "selected_for_building": False,
            }
        )
        return base_row, None

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    oof_metrics = compute_classification_metrics(
        oof_df["true_floor"].to_numpy(dtype=int),
        oof_df["predicted_floor"].to_numpy(dtype=int),
        labels=building_floors,
    )

    base_row.update(
        {
            "feasible": True,
            "status": "completed",
            "reason_code": "OK",
            "message": "Evaluated across all CV folds.",
            **_compute_spread_stats(fold_accuracies, "cv_accuracy"),
            **_compute_spread_stats(fold_errors, "cv_error"),
            **_compute_spread_stats(fold_macro_f1s, "cv_macro_f1"),
            "mean_cv_balanced_accuracy": float(np.mean(fold_bal_accs)),
            "std_cv_balanced_accuracy": float(np.std(fold_bal_accs, ddof=0)),
            "oof_accuracy": float(oof_metrics["accuracy"]),
            "oof_macro_f1": float(oof_metrics["macro_f1"]),
            "total_fit_seconds": float(total_fit_sec),
            "total_predict_seconds": float(total_pred_sec),
            "selected_for_building": False,
        }
    )
    return base_row, oof_df


def build_validation_curve_dataframe(
    search_rows: Sequence[Mapping[str, Any]],
    model_id: str = MODEL_ID_PCA_KNN,
    weights: str = "uniform",
) -> pd.DataFrame:
    'Build supervised classification CV validation-curve data from candidate search records.'
    curve_records: list[dict[str, Any]] = []
    for row in search_rows:
        if str(row.get("candidate_role", "tuned_grid")) != "tuned_grid":
            continue
        b_id = int(row["building_id"])
        k_val = int(row["n_neighbors"])
        n_comp = int(row["n_components"])
        sc_mode = str(row["scaling"])
        w_mode = str(row.get("weights", weights))
        n_splits = int(row.get("cv_folds", 0))
        common = {
            "model_id": str(row.get("model_id", model_id)),
            "building": b_id,
            "building_id": b_id,
            "k": k_val,
            "n_neighbors": k_val,
            "pca_components": n_comp,
            "n_components": n_comp,
            "scaling": sc_mode,
            "weights": w_mode,
            "cv_folds": n_splits,
            "candidate_id": str(row["candidate_id"]),
            "feasible": bool(row["feasible"]),
            "status": str(row["status"]),
            "reason_code": row.get("reason_code"),
            "mean_cv_accuracy": row.get("mean_cv_accuracy", np.nan),
            "std_cv_accuracy": row.get("std_cv_accuracy", np.nan),
            "min_cv_accuracy": row.get("min_cv_accuracy", np.nan),
            "max_cv_accuracy": row.get("max_cv_accuracy", np.nan),
            "iqr_cv_accuracy": row.get("iqr_cv_accuracy", np.nan),
            "range_cv_accuracy": row.get("range_cv_accuracy", np.nan),
            "mean_cv_error": row.get("mean_cv_error", np.nan),
            "std_cv_error": row.get("std_cv_error", np.nan),
            "min_cv_error": row.get("min_cv_error", np.nan),
            "max_cv_error": row.get("max_cv_error", np.nan),
            "iqr_cv_error": row.get("iqr_cv_error", np.nan),
            "range_cv_error": row.get("range_cv_error", np.nan),
            "mean_cv_macro_f1": row.get("mean_cv_macro_f1", np.nan),
            "std_cv_macro_f1": row.get("std_cv_macro_f1", np.nan),
            "min_cv_macro_f1": row.get("min_cv_macro_f1", np.nan),
            "max_cv_macro_f1": row.get("max_cv_macro_f1", np.nan),
            "iqr_cv_macro_f1": row.get("iqr_cv_macro_f1", np.nan),
            "range_cv_macro_f1": row.get("range_cv_macro_f1", np.nan),
            "mean_cv_balanced_accuracy": row.get("mean_cv_balanced_accuracy", np.nan),
            "k1_no_multi_neighbor_weighting_advantage": bool(k_val == 1),
            "selected_for_building": bool(row.get("selected_for_building", False)),
            "curve_type": "supervised_classification_cv_validation_curve_not_kmeans_elbow",
            "selection_rule": "deterministic_max_mean_cv_accuracy_tiebreak_macro_f1_not_visual_elbow",
        }
        # Per-fold rows
        for fold_id in range(n_splits):
            curve_records.append(
                {
                    **common,
                    "record_level": "fold",
                    "fold": str(fold_id),
                    "fold_cv_accuracy": row.get(f"fold_{fold_id}_accuracy", np.nan),
                    "fold_cv_error": row.get(f"fold_{fold_id}_error", np.nan),
                    "fold_cv_macro_f1": row.get(f"fold_{fold_id}_macro_f1", np.nan),
                }
            )
        # Across-fold summary row
        curve_records.append(
            {
                **common,
                "record_level": "mean_across_folds",
                "fold": "mean_across_folds",
                "fold_cv_accuracy": row.get("mean_cv_accuracy", np.nan),
                "fold_cv_error": row.get("mean_cv_error", np.nan),
                "fold_cv_macro_f1": row.get("mean_cv_macro_f1", np.nan),
            }
        )
    return pd.DataFrame(curve_records)


def build_matched_knn_comparison_table(
    uniform_search_df: pd.DataFrame,
    weighted_search_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analysis 1: Compare matched `(building_id, n_components, n_neighbors, scaling)` pairs where only `weights` differ.

    Explicitly separates `k == 1` (no multi-neighbor distance-weighting voting advantage) from `k > 1`.
    """
    u_df = uniform_search_df[
        uniform_search_df.get("candidate_role", pd.Series("tuned_grid", index=uniform_search_df.index))
        == "tuned_grid"
    ].copy()
    w_df = weighted_search_df[
        weighted_search_df.get("candidate_role", pd.Series("tuned_grid", index=weighted_search_df.index))
        == "tuned_grid"
    ].copy()

    if not (u_df["weights"].astype(str) == "uniform").all():
        raise ValueError("uniform_search_df must contain only weights='uniform' candidates.")
    if not (w_df["weights"].astype(str) == "distance").all():
        raise ValueError("weighted_search_df must contain only weights='distance' candidates.")

    join_keys = ["building_id", "n_components", "n_neighbors", "scaling", "cv_folds"]
    u_keys = set(map(tuple, u_df[join_keys].to_numpy()))
    w_keys = set(map(tuple, w_df[join_keys].to_numpy()))
    if u_keys != w_keys:
        raise ValueError(
            "Matched comparison requires identical candidate grids and cv_folds between uniform and "
            f"distance-weighted kNN. Symmetric difference size: {len(u_keys ^ w_keys)}."
        )

    keep_cols = [
        *join_keys,
        "candidate_id",
        "feasible",
        "status",
        "reason_code",
        "mean_cv_accuracy",
        "std_cv_accuracy",
        "mean_cv_error",
        "mean_cv_macro_f1",
        "std_cv_macro_f1",
        "selected_for_building",
    ]
    merged = pd.merge(
        u_df[keep_cols],
        w_df[keep_cols],
        on=join_keys,
        suffixes=("_uniform", "_weighted"),
        how="inner",
    ).sort_values(by=["building_id", "n_components", "n_neighbors", "scaling"]).reset_index(drop=True)

    merged["delta_mean_cv_accuracy_weighted_minus_uniform"] = (
        merged["mean_cv_accuracy_weighted"] - merged["mean_cv_accuracy_uniform"]
    )
    merged["delta_mean_cv_error_weighted_minus_uniform"] = (
        merged["mean_cv_error_weighted"] - merged["mean_cv_error_uniform"]
    )
    merged["delta_mean_cv_macro_f1_weighted_minus_uniform"] = (
        merged["mean_cv_macro_f1_weighted"] - merged["mean_cv_macro_f1_uniform"]
    )
    merged["k_equals_1_no_multi_neighbor_advantage"] = merged["n_neighbors"].astype(int) == 1
    merged["is_substantive_multi_neighbor_comparison"] = merged["n_neighbors"].astype(int) > 1
    return merged


def build_selected_vs_matched_knn_summary(
    uniform_selection_config: Mapping[str, Any],
    weighted_selection_config: Mapping[str, Any],
    matched_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    'Build a structured two-part comparison separating matched-parameter effects from best-selected pipelines.'
    if uniform_selection_config.get("dataset_hash") != weighted_selection_config.get("dataset_hash"):
        raise ValueError("Dataset hash mismatch between uniform and distance-weighted selection configs.")
    if (
        uniform_selection_config.get("split_manifest_hash")
        != weighted_selection_config.get("split_manifest_hash")
    ):
        raise ValueError("Split manifest hash mismatch between uniform and distance-weighted selection configs.")
    if uniform_selection_config.get("cv_folds") != weighted_selection_config.get("cv_folds"):
        raise ValueError("CV fold count mismatch between uniform and distance-weighted selection configs.")

    u_tuned = uniform_selection_config["variants"]["tuned"]["per_building"]
    w_tuned = weighted_selection_config["variants"]["tuned"]["per_building"]
    u_fixed = uniform_selection_config["variants"]["fixed_baseline"]["per_building"]
    w_fixed = weighted_selection_config["variants"]["fixed_baseline"]["per_building"]

    buildings = sorted(u_tuned.keys(), key=int)
    per_building_selected: dict[str, Any] = {}

    for b_key in buildings:
        ub = u_tuned[b_key]
        wb = w_tuned[b_key]
        uf = u_fixed.get(b_key, {})
        wf = w_fixed.get(b_key, {})

        same_hyperparams_selected = (
            int(ub["n_components"]) == int(wb["n_components"])
            and int(ub["n_neighbors"]) == int(wb["n_neighbors"])
            and str(ub["scaling"]) == str(wb["scaling"])
        )

        # Look up pure weighting effect at the uniform winner's (PCA, k, scaling) and weighted winner's (PCA, k, scaling)
        matched_delta_at_uniform_winner: float | None = None
        matched_delta_at_weighted_winner: float | None = None
        if matched_df is not None and not matched_df.empty:
            m_u = matched_df[
                (matched_df["building_id"] == int(b_key))
                & (matched_df["n_components"] == int(ub["n_components"]))
                & (matched_df["n_neighbors"] == int(ub["n_neighbors"]))
                & (matched_df["scaling"] == str(ub["scaling"]))
            ]
            if len(m_u) == 1 and pd.notna(m_u.iloc[0]["delta_mean_cv_accuracy_weighted_minus_uniform"]):
                matched_delta_at_uniform_winner = float(
                    m_u.iloc[0]["delta_mean_cv_accuracy_weighted_minus_uniform"]
                )

            m_w = matched_df[
                (matched_df["building_id"] == int(b_key))
                & (matched_df["n_components"] == int(wb["n_components"]))
                & (matched_df["n_neighbors"] == int(wb["n_neighbors"]))
                & (matched_df["scaling"] == str(wb["scaling"]))
            ]
            if len(m_w) == 1 and pd.notna(m_w.iloc[0]["delta_mean_cv_accuracy_weighted_minus_uniform"]):
                matched_delta_at_weighted_winner = float(
                    m_w.iloc[0]["delta_mean_cv_accuracy_weighted_minus_uniform"]
                )

        uf_acc = uf.get("mean_cv_accuracy")
        tuning_gain_within_uniform = (
            float(ub["mean_cv_accuracy"]) - float(uf_acc) if uf_acc is not None else None
        )

        per_building_selected[b_key] = {
            "best_selected_uniform": {
                "candidate_id": ub["candidate_id"],
                "n_components": int(ub["n_components"]),
                "n_neighbors": int(ub["n_neighbors"]),
                "scaling": str(ub["scaling"]),
                "weights": "uniform",
                "mean_cv_accuracy": float(ub["mean_cv_accuracy"]),
                "mean_cv_macro_f1": float(ub["mean_cv_macro_f1"]),
            },
            "best_selected_weighted": {
                "candidate_id": wb["candidate_id"],
                "n_components": int(wb["n_components"]),
                "n_neighbors": int(wb["n_neighbors"]),
                "scaling": str(wb["scaling"]),
                "weights": "distance",
                "mean_cv_accuracy": float(wb["mean_cv_accuracy"]),
                "mean_cv_macro_f1": float(wb["mean_cv_macro_f1"]),
                "k1_no_multi_neighbor_weighting_advantage": bool(int(wb["n_neighbors"]) == 1),
            },
            "fixed_k1_pca50_baseline": {
                "uniform_mean_cv_accuracy": float(uf_acc) if uf_acc is not None else None,
                "weighted_mean_cv_accuracy": (
                    float(wf["mean_cv_accuracy"]) if wf.get("mean_cv_accuracy") is not None else None
                ),
                "note": "At k=1, distance weighting provides no multi-neighbor voting advantage.",
            },
            "decomposition": {
                "same_hyperparameters_selected": bool(same_hyperparams_selected),
                "tuning_gain_within_uniform_over_fixed_k1": tuning_gain_within_uniform,
                "matched_weighting_delta_at_uniform_winner_params": matched_delta_at_uniform_winner,
                "matched_weighting_delta_at_weighted_winner_params": matched_delta_at_weighted_winner,
                "best_selected_weighted_minus_best_selected_uniform_cv_accuracy": (
                    float(wb["mean_cv_accuracy"]) - float(ub["mean_cv_accuracy"])
                ),
                "best_selected_weighted_minus_best_selected_uniform_cv_macro_f1": (
                    float(wb["mean_cv_macro_f1"]) - float(ub["mean_cv_macro_f1"])
                ),
            },
        }

    matched_overview: dict[str, Any] = {}
    if matched_df is not None and not matched_df.empty:
        comp_df = matched_df[
            (matched_df["status_uniform"] == "completed") & (matched_df["status_weighted"] == "completed")
        ]
        k1_df = comp_df[comp_df["k_equals_1_no_multi_neighbor_advantage"]]
        k_gt_1_df = comp_df[comp_df["is_substantive_multi_neighbor_comparison"]]
        matched_overview = {
            "total_matched_completed_pairs": int(len(comp_df)),
            "k_equal_1_pairs": int(len(k1_df)),
            "k_gt_1_substantive_pairs": int(len(k_gt_1_df)),
            "k_equal_1_max_abs_accuracy_delta": (
                float(np.max(np.abs(k1_df["delta_mean_cv_accuracy_weighted_minus_uniform"])))
                if not k1_df.empty
                else 0.0
            ),
            "k_gt_1_mean_accuracy_delta_weighted_minus_uniform": (
                float(k_gt_1_df["delta_mean_cv_accuracy_weighted_minus_uniform"].mean())
                if not k_gt_1_df.empty
                else None
            ),
            "k_gt_1_max_accuracy_gain_weighted_minus_uniform": (
                float(k_gt_1_df["delta_mean_cv_accuracy_weighted_minus_uniform"].max())
                if not k_gt_1_df.empty
                else None
            ),
        }

    return {
        "comparison_contract": {
            "dataset_hash": str(uniform_selection_config["dataset_hash"]),
            "split_manifest_hash": str(uniform_selection_config["split_manifest_hash"]),
            "cv_folds": int(uniform_selection_config["cv_folds"]),
            "confounding_guard": (
                "Tuned distance-weighted kNN is compared both at matched (k, PCA, scaling) settings "
                "and against independently CV-selected uniform kNN, never solely against fixed k=1."
            ),
        },
        "analysis_1_matched_configurations": matched_overview,
        "analysis_2_independently_selected_pipelines": per_building_selected,
    }


def _export_dev_train_pca_projection(
    pipeline: Pipeline,
    X_bldg: pd.DataFrame,
    y_bldg: pd.Series,
    row_ids_bldg: pd.Series,
    split_roles_bldg: pd.Series,
    building_id: int,
    variant_name: str,
    run_id: str,
    output_dir: Path,
    model_id: str = MODEL_ID_PCA_KNN,
) -> dict[str, Any]:
    """Export PCA-transformed features with explicit fitted-partition provenance.

    Never creates a single PCA dataset across CV folds; only projects through a pipeline
    whose fitted partition (`dev_train_building_{building_id}`) is explicitly labeled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_steps = Pipeline(pipeline.steps[:-1])
    Z = prep_steps.transform(X_bldg)
    n_comp = int(Z.shape[1])
    pc_cols = [f"PC{i:03d}" for i in range(1, n_comp + 1)]

    proj_df = pd.DataFrame(Z, columns=pc_cols)
    proj_df[ROW_ID_COLUMN] = row_ids_bldg.astype(str).to_numpy()
    proj_df[BUILDING_COLUMN] = int(building_id)
    proj_df[TARGET_COLUMN] = y_bldg.astype(int).to_numpy()
    proj_df["source_split_role"] = split_roles_bldg.astype(str).to_numpy()
    proj_df["fitted_on_partition"] = f"dev_train_building_{int(building_id)}_only"
    proj_df["variant"] = str(variant_name)
    proj_df["run_id"] = str(run_id)

    csv_path = output_dir / f"pca_projection_{variant_name}_building_{int(building_id)}.csv"
    meta_path = output_dir / f"pca_projection_{variant_name}_building_{int(building_id)}.provenance.json"
    if csv_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite PCA projection file: {csv_path}")

    proj_df.to_csv(csv_path, index=False)
    pca_step = pipeline.named_steps["pca"]
    provenance = {
        "run_id": str(run_id),
        "model_id": str(model_id),
        "variant": str(variant_name),
        "building_id": int(building_id),
        "fitted_on_partition": f"dev_train_building_{int(building_id)}_only",
        "single_cross_fold_pca_dataset_created": False,
        "n_components": n_comp,
        "whiten": bool(pca_step.whiten),
        "standard_scaler_fitted": "scaler" in pipeline.named_steps,
        "explained_variance_ratio_sum": float(np.sum(pca_step.explained_variance_ratio_)),
        "projection_csv_path": str(csv_path.resolve()),
        "projection_csv_sha256": compute_file_sha256(csv_path),
        "row_count": int(len(proj_df)),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=False)
    return provenance


def train_pca_knn(
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    results_base_dir: Path | str = DEFAULT_RESULTS_DIR,
    processed_data_base_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
    profile: str | None = None,
    cv_folds: int | None = None,
    k_values: Sequence[int] | None = None,
    pca_components: Sequence[int] | None = None,
    scaling_modes: Sequence[str] | None = None,
    search_workers: int | None = None,
    estimator_n_jobs: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    run_id: str | None = None,
    save_dev_train_pca: bool = False,
    expected_model_id: str = MODEL_ID_PCA_KNN,
    expected_weights: str = "uniform",
    uniform_reference_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute fold-local CV search, select per-building models, refit on dev_train, and save artifacts."""
    if "validationdata" in Path(train_csv_path).name.lower():
        raise PermissionError(
            "Forbidden operation: official validationData.csv must never be passed to train_pca_knn."
        )

    cfg = load_pca_knn_config(
        config_path,
        expected_model_id=expected_model_id,
        expected_weights=expected_weights,
    )
    profiles_cfg: Mapping[str, Any] = cfg.get("search_profiles", {})
    active_profile_name = profile or str(cfg.get("default_profile", "full"))
    if active_profile_name not in profiles_cfg:
        raise ValueError(
            f"Unknown search profile {active_profile_name!r}; expected one of {sorted(profiles_cfg.keys())}."
        )
    prof = profiles_cfg[active_profile_name]

    resolved_cv_folds = int(cv_folds if cv_folds is not None else prof["cv_folds"])
    resolved_k_values = [int(k) for k in (k_values if k_values is not None else prof["k_values"])]
    resolved_pca_components = [
        int(c) for c in (pca_components if pca_components is not None else prof["pca_components"])
    ]
    resolved_scaling_modes = [
        str(s) for s in (scaling_modes if scaling_modes is not None else prof["scaling_modes"])
    ]

    par_cfg = cfg.get("parallelism", {})
    req_search_workers = int(
        search_workers if search_workers is not None else par_cfg.get("search_workers", 1)
    )
    req_est_n_jobs = int(
        estimator_n_jobs if estimator_n_jobs is not None else par_cfg.get("estimator_n_jobs", 1)
    )
    eff_search_workers, eff_est_n_jobs, par_warnings = resolve_parallelism(
        search_workers=req_search_workers,
        estimator_n_jobs=req_est_n_jobs,
    )
    run_warnings: list[str] = list(par_warnings)

    prep_cfg = cfg.get("preprocessing", {})
    sentinel_val = prep_cfg.get("sentinel_value", UNDETECTED_SENTINEL)
    replacement_val = float(prep_cfg.get("replacement_value", DEFAULT_UNDETECTED_REPLACEMENT))
    encoding_sha256 = compute_encoding_config_sha256(
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
    )
    sel_tolerance = float(cfg.get("selection_policy", {}).get("tolerance", 1e-9))
    assumptions_tuple = get_model_assumptions(weights=expected_weights)

    # Load and validate training data (supports either raw trainingData.csv or cleaned development_cleaned.csv)
    train_path = Path(train_csv_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found at: {train_path}")

    raw_or_cleaned_df = pd.read_csv(train_path)
    if SPLIT_ROLE_OFFICIAL_VALIDATION in set(
        raw_or_cleaned_df.get("split_role", pd.Series(dtype=str)).astype(str).unique()
    ):
        raise PermissionError(
            "Forbidden operation: DataFrame containing official_validation split_role cannot be used in train.py."
        )

    manifest_df, split_manifest_sha256, _ = load_split_manifest(split_manifest_path, strict=True)
    manifest_source_sha = str(manifest_df["source_sha256"].iloc[0])

    train_file_sha256 = compute_file_sha256(train_path)
    if "source_sha256" in raw_or_cleaned_df.columns:
        df_source_shas = raw_or_cleaned_df["source_sha256"].astype(str).unique()
        if len(df_source_shas) != 1 or df_source_shas[0] != manifest_source_sha:
            raise SplitCreationError(
                f"Cleaned dataset source_sha256 ({df_source_shas.tolist()}) does not match "
                f"split manifest source_sha256 ({manifest_source_sha})."
            )
        expected_sha_for_align = manifest_source_sha
    else:
        df_sha = compute_dataframe_sha256(raw_or_cleaned_df)
        if train_file_sha256 == manifest_source_sha:
            expected_sha_for_align = train_file_sha256
        elif df_sha == manifest_source_sha:
            expected_sha_for_align = df_sha
        else:
            raise SplitCreationError(
                f"Training dataset hash (file={train_file_sha256}, df={df_sha}) does not match "
                f"split manifest source_sha256 ({manifest_source_sha})."
            )

    aligned_df = align_dataset_with_manifest(
        train_df=raw_or_cleaned_df,
        manifest_df=manifest_df,
        expected_source_sha256=expected_sha_for_align,
    )

    X_all, y_all, building_all, row_ids_all = extract_validated_wap_features_and_targets(
        aligned_df,
        require_row_id=True,
        allow_already_cleaned_replacement=True,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
    )
    assert row_ids_all is not None

    # Create unique non-overwriting run directories for artifacts and results
    resolved_run_id, run_artifacts_dir = create_unique_run_dir(
        base_dir=artifacts_base_dir,
        model_id=expected_model_id,
        evaluation_role="train",
        run_id=run_id,
    )
    _, run_results_dir = create_unique_run_dir(
        base_dir=results_base_dir,
        model_id=expected_model_id,
        evaluation_role="train",
        run_id=resolved_run_id,
    )

    fixed_artifacts_dir = run_artifacts_dir / "fixed_baseline"
    tuned_artifacts_dir = run_artifacts_dir / "tuned"
    fixed_artifacts_dir.mkdir(parents=True, exist_ok=False)
    tuned_artifacts_dir.mkdir(parents=True, exist_ok=False)

    candidate_grid = generate_candidate_grid(
        pca_components=resolved_pca_components,
        k_values=resolved_k_values,
        scaling_modes=resolved_scaling_modes,
        weights=expected_weights,
    )

    buildings = sorted(int(b) for b in building_all.unique())
    holdout_mask_global = aligned_df["split_role"].astype(str) == SPLIT_ROLE_INTERNAL_HOLDOUT
    holdout_indices_set = set(int(i) for i in np.flatnonzero(holdout_mask_global.to_numpy()))

    all_search_rows: list[dict[str, Any]] = []
    tuned_selected_by_building: dict[str, dict[str, Any]] = {}
    best_k_gt_1_by_building: dict[str, dict[str, Any]] = {}
    fixed_baseline_by_building: dict[str, dict[str, Any]] = {}

    tuned_oof_frames: list[pd.DataFrame] = []
    fixed_oof_frames: list[pd.DataFrame] = []

    tuned_artifact_paths: dict[str, str] = {}
    fixed_artifact_paths: dict[str, str] = {}
    pca_export_provenance: dict[str, Any] = {}

    total_tuned_cv_fit_sec = 0.0
    total_tuned_cv_pred_sec = 0.0
    total_tuned_refit_sec = 0.0

    total_fixed_cv_fit_sec = 0.0
    total_fixed_cv_pred_sec = 0.0
    total_fixed_refit_sec = 0.0

    fixed_baseline_all_feasible = True
    fixed_baseline_skip_reasons: list[dict[str, Any]] = []

    for b_id in buildings:
        b_key = str(b_id)
        b_dev_mask = (building_all.to_numpy() == b_id) & (
            aligned_df["split_role"].astype(str).to_numpy() == SPLIT_ROLE_DEV_TRAIN
        )
        b_dev_indices = np.flatnonzero(b_dev_mask)
        if b_dev_indices.size == 0:
            raise SplitCreationError(f"Building {b_id} has 0 dev_train rows.")
        assert set(int(i) for i in b_dev_indices).isdisjoint(holdout_indices_set)

        folds = list(iter_building_cv_folds(manifest_df, building_id=b_id, n_splits=resolved_cv_folds))

        # 1. Evaluate all tuned grid candidates for this building
        if eff_search_workers > 1 and len(candidate_grid) > 1:
            eval_outputs = joblib.Parallel(n_jobs=eff_search_workers)(
                joblib.delayed(evaluate_single_candidate_cv)(
                    candidate=cand,
                    building_id=b_id,
                    X_full=X_all,
                    y_full=y_all,
                    row_ids_full=row_ids_all,
                    folds=folds,
                    building_dev_train_indices=b_dev_indices,
                    holdout_indices_set=holdout_indices_set,
                    sentinel_value=sentinel_val,
                    replacement_value=replacement_val,
                    seed=seed,
                    estimator_n_jobs=eff_est_n_jobs,
                    candidate_role_label="tuned_grid",
                    weights=expected_weights,
                    model_id=expected_model_id,
                )
                for cand in candidate_grid
            )
        else:
            eval_outputs = [
                evaluate_single_candidate_cv(
                    candidate=cand,
                    building_id=b_id,
                    X_full=X_all,
                    y_full=y_all,
                    row_ids_full=row_ids_all,
                    folds=folds,
                    building_dev_train_indices=b_dev_indices,
                    holdout_indices_set=holdout_indices_set,
                    sentinel_value=sentinel_val,
                    replacement_value=replacement_val,
                    seed=seed,
                    estimator_n_jobs=eff_est_n_jobs,
                    candidate_role_label="tuned_grid",
                    weights=expected_weights,
                    model_id=expected_model_id,
                )
                for cand in candidate_grid
            ]

        b_rows: list[dict[str, Any]] = []
        b_oof_by_cand_id: dict[str, pd.DataFrame] = {}
        for row_dict, oof_df in eval_outputs:
            b_rows.append(row_dict)
            if oof_df is not None:
                b_oof_by_cand_id[str(row_dict["candidate_id"])] = oof_df

        best_cand = select_best_pca_knn_candidate(b_rows, tolerance=sel_tolerance)
        best_cand_id = str(best_cand["candidate_id"])
        for r in b_rows:
            if str(r["candidate_id"]) == best_cand_id:
                r["selected_for_building"] = True
            all_search_rows.append(r)

        # Also record best multi-neighbor (k > 1) candidate for diagnostic transparency
        k_gt_1_feasible = [
            r for r in b_rows if int(r["n_neighbors"]) > 1 and r.get("status") == "completed"
        ]
        if k_gt_1_feasible:
            best_k_gt_1 = select_best_pca_knn_candidate(k_gt_1_feasible, tolerance=sel_tolerance)
            best_k_gt_1_by_building[b_key] = {
                "candidate_id": str(best_k_gt_1["candidate_id"]),
                "n_components": int(best_k_gt_1["n_components"]),
                "n_neighbors": int(best_k_gt_1["n_neighbors"]),
                "scaling": str(best_k_gt_1["scaling"]),
                "weights": str(expected_weights),
                "mean_cv_accuracy": float(best_k_gt_1["mean_cv_accuracy"]),
                "mean_cv_macro_f1": float(best_k_gt_1["mean_cv_macro_f1"]),
            }

        if expected_weights == "distance" and int(best_cand["n_neighbors"]) == 1:
            run_warnings.append(
                f"Building {b_id} selected n_neighbors=1 under weights='distance': at k=1, inverse-distance "
                "weighting offers no multi-neighbor voting advantage over uniform 1NN."
            )

        tuned_oof_frames.append(b_oof_by_cand_id[best_cand_id])
        total_tuned_cv_fit_sec += float(best_cand["total_fit_seconds"])
        total_tuned_cv_pred_sec += float(best_cand["total_predict_seconds"])

        # Refit winning tuned pipeline strictly on dev_train rows for building b_id
        X_dev_b = X_all.iloc[b_dev_indices]
        y_dev_b = y_all.iloc[b_dev_indices].to_numpy(dtype=int)
        b_classes = sorted(int(c) for c in np.unique(y_dev_b))

        tuned_pipeline = build_configured_pca_knn_pipeline(
            n_components=int(best_cand["n_components"]),
            n_neighbors=int(best_cand["n_neighbors"]),
            scaling=str(best_cand["scaling"]),
            weights=expected_weights,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            enforce_wap_schema=True,
            random_state=seed,
            n_jobs=eff_est_n_jobs,
        )
        t_refit0 = time.perf_counter()
        tuned_pipeline.fit(X_dev_b, y_dev_b)
        refit_elapsed = time.perf_counter() - t_refit0
        total_tuned_refit_sec += refit_elapsed

        tuned_prep_desc = describe_preprocessing_config(
            scale=bool(best_cand["scale"]),
            use_pca=True,
            n_components=int(best_cand["n_components"]),
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        tuned_params_record = {
            "variant": "tuned",
            "candidate_id": best_cand_id,
            "n_components": int(best_cand["n_components"]),
            "n_neighbors": int(best_cand["n_neighbors"]),
            "scaling": str(best_cand["scaling"]),
            "scale": bool(best_cand["scale"]),
            "whiten": False,
            "weights": str(expected_weights),
            "metric": "euclidean",
            "k1_no_multi_neighbor_weighting_advantage": bool(int(best_cand["n_neighbors"]) == 1),
            "mean_cv_accuracy": float(best_cand["mean_cv_accuracy"]),
            "std_cv_accuracy": float(best_cand["std_cv_accuracy"]),
            "mean_cv_error": float(best_cand["mean_cv_error"]),
            "mean_cv_macro_f1": float(best_cand["mean_cv_macro_f1"]),
            "std_cv_macro_f1": float(best_cand["std_cv_macro_f1"]),
            "cv_folds": int(resolved_cv_folds),
            "refit_seconds": float(refit_elapsed),
            "dev_train_rows": int(len(b_dev_indices)),
        }
        tuned_selected_by_building[b_key] = tuned_params_record

        tuned_meta = save_pipeline_artifact(
            pipeline=tuned_pipeline,
            output_dir=tuned_artifacts_dir,
            building_id=b_id,
            model_id=expected_model_id,
            feature_names=WAP_COLUMNS,
            class_labels=b_classes,
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            preprocessing_config=tuned_prep_desc,
            selected_parameters=tuned_params_record,
            run_id=resolved_run_id,
        )
        tuned_artifact_paths[b_key] = str(tuned_meta["artifact_path"])

        if save_dev_train_pca:
            b_all_mask = building_all.to_numpy() == b_id
            pca_export_provenance[f"tuned_building_{b_id}"] = _export_dev_train_pca_projection(
                pipeline=tuned_pipeline,
                X_bldg=X_all.loc[b_all_mask],
                y_bldg=y_all.loc[b_all_mask],
                row_ids_bldg=row_ids_all.loc[b_all_mask],
                split_roles_bldg=aligned_df.loc[b_all_mask, "split_role"],
                building_id=b_id,
                variant_name="tuned",
                run_id=resolved_run_id,
                output_dir=Path(processed_data_base_dir) / resolved_run_id,
                model_id=expected_model_id,
            )

        # 2. Evaluate and refit fixed paper-inspired baseline (k=1, PCA=50, scaling='none', whiten=False)
        fixed_cand_spec = {
            "candidate_index": -1,
            "candidate_id": "fixed_baseline_pca50_k1_scale_none",
            "n_components": int(FIXED_BASELINE_PARAMS["n_components"]),
            "n_neighbors": int(FIXED_BASELINE_PARAMS["n_neighbors"]),
            "scaling": str(FIXED_BASELINE_PARAMS["scaling"]),
            "scale": False,
            "whiten": False,
            "weights": str(expected_weights),
            "metric": "euclidean",
            "is_fixed_baseline": True,
            "k1_no_multi_neighbor_weighting_advantage": True,
        }
        fixed_row, fixed_oof_df = evaluate_single_candidate_cv(
            candidate=fixed_cand_spec,
            building_id=b_id,
            X_full=X_all,
            y_full=y_all,
            row_ids_full=row_ids_all,
            folds=folds,
            building_dev_train_indices=b_dev_indices,
            holdout_indices_set=holdout_indices_set,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            seed=seed,
            estimator_n_jobs=eff_est_n_jobs,
            candidate_role_label="fixed_baseline",
            weights=expected_weights,
            model_id=expected_model_id,
        )
        if not any(bool(r.get("is_fixed_baseline", False)) for r in b_rows):
            all_search_rows.append(fixed_row)

        if fixed_row["status"] == "completed" and fixed_oof_df is not None:
            fixed_oof_frames.append(fixed_oof_df)
            total_fixed_cv_fit_sec += float(fixed_row["total_fit_seconds"])
            total_fixed_cv_pred_sec += float(fixed_row["total_predict_seconds"])

            fixed_pipeline = build_fixed_baseline_pipeline(
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
                enforce_wap_schema=True,
                random_state=seed,
                n_jobs=eff_est_n_jobs,
                weights=expected_weights,
            )
            tf0 = time.perf_counter()
            fixed_pipeline.fit(X_dev_b, y_dev_b)
            fixed_refit_sec = time.perf_counter() - tf0
            total_fixed_refit_sec += fixed_refit_sec

            fixed_prep_desc = describe_preprocessing_config(
                scale=False,
                use_pca=True,
                n_components=int(FIXED_BASELINE_PARAMS["n_components"]),
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
            )
            fixed_params_record = {
                "variant": "fixed_baseline",
                "candidate_id": fixed_cand_spec["candidate_id"],
                "n_components": int(FIXED_BASELINE_PARAMS["n_components"]),
                "n_neighbors": int(FIXED_BASELINE_PARAMS["n_neighbors"]),
                "scaling": "none",
                "scale": False,
                "whiten": False,
                "weights": str(expected_weights),
                "metric": "euclidean",
                "k1_no_multi_neighbor_weighting_advantage": True,
                "mean_cv_accuracy": float(fixed_row["mean_cv_accuracy"]),
                "mean_cv_macro_f1": float(fixed_row["mean_cv_macro_f1"]),
                "cv_folds": int(resolved_cv_folds),
                "refit_seconds": float(fixed_refit_sec),
                "dev_train_rows": int(len(b_dev_indices)),
                "status": "completed",
            }
            fixed_baseline_by_building[b_key] = fixed_params_record

            fixed_meta = save_pipeline_artifact(
                pipeline=fixed_pipeline,
                output_dir=fixed_artifacts_dir,
                building_id=b_id,
                model_id=expected_model_id,
                feature_names=WAP_COLUMNS,
                class_labels=b_classes,
                dataset_hash=expected_sha_for_align,
                split_manifest_hash=split_manifest_sha256,
                preprocessing_config=fixed_prep_desc,
                selected_parameters=fixed_params_record,
                run_id=resolved_run_id,
            )
            fixed_artifact_paths[b_key] = str(fixed_meta["artifact_path"])

            if save_dev_train_pca:
                b_all_mask = building_all.to_numpy() == b_id
                pca_export_provenance[f"fixed_baseline_building_{b_id}"] = _export_dev_train_pca_projection(
                    pipeline=fixed_pipeline,
                    X_bldg=X_all.loc[b_all_mask],
                    y_bldg=y_all.loc[b_all_mask],
                    row_ids_bldg=row_ids_all.loc[b_all_mask],
                    split_roles_bldg=aligned_df.loc[b_all_mask, "split_role"],
                    building_id=b_id,
                    variant_name="fixed_baseline",
                    run_id=resolved_run_id,
                    output_dir=Path(processed_data_base_dir) / resolved_run_id,
                    model_id=expected_model_id,
                )
        else:
            fixed_baseline_all_feasible = False
            skip_entry = {
                "building_id": b_id,
                "status": fixed_row["status"],
                "reason_code": fixed_row["reason_code"],
                "message": fixed_row["message"],
            }
            fixed_baseline_skip_reasons.append(skip_entry)
            fixed_baseline_by_building[b_key] = {
                "variant": "fixed_baseline",
                "n_components": int(FIXED_BASELINE_PARAMS["n_components"]),
                "n_neighbors": int(FIXED_BASELINE_PARAMS["n_neighbors"]),
                "scaling": "none",
                "scale": False,
                "whiten": False,
                "weights": str(expected_weights),
                "metric": "euclidean",
                "k1_no_multi_neighbor_weighting_advantage": True,
                "status": str(fixed_row["status"]),
                "reason_code": fixed_row["reason_code"],
                "message": fixed_row["message"],
            }

    # Save search_results.csv containing every evaluated, skipped, or failed candidate
    search_results_df = pd.DataFrame(all_search_rows)
    search_results_csv_path = run_results_dir / "search_results.csv"
    search_results_df.to_csv(search_results_csv_path, index=False)

    # Save classification CV validation_curve.csv
    validation_curve_df = build_validation_curve_dataframe(
        all_search_rows,
        model_id=expected_model_id,
        weights=expected_weights,
    )
    validation_curve_csv_path = run_results_dir / "validation_curve.csv"
    validation_curve_df.to_csv(validation_curve_csv_path, index=False)

    # Save out-of-fold CV predictions and shared-schema metrics.json for tuned models
    tuned_oof_all = pd.concat(tuned_oof_frames, axis=0, ignore_index=True)
    tuned_oof_csv_path = run_results_dir / "cv_tuned_oof_predictions.csv"
    save_predictions_csv(tuned_oof_all, tuned_oof_csv_path, evaluation_role=EVALUATION_ROLE_CV_VAL)

    t_per_bldg, t_pooled, t_supports, t_cms = evaluate_predictions_by_building(tuned_oof_all)
    for b_key, b_sel in tuned_selected_by_building.items():
        if b_key in t_per_bldg:
            t_per_bldg[b_key]["mean_fold_cv_accuracy"] = b_sel["mean_cv_accuracy"]
            t_per_bldg[b_key]["mean_fold_cv_macro_f1"] = b_sel["mean_cv_macro_f1"]

    scales_selected = {bool(v["scale"]) for v in tuned_selected_by_building.values()}
    pca_selected = sorted({int(v["n_components"]) for v in tuned_selected_by_building.values()})
    tuned_prep_summary = describe_preprocessing_config(
        scale=(len(scales_selected) == 1 and next(iter(scales_selected))),
        use_pca=True,
        n_components=pca_selected[0] if len(pca_selected) == 1 else None,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
    )
    tuned_prep_summary["encoding_config_sha256"] = encoding_sha256
    tuned_prep_summary["per_building_preprocessing"] = {
        b_key: describe_preprocessing_config(
            scale=bool(v["scale"]),
            use_pca=True,
            n_components=int(v["n_components"]),
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        for b_key, v in tuned_selected_by_building.items()
    }

    cv_tuned_result = build_completed_result(
        run_id=f"{resolved_run_id}_cv_tuned",
        model_id=expected_model_id,
        dataset_hash=expected_sha_for_align,
        split_manifest_hash=split_manifest_sha256,
        evaluation_role=EVALUATION_ROLE_CV_VAL,
        preprocessing=tuned_prep_summary,
        selected_parameters={
            "variant": "tuned",
            "weights": expected_weights,
            "profile": active_profile_name,
            "cv_folds": resolved_cv_folds,
            "per_building": tuned_selected_by_building,
            "best_k_gt_1_per_building": best_k_gt_1_by_building,
        },
        selection_metric={
            "primary_metric": "mean_cv_accuracy",
            "tie_breaker_metric": "mean_cv_macro_f1",
            "tolerance": sel_tolerance,
            "further_tie_breakers": list(DECLARED_TIE_BREAKER_POLICY),
        },
        per_building=t_per_bldg,
        pooled=t_pooled,
        supports=t_supports,
        confusion_matrices=t_cms,
        timing={
            "cv_fit_seconds": float(total_tuned_cv_fit_sec),
            "cv_predict_seconds": float(total_tuned_cv_pred_sec),
            "dev_train_refit_seconds": float(total_tuned_refit_sec),
            "total_seconds": float(
                total_tuned_cv_fit_sec + total_tuned_cv_pred_sec + total_tuned_refit_sec
            ),
        },
        prediction_file_path=tuned_oof_csv_path,
        model_artifact_paths=tuned_artifact_paths,
        assumptions=assumptions_tuple,
        warnings=run_warnings,
        extra_metadata={
            "variant": "tuned",
            "weights": expected_weights,
            "search_profile": active_profile_name,
            "search_results_csv": str(search_results_csv_path),
            "validation_curve_csv": str(validation_curve_csv_path),
        },
    )
    cv_tuned_metrics_path = run_results_dir / "cv_tuned_metrics.json"
    save_result_json(cv_tuned_result, cv_tuned_metrics_path)

    # Save out-of-fold CV predictions and shared-schema metrics.json for fixed_baseline
    fixed_baseline_spec_for_model = {
        **FIXED_BASELINE_PARAMS,
        "weights": str(expected_weights),
        "k1_no_multi_neighbor_weighting_advantage": True,
    }
    cv_fixed_metrics_path = run_results_dir / "cv_fixed_baseline_metrics.json"
    fixed_oof_csv_path: Path | None = None
    if fixed_baseline_all_feasible and len(fixed_oof_frames) == len(buildings):
        fixed_oof_all = pd.concat(fixed_oof_frames, axis=0, ignore_index=True)
        fixed_oof_csv_path = run_results_dir / "cv_fixed_baseline_oof_predictions.csv"
        save_predictions_csv(fixed_oof_all, fixed_oof_csv_path, evaluation_role=EVALUATION_ROLE_CV_VAL)
        f_per_bldg, f_pooled, f_supports, f_cms = evaluate_predictions_by_building(fixed_oof_all)
        fixed_prep_summary = describe_preprocessing_config(
            scale=False,
            use_pca=True,
            n_components=int(FIXED_BASELINE_PARAMS["n_components"]),
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        fixed_prep_summary["encoding_config_sha256"] = encoding_sha256
        cv_fixed_result = build_completed_result(
            run_id=f"{resolved_run_id}_cv_fixed_baseline",
            model_id=expected_model_id,
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            evaluation_role=EVALUATION_ROLE_CV_VAL,
            preprocessing=fixed_prep_summary,
            selected_parameters={
                "variant": "fixed_baseline",
                "fixed_baseline_spec": fixed_baseline_spec_for_model,
                "cv_folds": resolved_cv_folds,
                "per_building": fixed_baseline_by_building,
            },
            selection_metric="fixed_paper_inspired_baseline_no_tuning",
            per_building=f_per_bldg,
            pooled=f_pooled,
            supports=f_supports,
            confusion_matrices=f_cms,
            timing={
                "cv_fit_seconds": float(total_fixed_cv_fit_sec),
                "cv_predict_seconds": float(total_fixed_cv_pred_sec),
                "dev_train_refit_seconds": float(total_fixed_refit_sec),
                "total_seconds": float(
                    total_fixed_cv_fit_sec + total_fixed_cv_pred_sec + total_fixed_refit_sec
                ),
            },
            prediction_file_path=fixed_oof_csv_path,
            model_artifact_paths=fixed_artifact_paths,
            assumptions=assumptions_tuple,
            warnings=run_warnings,
            extra_metadata={
                "variant": "fixed_baseline",
                "weights": expected_weights,
                "k1_no_multi_neighbor_weighting_advantage": True,
                "search_profile": active_profile_name,
                "search_results_csv": str(search_results_csv_path),
                "validation_curve_csv": str(validation_curve_csv_path),
            },
        )
        save_result_json(cv_fixed_result, cv_fixed_metrics_path)
    else:
        cv_fixed_result = build_non_completed_result(
            run_id=f"{resolved_run_id}_cv_fixed_baseline",
            model_id=expected_model_id,
            status="skipped",
            evaluation_role=EVALUATION_ROLE_CV_VAL,
            reason=(
                "Fixed paper-inspired baseline (PCA=50, k=1) was not feasible across all buildings "
                "for the current training partition size."
            ),
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            preprocessing=describe_preprocessing_config(
                scale=False,
                use_pca=True,
                n_components=int(FIXED_BASELINE_PARAMS["n_components"]),
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
            ),
            selected_parameters={
                "variant": "fixed_baseline",
                "fixed_baseline_spec": fixed_baseline_spec_for_model,
                "per_building": fixed_baseline_by_building,
            },
            selection_metric="fixed_paper_inspired_baseline_no_tuning",
            skip_reasons=fixed_baseline_skip_reasons,
            assumptions=assumptions_tuple,
            warnings=run_warnings,
        )
        save_result_json(cv_fixed_result, cv_fixed_metrics_path)

    # Save frozen selection_config.json for test.py and cross-model comparison
    selection_config: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": resolved_run_id,
        "model_id": expected_model_id,
        "weights": expected_weights,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_split_role": SPLIT_ROLE_DEV_TRAIN,
        "internal_holdout_used_in_selection": False,
        "official_validation_used_in_selection": False,
        "dataset_hash": expected_sha_for_align,
        "split_manifest_hash": split_manifest_sha256,
        "encoding_config_sha256": encoding_sha256,
        "search_profile": active_profile_name,
        "cv_folds": resolved_cv_folds,
        "candidate_grid_spec": {
            "pca_components": resolved_pca_components,
            "k_values": resolved_k_values,
            "scaling_modes": resolved_scaling_modes,
            "weights": expected_weights,
        },
        "selection_policy": {
            "primary_metric": "mean_cv_accuracy",
            "tie_breaker_metric": "mean_cv_macro_f1",
            "tolerance": sel_tolerance,
            "deterministic_tie_breakers": list(DECLARED_TIE_BREAKER_POLICY),
        },
        "parallelism": {
            "requested_search_workers": req_search_workers,
            "effective_search_workers": eff_search_workers,
            "requested_estimator_n_jobs": req_est_n_jobs,
            "effective_estimator_n_jobs": eff_est_n_jobs,
            "warnings": par_warnings,
        },
        "variants": {
            "tuned": {
                "status": "completed",
                "per_building": tuned_selected_by_building,
                "best_k_gt_1_per_building": best_k_gt_1_by_building,
                "artifact_paths": tuned_artifact_paths,
                "cv_metrics_json": str(cv_tuned_metrics_path.resolve()),
                "cv_oof_predictions_csv": str(tuned_oof_csv_path.resolve()),
            },
            "fixed_baseline": {
                "status": "completed" if fixed_baseline_all_feasible else "skipped",
                "fixed_baseline_spec": fixed_baseline_spec_for_model,
                "per_building": fixed_baseline_by_building,
                "artifact_paths": fixed_artifact_paths,
                "cv_metrics_json": str(cv_fixed_metrics_path.resolve()),
                "cv_oof_predictions_csv": (
                    str(fixed_oof_csv_path.resolve()) if fixed_oof_csv_path is not None else None
                ),
            },
        },
        "search_results_csv": str(search_results_csv_path.resolve()),
        "validation_curve_csv": str(validation_curve_csv_path.resolve()),
        "pca_projection_exports": pca_export_provenance,
        "environment": collect_environment_info(),
    }

    # Optional matched & independently selected comparison when training distance-weighted kNN
    matched_comparison_csv_path: Path | None = None
    matched_summary_json_path: Path | None = None
    if uniform_reference_run_dir is not None:
        u_dir = Path(uniform_reference_run_dir)
        u_sel_path = u_dir / "selection_config.json"
        if not u_sel_path.is_file():
            raise FileNotFoundError(f"Uniform reference selection_config.json not found at: {u_sel_path}")
        with u_sel_path.open("r", encoding="utf-8") as fh:
            u_sel_cfg = json.load(fh)
        u_search_csv = Path(u_sel_cfg["search_results_csv"])
        if not u_search_csv.is_file():
            raise FileNotFoundError(f"Uniform reference search_results.csv not found at: {u_search_csv}")
        u_search_df = pd.read_csv(u_search_csv)
        matched_df = build_matched_knn_comparison_table(u_search_df, search_results_df)
        matched_comparison_csv_path = run_results_dir / "matched_uniform_vs_weighted_cv.csv"
        matched_df.to_csv(matched_comparison_csv_path, index=False)

        comp_summary = build_selected_vs_matched_knn_summary(
            uniform_selection_config=u_sel_cfg,
            weighted_selection_config=selection_config,
            matched_df=matched_df,
        )
        matched_summary_json_path = run_results_dir / "uniform_vs_weighted_comparison_summary.json"
        with matched_summary_json_path.open("w", encoding="utf-8") as fh:
            json.dump(comp_summary, fh, indent=2, sort_keys=False)

        selection_config["matched_uniform_comparison"] = {
            "uniform_run_id": u_sel_cfg.get("run_id"),
            "matched_cv_csv": str(matched_comparison_csv_path.resolve()),
            "comparison_summary_json": str(matched_summary_json_path.resolve()),
        }

    selection_config_results_path = run_results_dir / "selection_config.json"
    selection_config_artifacts_path = run_artifacts_dir / "selection_config.json"
    for sc_path in (selection_config_results_path, selection_config_artifacts_path):
        with sc_path.open("w", encoding="utf-8") as fh:
            json.dump(selection_config, fh, indent=2, sort_keys=False)

    return {
        "run_id": resolved_run_id,
        "model_id": expected_model_id,
        "weights": expected_weights,
        "artifacts_dir": str(run_artifacts_dir.resolve()),
        "results_dir": str(run_results_dir.resolve()),
        "search_results_csv": str(search_results_csv_path.resolve()),
        "validation_curve_csv": str(validation_curve_csv_path.resolve()),
        "selection_config_json": str(selection_config_results_path.resolve()),
        "cv_tuned_metrics_json": str(cv_tuned_metrics_path.resolve()),
        "cv_fixed_baseline_metrics_json": str(cv_fixed_metrics_path.resolve()),
        "matched_uniform_vs_weighted_cv_csv": (
            str(matched_comparison_csv_path.resolve()) if matched_comparison_csv_path is not None else None
        ),
        "uniform_vs_weighted_comparison_summary_json": (
            str(matched_summary_json_path.resolve()) if matched_summary_json_path is not None else None
        ),
        "selection_config": selection_config,
    }


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected comma-separated integers, got empty string.")
    return [int(p) for p in parts]


def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected comma-separated strings, got empty string.")
    return parts


def build_arg_parser(
    prog: str = "python -m models.pca_knn.train",
    description: str = (
        "Run fold-local cross-validation search for PCA + uniform-weight kNN on shared development "
        "splits, preserve the paper-inspired baseline (k=1, PCA=50, unscaled), and refit per-building "
        "pipelines on dev_train only."
    ),
    default_config_path: Path = DEFAULT_CONFIG_YAML,
    default_artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    default_results_dir: Path = DEFAULT_RESULTS_DIR,
    default_processed_data_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    include_uniform_reference_arg: bool = False,
) -> argparse.ArgumentParser:
    """Build CLI argument parser for PCA-kNN training scripts."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to trainingData.csv or cleaned development_cleaned.csv.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_CSV,
        help="Path to shared development_splits.csv manifest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Path to model config.yaml (default: {default_config_path}).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=("smoke", "lightweight", "full"),
        default=None,
        help="Staged search profile from config.yaml: 'smoke', 'lightweight', or 'full' (default: config default 'full').",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override CV fold count (e.g. 3 or 5 matching shared manifest columns fold_3 / fold_5).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Optional comma-separated neighbor counts override, e.g. '1,3,5,7,9,11,15'.",
    )
    parser.add_argument(
        "--pca-components",
        type=str,
        default=None,
        help="Optional comma-separated PCA component counts override, e.g. '5,50,200'.",
    )
    parser.add_argument(
        "--scaling-modes",
        type=str,
        default=None,
        help="Optional comma-separated scaling modes override: 'none', 'standard', or 'none,standard'.",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=None,
        help="Number of parallel candidate workers >= 1 (if > 1, estimator n_jobs is forced to 1).",
    )
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=None,
        help="KNeighborsClassifier n_jobs >= 1 (automatically forced to 1 if --search-workers > 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic PCA solver (default: 42).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom run_id (must not already exist in artifacts or results directories).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts_dir,
        help=f"Base directory for saved model artifacts (default: {default_artifacts_dir}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Base directory for CV results, search_results.csv, and validation_curve.csv (default: {default_results_dir}).",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=default_processed_data_dir,
        help=f"Base directory for optional PCA projection exports (default: {default_processed_data_dir}).",
    )
    parser.add_argument(
        "--save-dev-train-pca",
        action="store_true",
        help="Optionally export PCA-transformed features with explicit dev_train fitted-partition provenance.",
    )
    if include_uniform_reference_arg:
        parser.add_argument(
            "--uniform-reference-run-dir",
            type=Path,
            default=None,
            help=(
                "Optional path to a completed models/pca_knn/artifacts/<run_id> or results/<run_id> "
                "directory to emit matched-configuration and best-selected uniform vs weighted comparisons."
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m models.pca_knn.train`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = train_pca_knn(
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        config_path=args.config,
        artifacts_base_dir=args.artifacts_dir,
        results_base_dir=args.results_dir,
        processed_data_base_dir=args.processed_data_dir,
        profile=args.profile,
        cv_folds=args.cv_folds,
        k_values=_parse_int_list(args.k_values),
        pca_components=_parse_int_list(args.pca_components),
        scaling_modes=_parse_str_list(args.scaling_modes),
        search_workers=args.search_workers,
        estimator_n_jobs=args.estimator_n_jobs,
        seed=args.seed,
        run_id=args.run_id,
        save_dev_train_pca=args.save_dev_train_pca,
        expected_model_id=MODEL_ID_PCA_KNN,
        expected_weights="uniform",
    )
    print(
        f"[pca_knn.train] Completed training run '{summary['run_id']}'.\n"
        f"  Artifacts directory:  {summary['artifacts_dir']}\n"
        f"  Results directory:    {summary['results_dir']}\n"
        f"  Search results CSV:   {summary['search_results_csv']}\n"
        f"  Validation curve CSV: {summary['validation_curve_csv']}\n"
        f"  Selection config:     {summary['selection_config_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
