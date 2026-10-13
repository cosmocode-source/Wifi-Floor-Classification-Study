'Training, bounded deterministic randomized CV search, and pipeline serialization for Classical Gradient Boosting.'

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models.gradient_boosting.clean_data import (
    build_configured_gradient_boosting_pipeline,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_YAML,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    compute_encoding_config_sha256,
    extract_validated_wap_features_and_targets,
    load_gradient_boosting_config,
)
from models.pca_knn.train import _compute_spread_stats
from shared.artifacts import create_unique_run_dir, save_pipeline_artifact
from shared.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    EVALUATION_ROLE_CV_VAL,
    MODEL_ID_GRADIENT_BOOSTING,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    SPLIT_ROLE_OFFICIAL_VALIDATION,
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

STARTING_GB_PARAMS: dict[str, Any] = {
    "n_estimators": 189,
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_samples_leaf": 1,
    "subsample": 1.0,
    "max_features": None,
    "scale": False,
    "use_pca": False,
    "n_components": None,
    "n_iter_no_change": None,
}

# Alias for cross-module consistency with PCA-kNN fixed_baseline terminology
FIXED_BASELINE_PARAMS: dict[str, Any] = STARTING_GB_PARAMS

DECLARED_TIE_BREAKER_POLICY: tuple[str, ...] = (
    "1. Higher mean_cv_accuracy (tolerance 1e-9)",
    "2. Higher mean_cv_macro_f1 (tolerance 1e-9)",
    "3. Smaller max_depth (shallower trees / lower complexity)",
    "4. Smaller n_estimators (fewer boosting stages)",
    "5. Larger min_samples_leaf (stronger leaf regularization)",
    "6. Smaller learning_rate",
    "7. Lower candidate_index in deterministic sample order",
)

GRADIENT_BOOSTING_ASSUMPTIONS: tuple[str, ...] = (
    "Estimator is classical scikit-learn GradientBoostingClassifier on CPU (neither XGBoost nor HistGradientBoostingClassifier).",
    "Paper-inspired starting settings are max_depth=4 and n_estimators=189; learning_rate=0.1, min_samples_leaf=1, subsample=1.0, max_features=None, and no early stopping (n_iter_no_change=None) are documented implementation assumptions because the reference paper's boosting configuration is incompletely specified.",
    "Input features are raw sentinel-replaced WAP001..WAP520 (100 -> -110.0 dBm, genuine 0 dBm preserved as 0.0) without StandardScaler or PCA by default.",
    "Classical GradientBoostingClassifier requires finite numeric inputs and does not claim native NaN handling.",
    "Full-pipeline comparison: compares raw-feature classical gradient boosting against PCA-reduced kNN pipelines, not an isolated estimator-only comparison on identical PCA-transformed features.",
    "Floor classification is trained and evaluated separately within each known BUILDINGID (0, 1, 2) using identical development CV folds and selection rules as kNN.",
)


def resolve_gb_parallelism(
    search_workers: int = 1,
    blas_threads: int = 1,
) -> tuple[int, int, list[str]]:
    'Validate CPU worker and BLAS thread limits for classical GradientBoostingClassifier.'
    if (
        not isinstance(search_workers, (int, np.integer))
        or isinstance(search_workers, bool)
        or int(search_workers) < 1
    ):
        raise ValueError(
            f"search_workers must be a positive integer >= 1 (all-core -1 is prohibited); got {search_workers!r}."
        )
    if (
        not isinstance(blas_threads, (int, np.integer))
        or isinstance(blas_threads, bool)
        or int(blas_threads) < 1
    ):
        raise ValueError(
            f"blas_threads must be a positive integer >= 1 (all-core -1 is prohibited); got {blas_threads!r}."
        )

    eff_workers = int(search_workers)
    eff_threads = int(blas_threads)
    warnings: list[str] = []
    if eff_workers > 1 and eff_threads > 1:
        warnings.append(
            f"Parallelism safeguard: search_workers={eff_workers} > 1; forcing blas_threads=1 "
            f"(was {eff_threads}) to prevent nested CPU thread oversubscription."
        )
        eff_threads = 1

    os.environ["OMP_NUM_THREADS"] = str(eff_threads)
    os.environ["MKL_NUM_THREADS"] = str(eff_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(eff_threads)
    return eff_workers, eff_threads, warnings




def build_starting_gradient_boosting_pipeline(
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Construct the paper-inspired starting classical Gradient Boosting pipeline:
    `max_depth=4, n_estimators=189, learning_rate=0.1, min_samples_leaf=1, subsample=1.0`, unscaled, no PCA.
    """
    return build_configured_gradient_boosting_pipeline(
        n_estimators=int(STARTING_GB_PARAMS["n_estimators"]),
        learning_rate=float(STARTING_GB_PARAMS["learning_rate"]),
        max_depth=int(STARTING_GB_PARAMS["max_depth"]),
        min_samples_leaf=int(STARTING_GB_PARAMS["min_samples_leaf"]),
        subsample=float(STARTING_GB_PARAMS["subsample"]),
        max_features=STARTING_GB_PARAMS["max_features"],
        scale=False,
        use_pca=False,
        n_components=None,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
    )


def enumerate_full_candidate_space(
    n_estimators_list: Sequence[int] = (100, 189, 300),
    learning_rates: Sequence[float] = (0.05, 0.1),
    max_depths: Sequence[int] = (3, 4),
    min_samples_leaf_list: Sequence[int] = (1, 5),
    subsamples: Sequence[float] = (1.0,),
    max_features_list: Sequence[str | int | float | None] = (None,),
    scale: bool = False,
    use_pca: bool = False,
    n_components: int | None = None,
) -> list[dict[str, Any]]:
    'Enumerate the full candidate hyperparameter pool in deterministic complexity tie-breaking order.'
    ordered_depths = sorted(dict.fromkeys(int(d) for d in max_depths))
    ordered_n_est = sorted(dict.fromkeys(int(n) for n in n_estimators_list))
    ordered_leaf = sorted(dict.fromkeys(int(m) for m in min_samples_leaf_list), reverse=True)
    ordered_lr = sorted(dict.fromkeys(float(lr) for lr in learning_rates))
    ordered_sub = sorted(dict.fromkeys(float(s) for s in subsamples))
    ordered_mf = list(dict.fromkeys(max_features_list))

    full_pool: list[dict[str, Any]] = []
    pool_idx = 0
    for depth in ordered_depths:
        for n_est in ordered_n_est:
            for leaf in ordered_leaf:
                for lr in ordered_lr:
                    for sub in ordered_sub:
                        for mf in ordered_mf:
                            is_starting = (
                                n_est == int(STARTING_GB_PARAMS["n_estimators"])
                                and np.isclose(lr, float(STARTING_GB_PARAMS["learning_rate"]))
                                and depth == int(STARTING_GB_PARAMS["max_depth"])
                                and leaf == int(STARTING_GB_PARAMS["min_samples_leaf"])
                                and np.isclose(sub, float(STARTING_GB_PARAMS["subsample"]))
                                and mf == STARTING_GB_PARAMS["max_features"]
                                and not scale
                                and not use_pca
                            )
                            lr_tag = f"{lr:.3f}".rstrip("0").rstrip(".")
                            sub_tag = f"{sub:.2f}".rstrip("0").rstrip(".")
                            mf_tag = "all" if mf is None else str(mf)
                            cand_id = (
                                f"gb_est{n_est}_lr{lr_tag}_d{depth}_leaf{leaf}"
                                f"_sub{sub_tag}_mf{mf_tag}"
                            )
                            full_pool.append(
                                {
                                    "pool_index": pool_idx,
                                    "candidate_index": pool_idx,
                                    "candidate_id": cand_id,
                                    "n_estimators": int(n_est),
                                    "learning_rate": float(lr),
                                    "max_depth": int(depth),
                                    "min_samples_leaf": int(leaf),
                                    "subsample": float(sub),
                                    "max_features": mf,
                                    "scale": bool(scale),
                                    "use_pca": bool(use_pca),
                                    "n_components": int(n_components) if (use_pca and n_components is not None) else None,
                                    "n_iter_no_change": None,
                                    "is_starting_baseline": bool(is_starting),
                                }
                            )
                            pool_idx += 1
    return full_pool


def sample_gradient_boosting_candidates(
    n_estimators_list: Sequence[int] = (100, 189, 300),
    learning_rates: Sequence[float] = (0.05, 0.1),
    max_depths: Sequence[int] = (3, 4),
    min_samples_leaf_list: Sequence[int] = (1, 5),
    subsamples: Sequence[float] = (1.0,),
    max_features_list: Sequence[str | int | float | None] = (None,),
    n_trials: int | None = 16,
    search_strategy: str = "deterministic_random",
    seed: int = DEFAULT_RANDOM_SEED,
    include_starting_config_in_pool: bool = True,
    scale: bool = False,
    use_pca: bool = False,
    n_components: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    'Sample a bounded, deterministic list of Gradient Boosting candidates without replacement.'
    if search_strategy not in ("deterministic_random", "grid"):
        raise ValueError(
            f"search_strategy must be 'deterministic_random' or 'grid', got {search_strategy!r}."
        )

    full_pool = enumerate_full_candidate_space(
        n_estimators_list=n_estimators_list,
        learning_rates=learning_rates,
        max_depths=max_depths,
        min_samples_leaf_list=min_samples_leaf_list,
        subsamples=subsamples,
        max_features_list=max_features_list,
        scale=scale,
        use_pca=use_pca,
        n_components=n_components,
    )
    total_pool_size = len(full_pool)
    if total_pool_size == 0:
        raise ValueError("Candidate hyperparameter space is empty.")

    if search_strategy == "grid" or n_trials is None or int(n_trials) >= total_pool_size:
        sampled = [dict(c, candidate_index=i) for i, c in enumerate(full_pool)]
        effective_strategy = "grid" if search_strategy == "grid" else "deterministic_random_capped_at_pool_size"
    else:
        req_trials = int(n_trials)
        if req_trials < 1:
            raise ValueError(f"n_trials must be a positive integer >= 1, got {n_trials!r}.")
        rng = np.random.default_rng(int(seed))
        starting_indices = [
            i for i, c in enumerate(full_pool) if bool(c.get("is_starting_baseline", False))
        ]
        if include_starting_config_in_pool and starting_indices:
            anchor_idx = starting_indices[0]
            remaining_indices = [i for i in range(total_pool_size) if i != anchor_idx]
            if req_trials == 1:
                chosen_pool_indices = [anchor_idx]
            else:
                drawn = rng.choice(remaining_indices, size=req_trials - 1, replace=False).tolist()
                chosen_pool_indices = sorted([anchor_idx, *(int(x) for x in drawn)])
        else:
            drawn = rng.choice(total_pool_size, size=req_trials, replace=False).tolist()
            chosen_pool_indices = sorted(int(x) for x in drawn)

        sampled = [
            {**dict(full_pool[p_idx]), "candidate_index": sample_rank}
            for sample_rank, p_idx in enumerate(chosen_pool_indices)
        ]
        effective_strategy = "deterministic_random"

    sampling_metadata = {
        "search_strategy": effective_strategy,
        "requested_n_trials": int(n_trials) if n_trials is not None else total_pool_size,
        "sampled_candidate_count": int(len(sampled)),
        "total_cartesian_product_size": int(total_pool_size),
        "full_product_evaluated": bool(len(sampled) == total_pool_size),
        "random_seed": int(seed),
        "include_starting_config_in_pool": bool(include_starting_config_in_pool),
        "sampled_candidate_ids": [str(c["candidate_id"]) for c in sampled],
        "sampled_candidates": sampled,
    }
    return sampled, sampling_metadata


def select_best_gb_candidate(
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    'Select the winning feasible Gradient Boosting candidate using predeclared deterministic tie-breakers.'
    sorted_by_tie_breakers = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            int(c.get("max_depth", 999999)),
            int(c.get("n_estimators", 999999)),
            -int(c.get("min_samples_leaf", 0)),
            float(c.get("learning_rate", 999.0)),
            int(c.get("candidate_index", 999999)),
        ),
    )
    return select_best_candidate(
        sorted_by_tie_breakers,
        primary_key="mean_cv_accuracy",
        tie_breaker_key="mean_cv_macro_f1",
        tolerance=tolerance,
    )


def evaluate_single_gb_candidate_cv(
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
    candidate_role_label: str = "sampled_tuned_candidate",
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Evaluate a single Gradient Boosting candidate on one building across shared stratified CV folds.

    Never touches `holdout_indices_set`. Preserves original `FLOOR` labels without zero-indexing.
    """
    n_est = int(candidate["n_estimators"])
    lr = float(candidate["learning_rate"])
    depth = int(candidate["max_depth"])
    min_leaf = int(candidate["min_samples_leaf"])
    subsample = float(candidate.get("subsample", 1.0))
    max_features = candidate.get("max_features", None)
    scale_bool = bool(candidate.get("scale", False))
    use_pca_bool = bool(candidate.get("use_pca", False))
    n_comp = int(candidate["n_components"]) if (use_pca_bool and candidate.get("n_components") is not None) else None
    cand_id = str(candidate["candidate_id"])
    cand_idx = int(candidate.get("candidate_index", 0))
    n_splits = len(folds)

    y_dev_b = y_full.iloc[building_dev_train_indices].to_numpy(dtype=int)
    building_floors = sorted(int(f) for f in np.unique(y_dev_b))
    fold_train_sizes = [int(len(tr_idx)) for _, tr_idx, _ in folds]
    min_fold_rows = int(min(fold_train_sizes)) if fold_train_sizes else 0

    feas_cv: FeasibilityResult = check_candidate_feasibility(
        y_train=y_dev_b,
        n_features=int(X_full.shape[1]),
        n_splits=n_splits,
        n_components=n_comp if use_pca_bool else None,
        n_neighbors=None,
        fold_train_sizes=fold_train_sizes,
        strict=False,
    )

    # Also verify min_samples_leaf does not exceed half of the smallest fold training partition
    if feas_cv.feasible and (min_leaf < 1 or min_leaf * 2 > min_fold_rows):
        feas_cv = FeasibilityResult(
            feasible=False,
            reason_code="GB_MIN_SAMPLES_LEAF_INFEASIBLE",
            message=(
                f"min_samples_leaf={min_leaf} is too large for smallest fold training partition "
                f"({min_fold_rows} rows)."
            ),
            details={"min_samples_leaf": min_leaf, "min_fold_train_rows": min_fold_rows},
        )

    base_row: dict[str, Any] = {
        "model_id": MODEL_ID_GRADIENT_BOOSTING,
        "building_id": int(building_id),
        "candidate_index": cand_idx,
        "pool_index": int(candidate.get("pool_index", cand_idx)),
        "candidate_id": cand_id,
        "candidate_role": candidate_role_label,
        "is_starting_baseline": bool(candidate.get("is_starting_baseline", False)),
        "n_estimators": n_est,
        "learning_rate": lr,
        "max_depth": depth,
        "min_samples_leaf": min_leaf,
        "subsample": subsample,
        "max_features": max_features if max_features is not None else "all",
        "scale": scale_bool,
        "use_pca": use_pca_bool,
        "n_components": n_comp,
        "early_stopping_enabled": False,
        "n_iter_no_change": None,
        "cv_folds": int(n_splits),
        "min_fold_train_rows": min_fold_rows,
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

            pipeline = build_configured_gradient_boosting_pipeline(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                subsample=subsample,
                max_features=max_features,
                scale=scale_bool,
                use_pca=use_pca_bool,
                n_components=n_comp,
                sentinel_value=sentinel_value,
                replacement_value=replacement_value,
                enforce_wap_schema=True,
                random_state=int(seed),
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
                        "model_id": np.full(len(val_idx), MODEL_ID_GRADIENT_BOOSTING, dtype=object),
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


def train_gradient_boosting(
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    results_base_dir: Path | str = DEFAULT_RESULTS_DIR,
    processed_data_base_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
    profile: str | None = None,
    cv_folds: int | None = None,
    n_trials: int | None = None,
    search_strategy: str | None = None,
    n_estimators: Sequence[int] | None = None,
    learning_rates: Sequence[float] | None = None,
    max_depths: Sequence[int] | None = None,
    min_samples_leaf: Sequence[int] | None = None,
    search_workers: int | None = None,
    blas_threads: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute bounded CV search and starting-baseline evaluation, refit on dev_train, and save artifacts."""
    if "validationdata" in Path(train_csv_path).name.lower():
        raise PermissionError(
            "Forbidden operation: official validationData.csv must never be passed to train_gradient_boosting."
        )

    cfg = load_gradient_boosting_config(config_path)
    profiles_cfg: Mapping[str, Any] = cfg.get("search_profiles", {})
    active_profile_name = profile or str(cfg.get("default_profile", "full"))
    if active_profile_name not in profiles_cfg:
        raise ValueError(
            f"Unknown search profile {active_profile_name!r}; expected one of {sorted(profiles_cfg.keys())}."
        )
    prof = profiles_cfg[active_profile_name]
    tuned_space = cfg.get("tuned_space", {})

    resolved_cv_folds = int(cv_folds if cv_folds is not None else prof["cv_folds"])
    resolved_strategy = str(
        search_strategy if search_strategy is not None else prof.get("search_strategy", "deterministic_random")
    )
    resolved_n_trials = int(n_trials if n_trials is not None else prof.get("n_trials", 16))
    include_starting = bool(prof.get("include_starting_config_in_pool", True))

    resolved_n_estimators = [
        int(x) for x in (n_estimators if n_estimators is not None else prof.get("n_estimators", tuned_space["n_estimators"]))
    ]
    resolved_learning_rates = [
        float(x) for x in (learning_rates if learning_rates is not None else prof.get("learning_rate", tuned_space["learning_rate"]))
    ]
    resolved_max_depths = [
        int(x) for x in (max_depths if max_depths is not None else prof.get("max_depth", tuned_space["max_depth"]))
    ]
    resolved_min_samples_leaf = [
        int(x) for x in (min_samples_leaf if min_samples_leaf is not None else prof.get("min_samples_leaf", tuned_space["min_samples_leaf"]))
    ]

    par_cfg = cfg.get("parallelism", {})
    req_search_workers = int(
        search_workers if search_workers is not None else par_cfg.get("search_workers", 1)
    )
    req_blas_threads = int(
        blas_threads if blas_threads is not None else par_cfg.get("blas_threads", 1)
    )
    eff_search_workers, eff_blas_threads, par_warnings = resolve_gb_parallelism(
        search_workers=req_search_workers,
        blas_threads=req_blas_threads,
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

    resolved_run_id, run_artifacts_dir = create_unique_run_dir(
        base_dir=artifacts_base_dir,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        evaluation_role="train",
        run_id=run_id,
    )
    _, run_results_dir = create_unique_run_dir(
        base_dir=results_base_dir,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        evaluation_role="train",
        run_id=resolved_run_id,
    )

    fixed_artifacts_dir = run_artifacts_dir / "fixed_baseline"
    tuned_artifacts_dir = run_artifacts_dir / "tuned"
    fixed_artifacts_dir.mkdir(parents=True, exist_ok=False)
    tuned_artifacts_dir.mkdir(parents=True, exist_ok=False)

    sampled_candidates, sampling_metadata = sample_gradient_boosting_candidates(
        n_estimators_list=resolved_n_estimators,
        learning_rates=resolved_learning_rates,
        max_depths=resolved_max_depths,
        min_samples_leaf_list=resolved_min_samples_leaf,
        n_trials=resolved_n_trials,
        search_strategy=resolved_strategy,
        seed=seed,
        include_starting_config_in_pool=include_starting,
        scale=False,
        use_pca=False,
        n_components=None,
    )

    sampled_candidates_json_path = run_results_dir / "sampled_candidates.json"
    with sampled_candidates_json_path.open("w", encoding="utf-8") as fh:
        json.dump(sampling_metadata, fh, indent=2, sort_keys=False)

    buildings = sorted(int(b) for b in building_all.unique())
    holdout_mask_global = aligned_df["split_role"].astype(str) == SPLIT_ROLE_INTERNAL_HOLDOUT
    holdout_indices_set = set(int(i) for i in np.flatnonzero(holdout_mask_global.to_numpy()))

    all_search_rows: list[dict[str, Any]] = []
    tuned_selected_by_building: dict[str, dict[str, Any]] = {}
    fixed_baseline_by_building: dict[str, dict[str, Any]] = {}

    tuned_oof_frames: list[pd.DataFrame] = []
    fixed_oof_frames: list[pd.DataFrame] = []

    tuned_artifact_paths: dict[str, str] = {}
    fixed_artifact_paths: dict[str, str] = {}

    # Timing tracked strictly separately between CV candidate search and dev_train refit
    total_search_cv_fit_sec = 0.0
    total_search_cv_pred_sec = 0.0
    total_selected_tuned_cv_fit_sec = 0.0
    total_selected_tuned_cv_pred_sec = 0.0
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

        # 1. Evaluate sampled candidate pool for this building
        if eff_search_workers > 1 and len(sampled_candidates) > 1:
            eval_outputs = joblib.Parallel(n_jobs=eff_search_workers)(
                joblib.delayed(evaluate_single_gb_candidate_cv)(
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
                    candidate_role_label="tuned_grid",
                )
                for cand in sampled_candidates
            )
        else:
            eval_outputs = [
                evaluate_single_gb_candidate_cv(
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
                    candidate_role_label="tuned_grid",
                )
                for cand in sampled_candidates
            ]

        b_rows: list[dict[str, Any]] = []
        b_oof_by_cand_id: dict[str, pd.DataFrame] = {}
        b_search_fit_sec = 0.0
        b_search_pred_sec = 0.0
        for row_dict, oof_df in eval_outputs:
            b_rows.append(row_dict)
            b_search_fit_sec += float(row_dict.get("total_fit_seconds", 0.0))
            b_search_pred_sec += float(row_dict.get("total_predict_seconds", 0.0))
            if oof_df is not None:
                b_oof_by_cand_id[str(row_dict["candidate_id"])] = oof_df

        total_search_cv_fit_sec += b_search_fit_sec
        total_search_cv_pred_sec += b_search_pred_sec

        best_cand = select_best_gb_candidate(b_rows, tolerance=sel_tolerance)
        best_cand_id = str(best_cand["candidate_id"])
        for r in b_rows:
            if str(r["candidate_id"]) == best_cand_id:
                r["selected_for_building"] = True
            all_search_rows.append(r)

        tuned_oof_frames.append(b_oof_by_cand_id[best_cand_id])
        total_selected_tuned_cv_fit_sec += float(best_cand["total_fit_seconds"])
        total_selected_tuned_cv_pred_sec += float(best_cand["total_predict_seconds"])

        # Refit winning tuned pipeline strictly on dev_train rows for building b_id
        X_dev_b = X_all.iloc[b_dev_indices]
        y_dev_b = y_all.iloc[b_dev_indices].to_numpy(dtype=int)
        b_classes = sorted(int(c) for c in np.unique(y_dev_b))

        tuned_pipeline = build_configured_gradient_boosting_pipeline(
            n_estimators=int(best_cand["n_estimators"]),
            learning_rate=float(best_cand["learning_rate"]),
            max_depth=int(best_cand["max_depth"]),
            min_samples_leaf=int(best_cand["min_samples_leaf"]),
            subsample=float(best_cand["subsample"]),
            max_features=None if best_cand.get("max_features") in (None, "all") else best_cand["max_features"],
            scale=False,
            use_pca=False,
            n_components=None,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            enforce_wap_schema=True,
            random_state=seed,
        )
        t_refit0 = time.perf_counter()
        tuned_pipeline.fit(X_dev_b, y_dev_b)
        refit_elapsed = time.perf_counter() - t_refit0
        total_tuned_refit_sec += refit_elapsed

        # Verify original floor labels are preserved on the fitted estimator
        fitted_classes = [int(c) for c in tuned_pipeline.named_steps["classifier"].classes_.tolist()]
        assert fitted_classes == b_classes, (
            f"Classifier classes_ {fitted_classes} do not match original building floor labels {b_classes}."
        )

        tuned_prep_desc = describe_preprocessing_config(
            scale=False,
            use_pca=False,
            n_components=None,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        tuned_params_record = {
            "variant": "tuned",
            "candidate_id": best_cand_id,
            "n_estimators": int(best_cand["n_estimators"]),
            "learning_rate": float(best_cand["learning_rate"]),
            "max_depth": int(best_cand["max_depth"]),
            "min_samples_leaf": int(best_cand["min_samples_leaf"]),
            "subsample": float(best_cand["subsample"]),
            "max_features": None if best_cand.get("max_features") in (None, "all") else best_cand["max_features"],
            "scale": False,
            "use_pca": False,
            "n_components": None,
            "early_stopping_enabled": False,
            "n_iter_no_change": None,
            "is_starting_baseline": bool(best_cand.get("is_starting_baseline", False)),
            "mean_cv_accuracy": float(best_cand["mean_cv_accuracy"]),
            "std_cv_accuracy": float(best_cand["std_cv_accuracy"]),
            "mean_cv_error": float(best_cand["mean_cv_error"]),
            "mean_cv_macro_f1": float(best_cand["mean_cv_macro_f1"]),
            "std_cv_macro_f1": float(best_cand["std_cv_macro_f1"]),
            "cv_folds": int(resolved_cv_folds),
            "building_search_cv_fit_seconds": float(b_search_fit_sec),
            "building_search_cv_predict_seconds": float(b_search_pred_sec),
            "selected_candidate_cv_fit_seconds": float(best_cand["total_fit_seconds"]),
            "dev_train_refit_seconds": float(refit_elapsed),
            "dev_train_rows": int(len(b_dev_indices)),
            "class_labels": b_classes,
        }
        tuned_selected_by_building[b_key] = tuned_params_record

        tuned_meta = save_pipeline_artifact(
            pipeline=tuned_pipeline,
            output_dir=tuned_artifacts_dir,
            building_id=b_id,
            model_id=MODEL_ID_GRADIENT_BOOSTING,
            feature_names=WAP_COLUMNS,
            class_labels=b_classes,
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            preprocessing_config=tuned_prep_desc,
            selected_parameters=tuned_params_record,
            run_id=resolved_run_id,
        )
        tuned_artifact_paths[b_key] = str(tuned_meta["artifact_path"])

        # 2. Evaluate and refit paper-inspired starting configuration (fixed_baseline) separately
        starting_cand_spec = {
            "pool_index": -1,
            "candidate_index": -1,
            "candidate_id": "starting_baseline_gb_est189_lr0.1_d4_leaf1_sub1_mfall",
            "n_estimators": int(STARTING_GB_PARAMS["n_estimators"]),
            "learning_rate": float(STARTING_GB_PARAMS["learning_rate"]),
            "max_depth": int(STARTING_GB_PARAMS["max_depth"]),
            "min_samples_leaf": int(STARTING_GB_PARAMS["min_samples_leaf"]),
            "subsample": float(STARTING_GB_PARAMS["subsample"]),
            "max_features": STARTING_GB_PARAMS["max_features"],
            "scale": False,
            "use_pca": False,
            "n_components": None,
            "n_iter_no_change": None,
            "is_starting_baseline": True,
        }

        # Reuse CV evaluation if the exact starting config was already evaluated in b_rows
        existing_starting_rows = [
            r for r in b_rows if bool(r.get("is_starting_baseline", False))
        ]
        if existing_starting_rows:
            fixed_row = dict(existing_starting_rows[0])
            fixed_oof_df = b_oof_by_cand_id.get(str(fixed_row["candidate_id"]))
        else:
            fixed_row, fixed_oof_df = evaluate_single_gb_candidate_cv(
                candidate=starting_cand_spec,
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
                candidate_role_label="fixed_baseline",
            )
            all_search_rows.append(fixed_row)

        if fixed_row["status"] == "completed" and fixed_oof_df is not None:
            fixed_oof_frames.append(fixed_oof_df)
            total_fixed_cv_fit_sec += float(fixed_row["total_fit_seconds"])
            total_fixed_cv_pred_sec += float(fixed_row["total_predict_seconds"])

            fixed_pipeline = build_starting_gradient_boosting_pipeline(
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
                enforce_wap_schema=True,
                random_state=seed,
            )
            tf0 = time.perf_counter()
            fixed_pipeline.fit(X_dev_b, y_dev_b)
            fixed_refit_sec = time.perf_counter() - tf0
            total_fixed_refit_sec += fixed_refit_sec

            fixed_prep_desc = describe_preprocessing_config(
                scale=False,
                use_pca=False,
                n_components=None,
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
            )
            fixed_params_record = {
                "variant": "fixed_baseline",
                "candidate_id": str(fixed_row["candidate_id"]),
                "n_estimators": int(STARTING_GB_PARAMS["n_estimators"]),
                "learning_rate": float(STARTING_GB_PARAMS["learning_rate"]),
                "max_depth": int(STARTING_GB_PARAMS["max_depth"]),
                "min_samples_leaf": int(STARTING_GB_PARAMS["min_samples_leaf"]),
                "subsample": float(STARTING_GB_PARAMS["subsample"]),
                "max_features": STARTING_GB_PARAMS["max_features"],
                "scale": False,
                "use_pca": False,
                "n_components": None,
                "early_stopping_enabled": False,
                "n_iter_no_change": None,
                "paper_inspired_fields": ["max_depth=4", "n_estimators=189"],
                "implementation_assumption_fields": [
                    "learning_rate=0.1",
                    "min_samples_leaf=1",
                    "subsample=1.0",
                    "max_features=None",
                    "n_iter_no_change=None",
                ],
                "mean_cv_accuracy": float(fixed_row["mean_cv_accuracy"]),
                "mean_cv_macro_f1": float(fixed_row["mean_cv_macro_f1"]),
                "cv_folds": int(resolved_cv_folds),
                "cv_fit_seconds": float(fixed_row["total_fit_seconds"]),
                "dev_train_refit_seconds": float(fixed_refit_sec),
                "dev_train_rows": int(len(b_dev_indices)),
                "class_labels": b_classes,
                "status": "completed",
            }
            fixed_baseline_by_building[b_key] = fixed_params_record

            fixed_meta = save_pipeline_artifact(
                pipeline=fixed_pipeline,
                output_dir=fixed_artifacts_dir,
                building_id=b_id,
                model_id=MODEL_ID_GRADIENT_BOOSTING,
                feature_names=WAP_COLUMNS,
                class_labels=b_classes,
                dataset_hash=expected_sha_for_align,
                split_manifest_hash=split_manifest_sha256,
                preprocessing_config=fixed_prep_desc,
                selected_parameters=fixed_params_record,
                run_id=resolved_run_id,
            )
            fixed_artifact_paths[b_key] = str(fixed_meta["artifact_path"])
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
                **STARTING_GB_PARAMS,
                "status": str(fixed_row["status"]),
                "reason_code": fixed_row["reason_code"],
                "message": fixed_row["message"],
            }

    # Save search_results.csv containing every sampled/evaluated/skipped/failed candidate
    search_results_df = pd.DataFrame(all_search_rows)
    search_results_csv_path = run_results_dir / "search_results.csv"
    search_results_df.to_csv(search_results_csv_path, index=False)

    # Save out-of-fold CV predictions and shared-schema metrics.json for tuned models
    tuned_oof_all = pd.concat(tuned_oof_frames, axis=0, ignore_index=True)
    tuned_oof_csv_path = run_results_dir / "cv_tuned_oof_predictions.csv"
    save_predictions_csv(tuned_oof_all, tuned_oof_csv_path, evaluation_role=EVALUATION_ROLE_CV_VAL)

    t_per_bldg, t_pooled, t_supports, t_cms = evaluate_predictions_by_building(tuned_oof_all)
    for b_key, b_sel in tuned_selected_by_building.items():
        if b_key in t_per_bldg:
            t_per_bldg[b_key]["mean_fold_cv_accuracy"] = b_sel["mean_cv_accuracy"]
            t_per_bldg[b_key]["mean_fold_cv_macro_f1"] = b_sel["mean_cv_macro_f1"]

    tuned_prep_summary = describe_preprocessing_config(
        scale=False,
        use_pca=False,
        n_components=None,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
    )
    tuned_prep_summary["encoding_config_sha256"] = encoding_sha256
    tuned_prep_summary["pipeline_comparison_scope"] = (
        "raw_sentinel_replaced_wap_features_into_classical_gradient_boosting_vs_pca_knn"
    )

    cv_tuned_result = build_completed_result(
        run_id=f"{resolved_run_id}_cv_tuned",
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        dataset_hash=expected_sha_for_align,
        split_manifest_hash=split_manifest_sha256,
        evaluation_role=EVALUATION_ROLE_CV_VAL,
        preprocessing=tuned_prep_summary,
        selected_parameters={
            "variant": "tuned",
            "profile": active_profile_name,
            "search_strategy": sampling_metadata["search_strategy"],
            "sampled_candidate_count": sampling_metadata["sampled_candidate_count"],
            "total_cartesian_product_size": sampling_metadata["total_cartesian_product_size"],
            "cv_folds": resolved_cv_folds,
            "per_building": tuned_selected_by_building,
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
            "cv_search_fit_seconds": float(total_search_cv_fit_sec),
            "cv_search_predict_seconds": float(total_search_cv_pred_sec),
            "cv_search_total_seconds": float(total_search_cv_fit_sec + total_search_cv_pred_sec),
            "selected_candidate_cv_fit_seconds": float(total_selected_tuned_cv_fit_sec),
            "selected_candidate_cv_predict_seconds": float(total_selected_tuned_cv_pred_sec),
            "dev_train_refit_seconds": float(total_tuned_refit_sec),
            "fit_seconds": float(total_tuned_refit_sec),
            "predict_seconds": float(total_selected_tuned_cv_pred_sec),
            "total_seconds": float(
                total_search_cv_fit_sec + total_search_cv_pred_sec + total_tuned_refit_sec
            ),
        },
        prediction_file_path=tuned_oof_csv_path,
        model_artifact_paths=tuned_artifact_paths,
        assumptions=GRADIENT_BOOSTING_ASSUMPTIONS,
        warnings=run_warnings,
        extra_metadata={
            "variant": "tuned",
            "search_profile": active_profile_name,
            "search_results_csv": str(search_results_csv_path),
            "sampled_candidates_json": str(sampled_candidates_json_path),
        },
    )
    cv_tuned_metrics_path = run_results_dir / "cv_tuned_metrics.json"
    save_result_json(cv_tuned_result, cv_tuned_metrics_path)

    # Save out-of-fold CV predictions and shared-schema metrics.json for starting fixed_baseline
    cv_fixed_metrics_path = run_results_dir / "cv_fixed_baseline_metrics.json"
    fixed_oof_csv_path: Path | None = None
    if fixed_baseline_all_feasible and len(fixed_oof_frames) == len(buildings):
        fixed_oof_all = pd.concat(fixed_oof_frames, axis=0, ignore_index=True)
        fixed_oof_csv_path = run_results_dir / "cv_fixed_baseline_oof_predictions.csv"
        save_predictions_csv(fixed_oof_all, fixed_oof_csv_path, evaluation_role=EVALUATION_ROLE_CV_VAL)
        f_per_bldg, f_pooled, f_supports, f_cms = evaluate_predictions_by_building(fixed_oof_all)
        fixed_prep_summary = describe_preprocessing_config(
            scale=False,
            use_pca=False,
            n_components=None,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        fixed_prep_summary["encoding_config_sha256"] = encoding_sha256
        cv_fixed_result = build_completed_result(
            run_id=f"{resolved_run_id}_cv_fixed_baseline",
            model_id=MODEL_ID_GRADIENT_BOOSTING,
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            evaluation_role=EVALUATION_ROLE_CV_VAL,
            preprocessing=fixed_prep_summary,
            selected_parameters={
                "variant": "fixed_baseline",
                "starting_baseline_spec": dict(STARTING_GB_PARAMS),
                "cv_folds": resolved_cv_folds,
                "per_building": fixed_baseline_by_building,
            },
            selection_metric="paper_inspired_starting_gb_no_tuning",
            per_building=f_per_bldg,
            pooled=f_pooled,
            supports=f_supports,
            confusion_matrices=f_cms,
            timing={
                "cv_fit_seconds": float(total_fixed_cv_fit_sec),
                "cv_predict_seconds": float(total_fixed_cv_pred_sec),
                "dev_train_refit_seconds": float(total_fixed_refit_sec),
                "fit_seconds": float(total_fixed_refit_sec),
                "predict_seconds": float(total_fixed_cv_pred_sec),
                "total_seconds": float(
                    total_fixed_cv_fit_sec + total_fixed_cv_pred_sec + total_fixed_refit_sec
                ),
            },
            prediction_file_path=fixed_oof_csv_path,
            model_artifact_paths=fixed_artifact_paths,
            assumptions=GRADIENT_BOOSTING_ASSUMPTIONS,
            warnings=run_warnings,
            extra_metadata={
                "variant": "fixed_baseline",
                "search_profile": active_profile_name,
                "search_results_csv": str(search_results_csv_path),
            },
        )
        save_result_json(cv_fixed_result, cv_fixed_metrics_path)
    else:
        cv_fixed_result = build_non_completed_result(
            run_id=f"{resolved_run_id}_cv_fixed_baseline",
            model_id=MODEL_ID_GRADIENT_BOOSTING,
            status="skipped",
            evaluation_role=EVALUATION_ROLE_CV_VAL,
            reason="Starting gradient boosting baseline was not feasible across all buildings.",
            dataset_hash=expected_sha_for_align,
            split_manifest_hash=split_manifest_sha256,
            preprocessing=describe_preprocessing_config(
                scale=False,
                use_pca=False,
                n_components=None,
                sentinel_value=sentinel_val,
                replacement_value=replacement_val,
            ),
            selected_parameters={
                "variant": "fixed_baseline",
                "starting_baseline_spec": dict(STARTING_GB_PARAMS),
                "per_building": fixed_baseline_by_building,
            },
            selection_metric="paper_inspired_starting_gb_no_tuning",
            skip_reasons=fixed_baseline_skip_reasons,
            assumptions=GRADIENT_BOOSTING_ASSUMPTIONS,
            warnings=run_warnings,
        )
        save_result_json(cv_fixed_result, cv_fixed_metrics_path)

    selection_config: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": resolved_run_id,
        "model_id": MODEL_ID_GRADIENT_BOOSTING,
        "estimator_class": "sklearn.ensemble.GradientBoostingClassifier",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_split_role": SPLIT_ROLE_DEV_TRAIN,
        "internal_holdout_used_in_selection": False,
        "official_validation_used_in_selection": False,
        "early_stopping_enabled": False,
        "dataset_hash": expected_sha_for_align,
        "split_manifest_hash": split_manifest_sha256,
        "encoding_config_sha256": encoding_sha256,
        "search_profile": active_profile_name,
        "cv_folds": resolved_cv_folds,
        "sampling_metadata": sampling_metadata,
        "selection_policy": {
            "primary_metric": "mean_cv_accuracy",
            "tie_breaker_metric": "mean_cv_macro_f1",
            "tolerance": sel_tolerance,
            "deterministic_tie_breakers": list(DECLARED_TIE_BREAKER_POLICY),
        },
        "timing_summary": {
            "cv_search_fit_seconds": float(total_search_cv_fit_sec),
            "cv_search_predict_seconds": float(total_search_cv_pred_sec),
            "cv_search_total_seconds": float(total_search_cv_fit_sec + total_search_cv_pred_sec),
            "tuned_dev_train_refit_seconds": float(total_tuned_refit_sec),
            "fixed_baseline_cv_fit_seconds": float(total_fixed_cv_fit_sec),
            "fixed_baseline_dev_train_refit_seconds": float(total_fixed_refit_sec),
        },
        "parallelism": {
            "requested_search_workers": req_search_workers,
            "effective_search_workers": eff_search_workers,
            "requested_blas_threads": req_blas_threads,
            "effective_blas_threads": eff_blas_threads,
            "gpu_used": False,
            "warnings": par_warnings,
        },
        "variants": {
            "tuned": {
                "status": "completed",
                "per_building": tuned_selected_by_building,
                "artifact_paths": tuned_artifact_paths,
                "cv_metrics_json": str(cv_tuned_metrics_path.resolve()),
                "cv_oof_predictions_csv": str(tuned_oof_csv_path.resolve()),
            },
            "fixed_baseline": {
                "status": "completed" if fixed_baseline_all_feasible else "skipped",
                "starting_baseline_spec": dict(STARTING_GB_PARAMS),
                "per_building": fixed_baseline_by_building,
                "artifact_paths": fixed_artifact_paths,
                "cv_metrics_json": str(cv_fixed_metrics_path.resolve()),
                "cv_oof_predictions_csv": (
                    str(fixed_oof_csv_path.resolve()) if fixed_oof_csv_path is not None else None
                ),
            },
        },
        "search_results_csv": str(search_results_csv_path.resolve()),
        "sampled_candidates_json": str(sampled_candidates_json_path.resolve()),
        "environment": collect_environment_info(),
    }

    selection_config_results_path = run_results_dir / "selection_config.json"
    selection_config_artifacts_path = run_artifacts_dir / "selection_config.json"
    for sc_path in (selection_config_results_path, selection_config_artifacts_path):
        with sc_path.open("w", encoding="utf-8") as fh:
            json.dump(selection_config, fh, indent=2, sort_keys=False)

    return {
        "run_id": resolved_run_id,
        "model_id": MODEL_ID_GRADIENT_BOOSTING,
        "artifacts_dir": str(run_artifacts_dir.resolve()),
        "results_dir": str(run_results_dir.resolve()),
        "search_results_csv": str(search_results_csv_path.resolve()),
        "sampled_candidates_json": str(sampled_candidates_json_path.resolve()),
        "selection_config_json": str(selection_config_results_path.resolve()),
        "cv_tuned_metrics_json": str(cv_tuned_metrics_path.resolve()),
        "cv_fixed_baseline_metrics_json": str(cv_fixed_metrics_path.resolve()),
        "selection_config": selection_config,
    }


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected comma-separated integers, got empty string.")
    return [int(p) for p in parts]


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected comma-separated floats, got empty string.")
    return [float(p) for p in parts]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m models.gradient_boosting.train`."""
    parser = argparse.ArgumentParser(
        prog="python -m models.gradient_boosting.train",
        description=(
            "Run bounded deterministic randomized CV search for classical GradientBoostingClassifier "
            "on shared development splits, evaluate the paper-inspired starting baseline (max_depth=4, "
            "n_estimators=189) separately, and refit per-building pipelines on dev_train only."
        ),
    )
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
        default=DEFAULT_CONFIG_YAML,
        help=f"Path to model config.yaml (default: {DEFAULT_CONFIG_YAML}).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=("smoke", "lightweight", "full"),
        default=None,
        help="Staged search profile from config.yaml: 'smoke' (2 trials), 'lightweight' (8 trials), or 'full' (16 trials).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override CV fold count (e.g. 3 or 5 matching shared manifest columns fold_3 / fold_5).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of deterministic randomized search trials (e.g. 8 or 16).",
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=("deterministic_random", "grid"),
        default=None,
        help="Search strategy (default: 'deterministic_random').",
    )
    parser.add_argument(
        "--n-estimators",
        type=str,
        default=None,
        help="Optional comma-separated n_estimators candidates, e.g. '100,189,300'.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default=None,
        help="Optional comma-separated learning_rate candidates, e.g. '0.05,0.1'.",
    )
    parser.add_argument(
        "--max-depths",
        type=str,
        default=None,
        help="Optional comma-separated max_depth candidates, e.g. '3,4'.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=str,
        default=None,
        help="Optional comma-separated min_samples_leaf candidates, e.g. '1,5'.",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=None,
        help="Number of parallel candidate workers >= 1 (default: 1; all-core -1 is prohibited).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help="Number of BLAS/OpenMP threads per worker >= 1 (default: 1; forced to 1 when --search-workers > 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic candidate sampling and tree building (default: 42).",
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
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Base directory for saved model artifacts (default: {DEFAULT_ARTIFACTS_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Base directory for CV results and search_results.csv (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help=f"Base directory for processed data (default: {DEFAULT_PROCESSED_DATA_DIR}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m models.gradient_boosting.train`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = train_gradient_boosting(
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        config_path=args.config,
        artifacts_base_dir=args.artifacts_dir,
        results_base_dir=args.results_dir,
        processed_data_base_dir=args.processed_data_dir,
        profile=args.profile,
        cv_folds=args.cv_folds,
        n_trials=args.n_trials,
        search_strategy=args.search_strategy,
        n_estimators=_parse_int_list(args.n_estimators),
        learning_rates=_parse_float_list(args.learning_rates),
        max_depths=_parse_int_list(args.max_depths),
        min_samples_leaf=_parse_int_list(args.min_samples_leaf),
        search_workers=args.search_workers,
        blas_threads=args.blas_threads,
        seed=args.seed,
        run_id=args.run_id,
    )
    print(
        f"[gradient_boosting.train] Completed training run '{summary['run_id']}'.\n"
        f"  Artifacts directory:     {summary['artifacts_dir']}\n"
        f"  Results directory:       {summary['results_dir']}\n"
        f"  Sampled candidates JSON: {summary['sampled_candidates_json']}\n"
        f"  Search results CSV:      {summary['search_results_csv']}\n"
        f"  Selection config:        {summary['selection_config_json']}"
    )
    return 0


__all__ = [
    "DECLARED_TIE_BREAKER_POLICY",
    "FIXED_BASELINE_PARAMS",
    "GRADIENT_BOOSTING_ASSUMPTIONS",
    "STARTING_GB_PARAMS",
    "build_arg_parser",
    "build_configured_gradient_boosting_pipeline",
    "build_starting_gradient_boosting_pipeline",
    "enumerate_full_candidate_space",
    "evaluate_single_gb_candidate_cv",
    "main",
    "resolve_gb_parallelism",
    "sample_gradient_boosting_candidates",
    "select_best_gb_candidate",
    "train_gradient_boosting",
]


if __name__ == "__main__":
    raise SystemExit(main())
