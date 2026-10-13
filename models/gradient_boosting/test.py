'Inference-only holdout evaluation and guarded official validation for Classical Gradient Boosting.'

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from models.gradient_boosting.clean_data import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_YAML,
    DEFAULT_RESULTS_DIR,
    compute_encoding_config_sha256,
    extract_validated_wap_features_and_targets,
    load_gradient_boosting_config,
)
from models.gradient_boosting.train import (
    GRADIENT_BOOSTING_ASSUMPTIONS,
    build_configured_gradient_boosting_pipeline,
    build_starting_gradient_boosting_pipeline,
)
from models.pca_knn.test import (
    load_frozen_selection_config as _shared_load_frozen_selection_config,
    resolve_training_run_dir as _shared_resolve_training_run_dir,
)
from shared.artifacts import (
    UntrustedArtifactError,
    create_unique_run_dir,
    load_trusted_pipeline_artifact,
    save_pipeline_artifact,
)
from shared.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    DEFAULT_VALIDATION_CSV,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
    EXPECTED_VALIDATION_ROWS,
    MODEL_ID_GRADIENT_BOOSTING,
    PROJECT_ROOT,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    SPLIT_ROLE_OFFICIAL_VALIDATION,
    UNDETECTED_SENTINEL,
    WAP_COLUMNS,
)
from shared.create_splits import (
    SplitCreationError,
    align_dataset_with_manifest,
    load_split_manifest,
)
from shared.data_validation import (
    assign_stable_row_ids,
    compute_file_sha256,
    load_and_audit_csv,
    verify_training_csv_provenance,
)
from shared.metrics import evaluate_predictions_by_building
from shared.preprocessing import describe_preprocessing_config
from shared.result_schema import (
    build_completed_result,
    build_non_completed_result,
    save_predictions_csv,
    save_result_json,
)


def resolve_training_run_dir(
    train_run_dir: Path | str | None = None,
    train_run_id: str | None = None,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    """Locate a completed `gradient_boosting` training artifact directory containing `selection_config.json`."""
    return _shared_resolve_training_run_dir(
        train_run_dir=train_run_dir,
        train_run_id=train_run_id,
        artifacts_base_dir=artifacts_base_dir,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
    )


def load_frozen_selection_config(selection_config_path: Path | str) -> dict[str, Any]:
    """Load and validate a frozen `selection_config.json` produced by a `gradient_boosting` training run."""
    cfg = _shared_load_frozen_selection_config(
        selection_config_path,
        expected_model_id=MODEL_ID_GRADIENT_BOOSTING,
    )
    if cfg.get("early_stopping_enabled", False) is not False:
        raise ValueError(
            "Invalid gradient_boosting selection_config.json: early_stopping_enabled must be False."
        )
    return cfg


def _evaluate_single_gb_variant_pipelines(
    variant_name: str,
    variant_block: Mapping[str, Any],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    building_eval: pd.Series,
    row_ids_eval: pd.Series,
    evaluation_role: str,
    eval_run_id: str,
    run_results_dir: Path,
    dataset_hash: str,
    split_manifest_hash: str,
    expected_train_dataset_hash: str,
    expected_split_manifest_hash: str,
    sentinel_value: int | float,
    replacement_value: float,
    allowed_root: Path,
    refit_all_training: bool = False,
    X_all_train: pd.DataFrame | None = None,
    y_all_train: pd.Series | None = None,
    building_all_train: pd.Series | None = None,
    final_refit_artifacts_dir: Path | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Evaluate either `tuned` or `fixed_baseline` Gradient Boosting pipelines across buildings and save schema v1.0 outputs."""
    variant_dir = run_results_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    top_metrics_path = run_results_dir / f"{evaluation_role}_{variant_name}_metrics.json"
    top_preds_path = run_results_dir / f"{evaluation_role}_{variant_name}_predictions.csv"
    sub_metrics_path = variant_dir / "metrics.json"
    sub_preds_path = variant_dir / "predictions.csv"

    if variant_block.get("status") != "completed":
        reason = (
            f"Variant '{variant_name}' was not completed during training "
            f"(status={variant_block.get('status')!r})."
        )
        non_comp = build_non_completed_result(
            run_id=f"{eval_run_id}_{variant_name}",
            model_id=MODEL_ID_GRADIENT_BOOSTING,
            status="skipped",
            evaluation_role=evaluation_role,
            reason=reason,
            dataset_hash=dataset_hash,
            split_manifest_hash=split_manifest_hash,
            selected_parameters={"variant": variant_name, **dict(variant_block)},
            assumptions=GRADIENT_BOOSTING_ASSUMPTIONS,
        )
        save_result_json(non_comp, top_metrics_path)
        save_result_json(non_comp, sub_metrics_path)
        return {
            "variant": variant_name,
            "status": "skipped",
            "metrics_json": str(top_metrics_path.resolve()),
            "predictions_csv": None,
        }

    per_building_cfg: Mapping[str, Any] = variant_block.get("per_building", {})
    saved_artifact_paths: Mapping[str, str] = variant_block.get("artifact_paths", {})
    buildings = sorted(int(b) for b in building_eval.unique())

    pred_frames: list[pd.DataFrame] = []
    used_artifact_paths: dict[str, str] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0

    for b_id in buildings:
        b_key = str(b_id)
        if b_key not in per_building_cfg:
            raise ValueError(f"Missing building {b_id} configuration in variant '{variant_name}'.")
        b_params = per_building_cfg[b_key]

        b_eval_mask = building_eval.to_numpy() == b_id
        X_b_eval = X_eval.loc[b_eval_mask]
        y_b_eval = y_eval.loc[b_eval_mask].to_numpy(dtype=int)
        rids_b_eval = row_ids_eval.loc[b_eval_mask].astype(str).to_numpy()

        if refit_all_training:
            if evaluation_role != EVALUATION_ROLE_OFFICIAL_VALIDATION:
                raise PermissionError(
                    "Explicit refitting on all official training rows is only permitted in official_validation mode."
                )
            if (
                X_all_train is None
                or y_all_train is None
                or building_all_train is None
                or final_refit_artifacts_dir is None
            ):
                raise ValueError("Missing full training data or artifact directory for explicit final refit.")

            b_tr_mask = building_all_train.to_numpy() == b_id
            X_b_tr = X_all_train.loc[b_tr_mask]
            y_b_tr = y_all_train.loc[b_tr_mask].to_numpy(dtype=int)
            b_classes = sorted(int(c) for c in np.unique(y_b_tr))

            if variant_name == "fixed_baseline":
                pipeline = build_starting_gradient_boosting_pipeline(
                    sentinel_value=sentinel_value,
                    replacement_value=replacement_value,
                    enforce_wap_schema=True,
                    random_state=seed,
                )
            else:
                pipeline = build_configured_gradient_boosting_pipeline(
                    n_estimators=int(b_params["n_estimators"]),
                    learning_rate=float(b_params["learning_rate"]),
                    max_depth=int(b_params["max_depth"]),
                    min_samples_leaf=int(b_params["min_samples_leaf"]),
                    subsample=float(b_params.get("subsample", 1.0)),
                    max_features=None if b_params.get("max_features") in (None, "all") else b_params["max_features"],
                    scale=bool(b_params.get("scale", False)),
                    use_pca=bool(b_params.get("use_pca", False)),
                    n_components=b_params.get("n_components"),
                    sentinel_value=sentinel_value,
                    replacement_value=replacement_value,
                    enforce_wap_schema=True,
                    random_state=seed,
                )

            t0 = time.perf_counter()
            pipeline.fit(X_b_tr, y_b_tr)
            fit_sec = time.perf_counter() - t0
            total_fit_seconds += fit_sec

            prep_desc = describe_preprocessing_config(
                scale=bool(b_params.get("scale", False)),
                use_pca=bool(b_params.get("use_pca", False)),
                n_components=b_params.get("n_components"),
                sentinel_value=sentinel_value,
                replacement_value=replacement_value,
            )
            prep_desc["fit_scope"] = "all_official_training_rows_explicit_final_refit"

            refit_variant_dir = final_refit_artifacts_dir / variant_name
            refit_meta = save_pipeline_artifact(
                pipeline=pipeline,
                output_dir=refit_variant_dir,
                building_id=b_id,
                model_id=MODEL_ID_GRADIENT_BOOSTING,
                feature_names=WAP_COLUMNS,
                class_labels=b_classes,
                dataset_hash=expected_train_dataset_hash,
                split_manifest_hash=expected_split_manifest_hash,
                preprocessing_config=prep_desc,
                selected_parameters={
                    **dict(b_params),
                    "explicit_final_refit_all_training_rows": True,
                    "final_refit_train_rows": int(len(X_b_tr)),
                    "final_refit_seconds": float(fit_sec),
                },
                run_id=eval_run_id,
            )
            used_artifact_paths[b_key] = str(refit_meta["artifact_path"])
        else:
            if b_key not in saved_artifact_paths:
                raise FileNotFoundError(
                    f"Missing saved pipeline artifact path for building {b_id} in variant '{variant_name}'."
                )
            art_path = Path(saved_artifact_paths[b_key])
            pipeline, meta = load_trusted_pipeline_artifact(
                artifact_path=art_path,
                allowed_root=allowed_root,
            )
            if int(meta.get("building_id", -1)) != b_id:
                raise UntrustedArtifactError(
                    f"Artifact building_id mismatch: expected {b_id}, got {meta.get('building_id')}."
                )
            if meta.get("model_id") != MODEL_ID_GRADIENT_BOOSTING:
                raise UntrustedArtifactError(
                    f"Artifact model_id mismatch: expected '{MODEL_ID_GRADIENT_BOOSTING}', "
                    f"got {meta.get('model_id')!r}."
                )
            if meta.get("dataset_hash") != expected_train_dataset_hash:
                raise UntrustedArtifactError(
                    f"Artifact dataset_hash ({meta.get('dataset_hash')}) does not match training dataset "
                    f"hash ({expected_train_dataset_hash})."
                )
            if meta.get("split_manifest_hash") != expected_split_manifest_hash:
                raise UntrustedArtifactError(
                    f"Artifact split_manifest_hash ({meta.get('split_manifest_hash')}) does not match "
                    f"split manifest hash ({expected_split_manifest_hash})."
                )
            clf_step = pipeline.named_steps.get("classifier")
            if clf_step is None or type(clf_step) is not GradientBoostingClassifier:
                raise UntrustedArtifactError(
                    f"Loaded pipeline classifier must be exact sklearn.ensemble.GradientBoostingClassifier, "
                    f"got {type(clf_step)!r}."
                )
            if getattr(clf_step, "n_iter_no_change", None) is not None:
                raise UntrustedArtifactError(
                    "Loaded GradientBoostingClassifier has early stopping enabled (n_iter_no_change != None)."
                )
            used_artifact_paths[b_key] = str(art_path.resolve())

        # Inference-only transform + predict (never refits on evaluation rows)
        t_pred0 = time.perf_counter()
        y_pred_b = pipeline.predict(X_b_eval)
        pred_sec = time.perf_counter() - t_pred0
        total_predict_seconds += pred_sec

        pred_frames.append(
            pd.DataFrame(
                {
                    "row_id": rids_b_eval,
                    "building_id": np.full(len(X_b_eval), int(b_id), dtype=int),
                    "true_floor": y_b_eval.astype(int),
                    "predicted_floor": np.asarray(y_pred_b, dtype=int),
                    "evaluation_role": np.full(len(X_b_eval), str(evaluation_role), dtype=object),
                    "model_id": np.full(len(X_b_eval), MODEL_ID_GRADIENT_BOOSTING, dtype=object),
                }
            )
        )

    preds_df = pd.concat(pred_frames, axis=0, ignore_index=True)
    save_predictions_csv(preds_df, top_preds_path, evaluation_role=evaluation_role)
    save_predictions_csv(preds_df, sub_preds_path, evaluation_role=evaluation_role)

    per_bldg_metrics, pooled_metrics, supports_dict, cms_dict = evaluate_predictions_by_building(preds_df)

    prep_summary = describe_preprocessing_config(
        scale=False,
        use_pca=False,
        n_components=None,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
    )
    prep_summary["encoding_config_sha256"] = compute_encoding_config_sha256(
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
    )
    prep_summary["fit_scope"] = (
        "all_official_training_rows_explicit_final_refit"
        if refit_all_training
        else "dev_train_saved_pipeline_no_refit"
    )
    prep_summary["pipeline_comparison_scope"] = (
        "raw_sentinel_replaced_wap_features_into_classical_gradient_boosting_vs_pca_knn"
    )

    selection_metric_desc: str | dict[str, Any] = (
        "paper_inspired_starting_gb_no_tuning"
        if variant_name == "fixed_baseline"
        else {
            "primary_metric": "mean_cv_accuracy",
            "tie_breaker_metric": "mean_cv_macro_f1",
            "selected_on_split_role": SPLIT_ROLE_DEV_TRAIN,
        }
    )

    result_payload = build_completed_result(
        run_id=f"{eval_run_id}_{variant_name}",
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        dataset_hash=dataset_hash,
        split_manifest_hash=split_manifest_hash,
        evaluation_role=evaluation_role,
        preprocessing=prep_summary,
        selected_parameters={
            "variant": variant_name,
            "estimator_class": "sklearn.ensemble.GradientBoostingClassifier",
            "explicit_final_refit_all_training_rows": bool(refit_all_training),
            "per_building": dict(per_building_cfg),
        },
        selection_metric=selection_metric_desc,
        per_building=per_bldg_metrics,
        pooled=pooled_metrics,
        supports=supports_dict,
        confusion_matrices=cms_dict,
        timing={
            "fit_seconds": float(total_fit_seconds),
            "predict_seconds": float(total_predict_seconds),
            "total_seconds": float(total_fit_seconds + total_predict_seconds),
        },
        prediction_file_path=top_preds_path,
        model_artifact_paths=used_artifact_paths,
        assumptions=GRADIENT_BOOSTING_ASSUMPTIONS,
        extra_metadata={
            "variant": variant_name,
            "explicit_final_refit_all_training_rows": bool(refit_all_training),
            "canonical_variant_metrics_json": str(sub_metrics_path.resolve()),
            "canonical_variant_predictions_csv": str(sub_preds_path.resolve()),
        },
    )

    save_result_json(result_payload, top_metrics_path)
    save_result_json(result_payload, sub_metrics_path)

    return {
        "variant": variant_name,
        "status": "completed",
        "evaluation_role": evaluation_role,
        "pooled_accuracy": float(pooled_metrics["accuracy"]),
        "pooled_macro_f1": float(pooled_metrics["macro_f1"]),
        "metrics_json": str(top_metrics_path.resolve()),
        "predictions_csv": str(top_preds_path.resolve()),
        "variant_subdir_metrics_json": str(sub_metrics_path.resolve()),
        "variant_subdir_predictions_csv": str(sub_preds_path.resolve()),
        "model_artifact_paths": used_artifact_paths,
    }


def evaluate_gradient_boosting(
    eval_mode: str = EVALUATION_ROLE_INTERNAL_HOLDOUT,
    variant: str = "both",
    train_run_dir: Path | str | None = None,
    train_run_id: str | None = None,
    selection_config_path: Path | str | None = None,
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    val_csv_path: Path | str | None = None,
    confirm_final_evaluation: bool = False,
    refit_all_training: bool = False,
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    results_base_dir: Path | str = DEFAULT_RESULTS_DIR,
    allowed_root: Path | str = PROJECT_ROOT,
    seed: int = DEFAULT_RANDOM_SEED,
    eval_run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate trusted saved Gradient Boosting pipelines on `internal_holdout` (default) or `official_validation`."""
    if eval_mode not in (EVALUATION_ROLE_INTERNAL_HOLDOUT, EVALUATION_ROLE_OFFICIAL_VALIDATION):
        raise ValueError(
            f"eval_mode must be '{EVALUATION_ROLE_INTERNAL_HOLDOUT}' or "
            f"'{EVALUATION_ROLE_OFFICIAL_VALIDATION}', got {eval_mode!r}."
        )
    if variant not in ("both", "tuned", "fixed_baseline"):
        raise ValueError(f"variant must be 'both', 'tuned', or 'fixed_baseline', got {variant!r}.")

    # Strict development vs official-holdout separation guards
    if eval_mode == EVALUATION_ROLE_INTERNAL_HOLDOUT:
        if val_csv_path is not None:
            raise PermissionError(
                "Development mode (internal_holdout) cannot accept --val-csv / official validation data."
            )
        if confirm_final_evaluation:
            raise PermissionError(
                "--confirm-final-evaluation is only valid with --eval-mode official_validation."
            )
        if refit_all_training:
            raise PermissionError(
                "--refit-all-training is only valid with --eval-mode official_validation --confirm-final-evaluation."
            )
        if "validationdata" in Path(train_csv_path).name.lower():
            raise PermissionError(
                "Forbidden operation: official validationData.csv cannot be passed as --train-csv in internal_holdout mode."
            )
    else:
        if not confirm_final_evaluation:
            raise PermissionError(
                "Official validation holdout evaluation requires explicit `--confirm-final-evaluation`."
            )

    cfg = load_gradient_boosting_config(config_path)
    prep_cfg = cfg.get("preprocessing", {})
    sentinel_val = prep_cfg.get("sentinel_value", UNDETECTED_SENTINEL)
    replacement_val = float(prep_cfg.get("replacement_value", DEFAULT_UNDETECTED_REPLACEMENT))

    resolved_train_dir = resolve_training_run_dir(
        train_run_dir=train_run_dir,
        train_run_id=train_run_id,
        artifacts_base_dir=artifacts_base_dir,
    )
    resolved_sel_cfg_path = (
        Path(selection_config_path)
        if selection_config_path is not None
        else (resolved_train_dir / "selection_config.json")
    )
    selection_config = load_frozen_selection_config(resolved_sel_cfg_path)
    expected_train_sha = str(selection_config["dataset_hash"])
    expected_manifest_sha = str(selection_config["split_manifest_hash"])

    manifest_df, actual_manifest_sha, _ = load_split_manifest(split_manifest_path, strict=True)
    if actual_manifest_sha != expected_manifest_sha:
        raise SplitCreationError(
            f"Active split manifest hash ({actual_manifest_sha}) does not match frozen selection_config "
            f"split_manifest_hash ({expected_manifest_sha})."
        )

    train_path = Path(train_csv_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found at: {train_path}")
    train_df_raw = pd.read_csv(train_path)
    verify_training_csv_provenance(train_path, train_df_raw, expected_train_sha)
    if SPLIT_ROLE_OFFICIAL_VALIDATION in set(
        train_df_raw.get("split_role", pd.Series(dtype=str)).astype(str).unique()
    ):
        raise PermissionError(
            "Forbidden operation: training DataFrame contains official_validation split_role rows."
        )

    aligned_train_df = align_dataset_with_manifest(
        train_df=train_df_raw,
        manifest_df=manifest_df,
        expected_source_sha256=expected_train_sha,
    )
    X_train_all, y_train_all, bldg_train_all, rids_train_all = extract_validated_wap_features_and_targets(
        aligned_train_df,
        require_row_id=True,
        allow_already_cleaned_replacement=True,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
    )
    assert rids_train_all is not None

    if eval_mode == EVALUATION_ROLE_INTERNAL_HOLDOUT:
        hold_mask = aligned_train_df["split_role"].astype(str) == SPLIT_ROLE_INTERNAL_HOLDOUT
        if not hold_mask.any():
            raise SplitCreationError("No internal_holdout rows found in aligned dataset.")
        X_eval = X_train_all.loc[hold_mask].reset_index(drop=True)
        y_eval = y_train_all.loc[hold_mask].reset_index(drop=True)
        bldg_eval = bldg_train_all.loc[hold_mask].reset_index(drop=True)
        rids_eval = rids_train_all.loc[hold_mask].reset_index(drop=True)
        eval_dataset_hash = expected_train_sha
    else:
        resolved_val_path = Path(val_csv_path) if val_csv_path is not None else DEFAULT_VALIDATION_CSV
        if not resolved_val_path.is_file():
            raise FileNotFoundError(f"Official validation CSV not found at: {resolved_val_path}")
        val_df_peek = pd.read_csv(resolved_val_path)
        roles_in_val = set(val_df_peek.get("split_role", pd.Series(dtype=str)).astype(str).unique())
        if roles_in_val & {SPLIT_ROLE_DEV_TRAIN, SPLIT_ROLE_INTERNAL_HOLDOUT}:
            raise PermissionError(
                f"Official validation CSV at '{resolved_val_path}' contains development split roles: {sorted(roles_in_val)}."
            )

        val_file_sha = compute_file_sha256(resolved_val_path)
        if "source_sha256" in val_df_peek.columns:
            val_df = val_df_peek.copy()
            if "row_id" not in val_df.columns:
                val_df = assign_stable_row_ids(
                    val_df, dataset_role="val", file_sha256=val_file_sha, inplace=False
                )
            eval_dataset_hash = str(val_df["source_sha256"].iloc[0])
        else:
            val_df, val_audit = load_and_audit_csv(
                resolved_val_path,
                dataset_role="val",
                expected_row_count=EXPECTED_VALIDATION_ROWS,
                strict=True,
                attach_row_ids=True,
            )
            eval_dataset_hash = val_audit.file_sha256

        X_eval, y_eval, bldg_eval, rids_eval = extract_validated_wap_features_and_targets(
            val_df,
            require_row_id=True,
            allow_already_cleaned_replacement=True,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
        )
        assert rids_eval is not None

    resolved_eval_run_id, run_results_dir = create_unique_run_dir(
        base_dir=results_base_dir,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
        evaluation_role=eval_mode,
        run_id=eval_run_id,
    )

    final_refit_dir: Path | None = None
    if refit_all_training:
        _, final_refit_dir = create_unique_run_dir(
            base_dir=artifacts_base_dir,
            model_id=MODEL_ID_GRADIENT_BOOSTING,
            evaluation_role="final_refit_all_train",
            run_id=f"{resolved_eval_run_id}_final_refit_all_train",
        )

    variants_to_run = ["tuned", "fixed_baseline"] if variant == "both" else [variant]
    variant_outputs: dict[str, dict[str, Any]] = {}

    for var_name in variants_to_run:
        var_block = selection_config["variants"].get(var_name)
        if var_block is None:
            raise ValueError(f"Variant '{var_name}' not found in frozen selection_config.json.")
        variant_outputs[var_name] = _evaluate_single_gb_variant_pipelines(
            variant_name=var_name,
            variant_block=var_block,
            X_eval=X_eval,
            y_eval=y_eval,
            building_eval=bldg_eval,
            row_ids_eval=rids_eval,
            evaluation_role=eval_mode,
            eval_run_id=resolved_eval_run_id,
            run_results_dir=run_results_dir,
            dataset_hash=eval_dataset_hash,
            split_manifest_hash=actual_manifest_sha,
            expected_train_dataset_hash=expected_train_sha,
            expected_split_manifest_hash=expected_manifest_sha,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            allowed_root=Path(allowed_root),
            refit_all_training=refit_all_training,
            X_all_train=X_train_all if refit_all_training else None,
            y_all_train=y_train_all if refit_all_training else None,
            building_all_train=bldg_train_all if refit_all_training else None,
            final_refit_artifacts_dir=final_refit_dir,
            seed=seed,
        )

    summary = {
        "eval_run_id": resolved_eval_run_id,
        "train_run_id": str(selection_config.get("run_id", "")),
        "model_id": MODEL_ID_GRADIENT_BOOSTING,
        "estimator_class": "sklearn.ensemble.GradientBoostingClassifier",
        "evaluation_role": eval_mode,
        "confirm_final_evaluation": bool(confirm_final_evaluation),
        "explicit_final_refit_all_training_rows": bool(refit_all_training),
        "final_refit_artifacts_dir": str(final_refit_dir.resolve()) if final_refit_dir is not None else None,
        "results_dir": str(run_results_dir.resolve()),
        "selection_config_path": str(resolved_sel_cfg_path.resolve()),
        "variants": variant_outputs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = run_results_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m models.gradient_boosting.test`."""
    parser = argparse.ArgumentParser(
        prog="python -m models.gradient_boosting.test",
        description=(
            "Evaluate trusted saved classical GradientBoostingClassifier pipelines on the shared "
            "internal development holdout (default) or on the official validation holdout (requires "
            "frozen selection_config.json and explicit --confirm-final-evaluation)."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=(EVALUATION_ROLE_INTERNAL_HOLDOUT, EVALUATION_ROLE_OFFICIAL_VALIDATION),
        default=EVALUATION_ROLE_INTERNAL_HOLDOUT,
        help=f"Evaluation split role (default: '{EVALUATION_ROLE_INTERNAL_HOLDOUT}').",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=("both", "tuned", "fixed_baseline"),
        default="both",
        help="Which saved model variant(s) to evaluate: 'both', 'tuned', or 'fixed_baseline' (default: 'both').",
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        default=None,
        help=f"Path to specific training run artifact directory in {DEFAULT_ARTIFACTS_DIR}/<run_id>.",
    )
    parser.add_argument(
        "--train-run-id",
        type=str,
        default=None,
        help="Training run_id inside --artifacts-dir (defaults to most recent completed run).",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=None,
        help="Optional explicit path to frozen selection_config.json.",
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
        "--val-csv",
        type=Path,
        default=None,
        help="Path to official validationData.csv (ONLY permitted with --eval-mode official_validation --confirm-final-evaluation).",
    )
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Explicit confirmation required to unlock --eval-mode official_validation.",
    )
    parser.add_argument(
        "--refit-all-training",
        action="store_true",
        help="In official_validation mode, explicitly refit frozen configurations on 100%% of trainingData.csv and record artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_YAML,
        help=f"Path to model config.yaml (default: {DEFAULT_CONFIG_YAML}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Base directory containing saved model artifact run directories.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory to write evaluation result directories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used only if --refit-all-training is requested (default: 42).",
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="Optional custom evaluation run_id.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m models.gradient_boosting.test`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = evaluate_gradient_boosting(
        eval_mode=args.eval_mode,
        variant=args.variant,
        train_run_dir=args.train_run_dir,
        train_run_id=args.train_run_id,
        selection_config_path=args.selection_config,
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        val_csv_path=args.val_csv,
        confirm_final_evaluation=args.confirm_final_evaluation,
        refit_all_training=args.refit_all_training,
        config_path=args.config,
        artifacts_base_dir=args.artifacts_dir,
        results_base_dir=args.results_dir,
        seed=args.seed,
        eval_run_id=args.eval_run_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "build_arg_parser",
    "evaluate_gradient_boosting",
    "load_frozen_selection_config",
    "main",
    "resolve_training_run_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())
