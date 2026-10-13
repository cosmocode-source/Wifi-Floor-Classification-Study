'Offline live-demo inference entrypoint for known-building WiFi floor classification.'

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from shared.artifacts import UntrustedArtifactError, load_trusted_pipeline_artifact
from shared.config import (
    BUILDING_COLUMN,
    DEFAULT_UNDETECTED_REPLACEMENT,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    EXPECTED_BUILDINGS,
    FORBIDDEN_PREDICTOR_COLUMNS,
    MODELS_DIR,
    PROJECT_ROOT,
    ROW_ID_COLUMN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    SPLIT_ROLE_OFFICIAL_VALIDATION,
    TARGET_COLUMN,
    UNDETECTED_SENTINEL,
    VALID_MODEL_IDS,
    WAP_COLUMNS,
)
from shared.create_splits import align_dataset_with_manifest, load_split_manifest
from shared.data_validation import (
    DataValidationError,
    assign_stable_row_ids,
    validate_predictor_columns,
    validate_wap_numeric_and_ranges,
)


class DemoInferenceError(RuntimeError):
    """Raised when demo inputs, model artifacts, or schemas fail validation."""


def _resolve_building_artifacts(
    *,
    artifact_path: Path | str | None = None,
    selection_config_path: Path | str | None = None,
    train_run_dir: Path | str | None = None,
    model_id: str | None = None,
    train_run_id: str | None = None,
    variant: str = "tuned",
    building_id: int | None = None,
) -> dict[int, Path]:
    """Resolve a mapping `{building_id: artifact_joblib_path}` from explicit CLI/API arguments.

    Never guesses an arbitrary 'latest' directory; requires explicit artifact, selection config,
    or `(model_id, train_run_id)` coordinates.
    """
    if variant not in ("tuned", "fixed_baseline"):
        raise DemoInferenceError(
            f"Invalid variant={variant!r}; expected 'tuned' or 'fixed_baseline'."
        )

    if artifact_path is not None:
        resolved_art = Path(artifact_path)
        if not resolved_art.is_file():
            raise FileNotFoundError(
                f"Specified demo pipeline artifact not found: {resolved_art}"
            )
        if building_id is not None:
            return {int(building_id): resolved_art}
        # Inspect companion metadata sidecar to determine building_id
        stem = resolved_art.name.removesuffix(".joblib")
        meta_path = resolved_art.with_name(f"{stem}.metadata.json")
        if not meta_path.is_file():
            raise UntrustedArtifactError(
                f"Companion metadata sidecar not found for artifact: {meta_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "building_id" not in meta:
            raise DemoInferenceError(
                f"Metadata sidecar '{meta_path}' does not specify 'building_id'."
            )
        return {int(meta["building_id"]): resolved_art}

    resolved_sel_path: Path | None = None
    if selection_config_path is not None:
        resolved_sel_path = Path(selection_config_path)
    elif train_run_dir is not None:
        resolved_sel_path = Path(train_run_dir) / "selection_config.json"
    elif model_id is not None and train_run_id is not None:
        if model_id not in VALID_MODEL_IDS:
            raise DemoInferenceError(
                f"Invalid model_id={model_id!r}; expected one of {VALID_MODEL_IDS}."
            )
        resolved_sel_path = (
            MODELS_DIR / str(model_id) / "artifacts" / str(train_run_id) / "selection_config.json"
        )
    else:
        raise DemoInferenceError(
            "Demo requires an explicit model source: pass `--artifact-path`, "
            "`--selection-config`, `--train-run-dir`, or both `--model-id` and `--train-run-id`."
        )

    if not resolved_sel_path.is_file():
        raise FileNotFoundError(
            f"Selection configuration file not found at '{resolved_sel_path}'. "
            "Ensure the model has been trained before running the live demo."
        )

    sel_cfg = json.loads(resolved_sel_path.read_text(encoding="utf-8"))
    if model_id is not None and sel_cfg.get("model_id") != model_id:
        raise DemoInferenceError(
            f"selection_config.json model_id ({sel_cfg.get('model_id')!r}) does not match "
            f"requested --model-id ({model_id!r})."
        )

    variants_map = sel_cfg.get("variants", {})
    if variant not in variants_map:
        raise DemoInferenceError(
            f"Variant '{variant}' not found in '{resolved_sel_path}' "
            f"(available: {sorted(variants_map.keys())})."
        )
    v_block = variants_map[variant]
    if v_block.get("status") != "completed":
        raise DemoInferenceError(
            f"Variant '{variant}' in '{resolved_sel_path}' is not completed "
            f"(status={v_block.get('status')!r})."
        )

    art_paths_raw: Mapping[str, str] = v_block.get("artifact_paths", {})
    if not art_paths_raw:
        raise DemoInferenceError(
            f"Variant '{variant}' in '{resolved_sel_path}' contains no 'artifact_paths'."
        )

    result: dict[int, Path] = {}
    for b_key, p_str in art_paths_raw.items():
        b_int = int(b_key)
        if building_id is not None and b_int != int(building_id):
            continue
        p = Path(p_str)
        if not p.is_file():
            # Fallback relative to selection_config directory if moved within repo
            alt = resolved_sel_path.parent / variant / p.name
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(
                    f"Saved pipeline artifact for building {b_int} not found at '{p}'."
                )
        result[b_int] = p

    if not result:
        raise DemoInferenceError(
            f"No saved pipeline artifacts matched building_id={building_id} in '{resolved_sel_path}'."
        )
    return result


def prepare_demo_input_dataframe(
    input_df: pd.DataFrame,
    *,
    building_id_override: int | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    """Validate and split a demo DataFrame into `(X_wap, building_ids, row_ids, y_true_optional)`.

    Guarantees that `X_wap` contains strictly `WAP001`..`WAP520` and never includes `BUILDINGID`,
    `FLOOR`, coordinates, or identifier columns.
    """
    if input_df.empty:
        raise DataValidationError("Demo input DataFrame is empty (0 rows).")

    missing_waps = [c for c in WAP_COLUMNS if c not in input_df.columns]
    if missing_waps:
        raise DataValidationError(
            f"Demo input is missing {len(missing_waps)} required WAP columns "
            f"(first 5 missing: {missing_waps[:5]})."
        )

    df_work = input_df.copy()
    if ROW_ID_COLUMN not in df_work.columns:
        df_work = assign_stable_row_ids(df_work, dataset_role="demo")
    row_ids = df_work[ROW_ID_COLUMN].astype(str).reset_index(drop=True)

    if building_id_override is not None:
        if BUILDING_COLUMN in df_work.columns:
            mismatch_mask = df_work[BUILDING_COLUMN].astype(int) != int(building_id_override)
            if mismatch_mask.any():
                raise DataValidationError(
                    f"Input rows contain BUILDINGID values {sorted(df_work[BUILDING_COLUMN].unique().tolist())} "
                    f"that conflict with --building-id={building_id_override}."
                )
        building_series = pd.Series(
            np.full(len(df_work), int(building_id_override), dtype=int),
            name=BUILDING_COLUMN,
        )
    else:
        if BUILDING_COLUMN not in df_work.columns:
            raise DataValidationError(
                f"Demo input must include column '{BUILDING_COLUMN}' or specify `--building-id`."
            )
        if df_work[BUILDING_COLUMN].isna().any():
            raise DataValidationError(f"Demo input contains null values in '{BUILDING_COLUMN}'.")
        building_series = df_work[BUILDING_COLUMN].astype(int).reset_index(drop=True)

    invalid_buildings = set(building_series.unique().tolist()) - set(EXPECTED_BUILDINGS)
    if invalid_buildings:
        raise DataValidationError(
            f"Unknown BUILDINGID values {sorted(invalid_buildings)}; "
            f"known buildings are {list(EXPECTED_BUILDINGS)}."
        )

    # Optional ground truth kept strictly separate from predictors
    y_true_optional: pd.Series | None = None
    if TARGET_COLUMN in df_work.columns and not df_work[TARGET_COLUMN].isna().all():
        if df_work[TARGET_COLUMN].isna().any():
            raise DataValidationError(
                f"Column '{TARGET_COLUMN}' contains partial null values in demo input."
            )
        y_true_optional = df_work[TARGET_COLUMN].astype(int).reset_index(drop=True)

    X_wap = df_work.loc[:, list(WAP_COLUMNS)].reset_index(drop=True)
    validate_predictor_columns(list(X_wap.columns))
    assert set(X_wap.columns).isdisjoint(FORBIDDEN_PREDICTOR_COLUMNS)

    # Validate numeric values (accepting either raw 100 sentinel or already-cleaned -110.0)
    if X_wap.isna().any().any():
        raise DataValidationError("Demo WAP feature matrix contains NaN/null values.")
    try:
        X_num = X_wap.astype(float)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Non-numeric values found in WAP columns: {exc}") from exc

    arr = X_num.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise DataValidationError("Demo WAP feature matrix contains non-finite (inf) values.")

    if float(replacement_value) != float(sentinel_value):
        X_for_range = X_num.replace(float(replacement_value), float(sentinel_value))
    else:
        X_for_range = X_num
    validate_wap_numeric_and_ranges(X_for_range, strict=True)

    return X_wap, building_series, row_ids, y_true_optional


def load_dev_rehearsal_examples(
    *,
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    samples_per_building: int = 2,
    building_id: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select deterministic rehearsal rows from `internal_holdout` without touching `validationData.csv`."""
    if "validationdata" in Path(train_csv_path).name.lower():
        raise PermissionError(
            "Rehearsal loader refuses to read official validationData.csv by default."
        )
    train_path = Path(train_csv_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found for demo rehearsal: {train_path}")

    manifest_df, _, _ = load_split_manifest(split_manifest_path, strict=True)
    expected_sha = str(manifest_df["source_sha256"].iloc[0])
    raw_df = pd.read_csv(train_path)
    if SPLIT_ROLE_OFFICIAL_VALIDATION in set(
        raw_df.get("split_role", pd.Series(dtype=str)).astype(str).unique()
    ):
        raise PermissionError("Training CSV contains official_validation rows.")

    aligned = align_dataset_with_manifest(
        train_df=raw_df,
        manifest_df=manifest_df,
        expected_source_sha256=expected_sha,
    )
    holdout_df = aligned[aligned["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT].copy()
    if building_id is not None:
        holdout_df = holdout_df[holdout_df[BUILDING_COLUMN].astype(int) == int(building_id)].copy()
    if holdout_df.empty:
        raise DemoInferenceError("No internal_holdout rows matched the requested building filter.")

    rng = np.random.default_rng(seed)
    chosen_Parts: list[pd.DataFrame] = []
    for b_id in sorted(int(b) for b in holdout_df[BUILDING_COLUMN].unique()):
        b_rows = holdout_df[holdout_df[BUILDING_COLUMN].astype(int) == b_id].sort_values(
            by=ROW_ID_COLUMN
        )
        n_pick = min(int(samples_per_building), len(b_rows))
        if n_pick <= 0:
            continue
        indices = sorted(rng.choice(len(b_rows), size=n_pick, replace=False).tolist())
        chosen_Parts.append(b_rows.iloc[indices])

    if not chosen_Parts:
        raise DemoInferenceError("Could not sample any rehearsal examples.")
    return pd.concat(chosen_Parts, axis=0, ignore_index=True)


def run_offline_demo_inference(
    *,
    input_df: pd.DataFrame | None = None,
    input_csv_path: Path | str | None = None,
    use_dev_rehearsal: bool = False,
    samples_per_building: int = 2,
    artifact_path: Path | str | None = None,
    selection_config_path: Path | str | None = None,
    train_run_dir: Path | str | None = None,
    model_id: str | None = None,
    train_run_id: str | None = None,
    variant: str = "tuned",
    building_id: int | None = None,
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    allowed_root: Path | str = PROJECT_ROOT,
    output_json_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run offline floor prediction using saved trusted pipeline artifacts without training."""
    if input_df is not None:
        raw_df = input_df.copy()
        input_source_desc = "in_memory_dataframe"
    elif input_csv_path is not None:
        csv_p = Path(input_csv_path)
        if not csv_p.is_file():
            raise FileNotFoundError(f"Demo input CSV not found: {csv_p}")
        raw_df = pd.read_csv(csv_p)
        input_source_desc = str(csv_p.resolve())
    elif use_dev_rehearsal:
        raw_df = load_dev_rehearsal_examples(
            train_csv_path=train_csv_path,
            split_manifest_path=split_manifest_path,
            samples_per_building=samples_per_building,
            building_id=building_id,
            seed=seed,
        )
        input_source_desc = f"dev_rehearsal_internal_holdout({Path(train_csv_path).name})"
    else:
        raise DemoInferenceError(
            "Specify `--input-csv <path>` or `--use-dev-rehearsal` (or pass `input_df` via API)."
        )

    X_wap, building_series, row_ids, y_true_opt = prepare_demo_input_dataframe(
        raw_df,
        building_id_override=building_id,
    )

    building_artifacts = _resolve_building_artifacts(
        artifact_path=artifact_path,
        selection_config_path=selection_config_path,
        train_run_dir=train_run_dir,
        model_id=model_id,
        train_run_id=train_run_id,
        variant=variant,
        building_id=building_id,
    )

    buildings_needed = sorted(int(b) for b in building_series.unique())
    missing_b = [b for b in buildings_needed if b not in building_artifacts]
    if missing_b:
        raise DemoInferenceError(
            f"Input contains rows for building(s) {missing_b}, but artifacts were only resolved "
            f"for building(s) {sorted(building_artifacts.keys())}."
        )

    predictions_records: list[dict[str, Any]] = []
    loaded_metadata_by_building: dict[str, Any] = {}
    total_predict_sec = 0.0
    resolved_model_id: str | None = model_id

    for b_id in buildings_needed:
        art_file = building_artifacts[b_id]
        pipeline, meta = load_trusted_pipeline_artifact(
            artifact_path=art_file,
            allowed_root=allowed_root,
        )
        if int(meta.get("building_id", -1)) != int(b_id):
            raise UntrustedArtifactError(
                f"Artifact building_id ({meta.get('building_id')}) does not match "
                f"routed BUILDINGID ({b_id})."
            )
        if resolved_model_id is None:
            resolved_model_id = str(meta.get("model_id", "unknown"))
        elif str(meta.get("model_id")) != resolved_model_id:
            raise UntrustedArtifactError(
                f"Inconsistent model_id across building artifacts: {resolved_model_id!r} vs {meta.get('model_id')!r}."
            )

        loaded_metadata_by_building[str(b_id)] = {
            "artifact_path": str(Path(art_file).resolve()),
            "artifact_sha256": meta.get("artifact_sha256"),
            "model_id": meta.get("model_id"),
            "building_id": int(b_id),
            "class_labels": meta.get("class_labels", []),
            "selected_parameters": meta.get("selected_parameters", {}),
        }

        mask = (building_series.to_numpy() == b_id)
        X_b = X_wap.loc[mask]
        rids_b = row_ids.loc[mask].tolist()
        idxs_b = np.flatnonzero(mask).tolist()

        # Strictly inference-only: never calls fit()
        t0 = time.perf_counter()
        y_pred_b = np.asarray(pipeline.predict(X_b), dtype=int)
        elapsed = time.perf_counter() - t0
        total_predict_sec += elapsed

        # Validate that predicted labels belong to the building's known original FLOOR classes
        known_classes = [int(c) for c in meta.get("class_labels", [])]
        if known_classes:
            unexpected = set(y_pred_b.tolist()) - set(known_classes)
            if unexpected:
                raise DemoInferenceError(
                    f"Pipeline for building {b_id} predicted floor labels {sorted(unexpected)} "
                    f"outside known class_labels {known_classes}."
                )

        # Optional confidence score if predict_proba is available
        confidences: list[float | None] = [None] * len(X_b)
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = np.asarray(pipeline.predict_proba(X_b), dtype=float)
                if proba.ndim == 2 and proba.shape[0] == len(X_b):
                    confidences = [float(np.max(row_p)) for row_p in proba]
            except Exception:
                confidences = [None] * len(X_b)

        # Count detected WAPs (value != 100 and value != -110.0) as a helpful diagnostic
        X_b_arr = X_b.to_numpy(dtype=float)
        detected_counts = np.sum(
            (X_b_arr != float(UNDETECTED_SENTINEL))
            & (X_b_arr != float(DEFAULT_UNDETECTED_REPLACEMENT)),
            axis=1,
        )

        for local_i, global_idx in enumerate(idxs_b):
            rec: dict[str, Any] = {
                "input_order": int(global_idx),
                "row_id": str(rids_b[local_i]),
                "building_id": int(b_id),
                "predicted_floor": int(y_pred_b[local_i]),
                "detected_wap_count": int(detected_counts[local_i]),
                "max_class_probability": confidences[local_i],
                "model_id": str(resolved_model_id),
                "variant": str(variant),
            }
            if y_true_opt is not None:
                true_f = int(y_true_opt.iloc[global_idx])
                rec["true_floor"] = true_f
                rec["correct"] = bool(true_f == int(y_pred_b[local_i]))
            else:
                rec["true_floor"] = None
                rec["correct"] = None
            predictions_records.append(rec)

    predictions_records.sort(key=lambda r: int(r["input_order"]))
    out_df = pd.DataFrame(predictions_records).drop(columns=["input_order"])

    summary: dict[str, Any] = {
        "status": "completed",
        "mode": "offline_inference_only_no_training",
        "model_id": resolved_model_id,
        "variant": variant,
        "input_source": input_source_desc,
        "evaluated_row_count": int(len(out_df)),
        "predictor_column_count": len(WAP_COLUMNS),
        "ground_truth_included_in_predictors": False,
        "ground_truth_available_for_display": bool(y_true_opt is not None),
        "predict_seconds": float(total_predict_sec),
        "mean_latency_ms_per_sample": float((total_predict_sec * 1000.0) / max(1, len(out_df))),
        "loaded_artifacts": loaded_metadata_by_building,
        "predictions": out_df.to_dict(orient="records"),
    }

    if y_true_opt is not None:
        correct_cnt = int(sum(1 for r in summary["predictions"] if r["correct"] is True))
        total_cnt = int(len(summary["predictions"]))
        summary["rehearsal_accuracy"] = float(correct_cnt / total_cnt) if total_cnt > 0 else None
        summary["rehearsal_correct_count"] = correct_cnt
        summary["rehearsal_incorrect_count"] = total_cnt - correct_cnt

    if output_csv_path is not None:
        csv_out = Path(output_csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(csv_out, index=False)
        summary["output_csv_path"] = str(csv_out.resolve())

    if output_json_path is not None:
        json_out = Path(output_json_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_json_path"] = str(json_out.resolve())

    return summary


def format_demo_table(summary: Mapping[str, Any]) -> str:
    """Format offline demo predictions as a clean terminal table for live presentation."""
    lines: list[str] = [
        "=" * 86,
        f"OFFLINE WIFI FLOOR CLASSIFICATION DEMO | model={summary.get('model_id')} ({summary.get('variant')})",
        f"Input source: {summary.get('input_source')}",
        f"Evaluated rows: {summary.get('evaluated_row_count')} | "
        f"Predict time: {summary.get('predict_seconds', 0.0):.4f}s "
        f"({summary.get('mean_latency_ms_per_sample', 0.0):.2f} ms/sample)",
        "-" * 86,
        f"{'row_id':<16} {'bldg':>4} {'detected_waps':>13} {'pred_floor':>10} {'true_floor':>10} {'match':>8}",
        "-" * 86,
    ]
    for rec in summary.get("predictions", []):
        tf_str = str(rec["true_floor"]) if rec.get("true_floor") is not None else "N/A"
        match_str = (
            "OK" if rec.get("correct") is True else ("MISS" if rec.get("correct") is False else "N/A")
        )
        lines.append(
            f"{str(rec['row_id']):<16} {int(rec['building_id']):>4} "
            f"{int(rec['detected_wap_count']):>13} {int(rec['predicted_floor']):>10} "
            f"{tf_str:>10} {match_str:>8}"
        )
    lines.append("-" * 86)
    if summary.get("rehearsal_accuracy") is not None:
        lines.append(
            f"Post-prediction rehearsal check (ground truth withheld from predictors): "
            f"{summary['rehearsal_correct_count']}/{summary['evaluated_row_count']} "
            f"({summary['rehearsal_accuracy'] * 100.0:.2f}%)"
        )
    lines.append("=" * 86)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m shared.demo`."""
    parser = argparse.ArgumentParser(
        prog="python -m shared.demo",
        description=(
            "Offline live-demo inference for WiFi Fingerprint Floor Classification. "
            "Loads SHA-256-verified saved pipelines and predicts original FLOOR labels "
            "from WAP001..WAP520 + known BUILDINGID without retraining."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        choices=list(VALID_MODEL_IDS),
        default=None,
        help="Model identifier (pca_knn, pca_weighted_knn, or gradient_boosting).",
    )
    parser.add_argument(
        "--train-run-id",
        type=str,
        default=None,
        help="Training run ID under models/<model_id>/artifacts/<train_run_id>.",
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        default=None,
        help="Explicit path to a training artifact directory containing selection_config.json.",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=None,
        help="Explicit path to a frozen selection_config.json file.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help="Explicit path to a single building's .joblib pipeline artifact.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["tuned", "fixed_baseline"],
        default="tuned",
        help="Model variant to load from selection_config.json (default: tuned).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV file containing WAP001..WAP520 and BUILDINGID (and optional FLOOR for display).",
    )
    parser.add_argument(
        "--use-dev-rehearsal",
        action="store_true",
        help="Sample deterministic rehearsal rows from internal_holdout (never touches validationData.csv).",
    )
    parser.add_argument(
        "--samples-per-building",
        type=int,
        default=2,
        help="Number of rehearsal rows per building when --use-dev-rehearsal is set (default: 2).",
    )
    parser.add_argument(
        "--building-id",
        type=int,
        choices=list(EXPECTED_BUILDINGS),
        default=None,
        help="Optional known BUILDINGID filter or override (0, 1, or 2).",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to trainingData.csv (used only with --use-dev-rehearsal).",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_CSV,
        help="Path to development_splits.csv (used only with --use-dev-rehearsal).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the demo inference summary JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the demo predictions CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic rehearsal sampling (default: 42).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m shared.demo`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_offline_demo_inference(
        input_csv_path=args.input_csv,
        use_dev_rehearsal=bool(args.use_dev_rehearsal),
        samples_per_building=int(args.samples_per_building),
        artifact_path=args.artifact_path,
        selection_config_path=args.selection_config,
        train_run_dir=args.train_run_dir,
        model_id=args.model_id,
        train_run_id=args.train_run_id,
        variant=args.variant,
        building_id=args.building_id,
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        seed=int(args.seed),
    )
    print(format_demo_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
