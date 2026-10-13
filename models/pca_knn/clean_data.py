'Stateless data cleaning, schema validation, and sentinel replacement for PCA + uniform kNN.'

from __future__ import annotations

from sklearn.pipeline import Pipeline
from shared.config import DEFAULT_RANDOM_SEED
from shared.preprocessing import build_pca_knn_pipeline

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from shared.config import (
    BUILDING_COLUMN,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    DEFAULT_VALIDATION_CSV,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_VALIDATION_ROWS,
    FORBIDDEN_PREDICTOR_COLUMNS,
    MODEL_ID_PCA_KNN,
    ROW_ID_COLUMN,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    SPLIT_ROLE_OFFICIAL_VALIDATION,
    TARGET_COLUMN,
    UNDETECTED_SENTINEL,
    WAP_COLUMNS,
)
from shared.create_splits import align_dataset_with_manifest, load_split_manifest
from shared.data_validation import (
    DataValidationError,
    DatasetAuditReport,
    assign_stable_row_ids,
    audit_dataframe,
    compute_dataframe_sha256,
    compute_file_sha256,
    detect_duplicate_column_names,
    extract_predictors_and_targets,
    load_and_audit_csv,
    validate_predictor_columns,
)
from shared.preprocessing import SentinelReplacementTransformer

PCA_KNN_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_YAML: Path = PCA_KNN_DIR / "config.yaml"
DEFAULT_PROCESSED_DATA_DIR: Path = PCA_KNN_DIR / "processed_data"
DEFAULT_ARTIFACTS_DIR: Path = PCA_KNN_DIR / "artifacts"
DEFAULT_RESULTS_DIR: Path = PCA_KNN_DIR / "results"


def build_configured_pca_knn_pipeline(
    n_components: int,
    n_neighbors: int,
    scaling: str = "none",
    weights: str = "uniform",
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = 1,
) -> Pipeline:
    """Construct a leakage-safe Euclidean PCA-kNN pipeline with native sklearn `weights` ('uniform' or 'distance')."""
    if scaling not in ("none", "standard"):
        raise ValueError(f"scaling must be 'none' or 'standard', got {scaling!r}.")
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}.")
    scale_bool = scaling == "standard"
    pipeline = build_pca_knn_pipeline(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        weights=weights,
        metric="euclidean",
        scale=scale_bool,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
        n_jobs=int(n_jobs),
    )
    clf = pipeline.named_steps["classifier"]
    pca_step = pipeline.named_steps["pca"]
    assert isinstance(clf.weights, str) and clf.weights == weights, (
        f"PCA-kNN pipeline must use native sklearn weights={weights!r}, not a custom callable."
    )
    assert clf.metric == "euclidean", "PCA-kNN pipeline must have metric='euclidean'."
    assert pca_step.whiten is False, "PCA step must have whiten=False."
    return pipeline



def load_pca_knn_config(
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    expected_model_id: str = MODEL_ID_PCA_KNN,
    expected_weights: str = "uniform",
) -> dict[str, Any]:
    """Load and validate a PCA-kNN family YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PCA-kNN configuration file not found at '{path}'. "
            f"Expected config at '{DEFAULT_CONFIG_YAML}'."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected top-level mapping.")
    if cfg.get("model_id") != expected_model_id:
        raise ValueError(
            f"Configuration model_id mismatch in {path}: expected '{expected_model_id}', "
            f"got {cfg.get('model_id')!r}."
        )
    est_cfg = cfg.get("estimator", {})
    if est_cfg.get("weights") != expected_weights:
        raise ValueError(
            f"Configuration for {expected_model_id} requires estimator.weights={expected_weights!r}, "
            f"got {est_cfg.get('weights')!r}."
        )
    if est_cfg.get("metric", "euclidean") != "euclidean":
        raise ValueError(
            f"Configuration for {expected_model_id} requires estimator.metric='euclidean', "
            f"got {est_cfg.get('metric')!r}."
        )
    return cfg


def validate_no_extra_predictor_leakage(
    df: pd.DataFrame,
    candidate_predictor_columns: Sequence[str] | None = None,
) -> list[str]:
    'Verify predictor column allowlist (`WAP001..WAP520`) and reject any metadata/target leakage.'
    dup_cols = detect_duplicate_column_names(list(df.columns))
    if dup_cols:
        raise DataValidationError(f"DataFrame contains duplicate column names: {dup_cols}")

    if candidate_predictor_columns is not None:
        validate_predictor_columns(candidate_predictor_columns)
        missing_in_df = [c for c in candidate_predictor_columns if c not in df.columns]
        if missing_in_df:
            raise DataValidationError(f"Predictor columns missing from DataFrame: {missing_in_df[:5]}")
        return [str(c) for c in candidate_predictor_columns]

    present_waps = [str(c) for c in df.columns if str(c) in set(WAP_COLUMNS)]
    validate_predictor_columns(present_waps)
    return list(WAP_COLUMNS)


def extract_validated_wap_features_and_targets(
    df: pd.DataFrame,
    require_row_id: bool = False,
    allow_already_cleaned_replacement: bool = True,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    candidate_predictor_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    """Extract `(X, y, building_id, row_ids)` from either raw (`100` sentinel) or cleaned (`-110.0`) DataFrames.

    Enforces all shared predictor allowlist, numeric, finiteness, and RSSI range checks.
    """
    predictor_cols = validate_no_extra_predictor_leakage(
        df, candidate_predictor_columns=candidate_predictor_columns
    )

    if allow_already_cleaned_replacement:
        wap_sub = df.loc[:, predictor_cols]
        for col in predictor_cols:
            if not pd.api.types.is_numeric_dtype(wap_sub[col]) or pd.api.types.is_bool_dtype(wap_sub[col]):
                raise DataValidationError(f"Non-numeric WAP column detected: {col}")
        if wap_sub.isna().any().any():
            raise DataValidationError("WAP predictor matrix contains NaN/null values.")
        vals = wap_sub.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(vals).all():
            raise DataValidationError("WAP predictor matrix contains infinite values.")

        has_replacement = bool(np.any(vals == float(replacement_value)))
        if has_replacement:
            # Temporarily restore replacement_value -> sentinel_value on a copy so shared
            # extract_predictors_and_targets runs its full range check [-104, 0] U {100}.
            temp_df = df.copy()
            temp_wap = vals.copy()
            temp_wap[temp_wap == float(replacement_value)] = float(sentinel_value)
            temp_df.loc[:, predictor_cols] = temp_wap
            _, y, building, row_ids = extract_predictors_and_targets(
                temp_df, require_row_id=require_row_id
            )
            X = df.loc[:, predictor_cols].copy()
            return X, y, building, row_ids

    return extract_predictors_and_targets(df, require_row_id=require_row_id)


def compute_encoding_config_sha256(
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
) -> str:
    """Compute a deterministic SHA-256 hash of the missing-signal encoding and predictor allowlist contract."""
    import hashlib

    payload = json.dumps(
        {
            "predictor_columns": list(WAP_COLUMNS),
            "sentinel_value": float(sentinel_value),
            "replacement_value": float(replacement_value),
            "preserve_genuine_zero_dbm": True,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_pca_knn_dataframe(
    df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    dataset_role: str = "train",
    source_sha256: str | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    exclude_row_ids: Sequence[str] | None = None,
    exclusion_justification: str | None = None,
    model_id: str = MODEL_ID_PCA_KNN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    'Validate and apply stateless sentinel replacement (`100 -> -110.0`) without fitting scaling or PCA.'
    if dataset_role not in ("train", "official_validation"):
        raise ValueError(
            f"dataset_role must be 'train' or 'official_validation', got {dataset_role!r}."
        )

    resolved_sha = source_sha256 or compute_dataframe_sha256(df)
    work_df = df.copy()
    if ROW_ID_COLUMN not in work_df.columns:
        role_prefix = "train" if dataset_role == "train" else "val"
        work_df = assign_stable_row_ids(
            work_df, dataset_role=role_prefix, file_sha256=resolved_sha, inplace=False
        )

    audit_report: DatasetAuditReport = audit_dataframe(
        work_df,
        dataset_role="train" if dataset_role == "train" else "val",
        file_sha256=resolved_sha,
        strict=True,
    )

    if dataset_role == "train":
        if manifest_df is not None:
            work_df = align_dataset_with_manifest(
                train_df=work_df,
                manifest_df=manifest_df,
                expected_source_sha256=source_sha256,
            )
        else:
            required_split_cols = ("split_role", "fold", "fold_3", "fold_5")
            missing_split = [c for c in required_split_cols if c not in work_df.columns]
            if missing_split:
                raise ValueError(
                    f"Training dataset cleaning requires `manifest_df` or existing split columns "
                    f"{required_split_cols}; missing: {missing_split}."
                )
    else:
        work_df["split_role"] = SPLIT_ROLE_OFFICIAL_VALIDATION
        work_df["fold"] = -1
        work_df["fold_3"] = -1
        work_df["fold_5"] = -1
        work_df["seed"] = -1
        work_df["source_sha256"] = str(resolved_sha)

    input_row_count = int(len(work_df))
    excluded_ids_actual: list[str] = []

    if exclude_row_ids is not None and len(exclude_row_ids) > 0:
        if not exclusion_justification or not str(exclusion_justification).strip():
            raise ValueError(
                "Row exclusions are prohibited unless a non-empty `exclusion_justification` is provided."
            )
        exclude_set = {str(rid).strip() for rid in exclude_row_ids if str(rid).strip()}
        present_ids = set(work_df[ROW_ID_COLUMN].astype(str).tolist())
        unknown_ids = sorted(exclude_set - present_ids)
        if unknown_ids:
            raise ValueError(
                f"Configured exclude_row_ids contain {len(unknown_ids)} unknown row_ids "
                f"(first 5: {unknown_ids[:5]})."
            )
        keep_mask = ~work_df[ROW_ID_COLUMN].astype(str).isin(exclude_set)
        excluded_ids_actual = work_df.loc[~keep_mask, ROW_ID_COLUMN].astype(str).tolist()
        work_df = work_df.loc[keep_mask].reset_index(drop=True)

    X_raw, y, building, row_ids = extract_validated_wap_features_and_targets(
        work_df,
        require_row_id=True,
        allow_already_cleaned_replacement=True,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
    )

    transformer = SentinelReplacementTransformer(
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=True,
    )
    X_cleaned_arr = transformer.fit_transform(X_raw)
    if not np.isfinite(X_cleaned_arr).all():
        raise DataValidationError("Cleaned WAP feature matrix contains non-finite values.")

    cleaned_wap_df = pd.DataFrame(
        X_cleaned_arr,
        columns=list(WAP_COLUMNS),
        index=work_df.index,
    )
    non_wap_cols = [c for c in work_df.columns if c not in set(WAP_COLUMNS)]
    cleaned_df = pd.concat([cleaned_wap_df, work_df.loc[:, non_wap_cols]], axis=1)

    # Verify no forbidden predictor leakage in WAP columns
    validate_predictor_columns(list(WAP_COLUMNS))
    assert not (set(WAP_COLUMNS) & set(FORBIDDEN_PREDICTOR_COLUMNS))

    split_role_counts = {
        str(role): int(cnt)
        for role, cnt in cleaned_df["split_role"].value_counts().sort_index().items()
    }
    per_building_counts: dict[str, dict[str, Any]] = {}
    for b_id in sorted(int(b) for b in cleaned_df[BUILDING_COLUMN].astype(int).unique()):
        b_sub = cleaned_df[cleaned_df[BUILDING_COLUMN].astype(int) == b_id]
        per_building_counts[str(b_id)] = {
            "total_rows": int(len(b_sub)),
            "dev_train_rows": int((b_sub["split_role"] == SPLIT_ROLE_DEV_TRAIN).sum()),
            "internal_holdout_rows": int((b_sub["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT).sum()),
            "official_validation_rows": int((b_sub["split_role"] == SPLIT_ROLE_OFFICIAL_VALIDATION).sum()),
            "floors_present": sorted(int(f) for f in b_sub[TARGET_COLUMN].astype(int).unique()),
        }

    encoding_sha256 = compute_encoding_config_sha256(
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
    )

    cleaning_report: dict[str, Any] = {
        "model_id": str(model_id),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": dataset_role,
        "source_sha256": str(resolved_sha),
        "encoding_config_sha256": encoding_sha256,
        "cleaned_dataframe_sha256": compute_dataframe_sha256(cleaned_df),
        "sentinel_replacement": {
            "sentinel_value": sentinel_value,
            "replacement_value": float(replacement_value),
            "replacement_is_assumption": True,
            "preserve_genuine_zero_dbm": True,
            "encoding_config_sha256": encoding_sha256,
            "sentinel_100_replaced_count": int(audit_report.rssi_summary.get("sentinel_100_count", 0)),
            "genuine_zero_preserved_count": int(audit_report.rssi_summary.get("genuine_zero_count", 0)),
        },
        "leakage_safeguards": {
            "predictor_columns": f"{WAP_COLUMNS[0]}..{WAP_COLUMNS[-1]} ({len(WAP_COLUMNS)} columns)",
            "global_scaling_fitted": False,
            "global_pca_fitted": False,
            "fit_scope_policy": "stateless_sentinel_replacement_only_scaling_and_pca_fit_per_fold_in_train",
        },
        "row_counts": {
            "input_rows": input_row_count,
            "output_rows": int(len(cleaned_df)),
            "excluded_rows": int(len(excluded_ids_actual)),
            "split_role_counts": split_role_counts,
            "per_building": per_building_counts,
        },
        "exclusions": {
            "configured": bool(len(excluded_ids_actual) > 0),
            "excluded_row_count": int(len(excluded_ids_actual)),
            "excluded_row_ids": excluded_ids_actual,
            "justification": str(exclusion_justification).strip() if excluded_ids_actual else None,
        },
        "audit_diagnostics": {
            "all_undetected_row_count": int(audit_report.all_undetected_row_count),
            "duplicate_diagnostics": dict(audit_report.duplicate_diagnostics),
            "warnings": list(audit_report.warnings),
        },
    }
    return cleaned_df, cleaning_report


def _load_exclusion_ids_file(file_path: Path | str) -> list[str]:
    """Load a list of row_ids to exclude from a JSON list or newline-delimited text file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Exclusion row IDs file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list of row_ids in {path}, got {type(parsed).__name__}.")
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def run_clean_data(
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    output_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
    output_filename: str = "development_cleaned.csv",
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    val_csv_path: Path | str | None = None,
    include_official_validation: bool = False,
    exclude_row_ids: Sequence[str] | None = None,
    exclusion_justification: str | None = None,
    overwrite: bool = False,
    expected_model_id: str = MODEL_ID_PCA_KNN,
    expected_weights: str = "uniform",
) -> dict[str, Any]:
    """Validate raw CSVs, align with shared split manifest, and write cleaned development feature data."""
    cfg = load_pca_knn_config(
        config_path,
        expected_model_id=expected_model_id,
        expected_weights=expected_weights,
    )
    prep_cfg: Mapping[str, Any] = cfg.get("preprocessing", {})
    sentinel_val = prep_cfg.get("sentinel_value", UNDETECTED_SENTINEL)
    replacement_val = float(prep_cfg.get("replacement_value", DEFAULT_UNDETECTED_REPLACEMENT))

    if val_csv_path is not None and not include_official_validation:
        raise PermissionError(
            "Refusing to process official validation CSV without explicit `--include-official-validation` flag."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv_path = out_dir / output_filename
    manifest_json_path = out_dir / "cleaning_manifest.json"

    if not overwrite and (cleaned_csv_path.exists() or manifest_json_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing cleaned data in '{out_dir}'. "
            "Pass `--overwrite` if you intend to regenerate cleaned files."
        )

    train_df, train_audit = load_and_audit_csv(
        train_csv_path,
        dataset_role="train",
        expected_row_count=EXPECTED_TRAINING_ROWS,
        strict=True,
        attach_row_ids=True,
    )
    split_manifest_df, split_manifest_sha256, _ = load_split_manifest(
        split_manifest_path,
        strict=True,
    )

    cleaned_dev_df, dev_report = clean_pca_knn_dataframe(
        df=train_df,
        manifest_df=split_manifest_df,
        dataset_role="train",
        source_sha256=train_audit.file_sha256,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
        exclude_row_ids=exclude_row_ids,
        exclusion_justification=exclusion_justification,
        model_id=expected_model_id,
    )

    cleaned_dev_df.to_csv(cleaned_csv_path, index=False)
    cleaned_csv_sha256 = compute_file_sha256(cleaned_csv_path)

    dev_report["source_csv_path"] = str(Path(train_csv_path).resolve())
    dev_report["split_manifest_path"] = str(Path(split_manifest_path).resolve())
    dev_report["split_manifest_sha256"] = str(split_manifest_sha256)
    dev_report["output_csv_path"] = str(cleaned_csv_path.resolve())
    dev_report["output_csv_sha256"] = str(cleaned_csv_sha256)

    if include_official_validation:
        resolved_val_csv = Path(val_csv_path) if val_csv_path is not None else DEFAULT_VALIDATION_CSV
        val_out_path = out_dir / "official_validation_cleaned.csv"
        if not overwrite and val_out_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing {val_out_path} without --overwrite.")
        val_df, val_audit = load_and_audit_csv(
            resolved_val_csv,
            dataset_role="val",
            expected_row_count=EXPECTED_VALIDATION_ROWS,
            strict=True,
            attach_row_ids=True,
        )
        cleaned_val_df, val_report = clean_pca_knn_dataframe(
            df=val_df,
            manifest_df=None,
            dataset_role="official_validation",
            source_sha256=val_audit.file_sha256,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            exclude_row_ids=None,
            exclusion_justification=None,
            model_id=expected_model_id,
        )
        cleaned_val_df.to_csv(val_out_path, index=False)
        val_report["source_csv_path"] = str(resolved_val_csv.resolve())
        val_report["output_csv_path"] = str(val_out_path.resolve())
        val_report["output_csv_sha256"] = compute_file_sha256(val_out_path)
        dev_report["official_validation_cleaning"] = val_report

    with manifest_json_path.open("w", encoding="utf-8") as fh:
        json.dump(dev_report, fh, indent=2, sort_keys=False)

    return dev_report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m models.pca_knn.clean_data`."""
    parser = argparse.ArgumentParser(
        prog="python -m models.pca_knn.clean_data",
        description=(
            "Validate UJIIndoorLoc training data, align with shared development splits, "
            "and write stateless sentinel-replaced (100 -> -110.0) feature data without global scaling or PCA."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to immutable raw trainingData.csv (default: Dataset/trainingData.csv).",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_CSV,
        help="Path to shared development_splits.csv manifest (default: shared/manifests/development_splits.csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_YAML,
        help="Path to models/pca_knn/config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help="Directory to write cleaned CSV and cleaning_manifest.json (default: models/pca_knn/processed_data).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="development_cleaned.csv",
        help="Filename for cleaned development CSV (default: development_cleaned.csv).",
    )
    parser.add_argument(
        "--include-official-validation",
        action="store_true",
        help="Explicitly opt in to also writing official_validation_cleaned.csv from validationData.csv.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Optional path to official validationData.csv (requires --include-official-validation).",
    )
    parser.add_argument(
        "--exclude-row-ids-file",
        type=Path,
        default=None,
        help="Optional path to JSON/text file listing row_ids to exclude (requires --exclusion-justification).",
    )
    parser.add_argument(
        "--exclusion-justification",
        type=str,
        default=None,
        help="Explicit scientific justification required if --exclude-row-ids-file is provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in --output-dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m models.pca_knn.clean_data`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exclude_ids: list[str] | None = None
    if args.exclude_row_ids_file is not None:
        exclude_ids = _load_exclusion_ids_file(args.exclude_row_ids_file)

    report = run_clean_data(
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        config_path=args.config,
        val_csv_path=args.val_csv,
        include_official_validation=args.include_official_validation,
        exclude_row_ids=exclude_ids,
        exclusion_justification=args.exclusion_justification,
        overwrite=args.overwrite,
    )
    print(
        f"[pca_knn.clean_data] Cleaned development data written to: {report['output_csv_path']} "
        f"(rows={report['row_counts']['output_rows']}, excluded={report['row_counts']['excluded_rows']}, "
        f"sha256={report['output_csv_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
