'Stateless data cleaning, schema validation, and finite sentinel replacement for classical Gradient Boosting.'

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from shared.config import DEFAULT_RANDOM_SEED
from shared.preprocessing import build_gradient_boosting_pipeline

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from models.pca_knn.clean_data import (
    _load_exclusion_ids_file,
    clean_pca_knn_dataframe,
    compute_encoding_config_sha256,
    extract_validated_wap_features_and_targets,
    validate_no_extra_predictor_leakage,
)
from shared.config import (
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    DEFAULT_VALIDATION_CSV,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_VALIDATION_ROWS,
    MODEL_ID_GRADIENT_BOOSTING,
    UNDETECTED_SENTINEL,
)
from shared.create_splits import load_split_manifest
from shared.data_validation import compute_file_sha256, load_and_audit_csv

GRADIENT_BOOSTING_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_YAML: Path = GRADIENT_BOOSTING_DIR / "config.yaml"
DEFAULT_PROCESSED_DATA_DIR: Path = GRADIENT_BOOSTING_DIR / "processed_data"
DEFAULT_ARTIFACTS_DIR: Path = GRADIENT_BOOSTING_DIR / "artifacts"
DEFAULT_RESULTS_DIR: Path = GRADIENT_BOOSTING_DIR / "results"


def build_configured_gradient_boosting_pipeline(
    n_estimators: int = 189,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    max_features: str | int | float | None = None,
    scale: bool = False,
    use_pca: bool = False,
    n_components: int | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    'Construct a leakage-safe classical `GradientBoostingClassifier` pipeline.'
    pipeline = build_gradient_boosting_pipeline(
        learning_rate=float(learning_rate),
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        subsample=float(subsample),
        max_features=max_features,
        scale=bool(scale),
        use_pca=bool(use_pca),
        n_components=n_components,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=int(random_state),
    )
    clf = pipeline.named_steps["classifier"]
    assert type(clf) is GradientBoostingClassifier, (
        f"Expected exact sklearn.ensemble.GradientBoostingClassifier, got {type(clf)!r}."
    )
    assert clf.n_iter_no_change is None, (
        "Early stopping (n_iter_no_change) must be disabled by default to prevent validation leakage."
    )
    if not scale:
        assert "scaler" not in pipeline.named_steps, "StandardScaler must not be present when scale=False."
    if not use_pca:
        assert "pca" not in pipeline.named_steps, "PCA must not be present when use_pca=False."
    return pipeline



def load_gradient_boosting_config(
    config_path: Path | str = DEFAULT_CONFIG_YAML,
) -> dict[str, Any]:
    """Load and validate `models/gradient_boosting/config.yaml`."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gradient boosting configuration file not found at '{path}'. "
            f"Expected config at '{DEFAULT_CONFIG_YAML}'."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected top-level mapping.")
    if cfg.get("model_id") != MODEL_ID_GRADIENT_BOOSTING:
        raise ValueError(
            f"Configuration model_id mismatch in {path}: expected '{MODEL_ID_GRADIENT_BOOSTING}', "
            f"got {cfg.get('model_id')!r}."
        )
    est_cfg = cfg.get("estimator", {})
    expected_cls = "sklearn.ensemble.GradientBoostingClassifier"
    if est_cfg.get("class_name") != expected_cls:
        raise ValueError(
            f"Configuration for {MODEL_ID_GRADIENT_BOOSTING} requires estimator.class_name={expected_cls!r} "
            f"(no XGBoost or HistGradientBoostingClassifier substitution); got {est_cfg.get('class_name')!r}."
        )
    if est_cfg.get("n_iter_no_change") is not None or bool(est_cfg.get("early_stopping_enabled", False)):
        raise ValueError(
            "Configuration for gradient_boosting must disable early stopping by default "
            "(estimator.n_iter_no_change=null, early_stopping_enabled=false)."
        )
    prep_cfg = cfg.get("preprocessing", {})
    if bool(prep_cfg.get("scale", False)) or bool(prep_cfg.get("use_pca", False)):
        raise ValueError(
            "Default preprocessing for gradient_boosting must have scale=false and use_pca=false."
        )
    return cfg


def clean_gradient_boosting_dataframe(
    df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    dataset_role: str = "train",
    source_sha256: str | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    exclude_row_ids: Sequence[str] | None = None,
    exclusion_justification: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    'Validate schema, enforce finite WAP values, replace `100 -> -110.0`, and align split metadata.'
    cleaned_df, report = clean_pca_knn_dataframe(
        df=df,
        manifest_df=manifest_df,
        dataset_role=dataset_role,
        source_sha256=source_sha256,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        exclude_row_ids=exclude_row_ids,
        exclusion_justification=exclusion_justification,
        model_id=MODEL_ID_GRADIENT_BOOSTING,
    )
    report["leakage_safeguards"]["fit_scope_policy"] = (
        "stateless_sentinel_replacement_only_raw_wap_features_no_pca_no_scaling_by_default"
    )
    report["leakage_safeguards"]["native_nan_handling_claimed"] = False
    return cleaned_df, report


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
) -> dict[str, Any]:
    """Validate raw CSVs, align with shared split manifest, and write cleaned development feature data."""
    import json

    cfg = load_gradient_boosting_config(config_path)
    prep_cfg = cfg.get("preprocessing", {})
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

    cleaned_dev_df, dev_report = clean_gradient_boosting_dataframe(
        df=train_df,
        manifest_df=split_manifest_df,
        dataset_role="train",
        source_sha256=train_audit.file_sha256,
        sentinel_value=sentinel_val,
        replacement_value=replacement_val,
        exclude_row_ids=exclude_row_ids,
        exclusion_justification=exclusion_justification,
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
        cleaned_val_df, val_report = clean_gradient_boosting_dataframe(
            df=val_df,
            manifest_df=None,
            dataset_role="official_validation",
            source_sha256=val_audit.file_sha256,
            sentinel_value=sentinel_val,
            replacement_value=replacement_val,
            exclude_row_ids=None,
            exclusion_justification=None,
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
    """Build CLI argument parser for `python -m models.gradient_boosting.clean_data`."""
    parser = argparse.ArgumentParser(
        prog="python -m models.gradient_boosting.clean_data",
        description=(
            "Validate UJIIndoorLoc training data, align with shared development splits, "
            "and write finite sentinel-replaced (100 -> -110.0) raw WAP feature data for classical gradient boosting."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to immutable raw trainingData.csv.",
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
        help="Path to models/gradient_boosting/config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help="Directory to write cleaned CSV and cleaning_manifest.json (default: models/gradient_boosting/processed_data).",
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
    """CLI entrypoint for `python -m models.gradient_boosting.clean_data`."""
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
        f"[gradient_boosting.clean_data] Cleaned development data written to: {report['output_csv_path']} "
        f"(rows={report['row_counts']['output_rows']}, excluded={report['row_counts']['excluded_rows']}, "
        f"sha256={report['output_csv_sha256']})"
    )
    return 0


__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "GRADIENT_BOOSTING_DIR",
    "build_arg_parser",
    "clean_gradient_boosting_dataframe",
    "compute_encoding_config_sha256",
    "extract_validated_wap_features_and_targets",
    "load_gradient_boosting_config",
    "main",
    "run_clean_data",
    "validate_no_extra_predictor_leakage",
]


if __name__ == "__main__":
    raise SystemExit(main())
