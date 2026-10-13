'Stateless data cleaning, schema validation, and sentinel replacement for PCA + distance-weighted kNN.'

from __future__ import annotations

from sklearn.pipeline import Pipeline
from shared.config import DEFAULT_RANDOM_SEED
from models.pca_knn.clean_data import build_configured_pca_knn_pipeline

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from models.pca_knn.clean_data import (
    _load_exclusion_ids_file,
    clean_pca_knn_dataframe,
    compute_encoding_config_sha256,
    extract_validated_wap_features_and_targets,
    load_pca_knn_config,
    run_clean_data as _shared_run_clean_data,
)
from shared.config import (
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    MODEL_ID_PCA_WEIGHTED_KNN,
    UNDETECTED_SENTINEL,
)

PCA_WEIGHTED_KNN_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_YAML: Path = PCA_WEIGHTED_KNN_DIR / "config.yaml"
DEFAULT_PROCESSED_DATA_DIR: Path = PCA_WEIGHTED_KNN_DIR / "processed_data"
DEFAULT_ARTIFACTS_DIR: Path = PCA_WEIGHTED_KNN_DIR / "artifacts"
DEFAULT_RESULTS_DIR: Path = PCA_WEIGHTED_KNN_DIR / "results"


def build_distance_weighted_pca_knn_pipeline(
    n_components: int,
    n_neighbors: int,
    scaling: str = "none",
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = 1,
) -> Pipeline:
    """Construct a leakage-safe distance-weighted Euclidean PCA-kNN pipeline (`weights='distance'`)."""
    return build_configured_pca_knn_pipeline(
        n_components=n_components,
        n_neighbors=n_neighbors,
        scaling=scaling,
        weights="distance",
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
        n_jobs=n_jobs,
    )



def load_pca_weighted_knn_config(
    config_path: Path | str = DEFAULT_CONFIG_YAML,
) -> dict[str, Any]:
    """Load and validate ``models/pca_weighted_knn/config.yaml``."""
    return load_pca_knn_config(
        config_path=config_path,
        expected_model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        expected_weights="distance",
    )


def clean_pca_weighted_knn_dataframe(
    df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    dataset_role: str = "train",
    source_sha256: str | None = None,
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    exclude_row_ids: Sequence[str] | None = None,
    exclusion_justification: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate contract, replace missing-signal sentinels (`100 -> -110.0`), and attach split metadata.

    Delegates to ``clean_pca_knn_dataframe`` with ``model_id="pca_weighted_knn"`` so
    that uniform and distance-weighted kNN never diverge in row population or encoding.
    """
    return clean_pca_knn_dataframe(
        df=df,
        manifest_df=manifest_df,
        dataset_role=dataset_role,
        source_sha256=source_sha256,
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        exclude_row_ids=exclude_row_ids,
        exclusion_justification=exclusion_justification,
        model_id=MODEL_ID_PCA_WEIGHTED_KNN,
    )


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
    """Execute non-destructive cleaning for ``pca_weighted_knn`` and write provenance manifest."""
    return _shared_run_clean_data(
        train_csv_path=train_csv_path,
        split_manifest_path=split_manifest_path,
        output_dir=output_dir,
        output_filename=output_filename,
        config_path=config_path,
        val_csv_path=val_csv_path,
        include_official_validation=include_official_validation,
        exclude_row_ids=exclude_row_ids,
        exclusion_justification=exclusion_justification,
        overwrite=overwrite,
        expected_model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        expected_weights="distance",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``python -m models.pca_weighted_knn.clean_data``."""
    parser = argparse.ArgumentParser(
        prog="python -m models.pca_weighted_knn.clean_data",
        description=(
            "Validate UJIIndoorLoc training data, align with shared development splits, "
            "and write stateless sentinel-replaced (100 -> -110.0) feature data for models/pca_weighted_knn."
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
        help="Path to models/pca_weighted_knn/config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help="Directory to write cleaned CSV and cleaning_manifest.json (default: models/pca_weighted_knn/processed_data).",
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
    """CLI entrypoint for ``python -m models.pca_weighted_knn.clean_data``."""
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
        f"[pca_weighted_knn.clean_data] Cleaned development data written to: {report['output_csv_path']} "
        f"(rows={report['row_counts']['output_rows']}, excluded={report['row_counts']['excluded_rows']}, "
        f"sha256={report['output_csv_sha256']})"
    )
    return 0


__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "PCA_WEIGHTED_KNN_DIR",
    "build_arg_parser",
    "clean_pca_weighted_knn_dataframe",
    "compute_encoding_config_sha256",
    "extract_validated_wap_features_and_targets",
    "load_pca_weighted_knn_config",
    "main",
    "run_clean_data",
]


if __name__ == "__main__":
    raise SystemExit(main())
