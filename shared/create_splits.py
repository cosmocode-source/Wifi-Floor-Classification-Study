"""Shared development split and stratified CV manifest generation for UJIIndoorLoc."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from shared.config import (
    BUILDING_COLUMN,
    DEFAULT_FINAL_CV_FOLDS,
    DEFAULT_INTERNAL_HOLDOUT_FRACTION,
    DEFAULT_LIGHTWEIGHT_CV_FOLDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_SPLIT_MANIFEST_META_JSON,
    DEFAULT_TRAINING_CSV,
    ROW_ID_COLUMN,
    SPLIT_ROLE_DEV_TRAIN,
    SPLIT_ROLE_INTERNAL_HOLDOUT,
    TARGET_COLUMN,
)
from shared.data_validation import (
    DataValidationError,
    assign_stable_row_ids,
    compute_dataframe_sha256,
    compute_file_sha256,
    load_and_audit_csv,
)
from shared.feasibility import check_class_support_feasibility


class SplitCreationError(ValueError):
    """Raised when development split creation or manifest validation fails."""


@dataclass
class SplitValidationReport:
    """Validation summary verifying disjointness, completeness, and class coverage of a split manifest."""

    valid: bool
    manifest_sha256: str
    total_rows: int
    dev_train_rows: int
    internal_holdout_rows: int
    buildings: list[int] = field(default_factory=list)
    n_splits_checked: list[int] = field(default_factory=list)
    per_building_counts: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation report to a JSON-compatible dictionary."""
        return asdict(self)


MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    ROW_ID_COLUMN,
    "source_row_index",
    "building_id",
    "floor",
    "split_role",
    "fold",
    "fold_3",
    "fold_5",
    "seed",
    "source_sha256",
)


def create_development_splits(
    train_df: pd.DataFrame,
    source_sha256: str | None = None,
    holdout_fraction: float = DEFAULT_INTERNAL_HOLDOUT_FRACTION,
    primary_cv_folds: int = DEFAULT_FINAL_CV_FOLDS,
    lightweight_cv_folds: int = DEFAULT_LIGHTWEIGHT_CV_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
    source_name: str = "trainingData.csv",
) -> pd.DataFrame:
    'Create per-building stratified internal holdout and shared CV fold manifests from trainingData.csv.'
    if "validationdata" in str(source_name).lower():
        raise SplitCreationError(
            "Forbidden operation: official validationData.csv must never be used to generate development splits."
        )
    if not (0.0 < float(holdout_fraction) < 1.0):
        raise SplitCreationError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}.")
    if int(primary_cv_folds) < 2 or int(lightweight_cv_folds) < 2:
        raise SplitCreationError("CV fold counts must be integers >= 2.")

    for req_col in (BUILDING_COLUMN, TARGET_COLUMN):
        if req_col not in train_df.columns:
            raise SplitCreationError(f"Missing required column '{req_col}' in training DataFrame.")

    resolved_sha = source_sha256 or compute_dataframe_sha256(train_df)
    work_df = train_df.copy()
    if ROW_ID_COLUMN not in work_df.columns:
        work_df = assign_stable_row_ids(work_df, dataset_role="train", file_sha256=resolved_sha, inplace=False)

    n_rows = len(work_df)
    split_roles = np.full(n_rows, "", dtype=object)
    fold_primary = np.full(n_rows, -1, dtype=int)
    fold_3_arr = np.full(n_rows, -1, dtype=int)
    fold_5_arr = np.full(n_rows, -1, dtype=int)

    buildings = sorted(work_df[BUILDING_COLUMN].astype(int).unique())
    if not buildings:
        raise SplitCreationError("No buildings found in training DataFrame.")

    for idx_b, b_id in enumerate(buildings):
        b_mask = work_df[BUILDING_COLUMN].astype(int).to_numpy() == int(b_id)
        b_indices = np.flatnonzero(b_mask)
        y_b = work_df.iloc[b_indices][TARGET_COLUMN].astype(int).to_numpy()

        # Check class support before holdout split (need at least 2 per class to put >=1 in holdout and >=1 in train)
        pre_check = check_class_support_feasibility(y_b, n_splits=2, strict=False)
        if not pre_check.feasible:
            raise SplitCreationError(
                f"Building {b_id} failed initial holdout feasibility check: {pre_check.message}"
            )

        building_seed = int(seed) + int(b_id) * 1009 + idx_b
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(holdout_fraction),
            random_state=building_seed,
        )
        try:
            dev_rel_idx, holdout_rel_idx = next(sss.split(np.zeros(len(y_b)), y_b))
        except ValueError as exc:
            raise SplitCreationError(
                f"Building {b_id} could not be stratified into dev_train and internal_holdout: {exc}"
            ) from exc

        dev_abs_idx = b_indices[dev_rel_idx]
        holdout_abs_idx = b_indices[holdout_rel_idx]

        split_roles[dev_abs_idx] = SPLIT_ROLE_DEV_TRAIN
        split_roles[holdout_abs_idx] = SPLIT_ROLE_INTERNAL_HOLDOUT

        y_dev = y_b[dev_rel_idx]
        max_folds_needed = max(int(primary_cv_folds), int(lightweight_cv_folds))
        cv_check = check_class_support_feasibility(y_dev, n_splits=max_folds_needed, strict=False)
        if not cv_check.feasible:
            raise SplitCreationError(
                f"Building {b_id} dev_train pool failed CV class support check: {cv_check.message}"
            )

        # Generate lightweight folds (default 3)
        skf_light = StratifiedKFold(
            n_splits=int(lightweight_cv_folds),
            shuffle=True,
            random_state=building_seed + 17,
        )
        for fold_idx, (_, val_rel) in enumerate(skf_light.split(np.zeros(len(y_dev)), y_dev)):
            fold_3_arr[dev_abs_idx[val_rel]] = int(fold_idx)

        # Generate primary folds (default 5)
        skf_primary = StratifiedKFold(
            n_splits=int(primary_cv_folds),
            shuffle=True,
            random_state=building_seed + 31,
        )
        for fold_idx, (_, val_rel) in enumerate(skf_primary.split(np.zeros(len(y_dev)), y_dev)):
            fold_5_arr[dev_abs_idx[val_rel]] = int(fold_idx)
            fold_primary[dev_abs_idx[val_rel]] = int(fold_idx)

    manifest_df = pd.DataFrame(
        {
            ROW_ID_COLUMN: work_df[ROW_ID_COLUMN].astype(str).to_numpy(),
            "source_row_index": np.arange(n_rows, dtype=int),
            "building_id": work_df[BUILDING_COLUMN].astype(int).to_numpy(),
            "floor": work_df[TARGET_COLUMN].astype(int).to_numpy(),
            "split_role": split_roles,
            "fold": fold_primary,
            "fold_3": fold_3_arr,
            "fold_5": fold_5_arr,
            "seed": np.full(n_rows, int(seed), dtype=int),
            "source_sha256": np.full(n_rows, str(resolved_sha), dtype=object),
        }
    )

    report = validate_split_manifest(
        manifest_df,
        expected_n_splits_map={
            "fold": int(primary_cv_folds),
            "fold_3": int(lightweight_cv_folds),
            "fold_5": int(primary_cv_folds),
        },
        strict=True,
    )
    assert report.valid, "Split manifest validation failed unexpectedly."
    return manifest_df


def validate_split_manifest(
    manifest_df: pd.DataFrame,
    expected_n_splits_map: dict[str, int] | None = None,
    strict: bool = True,
) -> SplitValidationReport:
    """Validate disjointness, row uniqueness, fold completeness, and per-building floor coverage."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in manifest_df.columns]
    if missing_cols:
        msg = f"Manifest is missing required columns: {missing_cols}"
        if strict:
            raise SplitCreationError(msg)
        return SplitValidationReport(
            valid=False,
            manifest_sha256="",
            total_rows=len(manifest_df),
            dev_train_rows=0,
            internal_holdout_rows=0,
            errors=[msg],
        )

    manifest_sha256 = compute_dataframe_sha256(manifest_df)
    total_rows = int(len(manifest_df))
    if manifest_df[list(MANIFEST_REQUIRED_COLUMNS)].isna().any().any():
        errors.append("Manifest contains null values in required columns.")
    integer_columns = ("source_row_index", "building_id", "floor", "fold", "fold_3", "fold_5", "seed")
    for column in integer_columns:
        values = pd.to_numeric(manifest_df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            errors.append(f"Manifest {column} must contain finite integer values.")
    if not np.array_equal(manifest_df["source_row_index"].to_numpy(), np.arange(total_rows)):
        errors.append("Manifest source_row_index must match original row order 0..N-1.")
    if manifest_df["source_sha256"].astype(str).nunique() != 1:
        errors.append("Manifest must contain one shared source_sha256.")
    if errors:
        if strict:
            raise SplitCreationError("; ".join(errors))
        return SplitValidationReport(
            valid=False, manifest_sha256=manifest_sha256, total_rows=total_rows,
            dev_train_rows=int((manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN).sum()),
            internal_holdout_rows=int((manifest_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT).sum()),
            errors=errors,
        )

    if manifest_df[ROW_ID_COLUMN].duplicated().any():
        dup_count = int(manifest_df[ROW_ID_COLUMN].duplicated().sum())
        errors.append(f"Found {dup_count} duplicate row_id entries in manifest.")

    if manifest_df["source_row_index"].duplicated().any():
        errors.append("Duplicate source_row_index entries found in manifest.")

    roles = set(manifest_df["split_role"].unique())
    allowed_roles = {SPLIT_ROLE_DEV_TRAIN, SPLIT_ROLE_INTERNAL_HOLDOUT}
    invalid_roles = roles - allowed_roles
    if invalid_roles:
        errors.append(f"Invalid split_role values in development manifest: {sorted(invalid_roles)}")

    dev_mask = manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN
    holdout_mask = manifest_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT

    dev_ids = set(manifest_df.loc[dev_mask, ROW_ID_COLUMN])
    holdout_ids = set(manifest_df.loc[holdout_mask, ROW_ID_COLUMN])
    overlap_ids = dev_ids & holdout_ids
    if overlap_ids:
        errors.append(f"Leakage error: {len(overlap_ids)} row_ids appear in both dev_train and internal_holdout.")

    if int(dev_mask.sum()) + int(holdout_mask.sum()) != total_rows:
        errors.append("dev_train and internal_holdout row counts do not sum to total_rows.")

    if expected_n_splits_map is None:
        n_fold_primary = int(manifest_df.loc[dev_mask, "fold"].max()) + 1 if dev_mask.any() else 5
        n_fold_3 = int(manifest_df.loc[dev_mask, "fold_3"].max()) + 1 if dev_mask.any() else 3
        n_fold_5 = int(manifest_df.loc[dev_mask, "fold_5"].max()) + 1 if dev_mask.any() else 5
        expected_n_splits_map = {"fold": n_fold_primary, "fold_3": n_fold_3, "fold_5": n_fold_5}

    # Check that internal_holdout rows always have fold == -1 across all fold columns
    for fold_col in expected_n_splits_map:
        if fold_col in manifest_df.columns:
            bad_holdout_folds = manifest_df.loc[holdout_mask, fold_col] != -1
            if bad_holdout_folds.any():
                errors.append(
                    f"internal_holdout rows have non -1 assignments in column '{fold_col}'."
                )

    buildings = sorted(int(b) for b in manifest_df["building_id"].unique())
    per_building_counts: dict[str, dict[str, Any]] = {}

    for b_id in buildings:
        b_df = manifest_df[manifest_df["building_id"] == b_id]
        b_dev = b_df[b_df["split_role"] == SPLIT_ROLE_DEV_TRAIN]
        b_hold = b_df[b_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT]

        all_floors = sorted(int(f) for f in b_df["floor"].unique())
        dev_floors = sorted(int(f) for f in b_dev["floor"].unique())
        hold_floors = sorted(int(f) for f in b_hold["floor"].unique())

        if dev_floors != all_floors:
            errors.append(
                f"Building {b_id} dev_train is missing floor classes: expected {all_floors}, got {dev_floors}."
            )
        if hold_floors != all_floors:
            errors.append(
                f"Building {b_id} internal_holdout is missing floor classes: expected {all_floors}, got {hold_floors}."
            )

        fold_summaries: dict[str, Any] = {}
        for fold_col, expected_k in expected_n_splits_map.items():
            actual_folds = sorted(int(k) for k in b_dev[fold_col].unique())
            expected_folds = list(range(int(expected_k)))
            if actual_folds != expected_folds:
                errors.append(
                    f"Building {b_id} column '{fold_col}' has fold IDs {actual_folds}, expected {expected_folds}."
                )
            else:
                fold_sizes: dict[str, int] = {}
                for k in expected_folds:
                    val_fold_df = b_dev[b_dev[fold_col] == k]
                    train_fold_df = b_dev[b_dev[fold_col] != k]
                    fold_sizes[str(k)] = int(len(val_fold_df))
                    val_floors = sorted(int(f) for f in val_fold_df["floor"].unique())
                    train_floors = sorted(int(f) for f in train_fold_df["floor"].unique())
                    if train_floors != all_floors:
                        errors.append(
                            f"Building {b_id} {fold_col}={k} training partition is missing floors: "
                            f"expected {all_floors}, got {train_floors}."
                        )
                    if val_floors != all_floors:
                        errors.append(
                            f"Building {b_id} {fold_col}={k} validation partition is missing floors: "
                            f"expected {all_floors}, got {val_floors}."
                        )
                fold_summaries[fold_col] = fold_sizes

        per_building_counts[str(b_id)] = {
            "total_rows": int(len(b_df)),
            "dev_train_rows": int(len(b_dev)),
            "internal_holdout_rows": int(len(b_hold)),
            "floors": all_floors,
            "dev_train_floor_support": {
                str(int(f)): int(c) for f, c in b_dev["floor"].value_counts().sort_index().items()
            },
            "internal_holdout_floor_support": {
                str(int(f)): int(c) for f, c in b_hold["floor"].value_counts().sort_index().items()
            },
            "fold_validation_sizes": fold_summaries,
        }

    valid = len(errors) == 0
    if strict and not valid:
        raise SplitCreationError("; ".join(errors))

    return SplitValidationReport(
        valid=valid,
        manifest_sha256=manifest_sha256,
        total_rows=total_rows,
        dev_train_rows=int(dev_mask.sum()),
        internal_holdout_rows=int(holdout_mask.sum()),
        buildings=buildings,
        n_splits_checked=sorted(set(expected_n_splits_map.values())),
        per_building_counts=per_building_counts,
        errors=errors,
        warnings=warnings,
    )


def save_split_manifest(
    manifest_df: pd.DataFrame,
    manifest_csv_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    metadata_json_path: Path | str = DEFAULT_SPLIT_MANIFEST_META_JSON,
    holdout_fraction: float = DEFAULT_INTERNAL_HOLDOUT_FRACTION,
) -> tuple[str, SplitValidationReport]:
    """Validate and save the development split manifest CSV and its companion metadata JSON."""
    report = validate_split_manifest(manifest_df, strict=True)
    csv_path = Path(manifest_csv_path)
    meta_path = Path(metadata_json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df.to_csv(csv_path, index=False)
    file_sha256 = compute_file_sha256(csv_path)

    metadata = {
        "manifest_csv_path": str(csv_path),
        "split_manifest_hash": file_sha256,
        "dataframe_sha256": report.manifest_sha256,
        "source_sha256": str(manifest_df["source_sha256"].iloc[0]) if len(manifest_df) > 0 else "",
        "seed": int(manifest_df["seed"].iloc[0]) if len(manifest_df) > 0 else DEFAULT_RANDOM_SEED,
        "holdout_fraction": float(holdout_fraction),
        "validation_report": report.to_dict(),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return file_sha256, report


def load_split_manifest(
    manifest_csv_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    strict: bool = True,
) -> tuple[pd.DataFrame, str, SplitValidationReport]:
    """Load and validate a saved split manifest CSV, returning `(manifest_df, file_sha256, report)`."""
    path = Path(manifest_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found at: {path}")
    file_sha256 = compute_file_sha256(path)
    manifest_df = pd.read_csv(path)
    report = validate_split_manifest(manifest_df, strict=strict)
    return manifest_df, file_sha256, report


def align_dataset_with_manifest(
    train_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    expected_source_sha256: str | None = None,
) -> pd.DataFrame:
    """Verify row-for-row alignment between `train_df` and `manifest_df` and return merged DataFrame."""
    if len(train_df) != len(manifest_df):
        raise SplitCreationError(
            f"Row count mismatch between dataset ({len(train_df)}) and manifest ({len(manifest_df)})."
        )
    if not np.array_equal(manifest_df["source_row_index"].to_numpy(), np.arange(len(train_df))):
        raise SplitCreationError("Manifest source_row_index does not match dataset row order.")

    if expected_source_sha256 is not None:
        manifest_source_hashes = manifest_df["source_sha256"].astype(str).unique()
        if len(manifest_source_hashes) != 1 or manifest_source_hashes[0] != expected_source_sha256:
            raise SplitCreationError(
                f"Dataset SHA-256 ({expected_source_sha256}) does not match manifest source_sha256 "
                f"({manifest_source_hashes.tolist()})."
            )

    work_df = train_df.copy()
    if ROW_ID_COLUMN in work_df.columns:
        if not (work_df[ROW_ID_COLUMN].astype(str).to_numpy() == manifest_df[ROW_ID_COLUMN].astype(str).to_numpy()).all():
            raise SplitCreationError("row_id order in dataset does not match manifest row_id order.")
    else:
        work_df[ROW_ID_COLUMN] = manifest_df[ROW_ID_COLUMN].astype(str).to_numpy()

    if not (work_df[BUILDING_COLUMN].astype(int).to_numpy() == manifest_df["building_id"].astype(int).to_numpy()).all():
        raise SplitCreationError("BUILDINGID alignment mismatch between dataset and split manifest.")
    if not (work_df[TARGET_COLUMN].astype(int).to_numpy() == manifest_df["floor"].astype(int).to_numpy()).all():
        raise SplitCreationError("FLOOR alignment mismatch between dataset and split manifest.")

    for col in ("split_role", "fold", "fold_3", "fold_5", "seed", "source_sha256"):
        work_df[col] = manifest_df[col].to_numpy()

    return work_df


def iter_building_cv_folds(
    manifest_df: pd.DataFrame,
    building_id: int,
    n_splits: int = DEFAULT_FINAL_CV_FOLDS,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(fold_id, train_row_indices, val_row_indices)` for a building's `dev_train` partition.

    Indices returned are 0-based `source_row_index` values into the original training DataFrame.
    """
    if int(n_splits) == 3 and "fold_3" in manifest_df.columns:
        fold_col = "fold_3"
    elif int(n_splits) == 5 and "fold_5" in manifest_df.columns:
        fold_col = "fold_5"
    else:
        fold_col = "fold"

    b_dev = manifest_df[
        (manifest_df["building_id"] == int(building_id))
        & (manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN)
    ]
    if b_dev.empty:
        raise SplitCreationError(f"No dev_train rows found in manifest for building_id={building_id}.")

    unique_folds = sorted(int(k) for k in b_dev[fold_col].unique())
    if len(unique_folds) != int(n_splits):
        raise SplitCreationError(
            f"Requested {n_splits} folds for building {building_id}, but manifest column '{fold_col}' "
            f"contains {unique_folds}."
        )

    for fold_id in unique_folds:
        train_idx = b_dev.loc[b_dev[fold_col] != fold_id, "source_row_index"].to_numpy(dtype=int)
        val_idx = b_dev.loc[b_dev[fold_col] == fold_id, "source_row_index"].to_numpy(dtype=int)
        yield int(fold_id), train_idx, val_idx


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to generate and validate shared development splits from trainingData.csv."""
    parser = argparse.ArgumentParser(
        description="Generate seeded, stratified per-building development splits and CV folds from trainingData.csv."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to official trainingData.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_CSV,
        help="Path to save development split manifest CSV",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_META_JSON,
        help="Path to save split manifest metadata JSON",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_INTERNAL_HOLDOUT_FRACTION,
        help="Per-building stratified internal holdout fraction (default: 0.20)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_FINAL_CV_FOLDS,
        help="Primary CV fold count for final development selection (default: 5)",
    )
    parser.add_argument(
        "--lightweight-cv-folds",
        type=int,
        default=DEFAULT_LIGHTWEIGHT_CV_FOLDS,
        help="Lightweight CV fold count for fast exploration runs (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args(argv)

    train_df, audit_report = load_and_audit_csv(
        args.train_csv,
        dataset_role="train",
        strict=True,
        attach_row_ids=True,
    )
    manifest_df = create_development_splits(
        train_df=train_df,
        source_sha256=audit_report.file_sha256,
        holdout_fraction=args.holdout_fraction,
        primary_cv_folds=args.cv_folds,
        lightweight_cv_folds=args.lightweight_cv_folds,
        seed=args.seed,
        source_name=args.train_csv.name,
    )
    manifest_hash, report = save_split_manifest(
        manifest_df=manifest_df,
        manifest_csv_path=args.output_csv,
        metadata_json_path=args.output_meta,
        holdout_fraction=args.holdout_fraction,
    )
    print(
        f"Shared split manifest created: {args.output_csv} "
        f"(rows={report.total_rows}, dev_train={report.dev_train_rows}, "
        f"internal_holdout={report.internal_holdout_rows}, sha256={manifest_hash})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
