"""Data validation, schema checking, stable row IDs, and diagnostic audits for UJIIndoorLoc."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from shared.config import (
    BUILDING_COLUMN,
    DEFAULT_DATA_AUDIT_JSON,
    DEFAULT_TRAINING_CSV,
    DEFAULT_VALIDATION_CSV,
    EXPECTED_BUILDINGS,
    EXPECTED_FLOORS_GLOBAL,
    EXPECTED_TOTAL_COLUMNS,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_VALIDATION_ROWS,
    FORBIDDEN_PREDICTOR_COLUMNS,
    MAX_OBSERVED_RSSI,
    METADATA_COLUMNS,
    MIN_OBSERVED_RSSI,
    NUM_WAP_FEATURES,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    UNDETECTED_SENTINEL,
    WAP_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when dataset schema, numeric types, or predictor leakage checks fail."""


@dataclass
class DatasetAuditReport:
    """Comprehensive audit and provenance summary for a UJIIndoorLoc CSV dataset."""

    file_path: str
    dataset_role: str
    file_sha256: str
    dataframe_sha256: str
    row_count: int
    column_count: int
    expected_row_count: int | None
    matches_expected_row_count: bool | None
    schema_valid: bool
    wap_order_exact: bool
    missing_wap_columns: list[str] = field(default_factory=list)
    duplicate_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    missing_required_metadata: list[str] = field(default_factory=list)
    numeric_validation_passed: bool = False
    non_numeric_columns: list[str] = field(default_factory=list)
    null_count_total: int = 0
    rssi_summary: dict[str, Any] = field(default_factory=dict)
    all_undetected_row_count: int = 0
    all_undetected_row_ids: list[str] = field(default_factory=list)
    duplicate_diagnostics: dict[str, Any] = field(default_factory=dict)
    building_floor_support: dict[str, dict[str, int]] = field(default_factory=dict)
    building_row_counts: dict[str, int] = field(default_factory=dict)
    unexpected_buildings: list[int] = field(default_factory=list)
    unexpected_floors: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit report to a JSON-compatible dictionary."""
        return asdict(self)


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256; file does not exist: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 digest of a DataFrame's values and column names."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def make_row_id(dataset_role: str, row_index: int, file_sha256_prefix: str | None = None) -> str:
    """Create a deterministic, human-readable row identifier tied to dataset role and optional hash prefix."""
    clean_role = dataset_role.strip().lower()
    if not clean_role:
        raise ValueError("dataset_role must be a non-empty string.")
    if row_index < 0:
        raise ValueError(f"row_index must be non-negative, got {row_index}.")
    if file_sha256_prefix:
        prefix = file_sha256_prefix[:8].lower()
        return f"{clean_role}_{prefix}_{row_index:06d}"
    return f"{clean_role}_{row_index:06d}"


def assign_stable_row_ids(
    df: pd.DataFrame,
    dataset_role: str = "train",
    file_sha256: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Attach stable row IDs to a DataFrame without altering or dropping any rows."""
    target_df = df if inplace else df.copy()
    prefix = file_sha256[:8] if file_sha256 else None
    row_ids = [make_row_id(dataset_role, idx, prefix) for idx in range(len(target_df))]
    target_df[ROW_ID_COLUMN] = row_ids
    return target_df


def detect_duplicate_column_names(columns: Sequence[str]) -> list[str]:
    """Return a list of duplicated column names in order of second appearance."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for col in columns:
        col_str = str(col)
        if col_str in seen and col_str not in duplicates:
            duplicates.append(col_str)
        seen.add(col_str)
    return duplicates


def validate_predictor_columns(columns: Sequence[str]) -> None:
    """Strictly verify that a candidate predictor column list equals WAP001..WAP520 in exact order.

    Raises DataValidationError if metadata columns (e.g. BUILDINGID, FLOOR, coordinates),
    duplicate features, missing WAP features, or misordered WAP features are present.
    """
    col_tuple = tuple(str(c) for c in columns)
    duplicates = detect_duplicate_column_names(col_tuple)
    if duplicates:
        raise DataValidationError(f"Duplicate predictor columns detected: {duplicates}")

    forbidden_found = [c for c in col_tuple if c in FORBIDDEN_PREDICTOR_COLUMNS]
    if forbidden_found:
        raise DataValidationError(
            f"Predictor leakage violation: non-WAP metadata/target columns found in predictors: {forbidden_found}"
        )

    missing_waps = [c for c in WAP_COLUMNS if c not in col_tuple]
    if missing_waps:
        raise DataValidationError(
            f"Missing {len(missing_waps)} required WAP predictor columns (first 5: {missing_waps[:5]})."
        )

    extra_cols = [c for c in col_tuple if c not in WAP_COLUMNS]
    if extra_cols:
        raise DataValidationError(
            f"Unexpected non-allowlisted columns in predictor matrix: {extra_cols}"
        )

    if col_tuple != WAP_COLUMNS:
        raise DataValidationError(
            "Predictor columns contain all 520 WAPs but are not in the exact canonical order WAP001..WAP520."
        )


def extract_predictors_and_targets(
    df: pd.DataFrame,
    require_row_id: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    """Extract leakage-safe predictor DataFrame X (WAP001..WAP520 only), y (FLOOR), building_id, and optional row_id.

    Never drops rows silently. Validates that WAP columns exist, are unique, and are numeric.
    """
    dup_cols = detect_duplicate_column_names(list(df.columns))
    if dup_cols:
        raise DataValidationError(f"DataFrame contains duplicate column names: {dup_cols}")

    missing_waps = [c for c in WAP_COLUMNS if c not in df.columns]
    if missing_waps:
        raise DataValidationError(
            f"Cannot extract predictors; missing {len(missing_waps)} WAP columns (first 5: {missing_waps[:5]})."
        )

    for required_meta in (BUILDING_COLUMN, TARGET_COLUMN):
        if required_meta not in df.columns:
            raise DataValidationError(f"Missing required column '{required_meta}' in DataFrame.")

    if require_row_id and ROW_ID_COLUMN not in df.columns:
        raise DataValidationError(f"Missing required '{ROW_ID_COLUMN}' column in DataFrame.")

    X = df.loc[:, list(WAP_COLUMNS)].copy()
    validate_predictor_columns(list(X.columns))
    validate_wap_numeric_and_ranges(X, strict=True)

    y = df[TARGET_COLUMN].copy()
    building = df[BUILDING_COLUMN].copy()

    if y.isna().any() or not pd.api.types.is_numeric_dtype(y):
        raise DataValidationError("Target column FLOOR must be numeric and non-null.")
    if building.isna().any() or not pd.api.types.is_numeric_dtype(building):
        raise DataValidationError("Column BUILDINGID must be numeric and non-null.")

    validate_integer_labels(y, TARGET_COLUMN, EXPECTED_FLOORS_GLOBAL)
    validate_integer_labels(building, BUILDING_COLUMN, EXPECTED_BUILDINGS)
    y = y.astype(int)
    building = building.astype(int)
    row_ids = df[ROW_ID_COLUMN].copy() if ROW_ID_COLUMN in df.columns else None

    assert len(X) == len(df), "Row count invariant violated during predictor extraction."
    return X, y, building, row_ids


def validate_integer_labels(
    values: pd.Series | Sequence[Any],
    name: str,
    allowed: Sequence[int] | None = None,
) -> None:
    """Reject non-finite/fractional labels rather than silently truncating them."""
    series = pd.Series(values)
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise DataValidationError(f"{name} must contain numeric integer labels.")
    array = series.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise DataValidationError(f"{name} must contain finite integer labels.")
    if allowed is not None and not set(array.tolist()).issubset(set(allowed)):
        raise DataValidationError(f"{name} contains labels outside {list(allowed)}.")


def verify_training_csv_provenance(
    csv_path: Path | str, dataframe: pd.DataFrame, expected_sha256: str
) -> None:
    """Verify current raw data content, not merely the hash written in the manifest."""
    if "source_sha256" in dataframe.columns:
        # Cleaned exports carry the original source digest; their own bytes differ.
        hashes = dataframe["source_sha256"].astype(str).unique()
        if len(hashes) == 1 and hashes[0] == expected_sha256:
            return
    elif expected_sha256 in (
        compute_file_sha256(csv_path),
        compute_dataframe_sha256(dataframe),
    ):
        return
    raise DataValidationError(
        "Current training CSV does not match frozen training dataset provenance."
    )


def validate_wap_numeric_and_ranges(
    wap_df: pd.DataFrame,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate that WAP columns are numeric, non-null, and contain only sentinel 100 or RSSI in [-104, 0].

    Distinguishes sentinel 100 (undetected access point) from genuine 0 dBm (strong observed signal).
    """
    non_numeric: list[str] = []
    for col in wap_df.columns:
        if not pd.api.types.is_numeric_dtype(wap_df[col]) or pd.api.types.is_bool_dtype(wap_df[col]):
            non_numeric.append(str(col))

    null_count = int(wap_df.isna().sum().sum())
    if non_numeric or null_count > 0:
        msg = f"Numeric validation failed: non_numeric={non_numeric[:5]}, null_count={null_count}"
        if strict:
            raise DataValidationError(msg)
        return {
            "numeric_valid": False,
            "non_numeric_columns": non_numeric,
            "null_count": null_count,
        }

    values = wap_df.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        if strict:
            raise DataValidationError("WAP matrix contains non-finite (inf/-inf) values.")
        return {
            "numeric_valid": False,
            "non_numeric_columns": [],
            "null_count": null_count,
            "infinite_values": True,
        }

    is_sentinel = values == UNDETECTED_SENTINEL
    is_genuine_zero = values == 0.0
    is_valid_observed = (values >= MIN_OBSERVED_RSSI) & (values <= MAX_OBSERVED_RSSI)
    is_valid = is_sentinel | is_valid_observed
    invalid_mask = ~is_valid

    out_of_range_count = int(np.sum(invalid_mask))
    out_of_range_examples: list[float] = []
    if out_of_range_count > 0:
        unique_invalid = np.unique(values[invalid_mask])
        out_of_range_examples = [float(v) for v in unique_invalid[:10]]
        if strict:
            raise DataValidationError(
                f"Found {out_of_range_count} WAP values outside allowed range "
                f"[{MIN_OBSERVED_RSSI}, {MAX_OBSERVED_RSSI}] U {{{UNDETECTED_SENTINEL}}}. "
                f"Examples: {out_of_range_examples}"
            )

    observed_values = values[~is_sentinel]
    min_observed = float(np.min(observed_values)) if observed_values.size > 0 else None
    max_observed = float(np.max(observed_values)) if observed_values.size > 0 else None

    return {
        "numeric_valid": True,
        "non_numeric_columns": [],
        "null_count": 0,
        "sentinel_100_count": int(np.sum(is_sentinel)),
        "genuine_zero_count": int(np.sum(is_genuine_zero)),
        "observed_signal_count": int(observed_values.size),
        "min_observed_rssi": min_observed,
        "max_observed_rssi": max_observed,
        "out_of_range_count": out_of_range_count,
        "out_of_range_examples": out_of_range_examples,
    }


def diagnose_duplicates_and_undetected(
    df: pd.DataFrame,
    row_ids: Sequence[str] | None = None,
) -> tuple[int, list[str], dict[str, Any]]:
    """Diagnose all-undetected scans and duplicate/conflicting WAP fingerprints without dropping any rows."""
    wap_df = df.loc[:, list(WAP_COLUMNS)]
    wap_values = wap_df.to_numpy(dtype=np.float64, copy=False)

    if row_ids is None:
        if ROW_ID_COLUMN in df.columns:
            resolved_ids = [str(x) for x in df[ROW_ID_COLUMN].tolist()]
        else:
            resolved_ids = [f"row_{i:06d}" for i in range(len(df))]
    else:
        resolved_ids = [str(x) for x in row_ids]

    all_undetected_mask = np.all(wap_values == UNDETECTED_SENTINEL, axis=1)
    all_undetected_indices = np.flatnonzero(all_undetected_mask)
    all_undetected_row_ids = [resolved_ids[int(i)] for i in all_undetected_indices]

    # Exact duplicate rows across all feature + metadata columns (excluding synthetic row_id)
    cols_for_exact_dup = [c for c in df.columns if c != ROW_ID_COLUMN]
    exact_duplicate_count = int(df.duplicated(subset=cols_for_exact_dup, keep="first").sum())

    # Duplicate WAP fingerprint vectors
    wap_hashes = pd.util.hash_pandas_object(wap_df, index=False)
    dup_wap_mask = wap_df.duplicated(keep=False)
    duplicate_wap_rows_total = int(dup_wap_mask.sum())
    duplicate_wap_extra_copies = int(wap_df.duplicated(keep="first").sum())

    same_bldg_same_floor_groups = 0
    conflicting_floor_groups = 0
    cross_building_groups = 0
    conflicting_floor_row_count = 0

    if duplicate_wap_rows_total > 0 and BUILDING_COLUMN in df.columns and TARGET_COLUMN in df.columns:
        dup_sub = pd.DataFrame(
            {
                "wap_hash": wap_hashes[dup_wap_mask],
                BUILDING_COLUMN: df.loc[dup_wap_mask, BUILDING_COLUMN],
                TARGET_COLUMN: df.loc[dup_wap_mask, TARGET_COLUMN],
            }
        )
        # Group by WAP vector hash and verify exact match via group inspection
        for _, grp in dup_sub.groupby("wap_hash", sort=False):
            if len(grp) < 2:
                continue
            n_buildings = grp[BUILDING_COLUMN].nunique()
            if n_buildings > 1:
                cross_building_groups += 1
            # A global vector can cross buildings AND conflict within one building.
            for _, building_group in grp.groupby(BUILDING_COLUMN, sort=False):
                if len(building_group) < 2:
                    continue
                if building_group[TARGET_COLUMN].nunique() > 1:
                    conflicting_floor_groups += 1
                    conflicting_floor_row_count += int(len(building_group))
                else:
                    same_bldg_same_floor_groups += 1

    diagnostics = {
        "exact_duplicate_rows": exact_duplicate_count,
        "duplicate_wap_rows_total": duplicate_wap_rows_total,
        "duplicate_wap_extra_copies": duplicate_wap_extra_copies,
        "same_building_same_floor_duplicate_groups": same_bldg_same_floor_groups,
        "conflicting_floor_duplicate_groups": conflicting_floor_groups,
        "conflicting_floor_row_count": conflicting_floor_row_count,
        "cross_building_duplicate_groups": cross_building_groups,
        "within_building_group_policy": "group_by_fingerprint_then_building",
    }
    return int(len(all_undetected_row_ids)), all_undetected_row_ids, diagnostics


def audit_dataframe(
    df: pd.DataFrame,
    dataset_role: str = "train",
    file_path: Path | str = "<in-memory>",
    file_sha256: str | None = None,
    expected_row_count: int | None = None,
    strict: bool = False,
) -> DatasetAuditReport:
    """Run full schema, numeric, sentinel, duplicate, and class-support audits on a DataFrame.

    Never deletes or modifies rows in `df`. If `strict=True`, raises `DataValidationError` when
    any schema or numeric error is detected.
    """
    warnings: list[str] = []
    errors: list[str] = []

    raw_columns = [str(c) for c in df.columns]
    analysis_columns = [c for c in raw_columns if c != ROW_ID_COLUMN]

    dup_columns = detect_duplicate_column_names(raw_columns)
    if dup_columns:
        errors.append(f"Duplicate column names found: {dup_columns}")

    missing_waps = [c for c in WAP_COLUMNS if c not in analysis_columns]
    if missing_waps:
        errors.append(f"Missing {len(missing_waps)} WAP columns (e.g. {missing_waps[:5]}).")

    required_meta = [BUILDING_COLUMN, TARGET_COLUMN]
    missing_req_meta = [c for c in required_meta if c not in analysis_columns]
    if missing_req_meta:
        errors.append(f"Missing required metadata columns: {missing_req_meta}")

    allowed_all = set(WAP_COLUMNS) | set(METADATA_COLUMNS)
    extra_cols = [c for c in analysis_columns if c not in allowed_all]
    if extra_cols:
        warnings.append(f"Extra columns present outside standard 529 UJIIndoorLoc schema: {extra_cols}")

    present_waps_in_order = [c for c in analysis_columns if c in WAP_COLUMNS]
    wap_order_exact = tuple(present_waps_in_order) == WAP_COLUMNS
    if not wap_order_exact and not missing_waps:
        errors.append("All 520 WAP columns exist but are not in canonical order WAP001..WAP520.")

    schema_valid = (
        len(dup_columns) == 0
        and len(missing_waps) == 0
        and len(missing_req_meta) == 0
        and wap_order_exact
    )

    df_sha256 = compute_dataframe_sha256(df)
    resolved_file_sha256 = file_sha256 or df_sha256

    row_count = int(len(df))
    col_count = int(len(analysis_columns))
    matches_expected = (row_count == expected_row_count) if expected_row_count is not None else None
    if expected_row_count is not None and not matches_expected:
        warnings.append(
            f"Row count ({row_count}) differs from expected public reference count ({expected_row_count})."
        )
    if col_count != EXPECTED_TOTAL_COLUMNS:
        warnings.append(
            f"Column count excluding row_id ({col_count}) differs from expected {EXPECTED_TOTAL_COLUMNS}."
        )

    numeric_passed = False
    non_numeric_cols: list[str] = []
    null_total = int(df.isna().sum().sum())
    rssi_summary: dict[str, Any] = {}
    all_undetected_count = 0
    all_undetected_ids: list[str] = []
    dup_diagnostics: dict[str, Any] = {}
    bldg_floor_support: dict[str, dict[str, int]] = {}
    bldg_row_counts: dict[str, int] = {}
    unexpected_buildings: list[int] = []
    unexpected_floors: list[int] = []

    if not dup_columns and not missing_waps:
        wap_summary = validate_wap_numeric_and_ranges(df.loc[:, list(WAP_COLUMNS)], strict=False)
        rssi_summary = wap_summary
        non_numeric_cols.extend(wap_summary.get("non_numeric_columns", []))

        for meta_col in required_meta:
            if meta_col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[meta_col]) or pd.api.types.is_bool_dtype(df[meta_col]):
                    non_numeric_cols.append(meta_col)
                else:
                    allowed = EXPECTED_BUILDINGS if meta_col == BUILDING_COLUMN else EXPECTED_FLOORS_GLOBAL
                    try:
                        validate_integer_labels(df[meta_col], meta_col, allowed)
                    except DataValidationError as exc:
                        errors.append(str(exc))
                        non_numeric_cols.append(meta_col)

        numeric_passed = (
            bool(wap_summary.get("numeric_valid", False))
            and len(non_numeric_cols) == 0
            and null_total == 0
            and wap_summary.get("out_of_range_count", 0) == 0
        )
        if not numeric_passed:
            errors.append(
                f"Numeric/range validation failed: non_numeric={non_numeric_cols}, "
                f"nulls={null_total}, out_of_range={wap_summary.get('out_of_range_count', 0)}"
            )

        if numeric_passed:
            temp_ids = (
                [str(x) for x in df[ROW_ID_COLUMN].tolist()]
                if ROW_ID_COLUMN in df.columns
                else [make_row_id(dataset_role, i, resolved_file_sha256[:8]) for i in range(len(df))]
            )
            all_undetected_count, all_undetected_ids, dup_diagnostics = diagnose_duplicates_and_undetected(
                df, row_ids=temp_ids
            )
            if all_undetected_count > 0:
                warnings.append(
                    f"Found {all_undetected_count} scans where all {NUM_WAP_FEATURES} WAPs equal sentinel "
                    f"{UNDETECTED_SENTINEL} (all undetected). Rows were retained (no silent deletion)."
                )
            if dup_diagnostics.get("conflicting_floor_duplicate_groups", 0) > 0:
                warnings.append(
                    f"Found {dup_diagnostics['conflicting_floor_duplicate_groups']} duplicate WAP fingerprint "
                    f"groups with conflicting FLOOR labels inside the same building."
                )

            if len(missing_req_meta) == 0:
                bldg_series = df[BUILDING_COLUMN].astype(int)
                floor_series = df[TARGET_COLUMN].astype(int)
                unexpected_buildings = sorted(set(bldg_series.unique()) - set(EXPECTED_BUILDINGS))
                unexpected_floors = sorted(set(floor_series.unique()) - set(EXPECTED_FLOORS_GLOBAL))
                if unexpected_buildings:
                    errors.append(f"Unexpected BUILDINGID values found: {unexpected_buildings}")
                if unexpected_floors:
                    errors.append(f"Unexpected FLOOR values found: {unexpected_floors}")

                for b_id in sorted(bldg_series.unique()):
                    b_mask = bldg_series == b_id
                    bldg_row_counts[str(int(b_id))] = int(b_mask.sum())
                    f_counts = floor_series[b_mask].value_counts().sort_index()
                    bldg_floor_support[str(int(b_id))] = {
                        str(int(fl)): int(cnt) for fl, cnt in f_counts.items()
                    }

    if strict and errors:
        raise DataValidationError("; ".join(errors))

    return DatasetAuditReport(
        file_path=str(file_path),
        dataset_role=dataset_role,
        file_sha256=resolved_file_sha256,
        dataframe_sha256=df_sha256,
        row_count=row_count,
        column_count=col_count,
        expected_row_count=expected_row_count,
        matches_expected_row_count=matches_expected,
        schema_valid=schema_valid and len(errors) == 0,
        wap_order_exact=wap_order_exact,
        missing_wap_columns=missing_waps,
        duplicate_columns=dup_columns,
        extra_columns=extra_cols,
        missing_required_metadata=missing_req_meta,
        numeric_validation_passed=numeric_passed,
        non_numeric_columns=non_numeric_cols,
        null_count_total=null_total,
        rssi_summary=rssi_summary,
        all_undetected_row_count=all_undetected_count,
        all_undetected_row_ids=all_undetected_ids,
        duplicate_diagnostics=dup_diagnostics,
        building_floor_support=bldg_floor_support,
        building_row_counts=bldg_row_counts,
        unexpected_buildings=unexpected_buildings,
        unexpected_floors=unexpected_floors,
        warnings=warnings,
        errors=errors,
    )


def load_and_audit_csv(
    csv_path: Path | str,
    dataset_role: str = "train",
    expected_row_count: int | None = None,
    strict: bool = True,
    attach_row_ids: bool = True,
) -> tuple[pd.DataFrame, DatasetAuditReport]:
    """Load an immutable UJIIndoorLoc CSV file, verify schema/values, attach stable row IDs, and return report."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found at: {path}")

    file_sha256 = compute_file_sha256(path)

    # Check raw header line for duplicate column names before pandas mangle_dupe_cols renames them
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().strip()
    raw_header_cols = [c.strip().strip('"').strip("'") for c in header_line.split(",")] if header_line else []
    header_dups = detect_duplicate_column_names(raw_header_cols)
    if header_dups and strict:
        raise DataValidationError(f"Duplicate column names in raw CSV header of {path}: {header_dups}")

    df = pd.read_csv(path)
    if attach_row_ids:
        df = assign_stable_row_ids(df, dataset_role=dataset_role, file_sha256=file_sha256, inplace=False)

    report = audit_dataframe(
        df=df,
        dataset_role=dataset_role,
        file_path=path,
        file_sha256=file_sha256,
        expected_row_count=expected_row_count,
        strict=strict,
    )
    if header_dups and header_dups != report.duplicate_columns:
        report.duplicate_columns = header_dups
        report.schema_valid = False
        report.errors.append(f"Duplicate raw CSV header columns: {header_dups}")

    return df, report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to audit trainingData.csv and validationData.csv and save a JSON provenance report."""
    parser = argparse.ArgumentParser(
        description="Audit UJIIndoorLoc raw CSV files without modifying them."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="Path to official trainingData.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_VALIDATION_CSV,
        help="Path to official validationData.csv (optional if auditing training only)",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip auditing validationData.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_DATA_AUDIT_JSON,
        help="Path to write JSON audit report",
    )
    args = parser.parse_args(argv)

    _, train_report = load_and_audit_csv(
        args.train_csv,
        dataset_role="train",
        expected_row_count=EXPECTED_TRAINING_ROWS,
        strict=True,
    )
    payload: dict[str, Any] = {"training": train_report.to_dict()}

    if not args.skip_val and args.val_csv.is_file():
        _, val_report = load_and_audit_csv(
            args.val_csv,
            dataset_role="val",
            expected_row_count=EXPECTED_VALIDATION_ROWS,
            strict=True,
        )
        payload["validation"] = val_report.to_dict()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"Data audit completed. Report written to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
