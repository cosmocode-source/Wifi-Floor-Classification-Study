"""Unit tests for shared.data_validation (synthetic fixtures only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.config import UNDETECTED_SENTINEL, WAP_COLUMNS
from shared.data_validation import (
    DataValidationError,
    assign_stable_row_ids,
    audit_dataframe,
    extract_predictors_and_targets,
    load_and_audit_csv,
    validate_predictor_columns,
    validate_wap_numeric_and_ranges,
)


def test_validate_predictor_columns_accepts_exact_wap_order() -> None:
    """Exact ordered allowlist WAP001..WAP520 must pass."""
    validate_predictor_columns(list(WAP_COLUMNS))


@pytest.mark.parametrize(
    "forbidden_col",
    ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE", "SPACEID", "USERID", "PHONEID", "TIMESTAMP", "row_id"],
)
def test_validate_predictor_columns_rejects_metadata_leakage(forbidden_col: str) -> None:
    """Any metadata or target column entering predictors must raise DataValidationError."""
    leaked_cols = [*WAP_COLUMNS[:-1], forbidden_col]
    with pytest.raises(DataValidationError, match="leakage|Missing"):
        validate_predictor_columns(leaked_cols)


def test_validate_predictor_columns_rejects_misordered_or_duplicate_waps() -> None:
    """Misordered or duplicated WAP columns must be rejected."""
    misordered = [WAP_COLUMNS[1], WAP_COLUMNS[0], *WAP_COLUMNS[2:]]
    with pytest.raises(DataValidationError, match="canonical order"):
        validate_predictor_columns(misordered)

    duplicated = [WAP_COLUMNS[0], WAP_COLUMNS[0], *WAP_COLUMNS[2:]]
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_predictor_columns(duplicated)


def test_sentinel_100_distinguished_from_genuine_zero(synthetic_uji_df: pd.DataFrame) -> None:
    """Genuine 0 dBm must be counted separately from sentinel 100 and accepted as valid RSSI."""
    wap_df = synthetic_uji_df.loc[:, list(WAP_COLUMNS)].copy()
    wap_df.iloc[0, 0] = 0
    wap_df.iloc[0, 1] = 100
    wap_df.iloc[0, 2] = -104

    summary = validate_wap_numeric_and_ranges(wap_df, strict=True)
    assert summary["numeric_valid"] is True
    assert summary["genuine_zero_count"] >= 1
    assert summary["sentinel_100_count"] >= 1
    assert summary["min_observed_rssi"] >= -104.0
    assert summary["max_observed_rssi"] == 0.0


@pytest.mark.parametrize("bad_value", [-105, 1, 50, 99, 101])
def test_validate_wap_ranges_rejects_out_of_range_rssi(
    synthetic_uji_df: pd.DataFrame,
    bad_value: int,
) -> None:
    """Values outside [-104, 0] U {100} must raise DataValidationError in strict mode."""
    wap_df = synthetic_uji_df.loc[:, list(WAP_COLUMNS)].copy()
    wap_df.iloc[2, 5] = bad_value
    with pytest.raises(DataValidationError, match="outside allowed range"):
        validate_wap_numeric_and_ranges(wap_df, strict=True)


def test_audit_warns_on_all_undetected_and_conflicting_duplicates_without_row_deletion(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """All-undetected scans and conflicting floor duplicates must trigger warnings without dropping rows."""
    df = synthetic_uji_df.copy()
    initial_row_count = len(df)

    # Make row 10 an all-undetected scan (all 520 WAPs == 100)
    df.loc[10, list(WAP_COLUMNS)] = UNDETECTED_SENTINEL

    # Make row 20 have identical WAPs to row 0 in the same building, but a different FLOOR label
    df.loc[20, list(WAP_COLUMNS)] = df.loc[0, list(WAP_COLUMNS)].to_numpy()
    df.loc[20, "BUILDINGID"] = df.loc[0, "BUILDINGID"]
    df.loc[20, "FLOOR"] = int(df.loc[0, "FLOOR"]) + 1

    report = audit_dataframe(df, dataset_role="synthetic_train", strict=True)
    assert report.row_count == initial_row_count
    assert report.all_undetected_row_count >= 1
    assert report.duplicate_diagnostics["conflicting_floor_duplicate_groups"] >= 1
    assert any("all undetected" in w.lower() for w in report.warnings)
    assert any("conflicting floor" in w.lower() for w in report.warnings)

    X, y, bldg, _ = extract_predictors_and_targets(df)
    assert len(X) == initial_row_count
    assert len(y) == initial_row_count
    assert len(bldg) == initial_row_count


def test_load_and_audit_csv_roundtrip_and_stable_row_ids(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """CSV loader must compute file SHA-256, attach deterministic row_ids, and preserve immutability."""
    csv_path = tmp_path / "synthetic_trainingData.csv"
    synthetic_uji_df.to_csv(csv_path, index=False)

    df_loaded, report = load_and_audit_csv(
        csv_path,
        dataset_role="train",
        expected_row_count=len(synthetic_uji_df),
        strict=True,
        attach_row_ids=True,
    )
    assert report.schema_valid is True
    assert report.matches_expected_row_count is True
    assert len(report.file_sha256) == 64
    assert "row_id" in df_loaded.columns
    assert df_loaded["row_id"].nunique() == len(synthetic_uji_df)

    # Re-assigning with the same hash produces identical row IDs
    df_reassigned = assign_stable_row_ids(
        synthetic_uji_df, dataset_role="train", file_sha256=report.file_sha256
    )
    assert df_loaded["row_id"].tolist() == df_reassigned["row_id"].tolist()
