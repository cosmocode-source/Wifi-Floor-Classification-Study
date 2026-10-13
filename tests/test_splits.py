"""Unit tests for shared.create_splits (disjointness, class coverage, validationData protection)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.config import SPLIT_ROLE_DEV_TRAIN, SPLIT_ROLE_INTERNAL_HOLDOUT
from shared.create_splits import (
    SplitCreationError,
    align_dataset_with_manifest,
    create_development_splits,
    iter_building_cv_folds,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)


def test_create_development_splits_disjointness_and_class_coverage(
    synthetic_uji_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Development splits must partition every building disjointly and preserve all floor classes."""
    manifest_df = create_development_splits(
        train_df=synthetic_uji_df,
        source_sha256="a" * 64,
        holdout_fraction=0.20,
        primary_cv_folds=5,
        lightweight_cv_folds=3,
        seed=42,
    )
    report = validate_split_manifest(manifest_df, strict=True)
    assert report.valid is True
    assert report.total_rows == len(synthetic_uji_df)
    assert report.dev_train_rows + report.internal_holdout_rows == len(synthetic_uji_df)

    # Verify disjointness of row_id sets
    dev_ids = set(manifest_df.loc[manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN, "row_id"])
    hold_ids = set(manifest_df.loc[manifest_df["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT, "row_id"])
    assert len(dev_ids & hold_ids) == 0

    # Verify CV fold iterators for both 3-fold and 5-fold modes
    for b_id in (0, 1, 2):
        b_dev_indices = set(
            manifest_df.loc[
                (manifest_df["building_id"] == b_id) & (manifest_df["split_role"] == SPLIT_ROLE_DEV_TRAIN),
                "source_row_index",
            ].tolist()
        )
        for n_splits in (3, 5):
            seen_val: set[int] = set()
            for fold_id, train_idx, val_idx in iter_building_cv_folds(manifest_df, building_id=b_id, n_splits=n_splits):
                assert 0 <= fold_id < n_splits
                train_set = set(train_idx.tolist())
                val_set = set(val_idx.tolist())
                assert len(train_set & val_set) == 0
                assert train_set | val_set == b_dev_indices
                assert len(seen_val & val_set) == 0
                seen_val |= val_set
            assert seen_val == b_dev_indices

    # Verify save, load, and alignment round-trip
    csv_path = tmp_path / "development_splits.csv"
    meta_path = tmp_path / "development_splits_metadata.json"
    saved_hash, _ = save_split_manifest(manifest_df, csv_path, meta_path)
    loaded_df, loaded_hash, loaded_report = load_split_manifest(csv_path, strict=True)
    assert saved_hash == loaded_hash
    assert loaded_report.valid is True

    aligned = align_dataset_with_manifest(
        synthetic_uji_df, loaded_df, expected_source_sha256="a" * 64
    )
    assert len(aligned) == len(synthetic_uji_df)
    assert "split_role" in aligned.columns


def test_create_development_splits_rejects_official_validation_file(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """Passing validationData.csv as source_name must be rejected immediately."""
    with pytest.raises(SplitCreationError, match="validationData.csv must never be used"):
        create_development_splits(
            train_df=synthetic_uji_df,
            source_name="validationData.csv",
        )


def test_validate_split_manifest_detects_corrupted_holdout_fold(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """If an internal_holdout row is assigned a CV fold >= 0, manifest validation must fail."""
    manifest_df = create_development_splits(synthetic_uji_df, seed=42)
    corrupted = manifest_df.copy()
    holdout_idx = corrupted.index[corrupted["split_role"] == SPLIT_ROLE_INTERNAL_HOLDOUT][0]
    corrupted.loc[holdout_idx, "fold"] = 0

    with pytest.raises(SplitCreationError, match="internal_holdout rows have non -1"):
        validate_split_manifest(corrupted, strict=True)


def test_split_manifest_dataset_hash_matching_rejects_mismatched_source_hash(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """align_dataset_with_manifest must verify source_sha256 and reject tampered or mismatched datasets."""
    expected_hash = "1234567890abcdef" * 4
    manifest_df = create_development_splits(
        train_df=synthetic_uji_df,
        source_sha256=expected_hash,
        seed=42,
    )

    # Matching hash succeeds
    aligned = align_dataset_with_manifest(
        synthetic_uji_df, manifest_df, expected_source_sha256=expected_hash
    )
    assert len(aligned) == len(synthetic_uji_df)

    # Mismatched dataset SHA-256 raises SplitCreationError
    wrong_hash = "fedcba0987654321" * 4
    with pytest.raises(SplitCreationError, match="Dataset SHA-256 .* does not match manifest source_sha256"):
        align_dataset_with_manifest(
            synthetic_uji_df, manifest_df, expected_source_sha256=wrong_hash
        )

    # Mismatched FLOOR labels between dataset and manifest raises SplitCreationError
    tampered_df = synthetic_uji_df.copy()
    tampered_df.loc[0, "FLOOR"] = int(tampered_df.loc[0, "FLOOR"]) + 1
    with pytest.raises(SplitCreationError, match="FLOOR alignment mismatch"):
        align_dataset_with_manifest(
            tampered_df, manifest_df, expected_source_sha256=expected_hash
        )


def test_final_holdout_protection_rejects_official_validation_in_development_splits(
    synthetic_uji_df: pd.DataFrame,
) -> None:
    """Official validation role or file names must never enter development split manifests."""
    for forbidden_name in ("validationData.csv", "data/raw/validationData.csv", "VALIDATIONDATA.CSV"):
        with pytest.raises(SplitCreationError, match="validationData.csv must never be used"):
            create_development_splits(synthetic_uji_df, source_name=forbidden_name)

    manifest_df = create_development_splits(synthetic_uji_df, seed=42)
    injected = manifest_df.copy()
    injected.loc[0, "split_role"] = "official_validation"
    with pytest.raises(SplitCreationError, match="Invalid split_role values in development manifest"):
        validate_split_manifest(injected, strict=True)
