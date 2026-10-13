"""Unit tests for shared.feasibility (PCA dimension limits, kNN neighbor limits, stratified class supports)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.feasibility import (
    FeasibilityError,
    check_candidate_feasibility,
    check_class_support_feasibility,
    check_knn_feasibility,
    check_pca_feasibility,
    compute_max_feasible_pca_components,
)


def test_pca_feasibility_boundary_and_no_silent_clamping() -> None:
    """PCA components must not exceed min(n_features, min_train_rows - 1) and must never be clamped."""
    # 520 features, 30 training rows across 3 folds -> smallest training partition has 20 rows -> max = 19
    assert compute_max_feasible_pca_components(n_features=520, min_train_rows=20) == 19

    ok_res = check_pca_feasibility(n_components=19, n_features=520, train_row_counts=30, n_splits=3)
    assert ok_res.feasible is True
    assert ok_res.reason_code is None

    bad_res = check_pca_feasibility(n_components=20, n_features=520, train_row_counts=30, n_splits=3)
    assert bad_res.feasible is False
    assert bad_res.reason_code == "PCA_COMPONENTS_EXCEED_MAX"
    assert bad_res.details["requested_n_components"] == 20
    assert bad_res.details["max_feasible_components"] == 19

    with pytest.raises(FeasibilityError, match="Silent clamping is prohibited"):
        check_pca_feasibility(n_components=20, n_features=520, train_row_counts=30, n_splits=3, strict=True)


def test_pca_feasibility_with_explicit_fold_sizes() -> None:
    """When explicit CV fold training sizes are passed, the smallest fold governs feasibility."""
    fold_train_sizes = [100, 99, 85]  # smallest fold has 85 rows -> max PCA = min(520, 84) = 84
    assert check_pca_feasibility(84, n_features=520, train_row_counts=fold_train_sizes).feasible is True
    res = check_pca_feasibility(85, n_features=520, train_row_counts=fold_train_sizes)
    assert res.feasible is False
    assert res.reason_code == "PCA_COMPONENTS_EXCEED_MAX"


def test_knn_feasibility_rejects_k_greater_than_training_rows() -> None:
    """kNN neighbor count k must satisfy 1 <= k <= min_train_rows."""
    assert check_knn_feasibility(n_neighbors=15, train_row_counts=[15, 16, 16]).feasible is True

    res = check_knn_feasibility(n_neighbors=16, train_row_counts=[15, 16, 16])
    assert res.feasible is False
    assert res.reason_code == "KNN_NEIGHBORS_EXCEED_TRAIN_ROWS"

    invalid_k = check_knn_feasibility(n_neighbors=0, train_row_counts=50)
    assert invalid_k.feasible is False
    assert invalid_k.reason_code == "KNN_NEIGHBORS_INVALID"


def test_class_support_feasibility_preserves_rare_floors_and_reports_skip_reason() -> None:
    """Rare floors with support < n_splits must cause an explicit skip reason, never silent removal."""
    # Floor 4 has only 2 samples, which is feasible for 2 splits but infeasible for 5 splits
    y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 2, dtype=int)

    res_5 = check_class_support_feasibility(y, n_splits=5)
    assert res_5.feasible is False
    assert res_5.reason_code == "INSUFFICIENT_CLASS_SUPPORT_FOR_CV"
    assert res_5.details["failing_classes"] == {"4": 2}


def test_combined_candidate_feasibility_check() -> None:
    """Combined candidate check validates class support, PCA components, and kNN neighbors together."""
    y = np.array([0, 1, 2, 3] * 10, dtype=int)  # 40 rows, 5 splits -> 32 train rows per fold -> max PCA = 31
    ok = check_candidate_feasibility(
        y_train=y,
        n_features=520,
        n_splits=5,
        n_components=25,
        n_neighbors=10,
    )
    assert ok.feasible is True

    fail_pca = check_candidate_feasibility(
        y_train=y,
        n_features=520,
        n_splits=5,
        n_components=32,
        n_neighbors=10,
    )
    assert fail_pca.feasible is False
    assert fail_pca.reason_code == "PCA_COMPONENTS_EXCEED_MAX"
