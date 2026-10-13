"""Feasibility checks for PCA dimensions, kNN neighbors, and stratified CV class supports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


class FeasibilityError(ValueError):
    """Raised when strict feasibility validation fails for a model or split configuration."""


@dataclass
class FeasibilityResult:
    """Structured outcome of a feasibility check, including machine-readable skip reasons."""

    feasible: bool
    reason_code: str | None = None
    message: str = "Configuration is feasible."
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize feasibility result to a JSON-compatible dictionary."""
        return asdict(self)


def compute_max_feasible_pca_components(n_features: int, min_train_rows: int) -> int:
    """Return the maximum valid PCA component count: min(current_feature_count, training_rows - 1)."""
    if n_features < 1 or min_train_rows < 2:
        return 0
    return int(min(int(n_features), int(min_train_rows) - 1))


def resolve_min_training_rows(
    train_row_counts: int | Sequence[int],
    n_splits: int | None = None,
) -> int:
    'Resolve the smallest training partition size relevant to fitting (including across CV folds).'
    if isinstance(train_row_counts, (int, np.integer)):
        total_rows = int(train_row_counts)
        if n_splits is not None and int(n_splits) >= 2:
            # In k-fold CV, the validation fold receives at most ceil(total_rows / n_splits) rows,
            # so the smallest training partition has total_rows - ceil(total_rows / n_splits) rows.
            max_val_fold = (total_rows + int(n_splits) - 1) // int(n_splits)
            return max(0, total_rows - max_val_fold)
        return total_rows

    counts = [int(c) for c in train_row_counts]
    if not counts:
        return 0
    return int(min(counts))


def check_pca_feasibility(
    n_components: int,
    n_features: int,
    train_row_counts: int | Sequence[int],
    n_splits: int | None = None,
    strict: bool = False,
) -> FeasibilityResult:
    """Validate that `n_components` does not exceed `min(n_features, min_train_rows - 1)`.

    Never silently clamps `n_components`. Rejects infeasible dimensions explicitly.
    """
    min_train_rows = resolve_min_training_rows(train_row_counts, n_splits=n_splits)
    max_feasible = compute_max_feasible_pca_components(n_features=n_features, min_train_rows=min_train_rows)

    details = {
        "requested_n_components": n_components,
        "n_features": int(n_features),
        "min_train_rows": int(min_train_rows),
        "max_feasible_components": int(max_feasible),
        "n_splits": int(n_splits) if n_splits is not None else None,
    }

    if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool) or int(n_components) < 1:
        res = FeasibilityResult(
            feasible=False,
            reason_code="PCA_COMPONENTS_INVALID",
            message=f"Requested PCA n_components={n_components!r} must be a positive integer >= 1.",
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    if min_train_rows < 2 or n_features < 1:
        res = FeasibilityResult(
            feasible=False,
            reason_code="EMPTY_OR_DEGENERATE_TRAINING_PARTITION",
            message=(
                f"Training partition too small for PCA: min_train_rows={min_train_rows}, n_features={n_features}."
            ),
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    if int(n_components) > max_feasible:
        res = FeasibilityResult(
            feasible=False,
            reason_code="PCA_COMPONENTS_EXCEED_MAX",
            message=(
                f"Requested PCA n_components={n_components} exceeds maximum feasible "
                f"min(n_features={n_features}, min_train_rows - 1={min_train_rows - 1}) = {max_feasible}. "
                "Silent clamping is prohibited."
            ),
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    return FeasibilityResult(
        feasible=True,
        reason_code=None,
        message=f"PCA n_components={n_components} is feasible (max={max_feasible}).",
        details=details,
    )


def check_knn_feasibility(
    n_neighbors: int,
    train_row_counts: int | Sequence[int],
    n_splits: int | None = None,
    strict: bool = False,
) -> FeasibilityResult:
    """Validate that `1 <= n_neighbors <= min_train_rows` across relevant training partitions."""
    min_train_rows = resolve_min_training_rows(train_row_counts, n_splits=n_splits)
    details = {
        "requested_n_neighbors": n_neighbors,
        "min_train_rows": int(min_train_rows),
        "n_splits": int(n_splits) if n_splits is not None else None,
    }

    if not isinstance(n_neighbors, (int, np.integer)) or isinstance(n_neighbors, bool) or int(n_neighbors) < 1:
        res = FeasibilityResult(
            feasible=False,
            reason_code="KNN_NEIGHBORS_INVALID",
            message=f"Requested kNN n_neighbors={n_neighbors!r} must be a positive integer >= 1.",
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    if int(n_neighbors) > min_train_rows:
        res = FeasibilityResult(
            feasible=False,
            reason_code="KNN_NEIGHBORS_EXCEED_TRAIN_ROWS",
            message=(
                f"Requested kNN n_neighbors={n_neighbors} exceeds smallest training partition "
                f"size ({min_train_rows} rows)."
            ),
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    return FeasibilityResult(
        feasible=True,
        reason_code=None,
        message=f"kNN n_neighbors={n_neighbors} is feasible (min_train_rows={min_train_rows}).",
        details=details,
    )


def check_class_support_feasibility(
    y: Sequence[int] | np.ndarray | pd.Series,
    n_splits: int,
    strict: bool = False,
) -> FeasibilityResult:
    """Check whether class counts in `y` support `n_splits` stratified folds without dropping rare floors."""
    if not isinstance(n_splits, (int, np.integer)) or int(n_splits) < 2:
        res = FeasibilityResult(
            feasible=False,
            reason_code="INVALID_CV_SPLITS",
            message=f"n_splits must be an integer >= 2, got {n_splits!r}.",
            details={"requested_n_splits": n_splits},
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    y_arr = np.asarray(y)
    if y_arr.size == 0:
        res = FeasibilityResult(
            feasible=False,
            reason_code="EMPTY_TARGET_ARRAY",
            message="Target array y is empty.",
            details={"requested_n_splits": int(n_splits), "class_supports": {}},
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    unique_classes, counts = np.unique(y_arr, return_counts=True)
    class_supports = {str(int(cls)): int(cnt) for cls, cnt in zip(unique_classes, counts)}

    if len(unique_classes) < 2:
        res = FeasibilityResult(
            feasible=False,
            reason_code="SINGLE_CLASS_ONLY",
            message=f"At least 2 floor classes are required for classification; found {list(unique_classes)}.",
            details={"requested_n_splits": int(n_splits), "class_supports": class_supports},
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    failing_classes = {
        str(int(cls)): int(cnt)
        for cls, cnt in zip(unique_classes, counts)
        if int(cnt) < int(n_splits)
    }
    details = {
        "requested_n_splits": int(n_splits),
        "class_supports": class_supports,
        "min_class_support": int(np.min(counts)),
        "failing_classes": failing_classes,
    }

    if failing_classes:
        res = FeasibilityResult(
            feasible=False,
            reason_code="INSUFFICIENT_CLASS_SUPPORT_FOR_CV",
            message=(
                f"Cannot create {n_splits} stratified folds: floor classes {failing_classes} have fewer "
                f"than {n_splits} samples. Rare floors are never silently removed."
            ),
            details=details,
        )
        if strict:
            raise FeasibilityError(res.message)
        return res

    return FeasibilityResult(
        feasible=True,
        reason_code=None,
        message=f"All {len(unique_classes)} floor classes have support >= {n_splits}.",
        details=details,
    )


def check_candidate_feasibility(
    y_train: Sequence[int] | np.ndarray | pd.Series,
    n_features: int,
    n_splits: int | None = None,
    n_components: int | None = None,
    n_neighbors: int | None = None,
    fold_train_sizes: Sequence[int] | None = None,
    strict: bool = False,
) -> FeasibilityResult:
    """Combined feasibility check for class supports, PCA components, and kNN neighbor count."""
    y_arr = np.asarray(y_train)
    if n_splits is not None:
        support_check = check_class_support_feasibility(y_arr, n_splits=n_splits, strict=strict)
        if not support_check.feasible:
            return support_check

    train_counts: int | Sequence[int] = (
        fold_train_sizes if fold_train_sizes is not None else int(y_arr.size)
    )
    effective_splits = None if fold_train_sizes is not None else n_splits

    if n_components is not None:
        pca_check = check_pca_feasibility(
            n_components=n_components,
            n_features=n_features,
            train_row_counts=train_counts,
            n_splits=effective_splits,
            strict=strict,
        )
        if not pca_check.feasible:
            return pca_check

    if n_neighbors is not None:
        knn_check = check_knn_feasibility(
            n_neighbors=n_neighbors,
            train_row_counts=train_counts,
            n_splits=effective_splits,
            strict=strict,
        )
        if not knn_check.feasible:
            return knn_check

    min_train_rows = resolve_min_training_rows(train_counts, n_splits=effective_splits)
    return FeasibilityResult(
        feasible=True,
        reason_code=None,
        message="Candidate configuration passed all feasibility checks.",
        details={
            "n_features": int(n_features),
            "min_train_rows": int(min_train_rows),
            "max_feasible_components": compute_max_feasible_pca_components(n_features, min_train_rows),
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "n_splits": n_splits,
        },
    )
