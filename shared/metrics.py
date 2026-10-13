"""Evaluation metrics, pooled metric arithmetic, and model-selection rules."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from shared.config import POOLED_GLOBAL_FLOOR_LABELS, POOLED_MACRO_F1_LABEL_POLICY


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    'Compute required classification metrics, per-class supports, and original-label confusion matrix.'
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D sequences of integer floor labels.")
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.size} rows, y_pred has {y_pred_arr.size} rows."
        )
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set (0 rows).")

    if labels is None:
        ordered_labels = sorted(int(x) for x in (set(y_true_arr.tolist()) | set(y_pred_arr.tolist())))
    else:
        ordered_labels = [int(x) for x in labels]
        # Ensure any predicted/true label not in `labels` is caught
        missing_in_labels = (set(y_true_arr.tolist()) | set(y_pred_arr.tolist())) - set(ordered_labels)
        if missing_in_labels:
            ordered_labels = sorted(set(ordered_labels) | missing_in_labels)

    evaluated_row_count = int(y_true_arr.size)
    correct_count = int(np.sum(y_true_arr == y_pred_arr))
    incorrect_count = int(evaluated_row_count - correct_count)

    # Exact arithmetic from counts
    accuracy = float(correct_count / evaluated_row_count)
    error = float(incorrect_count / evaluated_row_count)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=ordered_labels,
        average=None,
        zero_division=0,
    )

    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=ordered_labels,
            average="macro",
            zero_division=0,
        )
    )

    # Compute balanced accuracy across classes present in y_true
    true_present_labels = sorted(int(x) for x in set(y_true_arr.tolist()))
    if len(true_present_labels) >= 1:
        balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    else:
        balanced_acc = 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ordered_labels)

    per_class: dict[str, dict[str, float | int]] = {}
    supports_dict: dict[str, int] = {}
    for idx, lbl in enumerate(ordered_labels):
        lbl_key = str(int(lbl))
        sup_val = int(supports[idx])
        supports_dict[lbl_key] = sup_val
        per_class[lbl_key] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "support": sup_val,
        }

    # Sanity check against sklearn accuracy_score
    assert np.isclose(accuracy, accuracy_score(y_true_arr, y_pred_arr), atol=1e-12)
    assert correct_count + incorrect_count == evaluated_row_count

    return {
        "accuracy": accuracy,
        "error": error,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "supports": supports_dict,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "evaluated_row_count": evaluated_row_count,
        "confusion_matrix": {
            "labels": [int(x) for x in ordered_labels],
            "matrix": [[int(cell) for cell in row] for row in cm.tolist()],
        },
    }


def compute_pooled_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    building_ids: Sequence[int] | np.ndarray | pd.Series | None = None,
    per_building_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    'Compute pooled metrics across buildings from pooled row-level predictions.'
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if fixed_labels is not None:
        labels_to_use = sorted(set(int(x) for x in fixed_labels) | set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    else:
        present = set(y_true_arr.tolist()) | set(y_pred_arr.tolist())
        if present == set(POOLED_GLOBAL_FLOOR_LABELS):
            labels_to_use = list(POOLED_GLOBAL_FLOOR_LABELS)
        else:
            labels_to_use = sorted(int(x) for x in present)

    pooled = compute_classification_metrics(y_true_arr, y_pred_arr, labels=labels_to_use)
    pooled["label_policy"] = POOLED_MACRO_F1_LABEL_POLICY

    resolved_per_building: dict[str, Mapping[str, Any]] = {}
    if per_building_metrics is not None:
        resolved_per_building = dict(per_building_metrics)
    elif building_ids is not None:
        b_arr = np.asarray(building_ids, dtype=int)
        if b_arr.size != y_true_arr.size:
            raise ValueError("building_ids length must match y_true length.")
        for b_id in sorted(int(x) for x in set(b_arr.tolist())):
            mask = b_arr == b_id
            resolved_per_building[str(b_id)] = compute_classification_metrics(
                y_true_arr[mask], y_pred_arr[mask]
            )

    if resolved_per_building:
        sum_bldg_correct = sum(int(m["correct_count"]) for m in resolved_per_building.values())
        sum_bldg_total = sum(int(m["evaluated_row_count"]) for m in resolved_per_building.values())
        if sum_bldg_total != pooled["evaluated_row_count"] or sum_bldg_correct != pooled["correct_count"]:
            raise ValueError(
                f"Pooled count mismatch vs sum of buildings: pooled=({pooled['correct_count']}/{pooled['evaluated_row_count']}), "
                f"sum_buildings=({sum_bldg_correct}/{sum_bldg_total})."
            )

        bldg_accs = [float(m["accuracy"]) for m in resolved_per_building.values()]
        bldg_f1s = [float(m["macro_f1"]) for m in resolved_per_building.values()]
        bldg_weights = [int(m["evaluated_row_count"]) for m in resolved_per_building.values()]

        pooled["unweighted_building_mean_accuracy"] = float(np.mean(bldg_accs))
        pooled["unweighted_building_mean_macro_f1"] = float(np.mean(bldg_f1s))
        pooled["support_weighted_building_macro_f1"] = float(
            np.average(bldg_f1s, weights=bldg_weights)
        )

    return pooled


def evaluate_predictions_by_building(
    predictions_df: pd.DataFrame,
    fixed_pooled_labels: Sequence[int] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute `per_building`, `pooled`, `supports`, and `confusion_matrices` dicts from a predictions DataFrame."""
    for col in ("building_id", "true_floor", "predicted_floor"):
        if col not in predictions_df.columns:
            raise ValueError(f"predictions_df is missing required column '{col}'.")

    if predictions_df.empty:
        raise ValueError("Cannot evaluate an empty predictions_df.")

    per_building: dict[str, dict[str, Any]] = {}
    supports: dict[str, Any] = {"per_building": {}, "pooled": {}}
    confusion_matrices: dict[str, Any] = {}

    buildings = sorted(int(b) for b in predictions_df["building_id"].unique())
    for b_id in buildings:
        b_sub = predictions_df[predictions_df["building_id"].astype(int) == b_id]
        b_key = str(b_id)
        b_metrics = compute_classification_metrics(
            b_sub["true_floor"].to_numpy(dtype=int),
            b_sub["predicted_floor"].to_numpy(dtype=int),
        )
        per_building[b_key] = b_metrics
        supports["per_building"][b_key] = {
            "evaluated_row_count": b_metrics["evaluated_row_count"],
            "class_supports": b_metrics["supports"],
        }
        confusion_matrices[b_key] = b_metrics["confusion_matrix"]

    pooled = compute_pooled_metrics(
        y_true=predictions_df["true_floor"].to_numpy(dtype=int),
        y_pred=predictions_df["predicted_floor"].to_numpy(dtype=int),
        per_building_metrics=per_building,
        fixed_labels=fixed_pooled_labels,
    )
    supports["pooled"] = {
        "evaluated_row_count": pooled["evaluated_row_count"],
        "class_supports": pooled["supports"],
    }
    confusion_matrices["pooled"] = pooled["confusion_matrix"]

    return per_building, pooled, supports, confusion_matrices


def select_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    primary_key: str = "mean_cv_accuracy",
    tie_breaker_key: str = "mean_cv_macro_f1",
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    'Select the best feasible candidate using CV accuracy with a declared macro-F1 tie-breaker.'
    feasible_candidates = [
        dict(c)
        for c in candidates
        if c.get("feasible", True) and c.get("status", "completed") == "completed"
    ]
    if not feasible_candidates:
        raise ValueError("No feasible completed candidates available for model selection.")

    for c in feasible_candidates:
        if primary_key not in c or c[primary_key] is None:
            raise ValueError(f"Candidate is missing primary selection metric '{primary_key}': {c}")
        if tie_breaker_key not in c or c[tie_breaker_key] is None:
            raise ValueError(f"Candidate is missing tie-breaker metric '{tie_breaker_key}': {c}")
        if not all(np.isfinite(float(c[key])) for key in (primary_key, tie_breaker_key)):
            raise ValueError("Candidate selection metrics must be finite.")

    best = feasible_candidates[0]
    for cand in feasible_candidates[1:]:
        p_diff = float(cand[primary_key]) - float(best[primary_key])
        if p_diff > tolerance:
            best = cand
        elif abs(p_diff) <= tolerance:
            t_diff = float(cand[tie_breaker_key]) - float(best[tie_breaker_key])
            if t_diff > tolerance:
                best = cand

    return best
