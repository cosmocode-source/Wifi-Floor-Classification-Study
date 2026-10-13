"""Result schema v1.0 contract, prediction CSV validation, and environment metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from shared.config import (
    EVALUATION_ROLE_CV_VAL,
    RESULT_SCHEMA_VERSION,
    VALID_EVALUATION_ROLES,
    VALID_MODEL_IDS,
    VALID_RUN_STATUSES,
)
from shared.data_validation import DataValidationError, validate_integer_labels


class ResultSchemaError(ValueError):
    """Raised when a result dictionary or predictions DataFrame violates schema_version='1.0'."""


REQUIRED_RESULT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "model_id",
    "status",
    "dataset_hash",
    "split_manifest_hash",
    "evaluation_role",
    "preprocessing",
    "selected_parameters",
    "selection_metric",
    "per_building",
    "pooled",
    "supports",
    "confusion_matrices",
    "timing",
    "prediction_file_path",
    "model_artifact_paths",
    "assumptions",
    "warnings",
    "environment",
)

REQUIRED_METRIC_BLOCK_FIELDS: tuple[str, ...] = (
    "accuracy",
    "error",
    "macro_f1",
    "balanced_accuracy",
    "per_class",
    "incorrect_count",
    "evaluated_row_count",
)

REQUIRED_PER_CLASS_FIELDS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "support",
)

BASE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "row_id",
    "building_id",
    "true_floor",
    "predicted_floor",
    "evaluation_role",
    "model_id",
)

OOF_PREDICTION_EXTRA_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "repeat_id",
)


def collect_environment_info() -> dict[str, Any]:
    """Capture runtime Python, OS, and core library versions."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "execution_policy": "cpu_only_no_nested_parallelism",
    }


def validate_predictions_dataframe(
    df: pd.DataFrame,
    evaluation_role: str | None = None,
) -> None:
    """Validate that a predictions DataFrame satisfies the canonical predictions.csv contract."""
    missing_base = [c for c in BASE_PREDICTION_COLUMNS if c not in df.columns]
    if missing_base:
        raise ResultSchemaError(f"predictions DataFrame is missing required columns: {missing_base}")

    if df.empty:
        raise ResultSchemaError("predictions DataFrame must not be empty for a completed evaluation.")

    if df[list(BASE_PREDICTION_COLUMNS)].isna().any().any():
        raise ResultSchemaError("predictions DataFrame contains null values in required columns.")

    if df["row_id"].duplicated().any():
        dup_cnt = int(df["row_id"].duplicated().sum())
        raise ResultSchemaError(f"predictions DataFrame contains {dup_cnt} duplicate row_id entries.")

    roles_present = set(df["evaluation_role"].astype(str).unique())
    invalid_roles = roles_present - set(VALID_EVALUATION_ROLES)
    if invalid_roles:
        raise ResultSchemaError(f"Invalid evaluation_role values in predictions DataFrame: {sorted(invalid_roles)}")

    if len(roles_present) > 1:
        raise ResultSchemaError(
            f"predictions DataFrame mixes multiple evaluation roles in one file: {sorted(roles_present)}"
        )
    for column in ("building_id", "true_floor", "predicted_floor"):
        try:
            validate_integer_labels(df[column], column)
        except DataValidationError as exc:
            raise ResultSchemaError(str(exc)) from exc

    effective_role = evaluation_role or next(iter(roles_present))
    if evaluation_role is not None and roles_present != {evaluation_role}:
        raise ResultSchemaError(
            f"predictions DataFrame role {sorted(roles_present)} does not match expected '{evaluation_role}'."
        )

    models_present = set(df["model_id"].astype(str).unique())
    invalid_models = models_present - set(VALID_MODEL_IDS)
    if invalid_models:
        raise ResultSchemaError(f"Invalid model_id values in predictions DataFrame: {sorted(invalid_models)}")
    if len(models_present) != 1:
        raise ResultSchemaError("One predictions file must contain exactly one model_id.")

    if effective_role == EVALUATION_ROLE_CV_VAL:
        missing_oof = [c for c in OOF_PREDICTION_EXTRA_COLUMNS if c not in df.columns]
        if missing_oof:
            raise ResultSchemaError(
                f"Out-of-fold (cv_val) predictions must include fold/repeat columns {OOF_PREDICTION_EXTRA_COLUMNS}; "
                f"missing: {missing_oof}"
            )
        if df[list(OOF_PREDICTION_EXTRA_COLUMNS)].isna().any().any():
            raise ResultSchemaError("Out-of-fold predictions contain null values in fold_id or repeat_id.")
        for column in OOF_PREDICTION_EXTRA_COLUMNS:
            try:
                validate_integer_labels(df[column], column)
            except DataValidationError as exc:
                raise ResultSchemaError(str(exc)) from exc


def _validate_metric_block(block: Mapping[str, Any], context_name: str) -> None:
    missing = [k for k in REQUIRED_METRIC_BLOCK_FIELDS if k not in block]
    if missing:
        raise ResultSchemaError(f"Metric block '{context_name}' is missing required keys: {missing}")

    for k in ("accuracy", "error", "macro_f1", "balanced_accuracy"):
        val = block[k]
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise ResultSchemaError(f"Metric '{context_name}.{k}' must be a finite float, got {val!r}.")
        if not np.isfinite(float(val)) or float(val) < -1e-9 or float(val) > 1.0 + 1e-9:
            raise ResultSchemaError(f"Metric '{context_name}.{k}' is out of [0, 1] bounds: {val}")

    eval_rows = block["evaluated_row_count"]
    inc_rows = block["incorrect_count"]
    if not isinstance(eval_rows, (int, np.integer)) or int(eval_rows) <= 0:
        raise ResultSchemaError(
            f"'{context_name}.evaluated_row_count' must be a positive integer, got {eval_rows!r}."
        )
    if not isinstance(inc_rows, (int, np.integer)) or int(inc_rows) < 0 or int(inc_rows) > int(eval_rows):
        raise ResultSchemaError(
            f"'{context_name}.incorrect_count' ({inc_rows}) invalid for evaluated_row_count ({eval_rows})."
        )

    expected_err = float(inc_rows) / float(eval_rows)
    expected_acc = float(eval_rows - inc_rows) / float(eval_rows)
    if not np.isclose(float(block["error"]), expected_err, atol=1e-7, rtol=0):
        raise ResultSchemaError(
            f"'{context_name}.error' ({block['error']}) does not match incorrect_count/evaluated_row_count ({expected_err})."
        )
    if not np.isclose(float(block["accuracy"]), expected_acc, atol=1e-7, rtol=0):
        raise ResultSchemaError(
            f"'{context_name}.accuracy' ({block['accuracy']}) does not match 1 - error ({expected_acc})."
        )

    per_class = block["per_class"]
    if not isinstance(per_class, Mapping) or len(per_class) == 0:
        raise ResultSchemaError(f"'{context_name}.per_class' must be a non-empty dictionary.")

    total_support = 0
    for cls_label, cls_metrics in per_class.items():
        if not isinstance(cls_metrics, Mapping):
            raise ResultSchemaError(f"'{context_name}.per_class[{cls_label}]' must be a dictionary.")
        missing_cls = [f for f in REQUIRED_PER_CLASS_FIELDS if f not in cls_metrics]
        if missing_cls:
            raise ResultSchemaError(
                f"'{context_name}.per_class[{cls_label}]' missing keys: {missing_cls}"
            )
        support = cls_metrics["support"]
        if isinstance(support, bool) or not isinstance(support, (int, np.integer)) or support < 0:
            raise ResultSchemaError(f"'{context_name}.per_class[{cls_label}].support' must be a nonnegative integer.")
        for metric in ("precision", "recall", "f1"):
            value = cls_metrics[metric]
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ResultSchemaError(f"Invalid per-class {metric} for class {cls_label}.")
        total_support += int(support)

    if total_support != int(eval_rows):
        raise ResultSchemaError(
            f"Sum of per-class supports ({total_support}) in '{context_name}' does not equal "
            f"evaluated_row_count ({eval_rows})."
        )


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    """Validate a result dictionary against schema_version='1.0'."""
    missing_top = [k for k in REQUIRED_RESULT_FIELDS if k not in payload]
    if missing_top:
        raise ResultSchemaError(f"Result payload is missing required top-level fields: {missing_top}")

    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultSchemaError(
            f"Unsupported schema_version={payload['schema_version']!r}; expected {RESULT_SCHEMA_VERSION!r}."
        )

    if not str(payload["run_id"]).strip():
        raise ResultSchemaError("run_id must be a non-empty string.")

    if payload["model_id"] not in VALID_MODEL_IDS:
        raise ResultSchemaError(
            f"Invalid model_id={payload['model_id']!r}; expected one of {VALID_MODEL_IDS}."
        )

    status = payload["status"]
    if status not in VALID_RUN_STATUSES:
        raise ResultSchemaError(
            f"Invalid status={status!r}; expected one of {VALID_RUN_STATUSES}."
        )

    if payload["evaluation_role"] not in VALID_EVALUATION_ROLES:
        raise ResultSchemaError(
            f"Invalid evaluation_role={payload['evaluation_role']!r}; expected one of {VALID_EVALUATION_ROLES}."
        )

    if status == "completed":
        if not str(payload["dataset_hash"]).strip() or not str(payload["split_manifest_hash"]).strip():
            raise ResultSchemaError("Completed runs must record non-empty dataset_hash and split_manifest_hash.")

        per_bldg = payload["per_building"]
        pooled = payload["pooled"]
        if not isinstance(per_bldg, Mapping) or len(per_bldg) == 0:
            raise ResultSchemaError("Completed runs must populate a non-empty 'per_building' metrics dictionary.")
        if not isinstance(pooled, Mapping) or len(pooled) == 0:
            raise ResultSchemaError("Completed runs must populate a non-empty 'pooled' metrics dictionary.")

        sum_eval = 0
        sum_incorrect = 0
        for b_key, b_metrics in per_bldg.items():
            if not isinstance(b_metrics, Mapping):
                raise ResultSchemaError(f"per_building['{b_key}'] must be a dictionary.")
            _validate_metric_block(b_metrics, context_name=f"per_building[{b_key}]")
            sum_eval += int(b_metrics["evaluated_row_count"])
            sum_incorrect += int(b_metrics["incorrect_count"])

        _validate_metric_block(pooled, context_name="pooled")

        # Enforce pooled accuracy arithmetic from pooled counts across buildings
        if int(pooled["evaluated_row_count"]) != sum_eval:
            raise ResultSchemaError(
                f"Pooled evaluated_row_count ({pooled['evaluated_row_count']}) does not equal sum of "
                f"per-building evaluated_row_count ({sum_eval})."
            )
        if int(pooled["incorrect_count"]) != sum_incorrect:
            raise ResultSchemaError(
                f"Pooled incorrect_count ({pooled['incorrect_count']}) does not equal sum of "
                f"per-building incorrect_count ({sum_incorrect})."
            )
        expected_pooled_acc = float(sum_eval - sum_incorrect) / float(sum_eval)
        if not np.isclose(float(pooled["accuracy"]), expected_pooled_acc, atol=1e-7, rtol=0):
            raise ResultSchemaError(
                f"Pooled accuracy ({pooled['accuracy']}) was not computed from pooled correct/total counts "
                f"({expected_pooled_acc})."
            )

        cms = payload["confusion_matrices"]
        if not isinstance(cms, Mapping) or "pooled" not in cms:
            raise ResultSchemaError("Completed runs must include 'confusion_matrices' with per-building and 'pooled' entries.")
        for cm_key, cm_obj in cms.items():
            if not isinstance(cm_obj, Mapping) or "labels" not in cm_obj or "matrix" not in cm_obj:
                raise ResultSchemaError(f"confusion_matrices['{cm_key}'] must contain 'labels' and 'matrix'.")
            labels = cm_obj["labels"]
            matrix = cm_obj["matrix"]
            if len(labels) == 0 or len(matrix) != len(labels):
                raise ResultSchemaError(f"confusion_matrices['{cm_key}'] dimensions do not match labels.")
            if any(not isinstance(row, (list, tuple)) or len(row) != len(labels) for row in matrix):
                raise ResultSchemaError(f"confusion_matrices['{cm_key}'] must be square.")
            counts = np.asarray(matrix)
            if (
                not np.issubdtype(counts.dtype, np.number)
                or not np.isfinite(counts).all()
                or (counts < 0).any()
                or not np.equal(counts, np.floor(counts)).all()
            ):
                raise ResultSchemaError(f"confusion_matrices['{cm_key}'] contains invalid counts.")
            metric_block = pooled if cm_key == "pooled" else per_bldg.get(cm_key)
            if metric_block is None:
                raise ResultSchemaError(f"Unexpected confusion-matrix key '{cm_key}'.")
            if (
                int(counts.sum()) != int(metric_block["evaluated_row_count"])
                or int(counts.sum() - np.trace(counts)) != int(metric_block["incorrect_count"])
            ):
                raise ResultSchemaError(f"confusion_matrices['{cm_key}'] disagrees with metric counts.")

        if not payload["prediction_file_path"]:
            raise ResultSchemaError("Completed runs must record a non-empty prediction_file_path.")
    else:
        # status in ("failed", "skipped", "unexecuted"): prohibit fabricated numerical metrics!
        if payload["per_building"] not in (None, {}) or payload["pooled"] not in (None, {}):
            raise ResultSchemaError(
                f"Run with status={status!r} must NOT contain fabricated numerical metrics in "
                "'per_building' or 'pooled' (expected None or empty dict)."
            )


def build_completed_result(
    run_id: str,
    model_id: str,
    dataset_hash: str,
    split_manifest_hash: str,
    evaluation_role: str,
    preprocessing: Mapping[str, Any],
    selected_parameters: Mapping[str, Any],
    selection_metric: Mapping[str, Any] | str,
    per_building: Mapping[str, Mapping[str, Any]],
    pooled: Mapping[str, Any],
    supports: Mapping[str, Any],
    confusion_matrices: Mapping[str, Any],
    timing: Mapping[str, Any],
    prediction_file_path: str | Path,
    model_artifact_paths: Mapping[str, str] | Sequence[str],
    assumptions: Sequence[str],
    warnings: Sequence[str] | None = None,
    environment: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate a schema_version='1.0' result dictionary for a completed run."""
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": str(run_id),
        "model_id": str(model_id),
        "status": "completed",
        "dataset_hash": str(dataset_hash),
        "split_manifest_hash": str(split_manifest_hash),
        "evaluation_role": str(evaluation_role),
        "preprocessing": dict(preprocessing),
        "selected_parameters": dict(selected_parameters),
        "selection_metric": selection_metric if isinstance(selection_metric, str) else dict(selection_metric),
        "per_building": {str(k): dict(v) for k, v in per_building.items()},
        "pooled": dict(pooled),
        "supports": dict(supports),
        "confusion_matrices": dict(confusion_matrices),
        "timing": dict(timing),
        "prediction_file_path": str(prediction_file_path),
        "model_artifact_paths": (
            {str(k): str(v) for k, v in model_artifact_paths.items()}
            if isinstance(model_artifact_paths, Mapping)
            else [str(p) for p in model_artifact_paths]
        ),
        "assumptions": [str(a) for a in assumptions],
        "warnings": [str(w) for w in (warnings or [])],
        "environment": dict(environment) if environment is not None else collect_environment_info(),
    }
    if extra_metadata:
        payload["extra_metadata"] = dict(extra_metadata)

    validate_result_payload(payload)
    return payload


def build_non_completed_result(
    run_id: str,
    model_id: str,
    status: str,
    evaluation_role: str,
    reason: str,
    dataset_hash: str = "",
    split_manifest_hash: str = "",
    preprocessing: Mapping[str, Any] | None = None,
    selected_parameters: Mapping[str, Any] | None = None,
    selection_metric: Mapping[str, Any] | str = "cv_accuracy_tiebreak_macro_f1",
    skip_reasons: Sequence[Mapping[str, Any]] | None = None,
    assumptions: Sequence[str] | None = None,
    warnings: Sequence[str] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate an explicit failed/skipped/unexecuted result record with NO fabricated metrics."""
    if status not in ("failed", "skipped", "unexecuted"):
        raise ValueError(f"status must be 'failed', 'skipped', or 'unexecuted', got {status!r}.")

    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": str(run_id),
        "model_id": str(model_id),
        "status": status,
        "status_reason": str(reason),
        "skip_reasons": [dict(r) for r in (skip_reasons or [])],
        "dataset_hash": str(dataset_hash),
        "split_manifest_hash": str(split_manifest_hash),
        "evaluation_role": str(evaluation_role),
        "preprocessing": dict(preprocessing or {}),
        "selected_parameters": dict(selected_parameters or {}),
        "selection_metric": selection_metric if isinstance(selection_metric, str) else dict(selection_metric),
        "per_building": None,
        "pooled": None,
        "supports": None,
        "confusion_matrices": None,
        "timing": {"fit_seconds": None, "predict_seconds": None, "total_seconds": None},
        "prediction_file_path": None,
        "model_artifact_paths": {},
        "assumptions": [str(a) for a in (assumptions or [])],
        "warnings": [str(w) for w in (warnings or [])],
        "environment": dict(environment) if environment is not None else collect_environment_info(),
    }
    validate_result_payload(payload)
    return payload


def save_result_json(payload: Mapping[str, Any], output_path: Path | str) -> Path:
    """Validate and write a result dictionary to disk as formatted JSON."""
    validate_result_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=False)
    return path


def load_result_json(input_path: Path | str, validate: bool = True) -> dict[str, Any]:
    """Load a result JSON file and optionally validate it against schema_version='1.0'."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if validate:
        validate_result_payload(data)
    return data


def save_predictions_csv(
    df: pd.DataFrame,
    output_path: Path | str,
    evaluation_role: str | None = None,
) -> Path:
    """Validate and save a predictions DataFrame to CSV."""
    validate_predictions_dataframe(df, evaluation_role=evaluation_role)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions_csv(
    input_path: Path | str,
    evaluation_role: str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load a predictions CSV file and optionally validate its schema."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions CSV file not found: {path}")
    df = pd.read_csv(path)
    if validate:
        validate_predictions_dataframe(df, evaluation_role=evaluation_role)
    return df
