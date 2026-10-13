"""Safe serialization, unique runtime directory management, and integrity-verified pipeline loading."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import joblib
from sklearn.base import BaseEstimator

from shared.config import PROJECT_ROOT, VALID_MODEL_IDS, WAP_COLUMNS
from shared.data_validation import compute_file_sha256, validate_predictor_columns
from shared.result_schema import collect_environment_info


class UntrustedArtifactError(RuntimeError):
    """Raised when an artifact path, metadata sidecar, or SHA-256 digest fails trust verification."""


def generate_unique_run_id(model_id: str, evaluation_role: str) -> str:
    """Create a unique, sortable run identifier including UTC timestamp and short random suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_uid = uuid.uuid4().hex[:8]
    return f"{ts}_{model_id}_{evaluation_role}_{short_uid}"


def create_unique_run_dir(
    base_dir: Path | str,
    model_id: str,
    evaluation_role: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """Create a new runtime output directory with `exist_ok=False` to prevent accidental overwriting."""
    resolved_run_id = run_id or generate_unique_run_id(model_id=model_id, evaluation_role=evaluation_role)
    out_dir = Path(base_dir) / resolved_run_id
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing runtime output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    return resolved_run_id, out_dir


def save_pipeline_artifact(
    pipeline: BaseEstimator,
    output_dir: Path | str,
    building_id: int,
    model_id: str,
    feature_names: Sequence[str] = WAP_COLUMNS,
    class_labels: Sequence[int] = (),
    dataset_hash: str = "",
    split_manifest_hash: str = "",
    preprocessing_config: Mapping[str, Any] | None = None,
    selected_parameters: Mapping[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Save a fitted preprocessing+estimator pipeline along with a cryptographic metadata sidecar.

    Never overwrites an existing `.joblib` or `.json` file in `output_dir`.
    """
    if model_id not in VALID_MODEL_IDS:
        raise ValueError(f"Invalid model_id={model_id!r}; expected one of {VALID_MODEL_IDS}.")

    validate_predictor_columns(list(feature_names))

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_filename = f"{model_id}_building_{int(building_id)}_pipeline.joblib"
    metadata_filename = f"{model_id}_building_{int(building_id)}_pipeline.metadata.json"

    artifact_path = target_dir / artifact_filename
    metadata_path = target_dir / metadata_filename

    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing serialized model artifact: {artifact_path}"
        )

    joblib.dump(pipeline, artifact_path)
    artifact_sha256 = compute_file_sha256(artifact_path)

    metadata = {
        "artifact_filename": artifact_filename,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "run_id": str(run_id),
        "model_id": str(model_id),
        "building_id": int(building_id),
        "target_column": "FLOOR",
        "building_selector_column": "BUILDINGID",
        "feature_names": [str(f) for f in feature_names],
        "class_labels": [int(c) for c in class_labels],
        "dataset_hash": str(dataset_hash),
        "split_manifest_hash": str(split_manifest_hash),
        "preprocessing_config": dict(preprocessing_config or {}),
        "selected_parameters": dict(selected_parameters or {}),
        "environment": collect_environment_info(),
    }

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return metadata


def _is_within_directory(target_path: Path, allowed_root: Path) -> bool:
    """Return True only if `target_path` resolves inside `allowed_root`."""
    try:
        target_path.resolve().relative_to(allowed_root.resolve())
        return True
    except ValueError:
        return False


def load_trusted_pipeline_artifact(
    artifact_path: Path | str,
    metadata_path: Path | str | None = None,
    allowed_root: Path | str = PROJECT_ROOT,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Load a local serialized pipeline ONLY after verifying path containment, metadata sidecar, and SHA-256 hash."""
    resolved_artifact = Path(artifact_path).resolve()
    resolved_root = Path(allowed_root).resolve()

    if not _is_within_directory(resolved_artifact, resolved_root):
        raise UntrustedArtifactError(
            f"Security restriction: artifact path '{resolved_artifact}' lies outside allowed root '{resolved_root}'."
        )

    if not resolved_artifact.is_file():
        raise FileNotFoundError(f"Serialized pipeline artifact not found: {resolved_artifact}")

    if metadata_path is None:
        # Derive companion .metadata.json path
        stem_without_joblib = resolved_artifact.name.removesuffix(".joblib")
        resolved_meta = resolved_artifact.with_name(f"{stem_without_joblib}.metadata.json")
    else:
        resolved_meta = Path(metadata_path).resolve()

    if not _is_within_directory(resolved_meta, resolved_root):
        raise UntrustedArtifactError(
            f"Security restriction: metadata path '{resolved_meta}' lies outside allowed root '{resolved_root}'."
        )

    if not resolved_meta.is_file():
        raise UntrustedArtifactError(
            f"Refusing to load serialized pipeline without companion metadata sidecar: {resolved_meta}"
        )

    with resolved_meta.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    expected_sha = metadata.get("artifact_sha256", "")
    if not expected_sha or not isinstance(expected_sha, str):
        raise UntrustedArtifactError("Metadata sidecar is missing a valid 'artifact_sha256' digest.")

    actual_sha = compute_file_sha256(resolved_artifact)
    if actual_sha.lower() != expected_sha.lower():
        raise UntrustedArtifactError(
            f"Artifact SHA-256 integrity check failed for {resolved_artifact}: "
            f"expected {expected_sha}, got {actual_sha}."
        )

    feature_names = metadata.get("feature_names", [])
    try:
        validate_predictor_columns(feature_names)
    except Exception as exc:
        raise UntrustedArtifactError(f"Artifact metadata failed feature allowlist validation: {exc}") from exc

    pipeline = joblib.load(resolved_artifact)
    return pipeline, metadata
