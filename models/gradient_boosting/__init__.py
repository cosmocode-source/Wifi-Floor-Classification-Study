'Stage 4 model package: Classical Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`).'

from __future__ import annotations

from typing import Any

from shared.config import MODEL_ID_GRADIENT_BOOSTING

__all__ = [
    "DECLARED_TIE_BREAKER_POLICY",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "FIXED_BASELINE_PARAMS",
    "GRADIENT_BOOSTING_ASSUMPTIONS",
    "GRADIENT_BOOSTING_DIR",
    "MODEL_ID",
    "STARTING_GB_PARAMS",
    "build_configured_gradient_boosting_pipeline",
    "build_starting_gradient_boosting_pipeline",
    "clean_gradient_boosting_dataframe",
    "compute_encoding_config_sha256",
    "enumerate_full_candidate_space",
    "evaluate_gradient_boosting",
    "evaluate_single_gb_candidate_cv",
    "extract_validated_wap_features_and_targets",
    "load_frozen_selection_config",
    "load_gradient_boosting_config",
    "resolve_gb_parallelism",
    "resolve_training_run_dir",
    "run_clean_data",
    "sample_gradient_boosting_candidates",
    "select_best_gb_candidate",
    "train_gradient_boosting",
]

MODEL_ID: str = MODEL_ID_GRADIENT_BOOSTING

_CLEAN_DATA_ATTRS = {
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "GRADIENT_BOOSTING_DIR",
    "clean_gradient_boosting_dataframe",
    "compute_encoding_config_sha256",
    "extract_validated_wap_features_and_targets",
    "load_gradient_boosting_config",
    "run_clean_data",
}

_TRAIN_ATTRS = {
    "DECLARED_TIE_BREAKER_POLICY",
    "FIXED_BASELINE_PARAMS",
    "GRADIENT_BOOSTING_ASSUMPTIONS",
    "STARTING_GB_PARAMS",
    "build_configured_gradient_boosting_pipeline",
    "build_starting_gradient_boosting_pipeline",
    "enumerate_full_candidate_space",
    "evaluate_single_gb_candidate_cv",
    "resolve_gb_parallelism",
    "sample_gradient_boosting_candidates",
    "select_best_gb_candidate",
    "train_gradient_boosting",
}

_TEST_ATTRS = {
    "evaluate_gradient_boosting",
    "load_frozen_selection_config",
    "resolve_training_run_dir",
}


def __getattr__(name: str) -> Any:
    if name in _CLEAN_DATA_ATTRS:
        from models.gradient_boosting import clean_data

        return getattr(clean_data, name)
    if name in _TRAIN_ATTRS:
        from models.gradient_boosting import train

        return getattr(train, name)
    if name in _TEST_ATTRS:
        from models.gradient_boosting import test

        return getattr(test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
