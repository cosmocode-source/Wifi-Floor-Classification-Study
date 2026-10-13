'Stage 2 model package: PCA + uniform-weight kNN floor classifier.'

from __future__ import annotations

from typing import Any

from shared.config import MODEL_ID_PCA_KNN

__all__ = [
    "DECLARED_TIE_BREAKER_POLICY",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "FIXED_BASELINE_PARAMS",
    "MODEL_ID",
    "PCA_KNN_ASSUMPTIONS",
    "PCA_KNN_DIR",
    "PCA_WEIGHTED_KNN_ASSUMPTIONS",
    "build_configured_pca_knn_pipeline",
    "build_fixed_baseline_pipeline",
    "build_matched_knn_comparison_table",
    "build_selected_vs_matched_knn_summary",
    "build_uniform_pca_knn_pipeline",
    "build_validation_curve_dataframe",
    "clean_pca_knn_dataframe",
    "compute_encoding_config_sha256",
    "evaluate_pca_knn",
    "evaluate_single_candidate_cv",
    "extract_validated_wap_features_and_targets",
    "generate_candidate_grid",
    "load_frozen_selection_config",
    "load_pca_knn_config",
    "resolve_parallelism",
    "resolve_training_run_dir",
    "run_clean_data",
    "select_best_pca_knn_candidate",
    "train_pca_knn",
]

MODEL_ID: str = MODEL_ID_PCA_KNN

_CLEAN_DATA_ATTRS = {
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "PCA_KNN_DIR",
    "clean_pca_knn_dataframe",
    "compute_encoding_config_sha256",
    "extract_validated_wap_features_and_targets",
    "load_pca_knn_config",
    "run_clean_data",
}

_TRAIN_ATTRS = {
    "DECLARED_TIE_BREAKER_POLICY",
    "FIXED_BASELINE_PARAMS",
    "PCA_KNN_ASSUMPTIONS",
    "PCA_WEIGHTED_KNN_ASSUMPTIONS",
    "build_configured_pca_knn_pipeline",
    "build_fixed_baseline_pipeline",
    "build_matched_knn_comparison_table",
    "build_selected_vs_matched_knn_summary",
    "build_uniform_pca_knn_pipeline",
    "build_validation_curve_dataframe",
    "evaluate_single_candidate_cv",
    "generate_candidate_grid",
    "resolve_parallelism",
    "select_best_pca_knn_candidate",
    "train_pca_knn",
}

_TEST_ATTRS = {
    "evaluate_pca_knn",
    "load_frozen_selection_config",
    "resolve_training_run_dir",
}


def __getattr__(name: str) -> Any:
    if name in _CLEAN_DATA_ATTRS:
        from models.pca_knn import clean_data

        return getattr(clean_data, name)
    if name in _TRAIN_ATTRS:
        from models.pca_knn import train

        return getattr(train, name)
    if name in _TEST_ATTRS:
        from models.pca_knn import test

        return getattr(test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
