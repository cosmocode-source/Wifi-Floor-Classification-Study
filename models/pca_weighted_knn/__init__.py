'Stage 3 model package: PCA + distance-weighted kNN floor classifier.'

from __future__ import annotations

from typing import Any

from shared.config import MODEL_ID_PCA_WEIGHTED_KNN

__all__ = [
    "DECLARED_TIE_BREAKER_POLICY",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "FIXED_BASELINE_PARAMS",
    "MODEL_ID",
    "PCA_WEIGHTED_KNN_ASSUMPTIONS",
    "PCA_WEIGHTED_KNN_DIR",
    "build_configured_pca_knn_pipeline",
    "build_distance_weighted_pca_knn_pipeline",
    "build_fixed_baseline_weighted_pipeline",
    "build_matched_knn_comparison_table",
    "build_selected_vs_matched_knn_summary",
    "build_validation_curve_dataframe",
    "clean_pca_weighted_knn_dataframe",
    "compare_holdout_with_uniform_pca_knn",
    "compare_with_uniform_pca_knn",
    "compute_encoding_config_sha256",
    "evaluate_pca_weighted_knn",
    "evaluate_single_candidate_cv",
    "extract_validated_wap_features_and_targets",
    "generate_candidate_grid",
    "load_frozen_selection_config",
    "load_pca_weighted_knn_config",
    "resolve_parallelism",
    "resolve_training_run_dir",
    "run_clean_data",
    "select_best_pca_knn_candidate",
    "train_pca_weighted_knn",
]

MODEL_ID: str = MODEL_ID_PCA_WEIGHTED_KNN

_CLEAN_DATA_ATTRS = {
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CONFIG_YAML",
    "DEFAULT_PROCESSED_DATA_DIR",
    "DEFAULT_RESULTS_DIR",
    "PCA_WEIGHTED_KNN_DIR",
    "clean_pca_weighted_knn_dataframe",
    "compute_encoding_config_sha256",
    "extract_validated_wap_features_and_targets",
    "load_pca_weighted_knn_config",
    "run_clean_data",
}

_TRAIN_ATTRS = {
    "DECLARED_TIE_BREAKER_POLICY",
    "FIXED_BASELINE_PARAMS",
    "PCA_WEIGHTED_KNN_ASSUMPTIONS",
    "build_configured_pca_knn_pipeline",
    "build_distance_weighted_pca_knn_pipeline",
    "build_fixed_baseline_weighted_pipeline",
    "build_matched_knn_comparison_table",
    "build_selected_vs_matched_knn_summary",
    "build_validation_curve_dataframe",
    "compare_with_uniform_pca_knn",
    "evaluate_single_candidate_cv",
    "generate_candidate_grid",
    "resolve_parallelism",
    "select_best_pca_knn_candidate",
    "train_pca_weighted_knn",
}

_TEST_ATTRS = {
    "compare_holdout_with_uniform_pca_knn",
    "evaluate_pca_weighted_knn",
    "load_frozen_selection_config",
    "resolve_training_run_dir",
}


def __getattr__(name: str) -> Any:
    if name in _CLEAN_DATA_ATTRS:
        from models.pca_weighted_knn import clean_data

        return getattr(clean_data, name)
    if name in _TRAIN_ATTRS:
        from models.pca_weighted_knn import train

        return getattr(train, name)
    if name in _TEST_ATTRS:
        from models.pca_weighted_knn import test

        return getattr(test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
