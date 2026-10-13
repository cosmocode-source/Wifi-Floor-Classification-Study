"""Lazy public exports; avoid importing CLI modules before python -m executes them."""

from importlib import import_module

_EXPORT_GROUPS = {
    'shared.artifacts': (
        'UntrustedArtifactError', 'create_unique_run_dir', 'generate_unique_run_id',
        'load_trusted_pipeline_artifact', 'save_pipeline_artifact',
    ),
    'shared.config': (
        'BUILDING_COLUMN', 'DEFAULT_FINAL_CV_FOLDS', 'DEFAULT_INTERNAL_HOLDOUT_FRACTION',
        'DEFAULT_LIGHTWEIGHT_CV_FOLDS', 'DEFAULT_N_JOBS', 'DEFAULT_RANDOM_SEED',
        'DEFAULT_UNDETECTED_REPLACEMENT', 'NUM_WAP_FEATURES', 'PROJECT_ROOT',
        'RESULT_SCHEMA_VERSION', 'ROW_ID_COLUMN', 'TARGET_COLUMN',
        'UNDETECTED_SENTINEL', 'VALID_EVALUATION_ROLES', 'VALID_MODEL_IDS',
        'WAP_COLUMNS',
    ),
    'shared.create_splits': (
        'SplitCreationError', 'SplitValidationReport', 'align_dataset_with_manifest',
        'create_development_splits', 'iter_building_cv_folds', 'load_split_manifest',
        'save_split_manifest', 'validate_split_manifest',
    ),
    'shared.data_validation': (
        'DataValidationError', 'DatasetAuditReport', 'assign_stable_row_ids',
        'audit_dataframe', 'compute_dataframe_sha256', 'compute_file_sha256',
        'extract_predictors_and_targets', 'load_and_audit_csv', 'make_row_id',
        'validate_predictor_columns', 'validate_wap_numeric_and_ranges',
    ),
    'shared.demo': (
        'DemoInferenceError', 'format_demo_table', 'load_dev_rehearsal_examples',
        'prepare_demo_input_dataframe', 'run_offline_demo_inference',
    ),
    'shared.feasibility': (
        'FeasibilityError', 'FeasibilityResult', 'check_candidate_feasibility',
        'check_class_support_feasibility', 'check_knn_feasibility', 'check_pca_feasibility',
        'compute_max_feasible_pca_components',
    ),
    'shared.metrics': (
        'compute_classification_metrics', 'compute_pooled_metrics', 'evaluate_predictions_by_building',
        'select_best_candidate',
    ),
    'shared.preprocessing': (
        'SentinelReplacementTransformer', 'build_gradient_boosting_pipeline', 'build_knn_preprocessing_pipeline',
        'build_pca_knn_pipeline', 'describe_preprocessing_config',
    ),
    'shared.result_schema': (
        'ResultSchemaError', 'build_completed_result', 'build_non_completed_result',
        'collect_environment_info', 'load_predictions_csv', 'load_result_json',
        'save_predictions_csv', 'save_result_json', 'validate_predictions_dataframe',
        'validate_result_payload',
    ),
}

_SYMBOL_TO_MODULE = {name: module for module, names in _EXPORT_GROUPS.items() for name in names}
__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_SYMBOL_TO_MODULE[name]), name)
    globals()[name] = value
    return value
