"""Regression tests for independent model command-line entry points."""

import importlib
import pytest

@pytest.mark.parametrize("model", ["pca_knn", "pca_weighted_knn", "gradient_boosting"])
def test_evaluation_main_forwards_seed_without_missing_namespace_attributes(model, monkeypatch):
    module = importlib.import_module(f"models.{model}.test")
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"eval_run_id": "fixture", "evaluation_role": "internal_holdout",
                "results_dir": "synthetic_fixture", "variants": {}}

    monkeypatch.setattr(module, f"evaluate_{model}", fake_evaluate)
    assert module.main(["--seed", "17"]) == 0
    assert captured["seed"] == 17

@pytest.mark.parametrize('model', ['pca_knn', 'pca_weighted_knn', 'gradient_boosting'])
def test_cleaning_file_owns_model_pipeline_factory(model):
    module = importlib.import_module(f'models.{model}.clean_data')
    if model == 'pca_knn':
        factory = module.build_configured_pca_knn_pipeline
        pipeline = factory(n_components=5, n_neighbors=1, scaling='standard')
    elif model == 'pca_weighted_knn':
        factory = module.build_distance_weighted_pca_knn_pipeline
        pipeline = factory(n_components=5, n_neighbors=1, scaling='standard')
        assert pipeline.named_steps['classifier'].weights == 'distance'
    else:
        factory = module.build_configured_gradient_boosting_pipeline
        pipeline = factory(scale=True, use_pca=True, n_components=5)
        default = factory()
        assert 'scaler' not in default.named_steps and 'pca' not in default.named_steps
    assert factory.__module__ == f'models.{model}.clean_data'
    assert 'sentinel_replacement' in pipeline.named_steps
    assert 'scaler' in pipeline.named_steps and 'pca' in pipeline.named_steps
    assert not hasattr(pipeline.named_steps['scaler'], 'mean_')

@pytest.mark.parametrize('model', ['pca_knn', 'pca_weighted_knn', 'gradient_boosting'])
def test_independent_clean_train_and_evaluate_cli_arguments(model):
    importlib.import_module(f'models.{model}.clean_data').build_arg_parser().parse_args([])
    train_args = ['--profile', 'smoke', '--run-id', 'experiment_01']
    eval_args = ['--train-run-dir', f'models/{model}/artifacts/experiment_01',
                 '--eval-mode', 'internal_holdout', '--variant', 'both',
                 '--eval-run-id', 'experiment_01_internal']
    if model == 'pca_weighted_knn':
        train_args += ['--uniform-reference-run-dir', 'models/pca_knn/artifacts/experiment_01']
        eval_args += ['--uniform-eval-run-dir', 'models/pca_knn/results/experiment_01_internal']
    importlib.import_module(f'models.{model}.train').build_arg_parser().parse_args(train_args)
    importlib.import_module(f'models.{model}.test').build_arg_parser().parse_args(eval_args)
