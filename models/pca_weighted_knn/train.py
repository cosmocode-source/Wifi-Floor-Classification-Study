'Training, fold-safe CV search, validation-curve export, and fair uniform-vs-weighted comparison.'

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from models.pca_knn.train import (
    DECLARED_TIE_BREAKER_POLICY,
    FIXED_BASELINE_PARAMS,
    PCA_WEIGHTED_KNN_ASSUMPTIONS,
    _parse_int_list,
    _parse_str_list,
    build_arg_parser as _shared_build_arg_parser,
    build_configured_pca_knn_pipeline,
    build_fixed_baseline_pipeline,
    build_matched_knn_comparison_table,
    build_selected_vs_matched_knn_summary,
    build_validation_curve_dataframe,
    evaluate_single_candidate_cv,
    generate_candidate_grid,
    resolve_parallelism,
    select_best_pca_knn_candidate,
    train_pca_knn as _shared_train_pca_knn,
)
from models.pca_weighted_knn.clean_data import (
    build_distance_weighted_pca_knn_pipeline,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_YAML,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_RESULTS_DIR,
)
from shared.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    DEFAULT_UNDETECTED_REPLACEMENT,
    MODEL_ID_PCA_WEIGHTED_KNN,
    UNDETECTED_SENTINEL,
)




def build_fixed_baseline_weighted_pipeline(
    sentinel_value: int | float = UNDETECTED_SENTINEL,
    replacement_value: float = DEFAULT_UNDETECTED_REPLACEMENT,
    enforce_wap_schema: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = 1,
) -> Pipeline:
    """Build the paper-inspired reference baseline (`k=1, PCA=50, unscaled, whiten=False`) with `weights='distance'`.

    Note: At ``k=1``, distance weighting has no multi-neighbor voting advantage over
    uniform 1NN. This variant is kept for pipeline parity and sanity verification.
    """
    return build_fixed_baseline_pipeline(
        sentinel_value=sentinel_value,
        replacement_value=replacement_value,
        enforce_wap_schema=enforce_wap_schema,
        random_state=random_state,
        n_jobs=n_jobs,
        weights="distance",
    )


def compare_with_uniform_pca_knn(
    uniform_search_df: pd.DataFrame,
    weighted_search_df: pd.DataFrame,
    uniform_selection_config: Mapping[str, Any],
    weighted_selection_config: Mapping[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    'Compute both matched-configuration and best-selected comparisons against uniform PCA-kNN.'
    matched_df = build_matched_knn_comparison_table(
        uniform_search_df=uniform_search_df,
        weighted_search_df=weighted_search_df,
    )
    summary = build_selected_vs_matched_knn_summary(
        uniform_selection_config=uniform_selection_config,
        weighted_selection_config=weighted_selection_config,
        matched_df=matched_df,
    )
    result: dict[str, Any] = {
        "matched_df": matched_df,
        "summary": summary,
    }
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        matched_csv_path = out_path / "matched_uniform_vs_weighted_cv.csv"
        summary_json_path = out_path / "uniform_vs_weighted_comparison_summary.json"
        matched_df.to_csv(matched_csv_path, index=False)
        summary_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        result["matched_csv_path"] = str(matched_csv_path.resolve())
        result["summary_json_path"] = str(summary_json_path.resolve())
    return result


def train_pca_weighted_knn(
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    results_base_dir: Path | str = DEFAULT_RESULTS_DIR,
    processed_data_base_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
    profile: str | None = None,
    cv_folds: int | None = None,
    k_values: Sequence[int] | None = None,
    pca_components: Sequence[int] | None = None,
    scaling_modes: Sequence[str] | None = None,
    search_workers: int | None = None,
    estimator_n_jobs: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    run_id: str | None = None,
    save_dev_train_pca: bool = False,
    uniform_reference_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run training-side CV search, validation-curve export, and dev_train refit for ``pca_weighted_knn``."""
    return _shared_train_pca_knn(
        train_csv_path=train_csv_path,
        split_manifest_path=split_manifest_path,
        config_path=config_path,
        artifacts_base_dir=artifacts_base_dir,
        results_base_dir=results_base_dir,
        processed_data_base_dir=processed_data_base_dir,
        profile=profile,
        cv_folds=cv_folds,
        k_values=k_values,
        pca_components=pca_components,
        scaling_modes=scaling_modes,
        search_workers=search_workers,
        estimator_n_jobs=estimator_n_jobs,
        seed=seed,
        run_id=run_id,
        save_dev_train_pca=save_dev_train_pca,
        expected_model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        expected_weights="distance",
        uniform_reference_run_dir=uniform_reference_run_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``python -m models.pca_weighted_knn.train``."""
    return _shared_build_arg_parser(
        prog="python -m models.pca_weighted_knn.train",
        description=(
            "Run fold-local cross-validation search for PCA + distance-weighted kNN on shared development "
            "splits, export classification validation curves, and optionally compare against a uniform "
            "PCA-kNN reference run."
        ),
        default_config_path=DEFAULT_CONFIG_YAML,
        default_artifacts_dir=DEFAULT_ARTIFACTS_DIR,
        default_results_dir=DEFAULT_RESULTS_DIR,
        default_processed_data_dir=DEFAULT_PROCESSED_DATA_DIR,
        include_uniform_reference_arg=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m models.pca_weighted_knn.train``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = train_pca_weighted_knn(
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        config_path=args.config,
        artifacts_base_dir=args.artifacts_dir,
        results_base_dir=args.results_dir,
        processed_data_base_dir=args.processed_data_dir,
        profile=args.profile,
        cv_folds=args.cv_folds,
        k_values=_parse_int_list(args.k_values),
        pca_components=_parse_int_list(args.pca_components),
        scaling_modes=_parse_str_list(args.scaling_modes),
        search_workers=args.search_workers,
        estimator_n_jobs=args.estimator_n_jobs,
        seed=args.seed,
        run_id=args.run_id,
        save_dev_train_pca=args.save_dev_train_pca,
        uniform_reference_run_dir=args.uniform_reference_run_dir,
    )
    print(
        f"[pca_weighted_knn.train] Completed training run '{summary['run_id']}'.\n"
        f"  Artifacts directory:  {summary['artifacts_dir']}\n"
        f"  Results directory:    {summary['results_dir']}\n"
        f"  Search results CSV:   {summary['search_results_csv']}\n"
        f"  Validation curve CSV: {summary['validation_curve_csv']}\n"
        f"  Selection config:     {summary['selection_config_json']}"
    )
    return 0


__all__ = [
    "DECLARED_TIE_BREAKER_POLICY",
    "FIXED_BASELINE_PARAMS",
    "PCA_WEIGHTED_KNN_ASSUMPTIONS",
    "build_arg_parser",
    "build_configured_pca_knn_pipeline",
    "build_distance_weighted_pca_knn_pipeline",
    "build_fixed_baseline_weighted_pipeline",
    "build_matched_knn_comparison_table",
    "build_selected_vs_matched_knn_summary",
    "build_validation_curve_dataframe",
    "compare_with_uniform_pca_knn",
    "evaluate_single_candidate_cv",
    "generate_candidate_grid",
    "main",
    "resolve_parallelism",
    "select_best_pca_knn_candidate",
    "train_pca_weighted_knn",
]


if __name__ == "__main__":
    raise SystemExit(main())
