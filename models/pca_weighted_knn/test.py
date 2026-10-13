'Holdout and protected final evaluation script for PCA + distance-weighted kNN.'

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from models.pca_knn.test import (
    build_arg_parser as _shared_build_arg_parser,
    evaluate_pca_knn as _shared_evaluate_pca_knn,
    load_frozen_selection_config as _shared_load_frozen_selection_config,
    resolve_training_run_dir,
)
from models.pca_weighted_knn.clean_data import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_YAML,
    DEFAULT_RESULTS_DIR,
)
from shared.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MANIFEST_CSV,
    DEFAULT_TRAINING_CSV,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    MODEL_ID_PCA_KNN,
    MODEL_ID_PCA_WEIGHTED_KNN,
    PROJECT_ROOT,
)


def load_frozen_selection_config(selection_config_path: Path | str) -> dict[str, Any]:
    """Load and verify ``selection_config.json`` from a ``pca_weighted_knn`` training run."""
    return _shared_load_frozen_selection_config(
        selection_config_path,
        expected_model_id=MODEL_ID_PCA_WEIGHTED_KNN,
    )


def compare_holdout_with_uniform_pca_knn(
    *,
    uniform_eval_dir: Path | str,
    weighted_eval_dir: Path | str,
    output_json_path: Path | str | None = None,
) -> dict[str, Any]:
    'Compare holdout ``metrics.json`` between ``pca_knn`` (uniform) and ``pca_weighted_knn`` (distance).'
    u_dir = Path(uniform_eval_dir)
    w_dir = Path(weighted_eval_dir)

    def _load_variant_metrics(base_dir: Path, variant: str) -> dict[str, Any] | None:
        sub_metrics = base_dir / variant / "metrics.json"
        if sub_metrics.is_file():
            return json.loads(sub_metrics.read_text(encoding="utf-8"))
        for candidate in base_dir.glob(f"*_{variant}_metrics.json"):
            return json.loads(candidate.read_text(encoding="utf-8"))
        return None

    comparison: dict[str, Any] = {
        "uniform_model_id": MODEL_ID_PCA_KNN,
        "weighted_model_id": MODEL_ID_PCA_WEIGHTED_KNN,
        "methodology_note": (
            "Fair holdout comparison between uniform and distance-weighted PCA-kNN. "
            "At k=1, distance weighting offers no multi-neighbor voting advantage over "
            "uniform 1NN. For tuned pipelines, per-building differences reflect "
            "both the distance-weighting rule and any training-CV-selected shift in (k, PCA, scaling)."
        ),
        "variants": {},
    }

    for variant in ("tuned", "fixed_baseline"):
        u_metrics = _load_variant_metrics(u_dir, variant)
        w_metrics = _load_variant_metrics(w_dir, variant)
        if u_metrics is None or w_metrics is None:
            continue
        if u_metrics.get("status") != "completed" or w_metrics.get("status") != "completed":
            continue

        if u_metrics.get("evaluation_role") != w_metrics.get("evaluation_role"):
            raise ValueError(
                f"Cannot compare holdout metrics across different evaluation roles: "
                f"{u_metrics.get('evaluation_role')!r} vs {w_metrics.get('evaluation_role')!r}."
            )

        u_pooled = u_metrics["pooled"]
        w_pooled = w_metrics["pooled"]
        u_pb = u_metrics.get("per_building", {})
        w_pb = w_metrics.get("per_building", {})
        u_params_by_b = u_metrics.get("selected_parameters", {}).get("per_building", {})
        w_params_by_b = w_metrics.get("selected_parameters", {}).get("per_building", {})

        per_building_deltas: dict[str, Any] = {}
        for b_key in sorted(set(u_pb.keys()) & set(w_pb.keys()), key=int):
            u_b = u_pb[b_key]
            w_b = w_pb[b_key]
            u_p = u_params_by_b.get(b_key, {})
            w_p = w_params_by_b.get(b_key, {})
            same_k_pca_scaling = (
                u_p.get("n_neighbors") == w_p.get("n_neighbors")
                and u_p.get("n_components") == w_p.get("n_components")
                and u_p.get("scaling") == w_p.get("scaling")
            )
            is_both_k1 = (
                u_p.get("n_neighbors") == 1 and w_p.get("n_neighbors") == 1
            )
            per_building_deltas[b_key] = {
                "building_id": int(b_key),
                "uniform_selected_params": u_p,
                "weighted_selected_params": w_p,
                "matched_k_pca_scaling": bool(same_k_pca_scaling),
                "is_both_k1_trivial_weighting": bool(is_both_k1),
                "uniform_accuracy": float(u_b["accuracy"]),
                "weighted_accuracy": float(w_b["accuracy"]),
                "accuracy_delta_weighted_minus_uniform": float(
                    w_b["accuracy"] - u_b["accuracy"]
                ),
                "uniform_macro_f1": float(u_b["macro_f1"]),
                "weighted_macro_f1": float(w_b["macro_f1"]),
                "macro_f1_delta_weighted_minus_uniform": float(
                    w_b["macro_f1"] - u_b["macro_f1"]
                ),
            }

        comparison["variants"][variant] = {
            "evaluation_role": u_metrics.get("evaluation_role"),
            "is_fixed_k1_baseline": variant == "fixed_baseline",
            "pooled_accuracy_uniform": float(u_pooled["accuracy"]),
            "pooled_accuracy_weighted": float(w_pooled["accuracy"]),
            "pooled_accuracy_delta_weighted_minus_uniform": float(
                w_pooled["accuracy"] - u_pooled["accuracy"]
            ),
            "pooled_macro_f1_uniform": float(u_pooled["macro_f1"]),
            "pooled_macro_f1_weighted": float(w_pooled["macro_f1"]),
            "pooled_macro_f1_delta_weighted_minus_uniform": float(
                w_pooled["macro_f1"] - u_pooled["macro_f1"]
            ),
            "per_building": per_building_deltas,
        }

    if output_json_path is not None:
        out_file = Path(output_json_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(comparison, indent=2, sort_keys=False),
            encoding="utf-8",
        )
    return comparison


def evaluate_pca_weighted_knn(
    eval_mode: str = EVALUATION_ROLE_INTERNAL_HOLDOUT,
    variant: str = "both",
    train_run_dir: Path | str | None = None,
    train_run_id: str | None = None,
    selection_config_path: Path | str | None = None,
    train_csv_path: Path | str = DEFAULT_TRAINING_CSV,
    split_manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_CSV,
    val_csv_path: Path | str | None = None,
    confirm_final_evaluation: bool = False,
    refit_all_training: bool = False,
    config_path: Path | str = DEFAULT_CONFIG_YAML,
    artifacts_base_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    results_base_dir: Path | str = DEFAULT_RESULTS_DIR,
    allowed_root: Path | str = PROJECT_ROOT,
    estimator_n_jobs: int = 1,
    seed: int = DEFAULT_RANDOM_SEED,
    eval_run_id: str | None = None,
    uniform_eval_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate frozen ``pca_weighted_knn`` pipelines without refitting."""
    summary = _shared_evaluate_pca_knn(
        eval_mode=eval_mode,
        variant=variant,
        train_run_dir=train_run_dir,
        train_run_id=train_run_id,
        selection_config_path=selection_config_path,
        train_csv_path=train_csv_path,
        split_manifest_path=split_manifest_path,
        val_csv_path=val_csv_path,
        confirm_final_evaluation=confirm_final_evaluation,
        refit_all_training=refit_all_training,
        config_path=config_path,
        artifacts_base_dir=artifacts_base_dir,
        results_base_dir=results_base_dir,
        allowed_root=allowed_root,
        estimator_n_jobs=estimator_n_jobs,
        seed=seed,
        eval_run_id=eval_run_id,
        expected_model_id=MODEL_ID_PCA_WEIGHTED_KNN,
        expected_weights="distance",
    )

    if uniform_eval_run_dir is not None:
        weighted_eval_dir = Path(summary["results_dir"])
        comparison_json_path = weighted_eval_dir / "uniform_vs_weighted_holdout_comparison.json"
        holdout_comparison = compare_holdout_with_uniform_pca_knn(
            uniform_eval_dir=uniform_eval_run_dir,
            weighted_eval_dir=weighted_eval_dir,
            output_json_path=comparison_json_path,
        )
        summary["uniform_vs_weighted_holdout_comparison_json"] = str(comparison_json_path.resolve())
        summary["uniform_vs_weighted_holdout_comparison"] = holdout_comparison

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``python -m models.pca_weighted_knn.test``."""
    parser = _shared_build_arg_parser(
        prog="python -m models.pca_weighted_knn.test",
        description=(
            "Evaluate trusted saved PCA + distance-weighted kNN pipelines on the shared internal "
            "development holdout (default) or on the official validation holdout (requires "
            "frozen selection_config.json and explicit --confirm-final-evaluation)."
        ),
        default_config_path=DEFAULT_CONFIG_YAML,
        default_artifacts_dir=DEFAULT_ARTIFACTS_DIR,
        default_results_dir=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--uniform-eval-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to a completed models/pca_knn/results/<eval_run_id> directory "
            "to write a fair holdout comparison JSON."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m models.pca_weighted_knn.test``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = evaluate_pca_weighted_knn(
        eval_mode=args.eval_mode,
        variant=args.variant,
        train_run_dir=args.train_run_dir,
        train_run_id=args.train_run_id,
        selection_config_path=args.selection_config,
        train_csv_path=args.train_csv,
        split_manifest_path=args.split_manifest,
        val_csv_path=args.val_csv,
        confirm_final_evaluation=args.confirm_final_evaluation,
        refit_all_training=args.refit_all_training,
        config_path=args.config,
        artifacts_base_dir=args.artifacts_dir,
        results_base_dir=args.results_dir,
        estimator_n_jobs=args.estimator_n_jobs,
        seed=args.seed,
        eval_run_id=args.eval_run_id,
        uniform_eval_run_dir=args.uniform_eval_run_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "build_arg_parser",
    "compare_holdout_with_uniform_pca_knn",
    "evaluate_pca_weighted_knn",
    "load_frozen_selection_config",
    "main",
    "resolve_training_run_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())
