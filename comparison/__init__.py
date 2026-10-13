'Multi-model comparison, plotting, and academic reporting package (`comparison/`).'

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ComparisonValidationError",
    "DEFAULT_COMPARISON_CONFIG_YAML",
    "DRAFT_BANNER",
    "align_and_validate_predictions_across_models",
    "build_academic_writeup_markdown",
    "build_comparison_report_markdown",
    "build_presentation_slides_markdown",
    "generate_academic_materials",
    "generate_all_comparison_figures",
    "load_comparison_config",
    "plot_accuracy_and_macro_f1_comparison",
    "plot_confusion_matrices",
    "plot_knn_validation_curves",
    "plot_runtime_comparison",
    "run_comparison",
]

_SYMBOL_TO_MODULE: dict[str, str] = {
    "ComparisonValidationError": "comparison.compare",
    "DEFAULT_COMPARISON_CONFIG_YAML": "comparison.compare",
    "align_and_validate_predictions_across_models": "comparison.compare",
    "load_comparison_config": "comparison.compare",
    "run_comparison": "comparison.compare",
    "generate_all_comparison_figures": "comparison.plots",
    "plot_accuracy_and_macro_f1_comparison": "comparison.plots",
    "plot_confusion_matrices": "comparison.plots",
    "plot_knn_validation_curves": "comparison.plots",
    "plot_runtime_comparison": "comparison.plots",
    "DRAFT_BANNER": "comparison.report_builder",
    "build_academic_writeup_markdown": "comparison.report_builder",
    "build_comparison_report_markdown": "comparison.report_builder",
    "build_presentation_slides_markdown": "comparison.report_builder",
    "generate_academic_materials": "comparison.report_builder",
}


def __getattr__(name: str) -> Any:
    if name in _SYMBOL_TO_MODULE:
        mod = importlib.import_module(_SYMBOL_TO_MODULE[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
