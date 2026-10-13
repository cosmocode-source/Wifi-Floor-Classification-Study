'Markdown comparison report and academic write-up/slide builder (`comparison/report_builder.py`).'

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from shared.config import (
    COMPARISON_RESULTS_DIR,
    PRESENTATION_DIR,
    REPORTS_DIR,
)


DRAFT_BANNER: str = "DRAFT — RESULTS PENDING (NO EXPERIMENTAL NUMBERS FABRICATED)"

MODEL_TITLES: dict[str, str] = {
    "pca_knn": "PCA + Uniform-Weight kNN (`pca_knn`)",
    "pca_weighted_knn": "PCA + Distance-Weighted kNN (`pca_weighted_knn`)",
    "gradient_boosting": "Classical Gradient Boosting (`gradient_boosting`)",
}


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None or val == "":
        return "N/A"
    try:
        f = float(val)
        return f"{f:.{digits}f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_pct(val: Any, digits: int = 2) -> str:
    if val is None or val == "":
        return "N/A"
    try:
        f = float(val) * 100.0
        return f"{f:.{digits}f}%"
    except (ValueError, TypeError):
        return "N/A"


def _summarize_model_params(model_id: str, selected_params: Mapping[str, Any]) -> str:
    pb = selected_params.get("per_building", {})
    if not pb:
        return json.dumps(dict(selected_params), sort_keys=True)
    parts: list[str] = []
    for b_key in sorted(pb.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        p = pb[b_key]
        if model_id in ("pca_knn", "pca_weighted_knn"):
            parts.append(
                f"B{b_key}: k={p.get('n_neighbors')}, PCA={p.get('n_components')}, scale={p.get('scaling')}"
            )
        else:
            parts.append(
                f"B{b_key}: n_est={p.get('n_estimators')}, lr={p.get('learning_rate')}, "
                f"depth={p.get('max_depth')}, leaf={p.get('min_samples_leaf')}"
            )
    return "; ".join(parts)


def _summarize_preprocessing(model_id: str, prep: Mapping[str, Any]) -> str:
    sentinel = prep.get("sentinel_value", 100)
    repl = prep.get("replacement_value", -110.0)
    use_pca = prep.get("use_pca", model_id != "gradient_boosting")
    if use_pca:
        return f"WAP001..WAP520 ({sentinel} -> {repl}), fold-local StandardScaler(optional) + PCA(whiten=False)"
    return f"Raw WAP001..WAP520 ({sentinel} -> {repl}), no PCA (use_pca=False), no scaling (scale=False)"


def build_comparison_report_markdown(combined_results: Mapping[str, Any]) -> str:
    """Render `comparison_report.md` strictly from `combined_results.json`."""
    status = str(combined_results.get("status", "incomplete"))
    comp_id = str(combined_results.get("comparison_id", "unidentified_comparison"))
    eval_role = str(combined_results.get("evaluation_role", "unknown"))
    ds_hash = str(combined_results.get("dataset_hash", "N/A"))
    manifest_hash = str(combined_results.get("split_manifest_hash", "N/A"))
    eval_rows = combined_results.get("evaluated_row_count", "N/A")

    lines: list[str] = [
        f"# Multi-Model Comparison Report (`{comp_id}`)",
        "",
    ]
    if status != "completed":
        lines.extend(
            [
                f"> **STATUS: {DRAFT_BANNER}**",
                f"> Comparison status is `{status}`. Missing or non-completed runs are reported honestly below without fabricated metrics.",
                "",
            ]
        )

    lines.extend(
        [
            "## 1. Evaluation Protocol & Provenance Verification",
            "",
            f"- **Comparison ID**: `{comp_id}`",
            f"- **Comparison Status**: `{status}`",
            f"- **Evaluation Split Role**: `{eval_role}` (strictly single-role; CV, internal holdout, and official validation scores are never mixed in an unlabeled leaderboard)",
            f"- **Target Definition**: Known-building `FLOOR` classification (`BUILDINGID` routes to per-building model `0`, `1`, `2`; strictly `WAP001..WAP520` predictors)",
            f"- **Evaluated Row Count**: `{eval_rows}` (aligned by stable `row_id`, verified identical ground-truth `true_floor` and `building_id`)",
            f"- **Source Dataset SHA-256**: `{ds_hash}`",
            f"- **Shared Split Manifest SHA-256**: `{manifest_hash}`",
            f"- **Pooled Accuracy Rule**: Exact pooled `correct_count / evaluated_row_count` across all buildings (never an unweighted mean of building accuracies)",
            f"- **Pooled Macro-F1 Policy**: `{combined_results.get('pooled_macro_f1_policy', 'fixed_global_floor_union_with_zero_division_0')}`",
            "",
            "---",
            "",
            "## 2. Main Three-Pipeline Summary (`Tuned / Primary Configurations`)",
            "",
            "| Model Configuration | Run ID | Status | Pooled Accuracy | Pooled Error | Pooled Macro-F1 | Balanced Acc | Correct / Evaluated (Incorrect) | Preprocessing & Feature Representation | Selected Parameters (Per Building) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    models_map: Mapping[str, Any] = combined_results.get("models", {})
    for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
        m_entry = models_map.get(m_id)
        title = MODEL_TITLES.get(m_id, m_id)
        if not m_entry or m_entry.get("status") != "completed":
            m_stat = m_entry.get("status", "missing") if m_entry else "missing"
            reason = m_entry.get("status_reason", "Artifact not provided") if m_entry else "Artifact not provided"
            lines.append(
                f"| {title} | `{m_entry.get('run_id', 'N/A') if m_entry else 'N/A'}` | `{m_stat}` | "
                f"N/A | N/A | N/A | N/A | N/A | N/A | {reason} |"
            )
            continue

        pooled = m_entry["pooled"]
        correct_c = int(pooled["evaluated_row_count"]) - int(pooled["incorrect_count"])
        eval_c = int(pooled["evaluated_row_count"])
        inc_c = int(pooled["incorrect_count"])
        prep_str = _summarize_preprocessing(m_id, m_entry.get("preprocessing", {}))
        param_str = _summarize_model_params(m_id, m_entry.get("selected_parameters", {}))

        lines.append(
            f"| {title} | `{m_entry['run_id']}` | `completed` | "
            f"**{_fmt_float(pooled['accuracy'])}** ({_fmt_pct(pooled['accuracy'])}) | "
            f"{_fmt_float(pooled['error'])} | "
            f"**{_fmt_float(pooled['macro_f1'])}** | "
            f"{_fmt_float(pooled['balanced_accuracy'])} | "
            f"`{correct_c} / {eval_c}` (`{inc_c}` err) | "
            f"{prep_str} | {param_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Per-Building Results Breakdown",
            "",
            "| Building | Model Configuration | Accuracy | Error | Macro-F1 | Balanced Acc | Correct / Evaluated (Incorrect) | Selected Building Hyperparameters |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    all_buildings = sorted(
        {
            str(b)
            for m_id, m_entry in models_map.items()
            if m_entry and m_entry.get("status") == "completed"
            for b in m_entry.get("per_building", {}).keys()
        },
        key=lambda x: int(x) if x.isdigit() else x,
    )
    for b_key in all_buildings:
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            m_entry = models_map.get(m_id)
            if not m_entry or m_entry.get("status") != "completed":
                continue
            pb = m_entry.get("per_building", {}).get(b_key)
            if not pb:
                continue
            b_corr = int(pb["evaluated_row_count"]) - int(pb["incorrect_count"])
            b_eval = int(pb["evaluated_row_count"])
            b_inc = int(pb["incorrect_count"])
            b_params = m_entry.get("selected_parameters", {}).get("per_building", {}).get(b_key, {})
            if m_id in ("pca_knn", "pca_weighted_knn"):
                b_param_str = (
                    f"`k={b_params.get('n_neighbors')}, PCA={b_params.get('n_components')}, "
                    f"scaling={b_params.get('scaling')}, weights={m_entry.get('selected_parameters', {}).get('weights', 'uniform' if m_id == 'pca_knn' else 'distance')}`"
                )
            else:
                b_param_str = (
                    f"`n_estimators={b_params.get('n_estimators')}, lr={b_params.get('learning_rate')}, "
                    f"max_depth={b_params.get('max_depth')}, min_samples_leaf={b_params.get('min_samples_leaf')}`"
                )
            lines.append(
                f"| Building `{b_key}` | `{m_id}` | {_fmt_float(pb['accuracy'])} | {_fmt_float(pb['error'])} | "
                f"{_fmt_float(pb['macro_f1'])} | {_fmt_float(pb['balanced_accuracy'])} | "
                f"`{b_corr} / {b_eval}` (`{b_inc}` err) | {b_param_str} |"
            )

    # Preprocessing & Fairness Disclosure
    prep_disc = combined_results.get("preprocessing_disclosure", {})
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Preprocessing & Controlled Comparison Disclosure",
            "",
            f"- **Pipeline Comparison Scope**: `{prep_disc.get('pipeline_comparison_scope', 'full_pipeline_comparison_pca_knn_vs_raw_feature_gradient_boosting')}`",
            f"- **kNN Sentinel Encoding Match**: `{prep_disc.get('knn_encoding_matched', True)}` (`100 -> -110.0` dBm, genuine `0` dBm preserved as `0.0`)",
            f"- **k=1 Equivalence Rule**: {prep_disc.get('k1_equivalence_note', 'At k=1, uniform and distance-weighted kNN are mathematically identical.')}",
        ]
    )

    knn_pair = prep_disc.get("uniform_vs_weighted_knn_analysis", {})
    if knn_pair:
        lines.extend(
            [
                "",
                "### 4.1 Uniform vs. Distance-Weighted kNN Per-Building Comparison Type",
                "",
                "| Building | Uniform Selected `(k, PCA, scale)` | Weighted Selected `(k, PCA, scale)` | Comparison Type | Both `k=1` (Trivial Weighting)? | Accuracy Delta (Weighted - Uniform) | Macro-F1 Delta (Weighted - Uniform) |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for b_key, b_info in sorted(knn_pair.get("per_building", {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
            u_p = b_info.get("uniform_params", {})
            w_p = b_info.get("weighted_params", {})
            lines.append(
                f"| Building `{b_key}` | `({u_p.get('n_neighbors')}, {u_p.get('n_components')}, {u_p.get('scaling')})` | "
                f"`({w_p.get('n_neighbors')}, {w_p.get('n_components')}, {w_p.get('scaling')})` | "
                f"`{b_info.get('comparison_label')}` | `{b_info.get('both_k_equals_1')}` | "
                f"{_fmt_float(b_info.get('accuracy_delta_weighted_minus_uniform'))} | "
                f"{_fmt_float(b_info.get('macro_f1_delta_weighted_minus_uniform'))} |"
            )

    # Supplementary Baselines Table
    supp_baselines: Mapping[str, Any] = combined_results.get("supplementary_baselines", {})
    if supp_baselines:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Supplementary Fixed / Paper-Inspired Baselines vs. Tuned Pipelines",
                "",
                "> Paper-inspired baselines (`k=1, PCA=50, scaling=none` for kNN; `max_depth=4, n_estimators=189, lr=0.1` for Gradient Boosting) are reported separately below so tuning gains are never conflated with distance-weighting or estimator effects.",
                "",
                "| Model ID | Baseline Variant | Run ID | Evaluation Role | Pooled Accuracy | Pooled Macro-F1 | Balanced Acc | Delta Acc (Tuned - Baseline) | Notes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            b_entry = supp_baselines.get(m_id)
            if not b_entry or b_entry.get("status") != "completed":
                continue
            bp = b_entry["pooled"]
            tuned_entry = models_map.get(m_id, {})
            delta_acc = (
                float(tuned_entry["pooled"]["accuracy"]) - float(bp["accuracy"])
                if tuned_entry.get("status") == "completed"
                else None
            )
            note = (
                "At k=1, uniform and weighted baselines are identical (no multi-neighbor voting)."
                if m_id in ("pca_knn", "pca_weighted_knn")
                else "Paper-inspired depth=4, n_estimators=189; lr=0.1 and raw -110 dBm encoding are project assumptions."
            )
            lines.append(
                f"| `{m_id}` | `fixed_baseline` | `{b_entry['run_id']}` | `{b_entry['evaluation_role']}` | "
                f"{_fmt_float(bp['accuracy'])} | {_fmt_float(bp['macro_f1'])} | {_fmt_float(bp['balanced_accuracy'])} | "
                f"{_fmt_float(delta_acc)} | {note} |"
            )

    # Supplementary Training CV Summary
    supp_cv: Mapping[str, Any] = combined_results.get("supplementary_training_cv", {})
    if supp_cv:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 6. Training-Side Cross-Validation Summary (`evaluation_role = cv_val`)",
                "",
                "> Model and hyperparameter selection were conducted strictly using these training-side CV metrics on `dev_train`, prior to evaluating frozen pipelines on the holdout split.",
                "",
                "| Model ID | CV Run ID | Pooled OOF Accuracy | Pooled OOF Macro-F1 | CV Search Trials / Candidates | Skipped Infeasible Candidates | CV Search Time (s) |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            cv_entry = supp_cv.get(m_id)
            if not cv_entry or cv_entry.get("status") != "completed":
                continue
            cv_p = cv_entry["pooled"]
            cv_t = cv_entry.get("timing", {})
            lines.append(
                f"| `{m_id}` | `{cv_entry['run_id']}` | {_fmt_float(cv_p['accuracy'])} | {_fmt_float(cv_p['macro_f1'])} | "
                f"`{cv_entry.get('total_candidates_evaluated', 'N/A')}` | `{cv_entry.get('skipped_candidate_count', 0)}` | "
                f"{_fmt_float(cv_t.get('cv_search_total_seconds'), 2)} |"
            )

    # Timing & Environment Section
    timing_comp: Mapping[str, Any] = combined_results.get("timing_comparison", {})
    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Runtime, Timing Scope, and Environment Metadata",
            "",
            f"- **Timing Comparable for Ranking?**: `{timing_comp.get('comparable_for_ranking', False)}`",
            f"- **Timing Scope Note**: {timing_comp.get('note', 'N/A')}",
            "",
            "| Model ID | Fit Scope | Batch Size (Rows) | Fit Seconds | Predict Seconds | Latency (ms / row) | Python / sklearn / Platform |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
        m_entry = models_map.get(m_id)
        if not m_entry or m_entry.get("status") != "completed":
            continue
        t_block = m_entry.get("timing", {})
        env = m_entry.get("environment", {})
        libs = env.get("libraries", {})
        fit_scope = m_entry.get("preprocessing", {}).get("fit_scope", "unknown")
        rows_n = int(m_entry["pooled"]["evaluated_row_count"])
        p_sec = float(t_block.get("predict_seconds", 0.0))
        ms_row = (p_sec * 1000.0) / max(1, rows_n)
        lines.append(
            f"| `{m_id}` | `{fit_scope}` | `{rows_n}` | {_fmt_float(t_block.get('fit_seconds'), 4)} | "
            f"{_fmt_float(p_sec, 4)} | {_fmt_float(ms_row, 3)} | "
            f"Py `{env.get('python_version', 'N/A')}`, sklearn `{libs.get('scikit_learn', 'N/A')}`, `{env.get('machine', 'N/A')}` |"
        )

    # Model-Selection Discipline & Statistical Honesty
    disc = combined_results.get("model_selection_discipline", {})
    lines.extend(
        [
            "",
            "---",
            "",
            "## 8. Model-Selection Discipline, Limitations, and Provenance Hashes",
            "",
            f"- **Selection Protocol**: {disc.get('selection_rule_statement', 'All hyperparameter and uniform-vs-weighted kNN choices are made strictly on dev_train CV folds.')}",
            f"- **Frozen Holdout Policy**: {disc.get('frozen_holdout_statement', 'Holdout evaluations compare frozen pipelines without retuning.')}",
            f"- **Statistical Reporting Policy**: {disc.get('statistical_reporting_statement', 'No parametric confidence intervals or p-values are fabricated from correlated CV folds.')}",
            "",
            "### 8.1 Source Artifact Provenance",
            "",
            "| Model ID | Source `metrics.json` Path | Source `predictions.csv` Path | Building 0 Artifact | Building 1 Artifact | Building 2 Artifact |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
        m_entry = models_map.get(m_id)
        if not m_entry or m_entry.get("status") != "completed":
            continue
        arts = m_entry.get("model_artifact_paths", {})
        b0 = Path(arts.get("0", "N/A")).name if isinstance(arts, Mapping) and "0" in arts else "N/A"
        b1 = Path(arts.get("1", "N/A")).name if isinstance(arts, Mapping) and "1" in arts else "N/A"
        b2 = Path(arts.get("2", "N/A")).name if isinstance(arts, Mapping) and "2" in arts else "N/A"
        lines.append(
            f"| `{m_id}` | `{m_entry.get('source_metrics_json_path', 'N/A')}` | "
            f"`{m_entry.get('prediction_file_path', 'N/A')}` | `{b0}` | `{b1}` | `{b2}` |"
        )

    warnings_list = combined_results.get("warnings", [])
    if warnings_list:
        lines.extend(["", "### 8.2 Recorded Warnings", ""])
        for w in warnings_list:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


def build_academic_writeup_markdown(
    combined_results: Mapping[str, Any] | None = None,
    *,
    length_mode: str = "two-page",
) -> str:
    'Generate a 1-page or 2-page academic write-up Markdown document from `combined_results.json`.'
    if length_mode not in ("one-page", "two-page"):
        raise ValueError(f"length_mode must be 'one-page' or 'two-page', got {length_mode!r}.")

    is_completed = bool(combined_results and combined_results.get("status") == "completed")
    eval_role = str(combined_results.get("evaluation_role", "PENDING")) if combined_results else "PENDING"
    eval_n = str(combined_results.get("evaluated_row_count", "PENDING")) if combined_results else "PENDING"
    ds_hash = str(combined_results.get("dataset_hash", "PENDING"))[:16] if combined_results else "PENDING"
    sp_hash = str(combined_results.get("split_manifest_hash", "PENDING"))[:16] if combined_results else "PENDING"

    header_status = (
        f"**Status**: Populated from verified `{eval_role}` artifacts (`comparison_id={combined_results.get('comparison_id')}`)"
        if is_completed
        else f"**Status**: `{DRAFT_BANNER}`"
    )

    # Build results table rows from actual combined_results if available
    table_rows: list[str] = []
    bldg_rows: list[str] = []
    if is_completed and combined_results is not None:
        models_map = combined_results.get("models", {})
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            m_entry = models_map.get(m_id, {})
            if m_entry.get("status") == "completed":
                p = m_entry["pooled"]
                corr = int(p["evaluated_row_count"]) - int(p["incorrect_count"])
                tot = int(p["evaluated_row_count"])
                table_rows.append(
                    f"| `{m_id}` | {_fmt_float(p['accuracy'])} ({_fmt_pct(p['accuracy'])}) | "
                    f"{_fmt_float(p['macro_f1'])} | {_fmt_float(p['balanced_accuracy'])} | "
                    f"`{corr}/{tot}` | {_summarize_model_params(m_id, m_entry.get('selected_parameters', {}))} |"
                )
                for b_key, pb in sorted(m_entry.get("per_building", {}).items(), key=lambda kv: int(kv[0])):
                    bldg_rows.append(
                        f"| Building {b_key} | `{m_id}` | {_fmt_float(pb['accuracy'])} | "
                        f"{_fmt_float(pb['macro_f1'])} | `{int(pb['evaluated_row_count']) - int(pb['incorrect_count'])}/{int(pb['evaluated_row_count'])}` |"
                    )
    else:
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            table_rows.append(
                f"| `{m_id}` | `[PENDING EXECUTION]` | `[PENDING EXECUTION]` | `[PENDING EXECUTION]` | `[PENDING]` | `[PENDING CV SELECTION]` |"
            )

    if length_mode == "one-page":
        lines = [
            "# Revisiting WiFi Fingerprint Floor Classification with PCA–kNN and Gradient Boosting (1-Page Version)",
            "",
            f"> {header_status}",
            "> **Note on Page-Length Conflict**: Course instructions contain a heading specifying a 1-page write-up and body text specifying a 2-page write-up. Confirm with faculty which version to export to PDF.",
            "",
            "## 1. Problem Formulation, Dataset Provenance, and Preprocessing",
            "We investigate multi-class indoor `FLOOR` classification within each known `BUILDINGID` (`0`, `1`, `2`) on the UCI UJIIndoorLoc benchmark (`DOI: 10.24432/C5MS59`; `19,937` training rows, `1,111` official validation rows, `529` columns). Strictly the 520 received signal strength indicator (RSSI) columns (`WAP001..WAP520`) enter the predictor matrix; `BUILDINGID` routes samples to building-specific classifiers and is never used as a feature, while coordinates, `SPACEID`, `USERID`, `PHONEID`, and `TIMESTAMP` are excluded to prevent target and device leakage. Undetected access points encoded as `+100` in UJIIndoorLoc are mapped to `-110.0` dBm (6 dB below the minimum observed `-104` dBm signal), whereas genuine `0` dBm measurements (`137` entries in `trainingData.csv`) are preserved as `0.0` dBm.",
            "",
            "## 2. Model Configurations and Fair Evaluation Protocol",
            "We compare two model families across three configurations on identical per-building stratified splits (`80%` `dev_train` = `15,948` rows; `20%` `internal_holdout` = `3,989` rows; shared 3-fold and 5-fold CV manifests):",
            "1. **Uniform PCA–kNN (`pca_knn`)**: `KNeighborsClassifier(weights='uniform', metric='euclidean')` with fold-local optional `StandardScaler` and `PCA(whiten=False)`, plus a fixed paper-inspired baseline (`k=1, PCA=50, unscaled`).",
            "2. **Distance-Weighted PCA–kNN (`pca_weighted_knn`)**: Controlled refinement using `weights='distance'` over the exact same `k in [1,3,5,7,9,11,15]`, `PCA in [5,50,200]` (enforcing `n_components <= min(n_fold_train - 1, 520)`), and `scaling in [none, standard]` grid. At `k=1`, distance weighting assigns 100% weight to the single neighbor and is mathematically identical to uniform 1NN.",
            "3. **Classical Gradient Boosting (`gradient_boosting`)**: `sklearn.ensemble.GradientBoostingClassifier` on raw sentinel-replaced WAP features (`scale=False, use_pca=False, n_iter_no_change=None`), starting from paper-inspired `max_depth=4, n_estimators=189` (`learning_rate=0.1` assumed) and searching a deterministic seeded subset of `n_estimators in [100,189,300]`, `learning_rate in [0.05,0.1]`, `max_depth in [3,4]`, `min_samples_leaf in [1,5]`. This represents a **full-pipeline comparison** (PCA–kNN vs. raw-feature boosting).",
            "",
            f"## 3. Experimental Results (Split: `{eval_role}`, Evaluated Rows: `{eval_n}`, Dataset Hash: `{ds_hash}...`)",
            "",
            "| Configuration | Pooled Accuracy | Pooled Macro-F1 | Balanced Acc | Correct / Total | Selected Per-Building Hyperparameters |",
            "| --- | --- | --- | --- | --- | --- |",
            *table_rows,
            "",
            "## 4. Key Findings, Limitations, and References",
            "- **Model-Selection Discipline**: Hyperparameters and uniform-vs-weighted kNN choices are selected strictly via training-side CV (`mean_cv_accuracy`, tie-breaker `mean_cv_macro_f1`). Holdout metrics evaluate frozen pipelines without retuning.",
            "- **Limitations**: The reference paper (*Indoor Positioning System Using Wifi Fingerprint*, Li et al.) incompletely specifies its boosting implementation; our `-110.0` dBm encoding and `learning_rate=0.1` are documented engineering assumptions rather than an exact numerical reproduction claim. Furthermore, `trainingData.csv` contains `76` all-undetected scans (`all WAPs == 100`) retained for diagnostic transparency.",
            "- **References**: (1) Torres-Sospedra et al., *UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems*, IPIN 2014 (`https://doi.org/10.24432/C5MS59`). (2) Dan Li, Le Wang, Shiqi Wu, *Indoor Positioning System Using Wifi Fingerprint*. (3) Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.",
        ]
        return "\n".join(lines) + "\n"

    # Two-page version
    lines = [
        "# Revisiting WiFi Fingerprint Floor Classification with PCA–kNN and Gradient Boosting (2-Page Version)",
        "",
        f"> {header_status}",
        "> **Note on Page-Length Conflict**: Course instructions contain a heading specifying a 1-page write-up and body text specifying a 2-page write-up. Confirm with faculty which version to export to PDF.",
        "",
        "## 1. Problem Statement and Known-Building Assumption",
        "Indoor floor localization from WiFi Received Signal Strength Indicator (RSSI) fingerprints is a core component of multi-story building navigation. Following the formulation in Li, Wang, and Wu (*Indoor Positioning System Using Wifi Fingerprint*), we address multi-class `FLOOR` classification separately within each known `BUILDINGID` (`0`, `1`, `2`) on the UCI UJIIndoorLoc dataset. A strict predictor allowlist enforces that only `WAP001` through `WAP520` (520 access point features in canonical order) enter the feature matrix `X`. `BUILDINGID` acts solely as a known model-routing key and is never passed as a predictor column. Metadata attributes (`LONGITUDE`, `LATITUDE`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`) are barred from `X` to prevent spatial or session leakage.",
        "",
        "## 2. Dataset Provenance, Audit Diagnostics, and Sentinel Handling",
        f"The immutable local UJIIndoorLoc files comprise `trainingData.csv` (`19,937` rows, `529` columns, SHA-256 prefix `{ds_hash}`) and `validationData.csv` (`1,111` rows, `529` columns, reserved strictly for locked final evaluation).Buildings `0` (`5,249` train rows) and `1` (`5,196` train rows) span 4 floors (`0..3`), while Building `2` (`9,492` train rows) spans 5 floors (`0..4`). Original integer floor labels are preserved throughout training and evaluation without zero-indexing.",
        "",
        "In UJIIndoorLoc, `+100` is an indicator sentinel denoting that an access point was not detected during a scan (`10,008,477` cells in `trainingData.csv`), whereas observed RSSI values lie in `[-104, 0]` dBm (including `137` genuine `0` dBm strong-signal readings). Leaving `+100` untreated distorts Euclidean distances and principal component directions by treating missing signals as stronger than `0` dBm. We replace `100 -> -110.0` dBm (an explicit project assumption 6 dB below `-104` dBm) while preserving `0 -> 0.0` dBm. Non-destructive dataset auditing also identified `76` all-undetected training rows (all 520 WAPs equal to `100`), which are retained and logged rather than silently deleted.",
        "",
        "## 3. Model Families, Feasibility Safeguards, and Leakage Prevention",
        "All development experiments share a single seeded per-building stratified split manifest (`shared/manifests/development_splits.csv`, SHA-256 prefix `{sp_hash}`): `80%` `dev_train` (`15,948` rows) and `20%` `internal_holdout` (`3,989` rows), partitioned into shared 3-fold (`lightweight`) and 5-fold (`full`) stratified CV folds.",
        "",
        "1. **Model 1 — Uniform PCA–kNN (`pca_knn`)**: `SentinelReplacementTransformer(100 -> -110.0)` -> optional `StandardScaler` -> `PCA(n_components, whiten=False)` -> `KNeighborsClassifier(weights='uniform', metric='euclidean')`. We evaluate both a fixed paper-inspired baseline (`k=1, PCA=50, scaling='none'`) and a tuned grid over `k in [1, 3, 5, 7, 9, 11, 15]`, `PCA in [5, 50, 200]`, and `scaling in ['none', 'standard']`.",
        "2. **Model 2 — Distance-Weighted PCA–kNN (`pca_weighted_knn`)**: Controlled refinement replacing uniform voting with inverse-distance weighting (`weights='distance'`) over the identical candidate grid and CV folds. Because a single nearest neighbor (`k=1`) receives 100% weight regardless of distance, uniform and distance-weighted kNN are mathematically equivalent at `k=1`. We therefore separate (a) **matched-configuration comparisons** (`k > 1` with identical `PCA` and `scaling`) from (b) **best-selected pipeline comparisons**.",
        "3. **Model 3 — Classical Gradient Boosting (`gradient_boosting`)**: CPU-compatible `sklearn.ensemble.GradientBoostingClassifier` (no XGBoost or `HistGradientBoostingClassifier`, no early stopping) trained on finite sentinel-replaced raw WAP features (`scale=False, use_pca=False`). Starting from the paper-inspired anchor (`max_depth=4, n_estimators=189` with `learning_rate=0.1, subsample=1.0` as documented implementation assumptions), we run a bounded deterministic randomized search (`8` trials in `lightweight`, `16` trials in `full` mode) over `n_estimators in [100, 189, 300]`, `learning_rate in [0.05, 0.1]`, `max_depth in [3, 4]`, and `min_samples_leaf in [1, 5]`.",
        "",
        "**Leakage & Feasibility Enforcement**: Scaling and PCA are fit strictly inside each CV training fold and refit only on `dev_train` after selection. Every candidate is checked against `n_components <= min(n_fold_train - 1, n_features)` and `1 <= k <= n_fold_train`; infeasible candidates are recorded with explicit skip reasons rather than silently clamped.",
        "",
        f"## 4. Quantitative Results (Split: `{eval_role}`, Total Evaluated Rows: `{eval_n}`)",
        "",
        "### 4.1 Pooled Multi-Model Summary",
        "Pooled accuracy is computed strictly as `total_correct / total_evaluated` across all three buildings (never as an unweighted average of per-building accuracies):",
        "",
        "| Configuration | Pooled Accuracy | Pooled Macro-F1 | Balanced Acc | Correct / Total | Selected Per-Building Hyperparameters |",
        "| --- | --- | --- | --- | --- | --- |",
        *table_rows,
        "",
    ]

    if bldg_rows:
        lines.extend(
            [
                "### 4.2 Per-Building Breakdown",
                "",
                "| Building | Configuration | Accuracy | Macro-F1 | Correct / Evaluated |",
                "| --- | --- | --- | --- | --- |",
                *bldg_rows,
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Discussion, Limitations, and References",
            "- **Pipeline vs. Estimator Attribution**: Comparing PCA–kNN against raw-feature `GradientBoostingClassifier` evaluates full end-to-end pipelines rather than isolated classifiers on identical PCA features. Axis-aligned regression trees partition raw sparse RSSI thresholds directly without requiring orthogonal linear projection.",
            "- **Model-Selection Discipline**: Hyperparameters are chosen exclusively on `dev_train` CV folds (`mean_cv_accuracy`, tie-breaker `mean_cv_macro_f1`). Holdout scores evaluate frozen pipelines; we do not retune on holdout data or fabricate parametric confidence intervals from correlated CV folds.",
            "- **Reproduction Scope**: Because the reference paper omits key boosting hyperparameters (learning rate, leaf constraints, sentinel replacement value, and exact split indices), our experiments provide a transparent, reproducible benchmark rather than a claim of exact numerical score reproduction.",
            "- **References**:",
            "  1. J. Torres-Sospedra et al., *UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems*, IEEE IPIN, 2014. DOI: `10.24432/C5MS59`.",
            "  2. D. Li, L. Wang, and S. Wu, *Indoor Positioning System Using Wifi Fingerprint*, Stanford CS229 Project Report.",
            "  3. F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825–2830, 2011.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_presentation_slides_markdown(
    combined_results: Mapping[str, Any] | None = None,
) -> str:
    """Build a 10-slide Marp/Pandoc-compatible presentation deck from `combined_results.json`."""
    is_completed = bool(combined_results and combined_results.get("status") == "completed")
    eval_role = str(combined_results.get("evaluation_role", "PENDING")) if combined_results else "PENDING"
    status_line = (
        f"Verified `{eval_role}` Results (`comparison_id={combined_results.get('comparison_id')}`)"
        if is_completed
        else DRAFT_BANNER
    )

    pooled_rows: list[str] = []
    if is_completed and combined_results is not None:
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            m_entry = combined_results.get("models", {}).get(m_id, {})
            if m_entry.get("status") == "completed":
                p = m_entry["pooled"]
                pooled_rows.append(
                    f"| `{m_id}` | {_fmt_float(p['accuracy'])} ({_fmt_pct(p['accuracy'])}) | "
                    f"{_fmt_float(p['macro_f1'])} | {_fmt_float(p['balanced_accuracy'])} | "
                    f"`{int(p['evaluated_row_count']) - int(p['incorrect_count'])}/{int(p['evaluated_row_count'])}` |"
                )
    if not pooled_rows:
        for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting"):
            pooled_rows.append(f"| `{m_id}` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |")

    slides = [
        "---",
        "marp: true",
        "theme: default",
        "paginate: true",
        "title: Revisiting WiFi Fingerprint Floor Classification with PCA-kNN and Gradient Boosting",
        "---",
        "",
        "# Slide 1: Revisiting WiFi Fingerprint Floor Classification with PCA–kNN and Gradient Boosting",
        f"- **Dataset**: UCI UJIIndoorLoc (`DOI: 10.24432/C5MS59`)",
        "- **Prediction Task**: Multi-class `FLOOR` classification within each known `BUILDINGID` (`0`, `1`, `2`)",
        "- **Team Members**: `[Member 1 Name & Roll ID — Pending]` | `[Member 2 Name & Roll ID — Pending]`",
        "- **Assigned Problem / Section**: `[Problem Serial & Section — Pending]`",
        f"- **Deck Status**: `{status_line}`",
        "",
        "---",
        "",
        "# Slide 2: Problem Formulation & Strict Predictor Contract",
        "- **Known-Building Routing**: `BUILDINGID` (`0`, `1`, `2`) selects the building-specific classifier; it is **never** a predictor feature.",
        "- **Predictor Allowlist**: Strictly `WAP001..WAP520` (520 WiFi RSSI columns in canonical order).",
        "- **Leakage Prevention**: `LONGITUDE`, `LATITUDE`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, and `TIMESTAMP` are excluded from predictors.",
        "- **Original Labels Preserved**: Buildings `0` & `1` have floors `0..3`; Building `2` has floors `0..4` (no zero-indexing).",
        "",
        "---",
        "",
        "# Slide 3: Dataset Provenance & Signal Sentinel Encoding",
        "- **Verified Local Files**:",
        "  - `trainingData.csv`: `19,937` rows, `529` columns (`SHA-256: 45ca0128...`)",
        "  - `validationData.csv`: `1,111` rows, `529` columns (`SHA-256: 5f90c536...`, reserved for final evaluation)",
        "- **Sentinel `100` vs. Genuine `0` dBm**:",
        "  - `100` means the access point was **not detected** (`10,008,477` cells).",
        "  - Observed RSSI spans `[-104, 0]` dBm, including `137` genuine `0` dBm measurements.",
        "  - **Shared Encoding**: `100 -> -110.0` dBm (documented project assumption); `0` dBm remains `0.0` dBm.",
        "- **Diagnostic Audit**: `76` all-undetected training scans retained and logged.",
        "",
        "---",
        "",
        "# Slide 4: Shared Splits, Leakage Prevention, & Feasibility Bounds",
        "- **Development Split (`shared/manifests/development_splits.csv`)**:",
        "  - Stratified by `(BUILDINGID, FLOOR)` with `seed=42`: `15,948` `dev_train` (80%) + `3,989` `internal_holdout` (20%).",
        "  - Shared 3-fold (`lightweight`) and 5-fold (`full`) CV assignments.",
        "- **In-Fold Preprocessing**: `StandardScaler` and `PCA(whiten=False)` are fit **strictly inside each CV training fold**.",
        "- **Feasibility Safeguards**:",
        "  - PCA bound: `1 <= n_components <= min(n_fold_train - 1, 520)`",
        "  - kNN bound: `1 <= n_neighbors <= n_fold_train`",
        "  - Infeasible candidates are logged with explicit `reason_code`s, never silently clamped.",
        "",
        "---",
        "",
        "# Slide 5: Two Model Families & Three Comparable Configurations",
        "1. **Model 1 — Uniform PCA–kNN (`pca_knn`)**:",
        "   - `KNeighborsClassifier(weights='uniform', metric='euclidean')`",
        "   - Fixed baseline (`k=1, PCA=50, scaling=none`) + tuned grid (`k in [1..15]`, `PCA in [5, 50, 200]`, `scaling in [none, standard]`).",
        "2. **Model 2 — Distance-Weighted PCA–kNN (`pca_weighted_knn`)**:",
        "   - Controlled refinement with `weights='distance'` over the exact same grid and CV folds.",
        "3. **Model 3 — Classical Gradient Boosting (`gradient_boosting`)**:",
        "   - `sklearn.ensemble.GradientBoostingClassifier` on raw finite WAP features (`scale=False, use_pca=False, n_iter_no_change=None`).",
        "   - Paper-inspired anchor (`max_depth=4, n_estimators=189, lr=0.1`) + bounded deterministic search (`8` lightweight / `16` full trials out of `24`).",
        "",
        "---",
        "",
        "# Slide 6: Fair Comparison Design & Validation Curves over `k`",
        "- **Supervised CV Validation Curve (Not an Unsupervised Elbow Plot)**:",
        "  - Neighbor count `k` is selected deterministically by maximizing `mean_cv_accuracy` (tie-breaker `mean_cv_macro_f1`).",
        "  - Validation curves plot fixed `(building_id, PCA, scaling)` panels so different preprocessing settings are never mixed into one line.",
        "- **Two-Part kNN Analysis**:",
        "  - **Analysis 1 (Matched Settings)**: Same `(k, PCA, scaling)` where only `weights` differs (`k > 1`).",
        "  - **Analysis 2 (Best-Selected Pipelines)**: Training-CV-selected uniform vs. weighted pipelines.",
        "  - **At `k=1`**: Distance weighting gives 100% weight to the single neighbor (`uniform == distance`).",
        "",
        "---",
        "",
        f"# Slide 7: Frozen Evaluation Results (Split: `{eval_role}`)",
        "",
        "| Model Configuration | Pooled Accuracy | Pooled Macro-F1 | Balanced Accuracy | Correct / Evaluated |",
        "| --- | --- | --- | --- | --- |",
        *pooled_rows,
        "",
        "- **Full-Pipeline Comparison Disclosure**: `pca_knn` and `pca_weighted_knn` operate on PCA-projected features, whereas `gradient_boosting` operates on 520 raw sentinel-replaced WAP features.",
        "",
        "---",
        "",
        "# Slide 8: Per-Building Error Analysis, Confusion Matrices, & Runtime",
        "- **Original-Label Confusion Matrices**:",
        "  - Generated from `combined_results.json` in `comparison/results/<comparison_id>/figures/`.",
        "  - Inspects adjacent-floor vs. multi-floor misclassifications per building (`0..3` in B0/B1, `0..4` in B2).",
        "- **Runtime Trade-Off**:",
        "  - **PCA–kNN**: Fast fitting (`PCA` SVD + lazy storage), higher per-query distance computation cost at inference.",
        "  - **Gradient Boosting**: Higher training cost (`189–300` sequential boosting stages across classes), fast tree-traversal inference.",
        "",
        "---",
        "",
        "# Slide 9: Offline Live Demo (`python -m shared.demo`)",
        "- **100% Offline & Deterministic**:",
        "  - Loads saved per-building pipelines via `load_trusted_pipeline_artifact` (verifying `.metadata.json` and SHA-256 digest).",
        "  - Routes each scan by known `BUILDINGID` and predicts original `FLOOR` labels from `WAP001..WAP520` without retraining.",
        "  - Withholds optional ground-truth `FLOOR` from predictors and uses it only for post-prediction display.",
        "- **Rehearsal Command**:",
        "  ```powershell",
        "  .\\.venv\\Scripts\\python.exe -m shared.demo --model-id pca_weighted_knn --train-run-id knn_weighted_full_dev --variant tuned --use-dev-rehearsal",
        "  ```",
        "",
        "---",
        "",
        "# Slide 10: Limitations, Reproduction Scope, & Conclusions",
        "- **Reproduction Honesty**:",
        "  - Li et al. do not specify learning rate, leaf regularization, sentinel replacement value, or CV fold indices for boosting; our results represent a controlled modern benchmark rather than an exact numerical claim.",
        "- **Model-Selection Discipline**:",
        "  - All hyperparameter choices are frozen on `dev_train` CV before holdout testing; no post-hoc retuning.",
        "- **Summary**:",
        "  - Controlled evaluation isolates the effect of `PCA`, `StandardScaler`, `k`, inverse-distance weighting (`k > 1`), and raw-feature tree boosting under strict anti-leakage guarantees.",
    ]
    return "\n".join(slides) + "\n"


def generate_academic_materials(
    *,
    combined_results_path: Path | str | None = None,
    output_reports_dir: Path | str = REPORTS_DIR,
    output_presentation_dir: Path | str = PRESENTATION_DIR,
    length_mode: str = "both",
) -> dict[str, Any]:
    """Generate 1-page / 2-page write-up markdown files and presentation slides from `combined_results.json`."""
    combined_data: Mapping[str, Any] | None = None
    if combined_results_path is not None:
        p = Path(combined_results_path)
        if not p.is_file():
            raise FileNotFoundError(f"combined_results.json not found at: {p}")
        combined_data = json.loads(p.read_text(encoding="utf-8"))

    rep_dir = Path(output_reports_dir)
    pres_dir = Path(output_presentation_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    pres_dir.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, str] = {}
    modes = ("one-page", "two-page") if length_mode == "both" else (length_mode,)
    for m in modes:
        suffix = "1page" if m == "one-page" else "2page"
        out_path = rep_dir / f"writeup_{suffix}.md"
        content = build_academic_writeup_markdown(combined_data, length_mode=m)
        out_path.write_text(content, encoding="utf-8")
        written_files[f"writeup_{suffix}"] = str(out_path.resolve())

    slides_out = pres_dir / "slides_populated.md"
    slides_content = build_presentation_slides_markdown(combined_data)
    slides_out.write_text(slides_content, encoding="utf-8")
    written_files["slides_populated"] = str(slides_out.resolve())

    return {
        "status": "completed" if (combined_data and combined_data.get("status") == "completed") else "draft_pending_results",
        "source_combined_results": str(Path(combined_results_path).resolve()) if combined_results_path else None,
        "written_files": written_files,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `python -m comparison.report_builder`."""
    parser = argparse.ArgumentParser(
        prog="python -m comparison.report_builder",
        description=(
            "Build Markdown comparison reports, 1-page/2-page academic write-ups, and presentation "
            "slides from an actual combined_results.json file (or in explicit --draft mode)."
        ),
    )
    parser.add_argument(
        "--combined-results",
        type=Path,
        default=None,
        help="Explicit path to comparison/results/<comparison_id>/combined_results.json.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Generate explicit DRAFT — RESULTS PENDING write-up and slide files before experiments are run.",
    )
    parser.add_argument(
        "--length-mode",
        type=str,
        choices=["one-page", "two-page", "both"],
        default="both",
        help="Academic write-up length variant to generate (default: both).",
    )
    parser.add_argument(
        "--output-report-md",
        type=Path,
        default=None,
        help="Optional explicit output path for comparison_report.md.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for generated writeup_1page.md and writeup_2page.md files.",
    )
    parser.add_argument(
        "--presentation-dir",
        type=Path,
        default=PRESENTATION_DIR,
        help="Directory for generated slides_populated.md file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m comparison.report_builder`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.combined_results is None and not args.draft:
        raise ValueError(
            "Pass `--combined-results <path_to_combined_results.json>` or explicit `--draft`."
        )

    if args.combined_results is not None:
        c_path = Path(args.combined_results)
        if not c_path.is_file():
            raise FileNotFoundError(f"combined_results.json not found: {c_path}")
        c_data = json.loads(c_path.read_text(encoding="utf-8"))
        rep_md = build_comparison_report_markdown(c_data)
        target_md = args.output_report_md or (c_path.parent / "comparison_report.md")
        Path(target_md).parent.mkdir(parents=True, exist_ok=True)
        Path(target_md).write_text(rep_md, encoding="utf-8")

    summary = generate_academic_materials(
        combined_results_path=args.combined_results,
        output_reports_dir=args.reports_dir,
        output_presentation_dir=args.presentation_dir,
        length_mode=args.length_mode,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
