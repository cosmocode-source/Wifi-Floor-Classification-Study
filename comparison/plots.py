'Plot generation from verified comparison results and training CV search artifacts.'

from __future__ import annotations

from html import escape as _xml_escape
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zlib

import numpy as np
import pandas as pd


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "pca_knn": "PCA + Uniform kNN",
    "pca_weighted_knn": "PCA + Weighted kNN",
    "gradient_boosting": "Classical Gradient Boosting (Raw WAPs)",
}

MODEL_COLORS_HEX: dict[str, str] = {
    "pca_knn": "#2563eb",
    "pca_weighted_knn": "#0d9488",
    "gradient_boosting": "#d97706",
}


def _write_deterministic_png_from_canvas(
    rgb_canvas: np.ndarray,
    output_path: Path,
) -> Path:
    """Encode an `(H, W, 3)` uint8 RGB array as a valid standard PNG file using stdlib `zlib`."""
    height, width, channels = rgb_canvas.shape
    assert channels == 3
    # Prepend filter byte 0 (None) to each scanline
    filter_col = np.zeros((height, 1), dtype=np.uint8)
    raw_scanlines = np.hstack([filter_col, rgb_canvas.reshape(height, width * 3)]).tobytes()

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    png_sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, level=6)
    iend = b""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(png_sig)
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"IDAT", idat))
        fh.write(_png_chunk(b"IEND", iend))
    return output_path


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _render_bar_canvas_png(
    values_by_group: Sequence[tuple[str, Sequence[tuple[str, float, str]]]],
    output_path: Path,
    width: int = 960,
    height: int = 540,
) -> Path:
    """Render a clean deterministic raster PNG bar chart directly from numeric values."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=120)
        group_names = [g[0] for g in values_by_group]
        series_defs = values_by_group[0][1] if values_by_group else []
        x = np.arange(len(group_names), dtype=float)
        n_series = max(1, len(series_defs))
        bar_w = 0.72 / n_series

        for s_idx in range(n_series):
            s_label = series_defs[s_idx][0]
            s_color = series_defs[s_idx][2]
            s_vals = [float(g[1][s_idx][1]) for g in values_by_group]
            offsets = x - 0.36 + (s_idx + 0.5) * bar_w
            rects = ax.bar(offsets, s_vals, width=bar_w * 0.9, label=s_label, color=s_color)
            for rect, val in zip(rects, s_vals):
                ax.annotate(
                    f"{val:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, fontsize=9)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
    except Exception:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        # Draw plot border
        left, right, top, bottom = 80, width - 40, 50, height - 70
        canvas[top:bottom, left:left + 2] = (60, 60, 60)
        canvas[bottom - 2:bottom, left:right] = (60, 60, 60)
        n_groups = max(1, len(values_by_group))
        group_span = (right - left) // n_groups
        for g_idx, (_, bars) in enumerate(values_by_group):
            n_bars = max(1, len(bars))
            bar_span = max(8, (group_span - 24) // n_bars)
            g_x0 = left + 12 + g_idx * group_span
            for b_idx, (_, val, hex_col) in enumerate(bars):
                clamped = float(np.clip(val, 0.0, 1.0))
                b_h = int(round(clamped * (bottom - top - 20)))
                bx0 = g_x0 + b_idx * bar_span
                bx1 = min(right - 2, bx0 + max(4, bar_span - 4))
                by0 = max(top, bottom - b_h)
                canvas[by0:bottom - 2, bx0:bx1] = _hex_to_rgb(hex_col)
        return _write_deterministic_png_from_canvas(canvas, output_path)


def plot_accuracy_and_macro_f1_comparison(
    combined_results: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Plot pooled and per-building Accuracy and Macro-F1 across the three compared pipelines."""
    models_block: Mapping[str, Any] = combined_results.get("models", {})
    completed_models = [
        m_id for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting")
        if m_id in models_block and models_block[m_id].get("status") == "completed"
    ]
    if not completed_models:
        return {
            "figure_id": "accuracy_macro_f1_comparison",
            "status": "skipped",
            "reason": "No completed models available in combined_results.json.",
            "paths": [],
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "accuracy_macro_f1_comparison.svg"
    png_path = out_dir / "accuracy_macro_f1_comparison.png"

    eval_role = str(combined_results.get("evaluation_role", "unknown"))
    total_support = int(combined_results.get("evaluated_row_count", 0))

    # Build groups: Pooled, Building 0, Building 1, Building 2
    all_buildings = sorted(
        {
            str(b)
            for m_id in completed_models
            for b in models_block[m_id].get("per_building", {}).keys()
        },
        key=lambda x: int(x) if x.isdigit() else x,
    )

    width, height = 1060, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 105, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    groups: list[tuple[str, int, list[tuple[str, float, float, str]]]] = []
    # Each group: (label, support, [(model_id, accuracy, macro_f1, color), ...])
    pooled_entries: list[tuple[str, float, float, str]] = []
    for m_id in completed_models:
        p = models_block[m_id]["pooled"]
        pooled_entries.append(
            (m_id, float(p["accuracy"]), float(p["macro_f1"]), MODEL_COLORS_HEX.get(m_id, "#4b5563"))
        )
    groups.append(("Pooled (All Buildings)", total_support, pooled_entries))

    for b_key in all_buildings:
        b_sup = 0
        b_entries: list[tuple[str, float, float, str]] = []
        for m_id in completed_models:
            pb = models_block[m_id].get("per_building", {}).get(b_key)
            if pb:
                b_sup = int(pb["evaluated_row_count"])
                b_entries.append(
                    (
                        m_id,
                        float(pb["accuracy"]),
                        float(pb["macro_f1"]),
                        MODEL_COLORS_HEX.get(m_id, "#4b5563"),
                    )
                )
        groups.append((f"Building {b_key}", b_sup, b_entries))

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#111827">'
        f'Floor Classification Accuracy &amp; Macro-F1 Comparison ({_xml_escape(eval_role)}, N={total_support})</text>',
        f'<text x="{width / 2:.1f}" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">'
        'Full-Pipeline Comparison: PCA-Reduced Euclidean kNN (Uniform vs. Distance-Weighted) vs. Raw-WAP Classical Gradient Boosting</text>',
        f'<text x="{width / 2:.1f}" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">'
        'Solid bars = Accuracy | Hatched/lighter bars = Macro-F1 | Computed strictly from row-aligned predictions</text>',
    ]

    # Y-axis gridlines
    for tick in np.linspace(0.0, 1.0, 6):
        y = margin_top + plot_h * (1.0 - float(tick))
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{tick:.2f}</text>'
        )

    svg_lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1.5"/>'
    )
    svg_lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1.5"/>'
    )

    n_groups = len(groups)
    group_w = plot_w / max(1, n_groups)

    png_groups: list[tuple[str, list[tuple[str, float, str]]]] = []

    for g_idx, (g_label, g_sup, entries) in enumerate(groups):
        gx0 = margin_left + g_idx * group_w
        inner_w = group_w * 0.82
        start_x = gx0 + (group_w - inner_w) / 2.0
        n_bars = max(1, len(entries) * 2)
        bar_w = inner_w / n_bars

        png_bars: list[tuple[str, float, str]] = []

        for m_idx, (m_id, acc_val, f1_val, color_hex) in enumerate(entries):
            # Accuracy bar
            bx_acc = start_x + (2 * m_idx) * bar_w
            bh_acc = max(1.0, float(np.clip(acc_val, 0.0, 1.0)) * plot_h)
            by_acc = margin_top + plot_h - bh_acc
            svg_lines.append(
                f'<rect x="{bx_acc + 1:.1f}" y="{by_acc:.1f}" width="{max(4.0, bar_w - 2):.1f}" height="{bh_acc:.1f}" fill="{color_hex}"/>'
            )
            svg_lines.append(
                f'<text x="{bx_acc + bar_w / 2:.1f}" y="{by_acc - 5:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9.5" font-weight="bold" fill="#111827">{acc_val:.3f}</text>'
            )

            # Macro-F1 bar
            bx_f1 = start_x + (2 * m_idx + 1) * bar_w
            bh_f1 = max(1.0, float(np.clip(f1_val, 0.0, 1.0)) * plot_h)
            by_f1 = margin_top + plot_h - bh_f1
            svg_lines.append(
                f'<rect x="{bx_f1 + 1:.1f}" y="{by_f1:.1f}" width="{max(4.0, bar_w - 2):.1f}" height="{bh_f1:.1f}" fill="{color_hex}" fill-opacity="0.55" stroke="{color_hex}" stroke-width="1"/>'
            )
            svg_lines.append(
                f'<text x="{bx_f1 + bar_w / 2:.1f}" y="{by_f1 - 5:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9.5" fill="#374151">{f1_val:.3f}</text>'
            )

            png_bars.append((f"{MODEL_DISPLAY_NAMES.get(m_id, m_id)} (Acc)", acc_val, color_hex))
            png_bars.append((f"{MODEL_DISPLAY_NAMES.get(m_id, m_id)} (F1)", f1_val, color_hex))

        png_groups.append((f"{g_label}\n(n={g_sup})", png_bars))

        cx = gx0 + group_w / 2.0
        svg_lines.append(
            f'<text x="{cx:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#111827">{_xml_escape(g_label)}</text>'
        )
        svg_lines.append(
            f'<text x="{cx:.1f}" y="{margin_top + plot_h + 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10.5" fill="#4b5563">(n = {g_sup})</text>'
        )

    # Legend
    leg_y = height - 48
    for idx, m_id in enumerate(completed_models):
        lx = margin_left + idx * 310
        col = MODEL_COLORS_HEX.get(m_id, "#4b5563")
        disp = MODEL_DISPLAY_NAMES.get(m_id, m_id)
        svg_lines.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="14" fill="{col}"/>')
        svg_lines.append(
            f'<rect x="{lx + 20}" y="{leg_y}" width="16" height="14" fill="{col}" fill-opacity="0.55" stroke="{col}" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{lx + 42}" y="{leg_y + 12}" font-family="Arial, sans-serif" font-size="11.5" fill="#111827">{_xml_escape(disp)} (Solid: Acc, Shaded: Macro-F1)</text>'
        )

    svg_lines.append("</svg>")
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")
    _render_bar_canvas_png(png_groups, png_path)

    return {
        "figure_id": "accuracy_macro_f1_comparison",
        "status": "completed",
        "evaluation_role": eval_role,
        "evaluated_row_count": total_support,
        "paths": [str(svg_path.resolve()), str(png_path.resolve())],
    }


def _render_confusion_matrices_svg_and_png(
    matrices: Sequence[tuple[str, Sequence[int], Sequence[Sequence[int]], int, str]],
    title: str,
    subtitle: str,
    svg_path: Path,
    png_path: Path,
) -> None:
    """Render a grid of original-label confusion matrices to SVG and PNG."""
    n_panels = max(1, len(matrices))
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    panel_w, panel_h = 340, 320
    width = cols * panel_w + 60
    height = rows * panel_h + 110

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="bold" fill="#111827">{_xml_escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="54" text-anchor="middle" font-family="Arial, sans-serif" font-size="11.5" fill="#4b5563">{_xml_escape(subtitle)}</text>',
    ]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for idx, (panel_title, labels, matrix, support, acc_text) in enumerate(matrices):
        r_idx = idx // cols
        c_idx = idx % cols
        px0 = 30 + c_idx * panel_w
        py0 = 75 + r_idx * panel_h

        svg_lines.append(
            f'<text x="{px0 + panel_w / 2:.1f}" y="{py0 + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12.5" font-weight="bold" fill="#111827">{_xml_escape(panel_title)}</text>'
        )
        svg_lines.append(
            f'<text x="{px0 + panel_w / 2:.1f}" y="{py0 + 34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10.5" fill="#374151">n={support} | {_xml_escape(acc_text)} | Original FLOOR Labels</text>'
        )

        n_lbl = max(1, len(labels))
        grid_size = 210
        cell_s = grid_size / n_lbl
        gx0 = px0 + 70
        gy0 = py0 + 52

        mat_arr = np.asarray(matrix, dtype=int)
        max_cell = max(1, int(np.max(mat_arr)) if mat_arr.size > 0 else 1)

        for i, row_lbl in enumerate(labels):
            cy = gy0 + i * cell_s
            svg_lines.append(
                f'<text x="{gx0 - 8:.1f}" y="{cy + cell_s / 2 + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10.5" font-weight="bold" fill="#1f2937">True {row_lbl}</text>'
            )
            for j, col_lbl in enumerate(labels):
                cx = gx0 + j * cell_s
                val = int(mat_arr[i, j])
                intensity = float(val) / float(max_cell)
                if i == j:
                    # Diagonal correct cells: blue tint
                    r_c = int(round(239 - intensity * 180))
                    g_c = int(round(246 - intensity * 110))
                    b_c = int(round(255 - intensity * 35))
                else:
                    # Off-diagonal errors: warm coral tint when non-zero
                    r_c = 255
                    g_c = int(round(248 - intensity * 140)) if val > 0 else 250
                    b_c = int(round(246 - intensity * 150)) if val > 0 else 250
                fill_hex = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
                txt_col = "#ffffff" if (i == j and intensity > 0.65) else "#111827"

                svg_lines.append(
                    f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_s:.1f}" height="{cell_s:.1f}" fill="{fill_hex}" stroke="#9ca3af" stroke-width="1"/>'
                )
                svg_lines.append(
                    f'<text x="{cx + cell_s / 2:.1f}" y="{cy + cell_s / 2 + 4:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{txt_col}">{val}</text>'
                )

                # Fill PNG canvas cell
                ix0, ix1 = int(round(cx)), int(round(cx + cell_s))
                iy0, iy1 = int(round(cy)), int(round(cy + cell_s))
                canvas[iy0:iy1, ix0:ix1] = (r_c, g_c, b_c)
                canvas[iy0:iy0 + 1, ix0:ix1] = (156, 163, 175)
                canvas[iy0:iy1, ix0:ix0 + 1] = (156, 163, 175)

        for j, col_lbl in enumerate(labels):
            cx = gx0 + j * cell_s
            svg_lines.append(
                f'<text x="{cx + cell_s / 2:.1f}" y="{gy0 + grid_size + 18:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10.5" font-weight="bold" fill="#1f2937">Pred {col_lbl}</text>'
            )

    svg_lines.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")
    _write_deterministic_png_from_canvas(canvas, png_path)


def plot_confusion_matrices(
    combined_results: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Generate pooled and per-building confusion matrix plots preserving original FLOOR labels."""
    models_block: Mapping[str, Any] = combined_results.get("models", {})
    completed_models = [
        m_id for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting")
        if m_id in models_block and models_block[m_id].get("status") == "completed"
    ]
    if not completed_models:
        return {
            "figure_id": "confusion_matrices",
            "status": "skipped",
            "reason": "No completed models available for confusion matrix plotting.",
            "paths": [],
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_role = str(combined_results.get("evaluation_role", "unknown"))

    pooled_panels: list[tuple[str, list[int], list[list[int]], int, str]] = []
    per_bldg_panels: list[tuple[str, list[int], list[list[int]], int, str]] = []

    for m_id in completed_models:
        m_data = models_block[m_id]
        disp = MODEL_DISPLAY_NAMES.get(m_id, m_id)
        cms = m_data.get("confusion_matrices", {})
        if "pooled" in cms:
            cm_p = cms["pooled"]
            p_metrics = m_data["pooled"]
            pooled_panels.append(
                (
                    disp,
                    [int(x) for x in cm_p["labels"]],
                    [[int(c) for c in row] for row in cm_p["matrix"]],
                    int(p_metrics["evaluated_row_count"]),
                    f"Acc={float(p_metrics['accuracy']):.4f}, F1={float(p_metrics['macro_f1']):.4f}",
                )
            )
        for b_key in sorted(m_data.get("per_building", {}).keys(), key=lambda x: int(x) if x.isdigit() else x):
            if b_key in cms:
                cm_b = cms[b_key]
                b_metrics = m_data["per_building"][b_key]
                per_bldg_panels.append(
                    (
                        f"{disp} — Building {b_key}",
                        [int(x) for x in cm_b["labels"]],
                        [[int(c) for c in row] for row in cm_b["matrix"]],
                        int(b_metrics["evaluated_row_count"]),
                        f"Acc={float(b_metrics['accuracy']):.4f}",
                    )
                )

    paths: list[str] = []
    if pooled_panels:
        p_svg = out_dir / "confusion_matrices_pooled.svg"
        p_png = out_dir / "confusion_matrices_pooled.png"
        _render_confusion_matrices_svg_and_png(
            pooled_panels,
            title=f"Pooled Confusion Matrices with Original FLOOR Labels ({eval_role})",
            subtitle="Rows = True FLOOR, Columns = Predicted FLOOR (No Zero-Indexing of Non-Contiguous Labels)",
            svg_path=p_svg,
            png_path=p_png,
        )
        paths.extend([str(p_svg.resolve()), str(p_png.resolve())])

    if per_bldg_panels:
        b_svg = out_dir / "confusion_matrices_per_building.svg"
        b_png = out_dir / "confusion_matrices_per_building.png"
        _render_confusion_matrices_svg_and_png(
            per_bldg_panels,
            title=f"Per-Building Confusion Matrices with Original FLOOR Labels ({eval_role})",
            subtitle="Buildings 0 & 1 have 4 floors (0..3); Building 2 has 5 floors (0..4)",
            svg_path=b_svg,
            png_path=b_png,
        )
        paths.extend([str(b_svg.resolve()), str(b_png.resolve())])

    return {
        "figure_id": "confusion_matrices",
        "status": "completed",
        "evaluation_role": eval_role,
        "paths": paths,
    }


def plot_knn_validation_curves(
    *,
    uniform_curve_df: pd.DataFrame | None,
    weighted_curve_df: pd.DataFrame | None,
    output_dir: Path | str,
    selected_settings_by_building: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    'Plot uniform vs. distance-weighted kNN supervised CV validation curves over neighbor count `k`.'
    if uniform_curve_df is None or weighted_curve_df is None or uniform_curve_df.empty or weighted_curve_df.empty:
        return {
            "figure_id": "knn_validation_curves_k",
            "status": "skipped",
            "reason": "Uniform and/or weighted kNN CV validation_curve / search_results CSVs were not provided.",
            "paths": [],
        }

    def _extract_mean_rows(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "record_level" in work.columns:
            work = work[work["record_level"] == "mean_across_folds"].copy()
        if "candidate_role" in work.columns:
            work = work[work["candidate_role"] == "tuned_grid"].copy()
        if "status" in work.columns:
            work = work[work["status"] == "completed"].copy()
        return work

    u_mean = _extract_mean_rows(uniform_curve_df)
    w_mean = _extract_mean_rows(weighted_curve_df)
    if u_mean.empty or w_mean.empty:
        return {
            "figure_id": "knn_validation_curves_k",
            "status": "skipped",
            "reason": "No completed mean_across_folds records found in kNN validation curve inputs.",
            "paths": [],
        }

    buildings = sorted(set(u_mean["building_id"].astype(int)).intersection(set(w_mean["building_id"].astype(int))))
    if not buildings:
        return {
            "figure_id": "knn_validation_curves_k",
            "status": "skipped",
            "reason": "No overlapping building_id values between uniform and weighted validation curve inputs.",
            "paths": [],
        }

    panels: list[dict[str, Any]] = []
    for b_id in buildings:
        u_b = u_mean[u_mean["building_id"].astype(int) == b_id]
        w_b = w_mean[w_mean["building_id"].astype(int) == b_id]
        common_pairs = sorted(
            set(zip(u_b["n_components"].astype(int), u_b["scaling"].astype(str))).intersection(
                set(zip(w_b["n_components"].astype(int), w_b["scaling"].astype(str)))
            )
        )
        if not common_pairs:
            continue

        # Choose the building's selected (n_components, scaling) if provided and available; otherwise choose
        # the pair with highest mean_cv_accuracy across k
        chosen_pair = common_pairs[0]
        if selected_settings_by_building and str(b_id) in selected_settings_by_building:
            sel_b = selected_settings_by_building[str(b_id)]
            cand_pair = (int(sel_b.get("n_components", -1)), str(sel_b.get("scaling", "")))
            if cand_pair in common_pairs:
                chosen_pair = cand_pair
        else:
            best_score = -1.0
            for pair in common_pairs:
                sub = u_b[(u_b["n_components"].astype(int) == pair[0]) & (u_b["scaling"].astype(str) == pair[1])]
                score = float(sub["mean_cv_accuracy"].max()) if not sub.empty else -1.0
                if score > best_score:
                    best_score = score
                    chosen_pair = pair

        n_comp, sc_mode = chosen_pair
        u_slice = u_b[(u_b["n_components"].astype(int) == n_comp) & (u_b["scaling"].astype(str) == sc_mode)].sort_values("n_neighbors")
        w_slice = w_b[(w_b["n_components"].astype(int) == n_comp) & (w_b["scaling"].astype(str) == sc_mode)].sort_values("n_neighbors")

        panels.append(
            {
                "building_id": b_id,
                "n_components": n_comp,
                "scaling": sc_mode,
                "cv_folds": int(u_slice["cv_folds"].iloc[0]) if "cv_folds" in u_slice.columns else 5,
                "dev_train_rows": int(u_slice["dev_train_rows"].iloc[0]) if "dev_train_rows" in u_slice.columns else None,
                "uniform_df": u_slice,
                "weighted_df": w_slice,
            }
        )

    if not panels:
        return {
            "figure_id": "knn_validation_curves_k",
            "status": "skipped",
            "reason": "No matched fixed-(PCA, scaling) panels found between uniform and weighted kNN.",
            "paths": [],
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "knn_validation_curves_k.svg"
    png_path = out_dir / "knn_validation_curves_k.png"

    n_panels = len(panels)
    panel_w, panel_h = 350, 360
    width = n_panels * panel_w + 60
    height = panel_h + 130

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="bold" fill="#111827">'
        'Supervised Classification CV Validation Curves over Neighbor Count k (Not a Clustering Elbow Plot)</text>',
        f'<text x="{width / 2:.1f}" y="52" text-anchor="middle" font-family="Arial, sans-serif" font-size="11.5" fill="#374151">'
        'Split: dev_train Stratified CV Folds | Metric: Mean CV Accuracy (±1 Fold Std) | Fixed (PCA, Scaling) Per Panel</text>',
        f'<text x="{width / 2:.1f}" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">'
        'Note: At k=1, uniform and inverse-distance weighting assign 100% weight to the single nearest neighbor and are identical.</text>',
    ]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for idx, p in enumerate(panels):
        px0 = 30 + idx * panel_w
        py0 = 90
        gx0, gy0 = px0 + 58, py0 + 50
        gw, gh = panel_w - 82, panel_h - 110

        sup_str = f", n_dev={p['dev_train_rows']}" if p.get("dev_train_rows") else ""
        svg_lines.append(
            f'<text x="{px0 + panel_w / 2:.1f}" y="{py0 + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#111827">'
            f'Building {p["building_id"]} (Fixed PCA={p["n_components"]}, scaling={_xml_escape(str(p["scaling"]))})</text>'
        )
        svg_lines.append(
            f'<text x="{px0 + panel_w / 2:.1f}" y="{py0 + 34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10.5" fill="#4b5563">'
            f'{p["cv_folds"]}-Fold Stratified CV on dev_train{sup_str}</text>'
        )

        u_df = p["uniform_df"]
        w_df = p["weighted_df"]
        all_accs = np.concatenate(
            [
                u_df["mean_cv_accuracy"].to_numpy(dtype=float),
                w_df["mean_cv_accuracy"].to_numpy(dtype=float),
            ]
        )
        y_min = max(0.0, float(np.nanmin(all_accs)) - 0.05)
        y_max = min(1.0, float(np.nanmax(all_accs)) + 0.03)
        if y_max - y_min < 0.08:
            y_min = max(0.0, y_max - 0.10)

        k_vals = sorted(set(u_df["n_neighbors"].astype(int)).union(set(w_df["n_neighbors"].astype(int))))
        k_min, k_max = min(k_vals), max(k_vals)
        k_span = max(1, k_max - k_min)

        # Axes
        svg_lines.append(
            f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy0 + gh}" stroke="#374151" stroke-width="1.5"/>'
        )
        svg_lines.append(
            f'<line x1="{gx0}" y1="{gy0 + gh}" x2="{gx0 + gw}" y2="{gy0 + gh}" stroke="#374151" stroke-width="1.5"/>'
        )
        canvas[gy0:gy0 + gh, gx0:gx0 + 2] = (55, 65, 81)
        canvas[gy0 + gh - 2:gy0 + gh, gx0:gx0 + gw] = (55, 65, 81)

        for tick in np.linspace(y_min, y_max, 5):
            ty = gy0 + gh * (1.0 - (float(tick) - y_min) / max(1e-9, y_max - y_min))
            svg_lines.append(
                f'<line x1="{gx0}" y1="{ty:.1f}" x2="{gx0 + gw}" y2="{ty:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            svg_lines.append(
                f'<text x="{gx0 - 6}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#374151">{tick:.3f}</text>'
            )

        for kv in k_vals:
            kx = gx0 + gw * ((kv - k_min) / k_span) if len(k_vals) > 1 else gx0 + gw / 2.0
            svg_lines.append(
                f'<text x="{kx:.1f}" y="{gy0 + gh + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#1f2937">{kv}</text>'
            )
        svg_lines.append(
            f'<text x="{gx0 + gw / 2:.1f}" y="{gy0 + gh + 36}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#111827">Neighbor Count k</text>'
        )

        def _plot_series(series_df: pd.DataFrame, color_hex: str, dash: str = "") -> None:
            pts: list[tuple[float, float]] = []
            rgb = _hex_to_rgb(color_hex)
            for _, row in series_df.iterrows():
                kv = int(row["n_neighbors"])
                acc = float(row["mean_cv_accuracy"])
                std = float(row.get("std_cv_accuracy", 0.0)) if pd.notna(row.get("std_cv_accuracy")) else 0.0
                kx = gx0 + gw * ((kv - k_min) / k_span) if len(k_vals) > 1 else gx0 + gw / 2.0
                ky = gy0 + gh * (1.0 - (acc - y_min) / max(1e-9, y_max - y_min))
                pts.append((kx, ky))
                if std > 0:
                    y_hi = gy0 + gh * (1.0 - (min(y_max, acc + std) - y_min) / max(1e-9, y_max - y_min))
                    y_lo = gy0 + gh * (1.0 - (max(y_min, acc - std) - y_min) / max(1e-9, y_max - y_min))
                    svg_lines.append(
                        f'<line x1="{kx:.1f}" y1="{y_hi:.1f}" x2="{kx:.1f}" y2="{y_lo:.1f}" stroke="{color_hex}" stroke-width="1.2" stroke-opacity="0.6"/>'
                    )
                svg_lines.append(
                    f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="4" fill="{color_hex}"/>'
                )
                ix, iy = int(round(kx)), int(round(ky))
                canvas[max(0, iy - 3):min(height, iy + 4), max(0, ix - 3):min(width, ix + 4)] = rgb

            if len(pts) >= 2:
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                svg_lines.append(
                    f'<polyline fill="none" stroke="{color_hex}" stroke-width="2.2"{dash_attr} points="{poly}"/>'
                )

        _plot_series(u_df, MODEL_COLORS_HEX["pca_knn"], dash="")
        _plot_series(w_df, MODEL_COLORS_HEX["pca_weighted_knn"], dash="5,3")

    leg_y = height - 28
    svg_lines.append(
        f'<line x1="{width / 2 - 240:.1f}" y1="{leg_y - 4}" x2="{width / 2 - 210:.1f}" y2="{leg_y - 4}" stroke="{MODEL_COLORS_HEX["pca_knn"]}" stroke-width="2.5"/>'
    )
    svg_lines.append(
        f'<text x="{width / 2 - 202:.1f}" y="{leg_y}" font-family="Arial, sans-serif" font-size="11.5" fill="#111827">Uniform kNN (weights="uniform")</text>'
    )
    svg_lines.append(
        f'<line x1="{width / 2 + 20:.1f}" y1="{leg_y - 4}" x2="{width / 2 + 50:.1f}" y2="{leg_y - 4}" stroke="{MODEL_COLORS_HEX["pca_weighted_knn"]}" stroke-width="2.5" stroke-dasharray="5,3"/>'
    )
    svg_lines.append(
        f'<text x="{width / 2 + 58:.1f}" y="{leg_y}" font-family="Arial, sans-serif" font-size="11.5" fill="#111827">Distance-Weighted kNN (weights="distance")</text>'
    )

    svg_lines.append("</svg>")
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")
    _write_deterministic_png_from_canvas(canvas, png_path)

    return {
        "figure_id": "knn_validation_curves_k",
        "status": "completed",
        "curve_terminology": "supervised_classification_cv_validation_curve_not_clustering_elbow",
        "mixed_pca_scaling_lines_avoided": True,
        "panels_plotted": [
            {
                "building_id": p["building_id"],
                "n_components": p["n_components"],
                "scaling": p["scaling"],
                "cv_folds": p["cv_folds"],
            }
            for p in panels
        ],
        "paths": [str(svg_path.resolve()), str(png_path.resolve())],
    }


def plot_runtime_comparison(
    combined_results: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Plot runtime comparison ONLY when timing scope and batch support are comparable across models."""
    timing_comp: Mapping[str, Any] = combined_results.get("timing_comparison", {})
    if not timing_comp.get("comparable_for_ranking", False):
        return {
            "figure_id": "runtime_comparison",
            "status": "skipped",
            "reason": str(
                timing_comp.get(
                    "incompatibility_reason",
                    "Timing scopes across models are not comparable; skipping runtime comparison plot.",
                )
            ),
            "paths": [],
        }

    models_block: Mapping[str, Any] = combined_results.get("models", {})
    completed_models = [
        m_id for m_id in ("pca_knn", "pca_weighted_knn", "gradient_boosting")
        if m_id in models_block and models_block[m_id].get("status") == "completed"
    ]
    if not completed_models:
        return {
            "figure_id": "runtime_comparison",
            "status": "skipped",
            "reason": "No completed models available for runtime plotting.",
            "paths": [],
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "runtime_comparison.svg"
    png_path = out_dir / "runtime_comparison.png"

    eval_role = str(combined_results.get("evaluation_role", "unknown"))
    total_support = int(combined_results.get("evaluated_row_count", 0))
    scope_desc = str(timing_comp.get("shared_fit_scope", "unknown"))

    width, height = 860, 460
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 95, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    pred_secs = [float(models_block[m]["timing"]["predict_seconds"]) for m in completed_models]
    max_sec = max(1e-4, max(pred_secs) * 1.18)

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="bold" fill="#111827">'
        f'Batch Inference Runtime Comparison ({_xml_escape(eval_role)}, N={total_support} rows)</text>',
        f'<text x="{width / 2:.1f}" y="54" text-anchor="middle" font-family="Arial, sans-serif" font-size="11.5" fill="#374151">'
        f'Verified Comparable Timing Scope: {_xml_escape(scope_desc)} | CPU Sequential Execution</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1.5"/>',
    ]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[margin_top:margin_top + plot_h, margin_left:margin_left + 2] = (55, 65, 81)
    canvas[margin_top + plot_h - 2:margin_top + plot_h, margin_left:width - margin_right] = (55, 65, 81)

    group_w = plot_w / len(completed_models)
    for idx, m_id in enumerate(completed_models):
        p_sec = float(models_block[m_id]["timing"]["predict_seconds"])
        ms_per_row = (p_sec * 1000.0) / max(1, total_support)
        col = MODEL_COLORS_HEX.get(m_id, "#4b5563")
        bh = max(2.0, (p_sec / max_sec) * plot_h)
        bx = margin_left + idx * group_w + group_w * 0.25
        bw = group_w * 0.50
        by = margin_top + plot_h - bh

        svg_lines.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}"/>'
        )
        svg_lines.append(
            f'<text x="{bx + bw / 2:.1f}" y="{by - 8:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#111827">'
            f'{p_sec:.4f}s ({ms_per_row:.3f} ms/row)</text>'
        )
        svg_lines.append(
            f'<text x="{bx + bw / 2:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#111827">'
            f'{_xml_escape(MODEL_DISPLAY_NAMES.get(m_id, m_id))}</text>'
        )

        ix0, ix1 = int(round(bx)), int(round(bx + bw))
        iy0, iy1 = int(round(by)), int(round(margin_top + plot_h - 2))
        canvas[iy0:iy1, ix0:ix1] = _hex_to_rgb(col)

    svg_lines.append("</svg>")
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")
    _write_deterministic_png_from_canvas(canvas, png_path)

    return {
        "figure_id": "runtime_comparison",
        "status": "completed",
        "evaluation_role": eval_role,
        "shared_fit_scope": scope_desc,
        "paths": [str(svg_path.resolve()), str(png_path.resolve())],
    }


def generate_all_comparison_figures(
    combined_results: Mapping[str, Any],
    output_dir: Path | str,
    *,
    uniform_curve_df: pd.DataFrame | None = None,
    weighted_curve_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate all applicable comparison figures from real artifacts into `output_dir`."""
    if combined_results.get("status") != "completed":
        return {
            "status": "skipped",
            "reason": (
                f"Combined results status is {combined_results.get('status')!r}; "
                "refusing to generate placeholder charts from incomplete results."
            ),
            "figures": {},
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_block = combined_results.get("models", {})
    selected_by_bldg: dict[str, dict[str, Any]] = {}
    if "pca_weighted_knn" in models_block and models_block["pca_weighted_knn"].get("status") == "completed":
        selected_by_bldg = dict(
            models_block["pca_weighted_knn"]
            .get("selected_parameters", {})
            .get("per_building", {})
        )

    fig_acc = plot_accuracy_and_macro_f1_comparison(combined_results, out_dir)
    fig_cm = plot_confusion_matrices(combined_results, out_dir)
    fig_vc = plot_knn_validation_curves(
        uniform_curve_df=uniform_curve_df,
        weighted_curve_df=weighted_curve_df,
        output_dir=out_dir,
        selected_settings_by_building=selected_by_bldg,
    )
    fig_rt = plot_runtime_comparison(combined_results, out_dir)

    return {
        "status": "completed",
        "figures_dir": str(out_dir.resolve()),
        "figures": {
            "accuracy_macro_f1_comparison": fig_acc,
            "confusion_matrices": fig_cm,
            "knn_validation_curves_k": fig_vc,
            "runtime_comparison": fig_rt,
        },
    }
