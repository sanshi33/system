#!/usr/bin/env python3
"""
Export a publication-style figure for GB/T 5.7 standard-sphere single-point evaluation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_SVG_NAME = "journal_standard_sphere_gbt57_figure.svg"
DEFAULT_PDF_NAME = "journal_standard_sphere_gbt57_figure.pdf"
DEFAULT_PNG_NAME = "journal_standard_sphere_gbt57_figure.png"

COLORS = {
    "residual": "#0B5FA5",
    "residual_fill": "#DCE7F1",
    "spacing": "#E66101",
    "spacing_fill": "#F3D6C3",
    "target_delta": "#2F4858",
    "reference": "#4D4D4D",
    "grid": "#D7DBE0",
    "overlay_border": "#E3E6EA",
    "summary_face": "#FAFBFC",
    "summary_edge": "#D9DEE5",
}


@dataclass
class FigureData:
    result_dir: Path
    summary: Dict[str, float | str]
    points_df: pd.DataFrame
    circle_center_df: Optional[pd.DataFrame]
    registration_overlay_path: Optional[Path]
    measurement_overlay_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GB/T 5.7 standard-sphere publication figure.")
    parser.add_argument("--result-dir", type=Path, help="Result directory containing GB/T 5.7 outputs.")
    parser.add_argument("--result-root", type=Path, default=Path("result/standard_sphere_loop"))
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=178.0)
    parser.add_argument("--font-size-pt", type=float, default=8.0)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def configure_matplotlib(font_size_pt: float, svg_fonttype: str) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 0.4,
        "ytick.labelsize": font_size_pt - 0.4,
        "legend.fontsize": font_size_pt - 0.6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.9,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.4,
        "ytick.major.size": 3.4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "svg.fonttype": svg_fonttype,
    })


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.0, fontweight="bold",
            ha="left", va="bottom")


def crop_light_background_image(image: np.ndarray, threshold: float = 0.995, pad_ratio: float = 0.02) -> np.ndarray:
    if image.ndim < 2:
        return image
    rgb = image[..., :3] if image.ndim == 3 else image
    if rgb.ndim == 2:
        mask = rgb < threshold
    else:
        mask = np.any(rgb < threshold, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        mask |= image[..., 3] < threshold
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return image
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    y_pad = max(2, int((y1 - y0) * pad_ratio))
    x_pad = max(2, int((x1 - x0) * pad_ratio))
    y0 = max(0, y0 - y_pad)
    y1 = min(image.shape[0], y1 + y_pad)
    x0 = max(0, x0 - x_pad)
    x1 = min(image.shape[1], x1 + x_pad)
    return image[y0:y1, x0:x1]


def enhance_light_background_overlay(image: np.ndarray, strength: float = 1.9) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32).copy()
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr
    rgb = arr[..., :3]
    rgb = 1.0 - np.clip((1.0 - rgb) * strength, 0.0, 1.0)
    arr[..., :3] = np.clip(rgb, 0.0, 1.0)
    return arr


def load_overlay_panel_image(path: Path) -> np.ndarray:
    image = plt.imread(path)
    image = crop_light_background_image(image, threshold=0.997, pad_ratio=0.02)
    image = enhance_light_background_overlay(image, strength=1.9)
    return image


def has_required_outputs(result_dir: Path) -> bool:
    return (
        (result_dir / "standard_sphere_gbt57_p2d_summary.csv").exists()
        and (result_dir / "standard_sphere_gbt57_p2d_points.csv").exists()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    candidates = [path for path in result_root.iterdir() if path.is_dir() and has_required_outputs(path)]
    if not candidates:
        raise FileNotFoundError(f"no valid result directory under: {result_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    return args.result_dir if args.result_dir is not None else find_latest_result_dir(args.result_root)


def load_summary_csv(path: Path) -> Dict[str, float | str]:
    df = pd.read_csv(path)
    summary: Dict[str, float | str] = {}
    for _, row in df.iterrows():
        metric = str(row.get("metric", "")).strip()
        unit = str(row.get("unit", "")).strip()
        key = metric if not unit else f"{metric}__{unit}"
        value = row.get("value", np.nan)
        if isinstance(value, str):
            summary[key] = value
        else:
            summary[key] = float(value) if math.isfinite(float(value)) else value
        if metric not in summary:
            summary[metric] = summary[key]
    return summary


def load_data(result_dir: Path) -> FigureData:
    summary_path = result_dir / "standard_sphere_gbt57_p2d_summary.csv"
    points_path = result_dir / "standard_sphere_gbt57_p2d_points.csv"
    if not summary_path.exists() or not points_path.exists():
        raise FileNotFoundError("GB/T 5.7 summary/points CSV not found in result directory.")
    summary = load_summary_csv(summary_path)
    points_df = pd.read_csv(points_path)
    circle_center_path = result_dir / "standard_sphere_circle_center_global.csv"
    circle_center_df = pd.read_csv(circle_center_path) if circle_center_path.exists() else None
    registration_overlay_path = None
    for candidate in ["standard_sphere_gbt57_p2d_edge_overlay.png",
                      "standard_sphere_stitched_overlay.png",
                      "standard_sphere_gbt57_p2d_overlay.png"]:
        path = result_dir / candidate
        if path.exists():
            registration_overlay_path = path
            break
    measurement_overlay_path = None
    for candidate in ["standard_sphere_gbt57_p2d_overlay.png",
                      "standard_sphere_gbt57_p2d_edge_overlay.png",
                      "standard_sphere_stitched_overlay.png"]:
        path = result_dir / candidate
        if path.exists():
            measurement_overlay_path = path
            break
    return FigureData(result_dir, summary, points_df, circle_center_df, registration_overlay_path, measurement_overlay_path)


def metric(summary: Dict[str, float | str], name: str, unit: str = "") -> float:
    key = name if not unit else f"{name}__{unit}"
    value = summary.get(key, summary.get(name, float("nan")))
    try:
        return float(value)
    except Exception:
        return float("nan")


def selected_points_df(data: FigureData) -> pd.DataFrame:
    selected = data.points_df.copy()
    selected["selected"] = pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int)
    selected = selected[selected["selected"] == 1].copy()
    for col in [
        "selected_angle_deg",
        "angle_delta_deg",
        "window_violation_deg",
        "residual_px",
        "corrected_residual_px",
        "selection_cost",
    ]:
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    return selected


def create_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 0.92))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]],
                              gridspec_kw={"wspace": 0.28, "hspace": 0.34})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.07)
    return fig, axes


def draw_overlay_panel(ax: plt.Axes, data: FigureData) -> None:
    panel_label(ax, "(a)")
    ax.set_axis_off()
    left_ax = ax.inset_axes([0.00, 0.04, 0.49, 0.88])
    right_ax = ax.inset_axes([0.51, 0.04, 0.49, 0.88])
    panels = [
        (left_ax, data.registration_overlay_path, "Registered map"),
        (right_ax, data.measurement_overlay_path, "Measurement map"),
    ]
    for sub_ax, image_path, title in panels:
        if image_path is None:
            sub_ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center",
                        transform=sub_ax.transAxes, color=COLORS["reference"])
            sub_ax.set_axis_off()
            continue
        image = load_overlay_panel_image(image_path)
        sub_ax.imshow(image, interpolation="nearest")
        sub_ax.set_title(title, fontsize=mpl.rcParams["font.size"] - 0.2, pad=3.0)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for spine in sub_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(COLORS["overlay_border"])
    ax.text(
        0.5,
        0.99,
        "Registration result and final GB/T 5.7 map",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=COLORS["reference"],
    )


def draw_residual_panel(ax: plt.Axes, data: FigureData) -> None:
    panel_label(ax, "(b)")
    selected = selected_points_df(data)
    angle = pd.to_numeric(selected["selected_angle_deg"], errors="coerce").to_numpy(dtype=float)
    residual_px = pd.to_numeric(selected.get("corrected_residual_px", selected["residual_px"]),
                                errors="coerce").to_numpy(dtype=float)
    pixel_size_um = metric(data.summary, "pixel_size", "um/px")
    residual_um = residual_px * pixel_size_um
    order = np.argsort(angle)
    angle = angle[order]
    residual_um = residual_um[order]
    rmse_um = metric(data.summary, "single_point_rmse", "um")
    if math.isfinite(rmse_um) and rmse_um > 0.0:
        ax.axhspan(-rmse_um, rmse_um, color=COLORS["residual_fill"], alpha=0.55, zorder=0)
    ax.plot(angle, residual_um, color=COLORS["residual"], marker="o", markersize=3.8, linewidth=1.0)
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Assigned angle (deg)")
    ax.set_ylabel("Corrected radial residual (μm)")
    ax.grid(True, axis="y")


def draw_spacing_panel(ax: plt.Axes, data: FigureData) -> None:
    panel_label(ax, "(c)")
    selected = selected_points_df(data)
    selected = selected.sort_values("selected_angle_deg")
    angle = pd.to_numeric(selected["selected_angle_deg"], errors="coerce").to_numpy(dtype=float)
    if len(angle) < 3:
        ax.text(0.5, 0.5, "Insufficient angular data", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["reference"])
        return
    spacing = np.diff(np.r_[angle, angle[0] + 360.0])
    target = 360.0 / len(angle)
    spacing_error = spacing - target
    target_delta = np.abs(pd.to_numeric(selected["angle_delta_deg"], errors="coerce").to_numpy(dtype=float))
    violation = np.maximum(
        pd.to_numeric(selected.get("window_violation_deg", 0.0), errors="coerce").to_numpy(dtype=float),
        0.0,
    )
    idx = np.arange(1, len(spacing_error) + 1, dtype=float)
    ax.bar(idx, spacing_error, color=COLORS["spacing_fill"], edgecolor=COLORS["spacing"],
           linewidth=0.8, alpha=0.95, width=0.72)
    ax.plot(idx, target_delta, color=COLORS["target_delta"], marker="o",
            markersize=3.2, markerfacecolor="white", linewidth=0.9)
    if np.any(violation > 1e-9):
        ax.scatter(idx[violation > 1e-9], target_delta[violation > 1e-9],
                   color="#CC3311", marker="x", s=28, linewidths=1.0, zorder=3)
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    legend_handles = [
        Line2D([0], [0], color=COLORS["spacing"], linewidth=6, alpha=0.65),
        Line2D([0], [0], color=COLORS["target_delta"], marker="o", markersize=4,
               markerfacecolor="white", linewidth=0.9),
        Line2D([0], [0], color="#CC3311", marker="x", linewidth=0, markersize=5),
    ]
    ax.legend(legend_handles, ["Spacing error", "Target-angle deviation", "Window violation"],
              loc="upper right", frameon=False, handlelength=1.8)
    ax.set_xlabel("Field / interval index")
    ax.set_ylabel("Spacing error (deg)")
    ax.grid(True, axis="y")


def draw_summary_panel(ax: plt.Axes, data: FigureData) -> None:
    panel_label(ax, "(d)")
    ax.axis("off")
    e_p2d_um = metric(data.summary, "e_p2d", "um")
    rmse_um = metric(data.summary, "single_point_rmse", "um")
    angle_err = metric(data.summary, "angle_spacing_max_error", "deg")
    spacing_rmse = metric(data.summary, "angle_spacing_rmse", "deg")
    target_delta_mean = metric(data.summary, "target_angle_delta_mean", "deg")
    target_delta_max = metric(data.summary, "target_angle_delta_max", "deg")
    window_violation_count = metric(data.summary, "window_violation_count", "count")
    window_violation_max = metric(data.summary, "window_violation_max", "deg")
    radius_px = metric(data.summary, "selected_circle_radius", "px")
    selected_count = metric(data.summary, "selected_point_count", "count")
    pixel_size = metric(data.summary, "pixel_size", "um/px")
    selected = selected_points_df(data)
    selection_cost = pd.to_numeric(selected.get("selection_cost", pd.Series(dtype=float)), errors="coerce")
    selection_cost = selection_cost[np.isfinite(selection_cost.to_numpy(dtype=float))] if len(selection_cost) > 0 else selection_cost
    lines = [
        f"Selected points: {selected_count:.0f}",
        f"Pixel size: {pixel_size:.3f} μm/px",
        f"Selected-circle radius: {radius_px:.3f} px",
        "",
        f"E_P2D: {e_p2d_um:.3f} μm",
        f"Single-point RMSE: {rmse_um:.3f} μm",
        f"Spacing RMSE: {spacing_rmse:.3f} deg",
        f"Max angle-spacing error: {angle_err:.3f} deg",
        f"Mean target-angle deviation: {target_delta_mean:.3f} deg",
        f"Max target-angle deviation: {target_delta_max:.3f} deg",
        f"Window violations: {window_violation_count:.0f}",
        f"Max window violation: {window_violation_max:.3f} deg",
    ]
    if len(selection_cost) > 0:
        lines.append(f"Mean selection cost: {float(selection_cost.mean()):.4f}")
    if data.circle_center_df is not None and not data.circle_center_df.empty and "local_normal_rmse_px" in data.circle_center_df:
        local_rmse = pd.to_numeric(data.circle_center_df["local_normal_rmse_px"], errors="coerce")
        local_rmse = local_rmse[np.isfinite(local_rmse)]
        if len(local_rmse) > 0 and math.isfinite(pixel_size):
            lines.extend([
                "",
                f"Circle-center local RMSE median: {float(np.median(local_rmse) * pixel_size):.3f} μm",
                f"Circle-center local RMSE max: {float(np.max(local_rmse) * pixel_size):.3f} μm",
            ])
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
            transform=ax.transAxes, family="monospace",
            fontsize=mpl.rcParams["font.size"] - 0.1,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["summary_face"],
                      edgecolor=COLORS["summary_edge"], linewidth=0.8))


def draw_summary_panel(ax: plt.Axes, data: FigureData) -> None:
    panel_label(ax, "(d)")
    ax.axis("off")
    e_p2d_um = metric(data.summary, "e_p2d", "um")
    rmse_um = metric(data.summary, "single_point_rmse", "um")
    angle_err = metric(data.summary, "angle_spacing_max_error", "deg")
    spacing_rmse = metric(data.summary, "angle_spacing_rmse", "deg")
    target_delta_mean = metric(data.summary, "target_angle_delta_mean", "deg")
    target_delta_max = metric(data.summary, "target_angle_delta_max", "deg")
    window_violation_count = metric(data.summary, "window_violation_count", "count")
    window_violation_max = metric(data.summary, "window_violation_max", "deg")
    radius_px = metric(data.summary, "selected_circle_radius", "px")
    selected_count = metric(data.summary, "selected_point_count", "count")
    pixel_size = metric(data.summary, "pixel_size", "um/px")
    selected = selected_points_df(data)
    selection_cost = pd.to_numeric(selected.get("selection_cost", pd.Series(dtype=float)), errors="coerce")
    selection_cost = selection_cost[np.isfinite(selection_cost.to_numpy(dtype=float))] if len(selection_cost) > 0 else selection_cost

    acquisition_lines = [
        f"Selected points: {selected_count:.0f}",
        f"Pixel size: {pixel_size:.3f} μm/px",
        f"Selected-circle radius: {radius_px:.1f} px",
    ]
    metric_lines = [
        f"E_P2D: {e_p2d_um:.3f} μm",
        f"Single-point RMSE: {rmse_um:.3f} μm",
        f"Spacing RMSE: {spacing_rmse:.3f} deg",
        f"Max spacing error: {angle_err:.3f} deg",
        f"Mean target Δθ: {target_delta_mean:.3f} deg",
        f"Max target Δθ: {target_delta_max:.3f} deg",
        f"Window violations: {window_violation_count:.0f}",
        f"Max window excess: {window_violation_max:.3f} deg",
    ]
    quality_lines: list[str] = []
    if len(selection_cost) > 0:
        quality_lines.append(f"Mean selection cost: {float(selection_cost.mean()):.4f}")
    if data.circle_center_df is not None and not data.circle_center_df.empty and "local_normal_rmse_px" in data.circle_center_df:
        local_rmse = pd.to_numeric(data.circle_center_df["local_normal_rmse_px"], errors="coerce")
        local_rmse = local_rmse[np.isfinite(local_rmse)]
        if len(local_rmse) > 0 and math.isfinite(pixel_size):
            quality_lines.extend([
                f"Center local RMSE median: {float(np.median(local_rmse) * pixel_size):.3f} μm",
                f"Center local RMSE max: {float(np.max(local_rmse) * pixel_size):.3f} μm",
            ])
    if not quality_lines:
        quality_lines.append("Circle-center local RMSE unavailable")

    def add_block(x: float, y: float, title: str, lines: list[str]) -> None:
        ax.text(
            x,
            y,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=COLORS["target_delta"],
            fontweight="bold",
        )
        ax.text(
            x,
            y - 0.08,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=COLORS["reference"],
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor=COLORS["summary_face"],
                edgecolor=COLORS["summary_edge"],
                linewidth=0.8,
            ),
        )

    add_block(0.00, 0.98, "Acquisition", acquisition_lines)
    add_block(0.52, 0.98, "GB/T 5.7 metrics", metric_lines)
    add_block(0.00, 0.45, "Selection quality", quality_lines)

    note = (
        "Interpretation:\n"
        "Target-angle uniformity and\n"
        "single-point accuracy should\n"
        "be reported together."
    )
    ax.text(
        0.52,
        0.45,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF9F5", edgecolor="#E2C6B8", linewidth=0.8),
    )


def export_figure(fig: plt.Figure, args: argparse.Namespace, result_dir: Path) -> None:
    svg_path = args.output_svg or (result_dir / DEFAULT_SVG_NAME)
    pdf_path = args.output_pdf or (result_dir / DEFAULT_PDF_NAME)
    png_path = args.output_png or (result_dir / DEFAULT_PNG_NAME)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    if not args.svg_only:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if not args.svg_only and not args.pdf_only:
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)


def main() -> None:
    args = parse_args()
    result_dir = resolve_result_dir(args).resolve()
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)
    data = load_data(result_dir)
    fig, axes = create_figure(args.figure_width_mm)
    draw_overlay_panel(axes["A"], data)
    draw_residual_panel(axes["B"], data)
    draw_spacing_panel(axes["C"], data)
    draw_summary_panel(axes["D"], data)
    export_figure(fig, args, result_dir)
    plt.close(fig)
    print(f"[OK] standard sphere figure exported under: {result_dir}")


if __name__ == "__main__":
    main()
