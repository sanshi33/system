#!/usr/bin/env python3
"""
Generate a CJSI-oriented analysis package for GB/T 5.7 standard-sphere results.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MAIN_SVG_NAME = "cjsi_standard_sphere_main_figure.svg"
MAIN_PDF_NAME = "cjsi_standard_sphere_main_figure.pdf"
MAIN_PNG_NAME = "cjsi_standard_sphere_main_figure.png"
ABLATION_SVG_NAME = "cjsi_standard_sphere_ablation_figure.svg"
ABLATION_PDF_NAME = "cjsi_standard_sphere_ablation_figure.pdf"
ABLATION_PNG_NAME = "cjsi_standard_sphere_ablation_figure.png"
COMPARISON_CSV_NAME = "cjsi_standard_sphere_method_comparison.csv"
DIAGNOSTICS_CSV_NAME = "cjsi_standard_sphere_field_diagnostics.csv"
REPORT_MD_NAME = "cjsi_standard_sphere_analysis_report.md"
FIGURE_LANGUAGES = ("zh", "en")

COLORS = {
    "accuracy_a": "#0B5FA5",
    "accuracy_b": "#E66101",
    "uniform_a": "#1B9E77",
    "uniform_b": "#7570B3",
    "window": "#CC3311",
    "target_delta": "#2F4858",
    "reference": "#4D4D4D",
    "grid": "#D7DBE0",
    "overlay_border": "#E8EBEF",
    "summary_face": "#FAFBFC",
    "summary_edge": "#D9DEE5",
    "ok": "#1B9E77",
    "warn": "#CC3311",
}

METHOD_LABELS = {
    "globalopt_v1": "Phase+GlobalOpt v1",
    "globalopt_v2": "Phase+GlobalOpt v2",
    "windowfit_v1": "WindowFit v1",
    "windowguard_v1": "WindowGuard v1",
    "codex_run_20260521_range256_restarts64_selectedrefine_v1-最好结果": "RangeOpt best",
    "codex_run_20260521_fixed_radius_uniform_angle_confbest_v1": "FixedRadius conf-best",
    "codex_run_20260528_uniform25_windowguard_v1": "WindowGuard v1",
}
METHOD_ORDER = ["globalopt_v1", "globalopt_v2", "windowfit_v1", "windowguard_v1"]


@dataclass
class ResultBundle:
    result_dir: Path
    label: str
    summary: Dict[str, float | str]
    points_df: pd.DataFrame
    circle_center_df: Optional[pd.DataFrame]
    registration_overlay_path: Optional[Path]
    measurement_overlay_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CJSI-ready analysis package for standard-sphere GB/T 5.7 outputs."
    )
    parser.add_argument("--result-dir", type=Path, help="Primary result directory.")
    parser.add_argument("--compare-dirs", nargs="*", type=Path, help="Optional comparison result directories.")
    parser.add_argument("--result-root", type=Path, default=Path("result/standard_sphere_loop"))
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to the primary result directory.")
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
        "font.family": ["SimSun", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.serif": ["SimSun", "Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
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
        "grid.alpha": 0.18,
        "grid.color": "#D7DBE0",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "axes.unicode_minus": False,
        "svg.fonttype": svg_fonttype,
    })


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.0, fontweight="bold",
            ha="left", va="bottom")


def tr(lang: str, zh_text: str, en_text: str) -> str:
    return zh_text if lang == "zh" else en_text


def localized_output_path(output_dir: Path, filename: str, lang: str) -> Path:
    if lang == "zh":
        return output_dir / filename
    path = Path(filename)
    return output_dir / f"{path.stem}_en{path.suffix}"


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
        metric_name = str(row.get("metric", "")).strip()
        unit = str(row.get("unit", "")).strip()
        key = metric_name if not unit else f"{metric_name}__{unit}"
        value = row.get("value", np.nan)
        if isinstance(value, str):
            summary[key] = value
        else:
            summary[key] = float(value) if math.isfinite(float(value)) else value
        if metric_name not in summary:
            summary[metric_name] = summary[key]
    return summary


def metric(summary: Dict[str, float | str], name: str, unit: str = "") -> float:
    key = name if not unit else f"{name}__{unit}"
    value = summary.get(key, summary.get(name, float("nan")))
    try:
        return float(value)
    except Exception:
        return float("nan")


def method_label_from_dir(result_dir: Path) -> str:
    lower = result_dir.name.lower()
    for key in METHOD_ORDER:
        if key in lower:
            return METHOD_LABELS[key]
    token = re.sub(r"^codex_run_\d+_", "", result_dir.name)
    return token or result_dir.name


def load_result_bundle(result_dir: Path) -> ResultBundle:
    summary_path = result_dir / "standard_sphere_gbt57_p2d_summary.csv"
    points_path = result_dir / "standard_sphere_gbt57_p2d_points.csv"
    if not summary_path.exists() or not points_path.exists():
        raise FileNotFoundError(f"missing standard sphere outputs under: {result_dir}")
    summary = load_summary_csv(summary_path)
    points_df = pd.read_csv(points_path)
    circle_center_path = result_dir / "standard_sphere_circle_center_global.csv"
    circle_center_df = pd.read_csv(circle_center_path) if circle_center_path.exists() else None
    registration_overlay_path = None
    for candidate in (
        "standard_sphere_gbt57_p2d_edge_overlay.png",
        "standard_sphere_stitched_overlay.png",
        "standard_sphere_gbt57_p2d_overlay.png",
    ):
        path = result_dir / candidate
        if path.exists():
            registration_overlay_path = path
            break
    measurement_overlay_path = None
    for candidate in (
        "standard_sphere_gbt57_p2d_overlay.png",
        "standard_sphere_gbt57_p2d_edge_overlay.png",
        "standard_sphere_stitched_overlay.png",
    ):
        path = result_dir / candidate
        if path.exists():
            measurement_overlay_path = path
            break
    return ResultBundle(
        result_dir=result_dir,
        label=method_label_from_dir(result_dir),
        summary=summary,
        points_df=points_df,
        circle_center_df=circle_center_df,
        registration_overlay_path=registration_overlay_path,
        measurement_overlay_path=measurement_overlay_path,
    )


def auto_compare_dirs(primary_dir: Path, result_root: Path) -> List[Path]:
    siblings = [path for path in result_root.iterdir() if path.is_dir()]
    if "uniform25" in primary_dir.name:
        same_date = re.search(r"(\d{8})", primary_dir.name)
        token = same_date.group(1) if same_date else ""
        picked: List[Path] = []
        for key in METHOD_ORDER:
            for path in siblings:
                lower = path.name.lower()
                if key in lower and (not token or token in lower):
                    picked.append(path)
                    break
        if primary_dir not in picked:
            picked.append(primary_dir)
        return picked
    return [primary_dir]


def selected_points_df(bundle: ResultBundle) -> pd.DataFrame:
    selected = bundle.points_df.copy()
    selected["selected"] = pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int)
    selected = selected[selected["selected"] == 1].copy()
    for column in (
        "selected_angle_deg",
        "view_angle_deg",
        "angle_delta_deg",
        "window_half_angle_deg",
        "window_violation_deg",
        "residual_px",
        "corrected_residual_px",
        "selection_cost",
        "confidence",
        "gradient",
    ):
        if column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def compute_field_diagnostics(bundle: ResultBundle) -> pd.DataFrame:
    selected = selected_points_df(bundle).sort_values("selected_angle_deg").reset_index(drop=True)
    count = len(selected)
    pixel_size_um = metric(bundle.summary, "pixel_size", "um/px")
    corrected_residual_px = pd.to_numeric(
        selected.get("corrected_residual_px", selected.get("residual_px", pd.Series(dtype=float))),
        errors="coerce",
    ).to_numpy(dtype=float)
    if corrected_residual_px.size != count:
        corrected_residual_px = np.full(count, np.nan, dtype=float)

    def _numeric_column(name: str) -> np.ndarray:
        if name not in selected.columns:
            return np.full(count, np.nan, dtype=float)
        values = pd.to_numeric(selected[name], errors="coerce")
        if len(values) != count:
            values = values.reindex(selected.index)
        return values.to_numpy(dtype=float)

    def _string_column(name: str, default_prefix: str = "field") -> np.ndarray:
        if name not in selected.columns:
            return np.array([f"{default_prefix}_{idx + 1}" for idx in range(count)], dtype=object)
        values = selected[name].astype(str)
        if len(values) != count:
            values = values.reindex(selected.index).fillna("")
        output = values.to_numpy(dtype=object)
        for idx, value in enumerate(output):
            if not value or value == "nan":
                output[idx] = f"{default_prefix}_{idx + 1}"
        return output

    diagnostics = pd.DataFrame({
        "angle_order": np.arange(1, count + 1, dtype=int),
        "image_name": _string_column("image_name", default_prefix="Pic"),
        "assigned_angle_deg": _numeric_column("view_angle_deg"),
        "selected_angle_deg": _numeric_column("selected_angle_deg"),
        "target_angle_delta_deg": _numeric_column("angle_delta_deg"),
        "window_half_angle_deg": _numeric_column("window_half_angle_deg"),
        "window_violation_deg": _numeric_column("window_violation_deg"),
        "corrected_residual_um": corrected_residual_px * pixel_size_um,
        "selection_cost": _numeric_column("selection_cost"),
        "confidence": _numeric_column("confidence"),
        "gradient": _numeric_column("gradient"),
    })
    diagnostics["abs_residual_um"] = np.abs(diagnostics["corrected_residual_um"])
    diagnostics["is_window_violation"] = diagnostics["window_violation_deg"] > 1e-9
    diagnostics["residual_rank"] = (
        diagnostics["abs_residual_um"].rank(method="first", ascending=False).astype(int)
        if not diagnostics.empty else pd.Series(dtype=int)
    )
    diagnostics["target_rank"] = (
        diagnostics["target_angle_delta_deg"].rank(method="first", ascending=False).astype(int)
        if not diagnostics.empty else pd.Series(dtype=int)
    )
    return diagnostics


def compute_comparison_df(bundles: Sequence[ResultBundle]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for bundle in bundles:
        rows.append({
            "label": bundle.label,
            "result_dir": bundle.result_dir.name,
            "e_p2d_um": metric(bundle.summary, "e_p2d", "um"),
            "single_point_rmse_um": metric(bundle.summary, "single_point_rmse", "um"),
            "angle_spacing_rmse_deg": metric(bundle.summary, "angle_spacing_rmse", "deg"),
            "angle_spacing_max_error_deg": metric(bundle.summary, "angle_spacing_max_error", "deg"),
            "target_angle_delta_mean_deg": metric(bundle.summary, "target_angle_delta_mean", "deg"),
            "target_angle_delta_max_deg": metric(bundle.summary, "target_angle_delta_max", "deg"),
            "window_violation_count": metric(bundle.summary, "window_violation_count", "count"),
            "window_violation_max_deg": metric(bundle.summary, "window_violation_max", "deg"),
        })
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    order_map = {METHOD_LABELS[key]: idx for idx, key in enumerate(METHOD_ORDER)}
    comparison["method_order"] = comparison["label"].map(order_map).fillna(999)
    comparison = comparison.sort_values(["method_order", "label"]).drop(columns=["method_order"]).reset_index(drop=True)
    return comparison


def create_main_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 0.92))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.34, "hspace": 0.34})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.98, bottom=0.07)
    return fig, axes


def create_ablation_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 0.84))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.34, "hspace": 0.34})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.98, bottom=0.09)
    return fig, axes


def draw_overlay_panel(ax: plt.Axes, bundle: ResultBundle, lang: str = "zh") -> None:
    panel_label(ax, "(a)")
    ax.set_axis_off()

    left_ax = ax.inset_axes([0.01, 0.06, 0.48, 0.84])
    right_ax = ax.inset_axes([0.51, 0.06, 0.48, 0.84])

    panels = [
        (left_ax, bundle.registration_overlay_path, tr(lang, "配准结果", "Registered")),
        (right_ax, bundle.measurement_overlay_path, tr(lang, "离散采样点", "Selected points")),
    ]
    for sub_ax, image_path, title in panels:
        if image_path is None:
            sub_ax.text(0.5, 0.5, tr(lang, "图像缺失", "Image unavailable"), ha="center", va="center",
                        transform=sub_ax.transAxes, color=COLORS["reference"])
            sub_ax.set_axis_off()
            continue
        image = load_overlay_panel_image(image_path)
        sub_ax.imshow(image, interpolation="nearest")
        sub_ax.text(
            0.03, 0.97, title, transform=sub_ax.transAxes,
            ha="left", va="top", fontsize=mpl.rcParams["font.size"] - 0.4,
            color=COLORS["reference"],
            bbox=dict(boxstyle="round,pad=0.16", facecolor=(1, 1, 1, 0.78), edgecolor="none"),
        )
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for spine in sub_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor(COLORS["overlay_border"])


def draw_residual_panel(ax: plt.Axes, bundle: ResultBundle, diagnostics: pd.DataFrame, lang: str = "zh") -> None:
    panel_label(ax, "(b)")
    if diagnostics.empty:
        ax.text(0.5, 0.5, tr(lang, "无有效采样点", "No selected points"), ha="center", va="center", transform=ax.transAxes)
        return
    angle = diagnostics["selected_angle_deg"].to_numpy(dtype=float)
    residual_um = diagnostics["corrected_residual_um"].to_numpy(dtype=float)
    violations = diagnostics["is_window_violation"].to_numpy(dtype=bool)
    order = np.argsort(angle)
    angle = angle[order]
    residual_um = residual_um[order]
    violations = violations[order]
    rmse_um = metric(bundle.summary, "single_point_rmse", "um")
    if math.isfinite(rmse_um) and rmse_um > 0.0:
        ax.axhspan(-rmse_um, rmse_um, color="#DEE8F2", alpha=0.60, zorder=0)
    ax.plot(angle, residual_um, color=COLORS["accuracy_a"], marker="o", markersize=3.2, linewidth=1.0)
    if np.any(violations):
        ax.scatter(angle[violations], residual_um[violations], s=28, marker="o",
                   facecolors="white", edgecolors=COLORS["window"], linewidths=1.0, zorder=3)
    ax.axhline(0.0, color=COLORS["reference"], linestyle=":", linewidth=0.8)
    legend_handles = [
        Line2D([0], [0], color=COLORS["accuracy_a"], marker="o", markersize=4, linewidth=1.0),
        Patch(facecolor="#DEE8F2", edgecolor="none", alpha=0.60),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["window"], markersize=5, linewidth=0),
    ]
    ax.legend(legend_handles, [tr(lang, "残差", "Residual"), tr(lang, "±RMSE带", "±RMSE band"), tr(lang, "警戒点", "Guard point")],
              loc="upper right", frameon=False, handlelength=1.6)
    ax.set_xlabel(tr(lang, "角度 (deg)", "Angle (deg)"))
    ax.set_ylabel(tr(lang, "残差 (μm)", "Residual (μm)"))
    ax.grid(True, axis="y")


def draw_spacing_panel(ax: plt.Axes, bundle: ResultBundle, diagnostics: pd.DataFrame, lang: str = "zh") -> None:
    panel_label(ax, "(c)")
    if diagnostics.empty or len(diagnostics) < 3:
        ax.text(0.5, 0.5, tr(lang, "角度数据不足", "Insufficient angular data"), ha="center", va="center", transform=ax.transAxes)
        return
    diagnostics = diagnostics.sort_values("selected_angle_deg").reset_index(drop=True)
    angle = diagnostics["selected_angle_deg"].to_numpy(dtype=float)
    spacing = np.diff(np.r_[angle, angle[0] + 360.0])
    target_spacing = 360.0 / len(angle)
    spacing_error = spacing - target_spacing
    target_delta = diagnostics["target_angle_delta_deg"].to_numpy(dtype=float)
    violation = diagnostics["window_violation_deg"].to_numpy(dtype=float)
    idx = np.arange(1, len(spacing_error) + 1, dtype=float)

    ax.bar(idx, spacing_error, width=0.72, color="#F3D6C3",
           edgecolor="none", linewidth=0.0)
    ax.plot(idx, target_delta, color=COLORS["target_delta"], marker="o",
            markersize=3.0, markerfacecolor="white", linewidth=0.9)
    if np.any(violation > 1e-9):
        ax.scatter(idx[violation > 1e-9], target_delta[violation > 1e-9],
                   color=COLORS["window"], marker="x", s=28, linewidths=1.0, zorder=3)
    ax.axhline(0.0, color=COLORS["reference"], linestyle=":", linewidth=0.8)
    ax.set_xlabel(tr(lang, "区间序号", "Interval index"))
    ax.set_ylabel(tr(lang, "角间隔误差 (deg)", "Spacing error (deg)"))
    ax.grid(True, axis="y")

    legend_handles = [
        Line2D([0], [0], color=COLORS["accuracy_b"], linewidth=6, alpha=0.6),
        Line2D([0], [0], color=COLORS["target_delta"], marker="o", markersize=4,
               markerfacecolor="white", linewidth=0.9),
        Line2D([0], [0], color=COLORS["window"], marker="x", linewidth=0, markersize=5),
    ]
    ax.legend(legend_handles, [tr(lang, "角间隔误差", "Spacing error"), tr(lang, "目标角偏差", "Target-angle deviation"), tr(lang, "窗口越界", "Window violation")],
              loc="upper right", frameon=False)


def draw_field_panel(ax: plt.Axes, bundle: ResultBundle, diagnostics: pd.DataFrame, lang: str = "zh") -> None:
    panel_label(ax, "(d)")
    if diagnostics.empty:
        ax.text(0.5, 0.5, tr(lang, "无字段诊断结果", "No field diagnostics available"), ha="center", va="center", transform=ax.transAxes)
        return
    top = diagnostics.nlargest(6, "abs_residual_um").sort_values("abs_residual_um", ascending=True)
    colors = [COLORS["warn"] if flag else COLORS["ok"] for flag in top["is_window_violation"]]
    ax.barh(top["image_name"].str.replace(".bmp", "", regex=False), top["abs_residual_um"],
            color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel(tr(lang, "|残差| (μm)", "|Residual| (μm)"))
    ax.set_ylabel(tr(lang, "图像", "Image"))
    ax.grid(True, axis="x")
    legend_handles = [
        Patch(facecolor=COLORS["ok"], edgecolor="white"),
        Patch(facecolor=COLORS["warn"], edgecolor="white"),
    ]
    ax.legend(legend_handles, [tr(lang, "窗口内", "Within window"), tr(lang, "超出窗口", "Window exceed")], loc="lower right", frameon=False)


def draw_comparison_bar_panel(
    ax: plt.Axes,
    comparison: pd.DataFrame,
    panel: str,
    title: str,
    left_col: str,
    right_col: str,
    left_label: str,
    right_label: str,
    colors: Tuple[str, str],
) -> None:
    panel_label(ax, panel)
    if comparison.empty:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center", transform=ax.transAxes)
        return
    x = np.arange(len(comparison), dtype=float)
    width = 0.36
    ax.bar(x - width / 2.0, comparison[left_col], width=width, color=colors[0], label=left_label)
    ax.bar(x + width / 2.0, comparison[right_col], width=width, color=colors[1], label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=18, ha="right")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")


def draw_tradeoff_panel(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    panel_label(ax, "(d)")
    if comparison.empty:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center", transform=ax.transAxes)
        return
    x = comparison["e_p2d_um"].to_numpy(dtype=float)
    y = comparison["target_angle_delta_max_deg"].to_numpy(dtype=float)
    palette = [COLORS["accuracy_a"], COLORS["accuracy_b"], COLORS["uniform_a"], COLORS["uniform_b"], COLORS["window"]]
    for idx, row in comparison.iterrows():
        color = palette[idx % len(palette)]
        ax.scatter(row["e_p2d_um"], row["target_angle_delta_max_deg"], s=60, color=color,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(row["label"], (row["e_p2d_um"], row["target_angle_delta_max_deg"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=mpl.rcParams["font.size"] - 0.8, color=COLORS["reference"])
    ax.set_xlabel("E_P2D (μm)")
    ax.set_ylabel("Max target-angle deviation (deg)")
    ax.grid(True)


def export_figure(fig: plt.Figure, svg_path: Path, pdf_path: Path, png_path: Path,
                  svg_only: bool, pdf_only: bool) -> None:
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    if not svg_only:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if not svg_only and not pdf_only:
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=600)


def build_main_figure(bundle: ResultBundle, diagnostics: pd.DataFrame, output_dir: Path,
                      args: argparse.Namespace, lang: str = "zh") -> None:
    fig, axes = create_main_figure(args.figure_width_mm)
    draw_overlay_panel(axes["A"], bundle, lang=lang)
    draw_residual_panel(axes["B"], bundle, diagnostics, lang=lang)
    draw_spacing_panel(axes["C"], bundle, diagnostics, lang=lang)
    draw_field_panel(axes["D"], bundle, diagnostics, lang=lang)
    export_figure(
        fig,
        localized_output_path(output_dir, MAIN_SVG_NAME, lang),
        localized_output_path(output_dir, MAIN_PDF_NAME, lang),
        localized_output_path(output_dir, MAIN_PNG_NAME, lang),
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def build_ablation_figure(comparison: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    fig, axes = create_ablation_figure(args.figure_width_mm)
    draw_comparison_bar_panel(
        axes["A"], comparison, "(a)", "Accuracy metrics",
        "e_p2d_um", "single_point_rmse_um",
        "E_P2D (μm)", "RMSE (μm)",
        (COLORS["accuracy_a"], COLORS["accuracy_b"]),
    )
    draw_comparison_bar_panel(
        axes["B"], comparison, "(b)", "Spacing metrics",
        "angle_spacing_rmse_deg", "angle_spacing_max_error_deg",
        "Spacing RMSE (deg)", "Max spacing error (deg)",
        (COLORS["uniform_a"], COLORS["uniform_b"]),
    )
    draw_comparison_bar_panel(
        axes["C"], comparison, "(c)", "Target-angle metrics",
        "target_angle_delta_mean_deg", "target_angle_delta_max_deg",
        "Mean target Δθ (deg)", "Max target Δθ (deg)",
        (COLORS["uniform_b"], COLORS["window"]),
    )
    draw_tradeoff_panel(axes["D"], comparison)
    export_figure(
        fig,
        output_dir / ABLATION_SVG_NAME,
        output_dir / ABLATION_PDF_NAME,
        output_dir / ABLATION_PNG_NAME,
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def best_method_text(comparison: pd.DataFrame, column: str) -> str:
    finite = comparison[np.isfinite(pd.to_numeric(comparison[column], errors="coerce"))].copy()
    if finite.empty:
        return "--"
    row = finite.loc[pd.to_numeric(finite[column], errors="coerce").idxmin()]
    return f"{row['label']} ({float(row[column]):.3f})"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def write_report(output_dir: Path, bundle: ResultBundle, comparison: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    top_violation = diagnostics.sort_values(["window_violation_deg", "abs_residual_um"], ascending=[False, False]).head(5)
    violation_count_metric = metric(bundle.summary, "window_violation_count", "count")
    violation_max_metric = metric(bundle.summary, "window_violation_max", "deg")
    if not math.isfinite(violation_count_metric):
        violation_count_metric = float(int(diagnostics["is_window_violation"].sum())) if "is_window_violation" in diagnostics else 0.0
    if not math.isfinite(violation_max_metric):
        if not diagnostics.empty and "window_violation_deg" in diagnostics:
            violation_values = diagnostics["window_violation_deg"].to_numpy(dtype=float)
            finite_mask = np.isfinite(violation_values)
            violation_max_metric = float(np.nanmax(violation_values[finite_mask])) if finite_mask.any() else 0.0
        else:
            violation_max_metric = 0.0
    comparison_rows = []
    for _, row in comparison.iterrows():
        comparison_rows.append([
            str(row["label"]),
            f"{row['e_p2d_um']:.3f}",
            f"{row['single_point_rmse_um']:.3f}",
            f"{row['angle_spacing_rmse_deg']:.3f}",
            f"{row['target_angle_delta_max_deg']:.3f}",
        ])
    violation_rows = []
    for _, row in top_violation.iterrows():
        violation_rows.append([
            str(int(row["angle_order"])),
            str(row["image_name"]),
            f"{row['abs_residual_um']:.3f}",
            f"{row['target_angle_delta_deg']:.3f}",
            f"{row['window_violation_deg']:.3f}",
        ])
    lines = [
        "# 标准球结果投稿分析摘要",
        "",
        "## 1. 当前结果概述",
        "",
        f"- 主结果目录：`{bundle.result_dir.name}`",
        f"- E_P2D：`{metric(bundle.summary, 'e_p2d', 'um'):.3f} μm`",
        f"- 单点 RMSE：`{metric(bundle.summary, 'single_point_rmse', 'um'):.3f} μm`",
        f"- 角间距 RMSE：`{metric(bundle.summary, 'angle_spacing_rmse', 'deg'):.3f} deg`",
        f"- 最大目标角偏差：`{metric(bundle.summary, 'target_angle_delta_max', 'deg'):.3f} deg`",
        f"- 窗口越界数量：`{int(round(violation_count_metric))} / 25`",
        f"- 最大窗口越界：`{violation_max_metric:.3f} deg`",
        "",
        "## 2. 同批次方法演进比较",
        "",
        markdown_table(
            ["方法", "E_P2D/μm", "RMSE/μm", "Spacing RMSE/deg", "Max target Δθ/deg"],
            comparison_rows,
        ),
        "",
        f"- 按 `E_P2D` 最优的方法：{best_method_text(comparison, 'e_p2d_um')}",
        f"- 按 `RMSE` 最优的方法：{best_method_text(comparison, 'single_point_rmse_um')}",
        f"- 按 `Max target Δθ` 最优的方法：{best_method_text(comparison, 'target_angle_delta_max_deg')}",
        "",
        "## 3. 字段级诊断",
        "",
        markdown_table(
            ["角顺序", "视场", "绝对残差/μm", "目标角偏差/deg", "窗口越界/deg"],
            violation_rows,
        ),
        "",
        "## 4. 面向《仪器仪表学报》的结果组织建议",
        "",
        "- 主图使用 `cjsi_standard_sphere_main_figure.*`，对应最终方法的全局覆盖、残差分布、角均匀性和字段诊断。",
        "- 对比图使用 `cjsi_standard_sphere_ablation_figure.*`，用于交代从全局优化到窗口约束的演进关系与精度-均匀性权衡。",
        "- 表格优先引用 `cjsi_standard_sphere_method_comparison.csv` 和 `cjsi_standard_sphere_field_diagnostics.csv`，避免在正文中堆砌重复数字。",
        "- 结果段表述建议突出“矩形顺时针采集条件下，最终测量窗口仍保持近似均匀且基本不重叠”，并诚实说明窗口约束与单点圆误差之间存在小幅权衡。",
        "",
        "## 5. 可直接用于结果段的表述草案",
        "",
        "在矩形顺时针采集而非等角采集条件下，本文采用基于全局相位拟合、候选点联合优化及窗口越界约束的 25 点测量窗口分配策略，实现了 GB/T 5.7 单点评定中测量窗口的近似均匀覆盖。最终结果表明，所选 25 个测量点的 E_P2D 为 "
        f"{metric(bundle.summary, 'e_p2d', 'um'):.3f} μm，单点 RMSE 为 "
        f"{metric(bundle.summary, 'single_point_rmse', 'um'):.3f} μm，角间距 RMSE 为 "
        f"{metric(bundle.summary, 'angle_spacing_rmse', 'deg'):.3f}°。",
        "",
        "进一步地，最终方法仅有 "
        f"{int(round(violation_count_metric))} 个测量点超出其对应非重叠窗口，且最大越界量仅为 "
        f"{violation_max_metric:.3f}°，说明在采集几何先验不理想的情况下，所提出的数据处理流程仍能有效维持测量窗口的均匀性与可解释性。",
        "",
    ]
    (output_dir / REPORT_MD_NAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    primary_dir = resolve_result_dir(args).resolve()
    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or primary_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_dirs = args.compare_dirs if args.compare_dirs else auto_compare_dirs(primary_dir, result_root)
    unique_compare_dirs: List[Path] = []
    for path in compare_dirs:
        resolved = path.resolve()
        if resolved not in unique_compare_dirs:
            unique_compare_dirs.append(resolved)

    bundles = [load_result_bundle(primary_dir)]
    for path in unique_compare_dirs:
        if path == primary_dir:
            continue
        bundles.append(load_result_bundle(path))

    primary_bundle = bundles[0]
    diagnostics = compute_field_diagnostics(primary_bundle)
    comparison = compute_comparison_df(bundles)
    diagnostics.to_csv(output_dir / DIAGNOSTICS_CSV_NAME, index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / COMPARISON_CSV_NAME, index=False, encoding="utf-8-sig")

    for lang in FIGURE_LANGUAGES:
        build_main_figure(primary_bundle, diagnostics, output_dir, args, lang=lang)
    build_ablation_figure(comparison, output_dir, args)
    write_report(output_dir, primary_bundle, comparison, diagnostics)
    print(f"[OK] CJSI analysis package exported under: {output_dir}")


if __name__ == "__main__":
    main()
