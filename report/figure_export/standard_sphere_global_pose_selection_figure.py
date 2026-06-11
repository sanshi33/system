#!/usr/bin/env python3
"""
Export a figure for the standard-sphere workflow:
global pose optimization -> point selection -> residual distribution.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from standard_sphere_optimized_presentation_figure import reconstruct_global_edge_cloud


OUTPUT_PNG_NAME = "standard_sphere_global_pose_selection_figure.png"
OUTPUT_PDF_NAME = "standard_sphere_global_pose_selection_figure.pdf"
OUTPUT_SVG_NAME = "standard_sphere_global_pose_selection_figure.svg"
OUTPUT_CSV_NAME = "standard_sphere_global_pose_selection_metrics.csv"

EDGE_THRESHOLD_GRAY = 128.0
EDGE_RESIDUAL_FILTER_PX = 1.2
MAX_EXPORT_SAMPLES = 1200
RIGHT_PANEL_YLIM_UM = 15.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standard-sphere global pose selection figure.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--image-indices", type=str, default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def configure_matplotlib(svg_fonttype: str) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "font.family": ["Microsoft YaHei", "SimSun", "Times New Roman", "DejaVu Sans"],
        "font.serif": ["SimSun", "Times New Roman", "DejaVu Serif", "serif"],
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": False,
        "ytick.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "svg.fonttype": svg_fonttype,
    })


def natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def resolve_input_dir(args: argparse.Namespace) -> Path:
    if args.input_dir is not None:
        return args.input_dir.resolve()
    root = project_root()
    candidates = sorted(root.rglob("image_80%_25"), key=lambda p: natural_key(str(p)))
    if not candidates:
        raise FileNotFoundError("cannot find input directory image_80%_25")
    return candidates[0]


def load_summary(path: Path) -> Dict[str, float | str]:
    df = pd.read_csv(path)
    summary: Dict[str, float | str] = {}
    for _, row in df.iterrows():
        metric_name = str(row.get("metric", "")).strip()
        unit = str(row.get("unit", "")).strip()
        key = metric_name if not unit else f"{metric_name}__{unit}"
        value = row.get("value", np.nan)
        try:
            summary[key] = float(value)
        except Exception:
            summary[key] = value
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


def load_overlay_image(path: Path) -> np.ndarray:
    image = plt.imread(path)
    image = crop_light_background_image(image, threshold=0.997, pad_ratio=0.02)
    return enhance_light_background_overlay(image, strength=1.8)


def image_name_from_index(index: int) -> str:
    return f"Pic_{index}.bmp"


def parse_indices(text: str, fallback: List[int]) -> List[int]:
    if not text.strip():
        return fallback
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or fallback


def extract_edge_points(image_path: Path, threshold_gray: float = EDGE_THRESHOLD_GRAY) -> np.ndarray:
    image = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    shifted = image - threshold_gray

    left = shifted[:, :-1]
    right = shifted[:, 1:]
    hmask = ((left <= 0.0) & (right > 0.0)) | ((left >= 0.0) & (right < 0.0))
    hmask &= np.abs(right - left) > 1e-12
    y_h, x_h = np.nonzero(hmask)

    top = shifted[:-1, :]
    bottom = shifted[1:, :]
    vmask = ((top <= 0.0) & (bottom > 0.0)) | ((top >= 0.0) & (bottom < 0.0))
    vmask &= np.abs(bottom - top) > 1e-12
    y_v, x_v = np.nonzero(vmask)

    if y_h.size == 0 and y_v.size == 0:
        return np.empty((0, 2), dtype=float)

    points: List[np.ndarray] = []
    if y_h.size:
        frac_h = -left[y_h, x_h] / (right[y_h, x_h] - left[y_h, x_h])
        points.append(np.column_stack([x_h.astype(np.float64) + frac_h, y_h.astype(np.float64)]))
    if y_v.size:
        frac_v = -top[y_v, x_v] / (bottom[y_v, x_v] - top[y_v, x_v])
        points.append(np.column_stack([x_v.astype(np.float64), y_v.astype(np.float64) + frac_v]))
    return np.vstack(points)


def sample_pair_series(values_a: np.ndarray, values_b: np.ndarray, max_samples: int = MAX_EXPORT_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    if values_a.size == 0 or values_b.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if values_a.size <= max_samples:
        return values_a, values_b
    sample_idx = np.linspace(0, values_a.size - 1, max_samples, dtype=int)
    return values_a[sample_idx], values_b[sample_idx]


def build_field_dataframe(result_dir: Path, input_dir: Path, indices: List[int]) -> pd.DataFrame:
    summary = load_summary(result_dir / "standard_sphere_gbt57_p2d_summary.csv")
    pixel_size_um = metric(summary, "pixel_size", "um/px")
    global_cx = metric(summary, "selected_circle_center_x", "px")
    global_cy = metric(summary, "selected_circle_center_y", "px")
    global_radius = metric(summary, "selected_circle_radius", "px")
    field_diag = pd.read_csv(result_dir / "cjsi_standard_sphere_field_diagnostics.csv")
    image_names = [image_name_from_index(index) for index in indices]
    edge_cloud_df = reconstruct_global_edge_cloud(
        result_dir,
        input_dir,
        pixel_size_um,
        global_cx,
        global_cy,
        global_radius,
        image_names=image_names,
    )

    rows: List[Dict[str, float | str]] = []
    for index in indices:
        image_name = image_name_from_index(index)
        diag_row = field_diag[field_diag["image_name"] == image_name]
        if diag_row.empty:
            continue
        diag = diag_row.iloc[0]
        image_cloud = edge_cloud_df[edge_cloud_df["image_name"] == image_name].copy()
        if image_cloud.empty:
            sampled_angles = np.empty(0, dtype=float)
            sampled_residual_um = np.empty(0, dtype=float)
            edge_rmse_um = float("nan")
            edge_mean_um = float("nan")
        else:
            angles = image_cloud["angle_deg"].to_numpy(dtype=float)
            residual_um = image_cloud["residual_um"].to_numpy(dtype=float)
            edge_rmse_um = float(np.sqrt(np.mean(np.square(residual_um)))) if residual_um.size else float("nan")
            edge_mean_um = float(np.mean(residual_um)) if residual_um.size else float("nan")
            sampled_angles, sampled_residual_um = sample_pair_series(angles, residual_um)

        corrected_residual_um = float(diag["corrected_residual_um"])
        diameter_error_um = 2.0 * corrected_residual_um
        rows.append({
            "image_index": index,
            "image_name": image_name,
            "label": image_name.replace(".bmp", ""),
            "assigned_angle_deg": float(diag["assigned_angle_deg"]),
            "selected_angle_deg": float(diag["selected_angle_deg"]),
            "target_angle_delta_deg": float(diag["target_angle_delta_deg"]),
            "window_half_angle_deg": float(diag["window_half_angle_deg"]),
            "window_violation_deg": float(diag["window_violation_deg"]),
            "corrected_residual_um": corrected_residual_um,
            "diameter_error_um": diameter_error_um,
            "abs_diameter_error_um": abs(diameter_error_um),
            "abs_residual_um": float(diag["abs_residual_um"]),
            "edge_rmse_um": edge_rmse_um,
            "edge_mean_um": edge_mean_um,
            "combined_um": math.sqrt(edge_rmse_um * edge_rmse_um + corrected_residual_um * corrected_residual_um)
            if math.isfinite(edge_rmse_um)
            else abs(corrected_residual_um),
            "angles_json": ";".join(f"{value:.4f}" for value in sampled_angles),
            "residual_json": ";".join(f"{value:.6f}" for value in sampled_residual_um),
        })

    df = pd.DataFrame(rows).sort_values("image_index").reset_index(drop=True)
    return df


def fit_linear_bias(df: pd.DataFrame) -> Tuple[float, float]:
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    angle = pd.to_numeric(df["selected_angle_deg"], errors="coerce").to_numpy(dtype=float)
    error = pd.to_numeric(df["diameter_error_um"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(angle) & np.isfinite(error)
    angle = angle[mask]
    error = error[mask]
    if len(angle) < 2:
        return 0.0, float(np.nanmean(error)) if len(error) else 0.0
    slope, intercept = np.polyfit(angle, error, 1)
    return float(slope), float(intercept)


def parse_series(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text:
        return np.empty(0, dtype=float)
    return np.array([float(item) for item in text.split(";") if item], dtype=float)


def draw_selected_points_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.34)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")
    circle = plt.Circle((0.0, 0.0), 1.0, facecolor="#111111", edgecolor="#111111", lw=0.8)
    ax.add_patch(circle)

    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(df)))
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        angle = math.radians(float(row["selected_angle_deg"]))
        x = math.cos(angle)
        y = math.sin(angle)
        ax.plot([0.0, x], [0.0, y], color="#222222", alpha=0.08, lw=0.7)
        ax.scatter(x, y, s=34, color=colors[idx - 1], edgecolors="white", linewidths=0.6, zorder=4)
        ax.text(1.05 * x, 1.05 * y, str(idx), fontsize=5.5, ha="center", va="center", color="#333333")

    ax.text(
        1.10,
        0.0,
        f"{len(df)} selected points",
        rotation=90,
        ha="left",
        va="center",
        fontsize=7.3,
        color="#555555",
    )
    ax.set_title("Selected point map")


def find_overlay_path(result_dir: Path) -> Optional[Path]:
    for candidate in [
        "standard_sphere_gbt57_p2d_overlay.png",
        "standard_sphere_gbt57_p2d_edge_overlay.png",
    ]:
        path = result_dir / candidate
        if path.exists():
            return path
    return None


def draw_global_detection_panel(ax: plt.Axes, result_dir: Path, summary: Dict[str, float | str]) -> None:
    ax.set_title("GB/T detection map")
    overlay_path = find_overlay_path(result_dir)
    if overlay_path is not None:
        image = load_overlay_image(overlay_path)
        ax.imshow(image, interpolation="nearest")
    else:
        ax.text(0.5, 0.5, "Overlay unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        (
            f"Global fit D = {metric(summary, 'global_circle_diameter', 'mm'):.4f} mm\n"
            f"Selected D = {metric(summary, 'selected_circle_diameter', 'mm'):.4f} mm\n"
            f"Pixel size = {metric(summary, 'pixel_size', 'um/px'):.3f} um/px\n"
            f"E_P2D = {metric(summary, 'e_p2d', 'um'):.3f} um"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D7D7D7", alpha=0.90),
    )


def draw_error_fluctuation_panel(ax: plt.Axes, df: pd.DataFrame, slope: float, intercept: float) -> None:
    ax.set_title("Diameter error fluctuation")
    ordered = df.sort_values("selected_angle_deg").reset_index(drop=True)
    angle = ordered["selected_angle_deg"].to_numpy(dtype=float)
    err = ordered["diameter_error_um"].to_numpy(dtype=float)
    fit = slope * angle + intercept
    detrended = err - fit

    ax.scatter(angle, err, s=15, color="#5C84C3", alpha=0.75, linewidths=0, label="Raw")
    ax.plot(angle, err, color="#5C84C3", lw=0.9, alpha=0.45)
    ax.plot(angle, fit, color="#D9791F", lw=1.2, label="Linear fit")
    ax.axhline(0.0, color="#555555", ls=":", lw=0.9)
    ax.set_xlabel("Selected angle (deg)")
    ax.set_ylabel("Diameter error (um)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.96,
        f"slope = {slope:.4f} um/deg\nbias-corrected RMSE = {float(np.sqrt(np.mean(detrended**2))):.3f} um",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D7D7D7", alpha=0.92),
    )


def draw_abs_hist_panel(ax: plt.Axes, df: pd.DataFrame, slope: float, intercept: float) -> None:
    ax.set_title("Absolute diameter error histogram")
    ordered = df.sort_values("selected_angle_deg").reset_index(drop=True)
    angle = ordered["selected_angle_deg"].to_numpy(dtype=float)
    err = ordered["diameter_error_um"].to_numpy(dtype=float)
    detrended = err - (slope * angle + intercept)
    abs_err = np.abs(detrended)

    bins = min(8, max(4, len(abs_err) // 3))
    ax.hist(abs_err, bins=bins, color="#D98A39", edgecolor="#A66519", alpha=0.92)
    ax.axvline(float(np.mean(abs_err)), color="#2F4858", ls="--", lw=1.0, label="Mean")
    ax.axvline(float(np.median(abs_err)), color="#5C84C3", ls=":", lw=1.0, label="Median")
    ax.set_xlabel("|Diameter error| (um)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")
    ax.text(
        0.98,
        0.96,
        f"mean = {float(np.mean(abs_err)):.3f} um\nmax = {float(np.max(abs_err)):.3f} um",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D7D7D7", alpha=0.92),
    )


def draw_edge_distribution_panel(ax: plt.Axes, row: pd.Series, is_left_edge: bool, is_bottom_edge: bool) -> None:
    angles = parse_series(str(row["angles_json"]))
    residual = parse_series(str(row["residual_json"]))
    selected_angle = float(row["selected_angle_deg"])
    corrected_residual_um = float(row["corrected_residual_um"])
    edge_rmse_um = float(row["edge_rmse_um"])

    if angles.size and residual.size:
        ax.scatter(angles, residual, s=1.8, color="#6A5ACD", alpha=0.60, linewidths=0)
        bin_edges = np.linspace(0.0, 360.0, 73)
        centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        means = np.full_like(centers, np.nan, dtype=float)
        for i in range(len(bin_edges) - 1):
            mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
            if np.any(mask):
                means[i] = float(np.mean(residual[mask]))
        finite = np.isfinite(means)
        if np.any(finite):
            finite_idx = np.flatnonzero(finite)
            split_points = np.where(np.diff(centers[finite_idx]) > 8.0)[0] + 1
            for segment in np.split(finite_idx, split_points):
                if segment.size >= 2:
                    ax.plot(centers[segment], means[segment], color="#26A69A", lw=0.95)

    ax.axhline(0.0, color="#4D4D4D", ls=":", lw=0.8)
    if math.isfinite(edge_rmse_um):
        ax.axhspan(-edge_rmse_um, edge_rmse_um, color="#CFE4D8", alpha=0.18, lw=0)
        ax.axhline(edge_rmse_um, color="#9BC3AE", lw=0.75)
        ax.axhline(-edge_rmse_um, color="#9BC3AE", lw=0.75)
    ax.scatter([selected_angle], [corrected_residual_um], s=23, color="#FF7043", edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-RIGHT_PANEL_YLIM_UM, RIGHT_PANEL_YLIM_UM)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks(np.linspace(-RIGHT_PANEL_YLIM_UM, RIGHT_PANEL_YLIM_UM, 5))
    ax.grid(True, color="#DDDDDD", lw=0.35)
    ax.set_title(f"{row['label']} global residuals", fontsize=6.8, pad=2.0)
    if is_left_edge:
        ax.set_ylabel("Residual (um)", fontsize=6.4)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if is_bottom_edge:
        ax.set_xlabel("Angle (deg)", fontsize=6.4)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)


def export_figure(
    result_dir: Path,
    df: pd.DataFrame,
    summary: Dict[str, float | str],
    output_dir: Path,
    svg_fonttype: str,
) -> None:
    fig = plt.figure(figsize=(24.0, 15.0), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)
    left = outer[0].subgridspec(2, 2, wspace=0.20, hspace=0.24)
    right = outer[1].subgridspec(5, 5, wspace=0.18, hspace=0.28)

    fig.text(
        0.33,
        0.988,
        (
            f"Global pose optimized standard sphere | D = {metric(summary, 'sphere_diameter_nominal', 'mm'):.4f} mm "
            f"| E_P2D = {metric(summary, 'e_p2d', 'um'):.3f} um"
        ),
        ha="center",
        va="top",
        fontsize=11.0,
        color="#333333",
    )

    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left[0, 1])
    ax_c = fig.add_subplot(left[1, 0])
    ax_d = fig.add_subplot(left[1, 1])

    draw_selected_points_panel(ax_a, df)
    draw_global_detection_panel(ax_b, result_dir, summary)

    slope, intercept = fit_linear_bias(df)
    draw_error_fluctuation_panel(ax_c, df, slope, intercept)
    draw_abs_hist_panel(ax_d, df, slope, intercept)

    ordered = df.sort_values("image_index").reset_index(drop=True)
    for idx, (_, row) in enumerate(ordered.iterrows()):
        ax = fig.add_subplot(right[idx // 5, idx % 5])
        draw_edge_distribution_panel(
            ax,
            row,
            is_left_edge=(idx % 5 == 0),
            is_bottom_edge=(idx // 5 == 4),
        )

    fig.text(
        0.17,
        0.028,
        "Global pose optimization before point selection",
        ha="center",
        va="center",
        color="white",
        fontsize=10.8,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#3F6FB6", edgecolor="none"),
    )
    fig.text(
        0.73,
        0.028,
        "Per-field reconstructed global subpixel residuals",
        ha="center",
        va="center",
        color="white",
        fontsize=10.8,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#3F6FB6", edgecolor="none"),
    )

    fig.savefig(output_dir / OUTPUT_PNG_NAME, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / OUTPUT_PDF_NAME, bbox_inches="tight")
    fig.savefig(output_dir / OUTPUT_SVG_NAME, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.svg_fonttype)

    result_dir = args.result_dir.resolve()
    input_dir = resolve_input_dir(args)
    output_dir = (args.output_dir or result_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_cleanup = pd.read_csv(result_dir / "standard_sphere_circle_edge_cleanup.csv")
    fallback_indices = [int(str(name).split("_")[1].split(".")[0]) for name in edge_cleanup["image_name"].tolist()]
    indices = parse_indices(args.image_indices, fallback=fallback_indices)

    df = build_field_dataframe(result_dir, input_dir, indices)
    if df.empty:
        raise RuntimeError("no fields available for plotting")

    slope, intercept = fit_linear_bias(df)
    ordered = df.sort_values("selected_angle_deg").reset_index(drop=True)
    df["linear_bias_diameter_um"] = slope * pd.to_numeric(df["selected_angle_deg"], errors="coerce").to_numpy(dtype=float) + intercept
    df["bias_corrected_diameter_error_um"] = df["diameter_error_um"] - df["linear_bias_diameter_um"]
    df["bias_corrected_abs_diameter_error_um"] = np.abs(df["bias_corrected_diameter_error_um"])
    df.to_csv(output_dir / OUTPUT_CSV_NAME, index=False, encoding="utf-8-sig")

    summary = load_summary(result_dir / "standard_sphere_gbt57_p2d_summary.csv")
    export_figure(result_dir, df, summary, output_dir, args.svg_fonttype)
    print(f"[OK] presentation figure exported under: {output_dir}")


if __name__ == "__main__":
    main()
