#!/usr/bin/env python3
"""
Export a presentation-style figure for the optimized standard-sphere result.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


OUTPUT_PNG_NAME = "standard_sphere_optimized_presentation_figure.png"
OUTPUT_PDF_NAME = "standard_sphere_optimized_presentation_figure.pdf"
OUTPUT_SVG_NAME = "standard_sphere_optimized_presentation_figure.svg"
OUTPUT_CSV_NAME = "standard_sphere_optimized_presentation_metrics.csv"
EDGE_THRESHOLD_GRAY = 128.0
EDGE_RESIDUAL_FILTER_PX = 1.2
MAX_EXPORT_SAMPLES = 1400
RIGHT_PANEL_YLIM_UM = 15.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export optimized standard-sphere presentation figure.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--image-indices", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def configure_matplotlib(svg_fonttype: str) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
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


def parse_indices(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("image-indices is empty")
    return values


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


def image_name_from_index(index: int) -> str:
    return f"Pic_{index}.bmp"


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


def build_selected_angle_offsets(result_dir: Path, cleanup_df: pd.DataFrame) -> pd.DataFrame:
    points_df = pd.read_csv(result_dir / "standard_sphere_gbt57_p2d_points.csv")
    selected_df = points_df.loc[
        points_df["selected"] == 1,
        ["image_name", "image_x_px", "image_y_px", "selected_angle_deg"],
    ].copy()
    merged = selected_df.merge(cleanup_df[["image_name", "center_x_px", "center_y_px"]], on="image_name", how="inner")
    merged["selected_local_angle_deg"] = (
        np.degrees(
            np.arctan2(
                merged["image_y_px"].to_numpy(dtype=float) - merged["center_y_px"].to_numpy(dtype=float),
                merged["image_x_px"].to_numpy(dtype=float) - merged["center_x_px"].to_numpy(dtype=float),
            )
        )
        + 360.0
    ) % 360.0
    merged["angle_offset_deg"] = (
        merged["selected_angle_deg"].to_numpy(dtype=float) - merged["selected_local_angle_deg"].to_numpy(dtype=float)
    )
    return merged[["image_name", "selected_angle_deg", "angle_offset_deg"]].copy()


def reconstruct_global_edge_cloud(
    result_dir: Path,
    input_dir: Path,
    pixel_size_um: float,
    global_cx: float,
    global_cy: float,
    global_radius: float,
    image_names: List[str] | None = None,
) -> pd.DataFrame:
    cleanup_df = pd.read_csv(result_dir / "standard_sphere_circle_edge_cleanup.csv")
    if image_names is not None:
        image_name_set = set(image_names)
        cleanup_df = cleanup_df[cleanup_df["image_name"].isin(image_name_set)].copy()
    if cleanup_df.empty:
        return pd.DataFrame()

    offset_df = build_selected_angle_offsets(result_dir, cleanup_df)
    merged = cleanup_df.merge(offset_df, on="image_name", how="inner")

    rows: List[Dict[str, float | str]] = []
    for _, row in merged.iterrows():
        image_name = str(row["image_name"])
        image_path = input_dir / image_name
        if not image_path.exists():
            continue

        points = extract_edge_points(image_path)
        if points.size == 0:
            continue

        cx = float(row["center_x_px"])
        cy = float(row["center_y_px"])
        radius = float(row["radius_px"])
        radial = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        residual_px = radial - radius
        keep = np.abs(residual_px) <= EDGE_RESIDUAL_FILTER_PX
        filtered = points[keep]
        residual_px = residual_px[keep]
        if filtered.size == 0:
            continue

        local_angle_deg = (np.degrees(np.arctan2(filtered[:, 1] - cy, filtered[:, 0] - cx)) + 360.0) % 360.0
        angle_offset_deg = float(row["angle_offset_deg"])
        global_angle_deg = (local_angle_deg + angle_offset_deg) % 360.0
        reconstructed_radius_px = global_radius + residual_px
        global_angle_rad = np.deg2rad(global_angle_deg)
        global_x_px = global_cx + reconstructed_radius_px * np.cos(global_angle_rad)
        global_y_px = global_cy + reconstructed_radius_px * np.sin(global_angle_rad)
        residual_um = residual_px * pixel_size_um

        for x_val, y_val, angle_val, residual_px_val, residual_um_val in zip(
            global_x_px,
            global_y_px,
            global_angle_deg,
            residual_px,
            residual_um,
        ):
            rows.append({
                "image_name": image_name,
                "global_x_px": float(x_val),
                "global_y_px": float(y_val),
                "angle_deg": float(angle_val),
                "residual_px": float(residual_px_val),
                "residual_um": float(residual_um_val),
            })

    return pd.DataFrame(rows)


def wrap_span(ax: plt.Axes, center_deg: float, half_deg: float, color: str, alpha: float) -> None:
    start = center_deg - half_deg
    end = center_deg + half_deg
    if start < 0:
        ax.axvspan(start + 360.0, 360.0, color=color, alpha=alpha, lw=0)
        ax.axvspan(0.0, end, color=color, alpha=alpha, lw=0)
    elif end > 360.0:
        ax.axvspan(start, 360.0, color=color, alpha=alpha, lw=0)
        ax.axvspan(0.0, end - 360.0, color=color, alpha=alpha, lw=0)
    else:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)


def build_representative_dataframe(
    result_dir: Path,
    input_dir: Path,
    indices: List[int],
) -> pd.DataFrame:
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
        if math.isfinite(edge_rmse_um):
            combined_um = math.sqrt(edge_rmse_um * edge_rmse_um + corrected_residual_um * corrected_residual_um)
        else:
            combined_um = abs(corrected_residual_um)
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
            "abs_residual_um": float(diag["abs_residual_um"]),
            "edge_rmse_um": edge_rmse_um,
            "edge_mean_um": edge_mean_um,
            "combined_um": combined_um,
            "e0_contribution_um": 2.0 * abs(corrected_residual_um),
            "diameter_error_um": 2.0 * corrected_residual_um,
            "angles_json": ";".join(f"{value:.4f}" for value in sampled_angles),
            "residual_json": ";".join(f"{value:.6f}" for value in sampled_residual_um),
        })

    return pd.DataFrame(rows).sort_values("image_index").reset_index(drop=True)


def parse_series(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text:
        return np.empty(0, dtype=float)
    return np.array([float(item) for item in text.split(";") if item], dtype=float)


def draw_circle_panel(ax: plt.Axes, rep_df: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    circle = plt.Circle((0.0, 0.0), 1.0, facecolor="black", edgecolor="black", lw=0.5)
    ax.add_patch(circle)

    colors = ["#17C0EB", "#FF66C4"]
    ys: List[float] = []
    for idx, row in rep_df.iterrows():
        angle = math.radians(float(row["selected_angle_deg"]))
        x = math.cos(angle)
        y = math.sin(angle)
        ys.append(y)
        ax.plot([0.0, x], [0.0, y], color="#111111", alpha=0.08, lw=0.7)
        ax.scatter(x, y, s=34, color=colors[idx % 2], edgecolors="white", linewidths=0.7, zorder=4)

    if ys:
        y0 = min(ys)
        y1 = max(ys)
        bx = 1.12
        ax.plot([bx, bx], [y0, y1], color="#4A4A4A", lw=1.0)
        ax.plot([bx - 0.03, bx], [y0, y0], color="#4A4A4A", lw=1.0)
        ax.plot([bx - 0.03, bx], [y1, y1], color="#4A4A4A", lw=1.0)
        ax.text(bx + 0.03, (y0 + y1) * 0.5, "9 representative fields", rotation=90, va="center", ha="left", fontsize=7.5)


def draw_residual_summary(ax: plt.Axes, rep_df: pd.DataFrame) -> None:
    x = np.arange(len(rep_df))
    ax.plot(x, rep_df["edge_rmse_um"], color="#26A69A", marker="o", ms=3.5, lw=1.2, label="Circle RMSE")
    ax.plot(x, rep_df["combined_um"], color="#5C84C3", marker="s", ms=3.2, lw=1.2, label="Combined")
    ax.axhline(1.0, color="#E57373", ls="--", lw=1.0, label="1 um target")
    ax.set_xticks(x)
    ax.set_xticklabels(rep_df["label"], rotation=28, ha="right")
    ax.set_ylabel("um")
    ax.set_title("Residual imaging error")
    ymax = max(3.0, float(np.nanmax(rep_df[["edge_rmse_um", "combined_um"]].to_numpy(dtype=float))) * 1.12)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")


def draw_e0_panel(ax: plt.Axes, rep_df: pd.DataFrame, overall_e_p2d_um: float) -> None:
    x = np.arange(len(rep_df))
    ax.bar(x, rep_df["e0_contribution_um"], color="#D8A76C", edgecolor="#C88A43", linewidth=0.6)
    ax.axhline(overall_e_p2d_um, color="#8C6B3F", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(rep_df["label"], rotation=28, ha="right")
    ax.set_ylabel("um")
    ax.set_title("GB/T E0 contribution")
    ax.set_ylim(0.0, max(1.6, float(rep_df["e0_contribution_um"].max()) * 1.35))
    ax.grid(True, axis="y")
    ax.text(0.98, 0.92, f"E_P2D = {overall_e_p2d_um:.3f} um", transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color="#8C6B3F")


def draw_diameter_panel(ax: plt.Axes, rep_df: pd.DataFrame) -> None:
    x = np.arange(len(rep_df))
    values = rep_df["diameter_error_um"].to_numpy(dtype=float)
    colors = np.where(values >= 0.0, "#E07A1F", "#D67A5C")
    ax.bar(x, values, color=colors, edgecolor="#B45F06", linewidth=0.6)
    ax.axhline(0.0, color="#666666", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(rep_df["label"], rotation=28, ha="right")
    ax.set_ylabel("um")
    ax.set_title("Diameter error")
    limit = max(1.5, float(np.max(np.abs(values))) * 1.35)
    ax.set_ylim(-limit, limit)
    ax.grid(True, axis="y")


def draw_angle_residual_panel(ax: plt.Axes, row: pd.Series) -> None:
    angles = parse_series(str(row["angles_json"]))
    residual = parse_series(str(row["residual_json"]))
    selected_angle = float(row["selected_angle_deg"])
    assigned_angle = float(row["assigned_angle_deg"])
    half_angle = float(row["window_half_angle_deg"])
    edge_rmse_um = float(row["edge_rmse_um"])

    wrap_span(ax, assigned_angle, half_angle, "#F8DDE6", 0.55)
    if angles.size and residual.size:
        ax.scatter(angles, residual, s=2.0, color="#6A5ACD", alpha=0.75, linewidths=0)
        bin_edges = np.linspace(0.0, 360.0, 73)
        centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        mean_values = np.full_like(centers, np.nan, dtype=float)
        for idx in range(len(bin_edges) - 1):
            mask = (angles >= bin_edges[idx]) & (angles < bin_edges[idx + 1])
            if np.any(mask):
                mean_values[idx] = float(np.mean(residual[mask]))
        finite = np.isfinite(mean_values)
        if np.any(finite):
            finite_idx = np.flatnonzero(finite)
            split_points = np.where(np.diff(centers[finite_idx]) > 7.5)[0] + 1
            for segment in np.split(finite_idx, split_points):
                if segment.size >= 2:
                    ax.plot(centers[segment], mean_values[segment], color="#26A69A", lw=1.0)
    ax.axhline(0.0, color="#4D4D4D", ls=":", lw=0.9)
    if math.isfinite(edge_rmse_um):
        ax.axhspan(-edge_rmse_um, edge_rmse_um, color="#CFE4D8", alpha=0.18, lw=0)
        ax.axhline(edge_rmse_um, color="#9BC3AE", lw=0.8)
        ax.axhline(-edge_rmse_um, color="#9BC3AE", lw=0.8)
    ax.scatter([selected_angle], [float(row["corrected_residual_um"])], s=26, color="#FF7043", edgecolors="white", linewidths=0.5, zorder=5)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-RIGHT_PANEL_YLIM_UM, RIGHT_PANEL_YLIM_UM)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax.set_yticks(np.linspace(-RIGHT_PANEL_YLIM_UM, RIGHT_PANEL_YLIM_UM, 5))
    ax.grid(True, color="#DDDDDD")
    ax.set_title(f"{row['label']} global residual-vs-angle", fontsize=7.0)
    ax.set_xlabel("Angle (deg)", fontsize=6.6)
    ax.set_ylabel("Residual error (um)", fontsize=6.4)


def export_figure(rep_df: pd.DataFrame, summary: Dict[str, float | str], output_dir: Path) -> None:
    fig = plt.figure(figsize=(15.2, 7.3), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.10)
    left = outer[0].subgridspec(2, 2, wspace=0.28, hspace=0.28)
    right = outer[1].subgridspec(3, 3, wspace=0.25, hspace=0.32)

    fig.text(
        0.34,
        0.985,
        f"Optimized standard-sphere display | D = {metric(summary, 'sphere_diameter_nominal', 'mm'):.4f} mm | E_P2D = {metric(summary, 'e_p2d', 'um'):.3f} um",
        ha="center",
        va="top",
        fontsize=10.0,
        color="#333333",
    )

    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left[0, 1])
    ax_c = fig.add_subplot(left[1, 0])
    ax_d = fig.add_subplot(left[1, 1])

    draw_circle_panel(ax_a, rep_df)
    draw_residual_summary(ax_b, rep_df)
    draw_e0_panel(ax_c, rep_df, metric(summary, "e_p2d", "um"))
    draw_diameter_panel(ax_d, rep_df)

    for idx, (_, row) in enumerate(rep_df.iterrows()):
        ax = fig.add_subplot(right[idx // 3, idx % 3])
        draw_angle_residual_panel(ax, row)

    fig.text(
        0.18,
        0.028,
        "Representative-field optimized result",
        ha="center",
        va="center",
        color="white",
        fontsize=11.0,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#3F6FB6", edgecolor="none"),
    )
    fig.text(
        0.72,
        0.028,
        "Reconstructed global subpixel residuals",
        ha="center",
        va="center",
        color="white",
        fontsize=11.0,
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

    indices = parse_indices(args.image_indices)
    rep_df = build_representative_dataframe(result_dir, input_dir, indices)
    if rep_df.empty:
        raise RuntimeError("no representative fields available for plotting")

    rep_df.to_csv(output_dir / OUTPUT_CSV_NAME, index=False, encoding="utf-8-sig")
    summary = load_summary(result_dir / "standard_sphere_gbt57_p2d_summary.csv")
    export_figure(rep_df, summary, output_dir)
    print(f"[OK] presentation figure exported under: {output_dir}")


if __name__ == "__main__":
    main()
