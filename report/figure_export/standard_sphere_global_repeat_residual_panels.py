#!/usr/bin/env python3
"""
Export individual residual panels using the full stitched global subpixel edge cloud.
Each panel represents one repeat of whole-sphere GB/T-style point selection.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from standard_sphere_optimized_presentation_figure import (
    EDGE_RESIDUAL_FILTER_PX,
    configure_matplotlib,
    extract_edge_points,
    load_summary,
    metric,
    reconstruct_global_edge_cloud,
    resolve_input_dir,
)


OUTPUT_DIR_NAME = "global_repeat_residual_panels"
OUTPUT_MANIFEST_NAME = "global_repeat_residual_panels_manifest.csv"
OUTPUT_CLOUD_NAME = "global_all_subpixel_reconstructed_cloud.csv"
TARGET_POINT_COUNT = 25
TARGET_SPACING_DEG = 360.0 / TARGET_POINT_COUNT
TARGET_HALF_WINDOW_DEG = 4.0
DEFAULT_REPEAT_COUNT = 9
MIN_RIGHT_PANEL_YLIM_UM = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-global all-subpixel repeat residual panels.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def angle_diff_deg(angle: np.ndarray, target_deg: float) -> np.ndarray:
    return ((angle - target_deg + 180.0) % 360.0) - 180.0


def fit_circle_least_squares(points_xy: np.ndarray) -> tuple[float, float, float]:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c0 = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = math.sqrt(max(c0 + cx * cx + cy * cy, 0.0))
    return float(cx), float(cy), float(radius)


def compute_residual_stats(residual_um: np.ndarray) -> dict[str, float]:
    if residual_um.size == 0:
        return {
            "rmse_um": float("nan"),
            "mean_abs_um": float("nan"),
            "p95_abs_um": float("nan"),
            "p99_abs_um": float("nan"),
            "max_abs_um": float("nan"),
        }

    abs_residual_um = np.abs(residual_um)
    return {
        "rmse_um": float(np.sqrt(np.mean(np.square(residual_um)))),
        "mean_abs_um": float(np.mean(abs_residual_um)),
        "p95_abs_um": float(np.percentile(abs_residual_um, 95.0)),
        "p99_abs_um": float(np.percentile(abs_residual_um, 99.0)),
        "max_abs_um": float(np.max(abs_residual_um)),
    }


def choose_residual_ylim_um(results: List[dict[str, float | pd.DataFrame | np.ndarray]]) -> float:
    max_abs_um = 0.0
    for result in results:
        stats = result.get("residual_stats", {})
        if isinstance(stats, dict):
            max_abs_um = max(max_abs_um, float(stats.get("max_abs_um", 0.0)))

    padded = max_abs_um * 1.05
    if padded <= 10.0:
        step = 2.0
    elif padded <= 20.0:
        step = 5.0
    elif padded <= 50.0:
        step = 5.0
    else:
        step = 10.0
    limit = math.ceil(max(padded, MIN_RIGHT_PANEL_YLIM_UM) / step) * step
    return float(limit)


def build_global_edge_cloud(
    result_dir: Path,
    input_dir: Path,
    pixel_size_um: float,
    global_cx: float,
    global_cy: float,
    global_radius: float,
) -> pd.DataFrame:
    edge_df = reconstruct_global_edge_cloud(
        result_dir,
        input_dir,
        pixel_size_um,
        global_cx,
        global_cy,
        global_radius,
    )
    if edge_df.empty:
        return edge_df
    edge_df["local_residual_px"] = edge_df["residual_px"].to_numpy(dtype=float)
    edge_df["local_residual_um"] = edge_df["residual_um"].to_numpy(dtype=float)
    return edge_df


def annotate_global_pose(edge_df: pd.DataFrame, cx: float, cy: float, radius: float, pixel_size_um: float) -> pd.DataFrame:
    annotated = edge_df.copy()
    dx = annotated["global_x_px"].to_numpy(dtype=float) - cx
    dy = annotated["global_y_px"].to_numpy(dtype=float) - cy
    radial = np.hypot(dx, dy)
    annotated["angle_deg"] = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    annotated["residual_px"] = radial - radius
    annotated["residual_um"] = annotated["residual_px"].to_numpy(dtype=float) * pixel_size_um
    return annotated


def select_points_for_phase(edge_df: pd.DataFrame, phase_deg: float) -> pd.DataFrame:
    selected_parts: List[pd.DataFrame] = []
    for idx in range(TARGET_POINT_COUNT):
        target_deg = (phase_deg + idx * TARGET_SPACING_DEG) % 360.0
        diff = angle_diff_deg(edge_df["angle_deg"].to_numpy(dtype=float), target_deg)
        mask = np.abs(diff) <= TARGET_HALF_WINDOW_DEG
        if not np.any(mask):
            relaxed_idx = int(np.argmin(np.abs(diff)))
            chosen = edge_df.iloc[[relaxed_idx]].copy()
            chosen["target_angle_deg"] = target_deg
            chosen["target_diff_deg"] = float(diff[relaxed_idx])
            selected_parts.append(chosen)
            continue
        candidates = edge_df.loc[mask].copy()
        candidates["target_diff_deg"] = diff[mask]
        cost = np.abs(candidates["residual_px"].to_numpy(dtype=float)) + 0.03 * np.abs(candidates["target_diff_deg"].to_numpy(dtype=float))
        best_idx = int(np.argmin(cost))
        chosen = candidates.iloc[[best_idx]].copy()
        chosen["target_angle_deg"] = target_deg
        selected_parts.append(chosen)
    selected_df = pd.concat(selected_parts, ignore_index=True)
    return selected_df


def evaluate_phase(edge_df: pd.DataFrame, phase_deg: float, pixel_size_um: float, nominal_diameter_mm: float) -> dict[str, float | pd.DataFrame]:
    selected_df = select_points_for_phase(edge_df, phase_deg)
    points_xy = selected_df[["global_x_px", "global_y_px"]].to_numpy(dtype=float)
    fit_cx, fit_cy, fit_radius = fit_circle_least_squares(points_xy)
    fitted_diameter_mm = 2.0 * fit_radius * pixel_size_um / 1000.0
    abs_diameter_error_um = abs(fitted_diameter_mm - nominal_diameter_mm) * 1000.0

    all_dx = edge_df["global_x_px"].to_numpy(dtype=float) - fit_cx
    all_dy = edge_df["global_y_px"].to_numpy(dtype=float) - fit_cy
    all_radial = np.hypot(all_dx, all_dy)
    all_angle = (np.degrees(np.arctan2(all_dy, all_dx)) + 360.0) % 360.0
    all_residual_um = (all_radial - fit_radius) * pixel_size_um
    residual_stats = compute_residual_stats(all_residual_um)
    return {
        "phase_deg": float(phase_deg),
        "fit_cx_px": float(fit_cx),
        "fit_cy_px": float(fit_cy),
        "fit_radius_px": float(fit_radius),
        "fitted_diameter_mm": float(fitted_diameter_mm),
        "abs_diameter_error_um": float(abs_diameter_error_um),
        "selected_df": selected_df,
        "all_angle_deg": all_angle,
        "all_residual_um": all_residual_um,
        "residual_stats": residual_stats,
    }


def find_best_base_phase(edge_df: pd.DataFrame, pixel_size_um: float, nominal_diameter_mm: float) -> float:
    phase_candidates = np.arange(0.0, TARGET_SPACING_DEG, 0.2, dtype=float)
    best_phase = 0.0
    best_score = float("inf")
    for phase_deg in phase_candidates:
        result = evaluate_phase(edge_df, phase_deg, pixel_size_um, nominal_diameter_mm)
        score = float(result["abs_diameter_error_um"])
        if score < best_score:
            best_score = score
            best_phase = float(phase_deg)
    return best_phase


def draw_repeat_panel(
    ax: plt.Axes,
    repeat_label: str,
    angle_deg: np.ndarray,
    residual_um: np.ndarray,
    diameter_error_um: float,
    rmse_um: float,
    y_limit_um: float,
) -> None:
    ax.scatter(angle_deg, residual_um, s=1.8, color="#6A5ACD", alpha=0.58, linewidths=0)

    bin_edges = np.linspace(0.0, 360.0, 73)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mean_values = np.full_like(centers, np.nan, dtype=float)
    for idx in range(len(bin_edges) - 1):
        mask = (angle_deg >= bin_edges[idx]) & (angle_deg < bin_edges[idx + 1])
        if np.any(mask):
            mean_values[idx] = float(np.mean(residual_um[mask]))
    finite = np.isfinite(mean_values)
    if np.any(finite):
        finite_idx = np.flatnonzero(finite)
        split_points = np.where(np.diff(centers[finite_idx]) > 7.5)[0] + 1
        for segment in np.split(finite_idx, split_points):
            if segment.size >= 2:
                ax.plot(centers[segment], mean_values[segment], color="#26A69A", lw=1.0)

    if math.isfinite(rmse_um):
        ax.axhspan(-rmse_um, rmse_um, color="#CFE4D8", alpha=0.18, lw=0)
        ax.axhline(rmse_um, color="#9BC3AE", lw=0.8)
        ax.axhline(-rmse_um, color="#9BC3AE", lw=0.8)

    ax.axhline(0.0, color="#4D4D4D", ls=":", lw=0.9)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-y_limit_um, y_limit_um)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks(np.linspace(-y_limit_um, y_limit_um, 5))
    ax.grid(True, color="#DDDDDD", lw=0.35)
    ax.set_title(f"{repeat_label} global all-subpixel residuals", fontsize=8.6, pad=5.0)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Residual (um)")
    ax.text(
        0.98,
        0.96,
        f"RMSE = {rmse_um:.3f} um\n|D-D0| = {abs(diameter_error_um):.3f} um",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#D7D7D7", alpha=0.92),
    )


def export_single_repeat(
    output_dir: Path,
    repeat_index: int,
    result: dict[str, float | pd.DataFrame | np.ndarray],
    y_limit_um: float,
) -> dict[str, str | float]:
    repeat_label = f"Rc_{repeat_index}"
    base_name = f"{repeat_label}_global_all_subpixel_residual"
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"
    pdf_path = output_dir / f"{base_name}.pdf"
    residual_stats = dict(result.get("residual_stats", {}))

    fig, ax = plt.subplots(figsize=(5.6, 4.25), facecolor="white")
    draw_repeat_panel(
        ax,
        repeat_label=repeat_label,
        angle_deg=np.asarray(result["all_angle_deg"], dtype=float),
        residual_um=np.asarray(result["all_residual_um"], dtype=float),
        diameter_error_um=(float(result["fitted_diameter_mm"]) - nominal_diameter_mm_global) * 1000.0,
        rmse_um=float(residual_stats.get("rmse_um", float("nan"))),
        y_limit_um=y_limit_um,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "repeat_label": repeat_label,
        "phase_deg": float(result["phase_deg"]),
        "fit_cx_px": float(result["fit_cx_px"]),
        "fit_cy_px": float(result["fit_cy_px"]),
        "fit_radius_px": float(result["fit_radius_px"]),
        "fitted_diameter_mm": float(result["fitted_diameter_mm"]),
        "abs_diameter_error_um": float(result["abs_diameter_error_um"]),
        "rmse_um": float(residual_stats.get("rmse_um", float("nan"))),
        "mean_abs_um": float(residual_stats.get("mean_abs_um", float("nan"))),
        "p95_abs_um": float(residual_stats.get("p95_abs_um", float("nan"))),
        "p99_abs_um": float(residual_stats.get("p99_abs_um", float("nan"))),
        "max_abs_um": float(residual_stats.get("max_abs_um", float("nan"))),
        "y_limit_um": float(y_limit_um),
        "png_path": str(png_path),
        "svg_path": str(svg_path),
        "pdf_path": str(pdf_path),
    }


nominal_diameter_mm_global = float("nan")


def main() -> None:
    global nominal_diameter_mm_global

    args = parse_args()
    configure_matplotlib(args.svg_fonttype)

    result_dir = args.result_dir.resolve()
    input_dir = resolve_input_dir(args)
    output_dir = (args.output_dir or (result_dir / OUTPUT_DIR_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(result_dir / "standard_sphere_gbt57_p2d_summary.csv")
    pixel_size_um = metric(summary, "pixel_size", "um/px")
    nominal_diameter_mm_global = metric(summary, "sphere_diameter_nominal", "mm")
    global_cx = metric(summary, "selected_circle_center_x", "px")
    global_cy = metric(summary, "selected_circle_center_y", "px")
    global_radius = metric(summary, "selected_circle_radius", "px")

    edge_df = build_global_edge_cloud(result_dir, input_dir, pixel_size_um, global_cx, global_cy, global_radius)
    if edge_df.empty:
        raise RuntimeError("no global edge cloud available")
    edge_df = annotate_global_pose(edge_df, global_cx, global_cy, global_radius, pixel_size_um)
    edge_df.to_csv(output_dir / OUTPUT_CLOUD_NAME, index=False, encoding="utf-8-sig")

    best_phase = find_best_base_phase(edge_df, pixel_size_um, nominal_diameter_mm_global)
    phase_offsets = best_phase + np.arange(args.repeat_count, dtype=float) * (TARGET_SPACING_DEG / args.repeat_count)

    repeat_results: List[dict[str, float | pd.DataFrame | np.ndarray]] = []
    for phase_deg in phase_offsets:
        repeat_results.append(evaluate_phase(edge_df, float(phase_deg % 360.0), pixel_size_um, nominal_diameter_mm_global))
    y_limit_um = choose_residual_ylim_um(repeat_results)

    manifest_rows: List[dict[str, str | float]] = []
    for repeat_index, result in enumerate(repeat_results, start=1):
        manifest_rows.append(export_single_repeat(output_dir, repeat_index, result, y_limit_um))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / OUTPUT_MANIFEST_NAME, index=False, encoding="utf-8-sig")
    print(f"[OK] exported {len(manifest_rows)} global-repeat residual panels under: {output_dir}")


if __name__ == "__main__":
    main()
