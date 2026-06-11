#!/usr/bin/env python3
"""
Batch repeatability validation for the proposed method and two recent literature
baselines on the fire-tube generatrix dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    ("proposed", "所提方法"),
    ("recent_profile_grinding_2025", "对比方法A（2025平滑曲线拼接）"),
    ("recent_telecentric_scan_stitch_2026", "对比方法B（2026远心扫描拼接）"),
]

METHOD_COLORS = {
    "proposed": "#1B9E77",
    "recent_profile_grinding_2025": "#E66101",
    "recent_telecentric_scan_stitch_2026": "#7570B3",
}

FIGURE_PNG_NAME = "repeatability_validation_figure.png"
FIGURE_PDF_NAME = "repeatability_validation_figure.pdf"
FIGURE_SVG_NAME = "repeatability_validation_figure.svg"
GROUP_METRICS_CSV_NAME = "repeatability_metrics_by_group.csv"
SUMMARY_CSV_NAME = "repeatability_summary_by_method.csv"
CURVE_STATS_CSV_NAME = "repeatability_curve_stats.csv"
REPORT_MD_NAME = "repeatability_validation_report.md"


@dataclass
class MethodResult:
    group_name: str
    method_code: str
    method_label: str
    normal_rmse_um: float
    profile_rms_um: float
    profile_p95_abs_um: float
    mean_normal_error_um: float
    used_count: int
    outlier_ratio: float
    worst_step_rmse_px: float
    worst_step_rmse_um: float
    curve_s_mm: np.ndarray
    curve_profile_error_um: np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run repeatability validation for the proposed method and recent literature baselines."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "火焰筒" / "母线拼接" / "重复性验证",
        help="Root directory of repeated image groups.",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=root / "build" / "bin" / "Release" / "pinjie_cli.exe",
        help="Path to pinjie_cli executable.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "result" / "repeatability_validation_recent_methods",
        help="Output root for all repeatability results.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of groups to run, e.g. 1 2 3.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.78,
        help="Configured overlap ratio for the proposed CLI run.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="x+",
        help="Motion direction for the proposed CLI run.",
    )
    parser.add_argument(
        "--timeout-sec-per-group",
        type=int,
        default=5400,
        help="Timeout for one proposed-method CLI run.",
    )
    parser.add_argument(
        "--rerun-proposed",
        action="store_true",
        help="Force rerunning the proposed-method CLI even if outputs already exist.",
    )
    parser.add_argument(
        "--svg-fonttype",
        choices=("path", "none"),
        default="path",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def collect_group_dirs(input_root: Path, groups: Sequence[str] | None) -> list[Path]:
    dirs = [path for path in input_root.iterdir() if path.is_dir()]
    dirs.sort(key=lambda p: natural_key(p.name))
    if not groups:
        return dirs
    wanted = {str(item) for item in groups}
    filtered = [path for path in dirs if path.name in wanted]
    missing = sorted(wanted - {path.name for path in filtered}, key=natural_key)
    if missing:
        raise FileNotFoundError(f"missing group directories: {missing}")
    return filtered


def collect_image_paths(group_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in group_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ]
    image_paths.sort(key=lambda p: natural_key(p.name))
    if not image_paths:
        raise FileNotFoundError(f"no supported images found under {group_dir}")
    return image_paths


def run_proposed_cli(
    cli_path: Path,
    group_dir: Path,
    output_dir: Path,
    image_count: int,
    overlap: float,
    direction: str,
    timeout_sec: int,
    rerun: bool,
) -> None:
    stitching_csv = output_dir / "stitching_data.csv"
    panorama_png = output_dir / "final_panorama.png"
    run_log = output_dir / "run.log"
    if stitching_csv.exists() and panorama_png.exists() and not rerun:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli_path),
        str(group_dir),
        str(image_count),
        "--preset",
        "gui",
        "--input-mode",
        "scan",
        "--run-mode",
        "registration",
        "--overlap",
        str(overlap),
        "--direction",
        direction,
        "--endpoint-probe-fast",
        "--no-process-vis",
        "--out",
        str(panorama_png),
        "--csv",
        str(stitching_csv),
    ]
    completed = subprocess.run(
        cmd,
        cwd=project_root(),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
    )
    run_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"proposed CLI failed for group {group_dir.name} with exit code {completed.returncode}. "
            f"See {run_log}"
        )


def worst_step_rmse_from_pairs(pair_transforms: Sequence[object], pixel_size_um: float) -> tuple[float, float]:
    values_px = [
        float(getattr(item, "overlap_profile_rmse_px", math.nan))
        for item in pair_transforms
        if math.isfinite(float(getattr(item, "overlap_profile_rmse_px", math.nan)))
    ]
    if not values_px:
        return math.nan, math.nan
    worst_px = float(max(values_px))
    return worst_px, worst_px * pixel_size_um


def result_from_eval(
    group_name: str,
    method_code: str,
    method_label: str,
    eval_result: object,
    worst_step_rmse_px: float,
    pixel_size_um: float,
) -> MethodResult:
    return MethodResult(
        group_name=group_name,
        method_code=method_code,
        method_label=method_label,
        normal_rmse_um=float(eval_result.normal_rmse_um),
        profile_rms_um=float(eval_result.profile_rms_um),
        profile_p95_abs_um=float(eval_result.profile_p95_abs_um),
        mean_normal_error_um=float(eval_result.mean_normal_error_um),
        used_count=int(eval_result.used_count),
        outlier_ratio=float(eval_result.outlier_ratio),
        worst_step_rmse_px=float(worst_step_rmse_px),
        worst_step_rmse_um=float(worst_step_rmse_px * pixel_size_um) if math.isfinite(worst_step_rmse_px) else math.nan,
        curve_s_mm=np.asarray(eval_result.s_used_mm, dtype=float),
        curve_profile_error_um=np.asarray(eval_result.profile_error_used_um, dtype=float),
    )


def write_pair_csv(path: Path, pair_transforms: Sequence[object]) -> None:
    rows = [
        {
            "step": int(getattr(item, "step", 0)),
            "image_a": int(getattr(item, "image_a", 0)),
            "image_b": int(getattr(item, "image_b", 0)),
            "dx_px": float(getattr(item, "dx", math.nan)),
            "dy_px": float(getattr(item, "dy", math.nan)),
            "angle_deg": float(getattr(item, "angle_deg", 0.0)),
            "score": float(getattr(item, "score", math.nan)),
            "overlap_profile_rmse_px": float(getattr(item, "overlap_profile_rmse_px", math.nan)),
            "method": str(getattr(item, "method", "")),
        }
        for item in pair_transforms
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def evaluate_group(
    group_dir: Path,
    proposed_dir: Path,
    compare_dir: Path,
    base_mod,
    recent_mod,
) -> list[MethodResult]:
    image_paths = collect_image_paths(group_dir)
    images = [base_mod.load_grayscale(path) for path in image_paths]
    contours = [base_mod.extract_top_contour(image) for image in images]
    pixel_size_um = float(base_mod.PIXEL_SIZE_MM) * 1000.0

    proposed_pairs = base_mod.load_stage5_pair_transforms(proposed_dir)
    proposed_samples = base_mod.collect_transformed_samples(contours, proposed_pairs)
    proposed_eval = base_mod.evaluate_method("proposed", proposed_samples)
    proposed_worst_px, _ = worst_step_rmse_from_pairs(proposed_pairs, pixel_size_um)

    recent_a_pairs = recent_mod.pairwise_recent_profile_grinding_baseline(base_mod, images, contours)
    recent_a_samples = base_mod.collect_transformed_samples(contours, recent_a_pairs)
    recent_a_eval = base_mod.evaluate_method("recent_a", recent_a_samples)
    recent_a_worst_px, _ = worst_step_rmse_from_pairs(recent_a_pairs, pixel_size_um)

    recent_b_pairs = recent_mod.pairwise_recent_telecentric_baseline(base_mod, images, contours)
    recent_b_samples = base_mod.collect_transformed_samples(contours, recent_b_pairs)
    recent_b_eval = base_mod.evaluate_method("recent_b", recent_b_samples)
    recent_b_worst_px, _ = worst_step_rmse_from_pairs(recent_b_pairs, pixel_size_um)

    compare_dir.mkdir(parents=True, exist_ok=True)
    write_pair_csv(compare_dir / "proposed_pair_transforms.csv", proposed_pairs)
    write_pair_csv(compare_dir / "recent_method_a_pair_transforms.csv", recent_a_pairs)
    write_pair_csv(compare_dir / "recent_method_b_pair_transforms.csv", recent_b_pairs)

    return [
        result_from_eval(group_dir.name, "proposed", "所提方法", proposed_eval, proposed_worst_px, pixel_size_um),
        result_from_eval(
            group_dir.name,
            "recent_profile_grinding_2025",
            "对比方法A（2025平滑曲线拼接）",
            recent_a_eval,
            recent_a_worst_px,
            pixel_size_um,
        ),
        result_from_eval(
            group_dir.name,
            "recent_telecentric_scan_stitch_2026",
            "对比方法B（2026远心扫描拼接）",
            recent_b_eval,
            recent_b_worst_px,
            pixel_size_um,
        ),
    ]


def build_group_metrics_dataframe(results: Sequence[MethodResult]) -> pd.DataFrame:
    rows = [
        {
            "group_name": item.group_name,
            "method_code": item.method_code,
            "method_label": item.method_label,
            "normal_rmse_um": item.normal_rmse_um,
            "profile_rms_um": item.profile_rms_um,
            "profile_p95_abs_um": item.profile_p95_abs_um,
            "mean_normal_error_um": item.mean_normal_error_um,
            "used_count": item.used_count,
            "outlier_ratio": item.outlier_ratio,
            "worst_step_rmse_px": item.worst_step_rmse_px,
            "worst_step_rmse_um": item.worst_step_rmse_um,
        }
        for item in results
    ]
    df = pd.DataFrame(rows)
    method_order = {code: index for index, (code, _) in enumerate(METHODS)}
    df["group_order"] = df["group_name"].map(lambda value: int(value))
    df["method_order"] = df["method_code"].map(method_order)
    df.sort_values(["group_order", "method_order"], inplace=True)
    df.drop(columns=["group_order", "method_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_summary_dataframe(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_code, method_label in METHODS:
        subset = group_df[group_df["method_code"] == method_code].copy()
        if subset.empty:
            continue
        abs_values = subset["normal_rmse_um"].to_numpy(dtype=float)
        form_values = subset["profile_rms_um"].to_numpy(dtype=float)
        worst_values = subset["worst_step_rmse_um"].to_numpy(dtype=float)
        rows.append(
            {
                "method_code": method_code,
                "method_label": method_label,
                "group_count": int(len(subset)),
                "abs_rmse_mean_um": float(np.mean(abs_values)),
                "abs_rmse_std_um": float(np.std(abs_values, ddof=1)) if len(abs_values) > 1 else 0.0,
                "abs_rmse_min_um": float(np.min(abs_values)),
                "abs_rmse_max_um": float(np.max(abs_values)),
                "profile_rms_mean_um": float(np.mean(form_values)),
                "profile_rms_std_um": float(np.std(form_values, ddof=1)) if len(form_values) > 1 else 0.0,
                "profile_rms_min_um": float(np.min(form_values)),
                "profile_rms_max_um": float(np.max(form_values)),
                "worst_step_rmse_mean_um": float(np.mean(worst_values)),
                "worst_step_rmse_std_um": float(np.std(worst_values, ddof=1)) if len(worst_values) > 1 else 0.0,
                "worst_step_pass_count": int(np.sum(subset["worst_step_rmse_px"].to_numpy(dtype=float) <= 0.25)),
                "used_count_mean": float(np.mean(subset["used_count"].to_numpy(dtype=float))),
                "outlier_ratio_mean": float(np.mean(subset["outlier_ratio"].to_numpy(dtype=float))),
            }
        )
    summary_df = pd.DataFrame(rows)
    method_order = {code: index for index, (code, _) in enumerate(METHODS)}
    summary_df["method_order"] = summary_df["method_code"].map(method_order)
    summary_df.sort_values("method_order", inplace=True)
    summary_df.drop(columns=["method_order"], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df


def build_curve_stats_dataframe(results: Sequence[MethodResult]) -> pd.DataFrame:
    grid = np.arange(0.0, 110.001, 0.5, dtype=float)
    rows: list[dict[str, float | str | int]] = []
    for method_code, method_label in METHODS:
        curves = []
        for item in results:
            if item.method_code != method_code:
                continue
            if item.curve_s_mm.size < 20:
                continue
            order = np.argsort(item.curve_s_mm)
            s = item.curve_s_mm[order]
            y = item.curve_profile_error_um[order]
            frame = pd.DataFrame({"s": s, "y": y}).dropna().groupby("s", as_index=False).mean()
            if len(frame) < 20:
                continue
            interp = np.interp(grid, frame["s"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float), left=np.nan, right=np.nan)
            valid = (grid >= frame["s"].min()) & (grid <= frame["s"].max())
            interp[~valid] = np.nan
            curves.append(interp)
        if not curves:
            continue
        stack = np.vstack(curves)
        count = np.sum(np.isfinite(stack), axis=0)
        mean = np.full(len(grid), np.nan, dtype=float)
        std = np.full(len(grid), np.nan, dtype=float)
        for idx, count_value in enumerate(count):
            if count_value <= 0:
                continue
            column = stack[:, idx]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                continue
            mean[idx] = float(np.mean(finite))
            std[idx] = float(np.std(finite, ddof=1 if finite.size > 1 else 0))
        for s_value, mean_value, std_value, count_value in zip(grid, mean, std, count):
            rows.append(
                {
                    "method_code": method_code,
                    "method_label": method_label,
                    "s_mm": float(s_value),
                    "profile_error_mean_um": float(mean_value) if math.isfinite(float(mean_value)) else math.nan,
                    "profile_error_std_um": float(std_value) if math.isfinite(float(std_value)) else math.nan,
                    "support_count": int(count_value),
                }
            )
    return pd.DataFrame(rows)


def configure_matplotlib(svg_fonttype: str) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
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
            "axes.titlepad": 4.0,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.0,
            "axes.unicode_minus": False,
            "svg.fonttype": svg_fonttype,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=mpl.rcParams["font.size"] + 1.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_metric_scatter_panel(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    panel: str,
    title: str,
    threshold: float | None = None,
) -> None:
    panel_label(ax, panel)
    x_positions = np.arange(len(METHODS), dtype=float)
    rng = np.random.default_rng(20260531)
    all_values = pd.to_numeric(group_df[metric_col], errors="coerce").to_numpy(dtype=float)
    finite_all = all_values[np.isfinite(all_values)]
    y_min = None
    y_max = None
    clipped_count = 0
    if finite_all.size > 0:
        upper_quantile = 0.80 if finite_all.size >= 12 else 0.88
        robust_low = float(np.quantile(finite_all, 0.02))
        robust_high = float(np.quantile(finite_all, upper_quantile))
        lower_bound = min(0.0, robust_low)
        span = max(1.0, robust_high - lower_bound)
        y_min = lower_bound - 0.08 * span
        y_max = robust_high + 0.18 * span
        if threshold is not None:
            y_min = min(y_min, -0.12 * threshold)
            y_max = max(y_max, 1.6 * threshold)
        clipped_count = int(np.sum(finite_all > y_max))

    for idx, (method_code, method_label) in enumerate(METHODS):
        subset = group_df[group_df["method_code"] == method_code]
        values = subset[metric_col].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        x_scatter = np.full(len(values), x_positions[idx]) + jitter
        finite_mask = np.isfinite(values)
        if y_min is None or y_max is None:
            shown_values = values
            shown_mask = finite_mask
            high_mask = np.zeros(len(values), dtype=bool)
            low_mask = np.zeros(len(values), dtype=bool)
        else:
            shown_values = values.copy()
            high_mask = finite_mask & (shown_values > y_max)
            low_mask = finite_mask & (shown_values < y_min)
            shown_values[high_mask] = y_max
            shown_values[low_mask] = y_min
            shown_mask = finite_mask & ~high_mask & ~low_mask

        if np.any(shown_mask):
            ax.scatter(
                x_scatter[shown_mask],
                shown_values[shown_mask],
                s=18,
                color=METHOD_COLORS[method_code],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        if np.any(high_mask):
            ax.scatter(
                x_scatter[high_mask],
                shown_values[high_mask],
                s=24,
                marker="^",
                color=METHOD_COLORS[method_code],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        if np.any(low_mask):
            ax.scatter(
                x_scatter[low_mask],
                shown_values[low_mask],
                s=24,
                marker="v",
                color=METHOD_COLORS[method_code],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean_y = mean
        text_y = mean
        text_va = "bottom"
        if y_min is not None and y_max is not None:
            mean_y = min(max(mean, y_min), y_max)
            if mean >= y_max:
                text_y = y_max - 0.10 * max(y_max - y_min, 1.0)
                text_va = "top"
            elif mean <= y_min:
                text_y = y_min + 0.06 * max(y_max - y_min, 1.0)
                text_va = "bottom"
            else:
                text_y = mean

        if y_min is None or y_max is None or (y_min <= mean - std and mean + std <= y_max):
            ax.errorbar(
                [x_positions[idx]],
                [mean],
                yerr=[[std], [std]],
                fmt="s",
                color="#222222",
                markersize=4.4,
                capsize=3.0,
                linewidth=0.9,
                zorder=4,
            )
        else:
            ax.scatter(
                [x_positions[idx]],
                [mean_y],
                s=26,
                marker="s",
                color="#222222",
                zorder=4,
            )
        ax.text(
            x_positions[idx],
            text_y,
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va=text_va,
            fontsize=7.0,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.82), edgecolor="none"),
        )
    if threshold is not None:
        ax.axhline(threshold, color="#7A8793", linestyle="--", linewidth=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in METHODS], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
        if clipped_count > 0:
            ax.text(
                0.98,
                0.96,
                f"为便于比较，裁剪显示 {clipped_count} 个极端离群点",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
                color="#5B6570",
            )
    ax.grid(True, axis="y")


def draw_repeatability_curve_panel(ax: plt.Axes, curve_df: pd.DataFrame, panel: str) -> None:
    panel_label(ax, panel)
    has_curve = False
    for method_code, method_label in METHODS:
        subset = curve_df[(curve_df["method_code"] == method_code) & (curve_df["support_count"] >= 5)].copy()
        if subset.empty:
            continue
        x = subset["s_mm"].to_numpy(dtype=float)
        std = subset["profile_error_std_um"].to_numpy(dtype=float)
        ax.plot(x, std, color=METHOD_COLORS[method_code], linewidth=1.05, label=method_label)
        has_curve = True
    if not has_curve:
        ax.text(0.5, 0.5, "至少完成5组后显示局部重复性曲线", transform=ax.transAxes, ha="center", va="center", color="#4D4D4D")
        ax.set_axis_off()
        return
    ax.set_xlabel("轴向坐标 s (mm)")
    ax.set_ylabel("局部重复性 σ (μm)")
    ax.set_title("沿母线的局部重复性分布")
    ax.grid(True)
    ax.legend(loc="upper right")


def draw_tradeoff_panel(ax: plt.Axes, summary_df: pd.DataFrame, panel: str) -> None:
    panel_label(ax, panel)
    for _, row in summary_df.iterrows():
        method_code = str(row["method_code"])
        ax.scatter(
            float(row["profile_rms_mean_um"]),
            float(row["abs_rmse_mean_um"]),
            s=80,
            color=METHOD_COLORS[method_code],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.errorbar(
            [float(row["profile_rms_mean_um"])],
            [float(row["abs_rmse_mean_um"])],
            xerr=[[float(row["profile_rms_std_um"])], [float(row["profile_rms_std_um"])]],
            yerr=[[float(row["abs_rmse_std_um"])], [float(row["abs_rmse_std_um"])]],
            fmt="none",
            color=METHOD_COLORS[method_code],
            linewidth=0.85,
            capsize=3.0,
            zorder=2,
        )
        ax.text(
            float(row["profile_rms_mean_um"]),
            float(row["abs_rmse_mean_um"]),
            str(row["method_label"]),
            fontsize=7.2,
            ha="left",
            va="bottom",
            color="#333333",
        )
    ax.set_xlabel("Profile RMS 均值 (μm)")
    ax.set_ylabel("Absolute RMSE 均值 (μm)")
    ax.set_title("重复性均值-波动折中")
    ax.grid(True)
    ax.text(0.02, 0.04, "左下角更优", transform=ax.transAxes, ha="left", va="bottom", color="#4D4D4D")


def save_repeatability_figure(
    output_root: Path,
    group_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    svg_fonttype: str,
) -> None:
    configure_matplotlib(svg_fonttype)
    fig = plt.figure(figsize=(178.0 / 25.4, 178.0 / 25.4 * 0.88))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.28, "hspace": 0.32})
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.10)

    draw_metric_scatter_panel(
        axes["A"],
        group_df,
        "normal_rmse_um",
        "Absolute RMSE (μm)",
        "(a)",
        "各方法绝对误差重复性",
    )
    draw_metric_scatter_panel(
        axes["B"],
        group_df,
        "profile_rms_um",
        "Profile RMS (μm)",
        "(b)",
        "各方法型面误差重复性",
    )
    draw_metric_scatter_panel(
        axes["C"],
        group_df,
        "worst_step_rmse_um",
        "Worst-step RMSE (μm)",
        "(c)",
        "最差单步拼接质量重复性",
        threshold=0.25 * 10.057,
    )
    draw_repeatability_curve_panel(axes["D"], curve_df, "(d)")

    fig.savefig(output_root / FIGURE_PNG_NAME, dpi=300)
    fig.savefig(output_root / FIGURE_PDF_NAME)
    fig.savefig(output_root / FIGURE_SVG_NAME)
    plt.close(fig)


def write_report(output_path: Path, summary_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    def df_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
        table = df.copy()
        for col in table.columns:
            if pd.api.types.is_float_dtype(table[col]):
                table[col] = table[col].map(lambda value: f"{value:.{digits}f}" if math.isfinite(float(value)) else "--")
        headers = list(table.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in table.iterrows():
            values = [str(row[col]) for col in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report = [
        "# 重复性验证报告",
        "",
        "## 方法说明",
        "",
        "- 所提方法：调用 `pinjie_cli` 的当前拼接链路，在 `registration` 模式下输出步间位姿，再用统一的外部设计型线评定器计算最终误差。",
        "- 对比方法A：2025 年平滑曲线拼接思路，采用高曲率锚点候选 + 重叠区残差精化。",
        "- 对比方法B：2026 年远心扫描拼接思路，采用灰度粗配准驱动的 scan-and-stitch 拼接。",
        "- 三种方法均基于同一批原始图像、同一条外部设计型线评定器，保证重复性比较口径一致。",
        "",
        "## 各方法重复性汇总",
        "",
        df_to_markdown(summary_df),
        "",
        "## 10组逐组结果",
        "",
        df_to_markdown(group_df),
        "",
    ]
    output_path.write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_mod = load_module(project_root() / "tools" / "workpiece_literature_comparison.py", "repeatability_base_module")
    recent_mod = load_module(project_root() / "literature_fire_tube_compare" / "compare_fire_tube_literature_baselines.py", "repeatability_recent_module")

    group_dirs = collect_group_dirs(args.input_root, args.groups)
    proposed_root = args.output_root / "proposed_runs"
    compare_root = args.output_root / "per_group_comparison"
    args.output_root.mkdir(parents=True, exist_ok=True)
    proposed_root.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)

    all_results: list[MethodResult] = []
    for index, group_dir in enumerate(group_dirs, start=1):
        image_paths = collect_image_paths(group_dir)
        proposed_dir = proposed_root / f"group_{int(group_dir.name):02d}"
        compare_dir = compare_root / f"group_{int(group_dir.name):02d}"
        print(f"[{index}/{len(group_dirs)}] Running proposed method for group {group_dir.name} ({len(image_paths)} images)...", flush=True)
        run_proposed_cli(
            cli_path=args.cli,
            group_dir=group_dir,
            output_dir=proposed_dir,
            image_count=len(image_paths),
            overlap=args.overlap,
            direction=args.direction,
            timeout_sec=args.timeout_sec_per_group,
            rerun=args.rerun_proposed,
        )
        print(f"[{index}/{len(group_dirs)}] Evaluating all methods for group {group_dir.name}...", flush=True)
        group_results = evaluate_group(group_dir, proposed_dir, compare_dir, base_mod, recent_mod)
        all_results.extend(group_results)
        group_df_partial = build_group_metrics_dataframe(all_results)
        group_df_partial.to_csv(args.output_root / GROUP_METRICS_CSV_NAME, index=False, encoding="utf-8-sig")

    group_df = build_group_metrics_dataframe(all_results)
    summary_df = build_summary_dataframe(group_df)
    curve_df = build_curve_stats_dataframe(all_results)

    group_df.to_csv(args.output_root / GROUP_METRICS_CSV_NAME, index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.output_root / SUMMARY_CSV_NAME, index=False, encoding="utf-8-sig")
    curve_df.to_csv(args.output_root / CURVE_STATS_CSV_NAME, index=False, encoding="utf-8-sig")
    save_repeatability_figure(args.output_root, group_df, summary_df, curve_df, args.svg_fonttype)
    write_report(args.output_root / REPORT_MD_NAME, summary_df, group_df)

    print(f"[OK] output_root={args.output_root}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
