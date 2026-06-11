from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PNG_NAME = "repeat10_workpiece_summary_figure.png"
DEFAULT_PDF_NAME = "repeat10_workpiece_summary_figure.pdf"
DEFAULT_SVG_NAME = "repeat10_workpiece_summary_figure.svg"
DEFAULT_METRICS_CSV_NAME = "repeat10_workpiece_metrics.csv"
DEFAULT_STATS_CSV_NAME = "repeat10_workpiece_summary_stats.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an aggregate summary figure for repeated workpiece runs."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Root directory that contains per-group result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to the result root.",
    )
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=178.0)
    parser.add_argument("--font-size-pt", type=float, default=8.0)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def configure_matplotlib(font_size_pt: float, svg_fonttype: str) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
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
            "axes.titlepad": 4.0,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.0,
            "axes.unicode_minus": False,
            "svg.fonttype": svg_fonttype,
        }
    )


def has_required_outputs(result_dir: Path) -> bool:
    return (
        (result_dir / "design_error_summary.csv").exists()
        and (result_dir / "quality_review.csv").exists()
        and (result_dir / "cjsi_workpiece_step_diagnostics.csv").exists()
    )


def natural_group_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def threshold_from_quality(quality_df: pd.DataFrame, check_name: str, fallback: float) -> float:
    matches = quality_df.loc[quality_df["check"] == check_name, "threshold"]
    if matches.empty:
        return fallback
    value = pd.to_numeric(matches, errors="coerce").dropna()
    if value.empty:
        return fallback
    return float(value.iloc[0])


def first_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def build_metrics(result_root: Path) -> tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, float | str]] = []
    thresholds: Dict[str, float] = {}
    result_dirs = [path for path in result_root.iterdir() if path.is_dir() and has_required_outputs(path)]
    for result_dir in sorted(result_dirs, key=natural_group_key):
        summary_df = pd.read_csv(result_dir / "design_error_summary.csv")
        quality_df = pd.read_csv(result_dir / "quality_review.csv")
        step_df = pd.read_csv(result_dir / "cjsi_workpiece_step_diagnostics.csv")
        stitching_df = pd.read_csv(result_dir / "stitching_data.csv")

        if not thresholds:
            thresholds = {
                "absolute_filtered_rmse_um": threshold_from_quality(
                    quality_df, "absolute_filtered_rmse_um", 80.0
                ),
                "profile_rms_um": threshold_from_quality(quality_df, "form_rmse_um", 30.0),
                "worst_step_normal_rmse_px": threshold_from_quality(
                    quality_df, "worst_step_normal_rmse_px", 0.25
                ),
                "outlier_ratio": threshold_from_quality(quality_df, "outlier_ratio", 0.03),
            }

        worst_step_idx = int(pd.to_numeric(step_df["step"], errors="coerce").iloc[
            pd.to_numeric(step_df["normal_rmse_px"], errors="coerce").idxmax()
        ])
        worst_step_rmse = float(pd.to_numeric(step_df["normal_rmse_px"], errors="coerce").max())
        step1_row = stitching_df.iloc[0]

        rows.append(
            {
                "group": result_dir.name,
                "group_index": natural_group_key(result_dir)[0],
                "absolute_filtered_rmse_um": first_numeric(summary_df, "absolute_filtered_rmse_um"),
                "profile_rms_um": first_numeric(summary_df, "profile_rms_um"),
                "outlier_ratio": first_numeric(summary_df, "outlier_ratio"),
                "worst_step_normal_rmse_px": worst_step_rmse,
                "worst_step_index": worst_step_idx,
                "step1_normal_rmse_px": float(step1_row["NormalRMSEInlier(px)"]),
                "step1_tangent_rmse_px": float(step1_row["TangentRMSEInlier(px)"]),
                "step1_tangent_corr": float(step1_row["TangentCorrInlier"]),
                "step1_mode": str(step1_row["SelectionMode"]),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values(["group_index", "group"]).reset_index(drop=True)
    return metrics_df, thresholds


def build_summary_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    stat_rows: List[Dict[str, float | str]] = []
    metric_columns = [
        "absolute_filtered_rmse_um",
        "profile_rms_um",
        "worst_step_normal_rmse_px",
        "outlier_ratio",
        "step1_normal_rmse_px",
        "step1_tangent_rmse_px",
        "step1_tangent_corr",
    ]
    for column in metric_columns:
        series = pd.to_numeric(metrics_df[column], errors="coerce")
        stat_rows.append(
            {
                "metric": column,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        )
    return pd.DataFrame(stat_rows)


def axis_limits(values: Iterable[float], threshold: float) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, max(1.0, threshold * 1.1)
    y_min = float(arr.min())
    y_max = float(arr.max())
    upper = max(y_max, threshold) * 1.10
    lower_candidates = [y_min * 0.90]
    if y_min >= 0.0:
        lower_candidates.append(0.0)
    lower = min(lower_candidates) if y_min < 0.0 else max(0.0, min(lower_candidates))
    if threshold < upper and threshold > lower:
        return lower, upper
    return lower, upper


def draw_metric_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    title: str,
    ylabel: str,
    panel: str,
) -> None:
    mean_value = float(np.nanmean(y))
    ax.plot(
        x,
        y,
        color="#4C555C",
        linewidth=0.95,
        marker="o",
        markersize=3.4,
        markerfacecolor="#4DB6AC",
        markeredgecolor="#3E8F88",
        markeredgewidth=0.45,
        zorder=3,
    )
    ax.axhline(
        mean_value,
        color="#7C848B",
        linewidth=0.9,
        linestyle="-.",
        label=f"mean={mean_value:.3f}",
        zorder=1,
    )
    ax.axhline(
        threshold,
        color="#E76F51",
        linewidth=0.9,
        linestyle="--",
        label=f"threshold={threshold:.3f}",
        zorder=1,
    )
    ax.set_title(f"{panel} {title}", loc="left", fontsize=mpl.rcParams["font.size"] + 0.1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    y_lower, y_upper = axis_limits(y, threshold)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(x)
    ax.grid(axis="y")
    ax.legend(loc="best")


def export_figure(fig: plt.Figure, output_dir: Path, args: argparse.Namespace) -> None:
    svg_path = output_dir / DEFAULT_SVG_NAME
    pdf_path = output_dir / DEFAULT_PDF_NAME
    png_path = output_dir / DEFAULT_PNG_NAME

    if not args.pdf_only:
        fig.savefig(svg_path, bbox_inches="tight")
    if not args.svg_only:
        fig.savefig(png_path, bbox_inches="tight")
    if not args.svg_only:
        fig.savefig(pdf_path, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or result_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, thresholds = build_metrics(result_root)
    if metrics_df.empty:
        raise FileNotFoundError(f"no valid per-group result directories found under: {result_root}")

    stats_df = build_summary_stats(metrics_df)
    metrics_df.to_csv(output_dir / DEFAULT_METRICS_CSV_NAME, index=False, encoding="utf-8-sig")
    stats_df.to_csv(output_dir / DEFAULT_STATS_CSV_NAME, index=False, encoding="utf-8-sig")

    x = metrics_df["group_index"].to_numpy(dtype=float)
    fig_width = args.figure_width_mm / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_width * 0.82), constrained_layout=True)
    axes = axes.ravel()

    draw_metric_panel(
        axes[0],
        x,
        metrics_df["absolute_filtered_rmse_um"].to_numpy(dtype=float),
        thresholds["absolute_filtered_rmse_um"],
        f"{len(metrics_df)}组完整流程绝对误差统计",
        "Absolute RMSE (um)",
        "(a)",
    )
    draw_metric_panel(
        axes[1],
        x,
        metrics_df["profile_rms_um"].to_numpy(dtype=float),
        thresholds["profile_rms_um"],
        f"{len(metrics_df)}组完整流程型面误差统计",
        "Profile RMS (um)",
        "(b)",
    )
    draw_metric_panel(
        axes[2],
        x,
        metrics_df["worst_step_normal_rmse_px"].to_numpy(dtype=float),
        thresholds["worst_step_normal_rmse_px"],
        f"{len(metrics_df)}组最差单步拼接质量统计",
        "Worst-step RMSE (px)",
        "(c)",
    )
    draw_metric_panel(
        axes[3],
        x,
        metrics_df["outlier_ratio"].to_numpy(dtype=float),
        thresholds["outlier_ratio"],
        f"{len(metrics_df)}组异常点比例统计",
        "Outlier ratio",
        "(d)",
    )

    for ax in axes[2:]:
        ax.set_xlabel("组号")

    export_figure(fig, output_dir, args)
    plt.close(fig)
    print(f"[OK] Repeat summary exported under: {output_dir}")


if __name__ == "__main__":
    main()
