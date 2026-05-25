from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SVG_NAME = "journal_design_profile_figure.svg"
DEFAULT_PNG_NAME = "journal_design_profile_figure.png"


@dataclass
class FigureData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export publication-ready SVG figures for design profile comparison results."
    )
    parser.add_argument("--result-dir", type=Path, help="One result directory containing design CSV outputs.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("result/workpiece"),
        help="Root directory used to auto-discover the latest result when --result-dir is omitted.",
    )
    parser.add_argument("--output-svg", type=Path, help="Explicit SVG output path.")
    parser.add_argument("--output-png", type=Path, help="Explicit PNG output path.")
    parser.add_argument("--svg-only", action="store_true", help="Export SVG only.")
    parser.add_argument("--figure-width-mm", type=float, default=180.0, help="Figure width in millimeters.")
    parser.add_argument("--font-size-pt", type=float, default=8.5, help="Base font size in points.")
    parser.add_argument(
        "--svg-fonttype",
        choices=("path", "none"),
        default="path",
        help="Whether to convert text to vector paths inside the SVG.",
    )
    return parser.parse_args()


def find_latest_result_dir(result_root: Path) -> Path:
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")

    candidates = [path for path in result_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no result directory found under: {result_root}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    if args.result_dir is not None:
        return args.result_dir
    return find_latest_result_dir(args.result_root)


def load_figure_data(result_dir: Path) -> FigureData:
    profile_csv = result_dir / "design_error_profile.csv"
    summary_csv = result_dir / "design_error_summary.csv"
    if not profile_csv.exists():
        raise FileNotFoundError(f"missing file: {profile_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing file: {summary_csv}")

    profile_df = pd.read_csv(profile_csv)
    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        raise ValueError(f"summary CSV is empty: {summary_csv}")

    return FigureData(result_dir=result_dir, profile_df=profile_df, summary_row=summary_df.iloc[0])


def configure_matplotlib(font_size_pt: float, svg_fonttype: str) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": font_size_pt,
            "axes.titlesize": font_size_pt + 0.8,
            "axes.labelsize": font_size_pt,
            "xtick.labelsize": font_size_pt - 0.4,
            "ytick.labelsize": font_size_pt - 0.4,
            "legend.fontsize": font_size_pt - 0.3,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Noto Serif CJK SC",
                "SimSun",
                "DejaVu Serif",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "svg.fonttype": svg_fonttype,
        }
    )


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=mpl.rcParams["axes.titlesize"] + 1.0,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finite_series(series: pd.Series) -> np.ndarray:
    array = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def format_metric(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def build_used_dataframe(profile_df: pd.DataFrame) -> pd.DataFrame:
    used_df = profile_df.copy()
    used_df["is_used"] = pd.to_numeric(used_df["is_used"], errors="coerce").fillna(0).astype(int)
    used_df = used_df[used_df["is_used"] == 1].copy()
    numeric_columns = [
        "s_aligned_mm",
        "r_aligned_mm",
        "r_design_mm",
        "normal_error_um",
        "profile_error_um",
    ]
    for column in numeric_columns:
        if column in used_df.columns:
            used_df[column] = pd.to_numeric(used_df[column], errors="coerce")
    return used_df


def build_summary_lines(summary: pd.Series) -> list[str]:
    return [
        f"Used count: {int(summary['used_count'])}",
        f"Reverse Z: {int(summary['design_reverse_z'])}",
        f"Left anchor: {int(summary['use_left_endpoint_anchor'])}",
        f"Profile-form mode: {int(summary['design_evaluate_profile_form'])}",
        "",
        f"Mean normal bias: {format_metric(summary['mean_normal_error_um'])} um",
        f"Normal RMSE: {format_metric(summary['normal_rmse_um'])} um",
        f"Normal P95: {format_metric(summary['normal_p95_abs_um'])} um",
        f"Normal PV: {format_metric(summary['normal_pv_um'])} um",
        "",
        f"Profile RMS: {format_metric(summary['profile_rms_um'])} um",
        f"Profile MAE: {format_metric(summary['profile_mae_um'])} um",
        f"Profile P95: {format_metric(summary['profile_p95_abs_um'])} um",
        f"Profile PV: {format_metric(summary['profile_pv_um'])} um",
    ]


def export_publication_figure(data: FigureData, output_svg: Path, output_png: Path | None, svg_only: bool, width_mm: float) -> None:
    summary = data.summary_row
    used_df = build_used_dataframe(data.profile_df)
    if len(used_df) < 50:
        raise ValueError("too few used points for figure export")

    s_all = pd.to_numeric(data.profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(data.profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(data.profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1

    width_in = mm_to_inches(width_mm)
    height_in = width_in * 0.78
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
    mosaic = """
    AB
    CD
    """
    axes = fig.subplot_mosaic(mosaic)

    measured_color = "#2457C5"
    design_color = "#198754"
    normal_color = "#D94841"
    profile_color = "#2457C5"
    excluded_color = "#C9CDD3"

    ax_a = axes["A"]
    panel_label(ax_a, "A")
    ax_a.scatter(
        s_all[~used_mask],
        r_all[~used_mask],
        s=6,
        color=excluded_color,
        linewidths=0,
        alpha=0.55,
        label="Excluded points",
    )
    ax_a.plot(
        used_df["s_aligned_mm"],
        used_df["r_aligned_mm"],
        color=measured_color,
        linewidth=1.2,
        label="Measured contour",
    )
    ax_a.plot(
        used_df["s_aligned_mm"],
        used_df["r_design_mm"],
        color=design_color,
        linewidth=1.0,
        linestyle="--",
        label="Design contour",
    )
    ax_a.set_title("Measured contour vs. design contour")
    ax_a.set_xlabel("Axial coordinate s (mm)")
    ax_a.set_ylabel("Radius r (mm)")
    ax_a.grid(True)
    ax_a.legend(loc="best", frameon=False)

    ax_b = axes["B"]
    panel_label(ax_b, "B")
    ax_b.plot(
        used_df["s_aligned_mm"],
        used_df["normal_error_um"],
        color=normal_color,
        linewidth=1.0,
        alpha=0.9,
        label="Absolute normal error",
    )
    ax_b.plot(
        used_df["s_aligned_mm"],
        used_df["profile_error_um"],
        color=profile_color,
        linewidth=1.0,
        label="Profile-form error",
    )
    ax_b.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax_b.set_title("Normal error and profile-form error")
    ax_b.set_xlabel("Axial coordinate s (mm)")
    ax_b.set_ylabel("Error (um)")
    ax_b.grid(True)
    ax_b.legend(loc="best", frameon=False)

    ax_c = axes["C"]
    panel_label(ax_c, "C")
    profile_errors = finite_series(used_df["profile_error_um"])
    bins = min(45, max(20, int(math.sqrt(len(profile_errors)))))
    ax_c.hist(
        profile_errors,
        bins=bins,
        color=profile_color,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_c.axvline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax_c.set_title("Distribution of profile-form error")
    ax_c.set_xlabel("Profile-form error (um)")
    ax_c.set_ylabel("Count")
    ax_c.grid(True, axis="y")

    ax_d = axes["D"]
    panel_label(ax_d, "D")
    ax_d.axis("off")
    summary_lines = build_summary_lines(summary)
    ax_d.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        transform=ax_d.transAxes,
        family="monospace",
        fontsize=mpl.rcParams["font.size"] - 0.2,
    )
    ax_d.set_title("Summary metrics", loc="left")

    fig.suptitle("Design Profile Comparison and Profile-Form Evaluation", y=1.01, fontsize=mpl.rcParams["font.size"] + 1.5)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    if not svg_only and output_png is not None:
        fig.savefig(output_png, format="png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = resolve_result_dir(args)
    data = load_figure_data(result_dir)
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    output_svg = args.output_svg or (result_dir / DEFAULT_SVG_NAME)
    output_png = None if args.svg_only else (args.output_png or (result_dir / DEFAULT_PNG_NAME))

    export_publication_figure(
        data=data,
        output_svg=output_svg,
        output_png=output_png,
        svg_only=args.svg_only,
        width_mm=args.figure_width_mm,
    )

    print(f"[OK] SVG figure saved to: {output_svg}")
    if output_png is not None:
        print(f"[OK] PNG preview saved to: {output_png}")


if __name__ == "__main__":
    main()
