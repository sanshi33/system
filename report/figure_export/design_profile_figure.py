from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from publication_figure import build_display_dataframe


DEFAULT_SVG_NAME = "journal_design_profile_figure.svg"
DEFAULT_PNG_NAME = "journal_design_profile_figure.png"
FIGURE_LANGUAGES = ("zh", "en")

COLORS = {
    "measured": "#0B5FA5",
    "design": "#1B9E77",
    "profile_error": "#E66101",
    "normal_error": "#7570B3",
    "excluded": "#BFC7D2",
    "reference": "#4D4D4D",
    "band": "#E8EEF6",
    "segment": "#C7CDD6",
}


@dataclass
class FigureData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export design-profile comparison figure.")
    parser.add_argument("--result-dir", type=Path, help="Result directory with design CSV outputs.")
    parser.add_argument("--result-root", type=Path, default=Path("result/workpiece"))
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=180.0)
    parser.add_argument("--font-size-pt", type=float, default=8.5)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def has_required_outputs(result_dir: Path) -> bool:
    return (
        (result_dir / "design_error_profile.csv").exists()
        and (result_dir / "design_error_summary.csv").exists()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")
    candidates = [path for path in result_root.iterdir() if path.is_dir() and has_required_outputs(path)]
    if not candidates:
        raise FileNotFoundError(f"no valid result directory found under: {result_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    return args.result_dir if args.result_dir is not None else find_latest_result_dir(args.result_root)


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
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 0.4,
        "ytick.labelsize": font_size_pt - 0.4,
        "legend.fontsize": font_size_pt - 0.55,
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
    ax.text(
        -0.08,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=mpl.rcParams["font.size"] + 1.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def format_metric(value: float, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except Exception:
        return "--"
    if not math.isfinite(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def tr(lang: str, zh_text: str, en_text: str) -> str:
    return zh_text if lang == "zh" else en_text


def localized_output_path(path: Path, lang: str) -> Path:
    if lang == "zh":
        return path
    return path.with_name(f"{path.stem}_en{path.suffix}")


def build_used_dataframe(profile_df: pd.DataFrame) -> pd.DataFrame:
    used_df = profile_df.copy()
    used_df["is_used"] = pd.to_numeric(used_df["is_used"], errors="coerce").fillna(0).astype(int)
    used_df = used_df[used_df["is_used"] == 1].copy()
    for column in ["s_aligned_mm", "r_aligned_mm", "r_design_mm", "normal_error_um", "profile_error_um"]:
        if column in used_df.columns:
            used_df[column] = pd.to_numeric(used_df[column], errors="coerce")
    return used_df


def clipped_limits(values: np.ndarray, low_q: float = 0.01, high_q: float = 0.99,
                   min_span: float = 24.0) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(finite, low_q))
    hi = float(np.quantile(finite, high_q))
    span = max(min_span, hi - lo)
    pad = 0.08 * span
    return lo - pad, hi + pad


def add_clipped_markers(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                        y_min: float, y_max: float, color: str) -> None:
    above = y > y_max
    below = y < y_min
    if np.any(above):
        ax.plot(x[above], np.full(np.count_nonzero(above), y_max), "^",
                color=color, markersize=2.2, linewidth=0, clip_on=False)
    if np.any(below):
        ax.plot(x[below], np.full(np.count_nonzero(below), y_min), "v",
                color=color, markersize=2.2, linewidth=0, clip_on=False)


def draw_segment_boundaries(ax: plt.Axes, s_coords: np.ndarray) -> None:
    boundaries_mm = [36.0, 55.0, 102.04]
    s_min = float(np.min(s_coords))
    s_max = float(np.max(s_coords))
    for boundary in boundaries_mm:
        if s_min <= boundary <= s_max:
            ax.axvline(boundary, color=COLORS["segment"], linewidth=0.6,
                       linestyle="--", alpha=0.50)


def export_publication_figure(data: FigureData, output_svg: Path,
                              output_png: Path | None, svg_only: bool, width_mm: float,
                              lang: str = "zh") -> None:
    summary = data.summary_row
    used_df = build_used_dataframe(data.profile_df)
    display_df = build_display_dataframe(data.profile_df, data.summary_row)
    if len(used_df) < 50:
        raise ValueError("too few used points for figure export")

    s_all = pd.to_numeric(data.profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(data.profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(data.profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1

    width_in = mm_to_inches(width_mm)
    height_in = width_in * 0.50
    fig = plt.figure(figsize=(width_in, height_in))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.06, 0.92], hspace=0.10)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.10)

    panel_label(ax_top, "(a)")
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax_top.scatter(
            s_all[excluded],
            r_all[excluded],
            s=2.2,
            color=COLORS["excluded"],
            linewidths=0,
            alpha=0.16,
            rasterized=True,
        )
    ax_top.plot(
        display_df["s_aligned_mm"],
        display_df["r_aligned_mm"],
        color=COLORS["measured"],
        linewidth=1.15,
        label=tr(lang, "实测轮廓", "Measured contour"),
    )
    ax_top.plot(
        display_df["s_aligned_mm"],
        display_df["r_design_display_mm"],
        color=COLORS["design"],
        linewidth=1.02,
        linestyle="--",
        dashes=(5, 2.6),
        label=tr(lang, "设计轮廓", "Design contour"),
    )
    s_used = display_df["s_aligned_mm"].to_numpy(dtype=float)
    r_used = display_df["r_aligned_mm"].to_numpy(dtype=float)
    r_design = display_df["r_design_display_mm"].to_numpy(dtype=float)
    draw_segment_boundaries(ax_top, s_used)
    s_min = float(np.nanmin(s_used))
    s_max = float(np.nanmax(s_used))
    s_pad = max(0.6, (s_max - s_min) * 0.015)
    r_min = float(np.nanmin(np.r_[r_used, r_design]))
    r_max = float(np.nanmax(np.r_[r_used, r_design]))
    r_pad = max(0.18, (r_max - r_min) * 0.10)
    ax_top.set_xlim(s_min - s_pad, s_max + s_pad)
    ax_top.set_ylim(r_min - r_pad, r_max + r_pad)
    ax_top.set_ylabel(tr(lang, "半径 r (mm)", "Radius r (mm)"))
    ax_top.grid(True, axis="y")
    ax_top.legend(loc="lower left")
    ax_top.tick_params(axis="x", labelbottom=False)

    panel_label(ax_bottom, "(b)")
    s = used_df["s_aligned_mm"].to_numpy(dtype=float)
    profile_err = used_df["profile_error_um"].to_numpy(dtype=float)
    normal_err = used_df["normal_error_um"].to_numpy(dtype=float)
    ax_bottom.plot(s, profile_err, color=COLORS["profile_error"], linewidth=0.95, label=tr(lang, "型面误差", "Profile error"))
    ax_bottom.plot(s, normal_err, color=COLORS["normal_error"], linewidth=0.65, alpha=0.25, label=tr(lang, "法向误差", "Normal error"))
    ax_bottom.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    p95_val = float(summary["profile_p95_abs_um"]) if math.isfinite(float(summary["profile_p95_abs_um"])) else float("nan")
    if math.isfinite(p95_val):
        ax_bottom.axhspan(-p95_val, p95_val, color=COLORS["band"], alpha=0.16, linewidth=0, label=tr(lang, "±P95带", "±P95 band"))
    y_min, y_max = clipped_limits(profile_err, low_q=0.005, high_q=0.995, min_span=28.0)
    ax_bottom.set_ylim(y_min, y_max)
    draw_segment_boundaries(ax_bottom, s)
    ax_bottom.set_xlabel(tr(lang, "轴向坐标 s (mm)", "Axial coordinate s (mm)"))
    ax_bottom.set_ylabel(tr(lang, "误差 (μm)", "Error (μm)"))
    ax_bottom.grid(True, axis="y")
    ax_bottom.legend(loc="upper left")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    if not svg_only and output_png is not None:
        fig.savefig(output_png, format="png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = resolve_result_dir(args)
    data = load_figure_data(result_dir)
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    output_svg = args.output_svg or (result_dir / DEFAULT_SVG_NAME)
    output_png = None if args.svg_only else (args.output_png or (result_dir / DEFAULT_PNG_NAME))

    for lang in FIGURE_LANGUAGES:
        export_publication_figure(
            data=data,
            output_svg=localized_output_path(output_svg, lang),
            output_png=None if output_png is None else localized_output_path(output_png, lang),
            svg_only=args.svg_only,
            width_mm=args.figure_width_mm,
            lang=lang,
        )

    print(f"[OK] SVG figure saved to: {output_svg}")
    if output_png is not None:
        print(f"[OK] PNG preview saved to: {output_png}")


if __name__ == "__main__":
    main()
