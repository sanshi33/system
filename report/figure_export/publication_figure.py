#!/usr/bin/env python3
"""
Publication-quality multi-panel figure for flame tube generatrix measurement.

Generates a 2x3 panel figure at journal double-column width (178 mm).
Reads design_error_profile.csv, design_error_summary.csv, and optionally
stitching_data.csv from a result directory.

Panels:
  (a) Stitched panorama + edge overlay (full-field overview)
  (b) Design contour vs. measured contour comparison
  (c) Per-step registration quality (normal RMSE + overlap coverage)
  (d) Profile-form error vs. axial coordinate with confidence bands
  (e) Error distribution histogram + KDE + Q-Q inset
  (f) Summary metrics table

Usage:
  python publication_figure.py --result-dir result/workpiece/run_20260507_120000
  python publication_figure.py --result-root result/workpiece  # auto latest
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SVG_NAME = "journal_figure.svg"
DEFAULT_PDF_NAME = "journal_figure.pdf"
DEFAULT_PNG_NAME = "journal_figure.png"

# ColorBrewer-inspired, colourblind-friendly palette
COLORS = {
    "measured":      "#1b9e77",   # teal
    "design":        "#d95f02",   # orange
    "normal_error":  "#7570b3",   # purple
    "profile_error": "#1b9e77",   # teal (same family as measured)
    "excluded":      "#bdbdbd",   # mid gray
    "histogram":     "#4292c6",   # blue
    "kde":           "#e41a1c",   # red
    "reference":     "#333333",   # dark gray
    "confidence":    "#a6cee3",   # light blue (CI band)
    "rmse_bar":      "#1f78b4",   # blue
    "coverage_line": "#e31a1c",   # red
    "anchor":        "#2166ac",   # dark blue
    "segment_line":  "#969696",   # medium gray
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FigureData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series
    stitching_df: Optional[pd.DataFrame] = None
    panorama_path: Optional[Path] = None


@dataclass
class PanelContext:
    fig: plt.Figure
    axes: Dict[str, plt.Axes]
    data: FigureData
    used_df: pd.DataFrame
    width_mm: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=mpl.rcParams["font.size"] + 1.2,
        fontweight="bold", va="bottom", ha="left",
    )


def finite_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def fmt_metric(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export publication-ready multi-panel figure for flame tube measurement."
    )
    parser.add_argument("--result-dir", type=Path, help="Result directory with design CSV outputs.")
    parser.add_argument(
        "--result-root", type=Path, default=Path("result/workpiece"),
        help="Root to auto-discover the latest result when --result-dir is omitted.",
    )
    parser.add_argument("--output-svg", type=Path, help="Explicit SVG output path.")
    parser.add_argument("--output-pdf", type=Path, help="Explicit PDF output path.")
    parser.add_argument("--output-png", type=Path, help="Explicit PNG output path.")
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=178.0)
    parser.add_argument("--font-size-pt", type=float, default=8.0)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def find_latest_result_dir(result_root: Path) -> Path:
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")
    candidates = [p for p in result_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no result directory under: {result_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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

    stitching_df: Optional[pd.DataFrame] = None
    stitching_csv = result_dir / "stitching_data.csv"
    if stitching_csv.exists():
        stitching_df = pd.read_csv(stitching_csv)

    panorama_path: Optional[Path] = None
    for candidate in ["final_panorama.png", "panorama.png", "stitching_result.png"]:
        p = result_dir / candidate
        if p.exists():
            panorama_path = p
            break

    return FigureData(
        result_dir=result_dir,
        profile_df=profile_df,
        summary_row=summary_df.iloc[0],
        stitching_df=stitching_df,
        panorama_path=panorama_path,
    )


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
def configure_matplotlib(font_size_pt: float, svg_fonttype: str) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": font_size_pt,
        "axes.titlesize": font_size_pt + 0.5,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 0.4,
        "ytick.labelsize": font_size_pt - 0.4,
        "legend.fontsize": font_size_pt - 0.6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.0,
        "svg.fonttype": svg_fonttype,
    })


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_used_dataframe(profile_df: pd.DataFrame) -> pd.DataFrame:
    used_df = profile_df.copy()
    used_df["is_used"] = pd.to_numeric(used_df["is_used"], errors="coerce").fillna(0).astype(int)
    used_df = used_df[used_df["is_used"] == 1].copy()
    for col in ["s_aligned_mm", "r_aligned_mm", "r_design_mm",
                "normal_error_um", "profile_error_um", "radial_error_um"]:
        if col in used_df.columns:
            used_df[col] = pd.to_numeric(used_df[col], errors="coerce")
    return used_df


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

# Panel A: Stitched panorama with edge overlay
def draw_panel_panorama(ctx: PanelContext) -> None:
    ax = ctx.axes["A"]
    panel_label(ax, "(a)")

    panorama_path = ctx.data.panorama_path
    if panorama_path is not None:
        try:
            img = plt.imread(str(panorama_path))
            ax.imshow(img, cmap="gray", aspect="auto")
            ax.set_title("Stitched panorama with edge overlay")
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        except Exception:
            _fallback_text(ax, f"Cannot load: {panorama_path.name}")
    else:
        _fallback_text(ax, "Panorama image not available\n(place final_panorama.png in result dir)")

    # Scale bar
    pixel_size_um = ctx.data.summary_row.get("pixel_size_mm", 0.010057) * 1000.0
    ax.text(0.98, 0.02, f"1 px = {pixel_size_um:.1f} um",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.0,
            color=COLORS["reference"], style="italic")


def _fallback_text(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=mpl.rcParams["font.size"] + 1, color=COLORS["reference"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# Panel B: Design vs measured contour comparison
def draw_panel_contour_comparison(ctx: PanelContext) -> None:
    ax = ctx.axes["B"]
    panel_label(ax, "(b)")
    profile_df = ctx.data.profile_df

    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1

    # Excluded points (scatter, gray)
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(s_all[excluded], r_all[excluded], s=4, color=COLORS["excluded"],
                   linewidths=0, alpha=0.45, rasterized=True, label="Excluded")

    # Measured contour (line)
    ax.plot(ctx.used_df["s_aligned_mm"], ctx.used_df["r_aligned_mm"],
            color=COLORS["measured"], linewidth=1.0, label="Measured")

    # Design contour (dashed)
    ax.plot(ctx.used_df["s_aligned_mm"], ctx.used_df["r_design_mm"],
            color=COLORS["design"], linewidth=1.0, linestyle="--", dashes=(6, 3),
            label="Design")

    # Anchor marker
    anchor_x = ctx.data.summary_row.get("anchor_x_px", None)
    anchor_y = ctx.data.summary_row.get("anchor_y_px", None)
    if anchor_x is not None and math.isfinite(float(anchor_x)):
        # Find nearest used point to anchor
        ax.axvline(x=0.0, color=COLORS["anchor"], linewidth=0.8, linestyle=":",
                   alpha=0.7)
        ax.annotate("Anchor", xy=(0.0, ctx.used_df["r_aligned_mm"].iloc[0]),
                    xytext=(3, 15), textcoords="offset points",
                    fontsize=mpl.rcParams["font.size"] - 1.2,
                    color=COLORS["anchor"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["anchor"], lw=0.8))

    # Segment boundaries
    _draw_segment_boundaries(ax, ctx.used_df["s_aligned_mm"])

    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=False, ncol=1)


# Panel C: Per-step registration quality
def draw_panel_registration_quality(ctx: PanelContext) -> None:
    ax = ctx.axes["C"]
    panel_label(ax, "(c)")

    stitching_df = ctx.data.stitching_df
    if stitching_df is None or stitching_df.empty:
        _fallback_text(ax, "Registration data not available\n(place stitching_data.csv in result dir)")
        return

    steps = list(range(1, len(stitching_df) + 1))

    # Normal RMSE bars
    rmse_col = _find_column(stitching_df, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)",
                                            "normal_rmse", "normal_inlier_rmse",
                                            "normal_rmse_px", "nrmse"])
    if rmse_col is not None:
        rmse_vals = finite_array(stitching_df[rmse_col])
        if len(rmse_vals) == len(steps):
            bars = ax.bar(steps, rmse_vals, color=COLORS["rmse_bar"], alpha=0.75,
                          width=0.55, label="Normal RMSE (px)", zorder=3)

    # Overlap coverage line
    cov_col = _find_column(stitching_df, ["InlierCoverageRatio", "OverlapCoverageRatio",
                                           "overlap_coverage", "overlap_coverage_ratio",
                                           "inlier_coverage_ratio", "coverage"])
    if cov_col is not None:
        cov_vals = finite_array(stitching_df[cov_col])
        if len(cov_vals) == len(steps):
            ax2 = ax.twinx()
            ax2.plot(steps, cov_vals, color=COLORS["coverage_line"], marker="o",
                     markersize=4, linewidth=1.2, label="Coverage ratio")
            ax2.set_ylabel("Overlap coverage", color=COLORS["coverage_line"])
            ax2.tick_params(axis="y", colors=COLORS["coverage_line"])
            ax2.set_ylim(0, 1.05)

    # Threshold line
    ax.axhline(y=0.1, color=COLORS["reference"], linewidth=0.7, linestyle="--",
               alpha=0.5)
    ax.text(len(steps) + 0.3, 0.1, "0.1 px", fontsize=mpl.rcParams["font.size"] - 1.2,
            color=COLORS["reference"], va="center")

    ax.set_xlabel("Stitching step")
    ax.set_ylabel("Normal RMSE (px)")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False)

    ax.set_title("Registration quality per step")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Panel D: Error profile with confidence band
def draw_panel_error_profile(ctx: PanelContext) -> None:
    ax = ctx.axes["D"]
    panel_label(ax, "(d)")

    s = finite_array(ctx.used_df["s_aligned_mm"])
    normal_err = finite_array(ctx.used_df["normal_error_um"])
    profile_err = finite_array(ctx.used_df["profile_error_um"])

    if len(s) < 10:
        _fallback_text(ax, "Insufficient error data")
        return

    # Running mean and ±2σ confidence band (sliding window)
    window_mm = 5.0
    s_smooth, mean_smooth, lower_smooth, upper_smooth = _sliding_stats(
        s, profile_err, window_mm=window_mm)

    # Confidence band
    ax.fill_between(s_smooth, lower_smooth, upper_smooth,
                    color=COLORS["confidence"], alpha=0.35, linewidth=0,
                    label=f"±2σ band ({window_mm:.0f} mm window)")

    # Profile-form error (main signal)
    ax.plot(s, profile_err, color=COLORS["profile_error"], linewidth=0.7,
            alpha=0.85, label="Profile-form error")

    # Running mean
    ax.plot(s_smooth, mean_smooth, color=COLORS["reference"], linewidth=0.9,
            linestyle="--", dashes=(4, 2), label="Running mean")

    # Normal error (faint, for reference)
    ax.plot(s, normal_err, color=COLORS["normal_error"], linewidth=0.5,
            alpha=0.35, label="Normal error (bias inc.)")

    # Zero line
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.7, linestyle=":")

    # Segment boundaries
    _draw_segment_boundaries(ax, s)

    # P95 reference lines
    p95 = ctx.data.summary_row.get("profile_p95_abs_um", None)
    if p95 is not None and math.isfinite(float(p95)):
        p95_val = float(p95)
        ax.axhline(+p95_val, color=COLORS["reference"], linewidth=0.5,
                   linestyle=":", alpha=0.4)
        ax.axhline(-p95_val, color=COLORS["reference"], linewidth=0.5,
                   linestyle=":", alpha=0.4)
        ax.text(s[-1], +p95_val, f" +P95={p95_val:.1f}",
                fontsize=mpl.rcParams["font.size"] - 1.5, va="bottom", ha="right",
                color=COLORS["reference"], alpha=0.7)
        ax.text(s[-1], -p95_val, f" -P95={p95_val:.1f}",
                fontsize=mpl.rcParams["font.size"] - 1.5, va="top", ha="right",
                color=COLORS["reference"], alpha=0.7)

    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Error (μm)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False, fontsize=mpl.rcParams["font.size"] - 1.0)
    ax.set_title("Profile-form error vs. axial coordinate")


def _sliding_stats(s: np.ndarray, y: np.ndarray, window_mm: float = 5.0
                   ) -> Tuple[np.ndarray, ...]:
    """Compute running mean and ±2σ band in sliding windows."""
    if len(s) < 5:
        return s, y, y, y

    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]

    n_out = min(200, len(s_sorted))
    s_out = np.linspace(s_sorted[0], s_sorted[-1], n_out)
    mean_out = np.full_like(s_out, np.nan)
    lower_out = np.full_like(s_out, np.nan)
    upper_out = np.full_like(s_out, np.nan)

    for i, sc in enumerate(s_out):
        mask = (s_sorted >= sc - window_mm / 2) & (s_sorted <= sc + window_mm / 2)
        if mask.sum() >= 5:
            yw = y_sorted[mask]
            m = np.mean(yw)
            sdev = np.std(yw, ddof=1)
            mean_out[i] = m
            lower_out[i] = m - 2.0 * sdev
            upper_out[i] = m + 2.0 * sdev

    valid = np.isfinite(mean_out)
    return s_out[valid], mean_out[valid], lower_out[valid], upper_out[valid]


def _draw_segment_boundaries(ax: plt.Axes, s_coords: np.ndarray) -> None:
    """Draw vertical lines at design profile segment transitions."""
    # Segment boundaries in comparison coordinates (reverse Z)
    # Linear → Polynomial at z=52.96 → s=155-52.96=102.04
    # Polynomial → Constant1 at z=100 → s=155-100=55
    # Constant1 → Constant2 at z=119 → s=155-119=36
    boundaries_mm = [36.0, 55.0, 102.04]
    labels = ["C2/C1", "C1/P", "P/L"]
    s_min, s_max = s_coords.min(), s_coords.max()
    for b, lbl in zip(boundaries_mm, labels):
        if s_min <= b <= s_max:
            ax.axvline(x=b, color=COLORS["segment_line"], linewidth=0.6,
                       linestyle="--", alpha=0.5)
            ax.text(b + 1, ax.get_ylim()[1] * 0.92, lbl,
                    fontsize=mpl.rcParams["font.size"] - 1.8,
                    color=COLORS["segment_line"], rotation=90, va="top")


# Panel E: Error distribution histogram + KDE + Q-Q inset
def draw_panel_error_distribution(ctx: PanelContext) -> None:
    ax = ctx.axes["E"]
    panel_label(ax, "(e)")

    profile_errors = finite_array(ctx.used_df["profile_error_um"])
    if len(profile_errors) < 10:
        _fallback_text(ax, "Insufficient error data")
        return

    # Histogram with KDE
    n_bins = min(50, max(18, int(math.sqrt(len(profile_errors)))))
    ax.hist(profile_errors, bins=n_bins, color=COLORS["histogram"], alpha=0.75,
            edgecolor="white", linewidth=0.4, density=True, label="Histogram")

    # KDE overlay (requires scipy)
    if _HAS_SCIPY:
        try:
            kde = sp_stats.gaussian_kde(profile_errors)
            x_kde = np.linspace(profile_errors.min(), profile_errors.max(), 300)
            ax.plot(x_kde, kde(x_kde), color=COLORS["kde"], linewidth=1.2, label="KDE")
        except Exception:
            pass

    # Vertical lines
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    mean_val = np.mean(profile_errors)
    ax.axvline(mean_val, color=COLORS["design"], linewidth=0.8, linestyle="--",
               alpha=0.6)
    ax.text(mean_val, ax.get_ylim()[1] * 0.95,
            f"  mean={mean_val:.2f}",
            fontsize=mpl.rcParams["font.size"] - 1.5, color=COLORS["design"])

    # Skewness / normality annotation
    if _HAS_SCIPY:
        skew = float(sp_stats.skew(profile_errors))
        kurt = float(sp_stats.kurtosis(profile_errors))
        try:
            _, sw_p = sp_stats.shapiro(profile_errors[:min(5000, len(profile_errors))])
            sw_text = f"SW p={sw_p:.3f}" if sw_p > 0.001 else "SW p<0.001"
        except Exception:
            sw_text = ""
        stats_text = f"skew={skew:.2f}  kurt={kurt:.2f}  {sw_text}"
    else:
        skew = float(np.mean((profile_errors - mean_val) ** 3) / (np.std(profile_errors) ** 3 + 1e-12))
        kurt = float(np.mean((profile_errors - mean_val) ** 4) / (np.std(profile_errors) ** 4 + 1e-12)) - 3.0
        stats_text = f"skew={skew:.2f}  kurt(excess)={kurt:.2f}"
    ax.text(0.98, 0.96, stats_text, transform=ax.transAxes,
            ha="right", va="top",
            fontsize=mpl.rcParams["font.size"] - 1.2, color=COLORS["reference"])

    ax.set_xlabel("Profile-form error (μm)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")

    # Q-Q plot inset
    ins_ax = ax.inset_axes([0.55, 0.08, 0.40, 0.40])
    if _HAS_SCIPY:
        try:
            sp_stats.probplot(profile_errors[:min(2000, len(profile_errors))],
                              dist="norm", plot=ins_ax)
            ins_ax.get_lines()[0].set_markersize(2.0)
            ins_ax.get_lines()[0].set_color(COLORS["profile_error"])
            ins_ax.get_lines()[1].set_color(COLORS["reference"])
            ins_ax.get_lines()[1].set_linewidth(0.8)
            ins_ax.set_title("")
            ins_ax.set_xlabel("")
            ins_ax.set_ylabel("")
            ins_ax.tick_params(labelsize=mpl.rcParams["font.size"] - 2.0)
        except Exception:
            ins_ax.text(0.5, 0.5, "Q-Q\nn/a", ha="center", va="center",
                        transform=ins_ax.transAxes, fontsize=6)
    else:
        ins_ax.text(0.5, 0.5, "Q-Q\n(scipy needed)", ha="center", va="center",
                    transform=ins_ax.transAxes, fontsize=6)

    ax.set_title("Error distribution")


# Panel F: Summary metrics table
def draw_panel_metrics_table(ctx: PanelContext) -> None:
    ax = ctx.axes["F"]
    panel_label(ax, "(f)")
    ax.axis("off")

    summary = ctx.data.summary_row

    def _s(key: str, digits: int = 2) -> str:
        val = summary.get(key, None)
        return fmt_metric(val, digits) if val is not None else "--"

    rows: List[Tuple[str, str, str]] = [
        # (group_header, metric, value)
        ("Sample", "Used points", str(int(float(_s("used_count", 0))))),
        ("", "Pixel size", f"{_s('pixel_size_mm', 4)} mm/px"),
        ("", "Reverse Z", "yes" if int(float(_s("design_reverse_z", 0))) else "no"),
        ("", "Left anchor", "yes" if int(float(_s("use_left_endpoint_anchor", 0))) else "no"),
        ("Normal error", "Mean bias", f"{_s('mean_normal_error_um')} μm"),
        ("", "RMSE", f"{_s('normal_rmse_um')} μm"),
        ("", "MAE", f"{_s('normal_mae_um')} μm"),
        ("", "P95", f"{_s('normal_p95_abs_um')} μm"),
        ("", "Peak-to-valley", f"{_s('normal_pv_um')} μm"),
        ("", "Max + / -", f"{_s('normal_max_pos_um')} / {_s('normal_max_neg_um')} μm"),
        ("Profile error", "RMS", f"{_s('profile_rms_um')} μm"),
        ("", "MAE", f"{_s('profile_mae_um')} μm"),
        ("", "P95", f"{_s('profile_p95_abs_um')} μm"),
        ("", "PV", f"{_s('profile_pv_um')} μm"),
    ]

    # Build table
    col_labels = ["Group", "Metric", "Value"]
    cell_text = [[r[0], r[1], r[2]] for r in rows]

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.18, 0.36, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(mpl.rcParams["font.size"] - 0.8)
    table.scale(1.0, 1.22)

    # Style: header row
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")

    # Style: group header rows
    for i, row in enumerate(rows):
        if row[0]:
            for j in range(3):
                cell = table[i + 1, j]
                if j == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#fafafa")

    # Right-align values
    for i in range(len(rows)):
        cell = table[i + 1, 2]
        cell.set_text_props(ha="right")

    ax.set_title("Summary metrics", loc="left", pad=12)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_figure(ctx: PanelContext) -> None:
    draw_panel_panorama(ctx)
    draw_panel_contour_comparison(ctx)
    draw_panel_registration_quality(ctx)
    draw_panel_error_profile(ctx)
    draw_panel_error_distribution(ctx)
    draw_panel_metrics_table(ctx)


def create_figure_layout(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    # Golden ratio-ish height: 2x3 layout
    height_in = width_in * 0.72

    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)

    # 2x3 mosaic
    mosaic = [
        ["A", "B"],
        ["C", "D"],
        ["E", "F"],
    ]
    axes = fig.subplot_mosaic(mosaic, gridspec_kw={
        "height_ratios": [1.0, 0.85, 0.95],
        "width_ratios": [1.0, 1.0],
        "hspace": 0.35,
        "wspace": 0.30,
    })

    return fig, axes


def export_figure(fig: plt.Figure, args: argparse.Namespace,
                  result_dir: Path) -> None:
    svg_path = args.output_svg or (result_dir / DEFAULT_SVG_NAME)
    pdf_path = args.output_pdf or (result_dir / DEFAULT_PDF_NAME)
    png_path = args.output_png or (result_dir / DEFAULT_PNG_NAME)

    svg_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.pdf_only:
        fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
        print(f"[OK] SVG → {svg_path}")

    if not args.svg_only:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[OK] PDF → {pdf_path}")

    if not args.svg_only and not args.pdf_only:
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
        print(f"[OK] PNG → {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    result_dir = resolve_result_dir(args)
    data = load_figure_data(result_dir)
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    used_df = build_used_dataframe(data.profile_df)
    if len(used_df) < 50:
        print(f"[ERROR] Too few used points ({len(used_df)}) for figure export.", file=sys.stderr)
        sys.exit(1)

    fig, axes = create_figure_layout(args.figure_width_mm)
    ctx = PanelContext(
        fig=fig, axes=axes, data=data, used_df=used_df,
        width_mm=args.figure_width_mm,
    )

    build_figure(ctx)

    # Overall suptitle
    fig.suptitle(
        "Flame Tube Generatrix Measurement — Design Profile Comparison",
        y=1.005, fontsize=mpl.rcParams["font.size"] + 1.0, fontweight="bold",
    )

    export_figure(fig, args, result_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
