#!/usr/bin/env python3
"""
Generate individual publication-quality figures for flame tube generatrix measurement.

Each figure is standalone with consistent formatting (no titles, panel labels only).
Outputs SVG + PDF + PNG to a dedicated figures/ subdirectory.

Usage:
  python paper_figures.py --result-dir result/workpiece/cli_20260514_151126
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Colour palette (ColorBrewer-inspired, colourblind-friendly)
# ---------------------------------------------------------------------------
C = {
    "measured":      "#1b9e77",
    "design":        "#d95f02",
    "normal_error":  "#7570b3",
    "profile_error": "#1b9e77",
    "excluded":      "#bdbdbd",
    "histogram":     "#4292c6",
    "kde":           "#e41a1c",
    "reference":     "#333333",
    "confidence":    "#a6cee3",
    "rmse_bar":      "#1f78b4",
    "coverage_line": "#e31a1c",
    "anchor":        "#2166ac",
    "segment_line":  "#969696",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mm_to_in(value_mm: float) -> float:
    return value_mm / 25.4


def finite_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def fmt_metric(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.5,
            fontweight="bold", va="bottom", ha="left")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def setup_axes_style(ax: plt.Axes) -> None:
    """Apply consistent styling to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


# ---------------------------------------------------------------------------
# Design profile function (reverseZ=True)
# ---------------------------------------------------------------------------
def eval_design(s_mm: float):
    """Returns (r_mm, dr_dz) at comparison coordinate s, reverseZ=True."""
    z = 155.0 - s_mm
    if z < 0.0 or z > 155.0:
        return None
    if z <= 52.958772:
        r = 220.11920702 - 0.50803027 * z
        dr_dz = -0.50803027
    elif z <= 100.0:
        xi = (z - 52.958772) / 47.041228
        xi2, xi3, xi4, xi5, xi6 = xi*xi, xi**3, xi**4, xi**5, xi**6
        r = (0.21387322*xi6 - 0.86957897*xi5 + 2.12875038*xi4
             - 3.85239806*xi3 + 15.01513050*xi2 - 23.91622723*xi + 193.21175337)
        dxi = (6*0.21387322*xi5 - 5*0.86957897*xi4 + 4*2.12875038*xi3
               - 3*3.85239806*xi2 + 2*15.01513050*xi - 23.91622723)
        dr_dz = dxi / 47.041228
    elif z <= 119.0:
        r = 181.931189
        dr_dz = 0.0
    else:
        r = 179.919242
        dr_dz = 0.0
    return (r, -dr_dz)


def draw_segment_boundaries(ax: plt.Axes, s_coords: np.ndarray) -> None:
    boundaries_mm = [36.0, 55.0, 102.04]
    labels = ["C2/C1", "C1/P", "P/L"]
    s_min, s_max = s_coords.min(), s_coords.max()
    for b, lbl in zip(boundaries_mm, labels):
        if s_min <= b <= s_max:
            ax.axvline(x=b, color=C["segment_line"], linewidth=0.6,
                       linestyle="--", alpha=0.45)
            ytop = ax.get_ylim()[1]
            ax.text(b + 1, ytop * 0.92, lbl,
                    fontsize=mpl.rcParams["font.size"] - 2.2,
                    color=C["segment_line"], rotation=90, va="top")


def sliding_stats(s: np.ndarray, y: np.ndarray, window_mm: float = 5.0):
    if len(s) < 5:
        return s, y, y, y
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]
    n_out = min(200, len(s_sorted))
    s_out = np.linspace(s_sorted[0], s_sorted[-1], n_out)
    mean_out = np.full_like(s_out, np.nan)
    lower_out = np.full_like(s_out, np.nan)
    upper_out = np.full_like(s_out, np.nan)
    for i, sc in enumerate(s_out):
        mask = (s_sorted >= sc - window_mm/2) & (s_sorted <= sc + window_mm/2)
        if mask.sum() >= 5:
            yw = y_sorted[mask]
            m = np.mean(yw)
            sdev = np.std(yw, ddof=1)
            mean_out[i] = m
            lower_out[i] = m - 2.0*sdev
            upper_out[i] = m + 2.0*sdev
    valid = np.isfinite(mean_out)
    return s_out[valid], mean_out[valid], lower_out[valid], upper_out[valid]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class RunData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series
    stitching_df: Optional[pd.DataFrame]
    panorama_path: Optional[Path]
    used_df: pd.DataFrame


def load_data(result_dir: Path) -> RunData:
    profile_csv = result_dir / "design_error_profile.csv"
    summary_csv = result_dir / "design_error_summary.csv"
    if not profile_csv.exists():
        raise FileNotFoundError(f"missing: {profile_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing: {summary_csv}")

    profile_df = pd.read_csv(profile_csv)
    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        raise ValueError(f"empty summary: {summary_csv}")

    stitching_df = None
    stitching_csv = result_dir / "stitching_data.csv"
    if stitching_csv.exists():
        stitching_df = pd.read_csv(stitching_csv)

    panorama_path = None
    for name in ["final_panorama.png", "panorama.png"]:
        p = result_dir / name
        if p.exists():
            panorama_path = p
            break

    # Build used dataframe
    used_df = profile_df.copy()
    used_df["is_used"] = pd.to_numeric(used_df["is_used"], errors="coerce").fillna(0).astype(int)
    used_df = used_df[used_df["is_used"] == 1].copy()
    for col in ["s_aligned_mm", "r_aligned_mm", "r_design_mm",
                "normal_error_um", "profile_error_um", "radial_error_um"]:
        if col in used_df.columns:
            used_df[col] = pd.to_numeric(used_df[col], errors="coerce")

    return RunData(
        result_dir=result_dir,
        profile_df=profile_df,
        summary_row=summary_df.iloc[0],
        stitching_df=stitching_df,
        panorama_path=panorama_path,
        used_df=used_df,
    )


# ---------------------------------------------------------------------------
# Individual figure builders
# ---------------------------------------------------------------------------

def fig_contour_comparison(data: RunData, fig_dir: Path, width_mm: float) -> None:
    """Figure (a): Design contour vs measured contour."""
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.62)))
    setup_axes_style(ax)

    profile_df = data.profile_df
    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1

    # Excluded points
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(s_all[excluded], r_all[excluded], s=3, color=C["excluded"],
                   linewidths=0, alpha=0.40, rasterized=True)

    # Measured
    ax.plot(data.used_df["s_aligned_mm"], data.used_df["r_aligned_mm"],
            color=C["measured"], linewidth=0.9)

    # Design (dense evaluation)
    s_min, s_max = data.used_df["s_aligned_mm"].min(), data.used_df["s_aligned_mm"].max()
    s_dense = np.linspace(max(0, s_min), s_max, 500)
    r_dense = []
    for sv in s_dense:
        de = eval_design(sv)
        r_dense.append(de[0] if de else np.nan)
    r_dense = np.array(r_dense)
    valid = np.isfinite(r_dense)
    ax.plot(s_dense[valid], r_dense[valid], color=C["design"], linewidth=1.0,
            linestyle="--", dashes=(6, 3))

    # Segment boundaries
    draw_segment_boundaries(ax, data.used_df["s_aligned_mm"].values)

    panel_label(ax, "(a)")
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True)

    # Legend
    ax.plot([], [], color=C["measured"], linewidth=0.9, label="Measured")
    ax.plot([], [], color=C["design"], linewidth=1.0, linestyle="--", label="Design")
    ax.legend(loc="lower left", frameon=False)

    _save_fig(fig, fig_dir, "fig_a_contour_comparison")


def fig_registration_quality(data: RunData, fig_dir: Path, width_mm: float) -> None:
    """Figure (b): Per-step registration quality."""
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.55)))
    setup_axes_style(ax)

    sdf = data.stitching_df
    if sdf is None or sdf.empty:
        ax.text(0.5, 0.5, "No registration data", ha="center", va="center",
                transform=ax.transAxes, color=C["reference"])
        _save_fig(fig, fig_dir, "fig_b_registration_quality")
        return

    steps = list(range(1, len(sdf) + 1))

    # Normal RMSE bars (use inlier RMSE)
    rmse_col = _find_column(sdf, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)"])
    if rmse_col:
        rmse_vals = finite_array(sdf[rmse_col])
        if len(rmse_vals) == len(steps):
            ax.bar(steps, rmse_vals, color=C["rmse_bar"], alpha=0.78,
                   width=0.55, zorder=3)

    # Overlap coverage line
    cov_col = _find_column(sdf, ["InlierCoverageRatio", "OverlapCoverageRatio"])
    if cov_col:
        cov_vals = finite_array(sdf[cov_col])
        if len(cov_vals) == len(steps):
            ax2 = ax.twinx()
            setup_axes_style(ax2)
            ax2.plot(steps, cov_vals, color=C["coverage_line"], marker="o",
                     markersize=5, linewidth=1.2)
            ax2.set_ylabel("Coverage ratio", color=C["coverage_line"])
            ax2.tick_params(axis="y", colors=C["coverage_line"])
            ax2.set_ylim(0, 1.05)

    # Threshold line at 0.1 px
    ax.axhline(y=0.1, color=C["reference"], linewidth=0.7, linestyle="--", alpha=0.45)
    ax.text(len(steps) + 0.3, 0.1, "0.1 px",
            fontsize=mpl.rcParams["font.size"] - 1.5,
            color=C["reference"], va="center")

    panel_label(ax, "(b)")
    ax.set_xlabel("Stitching step")
    ax.set_ylabel("Normal RMSE (px)")
    ax.set_xticks(steps)
    ax.grid(True, axis="y")

    _save_fig(fig, fig_dir, "fig_b_registration_quality")


def fig_error_profile(data: RunData, fig_dir: Path, width_mm: float) -> None:
    """Figure (c): Profile-form error vs axial coordinate."""
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.55)))
    setup_axes_style(ax)

    s = finite_array(data.used_df["s_aligned_mm"])
    profile_err = finite_array(data.used_df["profile_error_um"])

    if len(s) < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color=C["reference"])
        _save_fig(fig, fig_dir, "fig_c_error_profile")
        return

    # Confidence band
    s_sm, mean_sm, lo_sm, hi_sm = sliding_stats(s, profile_err, window_mm=5.0)
    ax.fill_between(s_sm, lo_sm, hi_sm, color=C["confidence"], alpha=0.30,
                    linewidth=0)

    # Profile error
    ax.plot(s, profile_err, color=C["profile_error"], linewidth=0.6, alpha=0.85)

    # Running mean
    ax.plot(s_sm, mean_sm, color=C["reference"], linewidth=0.8,
            linestyle="--", dashes=(4, 2))

    # Zero line
    ax.axhline(0.0, color=C["reference"], linewidth=0.7, linestyle=":")

    # P95 reference
    p95_val = float(data.summary_row.get("profile_p95_abs_um", 0))
    if p95_val > 0 and math.isfinite(p95_val):
        ymax = ax.get_ylim()[1]
        ax.axhline(+p95_val, color=C["reference"], linewidth=0.5, linestyle=":", alpha=0.35)
        ax.axhline(-p95_val, color=C["reference"], linewidth=0.5, linestyle=":", alpha=0.35)
        ax.text(s.max(), +p95_val, f"+P95={p95_val:.1f}",
                fontsize=mpl.rcParams["font.size"] - 2, va="bottom", ha="right",
                color=C["reference"], alpha=0.6)

    draw_segment_boundaries(ax, s)

    panel_label(ax, "(c)")
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Profile-form error (μm)")
    ax.grid(True)

    _save_fig(fig, fig_dir, "fig_c_error_profile")


def fig_error_distribution(data: RunData, fig_dir: Path, width_mm: float) -> None:
    """Figure (d): Error distribution histogram + KDE."""
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.60)))
    setup_axes_style(ax)

    profile_errors = finite_array(data.used_df["profile_error_um"])
    if len(profile_errors) < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color=C["reference"])
        _save_fig(fig, fig_dir, "fig_d_error_distribution")
        return

    n_bins = min(50, max(18, int(math.sqrt(len(profile_errors)))))
    ax.hist(profile_errors, bins=n_bins, color=C["histogram"], alpha=0.78,
            edgecolor="white", linewidth=0.3, density=True)

    if _HAS_SCIPY:
        try:
            kde = sp_stats.gaussian_kde(profile_errors)
            x_kde = np.linspace(profile_errors.min(), profile_errors.max(), 300)
            ax.plot(x_kde, kde(x_kde), color=C["kde"], linewidth=1.2)
        except Exception:
            pass

    ax.axvline(0.0, color=C["reference"], linewidth=0.8, linestyle=":")
    mean_val = np.mean(profile_errors)

    # Stats annotation
    if _HAS_SCIPY:
        skew = float(sp_stats.skew(profile_errors))
        kurt = float(sp_stats.kurtosis(profile_errors))
        try:
            _, sw_p = sp_stats.shapiro(profile_errors[:min(5000, len(profile_errors))])
            sw = f"SW p={sw_p:.3f}" if sw_p > 0.001 else "SW p<0.001"
        except Exception:
            sw = ""
        stats_text = f"μ={mean_val:.2f}  σ={np.std(profile_errors):.2f}  skew={skew:.2f}  kurt={kurt:.2f}  {sw}"
    else:
        stats_text = f"μ={mean_val:.2f}  σ={np.std(profile_errors):.2f}"

    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=mpl.rcParams["font.size"] - 1.0, color=C["reference"])

    panel_label(ax, "(d)")
    ax.set_xlabel("Profile-form error (μm)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")

    _save_fig(fig, fig_dir, "fig_d_error_distribution")


def fig_summary_table(data: RunData, fig_dir: Path, width_mm: float) -> None:
    """Figure (e): Summary metrics table."""
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.52)))
    ax.axis("off")

    summary = data.summary_row

    def _s(key: str, digits: int = 2) -> str:
        val = summary.get(key, None)
        return fmt_metric(val, digits) if val is not None else "--"

    rows = [
        ("Sample",  "Used points",     str(int(float(_s("used_count", 0))))),
        ("",        "Pixel size",      f"{_s('pixel_size_mm', 4)} mm/px"),
        ("",        "s range",         f"{data.used_df['s_aligned_mm'].min():.1f} – {data.used_df['s_aligned_mm'].max():.1f} mm"),
        ("Normal",  "Mean bias",       f"{_s('mean_normal_error_um')} μm"),
        ("",        "RMSE",            f"{_s('normal_rmse_um')} μm"),
        ("",        "P95",             f"{_s('normal_p95_abs_um')} μm"),
        ("",        "PV",              f"{_s('normal_pv_um')} μm"),
        ("Profile", "RMS",             f"{_s('profile_rms_um')} μm"),
        ("",        "MAE",             f"{_s('profile_mae_um')} μm"),
        ("",        "P95",             f"{_s('profile_p95_abs_um')} μm"),
        ("",        "PV",              f"{_s('profile_pv_um')} μm"),
    ]

    cell_text = [[r[0], r[1], r[2]] for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Group", "Metric", "Value"],
        cellLoc="left", loc="center",
        colWidths=[0.15, 0.38, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(mpl.rcParams["font.size"] - 0.5)
    table.scale(1.0, 1.28)

    # Header style
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(weight="bold")

    # Group headers bold
    for i, row in enumerate(rows):
        if row[0]:
            table[i+1, 0].set_text_props(weight="bold")
            table[i+1, 0].set_facecolor("#fafafa")

    # Right-align values
    for i in range(len(rows)):
        table[i+1, 2].set_text_props(ha="right")

    panel_label(ax, "(e)")
    _save_fig(fig, fig_dir, "fig_e_summary_table")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    for fmt in ["svg", "pdf", "png"]:
        path = fig_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.{{svg,pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate individual paper figures.")
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Result directory with design CSV outputs.")
    parser.add_argument("--figure-width-mm", type=float, default=140.0,
                        help="Figure width in mm (default 140 for 1.5-column).")
    parser.add_argument("--font-size-pt", type=float, default=8.5)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        print(f"[ERROR] result dir not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    # Setup matplotlib
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": args.font_size_pt,
        "axes.titlesize": args.font_size_pt + 0.5,
        "axes.labelsize": args.font_size_pt,
        "xtick.labelsize": args.font_size_pt - 0.4,
        "ytick.labelsize": args.font_size_pt - 0.4,
        "legend.fontsize": args.font_size_pt - 0.6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "grid.linewidth": 0.4, "grid.alpha": 0.25,
        "lines.linewidth": 1.0,
        "svg.fonttype": "path",
    })

    # Load data
    print(f"Loading data from: {result_dir}")
    data = load_data(result_dir)
    if len(data.used_df) < 50:
        print(f"[ERROR] Too few used points ({len(data.used_df)}).", file=sys.stderr)
        sys.exit(1)

    # Create figures directory
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {fig_dir}\n")

    width = args.figure_width_mm

    print("Generating figures...")
    fig_contour_comparison(data, fig_dir, width)
    fig_registration_quality(data, fig_dir, width)
    fig_error_profile(data, fig_dir, width)
    fig_error_distribution(data, fig_dir, width)
    fig_summary_table(data, fig_dir, width)

    print(f"\nDone. {len(list(fig_dir.glob('*.svg')))} figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
