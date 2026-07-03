#!/usr/bin/env python3
"""
Generate standalone publication-style figures for one measurement result directory.

Each output figure is intended to be directly reusable in a manuscript or response
package, with consistent typography, color palette, and axis presentation.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from publication_figure import build_display_dataframe

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False


COLORS = {
    "measured": "#0B5FA5",
    "design": "#1B9E77",
    "profile_error": "#E66101",
    "normal_error": "#7570B3",
    "excluded": "#BFC7D2",
    "bar": "#1F78B4",
    "hist": "#8DA0CB",
    "line": "#222222",
    "reference": "#4D4D4D",
    "band": "#E8EEF6",
    "segment": "#B0B7C3",
    "risk": "#CC3311",
    "prior": "#6A3D9A",
    "rescue": "#D95F02",
    "threshold": "#7A8793",
}


@dataclass
class RunData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series
    stitching_df: Optional[pd.DataFrame]
    used_df: pd.DataFrame


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
    ax.text(-0.10, 1.05, label, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.2,
            fontweight="bold", va="bottom", ha="left")


def setup_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prior_clamped_mask(df: pd.DataFrame) -> np.ndarray:
    prior_col = _find_column(df, ["PriorClamped", "prior_clamped", "prior_clamp"])
    if prior_col is not None:
        values = pd.to_numeric(df[prior_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return values > 0.5
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])
    if mode_col is not None:
        return df[mode_col].astype(str).str.contains("prior", case=False, na=False).to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def selection_mode_mask(df: pd.DataFrame, pattern: str) -> np.ndarray:
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])
    if mode_col is None:
        return np.zeros(len(df), dtype=bool)
    return df[mode_col].astype(str).str.contains(pattern, case=False, na=False, regex=False).to_numpy(dtype=bool)


def collect_motion_diagnostics(df: pd.DataFrame, rmse_px: np.ndarray) -> Optional[dict]:
    primary_col = _find_column(df, ["PrimaryShift(px)", "primary_shift_px", "primary_shift"])
    if primary_col is None:
        return None
    primary = pd.to_numeric(df[primary_col], errors="coerce").to_numpy(dtype=float)
    if len(primary) != len(df) or not np.isfinite(primary).any():
        return None

    nominal_col = _find_column(df, ["NominalPrimaryShift(px)", "nominal_primary_shift_px"])
    actual_overlap_col = _find_column(df, ["ActualOverlapRatio", "actual_overlap_ratio"])
    nominal_overlap_col = _find_column(df, ["NominalOverlapRatio", "nominal_overlap_ratio"])
    image_primary_col = _find_column(df, ["ImageCorrPrimaryShift(px)", "image_corr_primary_shift_px"])
    image_perp_col = _find_column(df, ["ImageCorrPerpShift(px)", "image_corr_perp_shift_px"])
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])

    prior_mask = prior_clamped_mask(df)
    rescue_mask = selection_mode_mask(df, "image_correlation_rescue")
    stable_mask = np.isfinite(primary) & ~prior_mask & ~rescue_mask
    if len(rmse_px) == len(df):
        stable_mask &= np.isfinite(rmse_px) & (rmse_px <= 0.25)
    if mode_col is not None:
        stable_mask &= ~df[mode_col].astype(str).str.contains(
            "trajectory_prior|wide_search|unfiltered|candidate_reselect",
            case=False,
            na=False,
            regex=True,
        ).to_numpy(dtype=bool)

    diagnostics: dict = {}
    if nominal_col is not None:
        nominal = pd.to_numeric(df[nominal_col], errors="coerce").to_numpy(dtype=float)
        finite_nominal = nominal[np.isfinite(nominal)]
        if finite_nominal.size > 0:
            diagnostics["nominal_primary_px"] = float(np.median(finite_nominal))
    if np.any(stable_mask):
        diagnostics["stable_primary_px"] = float(np.median(primary[stable_mask]))
    if actual_overlap_col is not None and np.any(stable_mask):
        actual_overlap = pd.to_numeric(df[actual_overlap_col], errors="coerce").to_numpy(dtype=float)
        stable_overlap = actual_overlap[stable_mask & np.isfinite(actual_overlap)]
        if stable_overlap.size > 0:
            diagnostics["stable_overlap_ratio"] = float(np.median(stable_overlap))
    if nominal_overlap_col is not None:
        nominal_overlap = pd.to_numeric(df[nominal_overlap_col], errors="coerce").to_numpy(dtype=float)
        finite_nominal_overlap = nominal_overlap[np.isfinite(nominal_overlap)]
        if finite_nominal_overlap.size > 0:
            diagnostics["nominal_overlap_ratio"] = float(np.median(finite_nominal_overlap))
    if len(rmse_px) == len(df) and np.isfinite(rmse_px).any():
        risk_idx = int(np.nanargmax(rmse_px))
        diagnostics["risk_step"] = risk_idx + 1
        if np.isfinite(primary[risk_idx]):
            diagnostics["risk_primary_px"] = float(primary[risk_idx])
        if image_primary_col is not None:
            image_primary = pd.to_numeric(df[image_primary_col], errors="coerce").to_numpy(dtype=float)
            if len(image_primary) == len(df) and np.isfinite(image_primary[risk_idx]):
                diagnostics["risk_image_primary_px"] = float(image_primary[risk_idx])
        if image_perp_col is not None:
            image_perp = pd.to_numeric(df[image_perp_col], errors="coerce").to_numpy(dtype=float)
            if len(image_perp) == len(df) and np.isfinite(image_perp[risk_idx]):
                diagnostics["risk_image_perp_px"] = float(image_perp[risk_idx])
    return diagnostics or None


def clipped_limits(values: np.ndarray,
                   low_q: float = 0.01,
                   high_q: float = 0.99,
                   min_span: float = 24.0) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(finite, low_q))
    hi = float(np.quantile(finite, high_q))
    span = max(min_span, hi - lo)
    pad = 0.08 * span
    return lo - pad, hi + pad


def add_clipped_markers(ax: plt.Axes,
                        x: np.ndarray,
                        y: np.ndarray,
                        y_min: float,
                        y_max: float,
                        color: str) -> None:
    above = y > y_max
    below = y < y_min
    if np.any(above):
        ax.plot(x[above], np.full(np.count_nonzero(above), y_max), "^",
                color=color, markersize=2.2, linewidth=0, clip_on=False)
    if np.any(below):
        ax.plot(x[below], np.full(np.count_nonzero(below), y_min), "v",
                color=color, markersize=2.2, linewidth=0, clip_on=False)


def sliding_stats(s: np.ndarray,
                  y: np.ndarray,
                  window_mm: float = 5.0) -> Tuple[np.ndarray, ...]:
    if len(s) < 5:
        return s, y, y, y
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    n_out = min(240, len(s_sorted))
    s_out = np.linspace(s_sorted[0], s_sorted[-1], n_out)
    mean_out = np.full_like(s_out, np.nan)
    lower_out = np.full_like(s_out, np.nan)
    upper_out = np.full_like(s_out, np.nan)
    for i, center in enumerate(s_out):
        mask = (s_sorted >= center - window_mm / 2.0) & (s_sorted <= center + window_mm / 2.0)
        if int(np.count_nonzero(mask)) >= 5:
            values = y_sorted[mask]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            mean_out[i] = mean_val
            lower_out[i] = mean_val - 2.0 * std_val
            upper_out[i] = mean_val + 2.0 * std_val
    valid = np.isfinite(mean_out)
    return s_out[valid], mean_out[valid], lower_out[valid], upper_out[valid]


def eval_design(s_mm: float) -> Optional[Tuple[float, float]]:
    z = 155.0 - s_mm
    if z < 0.0 or z > 155.0:
        return None
    if z <= 52.958772:
        radius = 220.11920702 - 0.50803027 * z
        dr_dz = -0.50803027
    elif z <= 100.0:
        xi = (z - 52.958772) / 47.041228
        xi2, xi3, xi4, xi5, xi6 = xi * xi, xi**3, xi**4, xi**5, xi**6
        radius = (0.21387322 * xi6 - 0.86957897 * xi5 + 2.12875038 * xi4
                  - 3.85239806 * xi3 + 15.01513050 * xi2 - 23.91622723 * xi + 193.21175337)
        dxi = (6 * 0.21387322 * xi5 - 5 * 0.86957897 * xi4 + 4 * 2.12875038 * xi3
               - 3 * 3.85239806 * xi2 + 2 * 15.01513050 * xi - 23.91622723)
        dr_dz = dxi / 47.041228
    elif z <= 119.0:
        radius = 181.931189
        dr_dz = 0.0
    else:
        radius = 179.919242
        dr_dz = 0.0
    return radius, -dr_dz


def draw_segment_boundaries(ax: plt.Axes, s_coords: np.ndarray) -> None:
    boundaries_mm = [36.0, 55.0, 102.04]
    s_min = float(np.min(s_coords))
    s_max = float(np.max(s_coords))
    for boundary in boundaries_mm:
        if s_min <= boundary <= s_max:
            ax.axvline(boundary, color=COLORS["segment"], linewidth=0.6,
                       linestyle="--", alpha=0.50)


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
        used_df=used_df,
    )


def fig_contour_comparison(data: RunData, fig_dir: Path, width_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.62)))
    setup_axes_style(ax)
    display_df = build_display_dataframe(data.profile_df, data.summary_row)

    profile_df = data.profile_df
    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(s_all[excluded], r_all[excluded], s=3, color=COLORS["excluded"],
                   linewidths=0, alpha=0.40, rasterized=True, label="Excluded")

    ax.plot(display_df["s_aligned_mm"], display_df["r_aligned_mm"],
            color=COLORS["measured"], linewidth=1.0, label="Measured")

    if "r_design_display_mm" in display_df.columns and np.isfinite(display_df["r_design_display_mm"]).any():
        ax.plot(display_df["s_aligned_mm"], display_df["r_design_display_mm"],
                color=COLORS["design"], linewidth=1.0,
                linestyle="--", dashes=(6, 3), label="Design")
    else:
        s_min = float(data.used_df["s_aligned_mm"].min())
        s_max = float(data.used_df["s_aligned_mm"].max())
        s_dense = np.linspace(max(0.0, s_min), s_max, 500)
        r_dense = np.array([eval_design(value)[0] if eval_design(value) else np.nan for value in s_dense])
        valid = np.isfinite(r_dense)
        ax.plot(s_dense[valid], r_dense[valid], color=COLORS["design"], linewidth=1.0,
                linestyle="--", dashes=(6, 3), label="Design")

    draw_segment_boundaries(ax, display_df["s_aligned_mm"].to_numpy(dtype=float))
    panel_label(ax, "(a)")
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=False)
    _save_fig(fig, fig_dir, "fig_a_contour_comparison")


def fig_registration_quality(data: RunData, fig_dir: Path, width_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.55)))
    setup_axes_style(ax)

    sdf = data.stitching_df
    if sdf is None or sdf.empty:
        ax.text(0.5, 0.5, "Registration data not available",
                transform=ax.transAxes, ha="center", va="center", color=COLORS["reference"])
        _save_fig(fig, fig_dir, "fig_b_registration_quality")
        return

    steps = np.arange(1, len(sdf) + 1)
    pixel_size_um = float(data.summary_row.get("pixel_size_mm", 0.0)) * 1000.0
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0.0:
        pixel_size_um = 1.0

    rmse_col = _find_column(sdf, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)",
                                  "normal_rmse", "normal_inlier_rmse",
                                  "normal_rmse_px", "nrmse"])
    corr_col = _find_column(sdf, ["TangentCorrInlier", "TangentCorrAll",
                                  "tangent_corr_inlier", "tangent_correlation"])

    if rmse_col is None:
        ax.text(0.5, 0.5, "Normal RMSE data not available",
                transform=ax.transAxes, ha="center", va="center", color=COLORS["reference"])
        _save_fig(fig, fig_dir, "fig_b_registration_quality")
        return

    rmse_px = pd.to_numeric(sdf[rmse_col], errors="coerce").to_numpy(dtype=float)
    if len(rmse_px) != len(steps) or not np.isfinite(rmse_px).all():
        ax.text(0.5, 0.5, "Registration data length mismatch",
                transform=ax.transAxes, ha="center", va="center", color=COLORS["reference"])
        _save_fig(fig, fig_dir, "fig_b_registration_quality")
        return

    rmse_um = rmse_px * pixel_size_um
    prior_mask = prior_clamped_mask(sdf)
    rescue_mask = selection_mode_mask(sdf, "image_correlation_rescue")
    bad_mask = np.isfinite(rmse_px) & (rmse_px > 0.25)
    bar_colors = [COLORS["bar"]] * len(steps)
    for idx, is_prior in enumerate(prior_mask):
        if bool(is_prior):
            bar_colors[idx] = COLORS["prior"]
        elif bool(rescue_mask[idx]):
            bar_colors[idx] = COLORS["rescue"]
    for idx, is_bad in enumerate(bad_mask):
        if bool(is_bad):
            bar_colors[idx] = COLORS["risk"]
    ax.bar(steps, rmse_um, color=bar_colors, edgecolor="#333333",
           linewidth=0.35, alpha=0.86, width=0.60, zorder=3)

    threshold_um = 0.25 * pixel_size_um
    ax.axhline(threshold_um, color=COLORS["threshold"], linewidth=0.85, linestyle="--", alpha=0.8)
    prior_steps = steps[prior_mask]
    if len(prior_steps) > 0:
        prior_values = rmse_um[prior_mask]
        ax.scatter(prior_steps, prior_values, marker="s", s=28, facecolors="white",
                   edgecolors=COLORS["prior"], linewidths=1.0, zorder=4)
    rescue_steps = steps[rescue_mask]
    if len(rescue_steps) > 0:
        rescue_values = rmse_um[rescue_mask]
        ax.scatter(rescue_steps, rescue_values, marker="D", s=26, facecolors="white",
                   edgecolors=COLORS["rescue"], linewidths=1.0, zorder=4)

    if corr_col is not None:
        corr_vals = pd.to_numeric(sdf[corr_col], errors="coerce").to_numpy(dtype=float)
        if len(corr_vals) == len(steps) and np.isfinite(corr_vals).all():
            ax2 = ax.twinx()
            setup_axes_style(ax2)
            ax2.plot(steps, corr_vals, color=COLORS["line"], marker="o",
                     markersize=4, linewidth=1.1)
            corr_min = float(np.min(corr_vals))
            corr_max = float(np.max(corr_vals))
            if corr_min > 0.95:
                ax2.set_ylim(max(0.97, corr_min - 0.004), min(1.001, corr_max + 0.001))
            else:
                ax2.set_ylim(max(-1.0, corr_min - 0.05), min(1.0, corr_max + 0.05))
            ax2.set_ylabel(r"Tangential correlation $\rho_t$", color=COLORS["line"])
            ax2.tick_params(axis="y", colors=COLORS["line"])

    legend_handles = [
        Patch(facecolor=COLORS["bar"], edgecolor="#333333", linewidth=0.35, label="Direct"),
        Line2D([0], [0], color=COLORS["threshold"], linestyle="--", linewidth=0.85, label="Acceptance limit"),
    ]
    if np.any(bad_mask):
        legend_handles.insert(
            1,
            Patch(facecolor=COLORS["risk"], edgecolor="#333333", linewidth=0.35, label="Exceeding step"),
        )
    if np.any(prior_mask):
        legend_handles.append(
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
                   markeredgecolor=COLORS["prior"], markersize=5, linewidth=0, label="Prior clamp")
        )
    if np.any(rescue_mask):
        legend_handles.append(
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                   markeredgecolor=COLORS["rescue"], markersize=5, linewidth=0, label="Raw-image rescue")
        )
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2)

    panel_label(ax, "(b)")
    ax.set_xlabel("Stitching step")
    ax.set_ylabel(r"Normal RMSE ($\mu$m)")
    ax.set_xticks(steps)
    ax.grid(True, axis="y")
    _save_fig(fig, fig_dir, "fig_b_registration_quality")


def fig_error_profile(data: RunData, fig_dir: Path, width_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.58)))
    setup_axes_style(ax)

    aligned = data.used_df[["s_aligned_mm", "normal_error_um", "profile_error_um"]].copy()
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    aligned = aligned[np.isfinite(aligned["s_aligned_mm"]) & np.isfinite(aligned["profile_error_um"])]
    s = aligned["s_aligned_mm"].to_numpy(dtype=float)
    profile_err = aligned["profile_error_um"].to_numpy(dtype=float)
    normal_err = aligned["normal_error_um"].to_numpy(dtype=float)

    if len(s) < 10:
        ax.text(0.5, 0.5, "Insufficient error data",
                transform=ax.transAxes, ha="center", va="center", color=COLORS["reference"])
        _save_fig(fig, fig_dir, "fig_c_error_profile")
        return

    s_sm, mean_sm, low_sm, high_sm = sliding_stats(s, profile_err, window_mm=5.0)
    ax.fill_between(s_sm, low_sm, high_sm, color=COLORS["band"], alpha=0.35,
                    linewidth=0, label=r"$\pm 2\sigma$ band")
    ax.plot(s, profile_err, color=COLORS["profile_error"], linewidth=0.75,
            alpha=0.88, label="Profile-form error")

    if len(normal_err) == len(s):
        ax.plot(s, normal_err, color=COLORS["normal_error"], linewidth=0.55,
                alpha=0.42, label="Normal error")

    ax.plot(s_sm, mean_sm, color=COLORS["reference"], linewidth=0.90,
            linestyle="--", dashes=(4, 2), label="Running mean")
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.7, linestyle=":")

    p95_val = data.summary_row.get("profile_p95_abs_um", None)
    if p95_val is not None and math.isfinite(float(p95_val)):
        p95_abs = float(p95_val)
        ax.axhline(+p95_abs, color=COLORS["reference"], linewidth=0.55, linestyle=":", alpha=0.35)
        ax.axhline(-p95_abs, color=COLORS["reference"], linewidth=0.55, linestyle=":", alpha=0.35)

    draw_segment_boundaries(ax, s)
    y_stack = np.r_[profile_err, normal_err] if len(normal_err) == len(s) else profile_err
    y_min, y_max = clipped_limits(y_stack, low_q=0.01, high_q=0.99, min_span=30.0)
    ax.set_ylim(y_min, y_max)
    add_clipped_markers(ax, s, profile_err, y_min, y_max, COLORS["profile_error"])
    if len(normal_err) == len(s):
        add_clipped_markers(ax, s, normal_err, y_min, y_max, COLORS["normal_error"])

    panel_label(ax, "(c)")
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel(r"Error ($\mu$m)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)
    _save_fig(fig, fig_dir, "fig_c_error_profile")


def fig_error_distribution(data: RunData, fig_dir: Path, width_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.60)))
    setup_axes_style(ax)

    profile_errors = finite_array(data.used_df["profile_error_um"])
    if len(profile_errors) < 10:
        ax.text(0.5, 0.5, "Insufficient error data",
                transform=ax.transAxes, ha="center", va="center", color=COLORS["reference"])
        _save_fig(fig, fig_dir, "fig_d_error_distribution")
        return

    n_bins = min(50, max(18, int(math.sqrt(len(profile_errors)))))
    ax.hist(profile_errors, bins=n_bins, color=COLORS["hist"], alpha=0.80,
            edgecolor="white", linewidth=0.35, density=True, label="Histogram")

    if _HAS_SCIPY:
        try:
            kde = sp_stats.gaussian_kde(profile_errors)
            x_kde = np.linspace(float(np.min(profile_errors)), float(np.max(profile_errors)), 300)
            ax.plot(x_kde, kde(x_kde), color=COLORS["line"], linewidth=1.0, label="KDE")
        except Exception:
            pass

    mean_val = float(np.mean(profile_errors))
    std_val = float(np.std(profile_errors, ddof=1)) if len(profile_errors) > 1 else 0.0
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    ax.axvline(mean_val, color=COLORS["profile_error"], linewidth=0.8,
               linestyle="--", alpha=0.65, label="Mean")

    stats_text = rf"$\mu$={mean_val:.2f}  $\sigma$={std_val:.2f}"
    if _HAS_SCIPY:
        try:
            skew = float(sp_stats.skew(profile_errors))
            kurt = float(sp_stats.kurtosis(profile_errors))
            stats_text += rf"  skew={skew:.2f}  kurt={kurt:.2f}"
        except Exception:
            pass
    stats_text = ""
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            ha="right", va="top",
            fontsize=mpl.rcParams["font.size"] - 1.0,
            color=COLORS["reference"])

    panel_label(ax, "(d)")
    ax.set_xlabel(r"Profile-form error ($\mu$m)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False)
    _save_fig(fig, fig_dir, "fig_d_error_distribution")


def fig_summary_table(data: RunData, fig_dir: Path, width_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(width_mm), mm_to_in(width_mm * 0.54)))
    ax.axis("off")

    summary = data.summary_row

    def _to_float(key: str, default: float = float("nan")) -> float:
        value = summary.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def _s(key: str, digits: int = 2) -> str:
        value = summary.get(key, None)
        return fmt_metric(value, digits) if value is not None else "--"

    s_min = float(data.used_df["s_aligned_mm"].min())
    s_max = float(data.used_df["s_aligned_mm"].max())
    reverse_axial_flag = _to_float("design_reverse_axial", _to_float("design_reverse_z", 0.0))
    reverse_axial_text = "yes" if reverse_axial_flag and int(reverse_axial_flag) else "no"
    used_count = _to_float("used_count", float(len(data.used_df)))
    motion_diag = None
    if data.stitching_df is not None and not data.stitching_df.empty:
        rmse_col = _find_column(data.stitching_df, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)",
                                                    "normal_rmse", "normal_inlier_rmse",
                                                    "normal_rmse_px", "nrmse"])
        if rmse_col is not None:
            rmse_px = pd.to_numeric(data.stitching_df[rmse_col], errors="coerce").to_numpy(dtype=float)
            motion_diag = collect_motion_diagnostics(data.stitching_df, rmse_px)

    rows = [
        ("Sample", "Used points", str(int(used_count))),
        ("", "Pixel size", f"{_s('pixel_size_mm', 4)} mm/px"),
        ("", "Reverse axial", reverse_axial_text),
        ("", "s range", f"{s_min:.1f} to {s_max:.1f} mm"),
        ("Normal", "Mean bias", rf"{_s('mean_normal_error_um')} $\mu$m"),
        ("", "RMSE", rf"{_s('normal_rmse_um')} $\mu$m"),
        ("", "P95", rf"{_s('normal_p95_abs_um')} $\mu$m"),
        ("", "PV", rf"{_s('normal_pv_um')} $\mu$m"),
        ("Profile", "RMS", rf"{_s('profile_rms_um')} $\mu$m"),
        ("", "MAE", rf"{_s('profile_mae_um')} $\mu$m"),
        ("", "P95", rf"{_s('profile_p95_abs_um')} $\mu$m"),
        ("", "PV", rf"{_s('profile_pv_um')} $\mu$m"),
    ]
    applied_bias_refine = pd.to_numeric(pd.Series([summary.get("applied_absolute_bias_refine", 0)]), errors="coerce").fillna(0.0).iloc[0]
    if applied_bias_refine > 0.5:
        rows.append(("Alignment", "Bias refine", rf"{_s('absolute_bias_correction_um')} $\mu$m"))
    if motion_diag is not None:
        nominal_primary = motion_diag.get("nominal_primary_px", np.nan)
        stable_primary = motion_diag.get("stable_primary_px", np.nan)
        risk_step = motion_diag.get("risk_step", None)
        risk_image = motion_diag.get("risk_image_primary_px", np.nan)
        risk_primary = motion_diag.get("risk_primary_px", np.nan)
        if math.isfinite(float(nominal_primary)):
            rows.append(("Stitch", "Nominal step", f"{float(nominal_primary):.1f} px"))
        if math.isfinite(float(stable_primary)):
            rows.append(("", "Stable step", f"{float(stable_primary):.1f} px"))
        if risk_step is not None and math.isfinite(float(risk_image)) and math.isfinite(float(risk_primary)):
            rows.append(("", f"Rescue S{int(risk_step)}", f"{float(risk_image):.1f}→{float(risk_primary):.1f} px"))

    table = ax.table(
        cellText=[[row[0], row[1], row[2]] for row in rows],
        colLabels=["Group", "Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.16, 0.38, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(mpl.rcParams["font.size"] - 0.9)
    table.scale(1.0, 1.14)

    for col in range(3):
        cell = table[0, col]
        cell.set_facecolor("#F2F4F7")
        cell.set_text_props(weight="bold")

    for row_index, row in enumerate(rows):
        if row[0]:
            table[row_index + 1, 0].set_text_props(weight="bold")
            table[row_index + 1, 0].set_facecolor("#FAFBFC")
        table[row_index + 1, 2].set_text_props(ha="right")

    panel_label(ax, "(e)")
    _save_fig(fig, fig_dir, "fig_e_summary_table")


def _save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    for fmt in ("svg", "pdf", "png"):
        path = fig_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.{{svg,pdf,png}}")


def configure_matplotlib(font_size_pt: float, svg_fonttype: str) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 0.4,
        "ytick.labelsize": font_size_pt - 0.4,
        "legend.fontsize": font_size_pt - 0.6,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
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
        "grid.alpha": 0.25,
        "lines.linewidth": 1.0,
        "svg.fonttype": svg_fonttype,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone paper figures.")
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Result directory with design CSV outputs.")
    parser.add_argument("--figure-width-mm", type=float, default=140.0,
                        help="Figure width in mm.")
    parser.add_argument("--font-size-pt", type=float, default=8.5)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        print(f"[ERROR] result dir not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    configure_matplotlib(args.font_size_pt, args.svg_fonttype)

    print(f"Loading data from: {result_dir}")
    data = load_data(result_dir)
    if len(data.used_df) < 50:
        print(f"[ERROR] Too few used points ({len(data.used_df)}).", file=sys.stderr)
        sys.exit(1)

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
