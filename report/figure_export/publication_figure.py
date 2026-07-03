#!/usr/bin/env python3
"""
Export a journal-style multi-panel figure from one stitching result directory.

Outputs:
  - journal_figure.svg
  - journal_figure.pdf
  - journal_figure.png
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False


DEFAULT_SVG_NAME = "journal_figure.svg"
DEFAULT_PDF_NAME = "journal_figure.pdf"
DEFAULT_PNG_NAME = "journal_figure.png"

COLORS = {
    "measured": "#0B5FA5",
    "design": "#1B9E77",
    "profile_error": "#E66101",
    "normal_error": "#7570B3",
    "excluded": "#BFC7D2",
    "bar": "#1F78B4",
    "line": "#222222",
    "hist": "#8DA0CB",
    "kde": "#222222",
    "reference": "#4D4D4D",
    "band": "#E8EEF6",
    "segment": "#C7CDD6",
    "risk": "#CC3311",
    "prior": "#6A3D9A",
    "rescue": "#D95F02",
    "threshold": "#7A8793",
}

_DESIGN_PROFILE_MIN_Z_MM = 0.0
_DESIGN_PROFILE_MAX_Z_MM = 155.0
_LINEAR_SEGMENT_END_Z_MM = 52.958772
_POLYNOMIAL_SEGMENT_END_Z_MM = 100.0
_CONSTANT_SEGMENT1_END_Z_MM = 119.0
_POLYNOMIAL_XI_SCALE = 47.041228


@dataclass
class FigureData:
    result_dir: Path
    profile_df: pd.DataFrame
    summary_row: pd.Series
    stitching_df: Optional[pd.DataFrame]
    panorama_path: Optional[Path]


@dataclass
class PanelContext:
    fig: plt.Figure
    axes: Dict[str, plt.Axes]
    data: FigureData
    used_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a publication-ready multi-panel figure.")
    parser.add_argument("--result-dir", type=Path, help="Result directory containing CSV outputs.")
    parser.add_argument("--result-root", type=Path, default=Path("result/workpiece"),
                        help="Root used to auto-discover the latest result when --result-dir is omitted.")
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=178.0)
    parser.add_argument("--font-size-pt", type=float, default=8.0)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 1.0, fontweight="bold",
            ha="left", va="bottom")


def finite_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def fmt_metric(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


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


def apply_metric_axis_ratio(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                            x_pad_ratio: float = 0.02, y_pad_ratio: float = 0.08) -> None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    x_pad = max(0.2, x_span * x_pad_ratio)
    y_pad = max(0.2, y_span * y_pad_ratio)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_box_aspect((y_span + 2.0 * y_pad) / (x_span + 2.0 * x_pad))


def _show_image_preserve_ratio(ax: plt.Axes, image: np.ndarray, cmap: Optional[str] = None) -> None:
    if image.ndim >= 2 and image.shape[0] > 0 and image.shape[1] > 0:
        ax.set_box_aspect(float(image.shape[0]) / float(image.shape[1]))
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    ax.set_anchor("C")
    ax.set_axis_off()


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
    })


def has_required_outputs(result_dir: Path) -> bool:
    return (
        (result_dir / "design_error_profile.csv").exists()
        and (result_dir / "design_error_summary.csv").exists()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")
    candidates = [p for p in result_root.iterdir() if p.is_dir() and has_required_outputs(p)]
    if not candidates:
        raise FileNotFoundError(f"no valid result directory under: {result_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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

    stitching_df = None
    stitching_csv = result_dir / "stitching_data.csv"
    if stitching_csv.exists():
        stitching_df = pd.read_csv(stitching_csv)

    panorama_path = None
    for candidate in ["final_panorama.png", "panorama.png", "stitching_result.png"]:
        path = result_dir / candidate
        if path.exists():
            panorama_path = path
            break

    return FigureData(
        result_dir=result_dir,
        profile_df=profile_df,
        summary_row=summary_df.iloc[0],
        stitching_df=stitching_df,
        panorama_path=panorama_path,
    )


def build_used_dataframe(profile_df: pd.DataFrame) -> pd.DataFrame:
    used_df = profile_df.copy()
    used_df["is_used"] = pd.to_numeric(used_df["is_used"], errors="coerce").fillna(0).astype(int)
    used_df = used_df[used_df["is_used"] == 1].copy()
    for col in ["index", "s_aligned_mm", "r_aligned_mm", "r_design_mm",
                "normal_error_um", "profile_error_um", "radial_error_um"]:
        if col in used_df.columns:
            used_df[col] = pd.to_numeric(used_df[col], errors="coerce")
    return used_df


def eval_design_compare(s_mm: float, reverse_axial: bool = False) -> Optional[Tuple[float, float]]:
    if not math.isfinite(float(s_mm)):
        return None
    z_mm = _DESIGN_PROFILE_MAX_Z_MM - float(s_mm) if reverse_axial else float(s_mm)
    if z_mm < _DESIGN_PROFILE_MIN_Z_MM or z_mm > _DESIGN_PROFILE_MAX_Z_MM:
        return None

    if z_mm <= _LINEAR_SEGMENT_END_Z_MM:
        radius = 220.11920702 - 0.50803027 * z_mm
        dr_dz = -0.50803027
    elif z_mm <= _POLYNOMIAL_SEGMENT_END_Z_MM:
        xi = (z_mm - _LINEAR_SEGMENT_END_Z_MM) / _POLYNOMIAL_XI_SCALE
        xi2 = xi * xi
        xi3 = xi2 * xi
        xi4 = xi3 * xi
        xi5 = xi4 * xi
        xi6 = xi5 * xi
        radius = (
            0.21387322 * xi6
            - 0.86957897 * xi5
            + 2.12875038 * xi4
            - 3.85239806 * xi3
            + 15.01513050 * xi2
            - 23.91622723 * xi
            + 193.21175337
        )
        dxi = (
            6.0 * 0.21387322 * xi5
            - 5.0 * 0.86957897 * xi4
            + 4.0 * 2.12875038 * xi3
            - 3.0 * 3.85239806 * xi2
            + 2.0 * 15.01513050 * xi
            - 23.91622723
        )
        dr_dz = dxi / _POLYNOMIAL_XI_SCALE
    elif z_mm <= _CONSTANT_SEGMENT1_END_Z_MM:
        radius = 181.931189
        dr_dz = 0.0
    else:
        radius = 179.919242
        dr_dz = 0.0

    return radius, (-dr_dz if reverse_axial else dr_dz)


def build_display_dataframe(profile_df: pd.DataFrame,
                            summary_row: Optional[pd.Series] = None) -> pd.DataFrame:
    display_df = profile_df.copy()
    for col in [
        "index",
        "s_aligned_mm",
        "r_aligned_mm",
        "r_design_mm",
        "normal_error_um",
        "profile_error_um",
        "radial_error_um",
    ]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

    if "is_used" in display_df.columns:
        display_df["is_used"] = pd.to_numeric(display_df["is_used"], errors="coerce").fillna(0).astype(int)
    else:
        display_df["is_used"] = 0

    finite_mask = np.isfinite(display_df["s_aligned_mm"]) & np.isfinite(display_df["r_aligned_mm"])
    # Main contour curves should only use points that actually participate in
    # the design comparison. Excluded left-end residual points are rendered
    # separately as scatter markers, otherwise they visually look like a
    # misaligned anchor segment.
    display_mask = display_df["is_used"].to_numpy(dtype=int) == 1
    display_df = display_df[finite_mask & display_mask].copy()
    display_df.sort_values("s_aligned_mm", inplace=True)
    display_df.reset_index(drop=True, inplace=True)

    reverse_axial = False
    if summary_row is not None:
        legacy_default = summary_row.get("design_reverse_z", 0)
        reverse_raw = pd.to_numeric(
            pd.Series([summary_row.get("design_reverse_axial", legacy_default)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
        reverse_axial = float(reverse_raw) > 0.5

    if "r_design_mm" in display_df.columns:
        design_display = np.array(display_df["r_design_mm"].to_numpy(dtype=float), copy=True)
    else:
        design_display = np.full(len(display_df), np.nan, dtype=float)
    missing_design = ~np.isfinite(design_display)
    if np.any(missing_design):
        s_values = display_df.loc[missing_design, "s_aligned_mm"].to_numpy(dtype=float)
        filled_values: List[float] = []
        for s_value in s_values:
            eval_result = eval_design_compare(s_value, reverse_axial)
            filled_values.append(eval_result[0] if eval_result is not None else np.nan)
        filled = np.array(filled_values, dtype=float)
        design_display[missing_design] = filled
    display_df["r_design_display_mm"] = design_display

    # Insert NaN separators across large axial gaps so excluded transition
    # regions are shown as breaks instead of misleading straight connectors.
    display_df = _insert_nan_rows_across_gaps(display_df, "s_aligned_mm")
    return display_df


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _prior_clamped_mask(df: pd.DataFrame) -> np.ndarray:
    prior_col = _find_column(df, ["PriorClamped", "prior_clamped", "prior_clamp"])
    if prior_col is not None:
        values = pd.to_numeric(df[prior_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return values > 0.5
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])
    if mode_col is not None:
        return df[mode_col].astype(str).str.contains("prior", case=False, na=False).to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _selection_mode_mask(df: pd.DataFrame, pattern: str) -> np.ndarray:
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])
    if mode_col is None:
        return np.zeros(len(df), dtype=bool)
    return df[mode_col].astype(str).str.contains(pattern, case=False, na=False, regex=False).to_numpy(dtype=bool)


def _collect_motion_diagnostics(df: pd.DataFrame, rmse_px: np.ndarray) -> Optional[Dict[str, float | int | str]]:
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

    prior_mask = _prior_clamped_mask(df)
    rescue_mask = _selection_mode_mask(df, "image_correlation_rescue")
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

    diagnostics: Dict[str, float | int | str] = {}
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
        if mode_col is not None:
            diagnostics["risk_mode"] = str(df.iloc[risk_idx][mode_col])

    return diagnostics or None


def _draw_segment_boundaries(ax: plt.Axes, s_coords: np.ndarray) -> None:
    boundaries_mm = [36.0, 55.0, 102.04]
    finite_s = np.asarray(s_coords, dtype=float)
    finite_s = finite_s[np.isfinite(finite_s)]
    if finite_s.size == 0:
        return
    s_min, s_max = float(np.min(finite_s)), float(np.max(finite_s))
    for boundary in boundaries_mm:
        if s_min <= boundary <= s_max:
            ax.axvline(boundary, color=COLORS["segment"], linewidth=0.6, linestyle="--", alpha=0.42)


def _gap_threshold_from_s(s_coords: np.ndarray) -> Optional[float]:
    s_array = np.asarray(s_coords, dtype=float)
    finite_s = s_array[np.isfinite(s_array)]
    if finite_s.size < 2:
        return None
    s_diffs = np.diff(finite_s)
    positive_diffs = s_diffs[np.isfinite(s_diffs) & (s_diffs > 0.0)]
    if positive_diffs.size == 0:
        return None
    return max(1.0, 80.0 * float(np.median(positive_diffs)))


def _gap_threshold_from_index(index_coords: np.ndarray) -> Optional[float]:
    index_array = np.asarray(index_coords, dtype=float)
    finite_index = index_array[np.isfinite(index_array)]
    if finite_index.size < 2:
        return None
    index_diffs = np.diff(finite_index)
    positive_diffs = index_diffs[np.isfinite(index_diffs) & (index_diffs > 0.0)]
    if positive_diffs.size == 0:
        return None
    return max(10.0, 20.0 * float(np.median(positive_diffs)))


def _insert_nan_rows_across_gaps(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or len(df) < 2:
        return df
    gap_threshold_mm = _gap_threshold_from_s(df[x_col].to_numpy(dtype=float))
    index_gap_threshold = None
    if "index" in df.columns:
        index_gap_threshold = _gap_threshold_from_index(df["index"].to_numpy(dtype=float))
    if gap_threshold_mm is None and index_gap_threshold is None:
        return df

    rows: List[pd.DataFrame] = []
    for idx in range(len(df) - 1):
        rows.append(df.iloc[[idx]])
        left_s = float(df.iloc[idx][x_col])
        right_s = float(df.iloc[idx + 1][x_col])
        left_index = float(df.iloc[idx]["index"]) if "index" in df.columns else math.nan
        right_index = float(df.iloc[idx + 1]["index"]) if "index" in df.columns else math.nan
        s_gap = (
            gap_threshold_mm is not None and
            math.isfinite(left_s) and
            math.isfinite(right_s) and
            (right_s - left_s) > gap_threshold_mm
        )
        index_gap = (
            index_gap_threshold is not None and
            math.isfinite(left_index) and
            math.isfinite(right_index) and
            (right_index - left_index) > index_gap_threshold
        )
        if s_gap or index_gap:
            gap_row = {col: np.nan for col in df.columns}
            rows.append(pd.DataFrame([gap_row]))
    rows.append(df.iloc[[-1]])
    return pd.concat(rows, ignore_index=True)


def _insert_nan_breaks_xy(x: np.ndarray,
                          *ys: np.ndarray,
                          segment_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    x_arr = np.asarray(x, dtype=float)
    y_arrs = [np.asarray(values, dtype=float) for values in ys]
    if any(len(values) != len(x_arr) for values in y_arrs) or len(x_arr) < 2:
        return (x_arr, *y_arrs)

    gap_threshold_mm = _gap_threshold_from_s(x_arr)
    ref_arr = None if segment_ref is None else np.asarray(segment_ref, dtype=float)
    if ref_arr is not None and len(ref_arr) != len(x_arr):
        ref_arr = None
    ref_gap_threshold = _gap_threshold_from_index(ref_arr) if ref_arr is not None else None
    if gap_threshold_mm is None and ref_gap_threshold is None:
        return (x_arr, *y_arrs)

    out_x: List[float] = []
    out_ys: List[List[float]] = [[] for _ in y_arrs]
    for idx in range(len(x_arr) - 1):
        out_x.append(float(x_arr[idx]))
        for y_idx, values in enumerate(y_arrs):
            out_ys[y_idx].append(float(values[idx]))
        left_s = float(x_arr[idx])
        right_s = float(x_arr[idx + 1])
        s_gap = (
            gap_threshold_mm is not None and
            math.isfinite(left_s) and
            math.isfinite(right_s) and
            (right_s - left_s) > gap_threshold_mm
        )
        ref_gap = False
        if ref_arr is not None and ref_gap_threshold is not None:
            left_ref = float(ref_arr[idx])
            right_ref = float(ref_arr[idx + 1])
            ref_gap = (
                math.isfinite(left_ref) and
                math.isfinite(right_ref) and
                (right_ref - left_ref) > ref_gap_threshold
            )
        if s_gap or ref_gap:
            out_x.append(np.nan)
            for values in out_ys:
                values.append(np.nan)

    out_x.append(float(x_arr[-1]))
    for y_idx, values in enumerate(out_ys):
        values.append(float(y_arrs[y_idx][-1]))
    return (np.asarray(out_x, dtype=float), *[np.asarray(values, dtype=float) for values in out_ys])


def _fallback_text(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=mpl.rcParams["font.size"] + 0.5, color=COLORS["reference"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _sliding_stats(s: np.ndarray,
                   y: np.ndarray,
                   window_mm: float = 5.0,
                   segment_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    if len(s) < 5:
        return s, y, y, y
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    ref_sorted = None
    if segment_ref is not None:
        ref_arr = np.asarray(segment_ref, dtype=float)
        if len(ref_arr) == len(s):
            ref_sorted = ref_arr[order]
    gap_threshold_mm = _gap_threshold_from_s(s_sorted)
    split_indices: List[int] = []
    if gap_threshold_mm is not None:
        diffs = np.diff(s_sorted)
        split_indices = list(np.where(np.isfinite(diffs) & (diffs > gap_threshold_mm))[0] + 1)
    if ref_sorted is not None:
        ref_gap_threshold = _gap_threshold_from_index(ref_sorted)
        if ref_gap_threshold is not None:
            ref_diffs = np.diff(ref_sorted)
            split_indices.extend(list(np.where(np.isfinite(ref_diffs) & (ref_diffs > ref_gap_threshold))[0] + 1))
            split_indices = sorted(set(split_indices))

    segment_bounds = [0, *split_indices, len(s_sorted)]
    s_parts: List[np.ndarray] = []
    mean_parts: List[np.ndarray] = []
    lower_parts: List[np.ndarray] = []
    upper_parts: List[np.ndarray] = []

    for segment_idx in range(len(segment_bounds) - 1):
        seg_start = segment_bounds[segment_idx]
        seg_end = segment_bounds[segment_idx + 1]
        s_segment = s_sorted[seg_start:seg_end]
        y_segment = y_sorted[seg_start:seg_end]
        if len(s_segment) == 0:
            continue
        if len(s_segment) < 5:
            s_valid = s_segment
            mean_valid = y_segment
            lower_valid = y_segment
            upper_valid = y_segment
        else:
            n_out = min(200, len(s_segment))
            s_out = np.linspace(s_segment[0], s_segment[-1], n_out)
            mean_out = np.full_like(s_out, np.nan)
            lower_out = np.full_like(s_out, np.nan)
            upper_out = np.full_like(s_out, np.nan)
            for i, sc in enumerate(s_out):
                mask = (s_segment >= sc - window_mm / 2) & (s_segment <= sc + window_mm / 2)
                if mask.sum() >= 5:
                    values = y_segment[mask]
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    mean_out[i] = mean_val
                    lower_out[i] = mean_val - 2.0 * std_val
                    upper_out[i] = mean_val + 2.0 * std_val
            valid = np.isfinite(mean_out)
            s_valid = s_out[valid]
            mean_valid = mean_out[valid]
            lower_valid = lower_out[valid]
            upper_valid = upper_out[valid]

        if len(s_valid) == 0:
            continue
        if s_parts:
            nan_sep = np.array([np.nan], dtype=float)
            s_parts.append(nan_sep)
            mean_parts.append(nan_sep.copy())
            lower_parts.append(nan_sep.copy())
            upper_parts.append(nan_sep.copy())
        s_parts.append(np.asarray(s_valid, dtype=float))
        mean_parts.append(np.asarray(mean_valid, dtype=float))
        lower_parts.append(np.asarray(lower_valid, dtype=float))
        upper_parts.append(np.asarray(upper_valid, dtype=float))

    if not s_parts:
        return s_sorted, y_sorted, y_sorted, y_sorted
    return (
        np.concatenate(s_parts),
        np.concatenate(mean_parts),
        np.concatenate(lower_parts),
        np.concatenate(upper_parts),
    )


def draw_panel_panorama(ctx: PanelContext) -> None:
    ax = ctx.axes["A"]
    panel_label(ax, "(a)")
    panorama_path = ctx.data.panorama_path
    if panorama_path is None:
        _fallback_text(ax, "Panorama image not available")
        return
    try:
        img = plt.imread(str(panorama_path))
    except Exception:
        _fallback_text(ax, f"Cannot load: {panorama_path.name}")
        return

    _show_image_preserve_ratio(ax, img, cmap="gray")
    return
    pixel_size_um = float(ctx.data.summary_row.get("pixel_size_mm", 0.010057)) * 1000.0
    ax.text(0.98, 0.02, f"1 px = {pixel_size_um:.1f} μm",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.0,
            color=COLORS["reference"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.82), edgecolor="none"))


def draw_panel_contour_comparison(ctx: PanelContext) -> None:
    ax = ctx.axes["B"]
    panel_label(ax, "(b)")
    profile_df = ctx.data.profile_df
    display_df = build_display_dataframe(profile_df, ctx.data.summary_row)

    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(s_all[excluded], r_all[excluded], s=4, color=COLORS["excluded"],
                   linewidths=0, alpha=0.45, rasterized=True, label="Excluded")

    ax.plot(display_df["s_aligned_mm"], display_df["r_aligned_mm"],
            color=COLORS["measured"], linewidth=1.0, label="Measured")
    ax.plot(display_df["s_aligned_mm"], display_df["r_design_display_mm"],
            color=COLORS["design"], linewidth=1.0, linestyle="--", dashes=(6, 3), label="Design")
    _draw_segment_boundaries(ax, display_df["s_aligned_mm"].to_numpy(dtype=float))
    apply_metric_axis_ratio(
        ax,
        display_df["s_aligned_mm"].to_numpy(dtype=float),
        np.r_[display_df["r_aligned_mm"].to_numpy(dtype=float), display_df["r_design_display_mm"].to_numpy(dtype=float)],
    )
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=False)


def draw_panel_registration_quality(ctx: PanelContext) -> None:
    ax = ctx.axes["C"]
    panel_label(ax, "(c)")
    stitching_df = ctx.data.stitching_df
    if stitching_df is None or stitching_df.empty:
        _fallback_text(ax, "Registration data not available")
        return

    steps = np.arange(1, len(stitching_df) + 1)
    pixel_size_um = float(ctx.data.summary_row.get("pixel_size_mm", 0.010057)) * 1000.0

    rmse_col = _find_column(stitching_df, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)",
                                           "normal_rmse", "normal_inlier_rmse",
                                           "normal_rmse_px", "nrmse"])
    corr_col = _find_column(stitching_df, ["TangentCorrInlier", "TangentCorrAll",
                                           "tangent_corr_inlier", "tangent_correlation"])
    if rmse_col is None:
        _fallback_text(ax, "Normal RMSE data not available")
        return

    rmse_px = pd.to_numeric(stitching_df[rmse_col], errors="coerce").to_numpy(dtype=float)
    if len(rmse_px) != len(steps) or not np.isfinite(rmse_px).all():
        _fallback_text(ax, "Registration data length mismatch")
        return
    rmse_um = rmse_px * pixel_size_um
    prior_mask = _prior_clamped_mask(stitching_df)
    rescue_mask = _selection_mode_mask(stitching_df, "image_correlation_rescue")
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
    ax.bar(steps, rmse_um, color=bar_colors, edgecolor="#333333", linewidth=0.35, alpha=0.85, width=0.60)
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
        corr_vals = pd.to_numeric(stitching_df[corr_col], errors="coerce").to_numpy(dtype=float)
        if len(corr_vals) == len(steps) and np.isfinite(corr_vals).all():
            ax2 = ax.twinx()
            ax2.plot(steps, corr_vals, color=COLORS["line"], marker="o", markersize=4, linewidth=1.1)
            corr_min = float(np.min(corr_vals))
            corr_max = float(np.max(corr_vals))
            if corr_min > 0.95:
                ax2.set_ylim(max(0.97, corr_min - 0.004), min(1.001, corr_max + 0.001))
            else:
                ax2.set_ylim(max(-1.0, corr_min - 0.05), min(1.0, corr_max + 0.05))
            ax2.set_ylabel(r"$\rho_t$", color=COLORS["line"])
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

    ax.set_xlabel("Stitching step")
    ax.set_ylabel("Normal RMSE (μm)")
    ax.set_xticks(steps)
    ax.grid(True, axis="y")


def draw_panel_error_profile(ctx: PanelContext) -> None:
    ax = ctx.axes["D"]
    panel_label(ax, "(d)")
    error_df = ctx.used_df[["index", "s_aligned_mm", "normal_error_um", "profile_error_um"]].copy()
    for col in ["index", "s_aligned_mm", "normal_error_um", "profile_error_um"]:
        error_df[col] = pd.to_numeric(error_df[col], errors="coerce")
    error_df = error_df[np.isfinite(error_df["s_aligned_mm"])
                        & np.isfinite(error_df["normal_error_um"])
                        & np.isfinite(error_df["profile_error_um"])].copy()
    error_df.sort_values("s_aligned_mm", inplace=True)
    s = error_df["s_aligned_mm"].to_numpy(dtype=float)
    source_index = error_df["index"].to_numpy(dtype=float)
    normal_err = error_df["normal_error_um"].to_numpy(dtype=float)
    profile_err = error_df["profile_error_um"].to_numpy(dtype=float)
    if len(s) < 10:
        _fallback_text(ax, "Insufficient error data")
        return

    window_mm = 5.0
    s_smooth, mean_smooth, lower_smooth, upper_smooth = _sliding_stats(
        s, profile_err, window_mm=window_mm, segment_ref=source_index
    )
    s, profile_err, normal_err = _insert_nan_breaks_xy(
        s, profile_err, normal_err, segment_ref=source_index
    )
    s_smooth, mean_smooth, lower_smooth, upper_smooth = _insert_nan_breaks_xy(
        s_smooth, mean_smooth, lower_smooth, upper_smooth
    )
    ax.fill_between(s_smooth, lower_smooth, upper_smooth,
                    color=COLORS["band"], alpha=0.35, linewidth=0,
                    label=f"±2σ band ({window_mm:.0f} mm window)")
    ax.plot(s, profile_err, color=COLORS["profile_error"], linewidth=0.75, alpha=0.88, label="Profile-form error")
    ax.plot(s_smooth, mean_smooth, color=COLORS["reference"], linewidth=0.9,
            linestyle="--", dashes=(4, 2), label="Running mean")
    ax.plot(s, normal_err, color=COLORS["normal_error"], linewidth=0.55,
            alpha=0.42, label="Normal error")
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.7, linestyle=":")
    _draw_segment_boundaries(ax, s)

    p95 = ctx.data.summary_row.get("profile_p95_abs_um", None)
    if p95 is not None and math.isfinite(float(p95)):
        p95_val = float(p95)
        ax.axhline(+p95_val, color=COLORS["reference"], linewidth=0.5, linestyle=":", alpha=0.35)
        ax.axhline(-p95_val, color=COLORS["reference"], linewidth=0.5, linestyle=":", alpha=0.35)

    y_min, y_max = clipped_limits(np.r_[profile_err, normal_err], low_q=0.01, high_q=0.99, min_span=30.0)
    ax.set_ylim(y_min, y_max)
    add_clipped_markers(ax, s, profile_err, y_min, y_max, COLORS["profile_error"])
    add_clipped_markers(ax, s, normal_err, y_min, y_max, COLORS["normal_error"])
    ax.set_xlabel("Axial coordinate s (mm)")
    ax.set_ylabel("Error (μm)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)


def draw_panel_error_distribution(ctx: PanelContext) -> None:
    ax = ctx.axes["E"]
    panel_label(ax, "(e)")
    profile_errors = finite_array(ctx.used_df["profile_error_um"])
    if len(profile_errors) < 10:
        _fallback_text(ax, "Insufficient error data")
        return

    n_bins = min(50, max(18, int(math.sqrt(len(profile_errors)))))
    ax.hist(profile_errors, bins=n_bins, color=COLORS["hist"], alpha=0.78,
            edgecolor="white", linewidth=0.35, density=True, label="Histogram")
    if _HAS_SCIPY:
        try:
            kde = sp_stats.gaussian_kde(profile_errors)
            x_kde = np.linspace(float(np.min(profile_errors)), float(np.max(profile_errors)), 300)
            ax.plot(x_kde, kde(x_kde), color=COLORS["kde"], linewidth=1.0, label="KDE")
        except Exception:
            pass

    mean_val = float(np.mean(profile_errors))
    std_val = float(np.std(profile_errors, ddof=1)) if len(profile_errors) > 1 else 0.0
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    ax.axvline(mean_val, color=COLORS["profile_error"], linewidth=0.8, linestyle="--", alpha=0.65, label="Mean")
    stats_text = f"μ={mean_val:.2f}  σ={std_val:.2f}"
    if _HAS_SCIPY:
        try:
            skew = float(sp_stats.skew(profile_errors))
            kurt = float(sp_stats.kurtosis(profile_errors))
            stats_text += f"  skew={skew:.2f}  kurt={kurt:.2f}"
        except Exception:
            pass
    stats_text = ""
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=mpl.rcParams["font.size"] - 1.1,
            color=COLORS["reference"])
    ax.set_xlabel("Profile-form error (μm)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False)


def draw_panel_metrics_table(ctx: PanelContext) -> None:
    ax = ctx.axes["F"]
    panel_label(ax, "(f)")
    ax.axis("off")
    summary = ctx.data.summary_row

    def _s(key: str, digits: int = 2) -> str:
        value = summary.get(key, None)
        return fmt_metric(value, digits) if value is not None else "--"

    reverse_axial_raw = pd.to_numeric(
        pd.Series([summary.get("design_reverse_axial", summary.get("design_reverse_z", 0))]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    reverse_axial_text = "yes" if int(float(reverse_axial_raw)) else "no"

    motion_diag = None
    if ctx.data.stitching_df is not None and not ctx.data.stitching_df.empty:
        rmse_col = _find_column(ctx.data.stitching_df, ["NormalRMSEInlier(px)", "NormalRMSEAll(px)",
                                                        "normal_rmse", "normal_inlier_rmse",
                                                        "normal_rmse_px", "nrmse"])
        if rmse_col is not None:
            rmse_px = pd.to_numeric(ctx.data.stitching_df[rmse_col], errors="coerce").to_numpy(dtype=float)
            motion_diag = _collect_motion_diagnostics(ctx.data.stitching_df, rmse_px)

    rows = [
        ("Sample", "Used points", str(int(float(_s("used_count", 0))))),
        ("", "Pixel size", f"{_s('pixel_size_mm', 4)} mm/px"),
        ("", "Reverse axial", reverse_axial_text),
        ("Normal", "Mean bias", f"{_s('mean_normal_error_um')} μm"),
        ("", "RMSE", f"{_s('normal_rmse_um')} μm"),
        ("", "P95", f"{_s('normal_p95_abs_um')} μm"),
        ("", "PV", f"{_s('normal_pv_um')} μm"),
        ("Profile", "RMS", f"{_s('profile_rms_um')} μm"),
        ("", "MAE", f"{_s('profile_mae_um')} μm"),
        ("", "P95", f"{_s('profile_p95_abs_um')} μm"),
        ("", "PV", f"{_s('profile_pv_um')} μm"),
    ]
    applied_bias_refine = pd.to_numeric(pd.Series([summary.get("applied_absolute_bias_refine", 0)]), errors="coerce").fillna(0.0).iloc[0]
    if applied_bias_refine > 0.5:
        rows.append(("Alignment", "Bias refine", f"{_s('absolute_bias_correction_um')} μm"))
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
        cellText=[[r[0], r[1], r[2]] for r in rows],
        colLabels=["Group", "Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.16, 0.38, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(mpl.rcParams["font.size"] - 1.0)
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


def build_figure(ctx: PanelContext) -> None:
    draw_panel_panorama(ctx)
    draw_panel_contour_comparison(ctx)
    draw_panel_registration_quality(ctx)
    draw_panel_error_profile(ctx)
    draw_panel_error_distribution(ctx)
    draw_panel_metrics_table(ctx)


def create_figure_layout(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    height_in = width_in * 0.98
    fig = plt.figure(figsize=(width_in, height_in))
    axes = fig.subplot_mosaic(
        [["A", "B"], ["C", "D"], ["E", "F"]],
        gridspec_kw={"height_ratios": [0.92, 0.92, 1.08], "wspace": 0.30, "hspace": 0.38},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06)
    return fig, axes


def export_figure(fig: plt.Figure, args: argparse.Namespace, result_dir: Path) -> None:
    svg_path = args.output_svg or (result_dir / DEFAULT_SVG_NAME)
    pdf_path = args.output_pdf or (result_dir / DEFAULT_PDF_NAME)
    png_path = args.output_png or (result_dir / DEFAULT_PNG_NAME)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.pdf_only:
        fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
        print(f"[OK] SVG -> {svg_path}")
    if not args.svg_only:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[OK] PDF -> {pdf_path}")
    if not args.svg_only and not args.pdf_only:
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
        print(f"[OK] PNG -> {png_path}")


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
    ctx = PanelContext(fig=fig, axes=axes, data=data, used_df=used_df)
    build_figure(ctx)
    export_figure(fig, args, result_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
