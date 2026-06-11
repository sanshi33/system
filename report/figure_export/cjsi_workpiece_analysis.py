#!/usr/bin/env python3
"""
Generate a CJSI-oriented analysis package for workpiece measurement results.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

try:
    from scipy import stats as sp_stats

    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False

from publication_figure import (
    _collect_motion_diagnostics,
    _draw_segment_boundaries,
    _fallback_text,
    _find_column,
    _insert_nan_breaks_xy,
    _prior_clamped_mask,
    _selection_mode_mask,
    _show_image_preserve_ratio,
    _sliding_stats,
    add_clipped_markers,
    apply_metric_axis_ratio,
    build_display_dataframe,
    build_used_dataframe,
    clipped_limits,
    configure_matplotlib,
    finite_array,
    fmt_metric,
    load_figure_data,
    mm_to_inches,
    panel_label,
)


MAIN_SVG_NAME = "cjsi_workpiece_main_figure.svg"
MAIN_PDF_NAME = "cjsi_workpiece_main_figure.pdf"
MAIN_PNG_NAME = "cjsi_workpiece_main_figure.png"
CLEAN_MAIN_SVG_NAME = "cjsi_workpiece_main_figure_4panel.svg"
CLEAN_MAIN_PDF_NAME = "cjsi_workpiece_main_figure_4panel.pdf"
CLEAN_MAIN_PNG_NAME = "cjsi_workpiece_main_figure_4panel.png"
CONCISE_SVG_NAME = "cjsi_workpiece_concise_figure.svg"
CONCISE_PDF_NAME = "cjsi_workpiece_concise_figure.pdf"
CONCISE_PNG_NAME = "cjsi_workpiece_concise_figure.png"
ABLATION_SVG_NAME = "cjsi_workpiece_ablation_figure.svg"
ABLATION_PDF_NAME = "cjsi_workpiece_ablation_figure.pdf"
ABLATION_PNG_NAME = "cjsi_workpiece_ablation_figure.png"
CONTOUR_SVG_NAME = "cjsi_workpiece_contour_showcase.svg"
CONTOUR_PDF_NAME = "cjsi_workpiece_contour_showcase.pdf"
CONTOUR_PNG_NAME = "cjsi_workpiece_contour_showcase.png"
COMPARISON_CSV_NAME = "cjsi_workpiece_method_comparison.csv"
STEP_DIAGNOSTICS_CSV_NAME = "cjsi_workpiece_step_diagnostics.csv"
REPORT_MD_NAME = "cjsi_workpiece_analysis_report.md"
FIGURE_LANGUAGES = ("zh", "en")

METHOD_LABELS = {
    "codex_stage2_workpiece": "S2 Baseline",
    "codex_stage3_workpiece": "S3 PriorClamp",
    "codex_stage4_workpiece": "S4 Rescue",
    "codex_stage5_workpiece": "S5 BiasRefine",
    "step9_overlap078": "S9 Overlap0.78",
}
METHOD_ORDER = [
    "codex_stage2_workpiece",
    "codex_stage3_workpiece",
    "codex_stage4_workpiece",
    "codex_stage5_workpiece",
    "step9_overlap078",
]

COLORS = {
    "measured": "#0B5FA5",
    "design": "#1B9E77",
    "profile": "#E66101",
    "normal": "#7570B3",
    "absolute": "#1F78B4",
    "bias": "#A6761D",
    "hist": "#8DA0CB",
    "kde": "#222222",
    "excluded": "#C9CFD7",
    "reference": "#4D4D4D",
    "band": "#E8EEF6",
    "grid": "#E2E6EB",
    "risk": "#CC3311",
    "rescue": "#D95F02",
    "prior": "#6A3D9A",
    "pass": "#1B9E77",
    "warn": "#CC3311",
    "overlap": "#2F4858",
    "summary_face": "#FAFBFC",
    "summary_edge": "#D9DEE5",
    "nominal": "#7A8793",
    "stable": "#222222",
}

BAD_STEP_RMSE_THRESHOLD_PX = 0.25
ERROR_PROFILE_Y_MIN_UM = -90.0
ERROR_PROFILE_Y_MAX_UM = 90.0


@dataclass
class WorkpieceBundle:
    result_dir: Path
    key: str
    label: str
    figure_data: object
    used_df: pd.DataFrame
    display_df: pd.DataFrame
    summary: pd.Series
    stitching_df: Optional[pd.DataFrame]
    quality_df: Optional[pd.DataFrame]
    motion_diag: Optional[Dict[str, float | int | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CJSI-ready analysis package for workpiece measurement results."
    )
    parser.add_argument("--result-dir", type=Path, help="Primary result directory.")
    parser.add_argument("--compare-dirs", nargs="*", type=Path, help="Optional comparison result directories.")
    parser.add_argument("--result-root", type=Path, default=Path("result/workpiece"))
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to the primary result directory.")
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--figure-width-mm", type=float, default=178.0)
    parser.add_argument("--font-size-pt", type=float, default=8.0)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def method_label_from_dir(result_dir: Path) -> str:
    return METHOD_LABELS.get(result_dir.name, result_dir.name)


def has_required_outputs(result_dir: Path) -> bool:
    return (
        (result_dir / "design_error_profile.csv").exists()
        and (result_dir / "design_error_summary.csv").exists()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    candidates = [path for path in result_root.iterdir() if path.is_dir() and has_required_outputs(path)]
    if not candidates:
        raise FileNotFoundError(f"no valid result directory under: {result_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    return args.result_dir if args.result_dir is not None else find_latest_result_dir(args.result_root)


def auto_compare_dirs(primary_dir: Path, result_root: Path) -> List[Path]:
    siblings = {path.name: path for path in result_root.iterdir() if path.is_dir()}
    picked: List[Path] = []
    if primary_dir.name in METHOD_LABELS:
        for name in METHOD_ORDER:
            if name in siblings:
                picked.append(siblings[name])
    if primary_dir not in picked:
        picked.append(primary_dir)
    return picked


def load_bundle(result_dir: Path) -> WorkpieceBundle:
    figure_data = load_figure_data(result_dir)
    used_df = build_used_dataframe(figure_data.profile_df)
    display_df = build_display_dataframe(figure_data.profile_df, figure_data.summary_row)
    summary = figure_data.summary_row.copy()
    stitching_df = figure_data.stitching_df
    quality_path = result_dir / "quality_review.csv"
    quality_df = pd.read_csv(quality_path) if quality_path.exists() else None
    motion_diag = None
    if stitching_df is not None and not stitching_df.empty:
        rmse_col = _find_column(
            stitching_df,
            [
                "NormalRMSEInlier(px)",
                "NormalRMSEAll(px)",
                "normal_rmse",
                "normal_inlier_rmse",
                "normal_rmse_px",
                "nrmse",
            ],
        )
        if rmse_col is not None:
            rmse_px = pd.to_numeric(stitching_df[rmse_col], errors="coerce").to_numpy(dtype=float)
            motion_diag = _collect_motion_diagnostics(stitching_df, rmse_px)
    return WorkpieceBundle(
        result_dir=result_dir,
        key=result_dir.name,
        label=method_label_from_dir(result_dir),
        figure_data=figure_data,
        used_df=used_df,
        display_df=display_df,
        summary=summary,
        stitching_df=stitching_df,
        quality_df=quality_df,
        motion_diag=motion_diag,
    )


def metric(summary: pd.Series, key: str) -> float:
    value = pd.to_numeric(pd.Series([summary.get(key, np.nan)]), errors="coerce").iloc[0]
    return float(value) if math.isfinite(float(value)) else float("nan")


def safe_int(value: float) -> int:
    return int(round(value)) if math.isfinite(value) else 0


def selection_mode_group(mode: str) -> str:
    mode_lower = str(mode).lower()
    if "rescue" in mode_lower:
        return "rescue"
    if "clamp" in mode_lower:
        return "prior"
    if "prior" in mode_lower:
        return "prior"
    return "direct"


def mode_display_label(mode: str) -> str:
    mode_lower = str(mode).lower()
    if "image_correlation_rescue" in mode_lower:
        return "image rescue"
    if "trajectory_prior_clamp" in mode_lower:
        return "prior clamp"
    if "local_prior_rescan" in mode_lower:
        return "prior rescan"
    if "direct_match" in mode_lower:
        return "direct match"
    return str(mode).replace("_", " ")


def load_quality_counts(quality_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    if quality_df is None or quality_df.empty or "status" not in quality_df.columns:
        return {"pass_count": 0, "warn_count": 0, "fail_count": 0}
    status = quality_df["status"].astype(str).str.upper()
    return {
        "pass_count": int((status == "PASS").sum()),
        "warn_count": int((status == "WARN").sum()),
        "fail_count": int((status == "FAIL").sum()),
    }


def compute_step_diagnostics(bundle: WorkpieceBundle) -> pd.DataFrame:
    if bundle.stitching_df is None or bundle.stitching_df.empty:
        return pd.DataFrame()
    df = bundle.stitching_df.copy()
    pixel_size_um = metric(bundle.summary, "pixel_size_mm") * 1000.0
    rmse_col = _find_column(
        df,
        [
            "NormalRMSEInlier(px)",
            "NormalRMSEAll(px)",
            "normal_rmse",
            "normal_inlier_rmse",
            "normal_rmse_px",
            "nrmse",
        ],
    )
    corr_col = _find_column(df, ["TangentCorrInlier", "TangentCorrAll", "tangent_corr_inlier", "tangent_correlation"])
    if rmse_col is None:
        return pd.DataFrame()

    rmse_px = pd.to_numeric(df[rmse_col], errors="coerce").to_numpy(dtype=float)
    rmse_um = rmse_px * pixel_size_um
    steps = pd.to_numeric(df.get("Step", pd.Series(np.arange(1, len(df) + 1))), errors="coerce").fillna(0).astype(int)
    primary = pd.to_numeric(df.get("PrimaryShift(px)", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    actual_overlap = pd.to_numeric(
        df.get("ActualOverlapRatio", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    image_primary = pd.to_numeric(
        df.get("ImageCorrPrimaryShift(px)", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    image_perp = pd.to_numeric(
        df.get("ImageCorrPerpShift(px)", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    tangent_corr = (
        pd.to_numeric(df[corr_col], errors="coerce").to_numpy(dtype=float)
        if corr_col is not None
        else np.full(len(df), np.nan, dtype=float)
    )
    mode_col = _find_column(df, ["SelectionMode", "selection_mode", "mode"])
    mode = df[mode_col].astype(str).to_numpy() if mode_col is not None else np.full(len(df), "direct_match", dtype=object)
    prior_clamped = _prior_clamped_mask(df)
    rescue = _selection_mode_mask(df, "image_correlation_rescue")
    motion_diag = bundle.motion_diag or {}
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    peak_idx = int(np.nanargmax(rmse_px)) if np.isfinite(rmse_px).any() else -1
    bad_mask = np.isfinite(rmse_px) & (rmse_px > BAD_STEP_RMSE_THRESHOLD_PX)

    diagnostics = pd.DataFrame(
        {
            "step": steps.to_numpy(dtype=int),
            "image_a": pd.to_numeric(df.get("ImageA", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float),
            "image_b": pd.to_numeric(df.get("ImageB", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float),
            "selection_mode": mode,
            "mode_group": [selection_mode_group(item) for item in mode],
            "prior_clamped": prior_clamped.astype(int),
            "rescue_step": rescue.astype(int),
            "primary_shift_px": primary,
            "primary_shift_minus_nominal_px": primary - nominal_primary if math.isfinite(nominal_primary) else np.full(len(df), np.nan),
            "primary_shift_minus_stable_px": primary - stable_primary if math.isfinite(stable_primary) else np.full(len(df), np.nan),
            "actual_overlap_ratio": actual_overlap,
            "actual_overlap_minus_nominal_ratio": (
                actual_overlap - nominal_overlap if math.isfinite(nominal_overlap) else np.full(len(df), np.nan)
            ),
            "actual_overlap_minus_stable_ratio": (
                actual_overlap - stable_overlap if math.isfinite(stable_overlap) else np.full(len(df), np.nan)
            ),
            "image_corr_primary_shift_px": image_primary,
            "image_corr_perp_shift_px": image_perp,
            "normal_rmse_px": rmse_px,
            "normal_rmse_um": rmse_um,
            "tangent_corr": tangent_corr,
            "is_peak_rmse_step": np.arange(len(df), dtype=int) == peak_idx,
            "is_bad_step": bad_mask,
        }
    )
    diagnostics["rmse_rank_desc"] = diagnostics["normal_rmse_um"].rank(method="first", ascending=False).astype(int)
    return diagnostics


def compute_method_comparison(bundles: Sequence[WorkpieceBundle]) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for bundle in bundles:
        quality_counts = load_quality_counts(bundle.quality_df)
        step_df = compute_step_diagnostics(bundle)
        highest_step = (
            step_df[step_df["is_peak_rmse_step"]].iloc[0]
            if not step_df.empty and step_df["is_peak_rmse_step"].any()
            else None
        )
        motion_diag = bundle.motion_diag or {}
        rows.append(
            {
                "label": bundle.label,
                "result_dir": bundle.result_dir.name,
                "normal_rmse_um": metric(bundle.summary, "normal_rmse_um"),
                "profile_rms_um": metric(bundle.summary, "profile_rms_um"),
                "profile_p95_abs_um": metric(bundle.summary, "profile_p95_abs_um"),
                "absolute_filtered_rmse_um": metric(bundle.summary, "absolute_filtered_rmse_um"),
                "mean_normal_error_um": metric(bundle.summary, "mean_normal_error_um"),
                "pre_refine_mean_normal_error_um": metric(bundle.summary, "pre_refine_mean_normal_error_um"),
                "pre_refine_absolute_filtered_rmse_um": metric(bundle.summary, "pre_refine_absolute_filtered_rmse_um"),
                "absolute_bias_correction_um": metric(bundle.summary, "absolute_bias_correction_um"),
                "outlier_ratio": metric(bundle.summary, "outlier_ratio"),
                "used_count": metric(bundle.summary, "used_count"),
                "worst_step": int(highest_step["step"]) if highest_step is not None else 0,
                "worst_step_mode": str(highest_step["selection_mode"]) if highest_step is not None else "--",
                "worst_step_normal_rmse_px": float(highest_step["normal_rmse_px"]) if highest_step is not None else float("nan"),
                "worst_step_normal_rmse_um": float(highest_step["normal_rmse_um"]) if highest_step is not None else float("nan"),
                "worst_step_is_bad": int(bool(highest_step["is_bad_step"])) if highest_step is not None else 0,
                "bad_step_count": int(step_df["is_bad_step"].sum()) if not step_df.empty else 0,
                "prior_clamped_step_count": int(step_df["prior_clamped"].sum()) if not step_df.empty else 0,
                "rescue_step_count": int(step_df["rescue_step"].sum()) if not step_df.empty else 0,
                "pass_count": quality_counts["pass_count"],
                "warn_count": quality_counts["warn_count"],
                "fail_count": quality_counts["fail_count"],
                "nominal_primary_shift_px": float(motion_diag.get("nominal_primary_px", np.nan)),
                "stable_primary_shift_px": float(motion_diag.get("stable_primary_px", np.nan)),
                "stable_primary_shift_minus_nominal_px": (
                    float(motion_diag.get("stable_primary_px", np.nan)) - float(motion_diag.get("nominal_primary_px", np.nan))
                    if math.isfinite(float(motion_diag.get("stable_primary_px", np.nan)))
                    and math.isfinite(float(motion_diag.get("nominal_primary_px", np.nan)))
                    else float("nan")
                ),
                "nominal_overlap_ratio": float(motion_diag.get("nominal_overlap_ratio", np.nan)),
                "stable_overlap_ratio": float(motion_diag.get("stable_overlap_ratio", np.nan)),
                "stable_overlap_minus_nominal_ratio": (
                    float(motion_diag.get("stable_overlap_ratio", np.nan)) - float(motion_diag.get("nominal_overlap_ratio", np.nan))
                    if math.isfinite(float(motion_diag.get("stable_overlap_ratio", np.nan)))
                    and math.isfinite(float(motion_diag.get("nominal_overlap_ratio", np.nan)))
                    else float("nan")
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    order_map = {METHOD_LABELS.get(name, name): idx for idx, name in enumerate(METHOD_ORDER)}
    comparison["method_order"] = comparison["label"].map(order_map).fillna(999)
    comparison = comparison.sort_values(["method_order", "label"]).drop(columns=["method_order"]).reset_index(drop=True)
    return comparison


def create_main_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 1.02))
    axes = fig.subplot_mosaic(
        [["A", "A"], ["C", "C"], ["B", "D"]],
        gridspec_kw={"height_ratios": [0.88, 0.88, 0.96], "wspace": 0.28, "hspace": 0.32},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.08)
    return fig, axes


def create_ablation_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 0.86))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.30, "hspace": 0.36})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
    return fig, axes


def create_contour_showcase_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 0.82))
    axes = fig.subplot_mosaic(
        [["A"], ["B"]],
        gridspec_kw={"height_ratios": [0.72, 0.62], "wspace": 0.20, "hspace": 0.34},
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.97, bottom=0.10)
    return fig, axes


def create_concise_figure(width_mm: float) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    width_in = mm_to_inches(width_mm)
    fig = plt.figure(figsize=(width_in, width_in * 1.02))
    axes = fig.subplot_mosaic(
        [["A", "A"], ["C", "C"], ["B", "D"]],
        gridspec_kw={"height_ratios": [0.88, 0.88, 0.96], "wspace": 0.28, "hspace": 0.32},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.08)
    return fig, axes


def mode_color(mode_group: str, is_bad_step: bool) -> str:
    if is_bad_step:
        return COLORS["risk"]
    if mode_group == "rescue":
        return COLORS["rescue"]
    if mode_group == "prior":
        return COLORS["prior"]
    return COLORS["absolute"]


def tr(lang: str, zh_text: str, en_text: str) -> str:
    return zh_text if lang == "zh" else en_text


def localized_output_path(output_dir: Path, filename: str, lang: str) -> Path:
    if lang == "zh":
        return output_dir / filename
    path = Path(filename)
    return output_dir / f"{path.stem}_en{path.suffix}"


def concise_step_color(row: pd.Series) -> str:
    if bool(row.get("is_bad_step", False)):
        return COLORS["risk"]
    if bool(row.get("rescue_step", False)):
        return COLORS["rescue"]
    return COLORS["absolute"]


def draw_panorama_panel(ax: plt.Axes, bundle: WorkpieceBundle) -> None:
    panel_label(ax, "(a)")
    panorama_path = bundle.figure_data.panorama_path
    if panorama_path is None:
        _fallback_text(ax, "Panorama image not available")
        return
    image = plt.imread(str(panorama_path))
    if image.ndim >= 2:
        gray = image[..., :3].mean(axis=-1) if image.ndim == 3 else image.astype(float)
        gray = gray.astype(float)
        if np.nanmax(gray) > 1.5:
            gray = gray / 255.0
        mask = gray < 0.88
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            y0 = max(0, int(np.min(ys)) - 6)
            y1 = min(mask.shape[0], int(np.max(ys)) + 7)
            x0 = max(0, int(np.min(xs)) - 6)
            x1 = min(mask.shape[1], int(np.max(xs)) + 7)
            mask = mask[y0:y1, x0:x1]
            inner = np.zeros_like(mask, dtype=bool)
            inner[1:-1, 1:-1] = (
                mask[1:-1, 1:-1]
                & mask[:-2, 1:-1]
                & mask[2:, 1:-1]
                & mask[1:-1, :-2]
                & mask[1:-1, 2:]
            )
            boundary = mask & ~inner
            stylized = np.ones((mask.shape[0], mask.shape[1], 3), dtype=float)
            stylized[mask] = 0.86
            stylized[boundary] = 0.58
            image = stylized
    _show_image_preserve_ratio(ax, image, cmap="gray")
    return
    motion_diag = bundle.motion_diag or {}
    text_lines = [
        "Rectangular clockwise acquisition",
        f"Used samples: {safe_int(metric(bundle.summary, 'used_count'))}",
        f"Pixel size: {metric(bundle.summary, 'pixel_size_mm') * 1000.0:.3f} μm/px",
    ]
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    if math.isfinite(stable_overlap):
        text_lines.append(f"Stable overlap: {stable_overlap:.4f}")
    ax.text(
        0.02,
        0.93,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.7),
    )


def draw_contour_panel(ax: plt.Axes, bundle: WorkpieceBundle, panel: str = "(b)", lang: str = "zh") -> None:
    panel_label(ax, panel)
    profile_df = bundle.figure_data.profile_df
    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(s_all[excluded], r_all[excluded], s=2.2, color=COLORS["excluded"], linewidths=0, alpha=0.18, rasterized=True)
    ax.plot(
        bundle.display_df["s_aligned_mm"],
        bundle.display_df["r_aligned_mm"],
        color=COLORS["measured"],
        linewidth=1.12,
        label=tr(lang, "实测轮廓", "Measured contour"),
    )
    ax.plot(
        bundle.display_df["s_aligned_mm"],
        bundle.display_df["r_design_display_mm"],
        color=COLORS["design"],
        linewidth=1.02,
        linestyle="--",
        dashes=(5, 2.6),
        label=tr(lang, "设计轮廓", "Design contour"),
    )
    _draw_segment_boundaries(ax, bundle.display_df["s_aligned_mm"].to_numpy(dtype=float))
    apply_metric_axis_ratio(
        ax,
        bundle.display_df["s_aligned_mm"].to_numpy(dtype=float),
        np.r_[bundle.display_df["r_aligned_mm"].to_numpy(dtype=float), bundle.display_df["r_design_display_mm"].to_numpy(dtype=float)],
    )
    ax.set_xlabel(tr(lang, "轴向坐标 s (mm)", "Axial coordinate s (mm)"))
    ax.set_ylabel(tr(lang, "半径 r (mm)", "Radius r (mm)"))
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", handlelength=2.2)
    return
    text = (
        f"dz = {metric(bundle.summary, 'dz_mm'):.3f} mm\n"
        f"dr = {metric(bundle.summary, 'dr_mm'):.3f} mm\n"
        f"dθ = {metric(bundle.summary, 'dtheta_deg'):.3f} deg"
    )
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.7),
    )


def contour_showcase_windows(bundle: WorkpieceBundle, lang: str = "zh") -> List[Tuple[str, float, float, str]]:
    s_values = bundle.display_df["s_aligned_mm"].to_numpy(dtype=float)
    s_min = float(np.nanmin(s_values))
    s_max = float(np.nanmax(s_values))
    default_windows = [
        (
            "B",
            58.0,
            82.0,
            tr(
                lang,
                "典型平滑轮廓区域局部重合结果",
                "Local overlap in a representative smooth-contour region",
            ),
        ),
    ]
    windows: List[Tuple[str, float, float, str]] = []
    for panel, start, end, title in default_windows:
        left = max(s_min, start)
        right = min(s_max, end)
        if right - left < 4.0:
            center = 0.5 * (left + right)
            left = max(s_min, center - 3.0)
            right = min(s_max, center + 3.0)
        windows.append((panel, left, right, title))
    return windows


def draw_contour_showcase_full(
    ax: plt.Axes,
    bundle: WorkpieceBundle,
    windows: Sequence[Tuple[str, float, float, str]],
    lang: str = "zh",
) -> None:
    panel_label(ax, "(a)")
    profile_df = bundle.figure_data.profile_df
    s_all = pd.to_numeric(profile_df["s_aligned_mm"], errors="coerce")
    r_all = pd.to_numeric(profile_df["r_aligned_mm"], errors="coerce")
    used_mask = pd.to_numeric(profile_df["is_used"], errors="coerce").fillna(0).astype(int) == 1
    excluded = (~used_mask) & np.isfinite(s_all) & np.isfinite(r_all)
    if excluded.any():
        ax.scatter(
            s_all[excluded],
            r_all[excluded],
            s=3.2,
            color=COLORS["excluded"],
            linewidths=0,
            alpha=0.34,
            label=tr(lang, "未参与点", "Excluded"),
        )
    ax.plot(
        bundle.display_df["s_aligned_mm"],
        bundle.display_df["r_aligned_mm"],
        color=COLORS["measured"],
        linewidth=1.10,
        label=tr(lang, "实测轮廓", "Measured contour"),
    )
    ax.plot(
        bundle.display_df["s_aligned_mm"],
        bundle.display_df["r_design_display_mm"],
        color=COLORS["design"],
        linewidth=1.0,
        linestyle="--",
        dashes=(6, 3),
        label=tr(lang, "设计轮廓", "Design contour"),
    )
    _draw_segment_boundaries(ax, bundle.display_df["s_aligned_mm"].to_numpy(dtype=float))
    apply_metric_axis_ratio(
        ax,
        bundle.display_df["s_aligned_mm"].to_numpy(dtype=float),
        np.r_[bundle.display_df["r_aligned_mm"].to_numpy(dtype=float), bundle.display_df["r_design_display_mm"].to_numpy(dtype=float)],
    )

    y_min, y_max = ax.get_ylim()
    colors = ["#D98B2B", "#6B7FD7", "#5B8E7D"]
    for idx, (_, left, right, title) in enumerate(windows):
        rect = Rectangle(
            (left, y_min),
            right - left,
            y_max - y_min,
            fill=False,
            edgecolor=colors[idx % len(colors)],
            linewidth=0.9,
            linestyle="--",
            alpha=0.95,
        )
        ax.add_patch(rect)
        title = ""
        ax.text(
            0.5 * (left + right),
            y_max - 0.02 * (y_max - y_min),
            title,
            ha="center",
            va="top",
            fontsize=mpl.rcParams["font.size"] - 0.8,
            color=colors[idx % len(colors)],
            bbox=dict(boxstyle="round,pad=0.15", facecolor=(1, 1, 1, 0.84), edgecolor="none"),
        )
    ax.set_xlabel(tr(lang, "轴向坐标 s (mm)", "Axial coordinate s (mm)"))
    ax.set_ylabel(tr(lang, "半径 r (mm)", "Radius r (mm)"))
    ax.grid(True)
    ax.legend(loc="lower left")
    return
    ax.text(
        0.985,
        0.035,
        "Metric ratio preserved after registration",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.7),
    )


def draw_contour_zoom_panel(ax: plt.Axes, bundle: WorkpieceBundle,
                            panel_label_text: str, s_left: float, s_right: float, title: str,
                            lang: str = "zh") -> None:
    panel_label(ax, panel_label_text)
    display_df = bundle.display_df.copy()
    local_df = display_df[(display_df["s_aligned_mm"] >= s_left) & (display_df["s_aligned_mm"] <= s_right)].copy()
    if local_df.empty:
        _fallback_text(ax, tr(lang, "局部窗口数据缺失", "Window data unavailable"))
        return

    x = local_df["s_aligned_mm"].to_numpy(dtype=float)
    y_measured = local_df["r_aligned_mm"].to_numpy(dtype=float)
    y_design = local_df["r_design_display_mm"].to_numpy(dtype=float)
    ax.plot(x, y_measured, color=COLORS["measured"], linewidth=1.10)
    ax.plot(x, y_design, color=COLORS["design"], linewidth=1.0, linestyle="--", dashes=(6, 3))
    apply_metric_axis_ratio(ax, x, np.r_[y_measured, y_design], x_pad_ratio=0.03, y_pad_ratio=0.12)
    ax.set_title(title, fontsize=mpl.rcParams["font.size"] + 0.1)
    ax.set_xlabel("s (mm)")
    ax.set_ylabel(tr(lang, "r (mm)", "r (mm)"))
    ax.grid(True)
    return
    ax.text(
        0.98,
        0.04,
        f"Peak |e| = {np.nanmax(np.abs(y_error_um)):.2f} μm",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.6),
    )


def draw_stitching_panel(ax: plt.Axes, bundle: WorkpieceBundle, step_df: pd.DataFrame, panel: str = "(c)") -> None:
    panel_label(ax, panel)
    if step_df.empty:
        _fallback_text(ax, "Stitching diagnostics unavailable")
        return

    steps = step_df["step"].to_numpy(dtype=float)
    rmse_um = step_df["normal_rmse_um"].to_numpy(dtype=float)
    overlap = step_df["actual_overlap_ratio"].to_numpy(dtype=float)
    colors = [mode_color(mode, bool(flag)) for mode, flag in zip(step_df["mode_group"], step_df["is_bad_step"])]
    ax.bar(steps, rmse_um, width=0.62, color=colors, edgecolor="#333333", linewidth=0.35, alpha=0.88)
    threshold_um = BAD_STEP_RMSE_THRESHOLD_PX * metric(bundle.summary, "pixel_size_mm") * 1000.0
    ax.axhline(threshold_um, color=COLORS["nominal"], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Stitching step")
    ax.set_ylabel("Inlier normal RMSE (μm)")
    ax.grid(True, axis="y")

    ax2 = ax.twinx()
    ax2.plot(steps, overlap, color=COLORS["overlap"], marker="o", markersize=3.2, linewidth=1.0, label="Actual overlap")
    motion_diag = bundle.motion_diag or {}
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    if math.isfinite(nominal_overlap):
        ax2.axhline(nominal_overlap, color=COLORS["nominal"], linestyle=":", linewidth=0.9)
    if math.isfinite(stable_overlap):
        ax2.axhline(stable_overlap, color=COLORS["stable"], linestyle="-.", linewidth=0.9)
    ax2.set_ylabel("Effective overlap ratio")
    ax2.spines["top"].set_visible(False)

    risk_rows = step_df[step_df["is_risk_step"]]
    if not risk_rows.empty:
        row = risk_rows.iloc[0]
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    risk_primary = float(motion_diag.get("risk_primary_px", np.nan))
    risk_image = float(motion_diag.get("risk_image_primary_px", np.nan))
    risk_perp = float(motion_diag.get("risk_image_perp_px", np.nan))
    lines: List[str] = []
    if math.isfinite(nominal_primary) and math.isfinite(stable_primary):
        lines.append(f"Shift preset {nominal_primary:.1f} px vs stable {stable_primary:.1f} px")
    if math.isfinite(nominal_overlap) and math.isfinite(stable_overlap):
        lines.append(f"Overlap {nominal_overlap:.4f} vs {stable_overlap:.4f}")
    if math.isfinite(risk_image) and math.isfinite(risk_primary):
        risk_line = f"S9 rescue {risk_image:.1f}→{risk_primary:.1f} px"
        if math.isfinite(risk_perp):
            risk_line += f", cross={risk_perp:.1f} px"
        lines.append(risk_line)
    legend_handles = [
        Patch(facecolor=COLORS["absolute"], edgecolor="#333333", linewidth=0.35, label="Direct"),
        Patch(facecolor=COLORS["rescue"], edgecolor="#333333", linewidth=0.35, label="Img rescue"),
        Patch(facecolor=COLORS["prior"], edgecolor="#333333", linewidth=0.35, label="Prior"),
        Line2D([0], [0], color=COLORS["nominal"], linestyle="--", linewidth=0.9, label="Acceptance"),
        Line2D([0], [0], color=COLORS["overlap"], marker="o", markersize=3.2, linewidth=1.0, label="Overlap"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")


def draw_error_profile_panel(ax: plt.Axes, bundle: WorkpieceBundle, panel: str = "(d)", lang: str = "zh") -> None:
    panel_label(ax, panel)
    error_df = bundle.used_df[["index", "s_aligned_mm", "profile_error_um"]].copy()
    for col in ["index", "s_aligned_mm", "profile_error_um"]:
        error_df[col] = pd.to_numeric(error_df[col], errors="coerce")
    error_df = error_df[
        np.isfinite(error_df["s_aligned_mm"]) & np.isfinite(error_df["profile_error_um"])
    ].copy()
    error_df.sort_values("s_aligned_mm", inplace=True)
    s = error_df["s_aligned_mm"].to_numpy(dtype=float)
    source_index = error_df["index"].to_numpy(dtype=float)
    profile_err = error_df["profile_error_um"].to_numpy(dtype=float)
    if len(s) < 10:
        _fallback_text(ax, tr(lang, "误差数据不足", "Insufficient error data"))
        return
    s_smooth, mean_smooth, lower_smooth, upper_smooth = _sliding_stats(
        s, profile_err, window_mm=5.0, segment_ref=source_index
    )
    s, profile_err = _insert_nan_breaks_xy(s, profile_err, segment_ref=source_index)
    ax.fill_between(
        s_smooth, lower_smooth, upper_smooth,
        color=COLORS["band"], alpha=0.22, linewidth=0,
        label=tr(lang, "局部2σ带", "Local 2σ band"),
    )
    ax.plot(s, profile_err, color=COLORS["profile"], linewidth=0.90, alpha=0.92, label=tr(lang, "型面误差", "Profile error"))
    ax.plot(
        s_smooth, mean_smooth, color=COLORS["reference"], linewidth=1.00, linestyle="--", dashes=(4, 2),
        label=tr(lang, "滑动均值", "Moving mean"),
    )
    ax.axhline(0.0, color=COLORS["reference"], linestyle=":", linewidth=0.75)
    _draw_segment_boundaries(ax, s)

    p95 = metric(bundle.summary, "profile_p95_abs_um")
    if math.isfinite(p95):
        ax.axhspan(-p95, p95, color=COLORS["band"], alpha=0.14, linewidth=0)

    ax.set_ylim(ERROR_PROFILE_Y_MIN_UM, ERROR_PROFILE_Y_MAX_UM)
    add_clipped_markers(
        ax,
        s,
        profile_err,
        ERROR_PROFILE_Y_MIN_UM,
        ERROR_PROFILE_Y_MAX_UM,
        COLORS["profile"],
    )
    ax.set_xlabel(tr(lang, "轴向坐标 s (mm)", "Axial coordinate s (mm)"))
    ax.set_ylabel(tr(lang, "误差 (μm)", "Error (μm)"))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", handlelength=2.0)
    return
    ax.text(
        0.98,
        0.04,
        f"Profile RMS = {metric(bundle.summary, 'profile_rms_um'):.3f} μm\nP95 = {p95:.3f} μm",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.7),
    )


def draw_distribution_panel(ax: plt.Axes, bundle: WorkpieceBundle, panel: str = "(e)") -> None:
    panel_label(ax, panel)
    profile_errors = finite_array(bundle.used_df["profile_error_um"])
    if len(profile_errors) < 10:
        _fallback_text(ax, "Insufficient error data")
        return

    mean_val = float(np.mean(profile_errors))
    std_val = float(np.std(profile_errors, ddof=1)) if len(profile_errors) > 1 else 0.0
    median_val = float(np.median(profile_errors))
    p95_val = metric(bundle.summary, "profile_p95_abs_um")
    clip_half = max(24.0, 1.35 * p95_val if math.isfinite(p95_val) else 4.5 * max(std_val, 1.0))
    clip_half = min(60.0, clip_half)
    central_mask = np.abs(profile_errors - median_val) <= clip_half
    plot_values = profile_errors[central_mask] if np.any(central_mask) else profile_errors
    clipped_count = int(len(profile_errors) - len(plot_values))

    n_bins = min(32, max(14, int(math.sqrt(len(plot_values)))))
    ax.hist(
        plot_values,
        bins=n_bins,
        color=COLORS["hist"],
        alpha=0.68,
        edgecolor="white",
        linewidth=0.28,
        density=True,
        label="Histogram",
    )
    if _HAS_SCIPY:
        try:
            kde = sp_stats.gaussian_kde(plot_values)
            x_kde = np.linspace(float(np.min(plot_values)), float(np.max(plot_values)), 400)
            ax.plot(x_kde, kde(x_kde), color=COLORS["kde"], linewidth=1.0, label="Density")
        except Exception:
            pass
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle=":")
    ax.axvline(mean_val, color=COLORS["profile"], linewidth=0.9, linestyle="--", alpha=0.75)
    if math.isfinite(p95_val):
        ax.axvline(+p95_val, color=COLORS["nominal"], linewidth=0.8, linestyle=":")
        ax.axvline(-p95_val, color=COLORS["nominal"], linewidth=0.8, linestyle=":")
    ax.set_xlim(-clip_half, clip_half)
    ax.set_xlabel("Profile error (μm)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", handlelength=2.0)


def draw_local_zoom_pair_panel(ax: plt.Axes, bundle: WorkpieceBundle, panel: str = "(d)", lang: str = "zh") -> None:
    panel_label(ax, panel)
    windows = contour_showcase_windows(bundle, lang=lang)
    if not windows:
        _fallback_text(ax, tr(lang, "局部窗口不可用", "Local window unavailable"))
        return

    _, left, right, title = windows[0]
    local_df = bundle.display_df[
        (bundle.display_df["s_aligned_mm"] >= left) & (bundle.display_df["s_aligned_mm"] <= right)
    ].copy()
    if local_df.empty:
        _fallback_text(ax, tr(lang, "局部窗口数据缺失", "Window data unavailable"))
        return

    x = local_df["s_aligned_mm"].to_numpy(dtype=float)
    y_measured = local_df["r_aligned_mm"].to_numpy(dtype=float)
    y_design = local_df["r_design_display_mm"].to_numpy(dtype=float)
    ax.plot(x, y_measured, color=COLORS["measured"], linewidth=1.08)
    ax.plot(x, y_design, color=COLORS["design"], linewidth=1.0, linestyle="--", dashes=(6, 3))
    apply_metric_axis_ratio(ax, x, np.r_[y_measured, y_design], x_pad_ratio=0.03, y_pad_ratio=0.14)
    ax.set_title(title, fontsize=mpl.rcParams["font.size"] + 0.1)
    ax.set_xlabel("s (mm)")
    ax.set_ylabel(tr(lang, "r (mm)", "r (mm)"))
    ax.grid(True)
    return
    stats_text = (
        f"μ = {mean_val:.2f} μm\n"
        f"median = {median_val:.2f} μm\n"
        f"σ = {std_val:.2f} μm\n"
        f"P95 = {p95_val:.2f} μm\n"
        f"central window = ±{clip_half:.1f} μm\n"
        f"clipped points = {clipped_count}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(1, 1, 1, 0.84), edgecolor=COLORS["summary_edge"], linewidth=0.7),
    )


def add_summary_block(ax: plt.Axes, x: float, y: float, title: str, lines: Sequence[str], color: str) -> None:
    ax.text(
        x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color=color,
    )
    ax.text(
        x,
        y - 0.08,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor=COLORS["summary_face"], edgecolor=COLORS["summary_edge"], linewidth=0.75),
    )


def draw_summary_panel(ax: plt.Axes, bundle: WorkpieceBundle, step_df: pd.DataFrame, panel: str = "(f)") -> None:
    panel_label(ax, panel)
    ax.axis("off")
    motion_diag = bundle.motion_diag or {}
    pre_mean = metric(bundle.summary, "pre_refine_mean_normal_error_um")
    pre_abs = metric(bundle.summary, "pre_refine_absolute_filtered_rmse_um")
    post_mean = metric(bundle.summary, "mean_normal_error_um")
    post_abs = metric(bundle.summary, "absolute_filtered_rmse_um")
    risk_row = step_df[step_df["is_risk_step"]].iloc[0] if not step_df.empty and step_df["is_risk_step"].any() else None
    core_lines = [
        f"Absolute RMSE: {post_abs:.3f} μm",
        f"Profile RMS: {metric(bundle.summary, 'profile_rms_um'):.3f} μm",
        f"Profile P95: {metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm",
        f"Mean normal bias: {post_mean:.3f} μm",
        f"Used points: {safe_int(metric(bundle.summary, 'used_count'))}",
    ]
    refine_lines = []
    if math.isfinite(pre_mean):
        refine_lines.append(f"Bias mean: {pre_mean:.3f} → {post_mean:.3f} μm")
    if math.isfinite(pre_abs):
        refine_lines.append(f"Abs RMSE: {pre_abs:.3f} → {post_abs:.3f} μm")
    bias_correction = metric(bundle.summary, "absolute_bias_correction_um")
    if math.isfinite(bias_correction):
        refine_lines.append(f"Correction: {bias_correction:.3f} μm")
    if not refine_lines:
        refine_lines.append("No post-fit bias refine applied")
    stitch_lines: List[str] = []
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    if math.isfinite(nominal_primary) and math.isfinite(stable_primary):
        stitch_lines.append(f"Shift: {nominal_primary:.1f} → {stable_primary:.1f} px")
    if math.isfinite(nominal_overlap) and math.isfinite(stable_overlap):
        stitch_lines.append(f"Overlap: {nominal_overlap:.4f} → {stable_overlap:.4f}")
    if risk_row is not None:
        stitch_lines.append(f"Risk step S{int(risk_row['step'])}: {mode_display_label(str(risk_row['selection_mode']))}")
        stitch_lines.append(f"Risk-step RMSE: {float(risk_row['normal_rmse_um']):.3f} μm")
    if not stitch_lines:
        stitch_lines.append("Stepwise motion diagnostics unavailable")

    add_summary_block(ax, 0.02, 0.96, "Metrology metrics", core_lines, COLORS["absolute"])
    add_summary_block(ax, 0.52, 0.96, "Bias-refine effect", refine_lines, COLORS["bias"])
    add_summary_block(ax, 0.02, 0.45, "Stitching diagnosis", stitch_lines, COLORS["risk"])

    caution = (
        "Result note:\n"
        "Absolute-error gain is reported\n"
        "after design-referenced bias refine.\n"
        "External traceable reference is still\n"
        "needed for strict absolute accuracy."
    )
    ax.text(
        0.52,
        0.45,
        caution,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["reference"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFF9F5", edgecolor="#E2C6B8", linewidth=0.75),
    )


def draw_stage_metric_panel(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    panel_label(ax, "(a)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    x = np.arange(len(comparison), dtype=float)
    ax.plot(x, comparison["absolute_filtered_rmse_um"], color=COLORS["absolute"], marker="o", linewidth=1.1, label="Absolute RMSE")
    ax.set_ylabel("Absolute RMSE (μm)", color=COLORS["absolute"])
    ax.tick_params(axis="y", colors=COLORS["absolute"])
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=18, ha="right")
    ax.grid(True, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, comparison["profile_rms_um"], color=COLORS["profile"], marker="s", linewidth=1.0, label="Profile RMS")
    ax2.set_ylabel("Profile RMS (μm)", color=COLORS["profile"])
    ax2.tick_params(axis="y", colors=COLORS["profile"])
    ax2.spines["top"].set_visible(False)
    ax.set_title("Absolute consistency and profile-form fidelity")

    handles = [
        Line2D([0], [0], color=COLORS["absolute"], marker="o", linewidth=1.1, label="Absolute RMSE"),
        Line2D([0], [0], color=COLORS["profile"], marker="s", linewidth=1.0, label="Profile RMS"),
    ]
    ax.legend(handles=handles, loc="upper right")


def draw_bias_panel(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    panel_label(ax, "(b)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    x = np.arange(len(comparison), dtype=float)
    bias = comparison["mean_normal_error_um"].to_numpy(dtype=float)
    ax.bar(x, bias, color=COLORS["bias"], edgecolor="#333333", linewidth=0.35, alpha=0.85, width=0.60)
    ax.axhline(0.0, color=COLORS["reference"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=18, ha="right")
    ax.set_ylabel("Mean normal bias (μm)")
    ax.grid(True, axis="y")
    ax.set_title("Bias evolution across processing stages")
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["bias"], edgecolor="#333333", linewidth=0.35, label="Post-refine bias"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COLORS["risk"], markersize=5, linewidth=0, label="Pre-refine bias"),
            Line2D([0], [0], color=COLORS["risk"], linestyle="--", linewidth=0.9, label="Refine shift"),
        ],
        loc="upper right",
    )

    for idx, row in comparison.iterrows():
        pre = float(row.get("pre_refine_mean_normal_error_um", np.nan))
        post = float(row["mean_normal_error_um"])
        if math.isfinite(pre):
            ax.vlines(idx, min(pre, post), max(pre, post), color=COLORS["risk"], linewidth=0.9, linestyle="--", zorder=3)
            ax.scatter(idx, pre, marker="D", s=34, facecolors="white", edgecolors=COLORS["risk"], linewidths=1.0, zorder=4)
            continue
            ax.annotate(
                f"{pre:.1f}→{post:.1f}",
                xy=(idx, post),
                xytext=(0, 12 if post >= 0 else -16),
                textcoords="offset points",
                ha="center",
                va="bottom" if post >= 0 else "top",
                fontsize=mpl.rcParams["font.size"] - 1.0,
                color=COLORS["risk"],
            )

    ax.legend(
        handles=[
            Patch(facecolor=COLORS["bias"], edgecolor="#333333", linewidth=0.35, label="Post-refine bias"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COLORS["risk"], markersize=5, linewidth=0, label="Pre-refine bias"),
            Line2D([0], [0], color=COLORS["risk"], linestyle="--", linewidth=0.9, label="Refine shift"),
        ],
        loc="upper right",
    )


def draw_worst_step_panel(ax: plt.Axes, comparison: pd.DataFrame, pixel_size_um: float) -> None:
    panel_label(ax, "(c)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    ax.text = lambda *args, **kwargs: None
    x = np.arange(len(comparison), dtype=float)
    values = comparison["worst_step_normal_rmse_um"].to_numpy(dtype=float)
    colors = [mode_color(selection_mode_group(mode), False) for mode in comparison["worst_step_mode"]]
    ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.35, alpha=0.88, width=0.60)
    threshold_um = 0.25 * pixel_size_um
    ax.axhline(threshold_um, color=COLORS["nominal"], linestyle="--", linewidth=0.9)
    ax.text(0.01, 0.96, f"Acceptance: 0.25 px ({threshold_um:.2f} μm)", transform=ax.transAxes, ha="left", va="top", color=COLORS["nominal"])
    ax.set_yscale("log")
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size > 0:
        ax.set_ylim(max(0.45, float(np.min(positive_values)) * 0.65), float(np.max(positive_values)) * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=18, ha="right")
    ax.set_ylabel("Worst-step RMSE (μm, log scale)")
    ax.grid(True, axis="y")
    ax.set_title("Worst-step stitching quality")
    ax.text = lambda *args, **kwargs: None
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["absolute"], edgecolor="#333333", linewidth=0.35, label="Direct"),
            Patch(facecolor=COLORS["rescue"], edgecolor="#333333", linewidth=0.35, label="Image rescue"),
            Patch(facecolor=COLORS["prior"], edgecolor="#333333", linewidth=0.35, label="Prior-based"),
            Line2D([0], [0], color=COLORS["nominal"], linestyle="--", linewidth=0.9, label="Acceptance limit"),
        ],
        loc="upper right",
    )
    for idx, row in comparison.iterrows():
        ax.text(
            idx,
            float(row["worst_step_normal_rmse_um"]) * 1.18,
            f"S{int(row['worst_step'])}\n{mode_display_label(str(row['worst_step_mode']))}",
            ha="center",
            va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.3,
            color=COLORS["reference"],
        )


def draw_tradeoff_panel(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    panel_label(ax, "(d)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    ax.annotate = lambda *args, **kwargs: None
    x = comparison["profile_rms_um"].to_numpy(dtype=float)
    y = comparison["absolute_filtered_rmse_um"].to_numpy(dtype=float)
    ax.axvspan(0.0, np.nanmin(x) + 1.5, color="#EEF5F1", alpha=0.55, zorder=0)
    ax.axhspan(0.0, np.nanmin(y) + 15.0, color="#EEF5F1", alpha=0.55, zorder=0)
    for idx, row in comparison.iterrows():
        color = mpl.cm.cividis(0.18 + 0.16 * idx)
        edge = COLORS["risk"] if "S5" in row["label"] else "white"
        ax.scatter(row["profile_rms_um"], row["absolute_filtered_rmse_um"], s=70, color=color, edgecolors=edge, linewidths=1.0, zorder=3)
        ax.annotate(
            row["label"],
            (row["profile_rms_um"], row["absolute_filtered_rmse_um"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=mpl.rcParams["font.size"] - 0.8,
            color=COLORS["reference"],
        )
    ax.set_xlabel("Profile RMS (μm)")
    ax.set_ylabel("Absolute RMSE (μm)")
    ax.set_title("Absolute-form trade-off map")
    ax.grid(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=mpl.cm.cividis(0.18 + 0.16 * idx), markeredgecolor="white", markersize=6, linewidth=0, label=label)
            for idx, label in enumerate(comparison["label"])
        ],
        loc="lower right",
    )
    ax.text = lambda *args, **kwargs: None
    ax.text(0.02, 0.05, "Lower-left is preferred", transform=ax.transAxes, ha="left", va="bottom", color=COLORS["reference"])


def draw_stitching_panel(ax: plt.Axes, bundle: WorkpieceBundle, step_df: pd.DataFrame, panel: str = "(c)", lang: str = "zh") -> None:
    panel_label(ax, panel)
    if step_df.empty:
        _fallback_text(ax, tr(lang, "拼接诊断数据缺失", "Stitching diagnostics unavailable"))
        return

    steps = step_df["step"].to_numpy(dtype=float)
    rmse_um = step_df["normal_rmse_um"].to_numpy(dtype=float)
    overlap = step_df["actual_overlap_ratio"].to_numpy(dtype=float)
    bar_colors = [concise_step_color(row) for _, row in step_df.iterrows()]
    ax.bar(steps, rmse_um, width=0.62, color=bar_colors, edgecolor="none", alpha=0.92)

    threshold_um = BAD_STEP_RMSE_THRESHOLD_PX * metric(bundle.summary, "pixel_size_mm") * 1000.0
    ax.axhline(threshold_um, color=COLORS["nominal"], linestyle="--", linewidth=0.9)
    ax.set_xlabel(tr(lang, "拼接步", "Step"))
    ax.set_ylabel("RMSE (μm)")
    ax.set_xticks(steps)
    ax.grid(True, axis="y")

    ax2 = ax.twinx()
    ax2.plot(steps, overlap, color=COLORS["reference"], marker="o", markersize=2.8, linewidth=0.95)
    motion_diag = bundle.motion_diag or {}
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    if math.isfinite(nominal_overlap):
        ax2.axhline(nominal_overlap, color=COLORS["nominal"], linestyle=":", linewidth=0.8)
    if math.isfinite(stable_overlap):
        ax2.axhline(stable_overlap, color=COLORS["stable"], linestyle="-.", linewidth=0.8)
    ax2.set_ylabel("")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLORS["reference"])
    ax2.yaxis.label.set_color(COLORS["reference"])

    legend_handles = [
        Patch(facecolor=COLORS["absolute"], edgecolor="none", label=tr(lang, "单步RMSE", "Single-step RMSE")),
        Line2D([0], [0], color=COLORS["nominal"], linestyle="--", linewidth=0.9, label=tr(lang, "接受阈值", "Acceptance limit")),
        Line2D([0], [0], color=COLORS["reference"], marker="o", markersize=2.8, linewidth=0.95, label=tr(lang, "重叠率", "Overlap ratio")),
    ]
    if bool(step_df["rescue_step"].any()):
        legend_handles.insert(1, Patch(facecolor=COLORS["rescue"], edgecolor="none", label=tr(lang, "救援步", "Rescued step")))
    if bool(step_df["is_bad_step"].any()):
        legend_handles.insert(2, Patch(facecolor=COLORS["risk"], edgecolor="none", label=tr(lang, "超阈值步", "Exceeding step")))
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=mpl.rcParams["font.size"] - 0.6)


def draw_summary_panel(ax: plt.Axes, bundle: WorkpieceBundle, step_df: pd.DataFrame, panel: str = "(f)") -> None:
    panel_label(ax, panel)
    ax.axis("off")
    motion_diag = bundle.motion_diag or {}
    post_abs = metric(bundle.summary, "absolute_filtered_rmse_um")
    peak_row = step_df[step_df["is_peak_rmse_step"]].iloc[0] if not step_df.empty and step_df["is_peak_rmse_step"].any() else None

    core_lines = [
        f"Absolute RMSE: {post_abs:.3f} μm",
        f"Profile RMS: {metric(bundle.summary, 'profile_rms_um'):.3f} μm",
        f"Profile P95: {metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm",
    ]

    quality_lines: List[str] = [
        f"Used points: {safe_int(metric(bundle.summary, 'used_count'))}",
        f"Outlier ratio: {metric(bundle.summary, 'outlier_ratio') * 100.0:.2f}%",
    ]
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    if math.isfinite(nominal_overlap) and math.isfinite(stable_overlap):
        quality_lines.append(f"Overlap ratio: {stable_overlap:.4f}")
    elif math.isfinite(nominal_primary) and math.isfinite(stable_primary):
        quality_lines.append(f"Shift: {stable_primary:.1f} px")
    if peak_row is not None:
        quality_lines.append(f"Peak-step RMSE: {float(peak_row['normal_rmse_um']):.3f} μm")
    add_summary_block(ax, 0.05, 0.92, "Key metrics", core_lines, COLORS["absolute"])
    add_summary_block(ax, 0.55, 0.92, "Quality summary", quality_lines, COLORS["overlap"])


def draw_worst_step_panel(ax: plt.Axes, comparison: pd.DataFrame, pixel_size_um: float) -> None:
    panel_label(ax, "(c)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    ax.text = lambda *args, **kwargs: None

    x = np.arange(len(comparison), dtype=float)
    values = comparison["worst_step_normal_rmse_um"].to_numpy(dtype=float)
    colors = [mode_color(selection_mode_group(mode), False) for mode in comparison["worst_step_mode"]]
    edgecolors = [COLORS["risk"] if bool(flag) else "#333333" for flag in comparison.get("worst_step_is_bad", pd.Series(0, index=comparison.index))]
    linewidths = [0.85 if bool(flag) else 0.35 for flag in comparison.get("worst_step_is_bad", pd.Series(0, index=comparison.index))]
    bars = ax.bar(x, values, color=colors, edgecolor=edgecolors, linewidth=linewidths, alpha=0.90, width=0.60)
    for bar, is_bad in zip(bars, comparison.get("worst_step_is_bad", pd.Series(0, index=comparison.index))):
        if bool(is_bad):
            bar.set_hatch("//")

    threshold_um = BAD_STEP_RMSE_THRESHOLD_PX * pixel_size_um
    ax.axhline(threshold_um, color=COLORS["nominal"], linestyle="--", linewidth=0.9)
    ax.text(0.01, 0.96, f"Acceptance: {BAD_STEP_RMSE_THRESHOLD_PX:.2f} px ({threshold_um:.2f} μm)", transform=ax.transAxes, ha="left", va="top", color=COLORS["nominal"])
    ax.set_yscale("log")
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size > 0:
        ax.set_ylim(max(0.45, float(np.min(positive_values)) * 0.65), float(np.max(positive_values)) * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=18, ha="right")
    ax.set_ylabel("Largest stepwise RMSE (μm, log scale)")
    ax.set_title("Largest stepwise residual across methods")
    ax.grid(True, axis="y")
    ax.text = lambda *args, **kwargs: None
    legend_handles = [
        Patch(facecolor=COLORS["absolute"], edgecolor="#333333", linewidth=0.35, label="Direct"),
        Patch(facecolor=COLORS["rescue"], edgecolor="#333333", linewidth=0.35, label="Image rescue"),
        Patch(facecolor=COLORS["prior"], edgecolor="#333333", linewidth=0.35, label="Prior-based"),
        Line2D([0], [0], color=COLORS["nominal"], linestyle="--", linewidth=0.9, label="Acceptance limit"),
    ]
    if bool(np.any(comparison.get("worst_step_is_bad", pd.Series(0, index=comparison.index)).astype(bool))):
        legend_handles.append(
            Patch(facecolor="white", edgecolor=COLORS["risk"], linewidth=0.85, hatch="//", label="Exceeding step")
        )
    ax.legend(handles=legend_handles, loc="upper right")
    for idx, row in comparison.iterrows():
        label = f"S{int(row['worst_step'])}\n{mode_display_label(str(row['worst_step_mode']))}"
        if bool(row.get("worst_step_is_bad", 0)):
            label += "\n> limit"
        ax.text(
            idx,
            float(row["worst_step_normal_rmse_um"]) * 1.18,
            label,
            ha="center",
            va="bottom",
            fontsize=mpl.rcParams["font.size"] - 1.3,
            color=COLORS["reference"],
        )


def draw_tradeoff_panel(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    panel_label(ax, "(d)")
    if comparison.empty:
        _fallback_text(ax, "No comparison data")
        return
    ax.annotate = lambda *args, **kwargs: None

    x = comparison["profile_rms_um"].to_numpy(dtype=float)
    y = comparison["absolute_filtered_rmse_um"].to_numpy(dtype=float)
    ax.axvspan(0.0, np.nanmin(x) + 1.5, color="#EEF5F1", alpha=0.55, zorder=0)
    ax.axhspan(0.0, np.nanmin(y) + 15.0, color="#EEF5F1", alpha=0.55, zorder=0)
    palette = [COLORS["absolute"], COLORS["design"], COLORS["profile"], COLORS["prior"], COLORS["bias"]]
    for idx, row in comparison.iterrows():
        color = palette[idx % len(palette)]
        ax.scatter(
            row["profile_rms_um"],
            row["absolute_filtered_rmse_um"],
            s=70,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (row["profile_rms_um"], row["absolute_filtered_rmse_um"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=mpl.rcParams["font.size"] - 0.8,
            color=COLORS["reference"],
        )
    ax.set_xlabel("Profile RMS (μm)")
    ax.set_ylabel("Absolute RMSE (μm)")
    ax.set_title("Absolute-form trade-off map")
    ax.grid(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[idx % len(palette)], markeredgecolor="white", markersize=6, linewidth=0, label=label)
            for idx, label in enumerate(comparison["label"])
        ],
        loc="lower right",
    )
    ax.text = lambda *args, **kwargs: None
    ax.text(0.02, 0.05, "Lower-left is preferred", transform=ax.transAxes, ha="left", va="bottom", color=COLORS["reference"])


def export_figure(fig: plt.Figure, svg_path: Path, pdf_path: Path, png_path: Path, svg_only: bool, pdf_only: bool) -> None:
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    if not svg_only:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if not svg_only and not pdf_only:
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=600)


def build_main_figure(bundle: WorkpieceBundle, step_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace, lang: str = "zh") -> None:
    fig, axes = create_main_figure(args.figure_width_mm)
    draw_contour_panel(axes["A"], bundle, panel="(a)", lang=lang)
    draw_stitching_panel(axes["B"], bundle, step_df, panel="(b)", lang=lang)
    draw_error_profile_panel(axes["C"], bundle, panel="(c)", lang=lang)
    draw_local_zoom_pair_panel(axes["D"], bundle, panel="(d)", lang=lang)
    export_figure(
        fig,
        localized_output_path(output_dir, MAIN_SVG_NAME, lang),
        localized_output_path(output_dir, MAIN_PDF_NAME, lang),
        localized_output_path(output_dir, MAIN_PNG_NAME, lang),
        args.svg_only,
        args.pdf_only,
    )
    export_figure(
        fig,
        localized_output_path(output_dir, CLEAN_MAIN_SVG_NAME, lang),
        localized_output_path(output_dir, CLEAN_MAIN_PDF_NAME, lang),
        localized_output_path(output_dir, CLEAN_MAIN_PNG_NAME, lang),
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def build_concise_figure(bundle: WorkpieceBundle, step_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace, lang: str = "zh") -> None:
    fig, axes = create_concise_figure(args.figure_width_mm)
    draw_contour_panel(axes["A"], bundle, panel="(a)", lang=lang)
    draw_stitching_panel(axes["B"], bundle, step_df, panel="(b)", lang=lang)
    draw_error_profile_panel(axes["C"], bundle, panel="(c)", lang=lang)
    draw_local_zoom_pair_panel(axes["D"], bundle, panel="(d)", lang=lang)
    export_figure(
        fig,
        localized_output_path(output_dir, CONCISE_SVG_NAME, lang),
        localized_output_path(output_dir, CONCISE_PDF_NAME, lang),
        localized_output_path(output_dir, CONCISE_PNG_NAME, lang),
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def build_ablation_figure(comparison: pd.DataFrame, output_dir: Path, args: argparse.Namespace, pixel_size_um: float) -> None:
    fig, axes = create_ablation_figure(args.figure_width_mm)
    draw_stage_metric_panel(axes["A"], comparison)
    draw_bias_panel(axes["B"], comparison)
    draw_worst_step_panel(axes["C"], comparison, pixel_size_um)
    draw_tradeoff_panel(axes["D"], comparison)
    export_figure(
        fig,
        output_dir / ABLATION_SVG_NAME,
        output_dir / ABLATION_PDF_NAME,
        output_dir / ABLATION_PNG_NAME,
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def build_contour_showcase_figure(bundle: WorkpieceBundle, output_dir: Path, args: argparse.Namespace) -> None:
    fig, axes = create_contour_showcase_figure(args.figure_width_mm)
    windows = contour_showcase_windows(bundle, lang="zh")
    draw_contour_showcase_full(axes["A"], bundle, windows, lang="zh")
    if windows:
        _, left, right, title = windows[0]
        draw_contour_zoom_panel(axes["B"], bundle, "(b)", left, right, title, lang="zh")
    export_figure(
        fig,
        output_dir / CONTOUR_SVG_NAME,
        output_dir / CONTOUR_PDF_NAME,
        output_dir / CONTOUR_PNG_NAME,
        args.svg_only,
        args.pdf_only,
    )
    plt.close(fig)


def best_method_text(comparison: pd.DataFrame, column: str, direction: str = "min") -> str:
    finite = comparison[np.isfinite(pd.to_numeric(comparison[column], errors="coerce"))].copy()
    if finite.empty:
        return "--"
    if direction == "max":
        row = finite.loc[pd.to_numeric(finite[column], errors="coerce").idxmax()]
    else:
        row = finite.loc[pd.to_numeric(finite[column], errors="coerce").idxmin()]
    return f"{row['label']} ({float(row[column]):.3f})"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def write_report(output_dir: Path, bundle: WorkpieceBundle, comparison: pd.DataFrame, step_df: pd.DataFrame) -> None:
    comparison_rows: List[List[str]] = []
    for _, row in comparison.iterrows():
        comparison_rows.append(
            [
                str(row["label"]),
                f"{row['absolute_filtered_rmse_um']:.3f}",
                f"{row['profile_rms_um']:.3f}",
                f"{row['mean_normal_error_um']:.3f}",
                f"S{int(row['worst_step'])} / {row['worst_step_mode']}",
            ]
        )

    step_rows: List[List[str]] = []
    if not step_df.empty:
        top_rows = step_df.sort_values(["normal_rmse_um", "step"], ascending=[False, True]).head(5)
        for _, row in top_rows.iterrows():
            step_rows.append(
                [
                    str(int(row["step"])),
                    str(row["selection_mode"]),
                    f"{row['normal_rmse_um']:.3f}",
                    f"{row['primary_shift_px']:.1f}" if math.isfinite(float(row["primary_shift_px"])) else "--",
                    f"{row['actual_overlap_ratio']:.4f}" if math.isfinite(float(row["actual_overlap_ratio"])) else "--",
                ]
            )

    motion_diag = bundle.motion_diag or {}
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    abs_best = best_method_text(comparison, "absolute_filtered_rmse_um")
    form_best = best_method_text(comparison, "profile_rms_um")
    worst_best = best_method_text(comparison, "worst_step_normal_rmse_um")
    pre_abs = metric(bundle.summary, "pre_refine_absolute_filtered_rmse_um")
    pre_mean = metric(bundle.summary, "pre_refine_mean_normal_error_um")

    lines = [
        "# 工件测量投稿分析摘要",
        "",
        "## 1. 当前主结果概述",
        "",
        f"- 主结果目录：`{bundle.result_dir.name}`",
        f"- 绝对误差 RMSE：`{metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm`",
        f"- 型线 RMS：`{metric(bundle.summary, 'profile_rms_um'):.3f} μm`",
        f"- 型线 P95：`{metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm`",
        f"- 法向均值偏差：`{metric(bundle.summary, 'mean_normal_error_um'):.3f} μm`",
        f"- 坏步拼接质量：`{step_df['normal_rmse_um'].max():.3f} μm`（step {int(step_df.loc[step_df['normal_rmse_um'].idxmax(), 'step']) if not step_df.empty else 0}）",
        "",
        "## 2. 阶段演进比较",
        "",
        markdown_table(
            ["方法", "Absolute RMSE/μm", "Profile RMS/μm", "Mean bias/μm", "Worst step"],
            comparison_rows,
        ),
        "",
        f"- 绝对误差最优：{abs_best}",
        f"- 型线误差最优：{form_best}",
        f"- 最差拼接步最优：{worst_best}",
        "",
        "## 3. 坏步与先验偏差诊断",
        "",
        markdown_table(["步号", "选择模式", "RMSE/μm", "Primary shift/px", "Overlap"], step_rows),
        "",
        (
            f"- 设定先验主位移为 `{nominal_primary:.1f} px`，稳定采集主位移约为 `{stable_primary:.1f} px`，"
            f"两者相差 `{stable_primary - nominal_primary:.1f} px`。"
            if math.isfinite(nominal_primary) and math.isfinite(stable_primary)
            else "- 当前结果缺少稳定主位移与设定先验的完整对照。"
        ),
        (
            f"- 设定重叠率为 `{nominal_overlap:.4f}`，稳定采集重叠率约为 `{stable_overlap:.4f}`，"
            f"差值为 `{stable_overlap - nominal_overlap:+.4f}`。"
            if math.isfinite(nominal_overlap) and math.isfinite(stable_overlap)
            else "- 当前结果缺少稳定重叠率与设定先验的完整对照。"
        ),
        "- step 9 仍是主风险步，但已由 `trajectory_prior_clamp` 转为 `image_correlation_rescue`，说明问题更接近真实采集步长/串扰变化，而不是简单的局部匹配噪声。",
        "",
        "## 4. 面向《中国仪器仪表学报》的结果组织建议",
        "",
        "- 主图使用 `cjsi_workpiece_main_figure.*`，强调全景拼接、设计型线对比、逐步拼接质量、型线误差分布与偏差修正效果。",
        "- 正文精简图使用 `cjsi_workpiece_concise_figure.*`，只保留设计型线对比、步间拼接质量、误差曲线与误差分布，更适合正文版面。",
        "- 对比图使用 `cjsi_workpiece_ablation_figure.*`，交代从 baseline、prior clamp、raw-image rescue 到 bias refine 的处理演进。",
        "- 轮廓展示图使用 `cjsi_workpiece_contour_showcase.*`，用于单独展示配准后轮廓比例、台阶过渡和大曲率段局部细节。",
        "- 正文表格优先引用 `cjsi_workpiece_method_comparison.csv` 与 `cjsi_workpiece_step_diagnostics.csv`，减少正文中重复堆数。",
        "- 结果段建议将“型线一致性”和“绝对偏差修正后的一致性”分开表述，避免在缺少外部真值时过度宣称绝对准确度。",
        "",
        "## 5. 可直接用于结果段的表述草案",
        "",
        (
            "在工件矩形顺时针多视场采集条件下，本文对拼接链路和设计型线评定链路进行了联合优化。"
            f"最终工件结果的绝对误差 RMSE 为 {metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm，"
            f"型线 RMS 为 {metric(bundle.summary, 'profile_rms_um'):.3f} μm，"
            f"型线 P95 绝对误差为 {metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm。"
        ),
        "",
        (
            f"与偏差修正前相比，法向均值偏差由 {pre_mean:.3f} μm 降至 {metric(bundle.summary, 'mean_normal_error_um'):.3f} μm，"
            f"绝对误差 RMSE 由 {pre_abs:.3f} μm 降至 {metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm，"
            f"表明基于设计参考的后验偏差修正对工件绝对一致性具有决定性作用。"
            if math.isfinite(pre_mean) and math.isfinite(pre_abs)
            else "当前结果未执行偏差修正或缺少修正前统计量。"
        ),
        "",
        (
            "进一步分析表明，step 9 是主要风险步。虽然其最终仍通过图像相关补救获得可接受的拼接结果，"
            "但稳定采集步长与设定先验之间仍存在系统偏差，说明实际采集过程中的步长/串扰变化是导致先验失配和坏步出现的重要原因。"
        ),
        "",
        "## 6. 投稿表述边界提醒",
        "",
        "- 本结果中的绝对误差改善建立在“设计型线参考 + bias refine”基础上，更适合表述为“设计一致性修正后的绝对偏差结果”。",
        "- 如果论文需要更强的绝对准确度结论，仍建议补充外部可溯源参考或独立标准件对照。",
        "",
    ]
    (output_dir / REPORT_MD_NAME).write_text("\n".join(lines), encoding="utf-8")


def write_report(output_dir: Path, bundle: WorkpieceBundle, comparison: pd.DataFrame, step_df: pd.DataFrame) -> None:
    comparison_rows: List[List[str]] = []
    for _, row in comparison.iterrows():
        peak_label = f"S{int(row['worst_step'])} / {row['worst_step_mode']}"
        if bool(row.get("worst_step_is_bad", 0)):
            peak_label += " / bad"
        comparison_rows.append(
            [
                str(row["label"]),
                f"{row['absolute_filtered_rmse_um']:.3f}",
                f"{row['profile_rms_um']:.3f}",
                f"{row['mean_normal_error_um']:.3f}",
                peak_label,
            ]
        )

    step_rows: List[List[str]] = []
    if not step_df.empty:
        top_rows = step_df.sort_values(["normal_rmse_um", "step"], ascending=[False, True]).head(5)
        for _, row in top_rows.iterrows():
            step_rows.append(
                [
                    str(int(row["step"])),
                    str(row["selection_mode"]),
                    f"{row['normal_rmse_um']:.3f}",
                    f"{row['primary_shift_px']:.1f}" if math.isfinite(float(row["primary_shift_px"])) else "--",
                    f"{row['actual_overlap_ratio']:.4f}" if math.isfinite(float(row["actual_overlap_ratio"])) else "--",
                    "Yes" if bool(row["is_bad_step"]) else "No",
                ]
            )

    motion_diag = bundle.motion_diag or {}
    nominal_primary = float(motion_diag.get("nominal_primary_px", np.nan))
    stable_primary = float(motion_diag.get("stable_primary_px", np.nan))
    nominal_overlap = float(motion_diag.get("nominal_overlap_ratio", np.nan))
    stable_overlap = float(motion_diag.get("stable_overlap_ratio", np.nan))
    peak_row = step_df[step_df["is_peak_rmse_step"]].iloc[0] if not step_df.empty and step_df["is_peak_rmse_step"].any() else None
    bad_rows = step_df[step_df["is_bad_step"]] if not step_df.empty else pd.DataFrame()

    abs_best = best_method_text(comparison, "absolute_filtered_rmse_um")
    form_best = best_method_text(comparison, "profile_rms_um")
    peak_best = best_method_text(comparison, "worst_step_normal_rmse_um")
    pre_abs = metric(bundle.summary, "pre_refine_absolute_filtered_rmse_um")
    pre_mean = metric(bundle.summary, "pre_refine_mean_normal_error_um")

    lines = [
        "# 工件测量投稿分析摘要",
        "",
        "## 1. 当前主结果概述",
        "",
        f"- 主结果目录：`{bundle.result_dir.name}`",
        f"- Absolute RMSE：`{metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm`",
        f"- Profile RMS：`{metric(bundle.summary, 'profile_rms_um'):.3f} μm`",
        f"- Profile P95：`{metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm`",
        f"- Mean normal bias：`{metric(bundle.summary, 'mean_normal_error_um'):.3f} μm`",
        (
            f"- 最高步间 RMSE：`{float(peak_row['normal_rmse_um']):.3f} μm`（Step {int(peak_row['step'])}）"
            if peak_row is not None
            else "- 最高步间 RMSE：`--`"
        ),
        f"- 超阈值坏步数量：`{len(bad_rows)}`（阈值 {BAD_STEP_RMSE_THRESHOLD_PX:.2f} px）",
        "",
        "## 2. 阶段演进比较",
        "",
        markdown_table(
            ["方法", "Absolute RMSE/μm", "Profile RMS/μm", "Mean bias/μm", "Highest-RMSE step"],
            comparison_rows,
        ),
        "",
        f"- Absolute RMSE 最优方法：{abs_best}",
        f"- Profile RMS 最优方法：{form_best}",
        f"- 最高步间 RMSE 最优方法：{peak_best}",
        "",
        "## 3. 步间拼接诊断",
        "",
        markdown_table(["步号", "选择模式", "RMSE/μm", "Primary shift/px", "Overlap", "Bad step"], step_rows),
        "",
        (
            f"- 设定先验主位移为 `{nominal_primary:.1f} px`，稳定采集主位移约为 `{stable_primary:.1f} px`，"
            f"两者相差 `{stable_primary - nominal_primary:.1f} px`。"
            if math.isfinite(nominal_primary) and math.isfinite(stable_primary)
            else "- 当前结果缺少稳定主位移与设定先验的完整对照。"
        ),
        (
            f"- 设定重叠率为 `{nominal_overlap:.4f}`，稳定采集重叠率约为 `{stable_overlap:.4f}`，"
            f"偏差为 `{stable_overlap - nominal_overlap:+.4f}`。"
            if math.isfinite(nominal_overlap) and math.isfinite(stable_overlap)
            else "- 当前结果缺少稳定重叠率与设定先验的完整对照。"
        ),
        (
            f"- 最高 RMSE 步为 Step {int(peak_row['step'])}，模式为 `{peak_row['selection_mode']}`，"
            f"但其 RMSE 仍低于坏步阈值。"
            if peak_row is not None and bad_rows.empty
            else ""
        ),
        (
            f"- 超阈值坏步共 {len(bad_rows)} 个，建议优先复查其采集步长、横向串扰与局部边缘质量。"
            if not bad_rows.empty
            else "- 当前结果未出现超阈值坏步，展示时不建议将最高 RMSE 步按风险步着色。"
        ),
        "",
        "## 4. 面向《中国仪器仪表学报》的图表组织建议",
        "",
        "- 主图使用 `cjsi_workpiece_main_figure.*`，突出全景采集、设计轮廓对比、步间拼接质量和误差统计。",
        "- 正文精简图使用 `cjsi_workpiece_concise_figure.*`，适合在版面受限时替代主图进入正文。",
        "- 对比图使用 `cjsi_workpiece_ablation_figure.*`，交代从基线、先验约束到偏差修正的性能演进。",
        "- 轮廓展示图使用 `cjsi_workpiece_contour_showcase.*`，专门展示配准后比例保持的整体轮廓及局部放大。",
        "- 表格优先引用 `cjsi_workpiece_method_comparison.csv` 和 `cjsi_workpiece_step_diagnostics.csv`，减少正文堆数。",
        "",
        "## 5. 可直接用于结果段的表述草案",
        "",
        (
            "在工件矩形顺时针多视场采集条件下，本文对步间拼接链路与设计型线评定链路进行了联合优化。"
            f"最终结果的 Absolute RMSE 为 {metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm，"
            f"Profile RMS 为 {metric(bundle.summary, 'profile_rms_um'):.3f} μm，"
            f"Profile P95 为 {metric(bundle.summary, 'profile_p95_abs_um'):.3f} μm。"
        ),
        "",
        (
            f"与偏差修正前相比，法向均值偏差由 {pre_mean:.3f} μm 降至 {metric(bundle.summary, 'mean_normal_error_um'):.3f} μm，"
            f"Absolute RMSE 由 {pre_abs:.3f} μm 降至 {metric(bundle.summary, 'absolute_filtered_rmse_um'):.3f} μm，"
            "表明设计参考约束下的后验偏差修正对累计绝对一致性具有决定性作用。"
            if math.isfinite(pre_mean) and math.isfinite(pre_abs)
            else "当前结果缺少偏差修正前的完整统计量，因此正文中更适合强调形貌一致性与拼接稳定性。"
        ),
        "",
        (
            f"最高步间 RMSE 出现在 Step {int(peak_row['step'])}，其数值为 {float(peak_row['normal_rmse_um']):.3f} μm。"
            "该步用于说明采集先验与实际步长存在偏差时的补救能力，而不应直接被表述为坏步。"
            if peak_row is not None and bad_rows.empty
            else "个别超阈值坏步提示采集步长或串扰条件发生突变，需结合原始采集参数进一步排查。"
        ),
        "",
    ]
    (output_dir / REPORT_MD_NAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.font_size_pt, args.svg_fonttype)
    mpl.rcParams.update(
        {
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "axes.titlepad": 5.0,
        }
    )

    primary_dir = resolve_result_dir(args).resolve()
    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or primary_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_dirs = args.compare_dirs if args.compare_dirs else auto_compare_dirs(primary_dir, result_root)
    unique_compare_dirs: List[Path] = []
    for path in compare_dirs:
        resolved = path.resolve()
        if resolved not in unique_compare_dirs:
            unique_compare_dirs.append(resolved)
    if primary_dir not in unique_compare_dirs:
        unique_compare_dirs.insert(0, primary_dir)

    bundles: List[WorkpieceBundle] = []
    for path in unique_compare_dirs:
        bundles.append(load_bundle(path))
    primary_bundle = next((bundle for bundle in bundles if bundle.result_dir == primary_dir), bundles[0])
    if bundles[0].result_dir != primary_dir:
        bundles.remove(primary_bundle)
        bundles.insert(0, primary_bundle)

    comparison = compute_method_comparison(bundles)
    step_df = compute_step_diagnostics(primary_bundle)
    comparison.to_csv(output_dir / COMPARISON_CSV_NAME, index=False, encoding="utf-8-sig")
    step_df.to_csv(output_dir / STEP_DIAGNOSTICS_CSV_NAME, index=False, encoding="utf-8-sig")

    pixel_size_um = metric(primary_bundle.summary, "pixel_size_mm") * 1000.0
    for lang in FIGURE_LANGUAGES:
        build_main_figure(primary_bundle, step_df, output_dir, args, lang=lang)
        build_concise_figure(primary_bundle, step_df, output_dir, args, lang=lang)
    build_ablation_figure(comparison, output_dir, args, pixel_size_um)
    build_contour_showcase_figure(primary_bundle, output_dir, args)
    write_report(output_dir, primary_bundle, comparison, step_df)
    print(f"[OK] CJSI workpiece analysis package exported under: {output_dir}")


if __name__ == "__main__":
    main()
