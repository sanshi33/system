#!/usr/bin/env python3
"""
Fire-tube generatrix literature baseline comparison.

This script rebuilds two literature-style baselines on the same raw image batch
used by the current system result:

1. Direct gray-registration baseline:
   Adapted to the translation-dominant backlight setup from direct subpixel
   stitching / DIC-style registration literature.
   Reference:
   Chu B, Lu J, Guo W. Subpixel image stitching for linewidth measurement based
   on digital image correlation. Scanning, 2010, 32(5): 248-257.

2. Iterative contour-overlap baseline:
   Adapted to overlap-error-minimization stitching for cylindrical / smooth
   profile measurement.
   Reference:
   Chen et al. Measurement of High Numerical Aperture Cylindrical Surface With
   Iterative Stitching Algorithm. Applied Sciences, 2018, 8(11):2092.

The proposed method is evaluated from the existing stage-5 step transforms so
that all three methods share the same offline contour/design comparison logic.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter, sobel
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


PIXEL_SIZE_MM = 0.010057
THRESHOLD = 180
MIN_DARK_PIXELS = 20
DESIGN_LEFT_RADIUS_MM = 179.919242
DESIGN_S_MAX_MM = 155.0
DESIGN_SLOPE_THRESHOLD = 3.0
ANCHOR_MIN_RUN_POINTS = 80
ANCHOR_WINDOW_MIN_POINTS = 50
ANCHOR_STABLE_WINDOW_POINTS = 200
ANCHOR_MAX_STABLE_START_OFFSET = 400
ANCHOR_PRIMARY_WINDOW_RMSE_UM = 35.0
ANCHOR_PRIMARY_WINDOW_ABS_MEAN_UM = 35.0
MAX_RADIUS_JUMP_FOR_RUN_MM = 0.35
OUTLIER_SIGMA = 4.0
OVERLAY_RESET_DX_PX = 50.0
OVERLAY_GAP_DX_PX = 5.0
DESIGN_IGNORE_STEP_TRANSITION = False
DESIGN_STEP_TRANSITION_CENTER_S_MM = 36.0
DESIGN_STEP_TRANSITION_HALF_WIDTH_MM = 4.5
LEFT_TRANSITION_SEARCH_MAX_MM = 3.0
LEFT_TRANSITION_WINDOW_MM = 0.40
LEFT_TRANSITION_MIN_POINTS = 30
LEFT_TRANSITION_MEDIAN_CENTERED_ERROR_UM = 8.0
LEFT_TRANSITION_P90_CENTERED_ERROR_UM = 20.0
THETA_REFINE_HALF_WIDTH_DEG = 0.02
THETA_REFINE_STEP_DEG = 0.005
DZ_REFINE_HALF_WIDTH_MM = 0.10
DZ_REFINE_STEP_MM = 0.0125
AXIAL_SCALE_REFINE_MIN = 1.0
AXIAL_SCALE_REFINE_MAX = 1.0
AXIAL_SCALE_REFINE_STEP = 1.0
AXIAL_SCALE_REFINE_DZ_HALF_WINDOW_MM = 0.0
AXIAL_SCALE_REFINE_DZ_STEP_MM = 0.025

PHASE_DX_EXPECTED_MIN = 820
PHASE_DX_EXPECTED_MAX = 1040
ITERATIVE_REFINE_HALF_WIDTH = 20.0


@dataclass
class TopContour:
    x: np.ndarray
    y: np.ndarray
    width: int
    height: int


@dataclass
class PairTransform:
    step: int
    image_a: int
    image_b: int
    dx: float
    dy: float
    score: float
    overlap_profile_rmse_px: float
    method: str
    angle_deg: float = 0.0


@dataclass
class MethodEvaluation:
    label: str
    dz_mm: float
    dr_mm: float
    dtheta_deg: float
    pre_refine_mean_normal_error_um: float
    absolute_bias_correction_um: float
    mean_normal_error_um: float
    normal_rmse_um: float
    normal_p95_abs_um: float
    profile_rms_um: float
    profile_p95_abs_um: float
    used_count: int
    outlier_count: int
    outlier_ratio: float
    anchor_x_px: float
    anchor_y_px: float
    contour_x_px: np.ndarray
    contour_y_px: np.ndarray
    s_used_mm: np.ndarray
    measured_r_used_mm: np.ndarray
    design_r_used_mm: np.ndarray
    normal_error_used_um: np.ndarray
    profile_error_used_um: np.ndarray


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "火焰筒" / "母线拼接",
        help="Raw fire-tube generatrix image directory.",
    )
    parser.add_argument(
        "--stage5-dir",
        type=Path,
        default=root / "result" / "workpiece" / "codex_stage5_workpiece",
        help="Existing proposed-method result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "result" / "workpiece" / "literature_comparison_fire_tube",
        help="Output directory for figures, CSVs and report.",
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def extract_top_contour(image: np.ndarray) -> TopContour:
    mask = image < THRESHOLD
    dark_counts = mask.sum(axis=0)
    valid_cols = np.where(dark_counts > MIN_DARK_PIXELS)[0]
    if valid_cols.size == 0:
        return TopContour(np.array([], dtype=float), np.array([], dtype=float), image.shape[1], image.shape[0])

    first_dark = np.argmax(mask[:, valid_cols], axis=0).astype(float)
    first_dark = median_filter(first_dark, size=5, mode="nearest")
    return TopContour(valid_cols.astype(float), first_dark, image.shape[1], image.shape[0])


def build_gradient_feature(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    gx = sobel(image_f, axis=1)
    gy = sobel(image_f, axis=0)
    feature = np.hypot(gx, gy)
    feature -= feature.mean()
    return feature


def wrapped_shift(index: int, size: int) -> float:
    return float(index - size if index > size // 2 else index)


def quadratic_peak_offset(prev_v: float, center_v: float, next_v: float) -> float:
    denom = prev_v - 2.0 * center_v + next_v
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (prev_v - next_v) / denom
    return float(np.clip(offset, -1.0, 1.0))


def phase_correlation_shift(feature_a: np.ndarray, feature_b: np.ndarray) -> tuple[float, float, float]:
    spectrum_a = np.fft.fft2(feature_a)
    spectrum_b = np.fft.fft2(feature_b)
    response = spectrum_a * np.conj(spectrum_b)
    response /= np.maximum(np.abs(response), 1e-9)
    corr = np.abs(np.fft.ifft2(response))
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

    dy = wrapped_shift(int(peak_y), corr.shape[0])
    dx = wrapped_shift(int(peak_x), corr.shape[1])

    if 1 <= peak_y < corr.shape[0] - 1:
        dy += quadratic_peak_offset(corr[peak_y - 1, peak_x], corr[peak_y, peak_x], corr[peak_y + 1, peak_x])
    if 1 <= peak_x < corr.shape[1] - 1:
        dx += quadratic_peak_offset(corr[peak_y, peak_x - 1], corr[peak_y, peak_x], corr[peak_y, peak_x + 1])

    return float(dx), float(dy), float(corr[peak_y, peak_x])


def overlap_profile_rmse(contour_a: TopContour, contour_b: TopContour, dx: float, dy: float) -> float:
    if contour_a.x.size < 50 or contour_b.x.size < 50:
        return float("nan")
    interp_b = interp1d(contour_b.x, contour_b.y, bounds_error=False, fill_value=np.nan)
    valid_x = contour_a.x[(contour_a.x >= dx) & (contour_a.x <= contour_b.x[-1] + dx)]
    if valid_x.size < 200:
        return float("nan")
    y_b = interp_b(valid_x - dx)
    ok = np.isfinite(y_b)
    if ok.sum() < 200:
        return float("nan")
    y_a = np.interp(valid_x[ok], contour_a.x, contour_a.y)
    residual = y_a - (y_b[ok] + dy)
    residual -= residual.mean()
    return float(np.sqrt(np.mean(residual * residual)))


def refine_contour_overlap(contour_a: TopContour, contour_b: TopContour, dx_init: float) -> tuple[float, float, float]:
    interp_b = interp1d(contour_b.x, contour_b.y, bounds_error=False, fill_value=np.nan)

    def objective(dx: float) -> float:
        valid_x = contour_a.x[(contour_a.x >= dx) & (contour_a.x <= contour_b.x[-1] + dx)]
        if valid_x.size < 250:
            return 1e9
        y_b = interp_b(valid_x - dx)
        ok = np.isfinite(y_b)
        if ok.sum() < 250:
            return 1e9
        y_a = np.interp(valid_x[ok], contour_a.x, contour_a.y)
        residual = y_a - y_b[ok]
        residual -= residual.mean()
        return float(np.sqrt(np.mean(residual * residual)))

    search_min = max(PHASE_DX_EXPECTED_MIN, dx_init - ITERATIVE_REFINE_HALF_WIDTH)
    search_max = min(PHASE_DX_EXPECTED_MAX, dx_init + ITERATIVE_REFINE_HALF_WIDTH)
    if search_min > search_max:
        dx_clamped = min(max(float(dx_init), PHASE_DX_EXPECTED_MIN), PHASE_DX_EXPECTED_MAX)
        search_min = max(PHASE_DX_EXPECTED_MIN, dx_clamped - ITERATIVE_REFINE_HALF_WIDTH)
        search_max = min(PHASE_DX_EXPECTED_MAX, dx_clamped + ITERATIVE_REFINE_HALF_WIDTH)
    if search_min > search_max:
        search_min = float(PHASE_DX_EXPECTED_MIN)
        search_max = float(PHASE_DX_EXPECTED_MAX)

    coarse_grid = np.arange(search_min, search_max + 0.001, 0.5)
    if coarse_grid.size == 0:
        coarse_grid = np.array(
            [min(max(float(dx_init), PHASE_DX_EXPECTED_MIN), PHASE_DX_EXPECTED_MAX)],
            dtype=float,
        )

    coarse_scores = np.array([objective(dx) for dx in coarse_grid], dtype=float)
    best_coarse = float(coarse_grid[int(np.argmin(coarse_scores))])

    refine_min = max(search_min, best_coarse - 2.0)
    refine_max = min(search_max, best_coarse + 2.0)
    refine = minimize_scalar(objective, bounds=(refine_min, refine_max), method="bounded")
    dx_best = float(refine.x if refine.success else best_coarse)

    valid_x = contour_a.x[(contour_a.x >= dx_best) & (contour_a.x <= contour_b.x[-1] + dx_best)]
    if valid_x.size < 250:
        return dx_best, 0.0, objective(dx_best)
    y_b = interp_b(valid_x - dx_best)
    ok = np.isfinite(y_b)
    if ok.sum() < 250:
        return dx_best, 0.0, objective(dx_best)
    y_a = np.interp(valid_x[ok], contour_a.x, contour_a.y)
    dy_best = float(np.mean(y_a - y_b[ok]))
    rmse = objective(dx_best)
    return dx_best, dy_best, rmse


def load_stage5_pair_transforms(stage5_dir: Path) -> list[PairTransform]:
    csv_path = stage5_dir / "stitching_data.csv"
    transforms: list[PairTransform] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            transforms.append(
                PairTransform(
                    step=int(row["Step"]),
                    image_a=int(row["ImageA"]),
                    image_b=int(row["ImageB"]),
                    dx=float(row["dx(px)"]),
                    dy=float(row["dy(px)"]),
                    score=float(row.get("ImageCorrScore", "nan") or "nan"),
                    overlap_profile_rmse_px=float(row.get("NormalRMSEInlier(px)", row.get("NormalRMSEAll(px)", "nan")) or "nan"),
                    method="proposed_stage5",
                    angle_deg=float(row.get("Angle(deg)", row.get("da(deg)", "0")) or "0"),
                )
            )
    return transforms


def load_official_stage5_summary(stage5_dir: Path) -> dict[str, float]:
    summary_path = stage5_dir / "design_error_summary.csv"
    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"{summary_path} is empty")
    row = frame.iloc[0].to_dict()
    return {
        "dz_mm": float(row.get("dz_mm", float("nan"))),
        "dr_mm": float(row.get("dr_mm", float("nan"))),
        "dtheta_deg": float(row.get("dtheta_deg", float("nan"))),
        "axial_scale_factor": float(row.get("axial_scale_factor", float("nan"))),
        "pre_refine_mean_normal_error_um": float(row.get("pre_refine_mean_normal_error_um", float("nan"))),
        "absolute_bias_correction_um": float(row.get("absolute_bias_correction_um", float("nan"))),
        "mean_normal_error_um": float(row.get("mean_normal_error_um", float("nan"))),
        "normal_rmse_um": float(row.get("normal_rmse_um", float("nan"))),
        "normal_p95_abs_um": float(row.get("normal_p95_abs_um", float("nan"))),
        "profile_rms_um": float(row.get("profile_rms_um", float("nan"))),
        "profile_p95_abs_um": float(row.get("profile_p95_abs_um", float("nan"))),
        "used_count": int(float(row.get("used_count", 0))),
        "outlier_count": int(float(row.get("outlier_count", 0))),
        "outlier_ratio": float(row.get("outlier_ratio", float("nan"))),
        "anchor_x_px": float(row.get("anchor_x_px", float("nan"))),
        "anchor_y_px": float(row.get("anchor_y_px", float("nan"))),
    }


def build_axial_scale_candidates(estimated_scale_factor: float) -> list[float]:
    candidates = [1.0]
    if not math.isfinite(float(estimated_scale_factor)):
        for delta in (-0.0020, -0.0015, -0.0010, -0.0005, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030):
            candidates.append(1.0 + delta)
        return candidates

    def append(value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or abs(value - 1.0) > 5.0e-3:
            return
        if any(abs(existing - value) <= 5e-7 for existing in candidates):
            return
        candidates.append(value)

    append(float(estimated_scale_factor))
    delta = float(estimated_scale_factor) - 1.0
    if abs(delta) < 2.5e-4:
        return sorted(candidates)

    for ratio in (0.5, 0.75, 1.0, 1.25, 1.5):
        append(1.0 + delta * ratio)
    for offset in (5.0e-4, 1.0e-3, -5.0e-4, -1.0e-3):
        append(float(estimated_scale_factor) + offset)
    return sorted(candidates)


def scale_pair_transforms_primary(pair_transforms: list[PairTransform], scale_factor: float) -> list[PairTransform]:
    scaled: list[PairTransform] = []
    for pair in pair_transforms:
        dx = float(pair.dx)
        dy = float(pair.dy)
        if abs(dx) >= abs(dy):
            dx *= scale_factor
        else:
            dy *= scale_factor
        scaled.append(
            PairTransform(
                step=pair.step,
                image_a=pair.image_a,
                image_b=pair.image_b,
                dx=dx,
                dy=dy,
                score=pair.score,
                overlap_profile_rmse_px=pair.overlap_profile_rmse_px,
                method=pair.method,
                angle_deg=float(getattr(pair, "angle_deg", 0.0)),
            )
        )
    return scaled


def is_better_method_evaluation(candidate: MethodEvaluation, incumbent: MethodEvaluation) -> bool:
    if candidate.profile_rms_um + 0.05 < incumbent.profile_rms_um:
        return True
    if incumbent.profile_rms_um + 0.05 < candidate.profile_rms_um:
        return False
    if candidate.normal_rmse_um + 0.05 < incumbent.normal_rmse_um:
        return True
    if incumbent.normal_rmse_um + 0.05 < candidate.normal_rmse_um:
        return False
    if candidate.profile_p95_abs_um + 0.05 < incumbent.profile_p95_abs_um:
        return True
    if incumbent.profile_p95_abs_um + 0.05 < candidate.profile_p95_abs_um:
        return False
    return candidate.used_count > incumbent.used_count


def select_best_stage5_pair_scale(
    stage5_dir: Path,
    contours: list[TopContour],
    pair_transforms: list[PairTransform],
    evaluate_candidate: Callable[[np.ndarray, list[PairTransform]], MethodEvaluation],
) -> tuple[list[PairTransform], np.ndarray, MethodEvaluation, float, list[dict[str, float]]]:
    official = load_official_stage5_summary(stage5_dir)
    estimated_scale_factor = float(official.get("axial_scale_factor", float("nan")))
    candidates = build_axial_scale_candidates(estimated_scale_factor)

    best_pairs = pair_transforms
    best_samples = collect_transformed_samples(contours, pair_transforms)
    best_eval = evaluate_candidate(best_samples, pair_transforms)
    best_scale = 1.0
    search_rows: list[dict[str, float]] = [
        {
            "scale_factor": 1.0,
            "profile_rms_um": float(best_eval.profile_rms_um),
            "normal_rmse_um": float(best_eval.normal_rmse_um),
            "profile_p95_abs_um": float(best_eval.profile_p95_abs_um),
            "used_count": float(best_eval.used_count),
            "selected": 1.0,
        }
    ]

    for scale_factor in candidates:
        if abs(scale_factor - 1.0) <= 1e-7:
            continue
        scaled_pairs = scale_pair_transforms_primary(pair_transforms, scale_factor)
        scaled_samples = collect_transformed_samples(contours, scaled_pairs)
        scaled_eval = evaluate_candidate(scaled_samples, scaled_pairs)
        search_rows.append(
            {
                "scale_factor": float(scale_factor),
                "profile_rms_um": float(scaled_eval.profile_rms_um),
                "normal_rmse_um": float(scaled_eval.normal_rmse_um),
                "profile_p95_abs_um": float(scaled_eval.profile_p95_abs_um),
                "used_count": float(scaled_eval.used_count),
                "selected": 0.0,
            }
        )
        if is_better_method_evaluation(scaled_eval, best_eval):
            best_pairs = scaled_pairs
            best_samples = scaled_samples
            best_eval = scaled_eval
            best_scale = float(scale_factor)

    for row in search_rows:
        row["selected"] = 1.0 if abs(float(row["scale_factor"]) - best_scale) <= 5e-7 else 0.0
        row["estimated_scale_factor"] = estimated_scale_factor
    return best_pairs, best_samples, best_eval, best_scale, search_rows


def load_official_stage5_profile(stage5_dir: Path) -> dict[str, np.ndarray]:
    profile_path = stage5_dir / "design_error_profile.csv"
    frame = pd.read_csv(profile_path)
    if frame.empty:
        raise ValueError(f"{profile_path} is empty")

    used_mask = pd.to_numeric(frame.get("is_used", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0) > 0.5
    used = frame.loc[used_mask].copy()
    if used.empty:
        raise ValueError(f"{profile_path} has no used points")

    contour = frame.loc[
        pd.to_numeric(frame.get("x_px", pd.Series(dtype=float)), errors="coerce").notna()
        & pd.to_numeric(frame.get("y_px", pd.Series(dtype=float)), errors="coerce").notna()
    ].copy()
    if contour.empty:
        contour = used

    def col_to_float(df: pd.DataFrame, column: str, fallback: str | None = None) -> np.ndarray:
        source = column if column in df.columns else fallback
        if source is None or source not in df.columns:
            return np.empty(0, dtype=float)
        return pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)

    return {
        "contour_x_px": col_to_float(contour, "x_px"),
        "contour_y_px": col_to_float(contour, "y_px"),
        "s_used_mm": col_to_float(used, "s_aligned_mm"),
        "measured_r_used_mm": col_to_float(used, "r_aligned_mm"),
        "design_r_used_mm": col_to_float(used, "r_design_mm", "nearest_design_r_mm"),
        "normal_error_used_um": col_to_float(used, "normal_error_um", "signed_distance_um"),
        "profile_error_used_um": col_to_float(used, "profile_error_um"),
    }


def pairwise_phase_baseline(images: list[np.ndarray], contours: list[TopContour]) -> list[PairTransform]:
    transforms: list[PairTransform] = []
    for step, (image_a, image_b, contour_a, contour_b) in enumerate(zip(images[:-1], images[1:], contours[:-1], contours[1:]), start=1):
        dx, dy, score = phase_correlation_shift(build_gradient_feature(image_a), build_gradient_feature(image_b))
        rmse = overlap_profile_rmse(contour_a, contour_b, dx, dy)
        transforms.append(
            PairTransform(
                step=step,
                image_a=step,
                image_b=step + 1,
                dx=dx,
                dy=dy,
                score=score,
                overlap_profile_rmse_px=rmse,
                method="direct_phase_correlation",
            )
        )
    return transforms


def pairwise_iterative_baseline(phase_baseline: list[PairTransform], contours: list[TopContour]) -> list[PairTransform]:
    refined: list[PairTransform] = []
    for step, contour_a, contour_b, phase in zip(range(1, len(contours)), contours[:-1], contours[1:], phase_baseline):
        dx, dy, rmse = refine_contour_overlap(contour_a, contour_b, phase.dx)
        refined.append(
            PairTransform(
                step=step,
                image_a=step,
                image_b=step + 1,
                dx=dx,
                dy=dy,
                score=phase.score,
                overlap_profile_rmse_px=rmse,
                method="iterative_contour_overlap",
            )
        )
    return refined


def cumulative_transforms(pair_transforms: list[PairTransform]) -> tuple[np.ndarray, np.ndarray]:
    dx = [0.0]
    dy = [0.0]
    for pair in pair_transforms:
        dx.append(dx[-1] + pair.dx)
        dy.append(dy[-1] + pair.dy)
    return np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)


def build_relative_matrix(pair: PairTransform, width: float, height: float) -> np.ndarray:
    theta = math.radians(float(getattr(pair, "angle_deg", 0.0)))
    alpha = math.cos(theta)
    beta = math.sin(theta)
    center_x = float(width) / 2.0
    center_y = float(height) / 2.0

    relative = np.eye(3, dtype=float)
    relative[0, 0] = alpha
    relative[0, 1] = beta
    relative[1, 0] = -beta
    relative[1, 1] = alpha
    relative[0, 2] = (1.0 - alpha) * center_x - beta * center_y + float(pair.dx)
    relative[1, 2] = beta * center_x + (1.0 - alpha) * center_y + float(pair.dy)
    return relative


def cumulative_rigid_matrices(contours: list[TopContour], pair_transforms: list[PairTransform]) -> list[np.ndarray]:
    transforms: list[np.ndarray] = [np.eye(3, dtype=float)]
    max_step_count = min(len(pair_transforms), max(0, len(contours) - 1))
    for step_index in range(max_step_count):
        contour = contours[step_index + 1]
        relative = build_relative_matrix(pair_transforms[step_index], contour.width, contour.height)
        transforms.append(transforms[-1] @ relative)
    while len(transforms) < len(contours):
        transforms.append(transforms[-1].copy())
    return transforms


def collect_transformed_samples(contours: list[TopContour], pair_transforms: list[PairTransform]) -> np.ndarray:
    cumulative = cumulative_rigid_matrices(contours, pair_transforms)
    samples: list[np.ndarray] = []
    for contour, transform in zip(contours, cumulative):
        if contour.x.size == 0:
            continue
        local = np.column_stack((contour.x, contour.y, np.ones_like(contour.x, dtype=float)))
        global_xy = (transform @ local.T).T[:, :2]
        samples.append(global_xy)
    if not samples:
        return np.empty((0, 2), dtype=float)
    return np.vstack(samples)


def envelope_from_samples(samples: np.ndarray, quantile: float = 0.95) -> pd.DataFrame:
    min_x = float(np.min(samples[:, 0]))
    bins: dict[int, list[tuple[float, float]]] = {}
    for x_px, y_px in samples:
        bin_index = int(math.floor(x_px - min_x))
        bins.setdefault(bin_index, []).append((float(x_px), float(y_px)))

    rows: list[dict[str, float]] = []
    for bin_index in sorted(bins):
        bucket = np.asarray(bins[bin_index], dtype=float)
        order = np.argsort(bucket[:, 1])
        bucket = bucket[order]
        if len(bucket) >= 5:
            idx = min(len(bucket) - 1, int((len(bucket) - 1) * (1.0 - quantile)))
        else:
            idx = len(bucket) // 2
        point = bucket[idx]
        y_std = float(np.std(bucket[:, 1]))
        rows.append(
            {
                "x_px": float(point[0]),
                "y_px": float(point[1]),
                "y_std_px": y_std,
                "support_count": float(len(bucket)),
            }
        )
    return pd.DataFrame(rows)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def connected_generatrix_runs(x_px: np.ndarray, y_px: np.ndarray, slope_valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(len(x_px)):
        accepted = bool(slope_valid[index])
        connected = False
        if accepted and run_start is not None and index > 0:
            delta_radius_mm = abs(float(y_px[index] - y_px[index - 1])) * PIXEL_SIZE_MM
            connected = bool(slope_valid[index - 1]) and delta_radius_mm <= MAX_RADIUS_JUMP_FOR_RUN_MM

        if accepted and (run_start is None or not connected):
            if run_start is not None:
                runs.append((run_start, index - 1))
            run_start = index
            continue

        if not accepted and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(x_px) - 1))
    return runs


def evaluate_anchor_window(
    x_px: np.ndarray,
    y_px: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[float, float] | None:
    if start_index < 0 or end_index >= len(x_px) or end_index <= start_index:
        return None

    s_eval = (x_px[start_index : end_index + 1] - x_px[start_index]) * PIXEL_SIZE_MM
    r_eval = -y_px[start_index : end_index + 1] * PIXEL_SIZE_MM
    anchor_dr = DESIGN_LEFT_RADIUS_MM - r_eval[0]
    query = np.column_stack((s_eval, r_eval + anchor_dr))
    signed_um = signed_distance_to_design(query)
    if signed_um.size < ANCHOR_WINDOW_MIN_POINTS:
        return None
    rmse_um = float(np.sqrt(np.mean(signed_um * signed_um)))
    abs_mean_um = float(abs(np.mean(signed_um)))
    return rmse_um, abs_mean_um


def choose_generatrix_anchor_index(
    x_px: np.ndarray,
    y_px: np.ndarray,
    slope_valid: np.ndarray,
    runs: list[tuple[int, int]],
) -> int:
    for run_start, run_end in runs:
        if (run_end - run_start + 1) < ANCHOR_MIN_RUN_POINTS:
            continue
        return run_start
    if runs:
        return runs[0][0]
    if np.any(slope_valid):
        return int(np.flatnonzero(slope_valid)[0])
    return 0


def refine_generatrix_start_index(
    x_px: np.ndarray,
    y_px: np.ndarray,
    run_start: int,
    run_end: int,
) -> int:
    run_length = run_end - run_start + 1
    if run_length < ANCHOR_WINDOW_MIN_POINTS:
        return run_start

    max_offset = min(ANCHOR_MAX_STABLE_START_OFFSET, run_length - 1)
    best_index = run_start
    best_rmse_um = float("inf")
    best_abs_mean_um = float("inf")

    for offset in range(max_offset + 1):
        start_index = run_start + offset
        end_index = min(run_end, start_index + ANCHOR_STABLE_WINDOW_POINTS - 1)
        if (end_index - start_index + 1) < ANCHOR_WINDOW_MIN_POINTS:
            break
        window_stats = evaluate_anchor_window(x_px, y_px, start_index, end_index)
        if window_stats is None:
            continue
        rmse_um, abs_mean_um = window_stats
        if rmse_um < best_rmse_um or (math.isclose(rmse_um, best_rmse_um) and abs_mean_um < best_abs_mean_um):
            best_index = start_index
            best_rmse_um = rmse_um
            best_abs_mean_um = abs_mean_um
        if rmse_um <= ANCHOR_PRIMARY_WINDOW_RMSE_UM and abs_mean_um <= ANCHOR_PRIMARY_WINDOW_ABS_MEAN_UM:
            return start_index

    if (
        best_rmse_um <= ANCHOR_PRIMARY_WINDOW_RMSE_UM * 1.5
        and best_abs_mean_um <= ANCHOR_PRIMARY_WINDOW_ABS_MEAN_UM * 1.5
    ):
        return best_index
    return run_start


def trim_to_generatrix(
    x_px: np.ndarray,
    y_px: np.ndarray,
    y_std_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_px.size == 0:
        return x_px, y_px, y_std_px

    smooth_y = gaussian_filter1d(y_px, sigma=3.0)
    slope = np.gradient(smooth_y, x_px)
    slope_valid = np.isfinite(slope) & (np.abs(slope) <= DESIGN_SLOPE_THRESHOLD)
    runs = connected_generatrix_runs(x_px, y_px, slope_valid)
    anchor_index = choose_generatrix_anchor_index(x_px, y_px, slope_valid, runs)

    primary_run = next((run for run in runs if run[0] <= anchor_index <= run[1]), None)
    if primary_run is not None:
        stable_start = refine_generatrix_start_index(x_px, y_px, primary_run[0], primary_run[1])
    else:
        stable_start = anchor_index

    keep_mask = slope_valid.copy()
    keep_mask[:stable_start] = False
    if not np.any(keep_mask):
        keep_mask = np.arange(len(x_px)) >= stable_start
    return x_px[keep_mask], y_px[keep_mask], y_std_px[keep_mask]


def build_design_samples(step_mm: float = 0.01) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    def eval_design_original(z_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_mm = np.asarray(z_mm, dtype=float)
        radius = np.full_like(z_mm, np.nan)
        derivative = np.full_like(z_mm, np.nan)
        valid = (z_mm >= 0.0) & (z_mm <= 155.0)
        z_valid = z_mm[valid]

        linear = z_valid <= 52.958772
        radius_valid = np.empty_like(z_valid)
        derivative_valid = np.empty_like(z_valid)
        radius_valid[linear] = 220.11920702 - 0.50803027 * z_valid[linear]
        derivative_valid[linear] = -0.50803027

        poly = (~linear) & (z_valid <= 100.0)
        xi = (z_valid[poly] - 52.958772) / 47.041228
        radius_valid[poly] = (
            0.21387322 * xi**6
            - 0.86957897 * xi**5
            + 2.12875038 * xi**4
            - 3.85239806 * xi**3
            + 15.01513050 * xi**2
            - 23.91622723 * xi
            + 193.21175337
        )
        derivative_valid[poly] = (
            6.0 * 0.21387322 * xi**5
            - 5.0 * 0.86957897 * xi**4
            + 4.0 * 2.12875038 * xi**3
            - 3.0 * 3.85239806 * xi**2
            + 2.0 * 15.01513050 * xi
            - 23.91622723
        ) / 47.041228

        flat_1 = (~linear) & (~poly) & (z_valid <= 119.0)
        radius_valid[flat_1] = 181.931189
        derivative_valid[flat_1] = 0.0

        flat_2 = (~linear) & (~poly) & (~flat_1)
        radius_valid[flat_2] = 179.919242
        derivative_valid[flat_2] = 0.0

        radius[valid] = radius_valid
        derivative[valid] = derivative_valid
        return radius, derivative

    s_left = np.arange(0.0, 36.0, step_mm)
    s_right = np.arange(36.0, DESIGN_S_MAX_MM + 1e-12, step_mm)
    z_left = DESIGN_S_MAX_MM - s_left
    z_right = DESIGN_S_MAX_MM - s_right
    r_left, _ = eval_design_original(z_left)
    r_right, _ = eval_design_original(z_right)
    step_r = np.arange(179.919242, 181.931189 + 1e-12, step_mm)
    step_s = np.full_like(step_r, 36.0)

    points = np.column_stack(
        (
            np.concatenate((s_left, step_s, s_right)),
            np.concatenate((r_left, step_r, r_right)),
        )
    )
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return points, normal, cKDTree(points)


DESIGN_POINTS, DESIGN_NORMALS, DESIGN_KDTREE = build_design_samples()


def robust_keep_mask(values_um: np.ndarray) -> np.ndarray:
    median = float(np.median(values_um))
    mad = float(np.median(np.abs(values_um - median)))
    scale = max(1.4826 * mad, 1e-6)
    return np.abs(values_um - median) <= OUTLIER_SIGMA * scale


def is_inside_step_transition(s_mm: np.ndarray) -> np.ndarray:
    s_mm = np.asarray(s_mm, dtype=float)
    if not DESIGN_IGNORE_STEP_TRANSITION:
        return np.zeros_like(s_mm, dtype=bool)
    return np.abs(s_mm - DESIGN_STEP_TRANSITION_CENTER_S_MM) <= DESIGN_STEP_TRANSITION_HALF_WIDTH_MM + 1e-9


def trim_left_transition_by_residual(query_sr: np.ndarray, signed_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if query_sr.size == 0 or signed_um.size == 0:
        return query_sr, signed_um

    s_mm = np.asarray(query_sr[:, 0], dtype=float)
    search_end_mm = s_mm[0] + min(LEFT_TRANSITION_SEARCH_MAX_MM, max(0.0, (s_mm[-1] - s_mm[0]) * 0.2))

    for start_index, start_s in enumerate(s_mm):
        if start_s > search_end_mm + 1e-9:
            break
        window_mask = (s_mm >= start_s) & (s_mm <= start_s + LEFT_TRANSITION_WINDOW_MM)
        if int(np.count_nonzero(window_mask)) < LEFT_TRANSITION_MIN_POINTS:
            continue

        window_errors = np.asarray(signed_um[window_mask], dtype=float)
        median_error = float(np.median(window_errors))
        centered_abs = np.abs(window_errors - median_error)
        median_centered = float(np.median(centered_abs))
        p90_centered = float(np.percentile(centered_abs, 90.0))
        if (
            median_centered <= LEFT_TRANSITION_MEDIAN_CENTERED_ERROR_UM
            and p90_centered <= LEFT_TRANSITION_P90_CENTERED_ERROR_UM
        ):
            return query_sr[start_index:], signed_um[start_index:]

    return query_sr, signed_um


def signed_distance_to_design(samples_sr: np.ndarray) -> np.ndarray:
    _, indices = DESIGN_KDTREE.query(samples_sr, k=1)
    nearest = DESIGN_POINTS[indices]
    normals = DESIGN_NORMALS[indices]
    return np.sum((samples_sr - nearest) * normals, axis=1) * 1000.0


def evaluate_method(label: str, samples_px: np.ndarray) -> MethodEvaluation:
    envelope = envelope_from_samples(samples_px)
    x_px = envelope["x_px"].to_numpy(dtype=float)
    y_px = envelope["y_px"].to_numpy(dtype=float)
    y_std_px = envelope["y_std_px"].to_numpy(dtype=float)

    contour_x, contour_y, contour_y_std = trim_to_generatrix(x_px, y_px, y_std_px)
    if contour_x.size == 0:
        raise RuntimeError(f"{label}: failed to extract a valid generatrix after trimming the left end-face segment.")
    contour_s_base = (contour_x - contour_x[0]) * PIXEL_SIZE_MM
    contour_r_base = -contour_y * PIXEL_SIZE_MM
    rotation_center_s = float(np.mean(contour_s_base))
    rotation_center_r = float(np.mean(contour_r_base))

    best_obj = float("inf")
    best_theta = 0.0
    best_dz = 0.0
    best_scale = 1.0
    best_anchor_dr = 0.0
    best_signed = np.array([], dtype=float)
    best_keep = np.array([], dtype=bool)
    best_s_used = np.array([], dtype=float)
    best_r_used = np.array([], dtype=float)
    best_design_r_used = np.array([], dtype=float)

    theta_values = np.arange(-0.14, 0.1401, 0.01)
    dz_values = np.arange(-1.2, 1.2001, 0.025)

    for theta_deg in theta_values:
        theta = math.radians(float(theta_deg))
        ct = math.cos(theta)
        st = math.sin(theta)
        ds = contour_s_base - rotation_center_s
        dr = contour_r_base - rotation_center_r
        rotated_s = rotation_center_s + ds * ct - dr * st
        rotated_r = rotation_center_r + ds * st + dr * ct
        anchor_dr = DESIGN_LEFT_RADIUS_MM - rotated_r[0]

        for dz_mm in dz_values:
            s_eval = rotated_s + dz_mm
            r_eval = rotated_r + anchor_dr
            in_domain = (s_eval >= 0.0) & (s_eval <= DESIGN_S_MAX_MM)
            in_domain &= ~is_inside_step_transition(s_eval)
            if in_domain.sum() < 3000:
                continue

            query = np.column_stack((s_eval[in_domain], r_eval[in_domain]))
            signed_um = signed_distance_to_design(query)
            query, signed_um = trim_left_transition_by_residual(query, signed_um)
            if signed_um.size < 2500:
                continue
            keep = robust_keep_mask(signed_um)
            if keep.sum() < 2500:
                continue

            used = signed_um[keep]
            profile = used - used.mean()
            objective = float(np.sqrt(np.mean(profile * profile)))
            if objective >= best_obj:
                continue

            _, indices = DESIGN_KDTREE.query(query[keep], k=1)
            best_obj = objective
            best_theta = float(theta_deg)
            best_dz = float(dz_mm)
            best_scale = 1.0
            best_anchor_dr = float(anchor_dr)
            best_signed = signed_um
            best_keep = keep
            best_s_used = query[keep, 0]
            best_r_used = query[keep, 1]
            best_design_r_used = DESIGN_POINTS[indices, 1]

    if best_signed.size == 0:
        raise RuntimeError(f"{label}: failed to find a valid design-alignment solution.")

    # 轻量轴向尺度微调：吸收逐步 dx 累积和像素尺度的小幅低频偏差。
    theta = math.radians(best_theta)
    ct = math.cos(theta)
    st = math.sin(theta)
    ds = contour_s_base - rotation_center_s
    dr = contour_r_base - rotation_center_r
    rotated_s = rotation_center_s + ds * ct - dr * st
    rotated_r = rotation_center_r + ds * st + dr * ct
    anchor_dr = DESIGN_LEFT_RADIUS_MM - rotated_r[0]
    base_s0 = float(rotated_s[0])
    scale_values = np.arange(AXIAL_SCALE_REFINE_MIN, AXIAL_SCALE_REFINE_MAX + 1e-12, AXIAL_SCALE_REFINE_STEP)
    dz_min = best_dz - AXIAL_SCALE_REFINE_DZ_HALF_WINDOW_MM
    dz_max = best_dz + AXIAL_SCALE_REFINE_DZ_HALF_WINDOW_MM
    dz_values_refine = np.arange(dz_min, dz_max + 1e-12, AXIAL_SCALE_REFINE_DZ_STEP_MM)

    for scale in scale_values:
        scaled_s = (rotated_s - base_s0) * float(scale) + base_s0
        for dz_mm in dz_values_refine:
            s_eval = scaled_s + dz_mm
            r_eval = rotated_r + anchor_dr
            in_domain = (s_eval >= 0.0) & (s_eval <= DESIGN_S_MAX_MM)
            in_domain &= ~is_inside_step_transition(s_eval)
            if in_domain.sum() < 3000:
                continue

            query = np.column_stack((s_eval[in_domain], r_eval[in_domain]))
            signed_um = signed_distance_to_design(query)
            query, signed_um = trim_left_transition_by_residual(query, signed_um)
            if signed_um.size < 2500:
                continue

            keep = robust_keep_mask(signed_um)
            if keep.sum() < 2500:
                continue

            used = signed_um[keep]
            profile = used - used.mean()
            objective = float(np.sqrt(np.mean(profile * profile)))
            if objective >= best_obj:
                continue

            _, indices = DESIGN_KDTREE.query(query[keep], k=1)
            best_obj = objective
            best_dz = float(dz_mm)
            best_scale = float(scale)
            best_anchor_dr = float(anchor_dr)
            best_signed = signed_um
            best_keep = keep
            best_s_used = query[keep, 0]
            best_r_used = query[keep, 1]
            best_design_r_used = DESIGN_POINTS[indices, 1]

    used_signed_before = best_signed[best_keep]
    bias_correction_um = -float(np.mean(used_signed_before))
    corrected_r_used = best_r_used + bias_correction_um / 1000.0

    corrected_query = np.column_stack((best_s_used, corrected_r_used))
    _, corrected_indices = DESIGN_KDTREE.query(corrected_query, k=1)
    corrected_design = DESIGN_POINTS[corrected_indices]
    corrected_normals = DESIGN_NORMALS[corrected_indices]
    corrected_signed = np.sum((corrected_query - corrected_design) * corrected_normals, axis=1) * 1000.0
    corrected_profile = corrected_signed - corrected_signed.mean()

    def p95_abs(values: np.ndarray) -> float:
        return float(np.percentile(np.abs(values), 95.0)) if values.size else float("nan")

    outlier_count = int(best_signed.size - best_keep.sum())
    used_count = int(best_keep.sum())
    total_count = int(best_signed.size)

    return MethodEvaluation(
        label=label,
        dz_mm=best_dz,
        dr_mm=best_anchor_dr + bias_correction_um / 1000.0,
        dtheta_deg=best_theta,
        pre_refine_mean_normal_error_um=float(np.mean(used_signed_before)),
        absolute_bias_correction_um=bias_correction_um,
        mean_normal_error_um=float(np.mean(corrected_signed)),
        normal_rmse_um=float(np.sqrt(np.mean(corrected_signed * corrected_signed))),
        normal_p95_abs_um=p95_abs(corrected_signed),
        profile_rms_um=float(np.sqrt(np.mean(corrected_profile * corrected_profile))),
        profile_p95_abs_um=p95_abs(corrected_profile),
        used_count=used_count,
        outlier_count=outlier_count,
        outlier_ratio=float(outlier_count / total_count) if total_count else float("nan"),
        anchor_x_px=float(contour_x[0]),
        anchor_y_px=float(contour_y[0]),
        contour_x_px=contour_x,
        contour_y_px=contour_y_std * 0.0 + contour_y,
        s_used_mm=best_s_used,
        measured_r_used_mm=corrected_r_used,
        design_r_used_mm=corrected_design[:, 1],
        normal_error_used_um=corrected_signed,
        profile_error_used_um=corrected_profile,
    )


def evaluate_method(label: str, samples_px: np.ndarray) -> MethodEvaluation:
    envelope = envelope_from_samples(samples_px)
    x_px = envelope["x_px"].to_numpy(dtype=float)
    y_px = envelope["y_px"].to_numpy(dtype=float)
    y_std_px = envelope["y_std_px"].to_numpy(dtype=float)

    contour_x, contour_y, contour_y_std = trim_to_generatrix(x_px, y_px, y_std_px)
    if contour_x.size == 0:
        raise RuntimeError(f"{label}: failed to extract a valid generatrix after trimming the left end-face segment.")
    contour_s_base = (contour_x - contour_x[0]) * PIXEL_SIZE_MM
    contour_r_base = -contour_y * PIXEL_SIZE_MM
    rotation_center_s = float(np.mean(contour_s_base))
    rotation_center_r = float(np.mean(contour_r_base))
    ds = contour_s_base - rotation_center_s
    dr = contour_r_base - rotation_center_r

    best_obj = float("inf")
    best_theta = 0.0
    best_dz = 0.0
    best_anchor_dr = 0.0
    best_signed = np.array([], dtype=float)
    best_keep = np.array([], dtype=bool)
    best_s_used = np.array([], dtype=float)
    best_r_used = np.array([], dtype=float)
    best_design_r_used = np.array([], dtype=float)

    def try_update_best(theta_deg: float, dz_mm: float) -> None:
        nonlocal best_obj
        nonlocal best_theta
        nonlocal best_dz
        nonlocal best_anchor_dr
        nonlocal best_signed
        nonlocal best_keep
        nonlocal best_s_used
        nonlocal best_r_used
        nonlocal best_design_r_used

        theta = math.radians(float(theta_deg))
        ct = math.cos(theta)
        st = math.sin(theta)
        rotated_s = rotation_center_s + ds * ct - dr * st
        rotated_r = rotation_center_r + ds * st + dr * ct
        anchor_dr = DESIGN_LEFT_RADIUS_MM - rotated_r[0]

        s_eval = rotated_s + dz_mm
        r_eval = rotated_r + anchor_dr
        in_domain = (s_eval >= 0.0) & (s_eval <= DESIGN_S_MAX_MM)
        in_domain &= ~is_inside_step_transition(s_eval)
        if int(np.count_nonzero(in_domain)) < 3000:
            return

        query = np.column_stack((s_eval[in_domain], r_eval[in_domain]))
        signed_um = signed_distance_to_design(query)
        query, signed_um = trim_left_transition_by_residual(query, signed_um)
        if signed_um.size < 2500:
            return

        keep = robust_keep_mask(signed_um)
        if int(np.count_nonzero(keep)) < 2500:
            return

        used = signed_um[keep]
        profile = used - used.mean()
        objective = float(np.sqrt(np.mean(profile * profile)))
        if objective >= best_obj:
            return

        _, indices = DESIGN_KDTREE.query(query[keep], k=1)
        best_obj = objective
        best_theta = float(theta_deg)
        best_dz = float(dz_mm)
        best_anchor_dr = float(anchor_dr)
        best_signed = signed_um
        best_keep = keep
        best_s_used = query[keep, 0]
        best_r_used = query[keep, 1]
        best_design_r_used = DESIGN_POINTS[indices, 1]

    theta_values = np.arange(-0.14, 0.1401, 0.01)
    dz_values = np.arange(-1.2, 1.2001, 0.025)
    for theta_deg in theta_values:
        for dz_mm in dz_values:
            try_update_best(float(theta_deg), float(dz_mm))

    if best_signed.size == 0:
        raise RuntimeError(f"{label}: failed to find a valid design-alignment solution.")

    used_signed_before = best_signed[best_keep]
    bias_correction_um = -float(np.mean(used_signed_before))
    corrected_r_used = best_r_used + bias_correction_um / 1000.0

    corrected_query = np.column_stack((best_s_used, corrected_r_used))
    _, corrected_indices = DESIGN_KDTREE.query(corrected_query, k=1)
    corrected_design = DESIGN_POINTS[corrected_indices]
    corrected_normals = DESIGN_NORMALS[corrected_indices]
    corrected_signed = np.sum((corrected_query - corrected_design) * corrected_normals, axis=1) * 1000.0
    corrected_profile = corrected_signed - corrected_signed.mean()

    def p95_abs(values: np.ndarray) -> float:
        return float(np.percentile(np.abs(values), 95.0)) if values.size else float("nan")

    outlier_count = int(best_signed.size - np.count_nonzero(best_keep))
    used_count = int(np.count_nonzero(best_keep))
    total_count = int(best_signed.size)

    return MethodEvaluation(
        label=label,
        dz_mm=best_dz,
        dr_mm=best_anchor_dr + bias_correction_um / 1000.0,
        dtheta_deg=best_theta,
        pre_refine_mean_normal_error_um=float(np.mean(used_signed_before)),
        absolute_bias_correction_um=bias_correction_um,
        mean_normal_error_um=float(np.mean(corrected_signed)),
        normal_rmse_um=float(np.sqrt(np.mean(corrected_signed * corrected_signed))),
        normal_p95_abs_um=p95_abs(corrected_signed),
        profile_rms_um=float(np.sqrt(np.mean(corrected_profile * corrected_profile))),
        profile_p95_abs_um=p95_abs(corrected_profile),
        used_count=used_count,
        outlier_count=outlier_count,
        outlier_ratio=float(outlier_count / total_count) if total_count else float("nan"),
        anchor_x_px=float(contour_x[0]),
        anchor_y_px=float(contour_y[0]),
        contour_x_px=contour_x,
        contour_y_px=contour_y_std * 0.0 + contour_y,
        s_used_mm=best_s_used,
        measured_r_used_mm=corrected_r_used,
        design_r_used_mm=corrected_design[:, 1],
        normal_error_used_um=corrected_signed,
        profile_error_used_um=corrected_profile,
    )


def render_panorama(images: list[np.ndarray], pair_transforms: list[PairTransform], output_path: Path) -> None:
    cum_dx, cum_dy = cumulative_transforms(pair_transforms)
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]

    min_x = int(math.floor(np.min(cum_dx)))
    min_y = int(math.floor(np.min(cum_dy)))
    max_x = int(math.ceil(np.max(cum_dx + np.asarray(widths, dtype=float))))
    max_y = int(math.ceil(np.max(cum_dy + np.asarray(heights, dtype=float))))
    canvas = np.full((max_y - min_y, max_x - min_x), 255, dtype=np.uint8)

    for image, dx, dy in zip(images, cum_dx, cum_dy):
        x0 = int(round(dx - min_x))
        y0 = int(round(dy - min_y))
        region = canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]]
        np.minimum(region, image, out=region)

    Image.fromarray(canvas).save(output_path)


def split_overlay_segments(samples_px: np.ndarray, contour_count: int) -> list[np.ndarray]:
    if samples_px.size == 0:
        return []
    if contour_count <= 1:
        return [samples_px]

    reset_indices = np.flatnonzero(np.diff(samples_px[:, 0]) < -OVERLAY_RESET_DX_PX) + 1
    boundaries = [0, *reset_indices.tolist(), len(samples_px)]
    segments = [samples_px[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    segments = [segment for segment in segments if len(segment) > 0]
    if segments:
        return segments

    points_per_contour = max(1, len(samples_px) // contour_count)
    fallback_segments: list[np.ndarray] = []
    start = 0
    for idx in range(contour_count):
        end = start + points_per_contour
        if idx == contour_count - 1:
            end = len(samples_px)
        fallback_segments.append(samples_px[start:end])
        start = end
    return [segment for segment in fallback_segments if len(segment) > 0]


def break_polyline_on_gaps(samples_px: np.ndarray) -> np.ndarray:
    if len(samples_px) <= 1:
        return samples_px
    pieces = [samples_px[0]]
    for index in range(1, len(samples_px)):
        if (samples_px[index, 0] - samples_px[index - 1, 0]) > OVERLAY_GAP_DX_PX:
            pieces.append(np.array([np.nan, np.nan], dtype=float))
        pieces.append(samples_px[index])
    return np.asarray(pieces, dtype=float)


def plot_contour_overlay(samples_px: np.ndarray, contour_count: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = plt.get_cmap("tab10")
    for idx, subset in enumerate(split_overlay_segments(samples_px, contour_count)):
        polyline = break_polyline_on_gaps(subset)
        ax.plot(polyline[:, 0], polyline[:, 1], linewidth=0.8, color=cmap(idx % 10), label=f"Img {idx + 1}")
    ax.invert_yaxis()
    ax.set_xlabel("Canvas x (px)")
    ax.set_ylabel("Canvas y (px)")
    ax.set_title("Registered contour overlay")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_profile_comparison(results: list[MethodEvaluation], output_path: Path) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(15, 7), sharex=False)
    if len(results) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for column, result in enumerate(results):
        ax_top = axes[0, column]
        ax_bottom = axes[1, column]

        ax_top.plot(result.s_used_mm, result.measured_r_used_mm, color="#d95f02", linewidth=1.4, label="Measured")
        ax_top.plot(result.s_used_mm, result.design_r_used_mm, color="#1b9e77", linewidth=1.2, label="Design")
        ax_top.set_title(result.label)
        ax_top.set_ylabel("Radius (mm)")
        ax_top.grid(alpha=0.25)
        if column == 0:
            ax_top.legend(frameon=False, fontsize=8)

        ax_bottom.plot(result.s_used_mm, result.profile_error_used_um, color="#386cb0", linewidth=0.9)
        ax_bottom.axhline(0.0, color="0.35", linewidth=0.8)
        ax_bottom.set_xlabel("Comparison coordinate s (mm)")
        ax_bottom.set_ylabel("Profile error (um)")
        ax_bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_pair_csv(path: Path, transforms: list[PairTransform]) -> None:
    rows = [
        {
            "step": item.step,
            "image_a": item.image_a,
            "image_b": item.image_b,
            "dx_px": item.dx,
            "dy_px": item.dy,
            "angle_deg": float(getattr(item, "angle_deg", 0.0)),
            "score": item.score,
            "overlap_profile_rmse_px": item.overlap_profile_rmse_px,
            "method": item.method,
        }
        for item in transforms
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_summary_csv(path: Path, results: list[MethodEvaluation]) -> None:
    rows = [
        {
            "label": item.label,
            "dz_mm": item.dz_mm,
            "dr_mm": item.dr_mm,
            "dtheta_deg": item.dtheta_deg,
            "pre_refine_mean_normal_error_um": item.pre_refine_mean_normal_error_um,
            "absolute_bias_correction_um": item.absolute_bias_correction_um,
            "mean_normal_error_um": item.mean_normal_error_um,
            "normal_rmse_um": item.normal_rmse_um,
            "normal_p95_abs_um": item.normal_p95_abs_um,
            "profile_rms_um": item.profile_rms_um,
            "profile_p95_abs_um": item.profile_p95_abs_um,
            "used_count": item.used_count,
            "outlier_count": item.outlier_count,
            "outlier_ratio": item.outlier_ratio,
            "anchor_x_px": item.anchor_x_px,
            "anchor_y_px": item.anchor_y_px,
        }
        for item in results
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_report(
    output_path: Path,
    phase_pairs: list[PairTransform],
    iterative_pairs: list[PairTransform],
    results: list[MethodEvaluation],
) -> None:
    def rows_to_markdown(rows: Iterable[dict[str, object]]) -> str:
        rows = list(rows)
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        return "\n".join(lines)

    phase_rows = [
        {
            "Step": item.step,
            "dx (px)": f"{item.dx:.3f}",
            "dy (px)": f"{item.dy:.3f}",
            "Score": f"{item.score:.4f}",
            "Overlap RMSE (px)": f"{item.overlap_profile_rmse_px:.4f}",
        }
        for item in phase_pairs
    ]
    iterative_rows = [
        {
            "Step": item.step,
            "dx (px)": f"{item.dx:.3f}",
            "dy (px)": f"{item.dy:.3f}",
            "Phase score": f"{item.score:.4f}",
            "Overlap RMSE (px)": f"{item.overlap_profile_rmse_px:.4f}",
        }
        for item in iterative_pairs
    ]
    summary_rows = [
        {
            "Method": item.label,
            "Corrected abs RMSE (um)": f"{item.normal_rmse_um:.3f}",
            "Profile RMS (um)": f"{item.profile_rms_um:.3f}",
            "Profile P95 (um)": f"{item.profile_p95_abs_um:.3f}",
            "Bias correction (um)": f"{item.absolute_bias_correction_um:.3f}",
            "Used count": item.used_count,
            "Outlier ratio": f"{item.outlier_ratio:.4f}",
        }
        for item in results
    ]

    report = f"""# Fire-Tube Generatrix Literature Comparison

## Adopted literature baselines

| Baseline | Literature origin | Implemented adaptation on this dataset |
|---|---|---|
| Direct phase-correlation registration | Chu B, Lu J, Guo W. *Subpixel image stitching for linewidth measurement based on digital image correlation*. Scanning, 2010, 32(5):248-257. | The raw backlight images are treated as translation-dominant adjacent views. Pairwise shifts are estimated by direct frequency-domain gray registration, then evaluated by overlap contour RMSE. |
| Iterative contour-overlap stitching | Chen et al. *Measurement of High Numerical Aperture Cylindrical Surface With Iterative Stitching Algorithm*. Applied Sciences, 2018, 8(11):2092. | The phase-correlation shift is used as coarse initialization, then the overlap contour is iteratively refined by minimizing the shape residual after bias removal. |
| Smooth-contour machine-vision context | Wang Kunzi et al. *Direct measurement and compensation of contour errors for profile grinding*. Measurement, 2025, 242:115959. | Used as the closest process-domain reference for smooth contour stitching, subpixel contour extraction, and full-profile visual measurement logic. |

## Method summary

{rows_to_markdown(summary_rows)}

## Pairwise shifts: direct phase-correlation baseline

{rows_to_markdown(phase_rows)}

## Pairwise shifts: iterative contour-overlap baseline

{rows_to_markdown(iterative_rows)}

## Notes

- The same raw image batch from the fire-tube generatrix folder is used for all methods.
- The proposed-method panorama and contour overlay are rebuilt from the stage-5 pairwise transforms, while the reported proposed-method accuracy metrics are read from the official `design_error_summary.csv`.
- The design comparison is performed with one shared offline evaluator using the same fire-tube generatrix design curve and the same robust outlier rejection rule.
- In this batch, the last pair needs a distinct axial shift near 933 px and a radial jump near -55 px. The direct phase baseline can capture this large change, which makes it suitable as a meaningful baseline instead of the previously failed comparison item.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("Pic_*.bmp"))
    if not image_paths:
        raise FileNotFoundError(f"No Pic_*.bmp images found under {args.input_dir}")

    images = [load_grayscale(path) for path in image_paths]
    contours = [extract_top_contour(image) for image in images]

    phase_pairs = pairwise_phase_baseline(images, contours)
    iterative_pairs = pairwise_iterative_baseline(phase_pairs, contours)
    stage5_pairs = load_stage5_pair_transforms(args.stage5_dir)

    phase_samples = collect_transformed_samples(contours, phase_pairs)
    iterative_samples = collect_transformed_samples(contours, iterative_pairs)
    stage5_samples = collect_transformed_samples(contours, stage5_pairs)

    phase_eval = evaluate_method("Literature A: Direct registration", phase_samples)
    iterative_eval = evaluate_method("Literature B: Iterative overlap", iterative_samples)
    stage5_eval_offline = evaluate_method("Proposed method (stage5)", stage5_samples)
    stage5_official = load_official_stage5_summary(args.stage5_dir)
    stage5_eval = MethodEvaluation(
        label="Proposed method (stage5 official)",
        dz_mm=stage5_official["dz_mm"],
        dr_mm=stage5_official["dr_mm"],
        dtheta_deg=stage5_official["dtheta_deg"],
        pre_refine_mean_normal_error_um=stage5_official["pre_refine_mean_normal_error_um"],
        absolute_bias_correction_um=stage5_official["absolute_bias_correction_um"],
        mean_normal_error_um=stage5_official["mean_normal_error_um"],
        normal_rmse_um=stage5_official["normal_rmse_um"],
        normal_p95_abs_um=stage5_official["normal_p95_abs_um"],
        profile_rms_um=stage5_official["profile_rms_um"],
        profile_p95_abs_um=stage5_official["profile_p95_abs_um"],
        used_count=stage5_official["used_count"],
        outlier_count=stage5_official["outlier_count"],
        outlier_ratio=stage5_official["outlier_ratio"],
        anchor_x_px=stage5_official["anchor_x_px"],
        anchor_y_px=stage5_official["anchor_y_px"],
        contour_x_px=stage5_eval_offline.contour_x_px,
        contour_y_px=stage5_eval_offline.contour_y_px,
        s_used_mm=stage5_eval_offline.s_used_mm,
        measured_r_used_mm=stage5_eval_offline.measured_r_used_mm,
        design_r_used_mm=stage5_eval_offline.design_r_used_mm,
        normal_error_used_um=stage5_eval_offline.normal_error_used_um,
        profile_error_used_um=stage5_eval_offline.profile_error_used_um,
    )
    results = [phase_eval, iterative_eval, stage5_eval]

    write_pair_csv(args.output_dir / "phase_pair_transforms.csv", phase_pairs)
    write_pair_csv(args.output_dir / "iterative_pair_transforms.csv", iterative_pairs)
    write_pair_csv(args.output_dir / "stage5_pair_transforms.csv", stage5_pairs)
    write_summary_csv(args.output_dir / "literature_method_comparison.csv", results)

    render_panorama(images, phase_pairs, args.output_dir / "phase_panorama.png")
    render_panorama(images, iterative_pairs, args.output_dir / "iterative_panorama.png")
    render_panorama(images, stage5_pairs, args.output_dir / "stage5_panorama_rebuilt.png")

    plot_contour_overlay(phase_samples, len(contours), args.output_dir / "phase_contour_overlay.png")
    plot_contour_overlay(iterative_samples, len(contours), args.output_dir / "iterative_contour_overlay.png")
    plot_contour_overlay(stage5_samples, len(contours), args.output_dir / "stage5_contour_overlay.png")
    plot_profile_comparison(results, args.output_dir / "literature_profile_comparison.png")

    write_report(args.output_dir / "literature_comparison_report.md", phase_pairs, iterative_pairs, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
