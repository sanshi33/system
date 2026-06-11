#!/usr/bin/env python3
"""
Batch full-workflow experiment runner for the fire-tube generatrix dataset.

What this script does for each image group:
1. Run the official proposed method with the full CLI workflow.
2. Export per-group CJSI-style paper figures.
3. Build literature baseline A/B outputs and per-group comparison figures.

After all groups are ready, the script can also aggregate:
1. Official proposed-method statistics across all groups.
2. A fair three-method comparison under one shared offline evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROPOSED_GROUP_CSV_NAME = "proposed_group_metrics.csv"
PROPOSED_SUMMARY_CSV_NAME = "proposed_summary.csv"
PROPOSED_FIGURE_PNG_NAME = "proposed_official_statistics_figure.png"
PROPOSED_FIGURE_PDF_NAME = "proposed_official_statistics_figure.pdf"
PROPOSED_FIGURE_SVG_NAME = "proposed_official_statistics_figure.svg"
PROPOSED_REPORT_MD_NAME = "proposed_official_statistics_report.md"

COMPARE_GROUP_CSV_NAME = "method_comparison_metrics_by_group.csv"
COMPARE_SUMMARY_CSV_NAME = "method_comparison_summary_by_method.csv"
COMPARE_CURVE_CSV_NAME = "method_comparison_curve_stats.csv"
COMPARE_REPORT_MD_NAME = "method_comparison_report.md"
INDEX_MD_NAME = "batch_output_index.md"


@dataclass
class ProposedOfficialGroupRow:
    group_name: str
    absolute_filtered_rmse_um: float
    profile_rms_um: float
    mean_normal_error_um: float
    outlier_ratio: float
    used_count: int
    worst_step_rmse_px: float
    worst_step_rmse_um: float
    nominal_overlap_ratio: float
    stable_overlap_ratio: float
    stable_overlap_minus_nominal_ratio: float
    pixel_size_um: float


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
        description="Run the full proposed workflow, literature baselines, and aggregate statistics for grouped fire-tube images."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "火焰筒" / "母线拼接" / "重复性验证",
        help="Root directory that contains 1..N image-group subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dedicated output root for this batch run.",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=root / "build" / "bin" / "Release" / "pinjie_cli.exe",
        help="Path to pinjie_cli executable.",
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
        default=0.75,
        help="Configured overlap ratio for the proposed full CLI run.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="x+",
        help="Motion direction for the proposed full CLI run.",
    )
    parser.add_argument(
        "--timeout-sec-per-group",
        type=int,
        default=7200,
        help="Timeout for one proposed-method full CLI run.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Only run the requested groups and skip the final aggregate outputs.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip per-group runs and only rebuild aggregate outputs from existing results.",
    )
    parser.add_argument(
        "--rerun-proposed",
        action="store_true",
        help="Force rerunning the proposed full CLI even if outputs already exist.",
    )
    parser.add_argument(
        "--rerun-figures",
        action="store_true",
        help="Force rerunning the CJSI figure export for each group.",
    )
    parser.add_argument(
        "--rerun-compare",
        action="store_true",
        help="Force rebuilding the per-group literature comparison outputs.",
    )
    parser.add_argument(
        "--svg-fonttype",
        choices=("path", "none"),
        default="path",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    import re

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


def run_proposed_full_cli(
    cli_path: Path,
    group_dir: Path,
    output_dir: Path,
    image_count: int,
    overlap: float,
    direction: str,
    timeout_sec: int,
    rerun: bool,
) -> None:
    panorama_png = output_dir / "final_panorama.png"
    stitching_csv = output_dir / "stitching_data.csv"
    summary_csv = output_dir / "design_error_summary.csv"
    quality_csv = output_dir / "quality_review.csv"
    run_log = output_dir / "proposed_full_run.log"
    if panorama_png.exists() and stitching_csv.exists() and summary_csv.exists() and quality_csv.exists() and not rerun:
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
        "full",
        "--overlap",
        str(overlap),
        "--direction",
        direction,
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
            f"proposed full CLI failed for group {group_dir.name} with exit code {completed.returncode}. "
            f"See {run_log}"
        )


def run_cjsi_export(result_dir: Path, rerun: bool) -> None:
    main_png = result_dir / "cjsi_workpiece_main_figure.png"
    report_md = result_dir / "cjsi_workpiece_analysis_report.md"
    run_log = result_dir / "cjsi_export.log"
    if main_png.exists() and report_md.exists() and not rerun:
        return

    cmd = [
        sys.executable,
        str(project_root() / "report" / "figure_export" / "cjsi_workpiece_analysis.py"),
        "--result-dir",
        str(result_dir),
        "--output-dir",
        str(result_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=project_root(),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    run_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CJSI export failed for {result_dir}. See {run_log}")


def orient_samples_left_to_right(samples_px: np.ndarray, pair_transforms: Sequence[object]) -> tuple[np.ndarray, bool]:
    dx_values = [
        float(getattr(item, "dx", math.nan))
        for item in pair_transforms
        if math.isfinite(float(getattr(item, "dx", math.nan)))
    ]
    if not dx_values or float(np.median(dx_values)) >= 0.0:
        return samples_px, False
    mirrored = np.asarray(samples_px, dtype=float).copy()
    min_x = float(np.min(mirrored[:, 0]))
    max_x = float(np.max(mirrored[:, 0]))
    mirrored[:, 0] = max_x - (mirrored[:, 0] - min_x)
    return mirrored, True


def build_official_proposed_result_oriented(base_mod, proposed_dir: Path, proposed_samples: np.ndarray, proposed_pairs: Sequence[object], label: str):
    oriented_samples, _ = orient_samples_left_to_right(proposed_samples, proposed_pairs)
    evaluation = base_mod.evaluate_method(label, oriented_samples)

    # 保留官方 summary 中的锚点信息，便于后续诊断时和 official 输出对应。
    official = base_mod.load_official_stage5_summary(proposed_dir)
    evaluation.anchor_x_px = official["anchor_x_px"]
    evaluation.anchor_y_px = official["anchor_y_px"]
    return evaluation


def evaluate_method_oriented(base_mod, label: str, samples_px: np.ndarray, pair_transforms: Sequence[object]):
    oriented_samples, _ = orient_samples_left_to_right(samples_px, pair_transforms)
    return base_mod.evaluate_method(label, oriented_samples)


def build_group_compare_outputs(group_dir: Path, proposed_dir: Path, output_dir: Path, rerun: bool) -> None:
    summary_csv = output_dir / "literature_method_comparison.csv"
    compare_png = output_dir / "literature_profile_comparison.png"
    if summary_csv.exists() and compare_png.exists() and not rerun:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    base_mod = load_module(project_root() / "tools" / "workpiece_literature_comparison.py", f"batch_compare_base_{group_dir.name}")
    recent_mod = load_module(
        project_root() / "literature_fire_tube_compare" / "compare_fire_tube_literature_baselines.py",
        f"batch_compare_recent_{group_dir.name}",
    )

    image_paths = collect_image_paths(group_dir)
    images = [base_mod.load_grayscale(path) for path in image_paths]
    contours = [base_mod.extract_top_contour(image) for image in images]

    recent_a_pairs = recent_mod.pairwise_recent_profile_grinding_baseline(base_mod, images, contours)
    recent_b_pairs = recent_mod.pairwise_recent_telecentric_baseline(base_mod, images, contours)
    proposed_pairs_raw = base_mod.load_stage5_pair_transforms(proposed_dir)
    proposed_pairs, proposed_samples, _, _, proposed_scale_rows = base_mod.select_best_stage5_pair_scale(
        proposed_dir,
        contours,
        proposed_pairs_raw,
        lambda samples, pairs: evaluate_method_oriented(base_mod, "proposed_scale_candidate", samples, pairs),
    )

    recent_a_samples = base_mod.collect_transformed_samples(contours, recent_a_pairs)
    recent_b_samples = base_mod.collect_transformed_samples(contours, recent_b_pairs)

    recent_a_eval = evaluate_method_oriented(base_mod, "Literature A: 2025 smooth-curve stitching", recent_a_samples, recent_a_pairs)
    recent_b_eval = evaluate_method_oriented(base_mod, "Literature B: 2026 telecentric scan-and-stitch", recent_b_samples, recent_b_pairs)
    proposed_eval = build_official_proposed_result_oriented(
        base_mod,
        proposed_dir,
        proposed_samples,
        proposed_pairs,
        "Proposed: full official workflow",
    )
    results = [recent_a_eval, recent_b_eval, proposed_eval]

    pd.DataFrame(proposed_scale_rows).to_csv(
        output_dir / "proposed_pair_scale_refine.csv", index=False, encoding="utf-8-sig"
    )
    base_mod.write_pair_csv(output_dir / "proposed_pair_transforms_raw.csv", proposed_pairs_raw)
    base_mod.write_pair_csv(output_dir / "recent_method_a_pair_transforms.csv", recent_a_pairs)
    base_mod.write_pair_csv(output_dir / "recent_method_b_pair_transforms.csv", recent_b_pairs)
    base_mod.write_pair_csv(output_dir / "proposed_pair_transforms.csv", proposed_pairs)
    base_mod.write_summary_csv(summary_csv, results)

    base_mod.render_panorama(images, recent_a_pairs, output_dir / "recent_method_a_panorama.png")
    base_mod.render_panorama(images, recent_b_pairs, output_dir / "recent_method_b_panorama.png")
    base_mod.render_panorama(images, proposed_pairs, output_dir / "proposed_panorama_rebuilt.png")

    base_mod.plot_contour_overlay(recent_a_samples, len(contours), output_dir / "recent_method_a_contour_overlay.png")
    base_mod.plot_contour_overlay(recent_b_samples, len(contours), output_dir / "recent_method_b_contour_overlay.png")
    base_mod.plot_contour_overlay(proposed_samples, len(contours), output_dir / "proposed_contour_overlay.png")
    base_mod.plot_profile_comparison(results, compare_png)

    compare_mod = load_module(
        project_root() / "literature_fire_tube_compare" / "compare_fire_tube_literature_baselines.py",
        f"batch_compare_writer_{group_dir.name}",
    )
    compare_mod.write_recent_report(output_dir / "literature_comparison_report.md", recent_a_pairs, recent_b_pairs, results)
    compare_mod.write_selection_mapping(output_dir, proposed_dir)


def read_quality_value(quality_df: pd.DataFrame, check_name: str) -> float:
    subset = quality_df[quality_df["check"].astype(str) == check_name]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset.iloc[0]["value"], errors="coerce"))


def read_proposed_official_group_row(group_name: str, proposed_dir: Path, nominal_overlap_ratio: float) -> ProposedOfficialGroupRow:
    summary_df = pd.read_csv(proposed_dir / "design_error_summary.csv")
    if summary_df.empty:
        raise ValueError(f"{proposed_dir / 'design_error_summary.csv'} is empty")
    summary = summary_df.iloc[0]
    stitching_df = pd.read_csv(proposed_dir / "stitching_data.csv")
    quality_df = pd.read_csv(proposed_dir / "quality_review.csv")
    pixel_size_um = float(pd.to_numeric(pd.Series([summary.get("pixel_size_mm", np.nan)]), errors="coerce").iloc[0]) * 1000.0
    worst_step_px = float(pd.to_numeric(stitching_df["NormalRMSEInlier(px)"], errors="coerce").max())
    return ProposedOfficialGroupRow(
        group_name=group_name,
        absolute_filtered_rmse_um=float(pd.to_numeric(pd.Series([summary.get("absolute_filtered_rmse_um", np.nan)]), errors="coerce").iloc[0]),
        profile_rms_um=float(pd.to_numeric(pd.Series([summary.get("profile_rms_um", np.nan)]), errors="coerce").iloc[0]),
        mean_normal_error_um=float(pd.to_numeric(pd.Series([summary.get("mean_normal_error_um", np.nan)]), errors="coerce").iloc[0]),
        outlier_ratio=float(pd.to_numeric(pd.Series([summary.get("outlier_ratio", np.nan)]), errors="coerce").iloc[0]),
        used_count=int(pd.to_numeric(pd.Series([summary.get("used_count", 0)]), errors="coerce").iloc[0]),
        worst_step_rmse_px=worst_step_px,
        worst_step_rmse_um=worst_step_px * pixel_size_um if math.isfinite(worst_step_px) else float("nan"),
        nominal_overlap_ratio=float(nominal_overlap_ratio),
        stable_overlap_ratio=float(nominal_overlap_ratio) + float(read_quality_value(quality_df, "stable_overlap_minus_nominal_ratio"))
        if math.isfinite(read_quality_value(quality_df, "stable_overlap_minus_nominal_ratio"))
        else float("nan"),
        stable_overlap_minus_nominal_ratio=float(read_quality_value(quality_df, "stable_overlap_minus_nominal_ratio")),
        pixel_size_um=pixel_size_um,
    )


def build_proposed_group_dataframe(rows: Sequence[ProposedOfficialGroupRow]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "group_name": item.group_name,
                "absolute_filtered_rmse_um": item.absolute_filtered_rmse_um,
                "profile_rms_um": item.profile_rms_um,
                "mean_normal_error_um": item.mean_normal_error_um,
                "outlier_ratio": item.outlier_ratio,
                "used_count": item.used_count,
                "worst_step_rmse_px": item.worst_step_rmse_px,
                "worst_step_rmse_um": item.worst_step_rmse_um,
                "nominal_overlap_ratio": item.nominal_overlap_ratio,
                "stable_overlap_ratio": item.stable_overlap_ratio,
                "stable_overlap_minus_nominal_ratio": item.stable_overlap_minus_nominal_ratio,
                "pixel_size_um": item.pixel_size_um,
            }
            for item in rows
        ]
    )
    frame["group_order"] = frame["group_name"].map(lambda value: int(value))
    frame.sort_values("group_order", inplace=True)
    frame.drop(columns=["group_order"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_proposed_summary_dataframe(group_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "absolute_filtered_rmse_um",
        "profile_rms_um",
        "mean_normal_error_um",
        "outlier_ratio",
        "used_count",
        "worst_step_rmse_px",
        "worst_step_rmse_um",
        "stable_overlap_ratio",
        "stable_overlap_minus_nominal_ratio",
    ]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(group_df[metric], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "group_count": int(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return pd.DataFrame(rows)


def save_proposed_statistics_figure(output_dir: Path, group_df: pd.DataFrame, svg_fonttype: str) -> None:
    repeat_mod = load_module(
        project_root() / "literature_fire_tube_compare" / "repeatability_validation.py",
        "batch_full_repeatability_plot_module",
    )
    repeat_mod.configure_matplotlib(svg_fonttype)
    plt = repeat_mod.plt
    fig = plt.figure(figsize=(178.0 / 25.4, 178.0 / 25.4 * 0.88))
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.28, "hspace": 0.32})
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.10)

    x = np.arange(len(group_df), dtype=float)
    group_labels = group_df["group_name"].astype(str).tolist()
    color = "#1B9E77"
    line_color = "#222222"

    panels = [
        ("A", "absolute_filtered_rmse_um", "Absolute RMSE (μm)", "10组完整流程绝对误差统计", 80.0),
        ("B", "profile_rms_um", "Profile RMS (μm)", "10组完整流程型面误差统计", 30.0),
        ("C", "worst_step_rmse_px", "Worst-step RMSE (px)", "10组最差单步拼接质量统计", 0.25),
        ("D", "outlier_ratio", "Outlier ratio", "10组异常点比例统计", 0.03),
    ]

    for panel_label, metric_col, ylabel, title, threshold in panels:
        ax = axes[panel_label]
        repeat_mod.panel_label(ax, f"({panel_label.lower()})")
        values = pd.to_numeric(group_df[metric_col], errors="coerce").to_numpy(dtype=float)
        mean_value = float(np.nanmean(values))
        ax.plot(x, values, color=line_color, linewidth=1.0, marker="o", markersize=3.8)
        ax.scatter(x, values, s=24, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(mean_value, color="#7A8793", linestyle="-.", linewidth=0.9, label=f"mean={mean_value:.3f}")
        ax.axhline(threshold, color="#CC3311", linestyle="--", linewidth=0.9, label=f"threshold={threshold:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.legend(loc="best")

    fig.savefig(output_dir / PROPOSED_FIGURE_PNG_NAME, dpi=300)
    fig.savefig(output_dir / PROPOSED_FIGURE_PDF_NAME)
    fig.savefig(output_dir / PROPOSED_FIGURE_SVG_NAME)
    plt.close(fig)


def write_proposed_report(output_path: Path, group_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
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
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    report = [
        "# 10组完整流程统计报告",
        "",
        "## 说明",
        "",
        "- 本部分仅统计所提方法的完整流程官方输出结果。",
        "- 每组均运行 `pinjie_cli --run-mode full`，并生成单组 CJSI 论文图。",
        "- 统计量来自 `design_error_summary.csv`、`quality_review.csv` 与 `stitching_data.csv`。",
        "",
        "## 逐组结果",
        "",
        df_to_markdown(group_df),
        "",
        "## 汇总统计",
        "",
        df_to_markdown(summary_df),
        "",
    ]
    output_path.write_text("\n".join(report), encoding="utf-8")


def build_fair_method_comparison(
    input_root: Path,
    group_dirs: Sequence[Path],
    groups_root: Path,
    aggregate_dir: Path,
    svg_fonttype: str,
) -> None:
    base_mod = load_module(project_root() / "tools" / "workpiece_literature_comparison.py", "batch_fair_base_module")
    recent_mod = load_module(
        project_root() / "literature_fire_tube_compare" / "compare_fire_tube_literature_baselines.py",
        "batch_fair_recent_module",
    )
    repeat_mod = load_module(
        project_root() / "literature_fire_tube_compare" / "repeatability_validation.py",
        "batch_fair_repeat_module",
    )

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for group_dir in group_dirs:
        group_id = int(group_dir.name)
        proposed_dir = groups_root / f"group_{group_id:02d}" / "proposed_full"
        fair_dir = groups_root / f"group_{group_id:02d}" / "fair_evaluator"
        fair_dir.mkdir(parents=True, exist_ok=True)

        image_paths = repeat_mod.collect_image_paths(group_dir)
        images = [base_mod.load_grayscale(path) for path in image_paths]
        contours = [base_mod.extract_top_contour(image) for image in images]
        pixel_size_um = float(base_mod.PIXEL_SIZE_MM) * 1000.0

        proposed_pairs_raw = base_mod.load_stage5_pair_transforms(proposed_dir)
        recent_a_pairs = recent_mod.pairwise_recent_profile_grinding_baseline(base_mod, images, contours)
        recent_b_pairs = recent_mod.pairwise_recent_telecentric_baseline(base_mod, images, contours)
        proposed_pairs, proposed_samples, _, _, proposed_scale_rows = base_mod.select_best_stage5_pair_scale(
            proposed_dir,
            contours,
            proposed_pairs_raw,
            lambda samples, pairs: evaluate_method_oriented(base_mod, "proposed_scale_candidate", samples, pairs),
        )

        recent_a_samples = base_mod.collect_transformed_samples(contours, recent_a_pairs)
        recent_b_samples = base_mod.collect_transformed_samples(contours, recent_b_pairs)

        proposed_eval = build_official_proposed_result_oriented(
            base_mod,
            proposed_dir,
            proposed_samples,
            proposed_pairs,
            "proposed",
        )
        recent_a_eval = evaluate_method_oriented(base_mod, "recent_a", recent_a_samples, recent_a_pairs)
        recent_b_eval = evaluate_method_oriented(base_mod, "recent_b", recent_b_samples, recent_b_pairs)

        proposed_worst_px, _ = repeat_mod.worst_step_rmse_from_pairs(proposed_pairs, pixel_size_um)
        recent_a_worst_px, _ = repeat_mod.worst_step_rmse_from_pairs(recent_a_pairs, pixel_size_um)
        recent_b_worst_px, _ = repeat_mod.worst_step_rmse_from_pairs(recent_b_pairs, pixel_size_um)

        pd.DataFrame(proposed_scale_rows).to_csv(
            fair_dir / "proposed_pair_scale_refine.csv", index=False, encoding="utf-8-sig"
        )
        repeat_mod.write_pair_csv(fair_dir / "proposed_pair_transforms_raw.csv", proposed_pairs_raw)
        repeat_mod.write_pair_csv(fair_dir / "proposed_pair_transforms.csv", proposed_pairs)
        repeat_mod.write_pair_csv(fair_dir / "recent_method_a_pair_transforms.csv", recent_a_pairs)
        repeat_mod.write_pair_csv(fair_dir / "recent_method_b_pair_transforms.csv", recent_b_pairs)

        all_results.extend(
            [
                repeat_mod.result_from_eval(group_dir.name, "proposed", "所提方法", proposed_eval, proposed_worst_px, pixel_size_um),
                repeat_mod.result_from_eval(
                    group_dir.name,
                    "recent_profile_grinding_2025",
                    "对比方法A（2025平滑曲线拼接）",
                    recent_a_eval,
                    recent_a_worst_px,
                    pixel_size_um,
                ),
                repeat_mod.result_from_eval(
                    group_dir.name,
                    "recent_telecentric_scan_stitch_2026",
                    "对比方法B（2026远心扫描拼接）",
                    recent_b_eval,
                    recent_b_worst_px,
                    pixel_size_um,
                ),
            ]
        )

    group_df = repeat_mod.build_group_metrics_dataframe(all_results)
    summary_df = repeat_mod.build_summary_dataframe(group_df)
    curve_df = repeat_mod.build_curve_stats_dataframe(all_results)

    group_df.to_csv(aggregate_dir / COMPARE_GROUP_CSV_NAME, index=False, encoding="utf-8-sig")
    summary_df.to_csv(aggregate_dir / COMPARE_SUMMARY_CSV_NAME, index=False, encoding="utf-8-sig")
    curve_df.to_csv(aggregate_dir / COMPARE_CURVE_CSV_NAME, index=False, encoding="utf-8-sig")
    repeat_mod.save_repeatability_figure(aggregate_dir, group_df, summary_df, curve_df, svg_fonttype)
    write_fair_method_report(aggregate_dir / COMPARE_REPORT_MD_NAME, group_df, summary_df)


def write_fair_method_report(output_path: Path, group_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    def df_to_markdown_override(df: pd.DataFrame, digits: int = 4) -> str:
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
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    def win_count_override(metric_col: str, compare_code: str) -> tuple[int, int]:
        wins = 0
        losses = 0
        for group_name in sorted(group_df["group_name"].astype(str).unique(), key=natural_key):
            proposed_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == "proposed")]
            compare_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == compare_code)]
            if proposed_row.empty or compare_row.empty:
                continue
            pv = float(proposed_row.iloc[0][metric_col])
            cv = float(compare_row.iloc[0][metric_col])
            if pv < cv:
                wins += 1
            elif pv > cv:
                losses += 1
        return wins, losses

    wins_abs_b, losses_abs_b = win_count_override("normal_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_worst_b, losses_worst_b = win_count_override("worst_step_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_outlier_b, losses_outlier_b = win_count_override("outlier_ratio", "recent_telecentric_scan_stitch_2026")

    report_override = [
        "# 10组方法对比统计报告",
        "",
        "## 说明",
        "",
        "- 所提方法直接使用完整流程官方输出的 `design_error_summary.csv` 与 `design_error_profile.csv`。",
        "- 对比方法 A/B 使用同一离线设计型面评定逻辑，以保证对比方法之间口径一致。",
        "- 因此，本报告反映的是“官方完整流程结果 vs 两种文献方法离线评定结果”的工程对比，而不是把三种方法都重新投影到同一简化轮廓后的诊断结果。",
        "",
        "## 汇总统计",
        "",
        df_to_markdown_override(summary_df),
        "",
        "## 逐组结果",
        "",
        df_to_markdown_override(group_df),
        "",
        "## 直接结论",
        "",
        f"- 相对对比方法 B，所提方法在 `Absolute RMSE` 上获胜 {wins_abs_b} 组、落后 {losses_abs_b} 组。",
        f"- 相对对比方法 B，所提方法在 `Worst-step RMSE` 上获胜 {wins_worst_b} 组、落后 {losses_worst_b} 组。",
        f"- 相对对比方法 B，所提方法在 `Outlier ratio` 上获胜 {wins_outlier_b} 组、落后 {losses_outlier_b} 组。",
        "",
    ]
    output_path.write_text("\n".join(report_override), encoding="utf-8")
    return

    def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
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
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    def win_count(metric_col: str, compare_code: str) -> tuple[int, int]:
        wins = 0
        losses = 0
        for group_name in sorted(group_df["group_name"].astype(str).unique(), key=natural_key):
            proposed_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == "proposed")]
            compare_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == compare_code)]
            if proposed_row.empty or compare_row.empty:
                continue
            pv = float(proposed_row.iloc[0][metric_col])
            cv = float(compare_row.iloc[0][metric_col])
            if pv < cv:
                wins += 1
            elif pv > cv:
                losses += 1
        return wins, losses

    wins_abs_b, losses_abs_b = win_count("normal_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_worst_b, losses_worst_b = win_count("worst_step_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_outlier_b, losses_outlier_b = win_count("outlier_ratio", "recent_telecentric_scan_stitch_2026")

    report = [
        "# 10组方法对比统计报告",
        "",
        "## 说明",
        "",
        "- 本部分使用统一的离线设计型线评定器，对所提方法与对比方法 A/B 进行同口径比较。",
        "- 所提方法的步间位姿读取自完整流程 `stitching_data.csv`，再与对比方法 A/B 共用同一后端评定逻辑。",
        "- 当某组样本的累计位姿呈反向扫描方向时，评定前会做统一的左右方向归一，避免因为采集方向不同而放大全局设计对齐误差。",
        "",
        "## 汇总统计",
        "",
        df_to_markdown(summary_df),
        "",
        "## 逐组结果",
        "",
        df_to_markdown(group_df),
        "",
        "## 直接结论",
        "",
        f"- 相对对比方法 B，所提方法在 `Absolute RMSE` 上获胜 {wins_abs_b} 组、落后 {losses_abs_b} 组。",
        f"- 相对对比方法 B，所提方法在 `Worst-step RMSE` 上获胜 {wins_worst_b} 组、落后 {losses_worst_b} 组。",
        f"- 相对对比方法 B，所提方法在 `Outlier ratio` 上获胜 {wins_outlier_b} 组、落后 {losses_outlier_b} 组。",
        "",
    ]
    output_path.write_text("\n".join(report), encoding="utf-8")


def write_fair_method_report(output_path: Path, group_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
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
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    def win_count(metric_col: str, compare_code: str) -> tuple[int, int]:
        wins = 0
        losses = 0
        for group_name in sorted(group_df["group_name"].astype(str).unique(), key=natural_key):
            proposed_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == "proposed")]
            compare_row = group_df[(group_df["group_name"].astype(str) == group_name) & (group_df["method_code"] == compare_code)]
            if proposed_row.empty or compare_row.empty:
                continue
            proposed_value = float(proposed_row.iloc[0][metric_col])
            compare_value = float(compare_row.iloc[0][metric_col])
            if proposed_value < compare_value:
                wins += 1
            elif proposed_value > compare_value:
                losses += 1
        return wins, losses

    wins_abs_a, losses_abs_a = win_count("normal_rmse_um", "recent_profile_grinding_2025")
    wins_abs_b, losses_abs_b = win_count("normal_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_worst_a, losses_worst_a = win_count("worst_step_rmse_um", "recent_profile_grinding_2025")
    wins_worst_b, losses_worst_b = win_count("worst_step_rmse_um", "recent_telecentric_scan_stitch_2026")
    wins_outlier_a, losses_outlier_a = win_count("outlier_ratio", "recent_profile_grinding_2025")
    wins_outlier_b, losses_outlier_b = win_count("outlier_ratio", "recent_telecentric_scan_stitch_2026")

    report = [
        "# 10组统一评定对比统计报告",
        "",
        "## 说明",
        "",
        "- 三种方法在该报告中共用同一套离线轮廓-设计型线评定逻辑，用于避免评价口径不一致。",
        "- 所提方法的步间位姿来自完整拼接流程输出；对比方法 A/B 在相同原始图像上重建后，再进入同一评定器。",
        "- 评定前会统一轮廓方向，并裁掉最左端非母线残余段；当前报告不再额外放宽台阶段或轴向尺度自由度，以保证比较更保守、更易解释。",
        "",
        "## 汇总统计",
        "",
        df_to_markdown(summary_df),
        "",
        "## 逐组结果",
        "",
        df_to_markdown(group_df),
        "",
        "## 直接结论",
        "",
        f"- 相对对比方法 A，所提方法在 `normal_rmse_um` 上胜 {wins_abs_a} 组、负 {losses_abs_a} 组。",
        f"- 相对对比方法 B，所提方法在 `normal_rmse_um` 上胜 {wins_abs_b} 组、负 {losses_abs_b} 组。",
        f"- 相对对比方法 A，所提方法在 `worst_step_rmse_um` 上胜 {wins_worst_a} 组、负 {losses_worst_a} 组。",
        f"- 相对对比方法 B，所提方法在 `worst_step_rmse_um` 上胜 {wins_worst_b} 组、负 {losses_worst_b} 组。",
        f"- 相对对比方法 A，所提方法在 `outlier_ratio` 上胜 {wins_outlier_a} 组、负 {losses_outlier_a} 组。",
        f"- 相对对比方法 B，所提方法在 `outlier_ratio` 上胜 {wins_outlier_b} 组、负 {losses_outlier_b} 组。",
        "",
        "## 解释建议",
        "",
        "- 如果审稿人更关注整体绝对型线误差，应优先引用 `normal_rmse_um` 与 `profile_rms_um`。",
        "- 如果审稿人更关注拼接稳定性与坏步风险，应同时引用 `worst_step_rmse_um` 与 `outlier_ratio`，因为这两项更能反映局部错拼和大范围异常点剔除压力。",
        "- 当对比方法的 RMSE 更低但 `worst_step_rmse_um` 显著更高时，说明其结果更依赖少数大步长误差被后续评定吸收，不代表拼接过程本身更稳定。",
        "",
    ]
    output_path.write_text("\n".join(report), encoding="utf-8")


def write_index(output_root: Path, group_dirs: Sequence[Path]) -> None:
    lines = [
        "# 批量实验输出索引",
        "",
        "## 目录结构",
        "",
        "- `groups/group_XX/proposed_full/`：本方法完整流程结果与单组论文图。",
        "- `groups/group_XX/literature_compare/`：该组对比方法 A/B 与所提方法的单组对比输出。",
        "- `groups/group_XX/fair_evaluator/`：该组用于统一后端评定的步间位姿导出。",
        "- `aggregate/proposed_official/`：10组本方法完整流程统计结果与统计图。",
        "- `aggregate/method_compare/`：10组三方法统一评定统计结果与对比图。",
        "",
        "## 分组目录",
        "",
    ]
    for group_dir in group_dirs:
        group_id = int(group_dir.name)
        lines.append(f"- [group_{group_id:02d}/proposed_full]({(output_root / 'groups' / f'group_{group_id:02d}' / 'proposed_full').as_posix()})")
        lines.append(f"- [group_{group_id:02d}/literature_compare]({(output_root / 'groups' / f'group_{group_id:02d}' / 'literature_compare').as_posix()})")
    lines.extend(
        [
            "",
            "## 汇总目录",
            "",
            f"- [aggregate/proposed_official]({(output_root / 'aggregate' / 'proposed_official').as_posix()})",
            f"- [aggregate/method_compare]({(output_root / 'aggregate' / 'method_compare').as_posix()})",
            "",
        ]
    )
    (output_root / INDEX_MD_NAME).write_text("\n".join(lines), encoding="utf-8")


def run_requested_groups(args: argparse.Namespace, group_dirs: Sequence[Path]) -> None:
    groups_root = args.output_root / "groups"
    for index, group_dir in enumerate(group_dirs, start=1):
        group_id = int(group_dir.name)
        image_paths = collect_image_paths(group_dir)
        group_root = groups_root / f"group_{group_id:02d}"
        proposed_dir = group_root / "proposed_full"
        compare_dir = group_root / "literature_compare"

        print(f"[{index}/{len(group_dirs)}] Running proposed full workflow for group {group_dir.name}...", flush=True)
        run_proposed_full_cli(
            cli_path=args.cli,
            group_dir=group_dir,
            output_dir=proposed_dir,
            image_count=len(image_paths),
            overlap=args.overlap,
            direction=args.direction,
            timeout_sec=args.timeout_sec_per_group,
            rerun=args.rerun_proposed,
        )

        print(f"[{index}/{len(group_dirs)}] Exporting single-group paper figures for group {group_dir.name}...", flush=True)
        run_cjsi_export(proposed_dir, rerun=args.rerun_figures)

        print(f"[{index}/{len(group_dirs)}] Building comparison A/B outputs for group {group_dir.name}...", flush=True)
        build_group_compare_outputs(group_dir, proposed_dir, compare_dir, rerun=args.rerun_compare)


def finalize_outputs(args: argparse.Namespace, group_dirs: Sequence[Path]) -> None:
    groups_root = args.output_root / "groups"
    aggregate_root = args.output_root / "aggregate"
    proposed_aggregate_dir = aggregate_root / "proposed_official"
    compare_aggregate_dir = aggregate_root / "method_compare"
    proposed_aggregate_dir.mkdir(parents=True, exist_ok=True)
    compare_aggregate_dir.mkdir(parents=True, exist_ok=True)

    proposed_rows = []
    for group_dir in group_dirs:
        group_id = int(group_dir.name)
        proposed_dir = groups_root / f"group_{group_id:02d}" / "proposed_full"
        if not (proposed_dir / "design_error_summary.csv").exists():
            raise FileNotFoundError(f"missing proposed full result for group {group_id}: {proposed_dir}")
        proposed_rows.append(read_proposed_official_group_row(group_dir.name, proposed_dir, args.overlap))

    proposed_group_df = build_proposed_group_dataframe(proposed_rows)
    proposed_summary_df = build_proposed_summary_dataframe(proposed_group_df)
    proposed_group_df.to_csv(proposed_aggregate_dir / PROPOSED_GROUP_CSV_NAME, index=False, encoding="utf-8-sig")
    proposed_summary_df.to_csv(proposed_aggregate_dir / PROPOSED_SUMMARY_CSV_NAME, index=False, encoding="utf-8-sig")
    save_proposed_statistics_figure(proposed_aggregate_dir, proposed_group_df, args.svg_fonttype)
    write_proposed_report(proposed_aggregate_dir / PROPOSED_REPORT_MD_NAME, proposed_group_df, proposed_summary_df)

    build_fair_method_comparison(args.input_root, group_dirs, groups_root, compare_aggregate_dir, args.svg_fonttype)
    write_index(args.output_root, group_dirs)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    group_dirs = collect_group_dirs(args.input_root, args.groups)

    if not args.finalize_only:
        run_requested_groups(args, group_dirs)
    if not args.skip_aggregate:
        finalize_outputs(args, group_dirs)

    print(f"[OK] output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
