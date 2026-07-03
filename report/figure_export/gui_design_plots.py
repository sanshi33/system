#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "measured": "#0057D9",
    "design": "#008F5A",
    "profile_error": "#E86F00",
    "normal_error": "#7A1FA2",
    "excluded": "#A8B2C0",
    "delta_x": "#0057D9",
    "delta_y": "#008F5A",
    "delta_z": "#D55E00",
    "radial": "#0072B2",
    "normal": "#C23B8E",
    "image_a": "#0057D9",
    "image_b": "#D55E00",
    "image_c": "#008F5A",
    "exceedance": "#D00027",
    "threshold": "#111827",
    "table_header": "#E8F0FA",
    "table_edge": "#D5DEE8",
    "text": "#1F2937",
}

ARGS: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw GUI design plots with matplotlib.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--compensation-output", type=Path)
    parser.add_argument("--contour-output", type=Path)
    parser.add_argument("--pointcloud-output", type=Path)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--width-px", type=int)
    parser.add_argument("--height-px", type=int)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_rows(path)
    return rows[0] if rows else {}


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or ""))
    except ValueError:
        return default


def first_number(row: dict[str, str], keys: Iterable[str]) -> float:
    for key in keys:
        value = to_float(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def finite_pairs(rows: Iterable[dict[str, str]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = to_float(row.get(x_key))
        y = to_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def row_series(rows: Iterable[dict[str, str]], keys: Iterable[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = first_number(row, keys)
        if math.isfinite(value):
            values.append(value)
    return values


def finite_values(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def symmetric_limit(series: Iterable[Iterable[float]], minimum: float = 1.0) -> float:
    values: list[float] = []
    for item in series:
        values.extend(abs(value) for value in item if math.isfinite(value))
    return max(minimum, max(values, default=minimum)) * 1.12


def padded_limits(values: Iterable[float], minimum_span: float = 1.0) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite:
        return -1.0, 1.0
    lo = min(finite)
    hi = max(finite)
    span = max(minimum_span, hi - lo)
    pad = span * 0.08
    return lo - pad, hi + pad


def generated_footer(result_dir: Path, source: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"result_dir={result_dir}  source={source}  generated={timestamp}"


def figure_size_inches(default_width_in: float,
                       default_height_in: float) -> tuple[float, float]:
    if ARGS is None:
        return default_width_in, default_height_in
    dpi = max(96, int(ARGS.dpi))
    width_px = max(0, int(ARGS.width_px or 0))
    height_px = max(0, int(ARGS.height_px or 0))
    if width_px > 0 and height_px > 0:
        return width_px / dpi, height_px / dpi
    if width_px > 0:
        width_in = width_px / dpi
        return width_in, width_in * default_height_in / max(default_width_in, 1e-9)
    if height_px > 0:
        height_in = height_px / dpi
        return height_in * default_width_in / max(default_height_in, 1e-9), height_in
    return default_width_in, default_height_in


def segmented_lines(xs: list[float], ys: list[float]) -> list[tuple[list[float], list[float]]]:
    if len(xs) != len(ys) or len(xs) < 2:
        return []
    points = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(points) < 2:
        return []

    finite_x = [point[0] for point in points]
    finite_y = [point[1] for point in points]
    x_span = max(finite_x) - min(finite_x)
    y_span = max(finite_y) - min(finite_y)
    distances = [
        math.hypot(points[index][0] - points[index - 1][0],
                   points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    ]
    nonzero_distances = sorted(distance for distance in distances if distance > 1e-9)
    typical_step = (
        nonzero_distances[len(nonzero_distances) // 2]
        if nonzero_distances else 0.0
    )
    jump_limit = max(1.1, typical_step * 14.0, 0.28 * max(x_span, y_span, 1.0))
    vertical_wall_dx_limit = max(0.18, typical_step * 3.5, 0.02 * max(x_span, 1.0))
    vertical_wall_dy_limit = max(2.4, 1.12 * max(y_span, 1.0))

    segments: list[tuple[list[float], list[float]]] = []

    def append_if_visual_segment(seg_x: list[float], seg_y: list[float]) -> None:
        if len(seg_x) < 2:
            return
        segments.append((seg_x, seg_y))

    current_x = [points[0][0]]
    current_y = [points[0][1]]
    for x, y in points[1:]:
        dx = x - current_x[-1]
        dy = y - current_y[-1]
        vertical_slot_wall = (
            abs(dx) <= vertical_wall_dx_limit and
            abs(dy) <= vertical_wall_dy_limit
        )
        if math.hypot(dx, dy) > jump_limit and not vertical_slot_wall:
            append_if_visual_segment(current_x, current_y)
            current_x = [x]
            current_y = [y]
        else:
            current_x.append(x)
            current_y.append(y)
    append_if_visual_segment(current_x, current_y)
    return segments


def plot_segmented_line(ax: plt.Axes,
                        xs: list[float],
                        ys: list[float],
                        label: str,
                        scatter_fallback: bool = True,
                        **kwargs) -> None:
    first = True
    for seg_x, seg_y in segmented_lines(xs, ys):
        ax.plot(seg_x, seg_y, label=label if first else None, **kwargs)
        first = False
    if first and scatter_fallback and xs and ys:
        ax.scatter(xs, ys, s=8, label=label, alpha=0.65)


def configure_matplotlib(dpi: int) -> None:
    effective_dpi = max(96, int(dpi))
    plt.rcParams.update({
        "figure.dpi": effective_dpi,
        "savefig.dpi": effective_dpi,
        "font.family": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "font.size": 15.2,
        "axes.titlesize": 18.2,
        "axes.labelsize": 15.8,
        "xtick.labelsize": 13.6,
        "ytick.labelsize": 13.6,
        "legend.fontsize": 12.8,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.05,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.34,
        "grid.linewidth": 0.62,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CBD5E1",
        "legend.framealpha": 0.96,
        "legend.borderpad": 0.55,
        "legend.handlelength": 2.8,
        "legend.handletextpad": 0.70,
        "legend.labelspacing": 0.55,
        "axes.formatter.useoffset": False,
    })


def metric(summary: dict[str, str], key: str, digits: int = 3) -> str:
    value = to_float(summary.get(key))
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def show_legend_if_any(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        legend_kwargs = {
            "frameon": True,
            "borderaxespad": 0.75,
            "columnspacing": 1.18,
            "handlelength": 2.8,
            "handletextpad": 0.70,
        }
        legend_kwargs.update(kwargs)
        legend = ax.legend(**legend_kwargs)
        if legend:
            legend.get_frame().set_linewidth(0.65)
            legend.get_frame().set_edgecolor("#D5DEE8")
            legend.get_frame().set_facecolor("white")


def style_plot_axes(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.margins(x=0.035, y=0.12)
    for spine in ax.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.95)
    ax.tick_params(colors=COLORS["text"], labelsize=12.8, pad=7)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])


def first_existing_image(result_dir: Path) -> Path | None:
    for name in (
        "origin_contour_overlay.png",
        "final_panorama.png",
        "panorama.png",
        "stitched_contour_profile.png",
    ):
        path = result_dir / name
        if path.exists():
            return path
    return None


def draw_image_panel(ax: plt.Axes, image_path: Path | None, title: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    if image_path is None:
        ax.text(0.5, 0.5, "未找到可显示的结果图", ha="center", va="center", fontsize=12)
        return
    try:
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.text(0.01, 0.02, image_path.name, transform=ax.transAxes,
                fontsize=9, color="white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"})
    except Exception as exc:
        ax.text(0.5, 0.5, f"图像读取失败\n{image_path.name}\n{exc}",
                ha="center", va="center", fontsize=10)


def draw_fallback_comparison(result_dir: Path, output: Path, dpi: int) -> None:
    image_path = first_existing_image(result_dir)
    fig = plt.figure(figsize=figure_size_inches(11.2, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9], wspace=0.24)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_note = fig.add_subplot(grid[0, 1])
    fig.suptitle("误差分析 - 轮廓预览兜底图", x=0.02, y=0.98,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.925,
             "本次运行未生成 CAD 误差 CSV，通常表示 CAD 未导入、未启用设计比对，或检测轮廓暂不能与 CAD 完成对齐。",
             fontsize=9.5, color="#8A5A00")
    draw_image_panel(ax_image, image_path, "当前可用图像证据")
    ax_note.axis("off")
    lines = [
        "GUI 显示说明",
        "该图由 Python matplotlib 生成，用于避免误差分析页空白。",
        "",
        "已完成内容",
        "- 图像读取/预处理",
        "- 单张图像直接展示或多张拼接输出",
        "- 轮廓/全景图结果写入目录",
        "",
        "未生成 CAD 误差图的常见原因",
        "- 未导入 CAD 模型或 CAD 采样轮廓不可用",
        "- 未提供真实标定或临时像素当量",
        "- 检测轮廓与 CAD 轮廓没有足够可对齐点",
        "",
        "需要 CAD 对齐后才会输出",
        "- design_error_profile.csv",
        "- design_3d_error_points.csv",
        "- design_compensation.csv",
    ]
    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10.5)
    fig.text(0.02, 0.02, generated_footer(result_dir, "fallback_comparison"), fontsize=8.8, color="#333333")
    save_figure(fig, output, dpi)


def draw_fallback_compensation(result_dir: Path, output: Path, dpi: int) -> None:
    image_path = first_existing_image(result_dir)
    fig = plt.figure(figsize=figure_size_inches(11.2, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_note = fig.add_subplot(grid[0, 1])
    fig.suptitle("补偿结算 - 等待 CAD 对齐结果", x=0.02, y=0.98,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.925,
             "补偿量必须基于 CAD 目标坐标与检测坐标对比；当前未生成 CAD 点级补偿 CSV，因此输出状态说明图。",
             fontsize=9.5, color="#8A5A00")
    draw_image_panel(ax_image, image_path, "当前可用轮廓/全景证据")
    ax_note.axis("off")
    lines = [
        "补偿结算状态",
        "当前无可用 CAD 点级补偿量。",
        "",
        "补偿输出条件",
        "1. CAD 模型已导入并生成 X/Y/Z 采样轮廓",
        "2. 图像轮廓提取成功",
        "3. 检测轮廓与 CAD 轮廓完成对齐",
        "4. 生成 design_compensation.csv",
        "",
        "正式输出字段",
        "cad_design_x/y/z_mm: CAD 目标点",
        "cad_measured_x/y/z_mm: 检测映射点",
        "compensated_cad_x/y/z_mm: 补偿后 CAD 绝对坐标",
        "delta_x/y/z_um: 从检测点到 CAD 目标点的补偿量",
        "",
        "若无法形成三维映射，软件会退回同一截面二维补偿演示。",
    ]
    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10.5)
    fig.text(0.02, 0.02, generated_footer(result_dir, "fallback_compensation"), fontsize=8.8, color="#333333")
    save_figure(fig, output, dpi)


def find_slot_feature(result_dir: Path) -> dict[str, str] | None:
    path = result_dir / "design_feature_compensation.csv"
    if not path.exists():
        return None
    candidates: list[dict[str, str]] = []
    for row in read_rows(path):
        feature_id = (row.get("feature_id") or "").lower()
        method = (row.get("method") or "").lower()
        status = (row.get("status") or "").lower()
        has_width = any(math.isfinite(to_float(row.get(key))) for key in (
            "target_slot_width_mm",
            "measured_slot_width_mm",
            "cad_slot_width_mm",
            "width_error_um",
        ))
        looks_like_slot = "slot" in feature_id or "slot" in method or has_width
        if looks_like_slot:
            candidates.append(row)
            if status == "ok":
                return row
    return candidates[0] if candidates else None


def estimate_slot_bounds(rows: list[dict[str, str]]) -> dict[str, float] | None:
    series = []
    for row in rows:
        s = to_float(row.get("s_aligned_mm"))
        r = to_float(row.get("r_aligned_mm"))
        if math.isfinite(s) and math.isfinite(r):
            series.append((s, r))
    if len(series) < 5:
        return None
    series.sort()
    radii = [r for _, r in series]
    top_r = sorted(radii)[max(0, min(len(radii) - 1, int(round(0.90 * (len(radii) - 1)))))]
    bottom_index, (_, bottom_r) = min(enumerate(series), key=lambda item: item[1][1])
    depth = top_r - bottom_r
    if not depth > 1e-9:
        return None
    threshold = top_r - 0.5 * depth
    left = bottom_index
    while left > 0 and series[left - 1][1] <= threshold:
        left -= 1
    right = bottom_index
    while right + 1 < len(series) and series[right + 1][1] <= threshold:
        right += 1
    left_s, _ = series[left]
    right_s, _ = series[right]
    bottom_s, _ = series[bottom_index]
    if not right_s > left_s:
        return None
    return {
        "left_s": left_s,
        "right_s": right_s,
        "center_s": 0.5 * (left_s + right_s),
        "bottom_s": bottom_s,
        "top_r": top_r,
        "bottom_r": bottom_r,
        "width": right_s - left_s,
        "depth": depth,
    }


def slot_value(feature: dict[str, str] | None, keys: Iterable[str]) -> float:
    if not feature:
        return math.nan
    return first_number(feature, keys)


def is_local_slot_result(feature: dict[str, str] | None,
                         summary: dict[str, str] | None = None) -> bool:
    fields = []
    for row in (feature or {}, summary or {}):
        fields.extend([
            row.get("design_source_type", ""),
            row.get("design_source_name", ""),
            row.get("cad_extraction_method", ""),
            row.get("method", ""),
            row.get("feature_id", ""),
        ])
    return any("local_slot" in str(value).lower() for value in fields)


def is_microchannel_result(feature: dict[str, str] | None,
                           summary: dict[str, str] | None = None) -> bool:
    fields = []
    for row in (feature or {}, summary or {}):
        fields.extend([
            row.get("design_source_type", ""),
            row.get("design_source_name", ""),
            row.get("cad_extraction_method", ""),
            row.get("method", ""),
            row.get("feature_id", ""),
            row.get("notes", ""),
        ])
    return any(
        "microchannel" in str(value).lower() or "微流" in str(value)
        for value in fields
    )


def slot_status(feature: dict[str, str] | None) -> str:
    return ((feature or {}).get("status") or "").strip().lower()


def slot_width_mismatch_ratio(feature: dict[str, str] | None) -> float:
    target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
    measured_width = slot_value(feature, ["measured_slot_width_mm"])
    if not (math.isfinite(target_width) and target_width > 1e-9 and math.isfinite(measured_width)):
        return math.nan
    return abs(measured_width - target_width) / target_width


def slot_width_mismatch_detected(feature: dict[str, str] | None, threshold: float = 0.25) -> bool:
    ratio = slot_width_mismatch_ratio(feature)
    status = slot_status(feature)
    return status == "mismatch" or (math.isfinite(ratio) and ratio > threshold)


def slot_mismatch_warning(feature: dict[str, str] | None, threshold: float = 0.25) -> str:
    if slot_width_mismatch_detected(feature, threshold):
        target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
        measured_width = slot_value(feature, ["measured_slot_width_mm"])
        return (
            "槽宽严重不一致："
            f"CAD/目标 {target_width:.6f} mm，检测 {measured_width:.6f} mm。"
            "该结果说明 CAD、图像 ROI 或标定参数不匹配，补偿量仅作诊断，不应作为正式修正量。"
        )
    return ""


def profile_span_coverage_ratio(rows: list[dict[str, str]], summary: dict[str, str]) -> float:
    if (summary.get("design_source_type") or "").strip() != "external_cad":
        return math.nan
    design_min = to_float(summary.get("design_profile_min_s_mm"))
    design_max = to_float(summary.get("design_profile_max_s_mm"))
    s_values = [to_float(row.get("s_aligned_mm")) for row in rows]
    finite_s = finite_values(s_values)
    if len(finite_s) < 2 or not (math.isfinite(design_min) and math.isfinite(design_max)):
        return math.nan
    design_span = design_max - design_min
    if not design_span > 1e-9:
        return math.nan
    return (max(finite_s) - min(finite_s)) / design_span


def section_coverage_warning(rows: list[dict[str, str]],
                             summary: dict[str, str],
                             threshold: float = 0.65,
                             feature: dict[str, str] | None = None) -> str:
    if is_local_slot_result(feature, summary):
        return ""
    ratio = profile_span_coverage_ratio(rows, summary)
    if math.isfinite(ratio) and ratio < threshold:
        return (
            f"当前为单槽 ROI 验收模式，检测轮廓覆盖 CAD 截面约 {ratio * 100.0:.1f}%；"
            "本图仅评价当前槽宽、槽边缘检测和补偿输出，不作为完整截面覆盖率判定。"
        )
    return ""


def combined_cad_warning(feature: dict[str, str] | None,
                         rows: list[dict[str, str]],
                         summary: dict[str, str]) -> str:
    warnings = [
        warning
        for warning in (
            slot_mismatch_warning(feature),
            section_coverage_warning(rows, summary, feature=feature),
        )
        if warning
    ]
    return "\n".join(warnings)


def local_exceedance_threshold_um(summary: dict[str, str] | None) -> float:
    if summary:
        for key in ("local_exceedance_threshold_um", "profile_exceedance_threshold_um"):
            value = to_float(summary.get(key))
            if math.isfinite(value) and value > 0.0:
                return value
    return 15.0


def row_profile_error_um(row: dict[str, str] | dict[str, float | str]) -> float:
    return first_number(row, ("profile_error_um", "normal_error_um", "radial_error_um"))


def computed_exceedance_stats(rows: Iterable[dict[str, str] | dict[str, float | str]],
                              threshold_um: float) -> dict[str, float]:
    count = 0
    max_abs = math.nan
    for row in rows:
        error = row_profile_error_um(row)
        if not math.isfinite(error):
            continue
        abs_error = abs(error)
        max_abs = abs_error if not math.isfinite(max_abs) else max(max_abs, abs_error)
        if abs_error > threshold_um:
            count += 1
    max_excess = max(0.0, max_abs - threshold_um) if math.isfinite(max_abs) else math.nan
    return {
        "threshold_um": threshold_um,
        "count": float(count),
        "max_abs_um": max_abs,
        "max_excess_um": max_excess,
    }


def local_exceedance_stats(rows: Iterable[dict[str, str] | dict[str, float | str]],
                           summary: dict[str, str] | None) -> dict[str, float]:
    threshold = local_exceedance_threshold_um(summary)
    stats = computed_exceedance_stats(rows, threshold)
    if summary:
        csv_count = to_float(summary.get("local_exceedance_count"))
        csv_max_abs = to_float(summary.get("local_max_abs_error_um"))
        csv_max_excess = to_float(summary.get("local_max_exceedance_um"))
        if math.isfinite(csv_count):
            stats["count"] = csv_count
        if math.isfinite(csv_max_abs):
            stats["max_abs_um"] = csv_max_abs
        if math.isfinite(csv_max_excess):
            stats["max_excess_um"] = csv_max_excess
    return stats


def exceedance_segments(rows: Iterable[dict[str, str] | dict[str, float | str]],
                        x_keys: Iterable[str],
                        y_keys: Iterable[str],
                        threshold_um: float) -> tuple[list[tuple[list[float], list[float]]], list[tuple[float, float]]]:
    segments: list[tuple[list[float], list[float]]] = []
    isolated: list[tuple[float, float]] = []
    current_x: list[float] = []
    current_y: list[float] = []

    def flush() -> None:
        nonlocal current_x, current_y
        if len(current_x) >= 2:
            segments.append((current_x, current_y))
        elif len(current_x) == 1:
            isolated.append((current_x[0], current_y[0]))
        current_x = []
        current_y = []

    for row in rows:
        x = first_number(row, x_keys)
        y = first_number(row, y_keys)
        error = row_profile_error_um(row)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(error) and abs(error) > threshold_um:
            current_x.append(x)
            current_y.append(y)
        else:
            flush()
    flush()
    return segments, isolated


def plot_exceedance_segments(ax: plt.Axes,
                             rows: Iterable[dict[str, str] | dict[str, float | str]],
                             x_keys: Iterable[str],
                             y_keys: Iterable[str],
                             summary: dict[str, str] | None,
                             label: str = "局部超差段") -> dict[str, float]:
    threshold = local_exceedance_threshold_um(summary)
    row_list = list(rows)
    segments, isolated = exceedance_segments(row_list, x_keys, y_keys, threshold)
    first = True
    for xs, ys in segments:
        ax.plot(xs,
                ys,
                color=COLORS["exceedance"],
                linewidth=3.4,
                solid_capstyle="round",
                alpha=0.95,
                zorder=8,
                label=label if first else None)
        first = False
    if isolated:
        xs, ys = zip(*isolated)
        ax.scatter(list(xs),
                   list(ys),
                   s=36,
                   marker="x",
                   linewidths=1.4,
                   color=COLORS["exceedance"],
                   zorder=9,
                   label=label if first else None)
    return local_exceedance_stats(row_list, summary)


def formatted_exceedance_line(stats: dict[str, float]) -> str:
    threshold = stats.get("threshold_um", math.nan)
    count = int(round(stats.get("count", 0.0))) if math.isfinite(stats.get("count", math.nan)) else 0
    max_abs = stats.get("max_abs_um", math.nan)
    if math.isfinite(threshold) and math.isfinite(max_abs):
        return f"局部超差: {count} 点；阈值 ±{threshold:.1f} um；最大 |误差| {max_abs:.3f} um"
    if math.isfinite(threshold):
        return f"局部超差: {count} 点；阈值 ±{threshold:.1f} um"
    return f"局部超差: {count} 点"


def compensation_result_nature(feature: dict[str, str] | None,
                               rows: list[dict[str, str]],
                               summary: dict[str, str]) -> str:
    if slot_width_mismatch_detected(feature):
        return "诊断，不可正式补偿"
    if is_local_slot_result(feature, summary):
        return "可用于补偿复核"
    ratio = profile_span_coverage_ratio(rows, summary)
    if math.isfinite(ratio) and ratio < 0.65:
        return "单槽ROI补偿复核"
    return "可用于补偿复核"


def draw_status_banner(fig: plt.Figure, text: str, y: float = 0.815) -> None:
    if not text:
        return
    fig.text(0.02, y, text, fontsize=12.0, color="#185A37",
             bbox={
                 "boxstyle": "round,pad=0.42",
                 "facecolor": "#E9F7EF",
                 "edgecolor": "#40A66B",
                 "linewidth": 1.0,
             })


def draw_warning_banner(fig: plt.Figure, warning: str, y: float = 0.905) -> None:
    if not warning:
        return
    fig.text(0.02, y, warning, fontsize=10.5, color="#8A2D00",
             bbox={
                 "boxstyle": "round,pad=0.38",
                 "facecolor": "#FFF3E0",
                 "edgecolor": "#F2A65A",
                 "linewidth": 0.9,
             })


def sort_profile_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            to_float(row.get("s_aligned_mm")) if math.isfinite(to_float(row.get("s_aligned_mm"))) else math.inf,
            to_float(row.get("r_aligned_mm")) if math.isfinite(to_float(row.get("r_aligned_mm"))) else math.inf,
        ),
    )


def microchannel_segment_key(row: dict[str, str] | dict[str, float | str]) -> tuple[int, int, float]:
    segment = str(row.get("segment", ""))
    match = re.search(r"microchannel_(\d+)_(left|right)", segment, flags=re.IGNORECASE)
    if match:
        side = 0 if match.group(2).lower() == "left" else 1
        return int(match.group(1)), side, to_float(row.get("source_index") or row.get("index"))
    return 10**9, 0, to_float(row.get("source_index") or row.get("index"))


def group_microchannel_rows(rows: Iterable[dict[str, str] | dict[str, float | str]]) -> list[list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = microchannel_segment_key(row)[:2]
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped):
        result.append(sorted(
            grouped[key],
            key=lambda item: to_float(item.get("s_aligned_mm")),
        ))
    return result


def focus_microchannel_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int | None]:
    channels: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        channel_index = microchannel_segment_key(row)[0]
        if channel_index >= 10**9:
            continue
        channels.setdefault(channel_index, []).append(row)
    if not channels:
        return rows, None
    ordered_channels = sorted(channels)
    focus_channel = ordered_channels[len(ordered_channels) // 2]
    focused = sorted(
        channels[focus_channel],
        key=lambda item: (
            microchannel_segment_key(item)[1],
            to_float(item.get("s_aligned_mm")),
            to_float(item.get("source_index") or item.get("index")),
        ),
    )
    return focused if focused else rows, focus_channel


def load_profile_data(result_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    profile_rows = read_rows(result_dir / "design_error_profile.csv")
    summary = read_first_row(result_dir / "design_error_summary.csv")
    used_rows = sort_profile_rows(row for row in profile_rows if to_int(row.get("is_used")) == 1)
    excluded_rows = sort_profile_rows(row for row in profile_rows if to_int(row.get("is_used")) != 1)
    return profile_rows, used_rows, excluded_rows, summary


def plot_source_contours(ax: plt.Axes,
                         result_dir: Path,
                         used_rows: list[dict[str, str]],
                         excluded_rows: list[dict[str, str]],
                         feature: dict[str, str] | None) -> None:
    origin_csv = result_dir / "origin_contour_overlay_points.csv"
    if origin_csv.exists():
        origin_rows = read_rows(origin_csv)
        roi_x, roi_y = finite_pairs(used_rows, "x_px", "y_px")
        roi_bounds: tuple[float, float, float, float] | None = None
        if feature and roi_x and roi_y:
            x_lo, x_hi = finite_range(roi_x)
            y_lo, y_hi = finite_range(roi_y)
            if all(math.isfinite(value) for value in (x_lo, x_hi, y_lo, y_hi)):
                x_pad = max(45.0, (x_hi - x_lo) * 0.38)
                y_pad = max(45.0, (y_hi - y_lo) * 0.38)
                roi_bounds = (x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad)
        fieldnames = list(origin_rows[0].keys()) if origin_rows else []
        image_indices = []
        for field in fieldnames:
            if field.startswith("Image") and field.endswith("_X(px)"):
                image_indices.append(field[len("Image"):field.index("_X(px)")])
        for index, image_id in enumerate(image_indices[:6]):
            xs = [to_float(row.get(f"Image{image_id}_X(px)")) for row in origin_rows]
            ys = [to_float(row.get(f"Image{image_id}_Y(px)")) for row in origin_rows]
            points = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if roi_bounds is not None:
                x_min, x_max, y_min, y_max = roi_bounds
                points = [
                    (x, y)
                    for x, y in points
                    if x >= x_min and x <= x_max and y >= y_min and y <= y_max
                ]
            if not points:
                continue
            px, py = zip(*points)
            color = [COLORS["image_a"], COLORS["image_b"], COLORS["image_c"], "#7B61FF", "#6B7280", "#D97706"][index]
            px_list = list(px)
            py_list = list(py)
            ax.scatter(px_list,
                       py_list,
                       s=5,
                       alpha=0.52,
                       color=color,
                       label=f"图像{image_id}目标槽亚像素点")
        ax.set_title("原图中轮廓对齐图")
        ax.set_xlabel("图像/画布 X / px")
        ax.set_ylabel("图像/画布 Y / px")
        ax.invert_yaxis()
        ax.grid(True)
        show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=10.0)
        return

    ex_x, ex_y = finite_pairs(excluded_rows, "x_px", "y_px")
    used_x, used_y = finite_pairs(used_rows, "x_px", "y_px")
    if ex_x and ex_y:
        ax.scatter(ex_x, ex_y, s=4, color=COLORS["excluded"], alpha=0.26, label="非ROI/过滤点")
    if used_x and used_y:
        label = "槽类特征ROI" if feature else "工件轮廓"
        ax.scatter(used_x, used_y, s=8, color=COLORS["measured"], alpha=0.62, label=label)
    ax.set_title("原图中轮廓对齐图")
    ax.set_xlabel("图像/展开 X / px")
    ax.set_ylabel("图像/展开 Y / px")
    ax.invert_yaxis()
    style_plot_axes(ax)
    show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=10.0)


def plot_profile_alignment(ax: plt.Axes,
                           used_rows: list[dict[str, str]],
                           excluded_rows: list[dict[str, str]],
                           feature: dict[str, str] | None,
                           summary: dict[str, str] | None = None,
                           reference_label: str = "CAD轮廓",
                           title: str = "提取特征轮廓与 CAD 轮廓对齐") -> None:
    local_slot = is_local_slot_result(feature, {})
    microchannel = is_microchannel_result(feature, summary)
    if microchannel:
        first_measured = True
        first_design = True
        for group in group_microchannel_rows(used_rows):
            xs, measured = finite_pairs(group, "s_aligned_mm", "r_aligned_mm")
            _, design = finite_pairs(group, "s_aligned_mm", "r_design_mm")
            if xs and measured:
                ax.plot(xs,
                        measured,
                        color=COLORS["measured"],
                        linewidth=2.05,
                        alpha=0.97,
                        label="检测微流道边缘" if first_measured else None)
                first_measured = False
            if xs and design:
                ax.plot(xs[:len(design)],
                        design,
                        color=COLORS["design"],
                        linewidth=1.85,
                        alpha=0.95,
                        linestyle="--",
                        label=reference_label if first_design else None)
                first_design = False
        ax.set_title(title)
        ax.set_xlabel("通道长度方向 / mm")
        ax.set_ylabel("通道横向边缘位置 / mm")
        style_plot_axes(ax)
        show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=11.4)
        return
    used_s, measured_r = finite_pairs(used_rows, "s_aligned_mm", "r_aligned_mm")
    _, design_r = finite_pairs(used_rows, "s_aligned_mm", "r_design_mm")
    if not design_r:
        _, design_r = finite_pairs(used_rows, "s_aligned_mm", "nearest_design_r_mm")
    ex_s, ex_r = finite_pairs(excluded_rows, "s_aligned_mm", "r_aligned_mm")

    if ex_s and ex_r and not local_slot:
        ax.scatter(ex_s, ex_r, s=4, color=COLORS["excluded"], alpha=0.18, label="过滤/未使用点")
    if used_s and measured_r:
        plot_segmented_line(ax,
                            used_s,
                            measured_r,
                            "检测槽边缘曲线" if local_slot else "检测特征轮廓",
                            color=COLORS["measured"],
                            linewidth=3.05 if local_slot else 2.65,
                            marker=None if local_slot else "o",
                            markersize=0 if local_slot else 2.6)
    if used_s and design_r:
        plot_segmented_line(ax,
                            used_s[:len(design_r)],
                            design_r,
                            reference_label,
                            color=COLORS["design"],
                            linewidth=2.75 if local_slot else 2.35,
                            linestyle="--")
    if used_s and measured_r:
        plot_exceedance_segments(ax,
                                 used_rows,
                                 ("s_aligned_mm",),
                                 ("r_aligned_mm", "measured_r_mm"),
                                 summary)

    bounds = estimate_slot_bounds(used_rows)
    if feature and bounds and not local_slot:
        ax.axvline(bounds["left_s"], color="#E66101", linewidth=1.6, linestyle="--", label="槽左边界")
        ax.axvline(bounds["right_s"], color="#1B9E77", linewidth=1.6, linestyle="--", label="槽右边界")
        ax.scatter([bounds["bottom_s"]], [bounds["bottom_r"]], color="#C76334", s=34, zorder=4,
                   label="槽底候选")
    ax.set_title(title)
    ax.set_xlabel("沿槽底方向 / mm" if local_slot else "统一截面坐标 s / mm")
    ax.set_ylabel("槽深方向 / mm" if local_slot else "轮廓半径/截面坐标 r / mm")
    style_plot_axes(ax)
    show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=11.4)
    y_min, y_max = padded_limits([*measured_r, *design_r, *ex_r], minimum_span=0.5)
    ax.set_ylim(y_min, y_max)


def plot_error_curve(ax: plt.Axes,
                     used_rows: list[dict[str, str]],
                     summary: dict[str, str] | None = None) -> None:
    if is_microchannel_result(None, summary) and any("microchannel" in str(row.get("segment", "")).lower() for row in used_rows):
        first_profile = True
        first_normal = True
        for group in group_microchannel_rows(used_rows):
            xs, profile_error = finite_pairs(group, "s_aligned_mm", "profile_error_um")
            _, normal_error = finite_pairs(group, "s_aligned_mm", "normal_error_um")
            if xs and profile_error:
                ax.plot(xs,
                        profile_error,
                        color=COLORS["profile_error"],
                        linewidth=1.65,
                        alpha=0.95,
                        label="横向轮廓误差" if first_profile else None)
                first_profile = False
            if xs and normal_error:
                ax.plot(xs[:len(normal_error)],
                        normal_error,
                        color=COLORS["normal_error"],
                        linewidth=1.45,
                        alpha=0.88,
                        label="法向/横向误差" if first_normal else None)
                first_normal = False
        threshold = local_exceedance_threshold_um(summary)
        limit = symmetric_limit([
            [row_profile_error_um(row) for row in used_rows],
            [threshold, -threshold],
        ], minimum=1.0)
        ax.axhline(threshold, color=COLORS["threshold"], linewidth=1.25, linestyle="--", alpha=0.86)
        ax.axhline(-threshold, color=COLORS["threshold"], linewidth=1.25, linestyle="--", alpha=0.86)
        ax.axhline(0.0, color="#6B7280", linewidth=0.9)
        ax.set_ylim(-limit, limit)
        ax.set_title("微流道边缘横向误差（按通道长度展开）")
        ax.set_xlabel("通道长度方向 / mm")
        ax.set_ylabel("误差 / um")
        style_plot_axes(ax)
        show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3, fontsize=10.4)
        return
    err_s, profile_error = finite_pairs(used_rows, "s_aligned_mm", "profile_error_um")
    norm_s, normal_error = finite_pairs(used_rows, "s_aligned_mm", "normal_error_um")
    if err_s and profile_error:
        plot_segmented_line(ax, err_s, profile_error, "轮廓误差", color=COLORS["profile_error"], linewidth=1.75)
    if norm_s and normal_error:
        plot_segmented_line(ax, norm_s, normal_error, "法向误差", color=COLORS["normal_error"], linewidth=1.55)
    threshold = local_exceedance_threshold_um(summary)
    exceed_points: list[tuple[float, float]] = []
    for row in used_rows:
        s = first_number(row, ("s_aligned_mm", "design_s_mm"))
        error = row_profile_error_um(row)
        if math.isfinite(s) and math.isfinite(error) and abs(error) > threshold:
            exceed_points.append((s, error))
    ax.axhline(threshold,
               color=COLORS["threshold"],
               linewidth=1.25,
               linestyle="--",
               alpha=0.86,
               label=f"局部超差阈值 ±{threshold:.1f} um")
    ax.axhline(-threshold,
               color=COLORS["threshold"],
               linewidth=1.1,
               linestyle="--",
               alpha=0.86)
    if exceed_points:
        xs, ys = zip(*exceed_points)
        ax.scatter(list(xs),
                   list(ys),
                   s=30,
                   color=COLORS["exceedance"],
                   edgecolors="white",
                   linewidths=0.45,
                   zorder=7,
                   label=f"局部超差 {len(exceed_points)} 点")
        max_s, max_error = max(exceed_points, key=lambda item: abs(item[1]))
        ax.annotate(f"最大超差 {max(0.0, abs(max_error) - threshold):.2f} um",
                    xy=(max_s, max_error),
                    xytext=(0.04, 0.90),
                    textcoords="axes fraction",
                    fontsize=9.6,
                    color=COLORS["exceedance"],
                    arrowprops={"arrowstyle": "->", "color": COLORS["exceedance"], "lw": 1.0})
    limit = symmetric_limit([profile_error, normal_error, [threshold, -threshold]], minimum=1.0)
    ax.axhline(0.0, color="#6B7280", linewidth=0.9)
    ax.set_ylim(-limit, limit)
    ax.set_title("误差分析图（按 s 展开，红色为局部超差）")
    ax.set_xlabel("统一截面坐标 s / mm")
    ax.set_ylabel("误差 / um")
    style_plot_axes(ax)
    show_legend_if_any(ax, loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3, fontsize=10.4)


def draw_comparison(result_dir: Path, output: Path, dpi: int) -> None:
    _, used_rows, excluded_rows, summary = load_profile_data(result_dir)
    feature = find_slot_feature(result_dir)
    local_slot = is_local_slot_result(feature, summary)
    microchannel = is_microchannel_result(feature, summary)

    fig = plt.figure(figsize=figure_size_inches(14.2, 9.2))
    grid = fig.add_gridspec(2, 2,
                            left=0.070, right=0.972, top=0.690, bottom=0.112,
                            width_ratios=[1.62, 0.88],
                            height_ratios=[1.08, 0.78],
                            hspace=0.70,
                            wspace=0.32)
    ax_profile = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[1, 0])
    ax_note = fig.add_subplot(grid[:, 1])

    mode = "微流道阵列检测" if microchannel else ("单槽宽度验收" if local_slot else ("槽类特征自动识别" if feature else "工件轮廓提取"))
    fig.suptitle(f"误差分析 - {mode}", x=0.02, y=0.985, ha="left", fontsize=21, fontweight="bold")
    subtitle = (
        "当前页展示微流道阵列边缘检测、CAD目标边缘对齐和横向误差；补偿后 CAD 坐标在下一模块输出。"
        if microchannel else
        "当前页只展示本次导入图中的单个槽边缘与 CAD/目标槽边缘对齐结果；补偿结算在下一模块单独展示。"
        if local_slot else
        "当前页只解释单槽 ROI 的检测轮廓与 CAD 对应关系；补偿结算在下一模块单独展示。"
    )
    fig.text(0.02, 0.910, subtitle, fontsize=12.4, color="#425466")
    target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
    measured_width = slot_value(feature, ["measured_slot_width_mm"])
    width_error = slot_value(feature, ["width_error_um"])
    if microchannel:
        channel_count = slot_value(feature, ["channel_count"])
        draw_status_banner(
            fig,
            f"微流道阵列检测完成：识别 {channel_count:.0f} 条通道，检测槽宽 {measured_width:.6f} mm，CAD目标槽宽 {target_width:.6f} mm，宽度误差 {width_error:.3f} um。"
            if math.isfinite(channel_count) and math.isfinite(width_error) else
            "微流道阵列检测完成：已生成边缘曲线、误差分析和 CAD 坐标补偿输出。",
            y=0.795,
        )
    elif local_slot and math.isfinite(width_error) and not slot_width_mismatch_detected(feature):
        draw_status_banner(
            fig,
            f"单槽宽度验收通过：检测槽宽 {measured_width:.6f} mm，基准槽宽 {target_width:.6f} mm，槽宽误差 {width_error:.3f} um。",
            y=0.795,
        )
    else:
        draw_warning_banner(fig, combined_cad_warning(feature, used_rows, summary), y=0.795)

    plot_profile_alignment(
        ax_profile,
        used_rows,
        excluded_rows,
        feature,
        summary,
        reference_label="CAD目标边缘" if microchannel else ("基准槽轮廓" if local_slot else "CAD轮廓"),
        title="微流道检测边缘与 CAD 目标边缘对齐" if microchannel else ("局部槽检测轮廓与基准槽轮廓对齐" if local_slot else "提取特征轮廓与 CAD 轮廓对齐"),
    )
    ax_profile.set_title("")
    ax_profile.set_xlabel("通道长度方向 / mm" if microchannel else ("沿槽底方向 / mm" if local_slot else "单槽截面位置 / mm"))
    ax_profile.set_ylabel("通道横向边缘位置 / mm" if microchannel else ("槽深方向 / mm" if local_slot else "CAD Y / mm"))
    ax_profile.tick_params(labelsize=12.6)

    plot_error_curve(ax_error, used_rows, summary)
    ax_error.set_title("")
    ax_error.tick_params(labelsize=12.0)

    measured_y = row_series(used_rows, ["r_aligned_mm", "measured_r_mm"])
    design_y = row_series(used_rows, ["r_design_mm", "nearest_design_r_mm", "design_r_mm", "design_radius_mm"])
    coverage_ratio = math.nan if local_slot else profile_span_coverage_ratio(used_rows, summary)
    roi_label = "金刚石微流道阵列 ROI" if microchannel else ("局部单槽截图 ROI" if local_slot else infer_cad_roi_label(summary, design_y))
    measured_y_min, measured_y_max = finite_range(measured_y)
    design_y_min, design_y_max = finite_range(design_y)
    bottom_gap = (
        abs(measured_y_min - design_y_min)
        if math.isfinite(measured_y_min) and math.isfinite(design_y_min)
        else math.nan
    )
    bottom_aligned = math.isfinite(bottom_gap) and bottom_gap <= 0.35
    measured_edge_description = (
        "槽底及左右槽壁内轮廓"
        if bottom_aligned else
        "槽口/槽壁上半段的局部边缘"
    )
    exceedance_stats = local_exceedance_stats(used_rows, summary)

    ax_note.axis("off")
    if microchannel:
        note_lines = [
            "微流道阵列检测对象",
            roi_label,
            "蓝线：检测到的微流道左右边缘",
            "绿线：对应 CAD 目标边缘",
            "",
            "槽宽结果",
            f"检测槽宽: {measured_width:.6f} mm" if math.isfinite(measured_width) else "检测槽宽: --",
            f"CAD目标槽宽: {target_width:.6f} mm" if math.isfinite(target_width) else "CAD目标槽宽: --",
            f"槽宽误差: {width_error:.3f} um" if math.isfinite(width_error) else "槽宽误差: --",
            formatted_exceedance_line(exceedance_stats),
            "",
            "对齐说明",
            "检测边缘已与 CAD 目标边缘完成同一尺度对齐。",
            "补偿结果输出为 CAD 模型坐标系下的 XYZ 坐标。",
            "该页展示边缘误差，补偿后曲线见模块 6。",
            "",
            f"有效边缘点: {summary.get('used_count', '--')}",
            "补偿后 XYZ 坐标见模块 6。",
        ]
    elif local_slot:
        note_lines = [
            "单槽验收对象",
            roi_label,
            "蓝线：检测到的单槽边缘曲线",
            "绿线：对应的 CAD/目标槽边缘",
            "",
            "槽宽结果",
            f"检测槽宽: {measured_width:.6f} mm" if math.isfinite(measured_width) else "检测槽宽: --",
            f"基准槽宽: {target_width:.6f} mm" if math.isfinite(target_width) else "基准槽宽: --",
            f"槽宽误差: {width_error:.3f} um" if math.isfinite(width_error) else "槽宽误差: --",
            formatted_exceedance_line(exceedance_stats),
            "",
            "对齐说明",
            "本页不再评价完整 CAD 截面覆盖率。",
            "当前只复检输入图中的单个槽宽。",
            "补偿曲线和补偿后坐标见模块 6。",
            "",
            f"有效边缘点: {summary.get('used_count', '--')}",
        ]
    else:
        note_lines = [
            "检测轮廓对应 CAD 哪一部分",
            roi_label,
            f"截面: {summary.get('cad_section_normal_axis', '--')} = {metric(summary, 'cad_section_coordinate_mm', 3)} mm",
            "",
            "当前检测到的蓝线",
            measured_edge_description,
            f"检测 Y: {range_text(measured_y)} mm",
            "",
            "当前 CAD 对照绿线",
            "同一单槽 ROI 的 CAD 目标轮廓",
            f"CAD Y: {range_text(design_y)} mm",
            "",
            "误差说明",
            f"槽宽误差: {width_error:.3f} um" if math.isfinite(width_error) else "槽宽误差: --",
            f"检测槽宽: {measured_width:.6f} mm" if math.isfinite(measured_width) else "检测槽宽: --",
            f"CAD槽宽: {target_width:.6f} mm" if math.isfinite(target_width) else "CAD槽宽: --",
            formatted_exceedance_line(exceedance_stats),
        ]
    if not local_slot and math.isfinite(measured_y_min) and math.isfinite(design_y_min):
        if bottom_aligned:
            note_lines.extend([
                "",
                f"基准槽底最低约 r={design_y_min:.3f} mm" if local_slot else
                f"CAD 槽底最低约 Y={design_y_min:.3f} mm",
                f"检测最低约 r={measured_y_min:.3f} mm" if local_slot else
                f"检测最低约 Y={measured_y_min:.3f} mm",
                "槽底深度已基本对齐，",
                "剩余误差主要来自边缘采样、",
                "像素当量和局部缺口。",
            ])
        else:
            note_lines.extend([
                "",
                f"CAD 槽底最低约 Y={design_y_min:.3f} mm",
                f"检测最低约 Y={measured_y_min:.3f} mm",
                "深度方向没有跟到槽底，",
                "因此整体误差曲线会很大。",
            ])
    if not local_slot:
        note_lines.extend([
            "",
            f"有效点: {summary.get('used_count', '--')}",
            f"单槽覆盖率: {coverage_ratio * 100.0:.1f}%" if math.isfinite(coverage_ratio) else "单槽覆盖率: --",
            f"RMSE: {metric(summary, 'profile_rms_um')} um",
            formatted_exceedance_line(exceedance_stats),
        ])
    ax_note.text(0.0, 1.0, "\n".join(note_lines),
                 va="top", ha="left", fontsize=12.2, linespacing=1.20,
                 bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F7FAFC",
                       "edgecolor": "#D7DBE0", "linewidth": 0.9})
    save_figure(fig, output, dpi)


def draw_single_slot_contour(result_dir: Path, output: Path, dpi: int) -> None:
    _, used_rows, excluded_rows, summary_row = load_profile_data(result_dir)
    feature = find_slot_feature(result_dir)
    local_slot = is_local_slot_result(feature, summary_row)
    microchannel = is_microchannel_result(feature, summary_row)
    if not used_rows:
        draw_comparison(result_dir, output, dpi)
        return
    display_rows = used_rows
    display_excluded_rows = excluded_rows
    focus_channel: int | None = None
    if microchannel:
        display_rows, focus_channel = focus_microchannel_rows(used_rows)
        display_excluded_rows = []

    fig = plt.figure(figsize=figure_size_inches(13.2, 8.4))
    grid = fig.add_gridspec(2, 1,
                            left=0.085, right=0.985, top=0.770, bottom=0.120,
                            height_ratios=[1.12, 1.0],
                            hspace=0.72)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_profile = fig.add_subplot(grid[1, 0])
    fig.suptitle(
        (f"特征提取 - 单条微流槽放大（第 {focus_channel} 条）"
         if microchannel and focus_channel is not None else
         "特征提取 - 微流道阵列边缘识别")
        if microchannel else "特征提取 - 单槽 ROI 轮廓识别",
                 x=0.02, y=0.98, ha="left", fontsize=19, fontweight="bold")
    draw_warning_banner(fig, combined_cad_warning(feature, used_rows, summary_row), y=0.875)

    plot_source_contours(ax_image, result_dir, display_rows, display_excluded_rows, feature)
    ax_image.set_title("")
    plot_profile_alignment(
        ax_profile,
        display_rows,
        [],
        feature,
        summary_row,
        reference_label="CAD目标边缘" if microchannel else ("基准槽轮廓" if local_slot else "CAD轮廓"),
        title="单条微流槽检测边缘与 CAD 目标边缘" if microchannel else ("局部槽 ROI 轮廓与基准槽轮廓" if local_slot else "单槽 ROI 轮廓与边界"),
    )
    ax_profile.set_title("")

    target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
    measured_width = slot_value(feature, ["measured_slot_width_mm"])
    width_error = slot_value(feature, ["width_error_um"])
    summary = [
        f"feature={feature.get('feature_id', '--') if feature else '--'}",
        f"status={feature.get('status', '--') if feature else '--'}",
        f"target_width={target_width:.6f} mm" if math.isfinite(target_width) else "target_width=--",
        f"measured_width={measured_width:.6f} mm" if math.isfinite(measured_width) else "measured_width=--",
        f"width_error={width_error:.3f} um" if math.isfinite(width_error) else "width_error=--",
        f"roi_points={len(display_rows)}",
        f"focus_channel={focus_channel}" if focus_channel is not None else "focus_channel=--",
        f"section_coverage={profile_span_coverage_ratio(used_rows, summary_row) * 100.0:.1f}%"
        if math.isfinite(profile_span_coverage_ratio(used_rows, summary_row)) else "section_coverage=--",
    ]
    save_figure(fig, output, dpi)


def load_compensation_rows(result_dir: Path) -> tuple[list[dict[str, str]], dict[str, str], Path]:
    preferred = result_dir / "design_compensation.csv"
    fallback = result_dir / "design_error_profile.csv"
    path = preferred if preferred.exists() else fallback
    rows = read_rows(path)
    summary = read_first_row(result_dir / "design_error_summary.csv")
    used_rows = sort_profile_rows(row for row in rows if to_int(row.get("is_used")) == 1)
    return used_rows, summary, path


def formatted_number(value: float, digits: int = 9) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def slot_edge_curve_rows(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    curve: list[dict[str, float | str]] = []
    for row in rows:
        s = first_number(row, ["s_aligned_mm", "design_s_mm"])
        measured_r = first_number(row, ["measured_r_mm", "r_aligned_mm"])
        compensated_r = first_number(row, ["compensated_target_r_mm", "design_r_mm", "nearest_design_r_mm"])
        compensation_radial_um = first_number(row, ["compensation_radial_um"])
        compensation_normal_um = first_number(row, ["compensation_normal_um"])
        profile_error_um = first_number(row, ["profile_error_um", "normal_error_um", "radial_error_um"])
        normal_error_um = first_number(row, ["normal_error_um", "profile_error_um", "radial_error_um"])
        radial_error_um = first_number(row, ["radial_error_um", "profile_error_um", "normal_error_um"])

        measured_x = first_number(row, ["cad_measured_x_mm", "measured_x_mm", "measured_cad_x_mm"])
        measured_y = first_number(row, ["cad_measured_y_mm", "measured_y_mm", "measured_cad_y_mm"])
        measured_z = first_number(row, ["cad_measured_z_mm", "measured_z_mm", "measured_cad_z_mm"])
        compensated_x = first_number(row, ["compensated_cad_x_mm", "cad_compensation_target_x_mm", "compensation_target_x_mm", "cad_design_x_mm", "design_x_mm"])
        compensated_y = first_number(row, ["compensated_cad_y_mm", "cad_compensation_target_y_mm", "compensation_target_y_mm", "cad_design_y_mm", "design_y_mm"])
        compensated_z = first_number(row, ["compensated_cad_z_mm", "cad_compensation_target_z_mm", "compensation_target_z_mm", "cad_design_z_mm", "design_z_mm"])

        has_xyz = all(math.isfinite(value) for value in (
            measured_x,
            measured_y,
            measured_z,
            compensated_x,
            compensated_y,
            compensated_z,
        ))
        has_section = math.isfinite(s) and math.isfinite(measured_r) and math.isfinite(compensated_r)
        if not (has_xyz or has_section):
            continue

        curve.append({
            "source_index": row.get("index", ""),
            "segment": row.get("segment", ""),
            "is_used": float(to_int(row.get("is_used"))),
            "s_aligned_mm": s,
            "measured_slot_edge_x_mm": measured_x,
            "measured_slot_edge_y_mm": measured_y if math.isfinite(measured_y) else measured_r,
            "measured_slot_edge_z_mm": measured_z,
            "compensated_slot_edge_x_mm": compensated_x,
            "compensated_slot_edge_y_mm": compensated_y if math.isfinite(compensated_y) else compensated_r,
            "compensated_slot_edge_z_mm": compensated_z,
            "measured_r_mm": measured_r,
            "compensated_r_mm": compensated_r,
            "compensation_radial_um": compensation_radial_um,
            "compensation_normal_um": compensation_normal_um,
            "profile_error_um": profile_error_um,
            "normal_error_um": normal_error_um,
            "radial_error_um": radial_error_um,
            "has_cad_xyz": 1.0 if has_xyz else 0.0,
        })

    if any("microchannel" in str(item.get("segment", "")).lower() for item in curve):
        curve.sort(key=lambda item: (
            microchannel_segment_key(item),
            float(item["s_aligned_mm"]) if isinstance(item["s_aligned_mm"], float) and math.isfinite(item["s_aligned_mm"]) else math.inf,
        ))
    else:
        curve.sort(key=lambda item: (
            float(item["s_aligned_mm"]) if isinstance(item["s_aligned_mm"], float) and math.isfinite(item["s_aligned_mm"]) else math.inf,
            str(item["source_index"]),
        ))
    return curve


def curve_series(curve: list[dict[str, float | str]], key: str) -> list[float]:
    values: list[float] = []
    for row in curve:
        value = row.get(key, math.nan)
        values.append(float(value) if isinstance(value, (int, float)) else math.nan)
    return values


def row_float(row: dict[str, float | str], key: str) -> float:
    value = row.get(key, math.nan)
    return float(value) if isinstance(value, (int, float)) else math.nan


def point_match_score(row: dict[str, float | str]) -> float:
    normal = abs(row_float(row, "compensation_normal_um"))
    radial = abs(row_float(row, "compensation_radial_um"))
    if math.isfinite(normal):
        return normal
    if math.isfinite(radial):
        return radial
    return math.inf


def best_rows_by_bin(rows: list[dict[str, float | str]],
                     key: str,
                     bin_width: float) -> list[dict[str, float | str]]:
    if not rows:
        return []
    width = max(bin_width, 1e-6)
    best: dict[int, dict[str, float | str]] = {}
    for row in rows:
        value = row_float(row, key)
        if not math.isfinite(value):
            continue
        bucket = int(round(value / width))
        existing = best.get(bucket)
        if existing is None or point_match_score(row) < point_match_score(existing):
            best[bucket] = row
    return list(best.values())


def slot_ordered_display_curve(curve: list[dict[str, float | str]],
                               feature: dict[str, str] | None) -> list[dict[str, float | str]]:
    if not feature or len(curve) < 24:
        return []
    left_s = slot_value(feature, ["cad_slot_left_s_mm"])
    right_s = slot_value(feature, ["cad_slot_right_s_mm"])
    width = slot_value(feature, ["cad_slot_width_mm", "target_slot_width_mm"])
    depth = slot_value(feature, ["cad_slot_depth_mm", "target_slot_depth_mm"])
    if not all(math.isfinite(value) for value in (left_s, right_s, width)):
        return []

    side_tol = max(0.22, width * 0.055)
    s_bin = max(0.035, width / 180.0)
    r_bin = max(0.035, (depth if math.isfinite(depth) and depth > 1e-9 else width) / 180.0)
    left_rows: list[dict[str, float | str]] = []
    bottom_rows: list[dict[str, float | str]] = []
    right_rows: list[dict[str, float | str]] = []
    for row in curve:
        s = row_float(row, "s_aligned_mm")
        if not math.isfinite(s):
            continue
        if abs(s - left_s) <= side_tol:
            left_rows.append(row)
        elif abs(s - right_s) <= side_tol:
            right_rows.append(row)
        elif left_s + side_tol < s < right_s - side_tol:
            bottom_rows.append(row)

    left_display = best_rows_by_bin(left_rows, "measured_r_mm", r_bin)
    bottom_display = best_rows_by_bin(bottom_rows, "s_aligned_mm", s_bin)
    right_display = best_rows_by_bin(right_rows, "measured_r_mm", r_bin)
    if len(left_display) + len(bottom_display) + len(right_display) < 8:
        return []

    left_display.sort(key=lambda row: row_float(row, "measured_r_mm"), reverse=True)
    bottom_display.sort(key=lambda row: row_float(row, "s_aligned_mm"))
    right_display.sort(key=lambda row: row_float(row, "measured_r_mm"))
    return left_display + bottom_display + right_display


def finite_range(values: Iterable[float]) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite:
        return math.nan, math.nan
    return min(finite), max(finite)


def finite_mean(values: Iterable[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def range_text(values: Iterable[float], digits: int = 3) -> str:
    lo, hi = finite_range(values)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "--"
    return f"{lo:.{digits}f} .. {hi:.{digits}f}"


def infer_cad_roi_label(summary: dict[str, str],
                        compensated_y: list[float]) -> str:
    radial_axis = (summary.get("cad_radial_axis") or "Y").strip() or "Y"
    min_r = to_float(summary.get("design_profile_min_r_mm"))
    max_r = to_float(summary.get("design_profile_max_r_mm"))
    target_min, target_max = finite_range(compensated_y)
    if math.isfinite(min_r) and math.isfinite(target_max) and target_max <= min_r + 2.0:
        return f"CAD {radial_axis} 底部中间 V 型槽口 ROI"
    if math.isfinite(max_r) and math.isfinite(target_min) and target_min >= max_r - 2.0:
        return f"CAD {radial_axis} 上侧槽口 ROI"
    return "CAD 局部槽口 ROI"
    _, target_max = finite_range(compensated_y)
    if math.isfinite(max_r) and math.isfinite(target_max) and target_max >= max_r - 0.5:
        return f"CAD {radial_axis} 正方向上侧中部槽口 ROI"
    if math.isfinite(min_r) and math.isfinite(target_max) and target_max <= min_r + 0.5:
        return f"CAD {radial_axis} 负方向槽口 ROI"
    return "CAD 局部槽口 ROI"


def display_slot_edge_curve(curve: list[dict[str, float | str]],
                            feature: dict[str, str] | None = None) -> list[dict[str, float | str]]:
    if is_microchannel_result(feature, {}):
        return curve
    ordered_curve = slot_ordered_display_curve(curve, feature)
    if ordered_curve:
        return ordered_curve
    if len(curve) < 9:
        return curve

    reference_r: list[float] = []
    for row in curve:
        measured_r = float(row.get("measured_r_mm", math.nan))
        compensated_r = float(row.get("compensated_r_mm", math.nan))
        if math.isfinite(measured_r) and math.isfinite(compensated_r):
            reference_r.append((measured_r + compensated_r) * 0.5)
        elif math.isfinite(measured_r):
            reference_r.append(measured_r)
        else:
            reference_r.append(compensated_r)

    finite_r = finite_values(reference_r)
    if len(finite_r) < 9:
        return curve

    bottom_r = min(finite_r)
    top_r = max(finite_r)
    radial_span = top_r - bottom_r
    if radial_span < 1.0:
        return curve

    bottom_limit = bottom_r + max(0.18, radial_span * 0.055)
    bottom_indices = [
        index
        for index, value in enumerate(reference_r)
        if math.isfinite(value) and value <= bottom_limit
    ]
    if len(bottom_indices) < 3:
        return curve

    center_index = bottom_indices[len(bottom_indices) // 2]
    fold_tolerance = max(0.32, radial_span * 0.075)
    keep_indices: set[int] = set()

    left_min = math.inf
    for index in range(0, center_index + 1):
        value = reference_r[index]
        if not math.isfinite(value):
            continue
        if value <= left_min + fold_tolerance:
            keep_indices.add(index)
            left_min = min(left_min, value)

    right_max = -math.inf
    for index in range(center_index, len(curve)):
        value = reference_r[index]
        if not math.isfinite(value):
            continue
        if value >= right_max - fold_tolerance:
            keep_indices.add(index)
            right_max = max(right_max, value)

    if len(keep_indices) < max(5, int(len(curve) * 0.6)):
        return curve
    return [row for index, row in enumerate(curve) if index in keep_indices]


def draw_key_value_table(ax: plt.Axes,
                         title: str,
                         rows: list[tuple[str, str]],
                         font_size: float = 9.2) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=10, color=COLORS["text"])
    table = ax.table(cellText=rows,
                     colLabels=["项目", "数值"],
                     cellLoc="left",
                     colLoc="left",
                     colWidths=[0.41, 0.59],
                     bbox=[0.0, 0.0, 1.0, 0.93])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.08, 1.72)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["table_edge"])
        cell.set_linewidth(0.70)
        cell.PAD = 0.23
        if row == 0:
            cell.set_facecolor(COLORS["table_header"])
            cell.set_text_props(weight="bold", color=COLORS["text"])
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7FAFC")
            cell.set_text_props(color=COLORS["text"])


def write_slot_edge_compensation_csv(result_dir: Path,
                                     curve: list[dict[str, float | str]],
                                     summary: dict[str, str] | None = None) -> Path:
    output = result_dir / "compensated_slot_edge_points.csv"
    threshold = local_exceedance_threshold_um(summary)
    fieldnames = [
        "source_index",
        "segment",
        "s_aligned_mm",
        "measured_slot_edge_x_mm",
        "measured_slot_edge_y_mm",
        "measured_slot_edge_z_mm",
        "compensated_slot_edge_x_mm",
        "compensated_slot_edge_y_mm",
        "compensated_slot_edge_z_mm",
        "compensated_cad_x_mm",
        "compensated_cad_y_mm",
        "compensated_cad_z_mm",
        "measured_r_mm",
        "compensated_r_mm",
        "profile_error_um",
        "is_local_exceedance",
        "compensation_radial_um",
        "compensation_normal_um",
        "has_cad_xyz",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in curve:
            writer.writerow({
                "source_index": row.get("source_index", ""),
                "segment": row.get("segment", ""),
                "s_aligned_mm": formatted_number(float(row.get("s_aligned_mm", math.nan)), 9),
                "measured_slot_edge_x_mm": formatted_number(float(row.get("measured_slot_edge_x_mm", math.nan)), 9),
                "measured_slot_edge_y_mm": formatted_number(float(row.get("measured_slot_edge_y_mm", math.nan)), 9),
                "measured_slot_edge_z_mm": formatted_number(float(row.get("measured_slot_edge_z_mm", math.nan)), 9),
                "compensated_slot_edge_x_mm": formatted_number(float(row.get("compensated_slot_edge_x_mm", math.nan)), 9),
                "compensated_slot_edge_y_mm": formatted_number(float(row.get("compensated_slot_edge_y_mm", math.nan)), 9),
                "compensated_slot_edge_z_mm": formatted_number(float(row.get("compensated_slot_edge_z_mm", math.nan)), 9),
                "compensated_cad_x_mm": formatted_number(float(row.get("compensated_slot_edge_x_mm", math.nan)), 9),
                "compensated_cad_y_mm": formatted_number(float(row.get("compensated_slot_edge_y_mm", math.nan)), 9),
                "compensated_cad_z_mm": formatted_number(float(row.get("compensated_slot_edge_z_mm", math.nan)), 9),
                "measured_r_mm": formatted_number(float(row.get("measured_r_mm", math.nan)), 9),
                "compensated_r_mm": formatted_number(float(row.get("compensated_r_mm", math.nan)), 9),
                "profile_error_um": formatted_number(float(row.get("profile_error_um", math.nan)), 6),
                "is_local_exceedance": (
                    1 if abs(float(row.get("profile_error_um", math.nan))) > threshold else 0
                ),
                "compensation_radial_um": formatted_number(float(row.get("compensation_radial_um", math.nan)), 6),
                "compensation_normal_um": formatted_number(float(row.get("compensation_normal_um", math.nan)), 6),
                "has_cad_xyz": int(float(row.get("has_cad_xyz", 0.0))),
            })
    return output


def write_all_matched_points_csv(result_dir: Path,
                                 rows: list[dict[str, str]]) -> Path:
    output = result_dir / "slot_roi_all_matched_points.csv"
    fieldnames = [
        "source_index",
        "is_used_for_compensation",
        "s_aligned_mm",
        "measured_r_mm",
        "nearest_design_s_mm",
        "nearest_design_r_mm",
        "normal_error_um",
        "radial_error_um",
        "cad_measured_x_mm",
        "cad_measured_y_mm",
        "cad_measured_z_mm",
        "cad_design_x_mm",
        "cad_design_y_mm",
        "cad_design_z_mm",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in sort_profile_rows(rows):
            s = first_number(row, ["s_aligned_mm", "design_s_mm"])
            measured_r = first_number(row, ["measured_r_mm", "r_aligned_mm"])
            design_s = first_number(row, ["design_s_mm", "nearest_design_s_mm"])
            design_r = first_number(row, ["design_r_mm", "nearest_design_r_mm", "r_design_mm"])
            if not any(math.isfinite(value) for value in (s, measured_r, design_s, design_r)):
                continue
            writer.writerow({
                "source_index": row.get("index", ""),
                "is_used_for_compensation": to_int(row.get("is_used")),
                "s_aligned_mm": formatted_number(s, 9),
                "measured_r_mm": formatted_number(measured_r, 9),
                "nearest_design_s_mm": formatted_number(design_s, 9),
                "nearest_design_r_mm": formatted_number(design_r, 9),
                "normal_error_um": formatted_number(first_number(row, ["normal_error_um"]), 6),
                "radial_error_um": formatted_number(first_number(row, ["radial_error_um"]), 6),
                "cad_measured_x_mm": formatted_number(first_number(row, ["cad_measured_x_mm", "measured_cad_x_mm"]), 9),
                "cad_measured_y_mm": formatted_number(first_number(row, ["cad_measured_y_mm", "measured_cad_y_mm"]), 9),
                "cad_measured_z_mm": formatted_number(first_number(row, ["cad_measured_z_mm", "measured_cad_z_mm"]), 9),
                "cad_design_x_mm": formatted_number(first_number(row, ["cad_design_x_mm", "nearest_design_cad_x_mm"]), 9),
                "cad_design_y_mm": formatted_number(first_number(row, ["cad_design_y_mm", "nearest_design_cad_y_mm"]), 9),
                "cad_design_z_mm": formatted_number(first_number(row, ["cad_design_z_mm", "nearest_design_cad_z_mm"]), 9),
            })
    return output


def read_xyz_series(rows: list[dict[str, str]]) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    s_values: list[float] = []
    design_x: list[float] = []
    design_y: list[float] = []
    design_z: list[float] = []
    measured_x: list[float] = []
    measured_y: list[float] = []
    measured_z: list[float] = []
    delta_x: list[float] = []
    delta_y: list[float] = []
    delta_z: list[float] = []

    for row in rows:
        has_cad = to_int(row.get("has_cad_coordinates")) == 1
        if not has_cad:
            continue
        s = first_number(row, ["s_aligned_mm", "design_s_mm"])
        dx = first_number(row, ["cad_design_x_mm", "design_x_mm", "nearest_design_cad_x_mm"])
        dy = first_number(row, ["cad_design_y_mm", "design_y_mm", "nearest_design_cad_y_mm"])
        dz = first_number(row, ["cad_design_z_mm", "design_z_mm", "nearest_design_cad_z_mm"])
        mx = first_number(row, ["cad_measured_x_mm", "measured_x_mm", "measured_cad_x_mm"])
        my = first_number(row, ["cad_measured_y_mm", "measured_y_mm", "measured_cad_y_mm"])
        mz = first_number(row, ["cad_measured_z_mm", "measured_z_mm", "measured_cad_z_mm"])
        cx = first_number(row, ["delta_x_um", "compensation_x_um", "compensation_delta_cad_x_um"])
        cy = first_number(row, ["delta_y_um", "compensation_y_um", "compensation_delta_cad_y_um"])
        cz = first_number(row, ["delta_z_um", "compensation_z_um", "compensation_delta_cad_z_um"])
        if all(math.isfinite(value) for value in (s, dx, dy, dz, mx, my, mz, cx, cy, cz)):
            s_values.append(s)
            design_x.append(dx)
            design_y.append(dy)
            design_z.append(dz)
            measured_x.append(mx)
            measured_y.append(my)
            measured_z.append(mz)
            delta_x.append(cx)
            delta_y.append(cy)
            delta_z.append(cz)

    order = sorted(range(len(s_values)), key=lambda index: s_values[index])
    return tuple([values[index] for index in order] for values in (
        s_values,
        design_x,
        design_y,
        design_z,
        measured_x,
        measured_y,
        measured_z,
        delta_x,
        delta_y,
        delta_z,
    ))  # type: ignore[return-value]


def axis_equal_3d(ax: plt.Axes, xs: list[float], ys: list[float], zs: list[float]) -> None:
    xlim = padded_limits(xs)
    ylim = padded_limits(ys)
    zlim = padded_limits(zs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except Exception:
        pass


def draw_3d_point_cloud(result_dir: Path, output: Path, dpi: int) -> None:
    all_rows = [row for row in read_rows(result_dir / "design_3d_error_points.csv") if to_int(row.get("is_used")) == 1]
    rows = [row for row in all_rows if to_int(row.get("has_cad_coordinates")) == 1]
    summary = read_first_row(result_dir / "design_error_summary.csv")

    points = []
    for row in rows:
        s = first_number(row, ["s_aligned_mm", "design_s_mm"])
        dx = first_number(row, ["design_x_mm"])
        dy = first_number(row, ["design_y_mm"])
        dz = first_number(row, ["design_z_mm"])
        mx = first_number(row, ["measured_x_mm"])
        my = first_number(row, ["measured_y_mm"])
        mz = first_number(row, ["measured_z_mm"])
        err3d = first_number(row, ["error_3d_um"])
        ex = first_number(row, ["error_x_um"])
        ey = first_number(row, ["error_y_um"])
        ez = first_number(row, ["error_z_um"])
        if all(math.isfinite(value) for value in (s, dx, dy, dz, mx, my, mz)):
            points.append((s, dx, dy, dz, mx, my, mz, err3d, ex, ey, ez))
    if len(points) < 2:
        draw_section_point_cloud(result_dir, output, dpi, all_rows, summary)
        return

    points.sort(key=lambda item: item[0])
    s_values = [point[0] for point in points]
    design_x = [point[1] for point in points]
    design_y = [point[2] for point in points]
    design_z = [point[3] for point in points]
    measured_x = [point[4] for point in points]
    measured_y = [point[5] for point in points]
    measured_z = [point[6] for point in points]
    error_3d = [point[7] for point in points]
    error_x = [point[8] for point in points]
    error_y = [point[9] for point in points]
    error_z = [point[10] for point in points]

    color_values = [
        value if math.isfinite(value) else 0.0
        for value in error_3d
    ]

    fig = plt.figure(figsize=figure_size_inches(13.8, 8.8))
    grid = fig.add_gridspec(2, 2,
                            left=0.070, right=0.965, top=0.845, bottom=0.105,
                            height_ratios=[1.18, 1.0],
                            hspace=0.58,
                            wspace=0.42)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_error = fig.add_subplot(grid[0, 1])
    ax_xy = fig.add_subplot(grid[1, 0])
    ax_note = fig.add_subplot(grid[1, 1])

    fig.suptitle("3D点云 - CAD目标点与检测映射点", x=0.02, y=0.985,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.947,
             "数据来源：design_3d_error_points.csv；该图只在 CAD 三维坐标对齐成功后生成。",
             fontsize=9.5, color="#425466")

    ax_3d.scatter(design_x, design_y, design_z, color=COLORS["design"], s=16, alpha=0.82, label="CAD目标点云")
    cloud = ax_3d.scatter(measured_x, measured_y, measured_z, c=color_values, cmap="magma",
                          s=18, alpha=0.90, label="检测点误差幅值")
    stride = max(1, len(points) // 28)
    for index in range(0, len(points), stride):
        ax_3d.plot([measured_x[index], design_x[index]],
                   [measured_y[index], design_y[index]],
                   [measured_z[index], design_z[index]],
                   color="#6B7280", linewidth=0.42, alpha=0.32)
    ax_3d.set_xlabel("CAD X / mm", labelpad=7, fontsize=10.6)
    ax_3d.set_ylabel("CAD Y / mm", labelpad=7, fontsize=10.6)
    ax_3d.set_zlabel("")
    ax_3d.view_init(elev=24, azim=-58)
    try:
        ax_3d.set_proj_type("ortho")
    except Exception:
        pass
    ax_3d.grid(True)
    ax_3d.tick_params(labelsize=8.2, pad=1)
    axis_equal_3d(ax_3d, [*design_x, *measured_x], [*design_y, *measured_y], [*design_z, *measured_z])
    ax_3d.set_zticks([])
    show_legend_if_any(ax_3d, loc="upper left", fontsize=9.4)
    cbar = fig.colorbar(cloud, ax=ax_3d, shrink=0.58, pad=0.16)
    cbar.set_label("")
    cbar.ax.set_title("3D误差 / um", fontsize=9.8, pad=8)
    cbar.ax.tick_params(labelsize=9.2)

    if any(math.isfinite(value) for value in error_3d):
        plot_segmented_line(ax_error, s_values, error_3d, "3D误差", color="#C76334", linewidth=1.75)
    if any(math.isfinite(value) for value in error_x):
        plot_segmented_line(ax_error, s_values, error_x, "X误差", color=COLORS["delta_x"], linewidth=1.35, alpha=0.84)
    if any(math.isfinite(value) for value in error_y):
        plot_segmented_line(ax_error, s_values, error_y, "Y误差", color=COLORS["delta_y"], linewidth=1.35, alpha=0.84)
    if any(math.isfinite(value) for value in error_z):
        plot_segmented_line(ax_error, s_values, error_z, "Z误差", color=COLORS["delta_z"], linewidth=1.35, alpha=0.84)
    ax_error.axhline(0.0, color="#777777", linewidth=0.8)
    ax_error.set_title("三维坐标误差（按 s 展开）")
    ax_error.set_xlabel("统一截面坐标 s / mm")
    ax_error.set_ylabel("误差 / um")
    ax_error.grid(True)
    style_plot_axes(ax_error)
    ax_error.set_title("")
    show_legend_if_any(ax_error, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=9.6)

    ax_xy.scatter(design_x, design_y, color=COLORS["design"], s=12, alpha=0.70, label="CAD目标XY")
    ax_xy.scatter(measured_x, measured_y, color=COLORS["measured"], s=12, alpha=0.70, label="检测映射XY")
    ax_xy.set_title("XY投影点云")
    ax_xy.set_xlabel("CAD X / mm")
    ax_xy.set_ylabel("CAD Y / mm")
    ax_xy.set_xlim(*padded_limits([*design_x, *measured_x]))
    ax_xy.set_ylim(*padded_limits([*design_y, *measured_y]))
    ax_xy.grid(True)
    style_plot_axes(ax_xy)
    show_legend_if_any(ax_xy, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9.6)

    finite_error_3d = finite_values(error_3d)
    finite_abs_components = finite_values([abs(value) for value in [*error_x, *error_y, *error_z]])
    ax_note.axis("off")
    lines = [
        "点云说明",
        f"有效三维对齐点: {len(points)}",
        f"CAD来源: {summary.get('design_source_type', '--')}",
        f"CAD提取: {summary.get('cad_extraction_method', '--')}",
        f"轴向/径向: {summary.get('cad_axial_axis', '--')} / {summary.get('cad_radial_axis', '--')}",
        "",
        "统计",
        f"3D误差均值: {sum(finite_error_3d) / len(finite_error_3d):.3f} um" if finite_error_3d else "3D误差均值: --",
        f"3D误差最大值: {max(finite_error_3d):.3f} um" if finite_error_3d else "3D误差最大值: --",
        f"XYZ分量最大绝对值: {max(finite_abs_components):.3f} um" if finite_abs_components else "XYZ分量最大绝对值: --",
        f"轮廓RMSE: {metric(summary, 'profile_rms_um')} um",
        f"轮廓P95: {metric(summary, 'profile_p95_abs_um')} um",
        "",
        "字段对应",
        "design_x/y/z_mm: CAD目标点",
        "measured_x/y/z_mm: 检测映射点",
        "error_x/y/z_um: 检测点 - CAD目标点",
    ]
    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10.2)

    fig.text(0.02,
             0.018,
             generated_footer(result_dir, "design_3d_error_points.csv"),
             fontsize=8.8,
             color="#333333")
    save_figure(fig, output, dpi)


def draw_section_point_cloud(result_dir: Path,
                             output: Path,
                             dpi: int,
                             rows: list[dict[str, str]],
                             summary: dict[str, str]) -> None:
    points = []
    for row in rows:
        s = first_number(row, ["s_aligned_mm", "design_s_mm"])
        measured_r = first_number(row, ["measured_r_mm"])
        design_s = first_number(row, ["design_s_mm", "s_aligned_mm"])
        design_r = first_number(row, ["design_r_mm"])
        normal_error = first_number(row, ["normal_error_um", "radial_error_um", "profile_error_um"])
        if math.isfinite(s) and math.isfinite(measured_r):
            points.append((s, measured_r, design_s, design_r, normal_error))
    if len(points) < 2:
        raise RuntimeError("not enough section points for point cloud plot")

    points.sort(key=lambda item: item[0])
    s_values = [point[0] for point in points]
    measured_r = [point[1] for point in points]
    design_s = [point[2] for point in points if math.isfinite(point[2]) and math.isfinite(point[3])]
    design_r = [point[3] for point in points if math.isfinite(point[2]) and math.isfinite(point[3])]
    errors = [point[4] for point in points]
    color_values = [value if math.isfinite(value) else 0.0 for value in errors]
    layer = [0.0 for _ in s_values]
    design_layer = [0.0 for _ in design_s]

    fig = plt.figure(figsize=figure_size_inches(12.0, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.34, wspace=0.24)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_error = fig.add_subplot(grid[0, 1])
    ax_2d = fig.add_subplot(grid[1, 0])
    ax_note = fig.add_subplot(grid[1, 1])

    fig.suptitle("点云 - 统一截面 2.5D 轮廓点", x=0.02, y=0.985,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.947,
             "当前 design_3d_error_points.csv 未包含可用 CAD XYZ 映射点，因此显示统一截面点云；这不是完整 CAD 三维点云。",
             fontsize=9.5, color="#8A5A00")

    if design_s and design_r:
        ax_3d.scatter(design_s, design_r, design_layer, color=COLORS["design"], s=13, alpha=0.72, label="CAD截面点")
    cloud = ax_3d.scatter(s_values, measured_r, layer, c=color_values, cmap="coolwarm",
                          s=16, alpha=0.86, label="检测截面点")
    ax_3d.set_xlabel("截面坐标 s / mm")
    ax_3d.set_ylabel("截面坐标 r / mm")
    ax_3d.set_zlabel("点云层")
    ax_3d.view_init(elev=28, azim=-58)
    try:
        ax_3d.set_proj_type("ortho")
    except Exception:
        pass
    ax_3d.set_zlim(-1.0, 1.0)
    ax_3d.grid(True)
    show_legend_if_any(ax_3d, loc="upper left", fontsize=8)
    cbar = fig.colorbar(cloud, ax=ax_3d, shrink=0.68, pad=0.02)
    cbar.set_label("截面误差 / um")

    finite_errors = finite_values(errors)
    if finite_errors:
        plot_segmented_line(ax_error, s_values[:len(errors)], errors, "截面误差", color=COLORS["normal"], linewidth=1.3)
    ax_error.axhline(0.0, color="#777777", linewidth=0.8)
    ax_error.set_title("截面误差（按 s 展开）")
    ax_error.set_xlabel("统一截面坐标 s / mm")
    ax_error.set_ylabel("误差 / um")
    ax_error.grid(True)
    show_legend_if_any(ax_error, loc="best", fontsize=8)

    if design_s and design_r:
        plot_segmented_line(ax_2d, design_s, design_r, "CAD截面", color=COLORS["design"], linewidth=1.4)
    ax_2d.scatter(s_values, measured_r, color=COLORS["measured"], s=12, alpha=0.75, label="检测截面")
    ax_2d.set_title("截面点云投影")
    ax_2d.set_xlabel("截面坐标 s / mm")
    ax_2d.set_ylabel("截面坐标 r / mm")
    ax_2d.grid(True)
    show_legend_if_any(ax_2d, loc="best", fontsize=8)

    ax_note.axis("off")
    lines = [
        "状态说明",
        "未形成 CAD XYZ 点云映射。",
        f"可显示截面点数: {len(points)}",
        f"CAD来源: {summary.get('design_source_type', '--')}",
        f"CAD提取: {summary.get('cad_extraction_method', '--')}",
        "",
        "可能原因",
        "检测轮廓 s 范围超出 CAD 采样范围",
        "CAD截面没有可识别槽特征",
        "真实标定/临时像素当量与图像不匹配",
        "",
        "完整3D点云需要",
        "has_cad_coordinates=1",
        "design_x/y/z_mm 与 measured_x/y/z_mm 有效",
    ]
    if finite_errors:
        lines.extend([
            "",
            f"截面误差均值: {sum(finite_errors) / len(finite_errors):.3f} um",
            f"截面误差最大绝对值: {max(abs(value) for value in finite_errors):.3f} um",
        ])
    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10.2)
    fig.text(0.02,
             0.018,
             generated_footer(result_dir, "design_3d_error_points.csv"),
             fontsize=8.8,
             color="#333333")
    save_figure(fig, output, dpi)


def draw_compensation_3d(result_dir: Path,
                         output: Path,
                         dpi: int,
                         rows: list[dict[str, str]],
                         summary: dict[str, str],
                          source_path: Path,
                          feature: dict[str, str] | None,
                          coordinate_csv: Path,
                          all_matched_csv: Path) -> None:
    curve = slot_edge_curve_rows(rows)
    display_curve = display_slot_edge_curve(curve, feature)
    s_values = curve_series(display_curve, "s_aligned_mm")
    measured_x = curve_series(display_curve, "measured_slot_edge_x_mm")
    measured_y = curve_series(display_curve, "measured_slot_edge_y_mm")
    compensated_x = curve_series(display_curve, "compensated_slot_edge_x_mm")
    compensated_y = curve_series(display_curve, "compensated_slot_edge_y_mm")
    measured_r = curve_series(display_curve, "measured_r_mm")
    compensated_r = curve_series(display_curve, "compensated_r_mm")
    all_curve = slot_edge_curve_rows(read_rows(source_path))
    has_xyz = len(finite_values([*measured_x, *measured_y, *compensated_x, *compensated_y])) >= 4
    if len(curve) < 2 or not has_xyz:
        draw_compensation_2d(result_dir, output, dpi, rows, summary, source_path, feature, coordinate_csv, all_matched_csv)
        return

    fig = plt.figure(figsize=figure_size_inches(9.4, 5.9))
    grid = fig.add_gridspec(1, 2,
                            left=0.075, right=0.975, top=0.735, bottom=0.145,
                            width_ratios=[1.42, 0.92],
                            wspace=0.20)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_note = fig.add_subplot(grid[0, 1])

    roi_label = infer_cad_roi_label(summary, compensated_y)
    fig.suptitle("补偿结算 - 单槽槽边缘补偿", x=0.02, y=0.982, ha="left",
                 fontsize=19, fontweight="bold")
    fig.text(0.02, 0.910,
             f"蓝线为检测到的槽边缘；绿线为对应 CAD 目标槽边缘/补偿后曲线。当前对应：{roi_label}。",
             fontsize=12.8, color="#425466")
    warning_text = combined_cad_warning(feature, rows, summary)
    draw_warning_banner(fig, warning_text, y=0.842)

    if len(all_curve) > len(curve):
        all_measured_x = curve_series(all_curve, "measured_slot_edge_x_mm")
        all_measured_y = curve_series(all_curve, "measured_slot_edge_y_mm")
        ax_xy.scatter(all_measured_x,
                      all_measured_y,
                      s=5,
                      alpha=0.18,
                      color=COLORS["excluded"],
                      label="全量亚像素-CAD近邻匹配点")
    plot_segmented_line(ax_xy, measured_x, measured_y, "检测槽边缘曲线",
                        scatter_fallback=False,
                        color=COLORS["measured"], linewidth=2.7)
    plot_segmented_line(ax_xy, compensated_x, compensated_y, "CAD目标/补偿后曲线",
                        scatter_fallback=False,
                        color=COLORS["design"], linewidth=2.6, linestyle="--")
    ax_xy.set_title("")
    ax_xy.set_xlabel("CAD X / mm")
    ax_xy.set_ylabel("CAD Y / mm")
    ax_xy.set_xlim(*padded_limits([*measured_x, *compensated_x]))
    ax_xy.set_ylim(*padded_limits([*measured_y, *compensated_y]))
    ax_xy.grid(True)
    ax_xy.tick_params(labelsize=11.8)
    show_legend_if_any(ax_xy, loc="upper center", fontsize=11.4)

    measured_y_min, measured_y_max = finite_range(measured_y)
    compensated_y_min, compensated_y_max = finite_range(compensated_y)
    bottom_gap = (
        abs(measured_y_min - compensated_y_min)
        if math.isfinite(measured_y_min) and math.isfinite(compensated_y_min)
        else math.nan
    )
    bottom_aligned = math.isfinite(bottom_gap) and bottom_gap <= 0.35
    if math.isfinite(measured_y_min) and math.isfinite(compensated_y_min):
        ax_xy.annotate("检测已贴近槽底/槽壁内轮廓" if bottom_aligned else "检测主要落在槽口/槽壁上半段",
                       xy=(finite_mean(measured_x), measured_y_min),
                       xytext=(0.03, 0.08),
                       textcoords="axes fraction",
                       fontsize=11.4,
                       color=COLORS["measured"],
                       arrowprops={"arrowstyle": "->", "color": COLORS["measured"], "lw": 1.1})
        ax_xy.annotate("CAD 槽底与检测槽底基本对齐" if bottom_aligned else "CAD 目标包含更深的槽底",
                       xy=(finite_mean(compensated_x), compensated_y_min),
                       xytext=(0.45, 0.26),
                       textcoords="axes fraction",
                       fontsize=11.4,
                       color=COLORS["design"],
                       arrowprops={"arrowstyle": "->", "color": COLORS["design"], "lw": 1.1})

    target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
    measured_width = slot_value(feature, ["measured_slot_width_mm"])
    width_error = slot_value(feature, ["width_error_um"])
    local_slot = is_local_slot_result(feature, summary)
    coverage_ratio = math.nan if local_slot else profile_span_coverage_ratio(rows, summary)
    section_axis = summary.get("cad_section_normal_axis", "--")
    section_coordinate = metric(summary, "cad_section_coordinate_mm", 3)
    ax_note.axis("off")
    note_lines = [
        "对应 CAD 区域",
        roi_label,
        f"截面: {section_axis} = {section_coordinate} mm",
        f"检测 X: {range_text(measured_x)} mm",
        f"检测 Y: {range_text(measured_y)} mm",
        f"CAD目标 Y: {range_text(compensated_y)} mm",
        "",
        "补偿结算",
        f"结果: {compensation_result_nature(feature, rows, summary)}",
        f"目标槽宽: {target_width:.6f} mm" if math.isfinite(target_width) else "目标槽宽: --",
        f"检测槽宽: {measured_width:.6f} mm" if math.isfinite(measured_width) else "检测槽宽: --",
        f"槽宽误差: {width_error:.3f} um" if math.isfinite(width_error) else "槽宽误差: --",
        f"有效点: {len(curve)}",
        "评价范围: 单槽宽度验收" if local_slot else
        (f"覆盖率: {coverage_ratio * 100.0:.1f}%" if math.isfinite(coverage_ratio) else "覆盖率: --"),
        "",
        "误差说明",
        "当前蓝线已贴近槽底/槽壁内轮廓；" if bottom_aligned else "当前蓝线主要是槽口/槽壁上半段；",
        f"CAD目标槽底最低约 Y={compensated_y_min:.3f} mm，" if math.isfinite(compensated_y_min) else "CAD目标槽底: --",
        f"检测最低约 Y={measured_y_min:.3f} mm。" if math.isfinite(measured_y_min) else "检测最低: --",
        "剩余补偿主要来自边缘采样与宽度差。" if bottom_aligned else "因此深度方向补偿会明显偏大。",
        "",
        "输出",
        coordinate_csv.name,
        all_matched_csv.name,
    ]
    ax_note.text(0.0, 1.0, "\n".join(note_lines),
                 va="top", ha="left", fontsize=12.3, linespacing=1.28,
                 bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F7FAFC",
                       "edgecolor": "#D7DBE0", "linewidth": 0.9})

    save_figure(fig, output, dpi)


def draw_compensation_2d(result_dir: Path,
                         output: Path,
                         dpi: int,
                         rows: list[dict[str, str]],
                         summary: dict[str, str],
                         source_path: Path,
                         feature: dict[str, str] | None,
                         coordinate_csv: Path,
                         all_matched_csv: Path) -> None:
    curve = slot_edge_curve_rows(rows)
    if len(curve) < 2:
        draw_fallback_compensation(result_dir, output, dpi)
        return
    all_curve = slot_edge_curve_rows(read_rows(source_path))
    display_curve = display_slot_edge_curve(curve, feature)
    microchannel = is_microchannel_result(feature, summary)
    s_values = curve_series(display_curve, "s_aligned_mm")
    measured_r = curve_series(display_curve, "measured_r_mm")
    compensated_r = curve_series(display_curve, "compensated_r_mm")

    fig = plt.figure(figsize=figure_size_inches(14.0, 9.8))
    grid = fig.add_gridspec(2, 2,
                            left=0.062, right=0.982, top=0.735, bottom=0.060,
                            height_ratios=[1.00, 1.30],
                            hspace=0.46,
                            wspace=0.28)
    ax_profile = fig.add_subplot(grid[0, :])
    ax_feature = fig.add_subplot(grid[1, 0])
    ax_note = fig.add_subplot(grid[1, 1])
    fig.suptitle("补偿结算 - 微流道边缘曲线补偿" if microchannel else "补偿结算 - 槽边缘曲线补偿", x=0.02, y=0.985,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(0.02, 0.925,
             "展示检测微流道边缘曲线与补偿后 CAD 目标边缘曲线；补偿后实际 XYZ 坐标已写入 CSV。"
             if microchannel else
             "展示检测槽边缘曲线与补偿后槽边缘曲线；补偿后实际槽边缘点坐标已写入 CSV。",
             fontsize=11.4, color="#425466")
    warning_text = combined_cad_warning(feature, rows, summary)
    draw_warning_banner(fig, warning_text, y=0.875)

    if len(all_curve) > len(curve):
        all_s = curve_series(all_curve, "s_aligned_mm")
        all_r = curve_series(all_curve, "measured_r_mm")
        ax_profile.scatter(all_s,
                           all_r,
                           s=3,
                           alpha=0.07,
                           color=COLORS["excluded"],
                           label="全量亚像素-CAD近邻匹配点")
    if s_values and measured_r:
        plot_segmented_line(ax_profile,
                            s_values,
                            measured_r,
                            "检测槽边缘曲线",
                            scatter_fallback=False,
                            color=COLORS["measured"],
                            linewidth=2.65)
    if s_values and compensated_r:
        plot_segmented_line(ax_profile,
                            s_values,
                            compensated_r,
                            "补偿后槽边缘曲线",
                            scatter_fallback=False,
                            color=COLORS["design"],
                            linewidth=2.45,
                            linestyle="--")
    exceedance_stats = (
        local_exceedance_stats(curve, summary)
        if microchannel else
        plot_exceedance_segments(ax_profile,
                                 curve,
                                 ("s_aligned_mm",),
                                 ("measured_r_mm",),
                                 summary)
    )
    ax_profile.set_title("微流道边缘曲线（按通道长度展开）" if microchannel else "槽边缘曲线（统一截面）")
    ax_profile.set_xlabel("通道长度方向 / mm" if microchannel else "统一截面坐标 s / mm")
    ax_profile.set_ylabel("通道横向边缘位置 / mm" if microchannel else "截面坐标 r / mm")
    ax_profile.set_title("")
    style_plot_axes(ax_profile)
    show_legend_if_any(ax_profile, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=11.3)

    ax_feature.axis("off")
    target_width = slot_value(feature, ["target_slot_width_mm", "cad_slot_width_mm"])
    measured_width = slot_value(feature, ["measured_slot_width_mm"])
    width_error = slot_value(feature, ["width_error_um"])
    compensation_width = slot_value(feature, ["compensation_width_um"])
    feature_rows = [
        ("类型", "微流道阵列补偿" if microchannel else ("槽特征补偿" if feature else "轮廓补偿")),
        ("特征ID", feature.get("feature_id", "--") if feature else "--"),
        ("状态", feature.get("status", "--") if feature else "--"),
        ("目标槽宽", f"{target_width:.6f} mm" if math.isfinite(target_width) else "--"),
        ("检测槽宽", f"{measured_width:.6f} mm" if math.isfinite(measured_width) else "--"),
        ("槽宽误差", f"{width_error:.3f} um" if math.isfinite(width_error) else "--"),
        ("宽度补偿", f"{compensation_width:.3f} um" if math.isfinite(compensation_width) else "--"),
    ]
    note = feature.get("notes", "") if feature else ""
    if note:
        compact_note = " ".join(note.split())
        if len(compact_note) > 42:
            compact_note = compact_note[:39] + "..."
        feature_rows.append(("说明", compact_note))
    draw_key_value_table(ax_feature, "槽特征表", feature_rows, font_size=10.6)

    local_slot = is_local_slot_result(feature, summary)
    coverage_ratio = math.nan if local_slot else profile_span_coverage_ratio(rows, summary)
    note_rows = [
        ("结果性质", compensation_result_nature(feature, rows, summary)),
        ("原始补偿CSV", source_path.name),
        ("槽边缘坐标CSV", coordinate_csv.name),
        ("全量匹配CSV", all_matched_csv.name),
        ("检测横向字段" if microchannel else "检测截面字段", "measured_r_mm"),
        ("补偿后横向字段" if microchannel else "补偿后字段", "compensated_r_mm"),
        ("CAD补偿后坐标", "compensated_cad_x/y/z_mm"),
        ("有效/显示点", f"{len(curve)} / {len(display_curve)}"),
        ("过滤点数/比例", f"{summary.get('outlier_count', '--')} / {metric(summary, 'outlier_ratio', 3)}"),
        ("局部超差阈值", f"±{exceedance_stats['threshold_um']:.1f} um"),
        ("局部超差点", f"{int(round(exceedance_stats['count']))}"),
        ("最大绝对误差", f"{exceedance_stats['max_abs_um']:.3f} um" if math.isfinite(exceedance_stats["max_abs_um"]) else "--"),
        ("评价范围", "微流道阵列边缘" if microchannel else (f"{coverage_ratio * 100.0:.1f}%" if math.isfinite(coverage_ratio) else "--")),
        ("RMSE", f"{metric(summary, 'profile_rms_um')} um"),
        ("P95", f"{metric(summary, 'profile_p95_abs_um')} um"),
    ]
    draw_key_value_table(ax_note, "补偿输出表", note_rows, font_size=9.6)
    save_figure(fig, output, dpi)


def draw_compensation(result_dir: Path, output: Path, dpi: int) -> None:
    rows, summary, source_path = load_compensation_rows(result_dir)
    feature = find_slot_feature(result_dir)
    curve = slot_edge_curve_rows(rows)
    coordinate_csv = write_slot_edge_compensation_csv(result_dir, curve, summary)
    all_source_rows = read_rows(source_path)
    all_matched_csv = write_all_matched_points_csv(result_dir, all_source_rows)
    local_slot = is_local_slot_result(feature, summary)
    has_3d = any(float(row.get("has_cad_xyz", 0.0)) > 0.5 for row in curve)
    if has_3d and not local_slot:
        draw_compensation_3d(result_dir, output, dpi, rows, summary, source_path, feature, coordinate_csv, all_matched_csv)
    else:
        draw_compensation_2d(result_dir, output, dpi, rows, summary, source_path, feature, coordinate_csv, all_matched_csv)


def main() -> None:
    global ARGS
    args = parse_args()
    ARGS = args
    configure_matplotlib(args.dpi)
    result_dir = args.result_dir
    if args.contour_output:
        draw_single_slot_contour(result_dir, args.contour_output, args.dpi)
    if args.comparison_output:
        draw_comparison(result_dir, args.comparison_output, args.dpi)
    if args.compensation_output:
        draw_compensation(result_dir, args.compensation_output, args.dpi)
    if args.pointcloud_output:
        draw_3d_point_cloud(result_dir, args.pointcloud_output, args.dpi)


if __name__ == "__main__":
    main()
