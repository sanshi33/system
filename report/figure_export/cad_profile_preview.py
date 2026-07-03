#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARGS: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw imported CAD profile preview with matplotlib.")
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="CAD 真实截面预览")
    parser.add_argument("--axial-axis", default="")
    parser.add_argument("--radial-axis", default="")
    parser.add_argument("--section-axis", default="")
    parser.add_argument("--section-coordinate", default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--width-px", type=int)
    parser.add_argument("--height-px", type=int)
    return parser.parse_args()


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def padded_limits(values: list[float], minimum_span: float = 1.0) -> tuple[float, float]:
    finite_values = finite(values)
    if not finite_values:
        return -1.0, 1.0
    lo = min(finite_values)
    hi = max(finite_values)
    span = max(minimum_span, hi - lo)
    pad = span * 0.10
    return lo - pad, hi + pad


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
    jump_limit = max(1.5, typical_step * 14.0, 0.28 * max(x_span, y_span, 1.0))
    vertical_wall_dx_limit = max(0.18, typical_step * 3.5, 0.02 * max(x_span, 1.0))
    vertical_wall_dy_limit = max(2.4, 1.12 * max(y_span, 1.0))

    def append_if_visual_segment(seg_x: list[float], seg_y: list[float]) -> None:
        if len(seg_x) < 2:
            return
        segments.append((seg_x, seg_y))

    segments: list[tuple[list[float], list[float]]] = []
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


def configure_matplotlib(dpi: int) -> None:
    effective_dpi = max(96, int(dpi))
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
        "font.size": 15.0,
        "axes.titlesize": 18.5,
        "axes.labelsize": 16.0,
        "xtick.labelsize": 13.8,
        "ytick.labelsize": 13.8,
        "legend.fontsize": 13.6,
        "figure.dpi": effective_dpi,
        "savefig.dpi": effective_dpi,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.05,
        "grid.color": "#D6DEE6",
        "grid.alpha": 0.50,
        "grid.linewidth": 0.75,
        "legend.frameon": False,
        "axes.formatter.useoffset": False,
    })


def draw_projection(ax: plt.Axes,
                    xs: list[float],
                    ys: list[float],
                    color_values: list[float],
                    xlabel: str,
                    ylabel: str,
                    title: str) -> None:
    ax.scatter(xs, ys, c=color_values, cmap="viridis", s=8, alpha=0.62, edgecolors="none", rasterized=True)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*padded_limits(xs))
    ax.set_ylim(*padded_limits(ys))
    ax.grid(True)


def draw_profile_preview(args: argparse.Namespace,
                         cad_points: list[tuple[float, float, float, float, float]]) -> None:
    s_values = [point[0] for point in cad_points]
    r_values = [point[1] for point in cad_points]
    x_values = [point[2] for point in cad_points]
    y_values = [point[3] for point in cad_points]
    z_values = [point[4] for point in cad_points]
    color_values = s_values if s_values else list(range(len(x_values)))

    fig = plt.figure(figsize=figure_size_inches(15.8, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.965, top=0.80, bottom=0.16,
                            hspace=0.42, wspace=0.30)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_yz = fig.add_subplot(grid[0, 2])
    ax_sr = fig.add_subplot(grid[1, :2])
    ax_info = fig.add_subplot(grid[1, 2])

    fig.suptitle(args.title, x=0.055, y=0.965, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.055, 0.905,
             "静态图优先显示 CAD 模型 XYZ 采样点投影；完整零件外形请查看“CAD零件3D”页签。",
             fontsize=16, color="#34495E")
    fig.text(0.055, 0.860,
             "XY/XZ/YZ 投影均为 CAD 模型坐标点，用于观察微流槽所在区域和后续检测对齐范围。",
             fontsize=15, color="#52616B")

    draw_projection(ax_xy, x_values, y_values, color_values, "X / mm", "Y / mm", "XY 投影")
    draw_projection(ax_xz, x_values, z_values, color_values, "X / mm", "Z / mm", "XZ 投影")
    draw_projection(ax_yz, y_values, z_values, color_values, "Y / mm", "Z / mm", "YZ 投影")

    sr_points = sorted((s, r) for s, r in zip(s_values, r_values) if math.isfinite(s) and math.isfinite(r))
    sr_s = [point[0] for point in sr_points]
    sr_y = [point[1] for point in sr_points]
    ax_sr.scatter(sr_s, sr_y, color="#17936F", s=5, alpha=0.34, edgecolors="none", rasterized=True)
    ax_sr.set_title("CAD XY 采样点展开", pad=10, fontweight="bold")
    ax_sr.set_xlabel("X / mm")
    radial_label = f"{args.radial_axis} / mm" if args.radial_axis else "截面径向坐标 / mm"
    ax_sr.set_ylabel(radial_label)
    ax_sr.set_xlim(*padded_limits(sr_s))
    ax_sr.set_ylim(*padded_limits(sr_y))
    ax_sr.grid(True)

    section_coordinate = to_float(args.section_coordinate)
    section_text = "未指定"
    if args.section_axis and math.isfinite(section_coordinate):
        section_text = f"{args.section_axis} = {section_coordinate:.3f} mm"

    ax_info.axis("off")
    info_lines = [
        "图示说明",
        "",
        f"采样点数：{len(cad_points)}",
        "数据源：CAD XYZ 采样点",
        f"轴向：{args.axial_axis or '--'}",
        f"径向：{args.radial_axis or '--'}",
        f"截面位置：{section_text}",
        "",
        "CAD 模型 XYZ 范围",
        f"X：{min(x_values):.3f} ~ {max(x_values):.3f} mm",
        f"Y：{min(y_values):.3f} ~ {max(y_values):.3f} mm",
        f"Z：{min(z_values):.3f} ~ {max(z_values):.3f} mm",
        f"径向：{min(r_values):.3f} ~ {max(r_values):.3f} mm",
    ]
    ax_info.text(0.0, 1.0, "\n".join(info_lines), va="top", fontsize=15.5,
                 linespacing=1.45, color="#263238")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_profile_fallback(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    s_values = [to_float(row.get("s_mm")) for row in rows]
    r_values = [to_float(row.get("r_mm")) for row in rows]
    points = sorted((s, r) for s, r in zip(s_values, r_values) if math.isfinite(s) and math.isfinite(r))
    if not points:
        raise RuntimeError("no finite CAD profile points")
    s_profile = [point[0] for point in points]
    r_profile = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=figure_size_inches(13.2, 7.2), constrained_layout=True)
    fig.suptitle(args.title, x=0.02, y=0.98, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.02, 0.91,
             "当前 CAD 采样缺少可用 XYZ 点，先显示截面二维轮廓；完整零件外形请查看“CAD零件3D”页签。",
             fontsize=15, color="#8A5A00")
    first_segment = True
    for seg_s, seg_r in segmented_lines(s_profile, r_profile):
        ax.plot(seg_s,
                seg_r,
                color="#17936F",
                linewidth=2.4,
                label="截面轮廓" if first_segment else None)
        first_segment = False
    ax.scatter(s_profile, r_profile, color="#17936F", s=24, alpha=0.76)
    ax.set_xlabel("采样位置 / mm")
    ax.set_ylabel("径向尺寸 / mm")
    ax.set_xlim(*padded_limits(s_profile))
    ax.set_ylim(*padded_limits(r_profile))
    ax.grid(True)
    ax.legend(loc="best")
    fig.text(0.02, 0.02, f"采样点数：{len(points)}", fontsize=13, color="#333333")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global ARGS
    args = parse_args()
    ARGS = args
    configure_matplotlib(args.dpi)
    rows = [row for row in read_rows(args.profile_csv) if row.get("has_cad_point", "1") != "0"]
    cad_points = []
    for row in rows:
        s = to_float(row.get("s_mm"))
        r = to_float(row.get("r_mm"))
        x = to_float(row.get("cad_x_mm"))
        y = to_float(row.get("cad_y_mm"))
        z = to_float(row.get("cad_z_mm"))
        if all(math.isfinite(value) for value in (s, r, x, y, z)):
            cad_points.append((s, r, x, y, z))

    if len(cad_points) >= 2:
        draw_profile_preview(args, cad_points)
    else:
        draw_profile_fallback(args, rows)


if __name__ == "__main__":
    main()
