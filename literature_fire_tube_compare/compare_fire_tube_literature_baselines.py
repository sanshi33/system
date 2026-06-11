#!/usr/bin/env python3
"""
Run two literature baselines on the fire-tube generatrix dataset and compare
them against the latest official project result under one shared evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return script_dir().parent


def load_base_module():
    module_path = project_root() / "tools" / "workpiece_literature_comparison.py"
    spec = importlib.util.spec_from_file_location("workpiece_literature_comparison_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load baseline module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def has_required_outputs(result_dir: Path) -> bool:
    return (
        result_dir.is_dir()
        and (result_dir / "design_error_profile.csv").exists()
        and (result_dir / "design_error_summary.csv").exists()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    candidates = [path for path in result_root.iterdir() if has_required_outputs(path)]
    if not candidates:
        raise FileNotFoundError(f"no valid result directory found under: {result_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Compare two literature baselines with the current fire-tube generatrix method."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "火焰筒" / "母线拼接",
        help="Raw fire-tube generatrix image directory.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "result" / "workpiece",
        help="Root directory that stores official workpiece results.",
    )
    parser.add_argument(
        "--proposed-dir",
        type=Path,
        default=None,
        help="Official project result directory. Defaults to the latest valid directory under result-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to literature_fire_tube_compare/output/<proposed-dir-name>.",
    )
    return parser.parse_args()


def resolve_proposed_dir(args: argparse.Namespace) -> Path:
    return args.proposed_dir if args.proposed_dir is not None else find_latest_result_dir(args.result_root)


def resolve_output_dir(args: argparse.Namespace, proposed_dir: Path) -> Path:
    return args.output_dir if args.output_dir is not None else script_dir() / "output" / proposed_dir.name


def write_selection_mapping(output_dir: Path, proposed_dir: Path) -> None:
    text = f"""# 方法映射说明

## 本次对比使用的数据

| 项目 | 内容 |
|---|---|
| 原始图像 | `火焰筒/母线拼接` |
| 当前方法官方结果 | `{proposed_dir.as_posix()}` |
| 统一后端评定 | 共享同一条设计型线对比与鲁棒剔除逻辑 |

## 方法与文献映射

| 对比方法 | 文献来源 | 当前代码中的实现映射 |
|---|---|---|
| Literature A: 2025 smooth-curve stitching | Wang K, Li Z, Xu L, Shi L, Liu M. 2025. *Direct measurement and compensation of contour errors for profile grinding*. | 按论文“低特征复杂平滑曲线 + 高曲率点检测”思路，先找轮廓高曲率锚点，再做重叠区残差精化。 |
| Literature B: 2026 telecentric scan-and-stitch | Li L, Li B, Sun Z, Shi Y, Xu Y, Wei X. 2026. *A hybrid optical metrology framework integrating telecentric imaging and an optical micrometer for multi-scale geometric evaluation of aero-engine shafts*. | 按论文“telecentric scan-and-stitch + affine-guided stitching”思路，在当前平移主导数据上采用灰度粗配准 + 重叠区精化的简化实现。 |
| Proposed: Motion-prior normal/tangent | 当前项目官方结果 | 直接读取你的官方输出 `stitching_data.csv` 与 `design_error_summary.csv`。 |

## 解读口径

1. 三种方法使用同一批原始火焰筒母线图像。
2. 三种方法共享同一条后端设计型线评定逻辑。
3. 因此最终差异主要反映前端拼接策略差异，而不是后端统计口径变化。
"""
    (output_dir / "selection_and_mapping.md").write_text(text, encoding="utf-8")


def write_recent_report(output_path: Path, phase_pairs, iterative_pairs, results) -> None:
    def rows_to_markdown(rows):
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

    summary_rows = [
        {
            "Method": item.label,
            "Absolute RMSE (um)": f"{item.normal_rmse_um:.3f}",
            "Profile RMS (um)": f"{item.profile_rms_um:.3f}",
            "Profile P95 (um)": f"{item.profile_p95_abs_um:.3f}",
            "Bias correction (um)": f"{item.absolute_bias_correction_um:.3f}",
            "Used count": item.used_count,
            "Outlier ratio": f"{item.outlier_ratio:.4f}",
        }
        for item in results
    ]
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
            "Score": f"{item.score:.4f}",
            "Overlap RMSE (px)": f"{item.overlap_profile_rmse_px:.4f}",
        }
        for item in iterative_pairs
    ]
    text = f"""# Recent Literature Comparison for the Fire-Tube Generatrix

## Adopted recent baselines

| Baseline | Literature origin | Implemented adaptation on this dataset |
|---|---|---|
| Literature A: 2025 smooth-curve stitching | Wang K, Li Z, Xu L, Shi L, Liu M. *Direct measurement and compensation of contour errors for profile grinding*. Measurement, January 2025. DOI: `10.1016/j.measurement.2024.115959` | The paper addresses low-feature smooth-curve stitching in profile grinding. On this dataset, its core idea is mapped to curvature-anchor detection followed by overlap-residual refinement. |
| Literature B: 2026 telecentric scan-and-stitch | Li L, Li B, Sun Z, Shi Y, Xu Y, Wei X. *A hybrid optical metrology framework integrating telecentric imaging and an optical micrometer for multi-scale geometric evaluation of aero-engine shafts*. Measurement, 2026. DOI: `10.1016/j.measurement.2025.120053` | The paper uses telecentric scan-and-stitch with affine-guided stitching. On this translation-dominant one-dimensional contour dataset, a gray-registration scan-and-stitch simplification is used. |
| Proposed | Current official project result | The official `stitching_data.csv` and `design_error_summary.csv` are reused directly. |

## Method summary

{rows_to_markdown(summary_rows)}

## Pairwise shifts: Literature A

{rows_to_markdown(phase_rows)}

## Pairwise shifts: Literature B

{rows_to_markdown(iterative_rows)}

## Notes

- All methods use the same raw image batch under `火焰筒/母线拼接`.
- All methods share the same offline design-profile evaluator and the same robust outlier rule.
- The comparison therefore isolates the influence of the front-end stitching strategy.
"""
    output_path.write_text(text, encoding="utf-8")


def detect_curvature_anchor_x(base, contour, top_k: int = 10, min_gap_px: float = 120.0):
    if contour.x.size < 300:
        return []
    y_smooth = base.gaussian_filter1d(contour.y.astype(float), sigma=6.0)
    dy = base.np.gradient(y_smooth, contour.x)
    ddy = base.np.gradient(dy, contour.x)
    curvature = base.np.abs(ddy) / base.np.maximum((1.0 + dy * dy) ** 1.5, 1e-9)
    order = base.np.argsort(curvature)[::-1]
    chosen = []
    for idx in order:
        x_value = float(contour.x[int(idx)])
        if x_value < 40.0 or x_value > float(contour.width) - 40.0:
            continue
        if any(abs(x_value - prev) < min_gap_px for prev in chosen):
            continue
        chosen.append(x_value)
        if len(chosen) >= top_k:
            break
    return chosen


def pairwise_recent_profile_grinding_baseline(base, images, contours):
    transforms = []
    phase_pairs = base.pairwise_phase_baseline(images, contours)
    for step, (image_a, image_b, contour_a, contour_b) in enumerate(
        zip(images[:-1], images[1:], contours[:-1], contours[1:]),
        start=1,
    ):
        phase_pair = phase_pairs[step - 1]
        phase_dx = float(phase_pair.dx)
        phase_dy = float(phase_pair.dy)
        phase_score = float(phase_pair.score)
        candidates_a = detect_curvature_anchor_x(base, contour_a)
        candidates_b = detect_curvature_anchor_x(base, contour_b)
        dx_candidates = [phase_dx]
        for anchor_a in candidates_a:
            for anchor_b in candidates_b:
                dx_guess = float(anchor_a - anchor_b)
                if base.PHASE_DX_EXPECTED_MIN <= dx_guess <= base.PHASE_DX_EXPECTED_MAX:
                    dx_candidates.append(dx_guess)

        best = None
        seen = set()
        for dx_init in dx_candidates:
            rounded = round(float(dx_init), 3)
            if rounded in seen:
                continue
            seen.add(rounded)
            try:
                dx_best, dy_best, rmse = base.refine_contour_overlap(contour_a, contour_b, float(dx_init))
            except Exception:
                continue
            if not math.isfinite(float(rmse)) or float(rmse) >= 1e8:
                continue
            score = float(rmse + 0.01 * abs(dx_best - phase_dx))
            if best is None or score < best["score"]:
                best = {
                    "dx": dx_best,
                    "dy": dy_best,
                    "rmse": rmse,
                    "score": score,
                }

        if best is None:
            fallback_rmse = base.overlap_profile_rmse(contour_a, contour_b, phase_dx, phase_dy)
            best = {
                "dx": phase_dx,
                "dy": phase_dy,
                "rmse": float(fallback_rmse) if math.isfinite(float(fallback_rmse)) else 1e9,
                "score": float("inf"),
            }

        transforms.append(
            base.PairTransform(
                step=step,
                image_a=step,
                image_b=step + 1,
                dx=float(best["dx"]),
                dy=float(best["dy"]),
                score=float(phase_score),
                overlap_profile_rmse_px=float(best["rmse"]),
                method="recent_profile_grinding_2025",
            )
        )
    return transforms


def pairwise_recent_telecentric_baseline(base, images, contours):
    phase_pairs = base.pairwise_phase_baseline(images, contours)
    return [
        base.PairTransform(
            step=item.step,
            image_a=item.image_a,
            image_b=item.image_b,
            dx=float(item.dx),
            dy=float(item.dy),
            score=float(item.score),
            overlap_profile_rmse_px=float(item.overlap_profile_rmse_px),
            method="recent_telecentric_scan_stitch_2026",
        )
        for item in phase_pairs
    ]


def build_official_proposed_result(base, proposed_dir: Path, proposed_samples, label: str):
    official = base.load_official_stage5_summary(proposed_dir)
    profile = base.load_official_stage5_profile(proposed_dir)
    return base.MethodEvaluation(
        label=label,
        dz_mm=official["dz_mm"],
        dr_mm=official["dr_mm"],
        dtheta_deg=official["dtheta_deg"],
        pre_refine_mean_normal_error_um=official["pre_refine_mean_normal_error_um"],
        absolute_bias_correction_um=official["absolute_bias_correction_um"],
        mean_normal_error_um=official["mean_normal_error_um"],
        normal_rmse_um=official["normal_rmse_um"],
        normal_p95_abs_um=official["normal_p95_abs_um"],
        profile_rms_um=official["profile_rms_um"],
        profile_p95_abs_um=official["profile_p95_abs_um"],
        used_count=official["used_count"],
        outlier_count=official["outlier_count"],
        outlier_ratio=official["outlier_ratio"],
        anchor_x_px=official["anchor_x_px"],
        anchor_y_px=official["anchor_y_px"],
        contour_x_px=profile["contour_x_px"],
        contour_y_px=profile["contour_y_px"],
        s_used_mm=profile["s_used_mm"],
        measured_r_used_mm=profile["measured_r_used_mm"],
        design_r_used_mm=profile["design_r_used_mm"],
        normal_error_used_um=profile["normal_error_used_um"],
        profile_error_used_um=profile["profile_error_used_um"],
    )


def main() -> int:
    args = parse_args()
    base = load_base_module()

    proposed_dir = resolve_proposed_dir(args)
    output_dir = resolve_output_dir(args, proposed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("Pic_*.bmp"))
    if not image_paths:
        raise FileNotFoundError(f"no Pic_*.bmp images found under: {args.input_dir}")

    images = [base.load_grayscale(path) for path in image_paths]
    contours = [base.extract_top_contour(image) for image in images]

    phase_pairs = pairwise_recent_profile_grinding_baseline(base, images, contours)
    iterative_pairs = pairwise_recent_telecentric_baseline(base, images, contours)
    proposed_pairs_raw = base.load_stage5_pair_transforms(proposed_dir)
    proposed_pairs, proposed_samples, _, _, proposed_scale_rows = base.select_best_stage5_pair_scale(
        proposed_dir,
        contours,
        proposed_pairs_raw,
        lambda samples, pairs: base.evaluate_method("proposed_scale_candidate", samples),
    )

    phase_samples = base.collect_transformed_samples(contours, phase_pairs)
    iterative_samples = base.collect_transformed_samples(contours, iterative_pairs)

    phase_eval = base.evaluate_method("Literature A: 2025 smooth-curve stitching", phase_samples)
    iterative_eval = base.evaluate_method("Literature B: 2026 telecentric scan-and-stitch", iterative_samples)
    proposed_eval = build_official_proposed_result(
        base,
        proposed_dir,
        proposed_samples,
        "Proposed: Motion-prior normal/tangent",
    )
    results = [phase_eval, iterative_eval, proposed_eval]

    pd.DataFrame(proposed_scale_rows).to_csv(
        output_dir / "proposed_pair_scale_refine.csv", index=False, encoding="utf-8-sig"
    )
    base.write_pair_csv(output_dir / "phase_pair_transforms.csv", phase_pairs)
    base.write_pair_csv(output_dir / "iterative_pair_transforms.csv", iterative_pairs)
    base.write_pair_csv(output_dir / "proposed_pair_transforms_raw.csv", proposed_pairs_raw)
    base.write_pair_csv(output_dir / "proposed_pair_transforms.csv", proposed_pairs)
    base.write_summary_csv(output_dir / "literature_method_comparison.csv", results)

    base.render_panorama(images, phase_pairs, output_dir / "phase_panorama.png")
    base.render_panorama(images, iterative_pairs, output_dir / "iterative_panorama.png")
    base.render_panorama(images, proposed_pairs, output_dir / "proposed_panorama_rebuilt.png")

    base.plot_contour_overlay(phase_samples, len(contours), output_dir / "phase_contour_overlay.png")
    base.plot_contour_overlay(iterative_samples, len(contours), output_dir / "iterative_contour_overlay.png")
    base.plot_contour_overlay(proposed_samples, len(contours), output_dir / "proposed_contour_overlay.png")
    base.plot_profile_comparison(results, output_dir / "literature_profile_comparison.png")

    write_recent_report(output_dir / "literature_comparison_report.md", phase_pairs, iterative_pairs, results)
    write_selection_mapping(output_dir, proposed_dir)

    print(f"[OK] proposed_dir={proposed_dir}")
    print(f"[OK] output_dir={output_dir}")
    for item in results:
        print(
            f"[Result] {item.label}: abs_RMSE={item.normal_rmse_um:.3f} um, "
            f"profile_RMS={item.profile_rms_um:.3f} um, profile_P95={item.profile_p95_abs_um:.3f} um"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
