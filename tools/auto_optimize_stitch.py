#!/usr/bin/env python3
"""Run stitch parameter sweeps and rank results by reconstruction quality."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path


def read_single_row_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def to_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def worst_step_rmse(stitching_csv: Path) -> tuple[float, int]:
    if not stitching_csv.exists():
        return math.nan, 0
    worst = 0.0
    worst_step = 0
    with stitching_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row.get("Step", "0"))
                rmse = float(row.get("NormalRMSEInlier(px)", "") or row.get("NormalRMSEAll(px)", "nan"))
            except ValueError:
                continue
            if math.isfinite(rmse) and rmse > worst:
                worst = rmse
                worst_step = step
    return worst, worst_step


def score_result(summary: dict[str, str], worst_rmse_px: float) -> float:
    absolute_rmse = to_float(summary, "absolute_filtered_rmse_um", to_float(summary, "normal_rmse_um", 1e9))
    absolute_p95 = to_float(summary, "absolute_filtered_p95_abs_um", to_float(summary, "normal_p95_abs_um", 1e9))
    form_rmse = to_float(summary, "profile_rms_um", 1e9)
    outlier_ratio = to_float(summary, "outlier_ratio", 1.0)
    dz = abs(to_float(summary, "dz_mm", 0.0))
    dtheta = abs(to_float(summary, "dtheta_deg", 0.0))
    worst_step = worst_rmse_px if math.isfinite(worst_rmse_px) else 10.0
    return (
        0.40 * absolute_rmse
        + 0.20 * absolute_p95
        + 0.20 * form_rmse
        + 200.0 * worst_step
        + 500.0 * max(0.0, outlier_ratio - 0.03)
        + 15.0 * dz
        + 100.0 * dtheta
    )


def write_rank_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    values = {
        "rotation_step": args.rotation_steps,
        "rotation_range": args.rotation_ranges,
        "tangent_residual_weight": args.tangent_residual_weights,
        "tangent_correlation_weight": args.tangent_correlation_weights,
        "filter_confidence_q": args.filter_confidence_q,
        "filter_gradient_q": args.filter_gradient_q,
        "filter_hampel_sigma": args.filter_hampel_sigma,
    }
    keys = list(values.keys())
    runs = []
    for combo in itertools.product(*(values[k] for k in keys)):
        runs.append(dict(zip(keys, combo)))
    return runs[: args.max_runs] if args.max_runs > 0 else runs


def run_one(args: argparse.Namespace, params: dict[str, object], index: int) -> dict[str, object]:
    run_dir = args.output_root / f"sweep_{index:03d}"
    out_png = run_dir / "final_panorama.png"
    out_csv = run_dir / "stitching_data.csv"
    cmd = [
        str(args.cli),
        str(args.input_dir),
        str(args.image_count),
        "--start-index",
        str(args.start_index),
        "--out",
        str(out_png),
        "--csv",
        str(out_csv),
        "--no-process-vis",
        "--rotation-range",
        str(params["rotation_range"]),
        "--rotation-step",
        str(params["rotation_step"]),
        "--tangent-residual-weight",
        str(params["tangent_residual_weight"]),
        "--tangent-correlation-weight",
        str(params["tangent_correlation_weight"]),
        "--filter-confidence-q",
        str(params["filter_confidence_q"]),
        "--filter-gradient-q",
        str(params["filter_gradient_q"]),
        "--filter-hampel-sigma",
        str(params["filter_hampel_sigma"]),
    ]

    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"run_dir": str(run_dir), **params}
    try:
        completed = subprocess.run(
            cmd,
            cwd=args.project_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
            check=False,
        )
        result["exit_code"] = completed.returncode
        result["timed_out"] = 0
        (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        result["exit_code"] = -1
        result["timed_out"] = 1
        (run_dir / "run.log").write_text(exc.stdout or "", encoding="utf-8")

    summary = read_single_row_csv(run_dir / "design_error_summary.csv")
    worst_rmse, worst_step = worst_step_rmse(run_dir / "stitching_data.csv")
    result.update(
        {
            "score": score_result(summary, worst_rmse) if summary else math.inf,
            "absolute_filtered_rmse_um": to_float(summary, "absolute_filtered_rmse_um"),
            "absolute_filtered_p95_abs_um": to_float(summary, "absolute_filtered_p95_abs_um"),
            "profile_rms_um": to_float(summary, "profile_rms_um"),
            "outlier_ratio": to_float(summary, "outlier_ratio"),
            "dz_mm": to_float(summary, "dz_mm"),
            "dtheta_deg": to_float(summary, "dtheta_deg"),
            "worst_step_normal_rmse_px": worst_rmse,
            "worst_step": worst_step,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cli", type=Path, default=Path("build/bin/Release/pinjie_cli.exe"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--image-count", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("result/workpiece/auto_optimize"))
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--stop-absolute-rmse-um", type=float, default=60.0)
    parser.add_argument("--stop-form-rmse-um", type=float, default=25.0)
    parser.add_argument("--rotation-steps", type=float, nargs="+", default=[0.05, 0.02])
    parser.add_argument("--rotation-ranges", type=float, nargs="+", default=[0.5, 0.3])
    parser.add_argument("--tangent-residual-weights", type=float, nargs="+", default=[0.05, 0.08])
    parser.add_argument("--tangent-correlation-weights", type=float, nargs="+", default=[0.25, 0.4])
    parser.add_argument("--filter-confidence-q", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--filter-gradient-q", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--filter-hampel-sigma", type=float, nargs="+", default=[3.0, 2.5])
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.cli = (args.project_root / args.cli).resolve() if not args.cli.is_absolute() else args.cli
    args.input_dir = (args.project_root / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir
    args.output_root = (args.project_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, params in enumerate(build_grid(args), start=1):
        print(f"[{index}] {params}", flush=True)
        row = run_one(args, params, index)
        rows.append(row)
        rows.sort(key=lambda item: float(item["score"]))
        write_rank_csv(args.output_root / "optimization_rank.csv", rows)
        best = rows[0]
        if (
            math.isfinite(float(best["score"]))
            and float(best["absolute_filtered_rmse_um"]) <= args.stop_absolute_rmse_um
            and float(best["profile_rms_um"]) <= args.stop_form_rmse_um
        ):
            break

    rows.sort(key=lambda item: float(item["score"]))
    write_rank_csv(args.output_root / "optimization_rank.csv", rows)
    if rows:
        def json_value(value: object) -> object:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        (args.output_root / "best_run_manifest.json").write_text(
            json.dumps({k: json_value(v) for k, v in rows[0].items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
