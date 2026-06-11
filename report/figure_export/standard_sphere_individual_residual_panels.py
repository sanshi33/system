#!/usr/bin/env python3
"""
Export individual all-subpixel residual panels for representative standard-sphere detections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from standard_sphere_global_pose_selection_figure import (
    build_field_dataframe,
    configure_matplotlib,
    image_name_from_index,
    parse_indices,
    resolve_input_dir,
)


OUTPUT_DIR_NAME = "individual_residual_panels"
OUTPUT_MANIFEST_NAME = "individual_residual_panels_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export separate all-subpixel residual panels.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--image-indices", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--svg-fonttype", choices=("path", "none"), default="path")
    return parser.parse_args()


def draw_single_panel(ax: plt.Axes, row: pd.Series) -> None:
    from standard_sphere_global_pose_selection_figure import draw_edge_distribution_panel

    draw_edge_distribution_panel(ax, row, is_left_edge=True, is_bottom_edge=True)
    ax.set_title(f"{row['label']} global all-subpixel residuals", fontsize=8.6, pad=5.0)


def export_single_panel(output_dir: Path, row: pd.Series) -> dict[str, str | float]:
    label = str(row["label"])
    base_name = f"{label}_all_subpixel_residual"
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"
    pdf_path = output_dir / f"{base_name}.pdf"

    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
    draw_single_panel(ax, row)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "label": label,
        "image_index": int(row["image_index"]),
        "image_name": str(row["image_name"]),
        "selected_angle_deg": float(row["selected_angle_deg"]),
        "diameter_error_um": float(row["diameter_error_um"]),
        "edge_rmse_um": float(row["edge_rmse_um"]),
        "png_path": str(png_path),
        "svg_path": str(svg_path),
        "pdf_path": str(pdf_path),
    }


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.svg_fonttype)

    result_dir = args.result_dir.resolve()
    input_dir = resolve_input_dir(args)
    output_dir = (args.output_dir or (result_dir / OUTPUT_DIR_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = parse_indices(args.image_indices, fallback=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    df = build_field_dataframe(result_dir, input_dir, indices)
    if df.empty:
        raise RuntimeError("no representative fields available for exporting")

    manifest_rows: list[dict[str, str | float]] = []
    for _, row in df.sort_values("image_index").iterrows():
        manifest_rows.append(export_single_panel(output_dir, row))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / OUTPUT_MANIFEST_NAME, index=False, encoding="utf-8-sig")
    print(f"[OK] exported {len(manifest_rows)} individual residual panels under: {output_dir}")


if __name__ == "__main__":
    main()
