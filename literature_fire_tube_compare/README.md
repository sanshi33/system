# 火焰筒母线文献基线对比

这个目录用于把你当前 `火焰筒/母线拼接` 数据，按同一批原始图像、同一条设计型线评定逻辑，与两种近年文献方法做可复现实验对比。

## 目录说明

| 文件 | 作用 |
|---|---|
| `compare_fire_tube_literature_baselines.py` | 主运行脚本。自动读取原始图像、两种文献基线、当前方法结果，并导出对比结果。 |
| `literature_selection.md` | 文献检索与方法筛选说明，交代为什么最终选择这两种方法作为可复现基线。 |
| `requirements.txt` | 运行该目录脚本所需的 Python 依赖。 |
| `output/` | 运行后生成的 CSV、对比图和 Markdown 报告目录。 |

## 当前对比对象

| 项目 | 内容 |
|---|---|
| 检测对象 | 回转体火焰筒类样件的母线轮廓 |
| 原始图像目录 | `火焰筒/母线拼接` |
| 当前方法结果目录 | 默认自动寻找 `result/workpiece` 下最新且包含 `design_error_summary.csv` 的结果目录 |
| 文献方法 A | 2025 年 profile grinding 平滑曲线拼接 |
| 文献方法 B | 2026 年 telecentric scan-and-stitch |

## 使用方法

```bash
python literature_fire_tube_compare/compare_fire_tube_literature_baselines.py
```

如果你想手动指定当前方法结果目录：

```bash
python literature_fire_tube_compare/compare_fire_tube_literature_baselines.py ^
  --proposed-dir result/workpiece/cli_20260531_100426
```

如果你想改输出目录：

```bash
python literature_fire_tube_compare/compare_fire_tube_literature_baselines.py ^
  --output-dir literature_fire_tube_compare/output/custom_run
```

## 输出内容

脚本默认会在 `literature_fire_tube_compare/output/<当前结果目录名>/` 下生成：

| 输出文件 | 说明 |
|---|---|
| `literature_method_comparison.csv` | 三种方法的最终误差汇总 |
| `phase_pair_transforms.csv` | 文献方法 A 的步间位移结果 |
| `iterative_pair_transforms.csv` | 文献方法 B 的步间位移结果 |
| `proposed_pair_transforms.csv` | 当前方法的步间位移结果 |
| `literature_comparison_report.md` | 自动生成的英文结果报告 |
| `selection_and_mapping.md` | 中文版方法映射说明 |
| `phase_panorama.png` / `iterative_panorama.png` / `proposed_panorama_rebuilt.png` | 三种方法的全景图 |
| `*_contour_overlay.png` | 三种方法的轮廓叠加图 |
| `literature_profile_comparison.png` | 三种方法的设计型线对比图 |

## 方法选择原则

| 原则 | 说明 |
|---|---|
| 同工艺背景优先 | 优先参考轮廓磨削、在线视觉测量、平滑轮廓拼接、回转体/圆柱类表面拼接文献。 |
| 年份尽量新 | 默认优先 2025-2026 年文献；年份较早的方法只作为候选背景，不作为主基线。 |
| 可迁移实现优先 | 只选能够在你当前这批背光母线图像上复现，且能得到正确拼接结果的文献方法。 |
| 对比解释性优先 | 选一条同工艺背景平滑曲线拼接路线，再选一条近年的远心扫描拼接路线，这样能更清楚说明你当前方法的增益来源。 |
