# Design Profile Figure Export

独立的论文图导出工具，用于把 `design_error_profile.csv` 与
`design_error_summary.csv` 自动组图为可投稿的 `SVG` 矢量图。

## 环境准备

```bash
python -m venv .venv-figure
.venv-figure\Scripts\activate
pip install -r report/figure_export/requirements.txt
```

## 用法

直接针对某次结果目录导出：

```bash
python report/figure_export/design_profile_figure.py ^
  --result-dir result/workpiece/cli_20260511_171726
```

默认会在结果目录下生成：

- `journal_design_profile_figure.svg`
- `journal_design_profile_figure.png`

如果只导出 SVG：

```bash
python report/figure_export/design_profile_figure.py ^
  --result-dir result/workpiece/cli_20260511_171726 ^
  --svg-only
```

如果希望自动寻找最新一次结果目录：

```bash
python report/figure_export/design_profile_figure.py --result-root result/workpiece
```

## 输出内容

自动组图包含四个子图：

1. 设计曲线与实测轮廓对比；
2. 绝对法向误差与去均值后的轮廓波动曲线；
3. 轮廓波动分布直方图；
4. 汇总统计信息面板。

## 字体与投稿兼容性

脚本默认优先使用：

- `Times New Roman`
- `Noto Serif CJK SC`
- `SimSun`
- `DejaVu Serif`

并默认将 SVG 字体转为 path，以减少期刊端缺字风险。

