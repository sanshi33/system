# 标准球闭环累计误差实验

这个目录用于把现有边缘提取、亚像素点过滤、单步配准算法放到标准球闭环实验里单独评估，不改动 GUI 主流程。

当前命令行默认复用工件完整拼接 pipeline：边缘预处理、顺序配准、运动先验更新、最后一步局部重扫、位姿图优化以及裁剪后的 `imageTransforms` 都来自 `stitch::runStitchingPipeline`。标准球模块只在这些拼接结果上额外计算圆拟合、累计误差和可视化。

## 构建目标

CMake 目标名：

```bash
standard_sphere_loop_eval
```

## 基本用法

```bash
standard_sphere_loop_eval <标准球图像目录> <图像数量> --start-index 1 --sphere-diameter 59.9986
```

在 `GB/T 5.7` 的 25 点单点评定模式下，如果提供了 `--sphere-diameter`，
程序会优先使用“实际球径 / 拼接全局圆直径”反算像素当量，再换算 `e_p2d`、`RMSE`
等微米指标；只有在没有可用球径时，才回退到 `--pixel-size` 或视场配置推导的比例尺。

如果需要在不破坏稳定拼接结果的前提下，尽量把被圆边清理删掉的边缘点也纳入 25 点候选，
可以额外打开 `--gbt57-supplement-precleanup-candidates`。这个模式会保留“清理后边缘”用于
全局拼接与全局圆拟合，但在单点候选池里补回清理前的边缘点，用于评估现有数据的极限覆盖能力。

默认按 `Pic_1.bmp, Pic_2.bmp ...` 读取。若文件命名不同，可以用：

```bash
standard_sphere_loop_eval <目录> <数量> --prefix Ball_ --ext .png
```

## 输出

默认输出到 `result/standard_sphere_loop/run_YYYYMMDD_HHMMSS/`：

| 文件 | 内容 |
|---|---|
| `standard_sphere_loop_summary.csv` | 闭环平移、闭环旋转、角点/中心漂移、拼接后圆拟合误差 |
| `standard_sphere_loop_pairs.csv` | 每一步 `i -> i+1` 以及最后 `N -> 1` 的配准结果和质量指标 |
| `standard_sphere_circle_fit.csv` | 每张标准球图像的圆拟合半径、直径和残差 |
| `standard_sphere_global_transforms.csv` | 每张图累乘到第 1 张图坐标系下的全局变换矩阵 |

核心评价量建议优先看：

| 指标 | 含义 |
|---|---|
| `closure_translation` | 顺序配准绕一圈后的平移闭合误差 |
| `closure_rotation` | 顺序配准绕一圈后的旋转闭合误差 |
| `closure_center_drift` | 第一张图中心点绕环变换后的漂移 |
| `stitched_circle_rmse` | 所有图像边缘累乘拼接后，对同一个标准球圆拟合的残差 |
| `stitched_circle_diameter_error` | 给定 `--pixel-size` 和 `--sphere-diameter` 后的标准球直径误差 |

## 2026-05-17 note

The standard-sphere refinement is a circle-consistency refinement, not a hard loop-closure correction.
It keeps the raw workpiece pipeline closure in `pre_optimization_closure_translation` and only applies
small regularized 2D pose corrections to reduce the stitched circle residual. The first/last closure is
reported after refinement but is never forced to zero. Random edge noise is reported with
`random_noise_per_image_rmse_mean`, `random_noise_pooled_rmse`, and robust stitched-circle MAD metrics.
