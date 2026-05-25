# 保留结果清单

生成日期：2026-05-25

本次清理仅处理：

- `result/workpiece`
- `result/standard_sphere_loop`

不会处理：

- `result/camera`
- `result/calibration`
- 其他非本次评估结论相关的数据目录

## 保留目录

| 类别 | 目录 | 保留原因 |
|---|---|---|
| 工件主结果 | `result/workpiece/step9_overlap078` | 当前工件绝对精度最好，`absolute_filtered_rmse_um = 14.011477` |
| 工件稳定备选 | `result/workpiece/step9_overlap078_local100` | 单步最差 RMSE 更稳，可作为稳定性对照 |
| 工件稳定备选 | `result/workpiece/step9_overlap078_local100_filterbest` | 过滤优化版本，保留作横向比较 |
| 标准球闭环主结果 | `result/standard_sphere_loop/codex_run_20260519_joint_pose_circle_v5` | 标准球闭环综合表现最好 |
| 标准球闭环备选 | `result/standard_sphere_loop/codex_run_20260519_joint_pose_circle_v6` | 闭环误差更小，可作对照 |
| GBT57 可信主结果 | `result/standard_sphere_loop/codex_run_20260521_fixed_radius_uniform_angle_confbest_v1` | 角度分布最规整，可靠性最好 |
| GBT57 极限结果 | `result/standard_sphere_loop/codex_run_20260521_range256_restarts64_selectedrefine_v1-最好结果` | 数值最优，保留作极限参考 |

## 删除策略

- 删除 `result/workpiece` 下除上述 3 个目录外的其他运行目录
- 删除 `result/standard_sphere_loop` 下除上述 4 个目录外的其他运行目录
- 不删除源码、输入图像、相机采集结果和标定结果
