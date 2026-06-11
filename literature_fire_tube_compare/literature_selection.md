# 文献检索与近年基线选择说明

检索日期：`2026-05-31`

## 1. 你的项目场景拆解

| 维度 | 你的项目特征 |
|---|---|
| 检测对象 | 回转体火焰筒类样件母线轮廓 |
| 检测方式 | 远心视觉 + 平行背光 |
| 数据形态 | 顺序采集的局部轮廓图像 |
| 核心困难 | 低纹理、平滑母线、相邻视场缺少稳定角点、顺序拼接易产生切向滑移与累计漂移 |
| 结果目标 | 设计型线偏差评定与轮廓重建稳定性比较 |

## 2. 检索到的最相关近年背景文献

| 类别 | 文献 | 相关性 | 用途 |
|---|---|---|---|
| 同工艺背景 | Xu L M, Fan F, Hu Y X, et al. *A vision-based processing methodology for profile grinding of contour surfaces*. Proc IMechE Part B, 2020. DOI: `10.1177/0954405419857401` | 高 | 说明“轮廓磨削 + 机器视觉在线检测”这条工艺背景是直接相关的。 |
| 同工艺背景 | Wang K, Li Z, Xu L, et al. *Direct measurement and compensation of contour errors for profile grinding*. Measurement, 2025. DOI: `10.1016/j.measurement.2024.115959` | 很高 | 说明“低特征平滑曲线拼接 + 在线轮廓评定”已成为该方向核心问题。 |
| 可复现基线 A | Wang K, Li Z, Xu L, et al. *Direct measurement and compensation of contour errors for profile grinding*. Measurement, January 2025. DOI: `10.1016/j.measurement.2024.115959` | 很高 | 文中明确提出“low-feature-based complex smooth curve image stitching algorithm”，并在结论中指出其基于高曲率点检测。 |
| 可复现基线 B | Li L, Li B, Sun Z, Shi Y, Xu Y, Wei X. *A hybrid optical metrology framework integrating telecentric imaging and an optical micrometer for multi-scale geometric evaluation of aero-engine shafts*. Measurement, 2026. DOI: `10.1016/j.measurement.2025.120053` | 高 | 文中明确采用 telecentric imaging + affine transformation-based stitching algorithm，适合作为远心扫描拼接方向的近年基线。 |
| 候选但未直接复现 | Schlagenhauf T, Brander T, Fleischer J. *A stitching algorithm for automated surface inspection of rotationally symmetric components*. CIRP Journal of Manufacturing Science and Technology, 2021. DOI: `10.1016/j.cirpj.2021.05.013` | 中高 | 与回转体背景接近，但年份不够新，且更依赖表面纹理/视频特征，不如前两种方法适配当前背光母线图像。 |

## 3. 最终为什么选这两种近年方法

| 选中方法 | 选择理由 | 对你当前方法的对比意义 |
|---|---|---|
| 文献方法 A：2025 smooth-curve stitching | 这是与你当前“轮廓磨削 + 低特征平滑母线”最同场景的近年方法。 | 可以回答“同工艺背景下，2025 年最新平滑曲线拼接思路做到什么程度”。 |
| 文献方法 B：2026 telecentric scan-and-stitch | 这是与你当前“远心视觉 + 扫描拼接”最接近的近年测量系统方法。 | 可以回答“在近年的远心扫描拼接框架下，常规 telecentric scan-and-stitch 能做到什么程度”。 |

## 4. 为什么不是直接照搬论文原系统

| 原因 | 说明 |
|---|---|
| 论文系统耦合较强 | 近年论文通常把相机、工装、扫描机构、补偿流程和拼接算法绑在一起，无法直接逐行复现。 |
| 当前任务目标是公平对比 | 你需要的是“在同一批火焰筒原始图像、同一条后端评定链路下”比较前端拼接思路，而不是重建整台别人的设备。 |
| 因此采用思想映射复现 | 代码只复现两篇近年论文中最核心、最能迁移到你数据上的拼接思想，并明确保持后端评定一致。 |

## 5. 代码中的方法映射

| 代码方法名 | 文献映射 | 当前实现方式 |
|---|---|---|
| `recent_profile_grinding_2025` | Wang et al., 2025 | 先在轮廓上检测高曲率锚点，再在候选位移附近做重叠区残差精化。 |
| `recent_telecentric_scan_stitch_2026` | Li et al., 2026 | 先做灰度粗配准，再在重叠轮廓上做平移主导的 scan-and-stitch 精化。 |
| `proposed` | 你当前系统结果 | 直接读取 `result/workpiece/<当前结果>/` 的官方输出，与前两者共用同一后端设计型线评定逻辑。 |

## 6. 结果解读建议

| 结果现象 | 建议写法 |
|---|---|
| 2025 基线误差高于当前方法 | 强调单纯依赖高曲率锚点和局部平滑曲线拼接，仍难以完全抑制顺序拼接中的漂移传播。 |
| 2026 基线仍弱于当前方法 | 强调常规 telecentric scan-and-stitch 即使能拼对，也仍缺少你当前方法中的运动先验、法切向联合约束与坏步处理。 |
| 当前方法在同一后端评定逻辑下最优 | 强调改进来自前端拼接约束，而不是后端评定口径变化。 |
