# 拼接数据结构说明

## 1. 文档目的

当前工程已经把拼接流程中的共享类型统一收口到：

- `stitch/StitchTypes.h`

把完整运行入口统一收口到：

- `stitch/StitchService.h`

因此现在工程对外的正式接口只有一套：

- 类型定义看 `stitch/StitchTypes.h`
- 调用入口看 `stitch/StitchService.h`

旧的兼容头 `stitch/Types.h` 已经删除，后续不再维护转发层。

## 2. 当前整体数据流

当前工件拼接主流程可以概括为：

```text
输入图像路径
    -> StitchRunRequest
    -> EdgeDetectConfig
    -> EdgeVariants
    -> TransformResult + AlignmentMetrics
    -> StitchStepRecord
    -> StitchingResult
    -> StitchRunResult
```

如果按职责划分，可以分成四层：

- 输入层：`StitchRunRequest`
- 算法层：`EdgeVariants`、`TransformResult`、`AlignmentMetrics`
- 步骤层：`StitchStepRecord`
- 输出层：`StitchingResult`、`StitchRunResult`

## 3. 模块与结构体对应关系

- `core/SubpixelEdgeDetector.*`
  负责亚像素边缘提取。

- `stitch/IO.*`
  负责图像加载与输入准备。

- `stitch/EdgeProcessing.*`
  负责把原始边缘点整理成 `EdgeVariants`。

- `stitch/Alignment.*`
  负责配准搜索，输出 `TransformResult`，并填充 `AlignmentMetrics`。

- `stitch/Pipeline.*`
  负责多步拼接流程，输出 `StitchStepRecord` 与 `StitchingResult`。

- `stitch/DebugVis.*`
  负责调试图、日志摘要与 CSV 文本导出。

- `stitch/StitchService.*`
  负责将“图像加载、边缘提取、拼接执行、结果保存、回调通知”组织成统一服务入口。

## 4. 核心类型

### 4.1 `AlignmentAxis`

作用：

- 描述当前配准采用 `X` 主轴还是 `Y` 主轴。

意义：

- `X` 表示将轮廓看作 `y = f(x)`
- `Y` 表示将轮廓看作 `x = f(y)`

它决定了后续插值、残差投影、覆盖率统计沿哪个方向进行。

### 4.2 `EdgeDetectConfig`

作用：

- 保存亚像素边缘提取所需参数。

主要字段：

- `cannyLow`、`cannyHigh`
  Canny 边缘检测阈值。

- `subpixWindow`
  ERF 亚像素拟合窗口大小。

- `subpixSigma`
  亚像素拟合前的预平滑强度。

- `minWarnPoints`
  边缘点过少时的告警阈值。

说明：

- 这个结构只描述“如何提取边缘”
- 不描述“如何拼接”与“如何保存输出”

### 4.3 `EdgeVariants`

作用：

- 保存同一条边缘在多种排序视图下的统一表示。

主要字段：

- `raw`
  原始亚像素边缘点。

- `x_sorted`
  按 `x` 升序排序后的点列。

- `y_sorted`
  按 `y` 升序排序后的点列。

- `negX_sorted`
  对 `x` 取反后再排序，用于反向位移假设。

- `negY_sorted`
  对 `y` 取反后再排序，用于反向位移假设。

辅助接口：

- `ordered(axis)`
  根据主轴返回对应排序视图。

- `mirroredOrdered(axis)`
  根据主轴返回对应镜像排序视图。

说明：

- 这是配准阶段的直接输入
- 它避免后续重复做排序和镜像准备

### 4.4 `ResidualStatistics`

作用：

- 统一封装一组残差样本的统计量。

这个结构同时用于：

- 法向残差统计
- 切向残差统计

主要字段：

- `sampleCount`
  参与统计的样本数量。

- `bias`
  带符号平均残差，用于观察系统偏移。

- `rmse`
  均方根误差，反映总体误差能量。

- `meanAbs`
  绝对误差均值。

- `medianAbs`
  绝对误差中位数。

- `p95Abs`
  绝对误差 95 分位。

- `maxAbs`
  绝对误差最大值。

说明：

- `bias` 强调系统性偏差
- `rmse` 强调整体误差水平
- `medianAbs` 与 `p95Abs` 更适合论文里描述稳态误差与长尾误差

### 4.5 `AlignmentMetrics`

作用：

- 保存单次配准结果的结构化质量指标。

当前实现把指标拆成三部分：

- 覆盖度指标
- 法向残差指标
- 切向残差指标

主要字段：

- `overlapCount`
  全部重叠样本数。

- `inlierCount`
  鲁棒筛选后的内点样本数。

- `inlierRatio`
  内点比例。

- `overlapSpan`
  全部重叠样本在主轴上的跨度。

- `overlapCoverageRatio`
  全部重叠跨度占参考边缘主轴跨度的比例。

- `inlierSpan`
  内点样本在主轴上的跨度。

- `inlierCoverageRatio`
  内点跨度占参考边缘主轴跨度的比例。

- `normalAll`
  全部重叠样本的法向残差统计，类型为 `ResidualStatistics`。

- `normalInlier`
  内点样本的法向残差统计，类型为 `ResidualStatistics`。

- `tangentAll`
  全部重叠样本的切向残差统计，类型为 `ResidualStatistics`。

- `tangentInlier`
  内点样本的切向残差统计，类型为 `ResidualStatistics`。

- `tangentCorrAll`
  全部重叠样本的切向相关性。

- `tangentCorrInlier`
  内点样本的切向相关性。

说明：

- `All` 指标反映原始重叠区域的整体质量
- `Inlier` 指标反映鲁棒筛选后的稳定质量
- 法向残差用于回答“贴合得准不准”
- 切向残差用于回答“沿轮廓方向是否一致”
- 切向相关性只能作为辅助指标，不能单独替代切向残差

### 4.6 `TransformResult`

作用：

- 保存单对图像配准后的几何结果与质量报告。

主要字段：

- `dx`、`dy`
  平移量。

- `da`
  旋转量。

- `score`
  内部搜索代价，用于候选优选与流程启发式判断。

- `axis`
  当前最优解对应主轴。

- `direction`
  当前最优方向标签，例如 `X+`、`X-`、`Y+`、`Y-`。

- `inlierErrors`
  内点法向残差，主要用于调试图可视化。

- `inlierCoordinates`
  与 `inlierErrors` 对应的主轴坐标。

- `metrics`
  结构化质量报告，应作为正式导出与论文汇报优先使用的数据。

说明：

- `score` 更偏算法内部启发式
- `metrics` 更偏结果解释、论文报告与后处理分析

### 4.7 `StitchStepRecord`

作用：

- 保存一次相邻图像拼接步骤的完整记录。

主要字段：

- `stepIndex`
  第几步拼接，从 1 开始。

- `referenceImageIndex`
  参考图索引，从 0 开始。

- `targetImageIndex`
  目标图索引，从 0 开始。

- `searchRangeX`、`searchRangeY`
  本步实际使用的搜索范围。

- `transform`
  本步最终采用的配准结果。

说明：

- 调试图、日志、CSV、GUI 步骤表格都可以直接复用该结构

## 5. 运行时交互类型

### 5.1 `LogCallback`

作用：

- 输出文本日志。

适合用于：

- CLI 终端输出
- GUI 日志面板

### 5.2 `ProgressCallback`

作用：

- 输出阶段进度。

参数：

- `stage`
  阶段名称，如 `load`、`preprocess`、`stitch`

- `current`
  当前完成数量

- `total`
  总数量

### 5.3 `StepCallback`

作用：

- 每完成一步拼接后把 `StitchStepRecord` 推送给上层。

### 5.4 `ImageCallback`

作用：

- 将过程图像回调给上层。

说明：

- 当前传递的是 `cv::Mat`
- Qt 层可自行转换成 `QImage` / `QPixmap`

### 5.5 `CancelCallback`

作用：

- 提供外部取消能力。

### 5.6 `StitchCallbacks`

作用：

- 把日志、进度、步骤、图像、取消等回调统一打包。

## 6. 运行配置与输入输出类型

### 6.1 `StitchPipelineConfig`

作用：

- 保存拼接流程本身的运行参数。

主要字段：

- `baseSearchRange`
  基础搜索范围。

- `approxShiftRatio`
  初始位移估计比例。

- `tangentResidualCostWeight`
  切向 RMSE^2 加入匹配代价的权重；设为 0 时关闭这一项。

- `tangentCorrelationCostWeight`
  `1 - 切向相关性` 加入匹配代价的权重；设为 0 时关闭这一项。

- `generateDebugVisualization`
  是否生成过程调试图。

说明：

- 它管理的是流程搜索行为
- 不负责文件路径与保存策略

### 6.1.1 匹配代价分解

当前候选的总匹配分数 `TransformResult::score` 由以下项组成：

- `normalMatchCost`
  原有法向/次轴 robust cost，是基础几何贴合项。

- `tangentResidualMatchCost`
  `tangentResidualCostWeight * tangentRMSE^2`，用于抑制切向滑移。

- `tangentCorrelationMatchCost`
  `tangentCorrelationCostWeight * max(0, 1 - tangentCorr)`，用于约束轮廓波动的一致性。

- `directionPenaltyMatchCost`
  与方向先验相反时加入的小惩罚。

这些字段会写入 `stitching_data.csv`，便于调参时判断是哪一项在主导选解。

### 6.2 `StitchRunRequest`

作用：

- 描述一次完整拼接运行所需的全部输入。

主要字段：

- `imagePaths`
  输入图像路径列表。

- `edgeConfig`
  边缘提取参数。

- `pipelineConfig`
  拼接流程参数。

- `panoramaOutputPath`
  最终拼接图输出路径。

- `csvOutputPath`
  CSV 输出路径。

- `debugImageOutputDir`
  调试图输出目录。

### 6.3 `StitchingResult`

作用：

- 保存拼接算法主流程的最终结果。

主要字段：

- `canvas`
  最终拼接图。

- `globalTransform`
  全局累计变换矩阵。

- `steps`
  所有步骤记录。

### 6.4 `StitchRunResult`

作用：

- 保存一次完整运行的最终返回结果。

主要字段：

- `ok`
  本次运行是否成功。

- `message`
  结果说明或错误信息。

- `stitching`
  拼接结果主体。

- `csvText`
  导出的 CSV 文本内容。

## 7. 统一服务入口

统一入口位于：

- `stitch/StitchService.h`

核心接口为：

```cpp
stitch::StitchRunResult runStitching(
    const stitch::StitchRunRequest& request,
    const stitch::StitchCallbacks& callbacks = {});
```

该接口负责组织：

- 输入图像加载
- 亚像素边缘提取
- 顺序拼接流程
- 调试图生成
- CSV 文本生成
- 可选文件保存
- 日志、进度、图像、取消回调

## 8. 当前论文级指标口径

当前导出的评估指标已经升级为“双口径”：

- 全部重叠样本统计 `All`
- 鲁棒内点样本统计 `Inlier`

论文里建议优先报告：

- `normalInlier.rmse`
- `normalInlier.meanAbs`
- `normalInlier.medianAbs`
- `normalInlier.p95Abs`
- `tangentInlier.rmse`
- `tangentInlier.meanAbs`
- `tangentCorrInlier`
- `overlapCoverageRatio`
- `inlierCoverageRatio`
- `inlierRatio`

同时建议附带：

- `normalAll.rmse`
- `tangentAll.rmse`

这样可以同时说明：

- 原始重叠质量如何
- 鲁棒筛选后稳定质量如何

## 9. Qt 接入建议

如果下一步做 Qt GUI，推荐：

- 主线程负责界面交互与参数填写
- 工作线程调用 `runStitching(...)`
- 将 `StitchCallbacks` 绑定到 Qt 信号槽
- 用 `onProgress` 更新进度条
- 用 `onLog` 更新日志窗口
- 用 `onStepFinished` 更新步骤表格
- 用 `onImage` 显示调试图与预览图
- 用 `isCancelled` 对接“停止任务”按钮

## 10. 维护规则

后续继续整理代码时，建议保持以下约定：

- 新的共享结构体优先加入 `stitch/StitchTypes.h`
- 不要重新引入 `Types.h` 之类的兼容转发头
- 新增误差指标优先落到 `ResidualStatistics` 或 `AlignmentMetrics`
- 单步过程数据统一沉淀到 `StitchStepRecord`
- 面向 GUI/CLI 的完整运行结果统一归入 `StitchRunResult`

## 11. 当前封装边界

现在这套拼接代码已经可以明确分成四个边界：

- 边缘检测边界：`core/SubpixelEdgeDetector.*`
- 类型定义边界：`stitch/StitchTypes.h`
- 服务入口边界：`stitch/StitchService.*`
- CLI 壳层边界：`main_clean.cpp`

这套边界划分的直接价值是：

- 后续替换 GUI 时不需要重写算法
- 后续补批处理时不需要改核心拼接逻辑
- 后续扩展论文级指标时，结构归属清晰，不容易再混乱
