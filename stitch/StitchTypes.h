#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace stitch {

/**
 * @brief 配准时选择的主轴方向。
 *
 * - `X`：把轮廓看成 `y = f(x)`，沿 X 方向建立对应关系
 * - `Y`：把轮廓看成 `x = f(y)`，沿 Y 方向建立对应关系
 */
enum class AlignmentAxis { X, Y };

/**
 * @brief 拼接搜索时使用的运动方向先验。
 *
 * 当已知采集过程只会沿某一个方向移动时，
 * 可以用它约束候选搜索，减少错误匹配。
 */
enum class MotionPriorDirection {
    Auto,
    XPositive,
    XNegative,
    YPositive,
    YNegative
};

enum class StitchRunMode {
    Full,
    Acquisition,
    Processing,
    Registration,
    Report
};

/**
 * @brief 亚像素边缘提取与前置滤波参数。
 *
 * 这组参数只负责“如何提取并清洗边缘点”，
 * 不涉及后续拼接搜索窗口与旋转范围。
 */
struct EdgeDetectConfig {
    double cannyLow = 50.0;
    double cannyHigh = 150.0;
    int subpixWindow = 7;
    double subpixSigma = 1.0;
    std::size_t minWarnPoints = 50;
    bool enablePointFiltering = true;      ///< 是否启用亚像素点前置滤波。
    double filterConfidenceQuantile = 0.15; ///< 按拟合置信度做分位数筛选。
    double filterGradientQuantile = 0.15;   ///< 按梯度强度做分位数筛选。
    int filterLocalLinearWindowRadius = 5;  ///< 局部线性 Hampel 滤波半窗口。
    double filterHampelSigma = 2.0;         ///< Hampel 判异阈值倍数。
    double filterHampelMinScale = 0.05;     ///< MAD 过小时使用的最小尺度，单位像素。
};

/**
 * @brief 同一条边缘在不同排序视图下的统一表示。
 *
 * 这些排序结果会直接用于后续插值匹配，
 * 避免在每次搜索时重复排序。
 */
struct EdgeVariants {
    std::vector<cv::Point2d> raw;
    std::vector<double> rawQualityWeights;
    std::vector<double> rawConfidences;
    std::vector<double> rawGradients;
    std::vector<unsigned char> rawSyntheticFlags;
    std::vector<cv::Point2d> x_sorted;
    std::vector<cv::Point2d> y_sorted;
    std::vector<cv::Point2d> negX_sorted;
    std::vector<cv::Point2d> negY_sorted;

    [[nodiscard]] bool empty() const noexcept { return raw.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return raw.size(); }

    [[nodiscard]] const std::vector<cv::Point2d>& ordered(AlignmentAxis axis) const noexcept
    {
        return axis == AlignmentAxis::X ? x_sorted : y_sorted;
    }

    [[nodiscard]] const std::vector<cv::Point2d>& mirroredOrdered(AlignmentAxis axis) const noexcept
    {
        return axis == AlignmentAxis::X ? negX_sorted : negY_sorted;
    }
};

struct ResidualStatistics {
    int sampleCount{0};
    double bias{0.0};
    double rmse{0.0};
    double meanAbs{0.0};
    double medianAbs{0.0};
    double p95Abs{0.0};
    double maxAbs{0.0};

    [[nodiscard]] bool valid() const noexcept { return sampleCount > 0; }
};

/**
 * @brief 单步配准后的质量统计。
 *
 * `All` 表示全部重叠样本，
 * `Inlier` 表示经过鲁棒筛选后的稳定样本。
 */
struct AlignmentMetrics {
    int overlapCount{0};
    int inlierCount{0};
    int trimmedOverlapCount{0};
    double inlierRatio{0.0};
    double trimmedOverlapRatio{0.0};
    double overlapSpan{0.0};
    double overlapCoverageRatio{0.0};
    double inlierSpan{0.0};
    double inlierCoverageRatio{0.0};
    double trimmedOverlapSpan{0.0};
    double trimmedOverlapCoverageRatio{0.0};

    ResidualStatistics normalAll;
    ResidualStatistics normalInlier;
    ResidualStatistics normalTrimmed;
    ResidualStatistics tangentAll;
    ResidualStatistics tangentInlier;
    ResidualStatistics tangentTrimmed;

    double tangentCorrAll{0.0};
    double tangentCorrInlier{0.0};
    double tangentCorrTrimmed{0.0};

    [[nodiscard]] bool hasOverlap() const noexcept { return overlapCount > 0; }
    [[nodiscard]] bool hasInliers() const noexcept { return inlierCount > 0; }
};

/**
 * @brief 单步相对变换与对应的质量评估结果。
 */
struct AlignmentCandidateDiagnostic {
    std::string direction;
    double dx{0.0};
    double dy{0.0};
    double da{0.0};
    double score{0.0};
    double normalMatchCost{0.0};
    double tangentResidualMatchCost{0.0};
    double tangentCorrelationMatchCost{0.0};
    double directionPenaltyMatchCost{0.0};
    AlignmentAxis axis{AlignmentAxis::X};
    AlignmentMetrics metrics;
};

struct TransformResult {
    using AlignAxis = AlignmentAxis;

    double dx{0.0};
    double dy{0.0};
    double da{0.0};
    double score{1e9};
    double normalMatchCost{0.0};
    double tangentResidualMatchCost{0.0};
    double tangentCorrelationMatchCost{0.0};
    double directionPenaltyMatchCost{0.0};
    AlignmentAxis axis{AlignmentAxis::X};
    std::string direction;
    std::vector<double> inlierErrors;
    std::vector<double> inlierCoordinates;
    std::vector<double> samplePrimaryCoordinates;
    std::vector<double> sampleRefSecondaryCoordinates;
    std::vector<double> sampleTargetSecondaryCoordinates;
    std::vector<double> sampleNormalErrors;
    std::vector<double> sampleTangentErrors;
    std::vector<int> sampleInlierFlags;
    AlignmentMetrics metrics;
    std::vector<AlignmentCandidateDiagnostic> candidateDiagnostics;

    [[nodiscard]] bool hasCandidate() const noexcept { return score < 1e8; }
};

/**
 * @brief 顺序拼接过程中每一步的记录。
 */
struct StitchStepRecord {
    std::size_t stepIndex{0};
    std::size_t referenceImageIndex{0};
    std::size_t targetImageIndex{0};
    double searchRangeX{0.0};
    double searchRangeY{0.0};
    TransformResult transform;
};

using LogCallback = std::function<void(const std::string&)>;
using ProgressCallback = std::function<void(const std::string& stage, std::size_t current, std::size_t total)>;
using StepCallback = std::function<void(const StitchStepRecord&)>;
using ImageCallback = std::function<void(const std::string& stage,
                                         std::size_t index,
                                         std::size_t total,
                                         const cv::Mat& image)>;
using CancelCallback = std::function<bool()>;

struct StitchCallbacks {
    LogCallback onLog;
    ProgressCallback onProgress;
    StepCallback onStepFinished;
    ImageCallback onImage;
    CancelCallback isCancelled;
};

/**
 * @brief 拼接搜索阶段的参数。
 *
 * 这里控制重叠度先验、方向约束、搜索窗口和旋转窗口。
 */
struct StitchPipelineConfig {
    double expectedOverlapRatio = -1.0; ///< 已知重叠度；合法时优先用于推算初始位移。
    MotionPriorDirection directionConstraint = MotionPriorDirection::Auto; ///< 方向先验。
    std::vector<MotionPriorDirection> stepDirectionConstraints; ///< 可选逐步方向先验；为空时使用 directionConstraint。
    std::vector<cv::Point2d> stepTranslationPriorsPx; ///< 可选逐步平移先验；为空时按重叠度和方向推算。
    double baseSearchRange = 200.0;     ///< 基础平移搜索范围，单位像素。
    double approxShiftRatio = 0.3;      ///< 初始位移比例；在没有显式重叠度时作为后备值。
    double rotationSearchMinDeg = -0.2; ///< 旋转搜索下界，单位度。
    double rotationSearchMaxDeg = 0.2;  ///< 旋转搜索上界，单位度。
    double rotationSearchStepDeg = 0.01; ///< 旋转搜索步长，单位度。
    double tangentResidualCostWeight = 0.05; ///< 切向残差加入匹配代价的权重。
    double tangentCorrelationCostWeight = 0.25; ///< 切向相关性不足加入匹配代价的权重。
    bool enableBadStepCandidateReselect = false; ///< 实验项：坏步候选重选，默认关闭。
    bool enableTranslationPriorFallback = false; ///< 实验项：匹配失败时回退到逐步平移先验，默认关闭。
    double translationPriorFallbackNormalRmseThreshold = 1e9; ///< 触发平移先验兜底的法向 RMSE 阈值。
    double translationPriorFallbackScoreThreshold = 1e8; ///< 触发平移先验兜底的 score 阈值。
    bool generateDebugVisualization = false; ///< 是否生成过程调试图。
    bool enableDesignComparison = true; ///< 是否执行设计母线比对。
    bool designEvaluateProfileForm = true; ///< 是否基于去均值后的法向残差评价轮廓波动。
    bool designReverseZ = true; ///< 实测从左到右是否对应原始设计 z 从大到小。
    bool designUseLeftEndpointAnchor = true; ///< 是否使用左端点作为轴向锚点。
    bool designAnchorRadialToLeftEndpoint = true; ///< 是否把径向零位锚定到左端点。
    bool designEnableBestFitTranslation = true; ///< 是否允许在锚定后做小范围平移微调。
    double designBestFitDzMinMm = -3.0; ///< 可选 best-fit 的 dz 下界。
    double designBestFitDzMaxMm = 3.0;  ///< 可选 best-fit 的 dz 上界。
    double designBestFitDrMinMm = -0.3; ///< 可选 best-fit 的 dr 下界。
    double designBestFitDrMaxMm = 0.3;  ///< 可选 best-fit 的 dr 上界。
    double designBestFitStepMm = 0.005;  ///< 可选 best-fit 的搜索步长。
    bool designEnableBestFitRotation = true; ///< 是否允许最优旋转对准。
    double designBestFitRotationMinDeg = -0.3; ///< 最优旋转搜索下界，单位度。
    double designBestFitRotationMaxDeg = 0.3;  ///< 最优旋转搜索上界，单位度。
    double designBestFitRotationStepDeg = 0.01; ///< 最优旋转搜索步长，单位度。
    double designPixelSizeMm = 0.010057; ///< 设计比对使用的像素尺寸，单位 mm/px。
    bool designInvertY = true; ///< 像素 y 向下时是否取负号映射到半径方向。
    bool designUseUpperEnvelope = true; ///< 是否保留每个 bin 的上包络点作为母线。
    double designUpperEnvelopeQuantile = 0.95; ///< 上包络分位数(0=min, 1=max)；1.0=绝对最外侧。
    double designProfileBinWidthPx = 1.0; ///< 提取母线时的主轴分箱宽度。
    bool designFilterEndFaceEdges = true; ///< 是否按局部斜率剔除端面投影边缘。
    bool designEnableProfileOutlierFilter = true; ///< 是否对轮廓误差做 Hampel 离群剔除。
    double designProfileOutlierSigma = 4.0; ///< 轮廓误差 Hampel 判异阈值。
    double designMaxAbsSlopeForGeneratrix = 3.0; ///< 母线允许的最大绝对斜率 |dy/dx|。
    int designSlopeWindow = 7; ///< 局部斜率估计滑窗长度。
    double designTrimLeftAfterEndpointMm = 0.0; ///< 左端点后额外裁掉的长度。
    double designTrimRightMm = 0.0; ///< 右端额外裁掉的长度。
};

/**
 * @brief 一次完整拼接运行的输入请求。
 */
struct StitchRunCache;

struct StitchRunRequest {
    std::vector<std::string> imagePaths;
    EdgeDetectConfig edgeConfig{};
    StitchPipelineConfig pipelineConfig{};
    std::string resultOutputDir;
    std::string panoramaOutputPath;
    std::string csvOutputPath;
    std::string designErrorProfileCsvOutputPath;
    std::string designErrorSummaryCsvOutputPath;
    std::string designComparisonOverlayOutputPath;
    std::string qualityReviewCsvOutputPath;
    std::string alignmentCandidateDiagnosticsCsvOutputPath;
    bool saveStepSummaryCsv{true};
    bool saveContourPointsCsv{false};
    bool saveStitchedContourProfileCsv{false};
    bool saveTangentStepCsv{false};
    bool saveNormalErrorProfileCsv{false};
    bool saveTangentProfileCsv{false};
    std::string contourPointsCsvOutputPath;
    std::string originContourOverlayCsvOutputPath;
    std::string stitchedContourProfileCsvOutputPath;
    std::string tangentStepCsvOutputPath;
    std::string normalErrorProfileCsvOutputPath;
    std::string tangentProfileCsvOutputPath;
    std::string originTangentPointMetricsCsvOutputPath;
    std::string debugImageOutputDir;
    std::string contourOverlayOutputPath;
    std::string stitchedContourProfilePlotOutputPath;
    std::string tangentCorrelationAllOutputPath;
    std::string tangentCorrelationInlierOutputPath;
    std::shared_ptr<const StitchRunCache> previousCache;
};

/**
 * @brief 主流程的核心输出结果。
 */
struct StitchingResult {
    cv::Mat canvas;
    cv::Mat globalTransform;
    std::vector<cv::Mat> imageTransforms;
    std::vector<StitchStepRecord> steps;

    [[nodiscard]] bool success() const noexcept { return !canvas.empty(); }
};

struct StitchRunCache {
    std::vector<std::string> imagePaths;
    EdgeDetectConfig edgeConfig{};
    StitchPipelineConfig pipelineConfig{};
    std::vector<cv::Mat> loadedImages;
    std::vector<EdgeVariants> preprocessedEdges;
    StitchingResult stitching;
    std::string csvText;

    [[nodiscard]] bool hasLoadedImages() const noexcept
    {
        return !loadedImages.empty() && loadedImages.size() == imagePaths.size();
    }

    [[nodiscard]] bool hasPreprocessedEdges() const noexcept
    {
        return hasLoadedImages() && preprocessedEdges.size() == loadedImages.size();
    }

    [[nodiscard]] bool hasStitching() const noexcept
    {
        return stitching.success();
    }
};

/**
 * @brief 一次完整运行的最终返回值。
 */
struct StitchRunResult {
    bool ok{false};
    StitchRunMode mode{StitchRunMode::Full};
    std::size_t loadedImageCount{0};
    std::size_t preprocessedImageCount{0};
    std::string message;
    StitchingResult stitching;
    std::string csvText;
    std::shared_ptr<StitchRunCache> cache;
};

} // namespace stitch
