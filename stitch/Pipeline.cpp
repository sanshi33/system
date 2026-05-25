#include "Pipeline.h"

#include "Alignment.h"
#include "Blending.h"
#include "DebugVis.h"
#include "PoseGraphOptimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace stitch {

namespace {

bool isCancelled(const StitchCallbacks& callbacks)
{
    return callbacks.isCancelled && callbacks.isCancelled();
}

void emitLog(const StitchCallbacks& callbacks, const std::string& message)
{
    if (callbacks.onLog) {
        callbacks.onLog(message);
    }
}

void emitProgress(const StitchCallbacks& callbacks,
                  const std::string& stage,
                  std::size_t current,
                  std::size_t total)
{
    if (callbacks.onProgress) {
        callbacks.onProgress(stage, current, total);
    }
}

double resolveShiftRatio(const StitchPipelineConfig& config)
{
    if (std::isfinite(config.expectedOverlapRatio) &&
        config.expectedOverlapRatio >= 0.0 &&
        config.expectedOverlapRatio <= 1.0) {
        return std::clamp(1.0 - config.expectedOverlapRatio, 0.0, 1.0);
    }

    return std::max(0.0, config.approxShiftRatio);
}

void initializeApproxShift(const cv::Mat& image,
                           const StitchPipelineConfig& config,
                           double& approxShiftX,
                           double& approxShiftY)
{
    const double shiftRatio = resolveShiftRatio(config);
    approxShiftX = 0.0;
    approxShiftY = 0.0;

    switch (config.directionConstraint) {
    case MotionPriorDirection::XPositive:
        approxShiftX = image.cols * shiftRatio;
        break;
    case MotionPriorDirection::XNegative:
        approxShiftX = -image.cols * shiftRatio;
        break;
    case MotionPriorDirection::YPositive:
        approxShiftY = image.rows * shiftRatio;
        break;
    case MotionPriorDirection::YNegative:
        approxShiftY = -image.rows * shiftRatio;
        break;
    case MotionPriorDirection::Auto:
    default:
        approxShiftX = image.cols * shiftRatio;
        approxShiftY = image.rows * shiftRatio;
        break;
    }
}

double preferredNormalRmse(const TransformResult& result)
{
    const ResidualStatistics& normal =
        result.metrics.normalInlier.valid() ? result.metrics.normalInlier : result.metrics.normalAll;
    return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
}

double preferredTangentRmse(const TransformResult& result)
{
    const ResidualStatistics& tangent =
        result.metrics.tangentInlier.valid() ? result.metrics.tangentInlier : result.metrics.tangentAll;
    return tangent.valid() ? tangent.rmse : 0.0;
}

bool shouldUseTranslationPriorFallback(const TransformResult& transform,
                                       const StitchPipelineConfig& config)
{
    if (!config.enableTranslationPriorFallback) {
        return false;
    }
    if (!transform.hasCandidate()) {
        return true;
    }
    const double normal = preferredNormalRmse(transform);
    if (std::isfinite(normal) &&
        normal > config.translationPriorFallbackNormalRmseThreshold) {
        return true;
    }
    return transform.score > config.translationPriorFallbackScoreThreshold;
}

TransformResult makeTranslationPriorFallback(double dx,
                                             double dy,
                                             double angleDeg,
                                             MotionPriorDirection direction)
{
    TransformResult fallback;
    fallback.dx = dx;
    fallback.dy = dy;
    fallback.da = angleDeg;
    fallback.score = 999.0;
    fallback.direction = "PRIOR";
    fallback.axis =
        (direction == MotionPriorDirection::YPositive ||
         direction == MotionPriorDirection::YNegative)
            ? AlignmentAxis::Y
            : AlignmentAxis::X;
    return fallback;
}

bool reselectBadStepCandidate(TransformResult& transform,
                              double approxShiftX,
                              double approxShiftY)
{
    const double currentNormal = preferredNormalRmse(transform);
    if (currentNormal <= 2.0 || transform.candidateDiagnostics.empty()) {
        return false;
    }

    double bestCost = std::numeric_limits<double>::infinity();
    const AlignmentCandidateDiagnostic* best = nullptr;
    for (const AlignmentCandidateDiagnostic& candidate : transform.candidateDiagnostics) {
        const ResidualStatistics& normal =
            candidate.metrics.normalInlier.valid() ? candidate.metrics.normalInlier
                                                   : candidate.metrics.normalAll;
        if (!normal.valid()) {
            continue;
        }
        const ResidualStatistics& tangent =
            candidate.metrics.tangentInlier.valid() ? candidate.metrics.tangentInlier
                                                    : candidate.metrics.tangentAll;
        const double normalRmse = normal.rmse;
        const double tangentRmse = tangent.valid() ? tangent.rmse : 0.0;
        if (normalRmse >= currentNormal) {
            continue;
        }

        const double primaryTolerance = 35.0;
        const double perpTolerance = 18.0;
        const double primaryDelta =
            candidate.axis == AlignmentAxis::X ? candidate.dx - approxShiftX
                                               : candidate.dy - approxShiftY;
        const double perpDelta =
            candidate.axis == AlignmentAxis::X ? candidate.dy - approxShiftY
                                               : candidate.dx - approxShiftX;
        const double priorCost =
            (primaryDelta * primaryDelta) / (primaryTolerance * primaryTolerance) +
            (perpDelta * perpDelta) / (perpTolerance * perpTolerance);
        const double angleCost = (candidate.da * candidate.da) / (0.2 * 0.2);
        const double cost = normalRmse * normalRmse +
                            0.04 * tangentRmse * tangentRmse +
                            0.12 * priorCost +
                            0.02 * angleCost;
        if (cost < bestCost) {
            bestCost = cost;
            best = &candidate;
        }
    }

    if (best == nullptr) {
        return false;
    }

    const double currentPriorDx = transform.dx - approxShiftX;
    const double currentPriorDy = transform.dy - approxShiftY;
    const double currentPriorCost =
        (currentPriorDx * currentPriorDx) / (35.0 * 35.0) +
        (currentPriorDy * currentPriorDy) / (18.0 * 18.0);
    const double currentCost = currentNormal * currentNormal +
                               0.04 * preferredTangentRmse(transform) * preferredTangentRmse(transform) +
                               0.12 * currentPriorCost +
                               0.02 * (transform.da * transform.da) / (0.2 * 0.2);
    if (bestCost >= currentCost * 0.98) {
        return false;
    }

    TransformResult selected;
    selected.dx = best->dx;
    selected.dy = best->dy;
    selected.da = best->da;
    selected.score = best->score;
    selected.normalMatchCost = best->normalMatchCost;
    selected.tangentResidualMatchCost = best->tangentResidualMatchCost;
    selected.tangentCorrelationMatchCost = best->tangentCorrelationMatchCost;
    selected.directionPenaltyMatchCost = best->directionPenaltyMatchCost;
    selected.axis = best->axis;
    selected.direction = best->direction;
    selected.metrics = best->metrics;
    selected.candidateDiagnostics = transform.candidateDiagnostics;
    transform = std::move(selected);
    return true;
}

} // namespace

StitchingResult runStitchingPipeline(const std::vector<cv::Mat>& images,
                                     const std::vector<EdgeVariants>& edgesPrepared,
                                     const StitchPipelineConfig& config,
                                     const StitchCallbacks& callbacks)
{
    StitchingResult result;
    if (images.empty() || edgesPrepared.size() < images.size()) {
        return result;
    }

    initCanvasAndPlaceFirst(images[0], result.canvas, result.globalTransform);
    result.imageTransforms.push_back(result.globalTransform.clone());

    EdgeVariants previousEdges = edgesPrepared[0];

    double approxShiftX = 0.0;
    double approxShiftY = 0.0;
    double approxAngleDeg = 0.0;
    bool hasReliableMotionPrior = false;
    TransformResult lastReliableTransform;
    bool hasLastReliableTransform = false;
    initializeApproxShift(images[0], config, approxShiftX, approxShiftY);
    double approxInitX = approxShiftX;
    double approxInitY = approxShiftY;
    MotionPriorDirection previousDirection = config.directionConstraint;

    const std::size_t totalSteps = images.size() > 1 ? images.size() - 1 : 0;

    for (std::size_t i = 0; i + 1 < images.size(); ++i) {
        if (isCancelled(callbacks)) {
            emitLog(callbacks, "[信息] 配准拼接已取消。");
            break;
        }

        emitProgress(callbacks, "stitch", i + 1, totalSteps);
        emitLog(callbacks,
                "\n>>> 正在拼接图像 " + std::to_string(i + 1) + " 与图像 " + std::to_string(i + 2) + "...");

        EdgeVariants nextEdges = edgesPrepared[i + 1];
        if (nextEdges.size() < 100) {
            emitLog(callbacks, "[错误] 图像 " + std::to_string(i + 2) + " 的边缘点数量不足。");
            continue;
        }

        const cv::Point2d center(images[i].cols / 2.0, images[i].rows / 2.0);
        double searchRangeX = 0.0;
        double searchRangeY = 0.0;
        const MotionPriorDirection stepDirection =
            i < config.stepDirectionConstraints.size()
                ? config.stepDirectionConstraints[i]
                : config.directionConstraint;
        if (i == 0 || stepDirection != previousDirection) {
            StitchPipelineConfig stepInitConfig = config;
            stepInitConfig.directionConstraint = stepDirection;
            initializeApproxShift(images[i], stepInitConfig, approxShiftX, approxShiftY);
            approxInitX = approxShiftX;
            approxInitY = approxShiftY;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
            hasLastReliableTransform = false;
        }
        if (i < config.stepTranslationPriorsPx.size()) {
            approxShiftX = config.stepTranslationPriorsPx[i].x;
            approxShiftY = config.stepTranslationPriorsPx[i].y;
            approxInitX = approxShiftX;
            approxInitY = approxShiftY;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
            hasLastReliableTransform = false;
        }
        TransformResult transform = matchOnePair(previousEdges, nextEdges, center,
                                                 approxShiftX, approxShiftY,
                                                 approxAngleDeg, hasReliableMotionPrior,
                                                 config.baseSearchRange,
                                                  stepDirection,
                                                  config.rotationSearchMinDeg,
                                                  config.rotationSearchMaxDeg,
                                                  config.rotationSearchStepDeg,
                                                  config.tangentResidualCostWeight,
                                                  config.tangentCorrelationCostWeight,
                                                  false,
                                                  0.0,
                                                  0.0,
                                                  searchRangeX, searchRangeY);

        bool useQualityLocalRescan = false;
        bool useCandidateReselect = false;
        bool usedMotionPriorFallback = false;
        const auto normalRmseFor = [](const TransformResult& result) {
            const ResidualStatistics& normal =
                result.metrics.normalInlier.valid() ? result.metrics.normalInlier : result.metrics.normalAll;
            return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
        };
        const bool needsQualityRescan =
            normalRmseFor(transform) > 0.35 ||
            std::abs(transform.da) > std::max(std::abs(config.rotationSearchMinDeg),
                                              std::abs(config.rotationSearchMaxDeg)) + 1e-9;
        if (needsQualityRescan) {
            double localSearchRangeX = 0.0;
            double localSearchRangeY = 0.0;
            const double localPriorX = hasLastReliableTransform ? lastReliableTransform.dx : approxShiftX;
            const double localPriorY = hasLastReliableTransform ? lastReliableTransform.dy : approxShiftY;
            const double localPriorAngle = hasLastReliableTransform ? lastReliableTransform.da : approxAngleDeg;
            const double localRotationHalfRangeDeg = 0.20;
            const double localRotationMinDeg =
                std::max(config.rotationSearchMinDeg, localPriorAngle - localRotationHalfRangeDeg);
            const double localRotationMaxDeg =
                std::min(config.rotationSearchMaxDeg, localPriorAngle + localRotationHalfRangeDeg);
            const double localRotationStepDeg = std::min(config.rotationSearchStepDeg, 0.01);
            TransformResult localTransform =
                matchOnePair(previousEdges, nextEdges, center,
                             localPriorX, localPriorY,
                             localPriorAngle, hasReliableMotionPrior || hasLastReliableTransform,
                             config.baseSearchRange,
                             stepDirection,
                             localRotationMinDeg,
                             localRotationMaxDeg,
                             localRotationStepDeg,
                             config.tangentResidualCostWeight,
                             config.tangentCorrelationCostWeight,
                             true,
                             100.0,
                             40.0,
                             localSearchRangeX, localSearchRangeY);
            if (normalRmseFor(localTransform) < normalRmseFor(transform)) {
                transform = localTransform;
                searchRangeX = localSearchRangeX;
                searchRangeY = localSearchRangeY;
                useQualityLocalRescan = true;
            } else {
                for (auto diagnostic : localTransform.candidateDiagnostics) {
                    diagnostic.direction = "local:" + diagnostic.direction;
                    transform.candidateDiagnostics.push_back(std::move(diagnostic));
                }
            }
        }
        if (config.enableBadStepCandidateReselect &&
            reselectBadStepCandidate(transform, approxShiftX, approxShiftY)) {
            useCandidateReselect = true;
        }
        if (shouldUseTranslationPriorFallback(transform, config)) {
            transform = makeTranslationPriorFallback(approxShiftX,
                                                     approxShiftY,
                                                     approxAngleDeg,
                                                     stepDirection);
            searchRangeX = 0.0;
            searchRangeY = 0.0;
            usedMotionPriorFallback = true;
        }

        StitchStepRecord step;
        step.stepIndex = i + 1;
        step.referenceImageIndex = i;
        step.targetImageIndex = i + 1;
        step.searchRangeX = searchRangeX;
        step.searchRangeY = searchRangeY;
        step.transform = transform;
        result.steps.push_back(step);

        if (useQualityLocalRescan) {
            emitLog(callbacks, "    [信息] 最后一步质量触发强先验局部重搜，并采用更低法向 RMSE 的结果。");
        }
        if (useCandidateReselect) {
            emitLog(callbacks, "    [Info] Bad-step candidate reselected by RMSE and motion-prior consistency.");
        }
        emitLog(callbacks, formatStepSummary(step));
        if (usedMotionPriorFallback) {
            emitLog(callbacks, "    [警告] 当前步未找到可信局部极值，已回退到上一可靠步的相对位姿作为保底。");
        }
        if (transform.direction == "N/A") {
            emitLog(callbacks, "    [警告] 未能选出可靠方向（可能是搜索代价过高或重叠区域过小）。");
        }

        if (callbacks.onStepFinished) {
            callbacks.onStepFinished(step);
        }

        if (config.generateDebugVisualization && callbacks.onImage) {
            const cv::Mat debugImage = buildDebugStepVisualization(
                images[i],
                images[i + 1],
                previousEdges.ordered(transform.axis),
                nextEdges.ordered(transform.axis),
                transform,
                static_cast<int>(step.stepIndex));
            callbacks.onImage("debug_step", step.stepIndex, totalSteps, debugImage);
        }

        applyTransformAndBlend(images[i + 1], result.canvas, result.globalTransform, center, transform);
        result.imageTransforms.push_back(result.globalTransform.clone());
        previousEdges = nextEdges;

        // 这里用法向 RMSE 判断下一步初值是否可信，避免切向代价权重改变 score 尺度后误触发重置。
        const ResidualStatistics& normalForReliability =
            transform.metrics.normalInlier.valid() ? transform.metrics.normalInlier : transform.metrics.normalAll;
        const bool reliableMatch = normalForReliability.valid()
                                       ? normalForReliability.rmse < std::sqrt(0.5)
                                       : transform.score < 0.5;
        if (reliableMatch || usedMotionPriorFallback) {
            // EMA 平滑运动先验，降低单步噪声传导
            constexpr double kPriorAlpha = 0.7;
            approxShiftX = kPriorAlpha * transform.dx + (1.0 - kPriorAlpha) * approxShiftX;
            approxShiftY = kPriorAlpha * transform.dy + (1.0 - kPriorAlpha) * approxShiftY;
            approxAngleDeg = transform.da;

            // 硬约束：与初始位移的最大偏离（防止积累漂移）
            constexpr double kMaxPriorDeviationPx = 50.0;
            const double deviation = std::hypot(approxShiftX - approxInitX, approxShiftY - approxInitY);
            if (deviation > kMaxPriorDeviationPx) {
                const double scale = kMaxPriorDeviationPx / deviation;
                approxShiftX = approxInitX + scale * (approxShiftX - approxInitX);
                approxShiftY = approxInitY + scale * (approxShiftY - approxInitY);
            }

            hasReliableMotionPrior = true;
            lastReliableTransform = transform;
            hasLastReliableTransform = true;
        } else {
            approxShiftX = approxInitX;
            approxShiftY = approxInitY;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
            emitLog(callbacks, "    [警告] 匹配得分过高，已将近似位移重置为初始值。");
        }
        previousDirection = stepDirection;
    }

    // --- 位姿图全局优化 ---
    // 用邻边 + 跳边约束优化所有图像的全局位姿，减少顺序拼接的误差积累
    if (result.imageTransforms.size() >= 2 && !result.steps.empty()) {
        auto poseEdges = buildAdjacentEdges(result.steps);

        // 补充跳边约束 (i -> i+2)，权重 0.25
        const auto skip2 = buildSkipEdges(result.imageTransforms, 1, 0.25);
        poseEdges.insert(poseEdges.end(), skip2.begin(), skip2.end());

        // 补充跳边约束 (i -> i+3)，权重 0.1
        const auto skip3 = buildSkipEdges(result.imageTransforms, 2, 0.10);
        poseEdges.insert(poseEdges.end(), skip3.begin(), skip3.end());

        const std::vector<cv::Mat> optimized =
            optimizePoseGraph(result.imageTransforms, poseEdges, 0.5, 30);

        if (optimized.size() == result.imageTransforms.size()) {
            double maxDeltaDx = 0.0, maxDeltaDy = 0.0;
            for (std::size_t i = 0; i < optimized.size(); ++i) {
                maxDeltaDx = std::max(maxDeltaDx,
                    std::abs(optimized[i].at<double>(0, 2) - result.imageTransforms[i].at<double>(0, 2)));
                maxDeltaDy = std::max(maxDeltaDy,
                    std::abs(optimized[i].at<double>(1, 2) - result.imageTransforms[i].at<double>(1, 2)));
            }
            result.imageTransforms = optimized;
            result.globalTransform = optimized.back().clone();
            emitLog(callbacks,
                    "    [位姿图] 优化完成，最大平移修正 dx=" +
                        std::to_string(static_cast<int>(maxDeltaDx)) +
                        " px, dy=" + std::to_string(static_cast<int>(maxDeltaDy)) + " px");
        }
    }

    if (!result.canvas.empty()) {
        const cv::Rect cropBox = cropCanvasAuto(result.canvas);
        if (cropBox.width > 0 && cropBox.height > 0) {
            cv::Mat cropTransform = cv::Mat::eye(3, 3, CV_64F);
            cropTransform.at<double>(0, 2) = -static_cast<double>(cropBox.x);
            cropTransform.at<double>(1, 2) = -static_cast<double>(cropBox.y);
            result.globalTransform = cropTransform * result.globalTransform;
            for (cv::Mat& transform : result.imageTransforms) {
                transform = cropTransform * transform;
            }
        }
    }

    return result;
}

} // namespace stitch
