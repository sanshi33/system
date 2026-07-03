#include "Pipeline.h"

#include "cad_design/DesignProfileAlignment.h"
#include "Alignment.h"
#include "Blending.h"
#include "DebugVis.h"
#include "GeometryUtils.h"
#include "PoseGraphOptimizer.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

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

double medianOfValues(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t midIndex = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(midIndex), values.end());
    double median = values[midIndex];
    if (values.size() % 2 == 0) {
        const auto lowerMid = std::max_element(values.begin(),
                                               values.begin() + static_cast<std::ptrdiff_t>(midIndex));
        median = 0.5 * (median + *lowerMid);
    }
    return median;
}

double medianAbsoluteDeviation(const std::vector<double>& values, double center)
{
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) {
        deviations.push_back(std::abs(value - center));
    }
    return medianOfValues(std::move(deviations));
}

bool directionUsesPrimaryX(MotionPriorDirection direction)
{
    return direction != MotionPriorDirection::YPositive &&
           direction != MotionPriorDirection::YNegative;
}

struct TrajectoryPriorEstimate {
    bool ok{false};
    double dx{0.0};
    double dy{0.0};
    double angleDeg{0.0};
    double primaryWindowPx{0.0};
    double perpWindowPx{0.0};
    double angleWindowDeg{0.0};
};

struct ImageCorrelationShiftEstimate {
    bool ok{false};
    double dx{0.0};
    double dy{0.0};
    double score{0.0};
};

TrajectoryPriorEstimate estimateTrajectoryPrior(const std::vector<TransformResult>& history,
                                                MotionPriorDirection stepDirection)
{
    TrajectoryPriorEstimate estimate;
    if (history.size() < 3) {
        return estimate;
    }

    std::vector<double> dxValues;
    std::vector<double> dyValues;
    std::vector<double> angleValues;
    std::vector<double> primaryValues;
    std::vector<double> perpValues;
    dxValues.reserve(history.size());
    dyValues.reserve(history.size());
    angleValues.reserve(history.size());
    primaryValues.reserve(history.size());
    perpValues.reserve(history.size());

    const bool primaryIsX = directionUsesPrimaryX(stepDirection);
    for (const TransformResult& transform : history) {
        if (!transform.hasCandidate()) {
            continue;
        }
        dxValues.push_back(transform.dx);
        dyValues.push_back(transform.dy);
        angleValues.push_back(transform.da);
        primaryValues.push_back(primaryIsX ? transform.dx : transform.dy);
        perpValues.push_back(primaryIsX ? transform.dy : transform.dx);
    }

    if (primaryValues.size() < 3) {
        return estimate;
    }

    const double dxMedian = medianOfValues(dxValues);
    const double dyMedian = medianOfValues(dyValues);
    const double angleMedian = medianOfValues(angleValues);
    const double primaryMedian = medianOfValues(primaryValues);
    const double perpMedian = medianOfValues(perpValues);
    const double primaryMad = medianAbsoluteDeviation(primaryValues, primaryMedian);
    const double perpMad = medianAbsoluteDeviation(perpValues, perpMedian);
    const double angleMad = medianAbsoluteDeviation(angleValues, angleMedian);

    const bool stableHistory =
        primaryMad <= 18.0 && perpMad <= 6.0 && angleMad <= 0.08;
    if (!stableHistory) {
        return estimate;
    }

    estimate.ok = true;
    estimate.dx = dxMedian;
    estimate.dy = dyMedian;
    estimate.angleDeg = angleMedian;
    estimate.primaryWindowPx = std::clamp(10.0 + 3.0 * std::max(primaryMad, 1.0), 12.0, 36.0);
    estimate.perpWindowPx = std::clamp(4.0 + 3.0 * std::max(perpMad, 0.5), 5.0, 14.0);
    estimate.angleWindowDeg = std::clamp(0.02 + 3.0 * std::max(angleMad, 0.008), 0.04, 0.12);
    if (!primaryIsX) {
        estimate.dx = perpMedian;
        estimate.dy = primaryMedian;
    }

    return estimate;
}

bool touchesRotationBoundary(double angleDeg,
                             double rotationMinDeg,
                             double rotationMaxDeg,
                             double toleranceDeg = 0.02)
{
    if (!std::isfinite(angleDeg) ||
        !std::isfinite(rotationMinDeg) ||
        !std::isfinite(rotationMaxDeg)) {
        return false;
    }

    if (rotationMinDeg > rotationMaxDeg) {
        std::swap(rotationMinDeg, rotationMaxDeg);
    }

    return std::abs(angleDeg - rotationMinDeg) <= toleranceDeg ||
           std::abs(angleDeg - rotationMaxDeg) <= toleranceDeg;
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

double preferredCoverageRatio(const TransformResult& result)
{
    if (result.metrics.inlierCount > 0) {
        return result.metrics.inlierCoverageRatio;
    }
    return result.metrics.overlapCoverageRatio;
}

double preferredTangentCorrelation(const TransformResult& result)
{
    return result.metrics.tangentInlier.valid() ? result.metrics.tangentCorrInlier
                                                : result.metrics.tangentCorrAll;
}

double primaryShiftDeltaPx(AlignmentAxis axis,
                           double dx,
                           double dy,
                           double approxShiftX,
                           double approxShiftY)
{
    return axis == AlignmentAxis::X ? dx - approxShiftX : dy - approxShiftY;
}

double perpendicularShiftDeltaPx(AlignmentAxis axis,
                                 double dx,
                                 double dy,
                                 double approxShiftX,
                                 double approxShiftY)
{
    return axis == AlignmentAxis::X ? dy - approxShiftY : dx - approxShiftX;
}

double computeStepRmseConsistencyCost(double normalRmse,
                                      double targetNormalRmsePx)
{
    if (!std::isfinite(normalRmse) || !std::isfinite(targetNormalRmsePx)) {
        return 0.0;
    }

    constexpr double kConsistencyDeadbandPx = 0.01;
    constexpr double kConsistencyTolerancePx = 0.04;
    const double excess = std::max(0.0, std::abs(normalRmse - targetNormalRmsePx) - kConsistencyDeadbandPx);
    if (!(excess > 0.0)) {
        return 0.0;
    }

    const double normalized = std::min(excess / kConsistencyTolerancePx, 3.0);
    return normalized * normalized;
}

double estimateRecentStableRmsePx(const std::vector<TransformResult>& reliableHistory,
                                  const TransformResult* lastReliableTransform)
{
    std::vector<double> values;
    values.reserve(4);
    constexpr std::size_t kHistorySize = 4;
    const std::size_t startIndex =
        reliableHistory.size() > kHistorySize ? reliableHistory.size() - kHistorySize : 0;
    for (std::size_t i = startIndex; i < reliableHistory.size(); ++i) {
        const double value = preferredNormalRmse(reliableHistory[i]);
        if (std::isfinite(value)) {
            values.push_back(value);
        }
    }

    if (values.empty() && lastReliableTransform != nullptr) {
        const double value = preferredNormalRmse(*lastReliableTransform);
        if (std::isfinite(value)) {
            values.push_back(value);
        }
    }

    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double computeSelectionCost(double normalRmse,
                            double tangentRmse,
                            double primaryDeltaPx,
                            double perpDeltaPx,
                            double angleDeg,
                            double targetNormalRmsePx)
{
    const double primaryTolerancePx = 32.0;
    const double perpTolerancePx = 14.0;
    const double angleToleranceDeg = 0.18;
    const double priorCost =
        (primaryDeltaPx * primaryDeltaPx) / (primaryTolerancePx * primaryTolerancePx) +
        (perpDeltaPx * perpDeltaPx) / (perpTolerancePx * perpTolerancePx);
    const double angleCost = (angleDeg * angleDeg) / (angleToleranceDeg * angleToleranceDeg);
    const double consistencyCost = computeStepRmseConsistencyCost(normalRmse, targetNormalRmsePx);
    return normalRmse * normalRmse +
           0.06 * tangentRmse * tangentRmse +
           0.03 * priorCost +
           0.01 * angleCost +
           0.004 * consistencyCost;
}

double transformSelectionCost(const TransformResult& result,
                              double approxShiftX,
                              double approxShiftY,
                              double targetNormalRmsePx)
{
    return computeSelectionCost(preferredNormalRmse(result),
                                preferredTangentRmse(result),
                                primaryShiftDeltaPx(result.axis, result.dx, result.dy, approxShiftX, approxShiftY),
                                perpendicularShiftDeltaPx(result.axis, result.dx, result.dy, approxShiftX, approxShiftY),
                                result.da,
                                targetNormalRmsePx);
}

void transformPointsByMatrix(const std::vector<cv::Point2d>& input,
                             std::vector<cv::Point2d>& output,
                             const cv::Mat& matrix)
{
    output.resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const cv::Point2d& point = input[index];
        const double x = matrix.at<double>(0, 0) * point.x +
                         matrix.at<double>(0, 1) * point.y +
                         matrix.at<double>(0, 2);
        const double y = matrix.at<double>(1, 0) * point.x +
                         matrix.at<double>(1, 1) * point.y +
                         matrix.at<double>(1, 2);
        const double w = matrix.at<double>(2, 0) * point.x +
                         matrix.at<double>(2, 1) * point.y +
                         matrix.at<double>(2, 2);
        if (std::abs(w) > 1e-12) {
            output[index] = cv::Point2d(x / w, y / w);
        } else {
            output[index] = cv::Point2d(x, y);
        }
    }
}

EdgeVariants buildMatchingEdgeView(const EdgeVariants& source, bool useUnfiltered)
{
    if (!useUnfiltered || !source.hasUnfiltered()) {
        return source;
    }

    EdgeVariants view;
    view.raw = source.unfiltered_raw;
    view.x_sorted = source.unfiltered_x_sorted;
    view.y_sorted = source.unfiltered_y_sorted;
    view.negX_sorted = source.unfiltered_negX_sorted;
    view.negY_sorted = source.unfiltered_negY_sorted;
    return view;
}

void fillOrderedViewsFromRaw(const std::vector<cv::Point2d>& rawPoints,
                             EdgeVariants& view)
{
    view.raw = rawPoints;
    view.x_sorted = rawPoints;
    view.y_sorted = rawPoints;
    view.negX_sorted = rawPoints;
    view.negY_sorted = rawPoints;

    sortContourByX(view.x_sorted);
    sortContourByY(view.y_sorted);

    for (auto& point : view.negX_sorted) {
        point.x = -point.x;
    }
    sortContourByX(view.negX_sorted);

    for (auto& point : view.negY_sorted) {
        point.y = -point.y;
    }
    sortContourByY(view.negY_sorted);
}

bool directionUsesNegativePrimary(MotionPriorDirection direction)
{
    return direction == MotionPriorDirection::XNegative ||
           direction == MotionPriorDirection::YNegative;
}

MotionPriorDirection reversePrimaryDirection(MotionPriorDirection direction)
{
    switch (direction) {
    case MotionPriorDirection::XPositive:
        return MotionPriorDirection::XNegative;
    case MotionPriorDirection::XNegative:
        return MotionPriorDirection::XPositive;
    case MotionPriorDirection::YPositive:
        return MotionPriorDirection::YNegative;
    case MotionPriorDirection::YNegative:
        return MotionPriorDirection::YPositive;
    default:
        return direction;
    }
}

EdgeVariants buildTrimmedMatchingEdgeView(const EdgeVariants& source,
                                          bool useUnfiltered,
                                          MotionPriorDirection stepDirection,
                                          double trimTailFraction)
{
    EdgeVariants baseView = buildMatchingEdgeView(source, useUnfiltered);
    if (trimTailFraction <= 0.0 ||
        trimTailFraction >= 0.90 ||
        baseView.raw.size() < 512) {
        return baseView;
    }

    std::vector<cv::Point2d> trimmed = baseView.raw;
    if (directionUsesPrimaryX(stepDirection)) {
        sortContourByX(trimmed);
    } else {
        sortContourByY(trimmed);
    }

    const std::size_t keepCount = std::clamp<std::size_t>(
        static_cast<std::size_t>(std::llround(static_cast<double>(trimmed.size()) * (1.0 - trimTailFraction))),
        512,
        trimmed.size());
    if (keepCount >= trimmed.size()) {
        return baseView;
    }

    const std::size_t trimCount = trimmed.size() - keepCount;
    if (directionUsesNegativePrimary(stepDirection)) {
        trimmed.erase(trimmed.begin(), trimmed.begin() + static_cast<std::ptrdiff_t>(trimCount));
    } else {
        trimmed.resize(keepCount);
    }

    EdgeVariants trimmedView;
    fillOrderedViewsFromRaw(trimmed, trimmedView);
    return trimmedView;
}

double candidatePreferredNormalRmse(const AlignmentCandidateDiagnostic& candidate)
{
    const ResidualStatistics& normal =
        candidate.metrics.normalInlier.valid() ? candidate.metrics.normalInlier
                                               : candidate.metrics.normalAll;
    return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
}

double candidatePreferredTangentRmse(const AlignmentCandidateDiagnostic& candidate)
{
    const ResidualStatistics& tangent =
        candidate.metrics.tangentInlier.valid() ? candidate.metrics.tangentInlier
                                                : candidate.metrics.tangentAll;
    return tangent.valid() ? tangent.rmse : 0.0;
}

double candidatePreferredCoverageRatio(const AlignmentCandidateDiagnostic& candidate)
{
    if (candidate.metrics.inlierCount > 0) {
        return candidate.metrics.inlierCoverageRatio;
    }
    return candidate.metrics.overlapCoverageRatio;
}

double candidatePreferredTangentCorrelation(const AlignmentCandidateDiagnostic& candidate)
{
    return candidate.metrics.tangentInlier.valid() ? candidate.metrics.tangentCorrInlier
                                                   : candidate.metrics.tangentCorrAll;
}

double candidateSelectionCost(const AlignmentCandidateDiagnostic& candidate,
                              double approxShiftX,
                              double approxShiftY,
                              double targetNormalRmsePx)
{
    return computeSelectionCost(candidatePreferredNormalRmse(candidate),
                                candidatePreferredTangentRmse(candidate),
                                primaryShiftDeltaPx(candidate.axis,
                                                    candidate.dx,
                                                    candidate.dy,
                                                    approxShiftX,
                                                    approxShiftY),
                                perpendicularShiftDeltaPx(candidate.axis,
                                                          candidate.dx,
                                                          candidate.dy,
                                                          approxShiftX,
                                                          approxShiftY),
                                candidate.da,
                                targetNormalRmsePx);
}

bool isSuspiciousTransform(const TransformResult& result,
                           double approxShiftX,
                           double approxShiftY)
{
    constexpr double kSuspiciousNormalRmsePx = 0.11;
    constexpr double kSuspiciousPrimaryDeltaPx = 18.0;
    constexpr double kSuspiciousPerpDeltaPx = 10.0;
    constexpr double kSuspiciousPenaltyCost = 0.08;

    const double normalRmse = preferredNormalRmse(result);
    const double primaryDelta =
        primaryShiftDeltaPx(result.axis, result.dx, result.dy, approxShiftX, approxShiftY);
    const double perpDelta =
        perpendicularShiftDeltaPx(result.axis, result.dx, result.dy, approxShiftX, approxShiftY);

    return normalRmse > kSuspiciousNormalRmsePx ||
           std::abs(primaryDelta) > kSuspiciousPrimaryDeltaPx ||
           std::abs(perpDelta) > kSuspiciousPerpDeltaPx ||
           result.directionPenaltyMatchCost > kSuspiciousPenaltyCost;
}

double continuityDistanceToPrior(const TransformResult& result,
                                 double priorShiftX,
                                 double priorShiftY)
{
    const double primaryDelta =
        primaryShiftDeltaPx(result.axis, result.dx, result.dy, priorShiftX, priorShiftY);
    const double perpDelta =
        perpendicularShiftDeltaPx(result.axis, result.dx, result.dy, priorShiftX, priorShiftY);
    return std::hypot(primaryDelta, perpDelta);
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

const char* motionDirectionLabel(MotionPriorDirection direction)
{
    switch (direction) {
    case MotionPriorDirection::XPositive:
        return "X+";
    case MotionPriorDirection::XNegative:
        return "X-";
    case MotionPriorDirection::YPositive:
        return "Y+";
    case MotionPriorDirection::YNegative:
        return "Y-";
    case MotionPriorDirection::Auto:
    default:
        return "AUTO";
    }
}

cv::Mat buildPrimaryToNormalShearMatrix(const cv::Point2d& center,
                                        MotionPriorDirection direction,
                                        double shearPrimaryToNormal)
{
    cv::Mat shear = cv::Mat::eye(3, 3, CV_64F);
    if (directionUsesPrimaryX(direction)) {
        shear.at<double>(1, 0) = shearPrimaryToNormal;
        shear.at<double>(1, 2) = -shearPrimaryToNormal * center.x;
    } else {
        shear.at<double>(0, 1) = shearPrimaryToNormal;
        shear.at<double>(0, 2) = -shearPrimaryToNormal * center.y;
    }
    return shear;
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

TransformResult buildEvaluatedPriorTransform(const EdgeVariants& previousEdges,
                                             const EdgeVariants& nextEdges,
                                             const cv::Point2d& center,
                                             double dx,
                                             double dy,
                                             double angleDeg,
                                             MotionPriorDirection direction,
                                             double tangentResidualCostWeight,
                                             double tangentCorrelationCostWeight,
                                             bool useUnfoldedTangentCorrelation = true)
{
    TransformResult fallback = makeTranslationPriorFallback(dx, dy, angleDeg, direction);
    fallback.score = 0.0;
    fallback.direction = motionDirectionLabel(direction);

    std::vector<cv::Point2d> rotatedTarget = nextEdges.raw;
    rotatePoints(nextEdges.raw, rotatedTarget, angleDeg, center);
    if (fallback.axis == AlignmentAxis::X) {
        sortContourByX(rotatedTarget);
    } else {
        sortContourByY(rotatedTarget);
    }

    populateAlignmentMetrics(previousEdges.ordered(fallback.axis),
                             rotatedTarget,
                             fallback,
                             useUnfoldedTangentCorrelation);

    const double normalRmse = preferredNormalRmse(fallback);
    const double tangentRmse = preferredTangentRmse(fallback);
    const double tangentCorrelation = std::clamp(preferredTangentCorrelation(fallback), -1.0, 1.0);
    fallback.normalMatchCost = std::isfinite(normalRmse) ? normalRmse * normalRmse : 1e6;
    fallback.tangentResidualMatchCost = tangentResidualCostWeight * tangentRmse * tangentRmse;
    fallback.tangentCorrelationMatchCost =
        tangentCorrelationCostWeight * std::max(0.0, 1.0 - tangentCorrelation);
    fallback.directionPenaltyMatchCost = 0.0;
    fallback.score = fallback.normalMatchCost +
                     fallback.tangentResidualMatchCost +
                     fallback.tangentCorrelationMatchCost;
    fallback.direction = "PRIOR";
    if (!fallback.metrics.hasOverlap()) {
        fallback.score = 999.0;
    }
    return fallback;
}

TransformResult buildEvaluatedCustomMatrixTransform(const EdgeVariants& previousEdges,
                                                    const EdgeVariants& nextEdges,
                                                    const cv::Mat& relativeMatrix,
                                                    const TransformResult& baseTransform,
                                                    MotionPriorDirection direction,
                                                    double tangentResidualCostWeight,
                                                    double tangentCorrelationCostWeight,
                                                    bool useUnfoldedTangentCorrelation = true)
{
    TransformResult evaluated = baseTransform;
    evaluated.score = 0.0;
    evaluated.direction = motionDirectionLabel(direction);
    evaluated.hasCustomRelativeMatrix = true;
    evaluated.customRelativeMatrix = relativeMatrix.clone();

    std::vector<cv::Point2d> transformedTarget;
    transformPointsByMatrix(nextEdges.raw, transformedTarget, relativeMatrix);
    if (evaluated.axis == AlignmentAxis::X) {
        sortContourByX(transformedTarget);
    } else {
        sortContourByY(transformedTarget);
    }

    populateAlignmentMetrics(previousEdges.ordered(evaluated.axis),
                             transformedTarget,
                             evaluated,
                             useUnfoldedTangentCorrelation);

    const double normalRmse = preferredNormalRmse(evaluated);
    const double tangentRmse = preferredTangentRmse(evaluated);
    const double tangentCorrelation = std::clamp(preferredTangentCorrelation(evaluated), -1.0, 1.0);
    evaluated.normalMatchCost = std::isfinite(normalRmse) ? normalRmse * normalRmse : 1e6;
    evaluated.tangentResidualMatchCost = tangentResidualCostWeight * tangentRmse * tangentRmse;
    evaluated.tangentCorrelationMatchCost =
        tangentCorrelationCostWeight * std::max(0.0, 1.0 - tangentCorrelation);
    evaluated.directionPenaltyMatchCost = 0.0;
    evaluated.score = evaluated.normalMatchCost +
                      evaluated.tangentResidualMatchCost +
                      evaluated.tangentCorrelationMatchCost;
    if (!evaluated.metrics.hasOverlap()) {
        evaluated.score = 999.0;
    }
    return evaluated;
}

ImageCorrelationShiftEstimate estimateImageCorrelationShift(const cv::Mat& referenceImage,
                                                            const cv::Mat& targetImage,
                                                            MotionPriorDirection stepDirection,
                                                            double primaryPriorShiftPx,
                                                            double secondaryPriorShiftPx,
                                                            double nominalPrimaryShiftPx)
{
    ImageCorrelationShiftEstimate estimate;
    if (referenceImage.empty() || targetImage.empty()) {
        return estimate;
    }

    const bool primaryIsY =
        stepDirection == MotionPriorDirection::YPositive || stepDirection == MotionPriorDirection::YNegative;
    const bool negativeDirection =
        stepDirection == MotionPriorDirection::XNegative || stepDirection == MotionPriorDirection::YNegative;

    cv::Mat refGray;
    cv::Mat targetGray;
    if (referenceImage.channels() == 1) {
        referenceImage.convertTo(refGray, CV_32F);
    } else {
        cv::cvtColor(referenceImage, refGray, cv::COLOR_BGR2GRAY);
        refGray.convertTo(refGray, CV_32F);
    }
    if (targetImage.channels() == 1) {
        targetImage.convertTo(targetGray, CV_32F);
    } else {
        cv::cvtColor(targetImage, targetGray, cv::COLOR_BGR2GRAY);
        targetGray.convertTo(targetGray, CV_32F);
    }

    if (primaryIsY) {
        cv::transpose(refGray, refGray);
        cv::transpose(targetGray, targetGray);
    }

    const int cropX = std::max(8, static_cast<int>(std::lround(refGray.cols * 0.02)));
    const int cropY = std::max(8, static_cast<int>(std::lround(refGray.rows * 0.04)));
    if (refGray.cols <= 2 * cropX + 64 || refGray.rows <= 2 * cropY + 64) {
        return estimate;
    }
    const cv::Rect cropRect(cropX, cropY, refGray.cols - 2 * cropX, refGray.rows - 2 * cropY);
    refGray = refGray(cropRect).clone();
    targetGray = targetGray(cropRect).clone();

    const double scale = std::min(0.25, 1200.0 / static_cast<double>(std::max(refGray.cols, refGray.rows)));
    cv::resize(refGray, refGray, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::resize(targetGray, targetGray, cv::Size(), scale, scale, cv::INTER_AREA);

    auto buildFeature = [](const cv::Mat& gray) {
        cv::Mat rowMean;
        cv::reduce(gray, rowMean, 1, cv::REDUCE_AVG, CV_32F);
        cv::Mat meanExpanded;
        cv::repeat(rowMean, 1, gray.cols, meanExpanded);
        cv::Mat centered = gray - meanExpanded;
        cv::Mat feature;
        cv::Sobel(centered, feature, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(feature, mean, stddev);
        feature -= mean[0];
        if (stddev[0] > 1e-6) {
            feature /= stddev[0];
        }
        return feature;
    };

    const cv::Mat refFeature = buildFeature(refGray);
    const cv::Mat targetFeature = buildFeature(targetGray);
    if (refFeature.cols < 160 || refFeature.rows < 120) {
        return estimate;
    }

    const int templateWidth = std::clamp(static_cast<int>(std::lround(refFeature.cols * 0.60)), 160, refFeature.cols - 60);
    const int templateHeight = std::clamp(static_cast<int>(std::lround(refFeature.rows * 0.70)), 120, refFeature.rows - 20);
    const int templateY0 = std::max(0, (targetFeature.rows - templateHeight) / 2);
    const int templateX0 = negativeDirection ? std::max(0, targetFeature.cols - templateWidth) : 0;
    const cv::Rect templateRect(templateX0, templateY0, templateWidth, templateHeight);
    const cv::Mat templ = targetFeature(templateRect);

    const double primaryPriorDs = primaryPriorShiftPx * scale;
    const double nominalPrimaryDs = nominalPrimaryShiftPx * scale;
    const double secondaryPriorDs = secondaryPriorShiftPx * scale;
    const double predictedAnchorX = negativeDirection
                                        ? templateX0 + primaryPriorDs
                                        : primaryPriorDs;
    const double nominalAnchorX = negativeDirection
                                      ? templateX0 + nominalPrimaryDs
                                      : nominalPrimaryDs;
    const double centerX = std::isfinite(predictedAnchorX)
                               ? predictedAnchorX
                               : nominalAnchorX;
    const double centerY = templateY0 + (std::isfinite(secondaryPriorDs) ? secondaryPriorDs : 0.0);
    const int halfWindowX =
        std::max(40, static_cast<int>(std::lround(scale *
                                                  std::max(220.0,
                                                           std::abs(primaryPriorShiftPx - nominalPrimaryShiftPx) + 140.0))));
    const int halfWindowY = std::max(18, static_cast<int>(std::lround(scale * 90.0)));
    const int searchX0 = std::clamp(static_cast<int>(std::lround(centerX)) - halfWindowX,
                                    0,
                                    refFeature.cols - templateWidth);
    const int searchX1 = std::clamp(static_cast<int>(std::lround(centerX)) + halfWindowX,
                                    searchX0,
                                    refFeature.cols - templateWidth);
    const int searchY0 = std::clamp(static_cast<int>(std::lround(centerY)) - halfWindowY,
                                    0,
                                    refFeature.rows - templateHeight);
    const int searchY1 = std::clamp(static_cast<int>(std::lround(centerY)) + halfWindowY,
                                    searchY0,
                                    refFeature.rows - templateHeight);
    const cv::Rect searchRect(searchX0,
                              searchY0,
                              std::max(1, searchX1 - searchX0 + templateWidth),
                              std::max(1, searchY1 - searchY0 + templateHeight));
    if (searchRect.width < templateWidth || searchRect.height < templateHeight) {
        return estimate;
    }

    cv::Mat matchScores;
    cv::matchTemplate(refFeature(searchRect), templ, matchScores, cv::TM_CCOEFF_NORMED);
    double maxScore = 0.0;
    cv::Point maxLoc;
    cv::minMaxLoc(matchScores, nullptr, &maxScore, nullptr, &maxLoc);
    if (!std::isfinite(maxScore) || maxScore < 0.10) {
        return estimate;
    }

    const int matchedX = searchRect.x + maxLoc.x;
    const int matchedY = searchRect.y + maxLoc.y;
    const double primaryShiftDs = negativeDirection ? matchedX - templateX0 : matchedX;
    const double secondaryShiftDs = matchedY - templateY0;
    const double primaryShiftPx = primaryShiftDs / scale;
    const double secondaryShiftPx = secondaryShiftDs / scale;

    estimate.ok = true;
    estimate.score = maxScore;
    if (primaryIsY) {
        estimate.dx = secondaryShiftPx;
        estimate.dy = primaryShiftPx;
    } else {
        estimate.dx = primaryShiftPx;
        estimate.dy = secondaryShiftPx;
    }
    return estimate;
}

TransformResult refineEndpointStepWithImageGuidedNormalSearchViews(const EdgeVariants& previousView,
                                                                   const EdgeVariants& nextView,
                                                                   const cv::Point2d& center,
                                                                   MotionPriorDirection stepDirection,
                                                                   const StitchPipelineConfig& config,
                                                                   const TransformResult& currentTransform,
                                                                   const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
                                                                   double targetNormalRmsePx,
                                                                   bool aggressiveSearch,
                                                                   bool relaxQualityGuards)
{
    if (!imageCorrelationEstimate.ok || !currentTransform.hasCandidate()) {
        return currentTransform;
    }
    const double currentNormal = preferredNormalRmse(currentTransform);
    const double currentCoverage = preferredCoverageRatio(currentTransform);
    const double currentCorrelation = preferredTangentCorrelation(currentTransform);
    const double currentCost =
        transformSelectionCost(currentTransform,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    const double currentContinuity =
        continuityDistanceToPrior(currentTransform,
                                  imageCorrelationEstimate.dx,
                                  imageCorrelationEstimate.dy);
    if (!std::isfinite(currentNormal) || !std::isfinite(currentCoverage)) {
        return currentTransform;
    }
    if (!aggressiveSearch &&
        currentNormal <= 0.12 &&
        currentCoverage >= 0.70 &&
        currentCorrelation >= 0.85 &&
        currentContinuity <= 3.0) {
        return currentTransform;
    }

    const bool primaryIsX = directionUsesPrimaryX(stepDirection);
    const auto primaryShiftOf = [&](double dx, double dy) {
        return primaryIsX ? dx : dy;
    };
    const auto perpShiftOf = [&](double dx, double dy) {
        return primaryIsX ? dy : dx;
    };
    const auto makeTransform = [&](double primaryShift, double perpShift, double angleDeg) {
        const double dx = primaryIsX ? primaryShift : perpShift;
        const double dy = primaryIsX ? perpShift : primaryShift;
        return buildEvaluatedPriorTransform(previousView,
                                           nextView,
                                           center,
                                           dx,
                                           dy,
                                           angleDeg,
                                           stepDirection,
                                           config.tangentResidualCostWeight,
                                           config.tangentCorrelationCostWeight,
                                           false);
    };

    TransformResult bestTransform = currentTransform;
    double bestNormal = currentNormal;
    double bestCoverage = currentCoverage;
    double bestCost = currentCost;
    const double coverageSlack = relaxQualityGuards ? 0.06 : 0.015;
    const double correlationSlack = relaxQualityGuards ? 0.03 : 0.01;
    const double continuitySlack = relaxQualityGuards ? 8.0 : 4.0;
    const double normalContinuityGuard = relaxQualityGuards ? 0.015 : 0.008;
    const double betterNormalMargin = relaxQualityGuards ? 0.001 : 0.0015;
    const double sameNormalMargin = relaxQualityGuards ? 0.004 : 0.0015;
    const double betterCoverageMargin = relaxQualityGuards ? 0.005 : 0.01;

    const auto considerCandidate = [&](double primaryShift,
                                       double perpShift,
                                       double angleDeg) {
        TransformResult candidate = makeTransform(primaryShift, perpShift, angleDeg);
        candidate.direction = motionDirectionLabel(stepDirection);
        const double candidateNormal = preferredNormalRmse(candidate);
        if (!candidate.hasCandidate() || !std::isfinite(candidateNormal)) {
            return;
        }

        const double candidateCoverage = preferredCoverageRatio(candidate);
        if (candidateCoverage + coverageSlack < currentCoverage) {
            return;
        }

        const double candidateCorrelation = preferredTangentCorrelation(candidate);
        if (candidateCorrelation + correlationSlack < currentCorrelation) {
            return;
        }

        const double candidateContinuity =
            continuityDistanceToPrior(candidate,
                                      imageCorrelationEstimate.dx,
                                      imageCorrelationEstimate.dy);
        if (candidateContinuity > std::max(currentContinuity + continuitySlack, 10.0) &&
            candidateNormal + normalContinuityGuard >= currentNormal) {
            return;
        }

        const double candidateCost =
            transformSelectionCost(candidate,
                                   imageCorrelationEstimate.dx,
                                   imageCorrelationEstimate.dy,
                                   targetNormalRmsePx);
        const bool betterNormal = candidateNormal < bestNormal - betterNormalMargin;
        const bool sameNormalButBetterCost =
            std::abs(candidateNormal - bestNormal) <= sameNormalMargin &&
            candidateCost < bestCost * 0.995;
        const bool sameNormalButBetterCoverage =
            std::abs(candidateNormal - bestNormal) <= (relaxQualityGuards ? 0.006 : 0.0025) &&
            candidateCoverage > bestCoverage + betterCoverageMargin &&
            candidateCost <= bestCost * 1.01;
        if (betterNormal || sameNormalButBetterCost || sameNormalButBetterCoverage) {
            bestTransform = std::move(candidate);
            bestNormal = candidateNormal;
            bestCoverage = candidateCoverage;
            bestCost = candidateCost;
        }
    };

    const double currentPrimary = primaryShiftOf(currentTransform.dx, currentTransform.dy);
    const double currentPerp = perpShiftOf(currentTransform.dx, currentTransform.dy);
    const double imagePrimary =
        primaryShiftOf(imageCorrelationEstimate.dx, imageCorrelationEstimate.dy);
    const double imagePerp =
        perpShiftOf(imageCorrelationEstimate.dx, imageCorrelationEstimate.dy);
    const double primaryGapPx = std::abs(imagePrimary - currentPrimary);
    const double perpGapPx = std::abs(imagePerp - currentPerp);
    const bool compactLocalSearch = !aggressiveSearch && currentContinuity <= 6.0;
    std::vector<double> seedBlendFactors = compactLocalSearch
                                               ? std::vector<double>{0.0, 1.0}
                                               : std::vector<double>{0.0, 1.0, 0.5};
    if (aggressiveSearch) {
        seedBlendFactors.push_back(0.25);
        seedBlendFactors.push_back(0.75);
    }

    std::vector<double> seedPrimary;
    std::vector<double> seedPerp;
    seedPrimary.reserve(seedBlendFactors.size());
    seedPerp.reserve(seedBlendFactors.size());
    for (const double blend : seedBlendFactors) {
        seedPrimary.push_back(currentPrimary * (1.0 - blend) + imagePrimary * blend);
        seedPerp.push_back(currentPerp * (1.0 - blend) + imagePerp * blend);
    }

    const double coarseAngleHalfRangeDeg = aggressiveSearch ? 0.06 : 0.03;
    const double coarseAngleStepDeg = aggressiveSearch ? 0.01 : 0.01;
    const double coarsePrimaryHalfRangePx =
        aggressiveSearch ? std::clamp(primaryGapPx + 3.0, 3.5, 6.0)
                         : (compactLocalSearch ? std::clamp(primaryGapPx + 0.5, 0.6, 1.4)
                                               : std::clamp(primaryGapPx + 0.8, 0.8, 2.0));
    const double coarsePerpHalfRangePx =
        aggressiveSearch ? std::clamp(perpGapPx + 2.0, 2.5, 4.0)
                         : (compactLocalSearch ? std::clamp(perpGapPx + 0.5, 0.8, 1.4)
                                               : std::clamp(perpGapPx + 0.8, 1.0, 2.0));
    const double coarsePrimaryStepPx = aggressiveSearch ? 0.75 : (compactLocalSearch ? 0.25 : 0.2);
    const double coarsePerpStepPx = aggressiveSearch ? 0.75 : (compactLocalSearch ? 0.25 : 0.2);

    for (std::size_t seedIndex = 0; seedIndex < seedPrimary.size(); ++seedIndex) {
        for (double angleOffsetDeg = -coarseAngleHalfRangeDeg;
             angleOffsetDeg <= coarseAngleHalfRangeDeg + 1e-9;
             angleOffsetDeg += coarseAngleStepDeg) {
            for (double primaryOffsetPx = -coarsePrimaryHalfRangePx;
                 primaryOffsetPx <= coarsePrimaryHalfRangePx + 1e-9;
                 primaryOffsetPx += coarsePrimaryStepPx) {
                for (double perpOffsetPx = -coarsePerpHalfRangePx;
                     perpOffsetPx <= coarsePerpHalfRangePx + 1e-9;
                     perpOffsetPx += coarsePerpStepPx) {
                    considerCandidate(seedPrimary[seedIndex] + primaryOffsetPx,
                                      seedPerp[seedIndex] + perpOffsetPx,
                                      currentTransform.da + angleOffsetDeg);
                }
            }
        }
    }

    const double refinedPrimary = primaryShiftOf(bestTransform.dx, bestTransform.dy);
    const double refinedPerp = perpShiftOf(bestTransform.dx, bestTransform.dy);
    const double fineAngleHalfRangeDeg = aggressiveSearch ? 0.02 : 0.01;
    const double finePrimaryHalfRangePx = aggressiveSearch ? 1.0 : (compactLocalSearch ? 0.3 : 0.5);
    const double finePerpHalfRangePx = aggressiveSearch ? 1.0 : (compactLocalSearch ? 0.3 : 0.5);
    for (double angleOffsetDeg = -fineAngleHalfRangeDeg;
         angleOffsetDeg <= fineAngleHalfRangeDeg + 1e-9;
         angleOffsetDeg += 0.005) {
        for (double primaryOffsetPx = -finePrimaryHalfRangePx;
             primaryOffsetPx <= finePrimaryHalfRangePx + 1e-9;
             primaryOffsetPx += 0.1) {
            for (double perpOffsetPx = -finePerpHalfRangePx;
                 perpOffsetPx <= finePerpHalfRangePx + 1e-9;
                 perpOffsetPx += 0.1) {
                considerCandidate(refinedPrimary + primaryOffsetPx,
                                  refinedPerp + perpOffsetPx,
                                  bestTransform.da + angleOffsetDeg);
            }
        }
    }

    const double ultraFineAngleHalfRangeDeg = aggressiveSearch ? 0.01 : 0.005;
    const double ultraFinePrimaryHalfRangePx = aggressiveSearch ? 0.35 : (compactLocalSearch ? 0.18 : 0.25);
    const double ultraFinePerpHalfRangePx = aggressiveSearch ? 0.35 : (compactLocalSearch ? 0.18 : 0.25);
    const double ultraFinePrimary = primaryShiftOf(bestTransform.dx, bestTransform.dy);
    const double ultraFinePerp = perpShiftOf(bestTransform.dx, bestTransform.dy);
    for (double angleOffsetDeg = -ultraFineAngleHalfRangeDeg;
         angleOffsetDeg <= ultraFineAngleHalfRangeDeg + 1e-9;
         angleOffsetDeg += 0.0025) {
        for (double primaryOffsetPx = -ultraFinePrimaryHalfRangePx;
             primaryOffsetPx <= ultraFinePrimaryHalfRangePx + 1e-9;
             primaryOffsetPx += 0.05) {
            for (double perpOffsetPx = -ultraFinePerpHalfRangePx;
                 perpOffsetPx <= ultraFinePerpHalfRangePx + 1e-9;
                 perpOffsetPx += 0.05) {
                considerCandidate(ultraFinePrimary + primaryOffsetPx,
                                  ultraFinePerp + perpOffsetPx,
                                  bestTransform.da + angleOffsetDeg);
            }
        }
    }

    const std::vector<AlignmentCandidateDiagnostic> candidateDiagnostics = bestTransform.candidateDiagnostics;
    TransformResult preciseBestTransform =
        buildEvaluatedPriorTransform(previousView,
                                     nextView,
                                     center,
                                     bestTransform.dx,
                                     bestTransform.dy,
                                     bestTransform.da,
                                     stepDirection,
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight,
                                     true);
    preciseBestTransform.direction = motionDirectionLabel(stepDirection);
    if (preciseBestTransform.candidateDiagnostics.empty()) {
        if (!candidateDiagnostics.empty()) {
            preciseBestTransform.candidateDiagnostics = candidateDiagnostics;
        } else if (!currentTransform.candidateDiagnostics.empty()) {
            preciseBestTransform.candidateDiagnostics = currentTransform.candidateDiagnostics;
        }
    }

    return preciseBestTransform.hasCandidate() ? preciseBestTransform : bestTransform;
}

TransformResult refineEndpointStepWithImageGuidedNormalSearch(const EdgeVariants& previousEdges,
                                                              const EdgeVariants& nextEdges,
                                                              const cv::Point2d& center,
                                                              MotionPriorDirection stepDirection,
                                                              const StitchPipelineConfig& config,
                                                              const TransformResult& currentTransform,
                                                              const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
                                                              double targetNormalRmsePx,
                                                              bool useUnfilteredEdges,
                                                              bool aggressiveSearch)
{
    const EdgeVariants previousView = buildMatchingEdgeView(previousEdges, useUnfilteredEdges);
    const EdgeVariants nextView = buildMatchingEdgeView(nextEdges, useUnfilteredEdges);
    return refineEndpointStepWithImageGuidedNormalSearchViews(previousView,
                                                              nextView,
                                                              center,
                                                              stepDirection,
                                                              config,
                                                              currentTransform,
                                                              imageCorrelationEstimate,
                                                              targetNormalRmsePx,
                                                              aggressiveSearch,
                                                              false);
}

TransformResult refineLastEndpointStepWithTailTrimSearch(const EdgeVariants& previousEdges,
                                                         const EdgeVariants& nextEdges,
                                                         const cv::Point2d& center,
                                                         MotionPriorDirection stepDirection,
                                                         const StitchPipelineConfig& config,
                                                         const TransformResult& currentTransform,
                                                         const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
                                                         double targetNormalRmsePx,
                                                         bool useUnfilteredEdges)
{
    if (!imageCorrelationEstimate.ok || !currentTransform.hasCandidate()) {
        return currentTransform;
    }

    const EdgeVariants fullPreviousView = buildMatchingEdgeView(previousEdges, useUnfilteredEdges);
    const EdgeVariants fullNextView = buildMatchingEdgeView(nextEdges, useUnfilteredEdges);
    TransformResult bestTransform =
        buildEvaluatedPriorTransform(fullPreviousView,
                                     fullNextView,
                                     center,
                                     currentTransform.dx,
                                     currentTransform.dy,
                                     currentTransform.da,
                                     stepDirection,
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight);
    bestTransform.direction = motionDirectionLabel(stepDirection);
    if (bestTransform.candidateDiagnostics.empty() &&
        !currentTransform.candidateDiagnostics.empty()) {
        bestTransform.candidateDiagnostics = currentTransform.candidateDiagnostics;
    }

    double bestNormal = preferredNormalRmse(bestTransform);
    double bestCoverage = preferredCoverageRatio(bestTransform);
    double bestCorrelation = preferredTangentCorrelation(bestTransform);
    double bestCost =
        transformSelectionCost(bestTransform,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    double bestContinuity =
        continuityDistanceToPrior(bestTransform,
                                  imageCorrelationEstimate.dx,
                                  imageCorrelationEstimate.dy);
    if (!std::isfinite(bestNormal)) {
        return currentTransform;
    }

    const std::vector<double> trimTailFractions =
        config.endpointProbeFastMode ? std::vector<double>{0.12, 0.20}
                                     : std::vector<double>{0.08, 0.12, 0.16, 0.20, 0.24};
    for (const double trimTailFraction : trimTailFractions) {
        const EdgeVariants trimmedPreviousView =
            buildTrimmedMatchingEdgeView(previousEdges,
                                         useUnfilteredEdges,
                                         reversePrimaryDirection(stepDirection),
                                         trimTailFraction);
        const EdgeVariants trimmedNextView =
            buildTrimmedMatchingEdgeView(nextEdges, useUnfilteredEdges, stepDirection, trimTailFraction);
        if (trimmedPreviousView.raw.size() + 64 >= fullPreviousView.raw.size() ||
            trimmedNextView.raw.size() + 64 >= fullNextView.raw.size()) {
            continue;
        }

        TransformResult trimmedSeed =
            buildEvaluatedPriorTransform(trimmedPreviousView,
                                         trimmedNextView,
                                         center,
                                         currentTransform.dx,
                                         currentTransform.dy,
                                         currentTransform.da,
                                         stepDirection,
                                         config.tangentResidualCostWeight,
                                         config.tangentCorrelationCostWeight);
        trimmedSeed.direction = motionDirectionLabel(stepDirection);
        TransformResult trimmedCandidate =
            refineEndpointStepWithImageGuidedNormalSearchViews(trimmedPreviousView,
                                                               trimmedNextView,
                                                               center,
                                                               stepDirection,
                                                               config,
                                                               trimmedSeed,
                                                               imageCorrelationEstimate,
                                                               targetNormalRmsePx,
                                                               true,
                                                               true);
        if (!trimmedCandidate.hasCandidate()) {
            continue;
        }

        TransformResult fullCandidateSeed =
            buildEvaluatedPriorTransform(fullPreviousView,
                                         fullNextView,
                                         center,
                                         trimmedCandidate.dx,
                                         trimmedCandidate.dy,
                                         trimmedCandidate.da,
                                         stepDirection,
                                         config.tangentResidualCostWeight,
                                         config.tangentCorrelationCostWeight);
        fullCandidateSeed.direction = motionDirectionLabel(stepDirection);
        TransformResult fullCandidate =
            refineEndpointStepWithImageGuidedNormalSearchViews(fullPreviousView,
                                                               fullNextView,
                                                               center,
                                                               stepDirection,
                                                               config,
                                                               fullCandidateSeed,
                                                               imageCorrelationEstimate,
                                                               targetNormalRmsePx,
                                                               true,
                                                               true);
        const double candidateNormal = preferredNormalRmse(fullCandidate);
        if (!fullCandidate.hasCandidate() || !std::isfinite(candidateNormal)) {
            continue;
        }

        const double candidateCoverage = preferredCoverageRatio(fullCandidate);
        const double candidateCorrelation = preferredTangentCorrelation(fullCandidate);
        const double candidateCost =
            transformSelectionCost(fullCandidate,
                                   imageCorrelationEstimate.dx,
                                   imageCorrelationEstimate.dy,
                                   targetNormalRmsePx);
        const double candidateContinuity =
            continuityDistanceToPrior(fullCandidate,
                                      imageCorrelationEstimate.dx,
                                      imageCorrelationEstimate.dy);
        const bool coverageGuardPass =
            candidateCoverage >= 0.66 ||
            candidateCoverage + 0.05 >= bestCoverage;
        if (!coverageGuardPass || candidateCorrelation + 0.02 < bestCorrelation) {
            continue;
        }
        if (candidateContinuity > std::max(bestContinuity + 5.0, 12.0) &&
            candidateNormal + 0.004 >= bestNormal) {
            continue;
        }

        const bool strongNormalImprovement = candidateNormal + 0.004 < bestNormal;
        const bool sameNormalButBetterCost =
            std::abs(candidateNormal - bestNormal) <= 0.002 &&
            candidateCost < bestCost * 0.992;
        const bool pushedBelowTarget =
            candidateNormal < 0.1 &&
            candidateNormal + 0.0015 < bestNormal;
        if (strongNormalImprovement || sameNormalButBetterCost || pushedBelowTarget) {
            bestTransform = std::move(fullCandidate);
            bestNormal = candidateNormal;
            bestCoverage = candidateCoverage;
            bestCorrelation = candidateCorrelation;
            bestCost = candidateCost;
            bestContinuity = candidateContinuity;
        }
    }

    if (bestTransform.candidateDiagnostics.empty() &&
        !currentTransform.candidateDiagnostics.empty()) {
        bestTransform.candidateDiagnostics = currentTransform.candidateDiagnostics;
    }
    return bestTransform;
}

TransformResult refineLastEndpointStepWithReferencePriorHalfOverlapProbe(
    const EdgeVariants& previousEdges,
    const EdgeVariants& nextEdges,
    const cv::Point2d& center,
    MotionPriorDirection stepDirection,
    const StitchPipelineConfig& config,
    const TransformResult& referencePriorTransform,
    double& outSearchRangeX,
    double& outSearchRangeY,
    double* outSelectedTrimFraction = nullptr)
{
    outSearchRangeX = 0.0;
    outSearchRangeY = 0.0;
    if (outSelectedTrimFraction != nullptr) {
        *outSelectedTrimFraction = 0.0;
    }
    const EdgeVariants fullPreviousView = buildMatchingEdgeView(previousEdges, false);
    const EdgeVariants fullNextView = buildMatchingEdgeView(nextEdges, false);
    if (!referencePriorTransform.hasCandidate()) {
        return buildEvaluatedPriorTransform(fullPreviousView,
                                            fullNextView,
                                            center,
                                            referencePriorTransform.dx,
                                            referencePriorTransform.dy,
                                            referencePriorTransform.da,
                                            stepDirection,
                                            config.tangentResidualCostWeight,
                                            config.tangentCorrelationCostWeight);
    }

    constexpr double kExpandedBaseSearchRangePx = 600.0;
    constexpr double kExpandedPrimaryHalfRangePx = 120.0;
    constexpr double kExpandedPerpHalfRangePx = 100.0;
    constexpr double kRotationHalfRangeDeg = 0.20;

    TransformResult evaluatedPrior =
        buildEvaluatedPriorTransform(fullPreviousView,
                                     fullNextView,
                                     center,
                                     referencePriorTransform.dx,
                                     referencePriorTransform.dy,
                                     referencePriorTransform.da,
                                     stepDirection,
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight);
    evaluatedPrior.direction = motionDirectionLabel(stepDirection);

    double rotationMinDeg =
        std::max(config.rotationSearchMinDeg, referencePriorTransform.da - kRotationHalfRangeDeg);
    double rotationMaxDeg =
        std::min(config.rotationSearchMaxDeg, referencePriorTransform.da + kRotationHalfRangeDeg);
    if (rotationMinDeg > rotationMaxDeg) {
        rotationMinDeg = config.rotationSearchMinDeg;
        rotationMaxDeg = config.rotationSearchMaxDeg;
    }

    const std::vector<double> trimFractions =
        config.endpointProbeFastMode
            ? std::vector<double>{0.32, 0.40, 0.48, 0.56, 0.64}
            : std::vector<double>{0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64};

    TransformResult bestTrimmedTransform;
    double bestTrimmedObjective = std::numeric_limits<double>::infinity();
    double bestTrimmedNormal = std::numeric_limits<double>::infinity();
    double bestTrimmedCoverage = 0.0;
    double bestTrimmedCorrelation = 0.0;
    double bestTrimmedSearchRangeX = 0.0;
    double bestTrimmedSearchRangeY = 0.0;
    double bestTrimFraction = 0.0;

    for (const double trimFraction : trimFractions) {
        const EdgeVariants trimmedPreviousView =
            buildTrimmedMatchingEdgeView(previousEdges,
                                         false,
                                         reversePrimaryDirection(stepDirection),
                                         trimFraction);
        const EdgeVariants trimmedNextView =
            buildTrimmedMatchingEdgeView(nextEdges, false, stepDirection, trimFraction);
        if (trimmedPreviousView.raw.size() + 64 >= fullPreviousView.raw.size() ||
            trimmedNextView.raw.size() + 64 >= fullNextView.raw.size()) {
            continue;
        }

        double localSearchRangeX = 0.0;
        double localSearchRangeY = 0.0;
        TransformResult trimmedTransform =
            matchOnePair(trimmedPreviousView,
                         trimmedNextView,
                         center,
                         referencePriorTransform.dx,
                         referencePriorTransform.dy,
                         referencePriorTransform.da,
                         true,
                         std::max(config.baseSearchRange, kExpandedBaseSearchRangePx),
                         stepDirection,
                         rotationMinDeg,
                         rotationMaxDeg,
                         std::min(config.rotationSearchStepDeg, 0.01),
                         config.tangentResidualCostWeight,
                         config.tangentCorrelationCostWeight,
                         true,
                         kExpandedPrimaryHalfRangePx,
                         kExpandedPerpHalfRangePx,
                         localSearchRangeX,
                         localSearchRangeY);
        if (!trimmedTransform.hasCandidate()) {
            continue;
        }

        trimmedTransform.direction = motionDirectionLabel(stepDirection);
        const double trimmedNormal = preferredNormalRmse(trimmedTransform);
        const double trimmedCoverage = preferredCoverageRatio(trimmedTransform);
        const double trimmedCorrelation = preferredTangentCorrelation(trimmedTransform);
        const int supportCount =
            std::max(trimmedTransform.metrics.inlierCount, trimmedTransform.metrics.overlapCount);
        if (!std::isfinite(trimmedNormal) ||
            trimmedCoverage < 0.16 ||
            supportCount < 280) {
            continue;
        }

        const double coveragePenalty =
            trimmedCoverage < 0.22 ? (0.22 - trimmedCoverage) * 0.10 : 0.0;
        const double supportPenalty =
            supportCount < 360
                ? (static_cast<double>(360 - supportCount) / 360.0) * 0.012
                : 0.0;
        const double trimmedObjective = trimmedNormal + coveragePenalty + supportPenalty;
        const bool betterObjective = trimmedObjective + 0.002 < bestTrimmedObjective;
        const bool strongerNormalWithComparableCoverage =
            trimmedNormal + 0.006 < bestTrimmedNormal &&
            trimmedCoverage + 0.03 >= bestTrimmedCoverage;
        const bool sameObjectiveButBetterCoverage =
            std::abs(trimmedObjective - bestTrimmedObjective) <= 0.002 &&
            trimmedCoverage > bestTrimmedCoverage + 0.015;
        const bool sameObjectiveButBetterCorrelation =
            std::abs(trimmedObjective - bestTrimmedObjective) <= 0.001 &&
            trimmedCorrelation > bestTrimmedCorrelation + 0.002;
        if (betterObjective ||
            strongerNormalWithComparableCoverage ||
            sameObjectiveButBetterCoverage ||
            sameObjectiveButBetterCorrelation) {
            bestTrimmedTransform = std::move(trimmedTransform);
            bestTrimmedObjective = trimmedObjective;
            bestTrimmedNormal = trimmedNormal;
            bestTrimmedCoverage = trimmedCoverage;
            bestTrimmedCorrelation = trimmedCorrelation;
            bestTrimmedSearchRangeX = localSearchRangeX;
            bestTrimmedSearchRangeY = localSearchRangeY;
            bestTrimFraction = trimFraction;
        }
    }

    if (!bestTrimmedTransform.hasCandidate()) {
        return evaluatedPrior;
    }

    outSearchRangeX = bestTrimmedSearchRangeX;
    outSearchRangeY = bestTrimmedSearchRangeY;
    if (outSelectedTrimFraction != nullptr) {
        *outSelectedTrimFraction = bestTrimFraction;
    }
    return bestTrimmedTransform;
}

TransformResult refineLastEndpointStepWithReferencePriorHalfOverlapSoftPullback(
    const EdgeVariants& previousEdges,
    const EdgeVariants& nextEdges,
    const cv::Point2d& center,
    MotionPriorDirection stepDirection,
    const StitchPipelineConfig& config,
    const TransformResult& localRegionAnchor,
    const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
    double targetNormalRmsePx,
    TransformResult* outFullAnchorTransform = nullptr,
    TransformResult* outPullbackCandidateTransform = nullptr)
{
    if (!imageCorrelationEstimate.ok || !localRegionAnchor.hasCandidate()) {
        return localRegionAnchor;
    }

    const EdgeVariants fullPreviousView = buildMatchingEdgeView(previousEdges, false);
    const EdgeVariants fullNextView = buildMatchingEdgeView(nextEdges, false);
    TransformResult fullAnchor =
        buildEvaluatedPriorTransform(fullPreviousView,
                                     fullNextView,
                                     center,
                                     localRegionAnchor.dx,
                                     localRegionAnchor.dy,
                                     localRegionAnchor.da,
                                     stepDirection,
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight);
    fullAnchor.direction = motionDirectionLabel(stepDirection);
    if (outFullAnchorTransform != nullptr) {
        *outFullAnchorTransform = fullAnchor;
    }
    if (!fullAnchor.hasCandidate()) {
        return localRegionAnchor;
    }

    TransformResult pullbackCandidate =
        refineEndpointStepWithImageGuidedNormalSearchViews(fullPreviousView,
                                                           fullNextView,
                                                           center,
                                                           stepDirection,
                                                           config,
                                                           fullAnchor,
                                                           imageCorrelationEstimate,
                                                           targetNormalRmsePx,
                                                           true,
                                                           true);
    pullbackCandidate.direction = motionDirectionLabel(stepDirection);
    if (outPullbackCandidateTransform != nullptr) {
        *outPullbackCandidateTransform = pullbackCandidate;
    }
    if (!pullbackCandidate.hasCandidate()) {
        return localRegionAnchor;
    }

    const bool primaryIsX = directionUsesPrimaryX(stepDirection);
    const auto primaryShiftOf = [&](const TransformResult& transform) {
        return primaryIsX ? transform.dx : transform.dy;
    };
    const auto perpShiftOf = [&](const TransformResult& transform) {
        return primaryIsX ? transform.dy : transform.dx;
    };

    const double anchorNormal = preferredNormalRmse(fullAnchor);
    const double anchorCoverage = preferredCoverageRatio(fullAnchor);
    const double anchorCorrelation = preferredTangentCorrelation(fullAnchor);
    const double anchorCost =
        transformSelectionCost(fullAnchor,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    const double candidateNormal = preferredNormalRmse(pullbackCandidate);
    const double candidateCoverage = preferredCoverageRatio(pullbackCandidate);
    const double candidateCorrelation = preferredTangentCorrelation(pullbackCandidate);
    const double candidateCost =
        transformSelectionCost(pullbackCandidate,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    const double primaryDriftPx =
        std::abs(primaryShiftOf(pullbackCandidate) - primaryShiftOf(localRegionAnchor));
    const double perpDriftPx =
        std::abs(perpShiftOf(pullbackCandidate) - perpShiftOf(localRegionAnchor));
    const double angleDriftDeg =
        std::abs(pullbackCandidate.da - localRegionAnchor.da);

    const bool staysCloseToLocalAnchor =
        primaryDriftPx <= 1.5 &&
        perpDriftPx <= 1.25 &&
        angleDriftDeg <= 0.03;
    const bool preservesFullViewQuality =
        std::isfinite(candidateNormal) &&
        candidateCorrelation + 0.01 >= anchorCorrelation &&
        (candidateCoverage + 0.03 >= anchorCoverage ||
         candidateNormal + 0.015 < anchorNormal);
    const bool meaningfullyImprovesFullView =
        candidateNormal + 0.002 < anchorNormal ||
        candidateCost < anchorCost * 0.985 ||
        (candidateNormal <= anchorNormal + 0.002 &&
         candidateCoverage > anchorCoverage + 0.01);
    if (staysCloseToLocalAnchor &&
        preservesFullViewQuality &&
        meaningfullyImprovesFullView) {
        return pullbackCandidate;
    }

    return localRegionAnchor;
}

TransformResult refineLastEndpointStepWithDirectNormalGridSearch(const EdgeVariants& previousEdges,
                                                                 const EdgeVariants& nextEdges,
                                                                 const cv::Point2d& center,
                                                                 MotionPriorDirection stepDirection,
                                                                 const StitchPipelineConfig& config,
                                                                 const TransformResult& currentTransform,
                                                                 const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
                                                                 double targetNormalRmsePx,
                                                                 bool useUnfilteredEdges)
{
    if (!currentTransform.hasCandidate()) {
        return currentTransform;
    }

    const EdgeVariants previousView = buildMatchingEdgeView(previousEdges, useUnfilteredEdges);
    const EdgeVariants nextView = buildMatchingEdgeView(nextEdges, useUnfilteredEdges);
    const auto evaluateCandidate = [&](double dx, double dy, double angleDeg) {
        TransformResult candidate =
            buildEvaluatedPriorTransform(previousView,
                                         nextView,
                                         center,
                                         dx,
                                         dy,
                                         angleDeg,
                                         stepDirection,
                                         config.tangentResidualCostWeight,
                                         config.tangentCorrelationCostWeight);
        candidate.direction = motionDirectionLabel(stepDirection);
        if (candidate.candidateDiagnostics.empty() &&
            !currentTransform.candidateDiagnostics.empty()) {
            candidate.candidateDiagnostics = currentTransform.candidateDiagnostics;
        }
        return candidate;
    };

    TransformResult bestTransform =
        evaluateCandidate(currentTransform.dx, currentTransform.dy, currentTransform.da);
    double bestNormal = preferredNormalRmse(bestTransform);
    double bestCoverage = preferredCoverageRatio(bestTransform);
    double bestCorrelation = preferredTangentCorrelation(bestTransform);
    double bestCost =
        transformSelectionCost(bestTransform,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    double bestContinuity =
        continuityDistanceToPrior(bestTransform,
                                  imageCorrelationEstimate.dx,
                                  imageCorrelationEstimate.dy);
    if (!std::isfinite(bestNormal)) {
        return currentTransform;
    }

    const auto considerCandidate = [&](double dx, double dy, double angleDeg) {
        TransformResult candidate = evaluateCandidate(dx, dy, angleDeg);
        const double candidateNormal = preferredNormalRmse(candidate);
        if (!candidate.hasCandidate() || !std::isfinite(candidateNormal)) {
            return;
        }

        const double candidateCoverage = preferredCoverageRatio(candidate);
        const double candidateCorrelation = preferredTangentCorrelation(candidate);
        if ((candidateCoverage + 0.06 < bestCoverage && candidateCoverage < 0.66) ||
            candidateCorrelation + 0.03 < bestCorrelation) {
            return;
        }

        const double candidateContinuity =
            continuityDistanceToPrior(candidate,
                                      imageCorrelationEstimate.dx,
                                      imageCorrelationEstimate.dy);
        if (candidateContinuity > std::max(bestContinuity + 8.0, 12.0) &&
            candidateNormal + 0.003 >= bestNormal) {
            return;
        }

        const double candidateCost =
            transformSelectionCost(candidate,
                                   imageCorrelationEstimate.dx,
                                   imageCorrelationEstimate.dy,
                                   targetNormalRmsePx);
        const bool betterNormal = candidateNormal < bestNormal - 0.0005;
        const bool sameNormalButBetterCost =
            std::abs(candidateNormal - bestNormal) <= 0.002 &&
            candidateCost < bestCost * 0.995;
        const bool sameNormalButBetterCoverage =
            std::abs(candidateNormal - bestNormal) <= 0.003 &&
            candidateCoverage > bestCoverage + 0.005 &&
            candidateCost <= bestCost * 1.01;
        if (betterNormal || sameNormalButBetterCost || sameNormalButBetterCoverage) {
            bestTransform = std::move(candidate);
            bestNormal = candidateNormal;
            bestCoverage = candidateCoverage;
            bestCorrelation = candidateCorrelation;
            bestCost = candidateCost;
            bestContinuity = candidateContinuity;
        }
    };

    const bool primaryIsX = directionUsesPrimaryX(stepDirection);
    const auto currentPrimary = primaryIsX ? currentTransform.dx : currentTransform.dy;
    const auto currentPerp = primaryIsX ? currentTransform.dy : currentTransform.dx;
    const auto imagePrimary = primaryIsX ? imageCorrelationEstimate.dx : imageCorrelationEstimate.dy;
    const auto imagePerp = primaryIsX ? imageCorrelationEstimate.dy : imageCorrelationEstimate.dx;
    const std::array<std::pair<double, double>, 3> coarseSeeds{{
        {currentPrimary, currentPerp},
        {imagePrimary, imagePerp},
        {0.5 * (currentPrimary + imagePrimary), 0.5 * (currentPerp + imagePerp)}
    }};

    for (const auto& coarseSeed : coarseSeeds) {
        for (double angleOffsetDeg = -0.04; angleOffsetDeg <= 0.0400001; angleOffsetDeg += 0.005) {
            for (double primaryOffsetPx = -4.0; primaryOffsetPx <= 4.0001; primaryOffsetPx += 0.5) {
                for (double perpOffsetPx = -4.0; perpOffsetPx <= 4.0001; perpOffsetPx += 0.5) {
                    const double primaryShift = coarseSeed.first + primaryOffsetPx;
                    const double perpShift = coarseSeed.second + perpOffsetPx;
                    const double dx = primaryIsX ? primaryShift : perpShift;
                    const double dy = primaryIsX ? perpShift : primaryShift;
                    considerCandidate(dx, dy, currentTransform.da + angleOffsetDeg);
                }
            }
        }
    }

    const double refinedPrimary = primaryIsX ? bestTransform.dx : bestTransform.dy;
    const double refinedPerp = primaryIsX ? bestTransform.dy : bestTransform.dx;
    for (double angleOffsetDeg = -0.015; angleOffsetDeg <= 0.0150001; angleOffsetDeg += 0.0025) {
        for (double primaryOffsetPx = -1.5; primaryOffsetPx <= 1.5001; primaryOffsetPx += 0.1) {
            for (double perpOffsetPx = -1.5; perpOffsetPx <= 1.5001; perpOffsetPx += 0.1) {
                const double primaryShift = refinedPrimary + primaryOffsetPx;
                const double perpShift = refinedPerp + perpOffsetPx;
                const double dx = primaryIsX ? primaryShift : perpShift;
                const double dy = primaryIsX ? perpShift : primaryShift;
                considerCandidate(dx, dy, bestTransform.da + angleOffsetDeg);
            }
        }
    }

    return bestTransform;
}

TransformResult refineLastEndpointStepWithAffineShearSearch(const EdgeVariants& previousEdges,
                                                            const EdgeVariants& nextEdges,
                                                            const cv::Point2d& center,
                                                            MotionPriorDirection stepDirection,
                                                            const StitchPipelineConfig& config,
                                                            const TransformResult& currentTransform,
                                                            const ImageCorrelationShiftEstimate& imageCorrelationEstimate,
                                                            double targetNormalRmsePx,
                                                            bool useUnfilteredEdges)
{
    if (!currentTransform.hasCandidate()) {
        return currentTransform;
    }

    const EdgeVariants previousView = buildMatchingEdgeView(previousEdges, useUnfilteredEdges);
    const EdgeVariants nextView = buildMatchingEdgeView(nextEdges, useUnfilteredEdges);
    const auto evaluateCandidate =
        [&](double dx, double dy, double angleDeg, double shearPrimaryToNormal) {
            TransformResult baseTransform = currentTransform;
            baseTransform.dx = dx;
            baseTransform.dy = dy;
            baseTransform.da = angleDeg;
            baseTransform.shearPrimaryToNormal = shearPrimaryToNormal;
            baseTransform.hasCustomRelativeMatrix = false;
            baseTransform.customRelativeMatrix.release();
            cv::Mat rotation2x3 = cv::getRotationMatrix2D(center, angleDeg, 1.0);
            cv::Mat rigid = cv::Mat::eye(3, 3, CV_64F);
            rotation2x3.copyTo(rigid(cv::Rect(0, 0, 3, 2)));
            rigid.at<double>(0, 2) += dx;
            rigid.at<double>(1, 2) += dy;
            const cv::Mat shear =
                buildPrimaryToNormalShearMatrix(center, stepDirection, shearPrimaryToNormal);
            const cv::Mat relative = rigid * shear;
            return buildEvaluatedCustomMatrixTransform(previousView,
                                                       nextView,
                                                       relative,
                                                       baseTransform,
                                                       stepDirection,
                                                       config.tangentResidualCostWeight,
                                                       config.tangentCorrelationCostWeight);
        };

    TransformResult bestTransform =
        evaluateCandidate(currentTransform.dx,
                          currentTransform.dy,
                          currentTransform.da,
                          currentTransform.shearPrimaryToNormal);
    double bestNormal = preferredNormalRmse(bestTransform);
    double bestCoverage = preferredCoverageRatio(bestTransform);
    double bestCorrelation = preferredTangentCorrelation(bestTransform);
    double bestCost =
        transformSelectionCost(bestTransform,
                               imageCorrelationEstimate.dx,
                               imageCorrelationEstimate.dy,
                               targetNormalRmsePx);
    double bestContinuity =
        continuityDistanceToPrior(bestTransform,
                                  imageCorrelationEstimate.dx,
                                  imageCorrelationEstimate.dy);
    if (!std::isfinite(bestNormal)) {
        return currentTransform;
    }

    const auto considerCandidate =
        [&](double dx, double dy, double angleDeg, double shearPrimaryToNormal) {
            TransformResult candidate =
                evaluateCandidate(dx, dy, angleDeg, shearPrimaryToNormal);
            const double candidateNormal = preferredNormalRmse(candidate);
            if (!candidate.hasCandidate() || !std::isfinite(candidateNormal)) {
                return;
            }

            const double candidateCoverage = preferredCoverageRatio(candidate);
            const double candidateCorrelation = preferredTangentCorrelation(candidate);
            if ((candidateCoverage + 0.06 < bestCoverage && candidateCoverage < 0.66) ||
                candidateCorrelation + 0.03 < bestCorrelation) {
                return;
            }

            const double candidateContinuity =
                continuityDistanceToPrior(candidate,
                                          imageCorrelationEstimate.dx,
                                          imageCorrelationEstimate.dy);
            if (candidateContinuity > std::max(bestContinuity + 8.0, 12.0) &&
                candidateNormal + 0.003 >= bestNormal) {
                return;
            }

            const double candidateCost =
                transformSelectionCost(candidate,
                                       imageCorrelationEstimate.dx,
                                       imageCorrelationEstimate.dy,
                                       targetNormalRmsePx);
            const bool betterNormal = candidateNormal < bestNormal - 0.0005;
            const bool sameNormalButBetterCost =
                std::abs(candidateNormal - bestNormal) <= 0.002 &&
                candidateCost < bestCost * 0.995;
            const bool sameNormalButBetterCoverage =
                std::abs(candidateNormal - bestNormal) <= 0.003 &&
                candidateCoverage > bestCoverage + 0.005 &&
                candidateCost <= bestCost * 1.01;
            if (betterNormal || sameNormalButBetterCost || sameNormalButBetterCoverage) {
                bestTransform = std::move(candidate);
                bestNormal = candidateNormal;
                bestCoverage = candidateCoverage;
                bestCorrelation = candidateCorrelation;
                bestCost = candidateCost;
                bestContinuity = candidateContinuity;
            }
        };

    const bool primaryIsX = directionUsesPrimaryX(stepDirection);
    const double currentPrimary = primaryIsX ? currentTransform.dx : currentTransform.dy;
    const double currentPerp = primaryIsX ? currentTransform.dy : currentTransform.dx;
    const double imagePrimary = primaryIsX ? imageCorrelationEstimate.dx : imageCorrelationEstimate.dy;
    const double imagePerp = primaryIsX ? imageCorrelationEstimate.dy : imageCorrelationEstimate.dx;
    const std::array<std::pair<double, double>, 3> coarseSeeds{{
        {currentPrimary, currentPerp},
        {imagePrimary, imagePerp},
        {0.5 * (currentPrimary + imagePrimary), 0.5 * (currentPerp + imagePerp)}
    }};

    for (const auto& coarseSeed : coarseSeeds) {
        for (double shear = -0.00030; shear <= 0.0003001; shear += 0.00005) {
            for (double angleOffsetDeg = -0.01; angleOffsetDeg <= 0.0100001; angleOffsetDeg += 0.0025) {
                for (double primaryOffsetPx = -0.75; primaryOffsetPx <= 0.7501; primaryOffsetPx += 0.25) {
                    for (double perpOffsetPx = -0.75; perpOffsetPx <= 0.7501; perpOffsetPx += 0.25) {
                        const double primaryShift = coarseSeed.first + primaryOffsetPx;
                        const double perpShift = coarseSeed.second + perpOffsetPx;
                        const double dx = primaryIsX ? primaryShift : perpShift;
                        const double dy = primaryIsX ? perpShift : primaryShift;
                        considerCandidate(dx, dy, currentTransform.da + angleOffsetDeg, shear);
                    }
                }
            }
        }
    }

    const double refinedPrimary = primaryIsX ? bestTransform.dx : bestTransform.dy;
    const double refinedPerp = primaryIsX ? bestTransform.dy : bestTransform.dx;
    for (double shear = bestTransform.shearPrimaryToNormal - 0.00008;
         shear <= bestTransform.shearPrimaryToNormal + 0.0000801;
         shear += 0.00001) {
        for (double angleOffsetDeg = -0.004; angleOffsetDeg <= 0.0040001; angleOffsetDeg += 0.001) {
            for (double primaryOffsetPx = -0.4; primaryOffsetPx <= 0.4001; primaryOffsetPx += 0.1) {
                for (double perpOffsetPx = -0.4; perpOffsetPx <= 0.4001; perpOffsetPx += 0.1) {
                    const double primaryShift = refinedPrimary + primaryOffsetPx;
                    const double perpShift = refinedPerp + perpOffsetPx;
                    const double dx = primaryIsX ? primaryShift : perpShift;
                    const double dy = primaryIsX ? perpShift : primaryShift;
                    considerCandidate(dx, dy, bestTransform.da + angleOffsetDeg, shear);
                }
            }
        }
    }

    return bestTransform;
}

bool reselectBadStepCandidate(TransformResult& transform,
                              double approxShiftX,
                              double approxShiftY,
                              double targetNormalRmsePx)
{
    if (transform.candidateDiagnostics.empty()) {
        return false;
    }

    const double currentNormal = preferredNormalRmse(transform);
    const double currentCoverage = preferredCoverageRatio(transform);
    const double currentCorrelation = preferredTangentCorrelation(transform);
    const double currentCost = transformSelectionCost(transform, approxShiftX, approxShiftY, targetNormalRmsePx);

    double bestCost = std::numeric_limits<double>::infinity();
    double bestNormal = currentNormal;
    double bestCoverage = currentCoverage;
    const AlignmentCandidateDiagnostic* best = nullptr;
    for (const AlignmentCandidateDiagnostic& candidate : transform.candidateDiagnostics) {
        const double normalRmse = candidatePreferredNormalRmse(candidate);
        if (!std::isfinite(normalRmse)) {
            continue;
        }
        if (!(normalRmse + 0.015 < currentNormal || normalRmse <= currentNormal * 0.82)) {
            continue;
        }
        const double coverage = candidatePreferredCoverageRatio(candidate);
        if (coverage + 0.04 < currentCoverage) {
            continue;
        }
        const double tangentCorrelation = candidatePreferredTangentCorrelation(candidate);
        if (tangentCorrelation + 0.01 < currentCorrelation) {
            continue;
        }

        const double cost = candidateSelectionCost(candidate, approxShiftX, approxShiftY, targetNormalRmsePx);
        const bool betterNormal = normalRmse < bestNormal - 0.005;
        const bool sameNormalButBetterCost =
            std::abs(normalRmse - bestNormal) <= 0.005 && cost < bestCost;
        if (betterNormal || sameNormalButBetterCost) {
            bestCost = cost;
            best = &candidate;
            bestNormal = normalRmse;
            bestCoverage = coverage;
        }
    }

    if (best == nullptr) {
        return false;
    }

    const bool strongNormalImprovement =
        bestNormal + 0.03 < currentNormal && bestCoverage + 0.03 >= currentCoverage;
    if (!strongNormalImprovement && bestCost >= currentCost * 0.96) {
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

cv::Point2d transformPointLocal(const cv::Mat& transform, const cv::Point2d& point)
{
    const double x = transform.at<double>(0, 0) * point.x +
                     transform.at<double>(0, 1) * point.y +
                     transform.at<double>(0, 2);
    const double y = transform.at<double>(1, 0) * point.x +
                     transform.at<double>(1, 1) * point.y +
                     transform.at<double>(1, 2);
    const double w = transform.rows > 2
                         ? transform.at<double>(2, 0) * point.x +
                               transform.at<double>(2, 1) * point.y +
                               transform.at<double>(2, 2)
                         : 1.0;
    if (std::abs(w) < 1e-12) {
        return {x, y};
    }
    return {x / w, y / w};
}

cv::Mat buildRelativeMatrix(const cv::Point2d& center, const TransformResult& transform)
{
    if (transform.hasCustomRelativeMatrix &&
        !transform.customRelativeMatrix.empty() &&
        transform.customRelativeMatrix.rows == 3 &&
        transform.customRelativeMatrix.cols == 3) {
        return transform.customRelativeMatrix.clone();
    }

    cv::Mat rotation2x3 = cv::getRotationMatrix2D(center, transform.da, 1.0);
    cv::Mat relative = cv::Mat::eye(3, 3, CV_64F);
    rotation2x3.copyTo(relative(cv::Rect(0, 0, 3, 2)));
    relative.at<double>(0, 2) += transform.dx;
    relative.at<double>(1, 2) += transform.dy;
    return relative;
}

TransformResult extractRelativeTransform(const cv::Mat& fromGlobal,
                                         const cv::Mat& toGlobal,
                                         const cv::Point2d& center,
                                         AlignmentAxis axis)
{
    cv::Mat inverseFrom;
    if (!cv::invert(fromGlobal, inverseFrom)) {
        return {};
    }

    const cv::Mat relative = inverseFrom * toGlobal;
    TransformResult transform;
    transform.axis = axis;
    transform.da = std::atan2(relative.at<double>(1, 0), relative.at<double>(0, 0)) * 180.0 / CV_PI;

    const cv::Mat rotation2x3 = cv::getRotationMatrix2D(center, transform.da, 1.0);
    transform.dx = relative.at<double>(0, 2) - rotation2x3.at<double>(0, 2);
    transform.dy = relative.at<double>(1, 2) - rotation2x3.at<double>(1, 2);
    return transform;
}

MotionPriorDirection inferStepDirectionConstraint(const StitchStepRecord& step,
                                                  const TransformResult& priorTransform)
{
    const auto label = step.transform.direction;
    if (label.find("X+") != std::string::npos) {
        return MotionPriorDirection::XPositive;
    }
    if (label.find("X-") != std::string::npos) {
        return MotionPriorDirection::XNegative;
    }
    if (label.find("Y+") != std::string::npos) {
        return MotionPriorDirection::YPositive;
    }
    if (label.find("Y-") != std::string::npos) {
        return MotionPriorDirection::YNegative;
    }

    if (step.motionAxis == AlignmentAxis::Y) {
        double primaryShift = priorTransform.dy;
        if (std::abs(primaryShift) < 1e-6) {
            primaryShift = step.hasTrajectoryPrior ? step.trajectoryPriorDy
                                                   : (step.hasPairPrior ? step.pairPriorDy : step.nominalPriorDy);
        }
        return primaryShift < 0.0 ? MotionPriorDirection::YNegative : MotionPriorDirection::YPositive;
    }

    double primaryShift = priorTransform.dx;
    if (std::abs(primaryShift) < 1e-6) {
        primaryShift = step.hasTrajectoryPrior ? step.trajectoryPriorDx
                                               : (step.hasPairPrior ? step.pairPriorDx : step.nominalPriorDx);
    }
    return primaryShift < 0.0 ? MotionPriorDirection::XNegative : MotionPriorDirection::XPositive;
}

AlignmentAxis inferPrimaryAxisFromTransform(const TransformResult& transform)
{
    return std::abs(transform.dy) > std::abs(transform.dx)
               ? AlignmentAxis::Y
               : AlignmentAxis::X;
}

MotionPriorDirection inferDirectionConstraintFromTransform(const TransformResult& transform)
{
    const AlignmentAxis axis = inferPrimaryAxisFromTransform(transform);
    const double primaryShift = axis == AlignmentAxis::Y ? transform.dy : transform.dx;
    if (axis == AlignmentAxis::Y) {
        return primaryShift < 0.0 ? MotionPriorDirection::YNegative : MotionPriorDirection::YPositive;
    }
    return primaryShift < 0.0 ? MotionPriorDirection::XNegative : MotionPriorDirection::XPositive;
}

bool isMeasuredSkipEdgeAcceptable(const TransformResult& priorTransform,
                                  const TransformResult& measuredTransform,
                                  const std::size_t skipDistance)
{
    if (!measuredTransform.hasCandidate()) {
        return false;
    }

    const double normalRmse = preferredNormalRmse(measuredTransform);
    const double coverage = preferredCoverageRatio(measuredTransform);
    if (!std::isfinite(normalRmse) || !std::isfinite(coverage)) {
        return false;
    }

    const double primaryDelta = std::abs(primaryShiftDeltaPx(measuredTransform.axis,
                                                             measuredTransform.dx,
                                                             measuredTransform.dy,
                                                             priorTransform.dx,
                                                             priorTransform.dy));
    const double perpDelta = std::abs(perpendicularShiftDeltaPx(measuredTransform.axis,
                                                                measuredTransform.dx,
                                                                measuredTransform.dy,
                                                                priorTransform.dx,
                                                                priorTransform.dy));
    const double angleDelta = std::abs(measuredTransform.da - priorTransform.da);

    const double normalLimit = skipDistance <= 1 ? 0.18 : 0.24;
    const double coverageLimit = skipDistance <= 1 ? 0.30 : 0.16;
    const double primaryLimit = skipDistance <= 1 ? 42.0 : 70.0;
    const double perpLimit = skipDistance <= 1 ? 16.0 : 24.0;
    const double angleLimit = skipDistance <= 1 ? 0.06 : 0.10;

    return normalRmse <= normalLimit &&
           coverage >= coverageLimit &&
           primaryDelta <= primaryLimit &&
           perpDelta <= perpLimit &&
           angleDelta <= angleLimit;
}

struct MeasuredPoseEdgeBuildResult {
    std::vector<PoseGraphEdge> edges;
    std::size_t attemptedCount{0};
    std::size_t acceptedCount{0};
};

MeasuredPoseEdgeBuildResult buildMeasuredSkipEdges(const std::vector<cv::Mat>& images,
                                                   const std::vector<EdgeVariants>& edgesPrepared,
                                                   const std::vector<cv::Mat>& globalTransforms,
                                                   const StitchPipelineConfig& config,
                                                   const std::size_t skipDistance,
                                                   const double weightScale)
{
    MeasuredPoseEdgeBuildResult result;
    if (images.size() != globalTransforms.size() || edgesPrepared.size() != images.size()) {
        return result;
    }

    const std::size_t imageCount = images.size();
    if (imageCount <= skipDistance + 1) {
        return result;
    }

    const double rotationHalfRangeDeg =
        std::clamp(std::max(config.rotationSearchStepDeg * 5.0, 0.05), 0.05, 0.15);
    const double primaryHalfRangePx = skipDistance <= 1 ? 36.0 : 56.0;
    const double perpHalfRangePx = skipDistance <= 1 ? 12.0 : 18.0;

    for (std::size_t referenceIndex = 0; referenceIndex + skipDistance + 1 < imageCount; ++referenceIndex) {
        const std::size_t targetIndex = referenceIndex + skipDistance + 1;
        if (images[referenceIndex].empty() || images[targetIndex].empty()) {
            continue;
        }

        ++result.attemptedCount;
        const cv::Point2d center(images[referenceIndex].cols / 2.0,
                                 images[referenceIndex].rows / 2.0);
        TransformResult priorTransform =
            extractRelativeTransform(globalTransforms[referenceIndex],
                                     globalTransforms[targetIndex],
                                     center,
                                     AlignmentAxis::X);
        const MotionPriorDirection direction =
            inferDirectionConstraintFromTransform(priorTransform);

        double measuredSearchRangeX = 0.0;
        double measuredSearchRangeY = 0.0;
        TransformResult measuredTransform =
            matchOnePair(edgesPrepared[referenceIndex],
                         edgesPrepared[targetIndex],
                         center,
                         priorTransform.dx,
                         priorTransform.dy,
                         priorTransform.da,
                         true,
                         std::min(config.baseSearchRange, 160.0),
                         direction,
                         std::max(config.rotationSearchMinDeg, priorTransform.da - rotationHalfRangeDeg),
                         std::min(config.rotationSearchMaxDeg, priorTransform.da + rotationHalfRangeDeg),
                         std::min(config.rotationSearchStepDeg, 0.01),
                         config.tangentResidualCostWeight,
                         config.tangentCorrelationCostWeight,
                         true,
                         primaryHalfRangePx,
                         perpHalfRangePx,
                         measuredSearchRangeX,
                         measuredSearchRangeY);

        if (!isMeasuredSkipEdgeAcceptable(priorTransform, measuredTransform, skipDistance)) {
            continue;
        }

        PoseGraphEdge edge;
        edge.fromIndex = referenceIndex;
        edge.toIndex = targetIndex;
        edge.dx = measuredTransform.dx;
        edge.dy = measuredTransform.dy;
        edge.da = measuredTransform.da;

        const double coverage = preferredCoverageRatio(measuredTransform);
        const double normalRmse = preferredNormalRmse(measuredTransform);
        edge.weight = weightScale / std::max(0.01, normalRmse * normalRmse);
        edge.weight *= std::clamp(coverage, 0.25, 1.0);
        edge.weight = std::clamp(edge.weight, weightScale * 0.1, weightScale * 40.0);

        result.edges.push_back(edge);
        ++result.acceptedCount;
    }

    return result;
}

TransformResult refineStepWithGlobalPosePrior(const cv::Mat& referenceImage,
                                              const EdgeVariants& referenceEdges,
                                              const EdgeVariants& targetEdges,
                                              const StitchStepRecord& step,
                                              const StitchPipelineConfig& config,
                                              const TransformResult& optimizedPrior,
                                              double& outSearchRangeX,
                                              double& outSearchRangeY,
                                              bool& usedLocalRefinement)
{
    outSearchRangeX = 0.0;
    outSearchRangeY = 0.0;
    usedLocalRefinement = false;

    const cv::Point2d center(referenceImage.cols / 2.0, referenceImage.rows / 2.0);
    const MotionPriorDirection direction = inferStepDirectionConstraint(step, optimizedPrior);

    TransformResult evaluated =
        buildEvaluatedPriorTransform(referenceEdges,
                                     targetEdges,
                                     center,
                                     optimizedPrior.dx,
                                     optimizedPrior.dy,
                                     optimizedPrior.da,
                                     direction,
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight);
    evaluated.axis = step.motionAxis;
    if (evaluated.direction == "PRIOR") {
        evaluated.direction = motionDirectionLabel(direction);
    }

    const double localRotationHalfRangeDeg =
        std::clamp(std::max(config.rotationSearchStepDeg * 4.0, 0.03), 0.03, 0.10);
    const double localRotationMinDeg =
        std::max(config.rotationSearchMinDeg, optimizedPrior.da - localRotationHalfRangeDeg);
    const double localRotationMaxDeg =
        std::min(config.rotationSearchMaxDeg, optimizedPrior.da + localRotationHalfRangeDeg);

    double localSearchRangeX = 0.0;
    double localSearchRangeY = 0.0;
    TransformResult local =
        matchOnePair(referenceEdges,
                     targetEdges,
                     center,
                     optimizedPrior.dx,
                     optimizedPrior.dy,
                     optimizedPrior.da,
                     true,
                     std::min(config.baseSearchRange, 120.0),
                     direction,
                     localRotationMinDeg,
                     localRotationMaxDeg,
                     std::min(config.rotationSearchStepDeg, 0.01),
                     config.tangentResidualCostWeight,
                     config.tangentCorrelationCostWeight,
                     true,
                     18.0,
                     8.0,
                     localSearchRangeX,
                     localSearchRangeY);

    if (!local.hasCandidate()) {
        return evaluated;
    }

    const double targetNormalRmsePx = preferredNormalRmse(step.transform);
    const double evaluatedNormal = preferredNormalRmse(evaluated);
    const double localNormal = preferredNormalRmse(local);
    const double evaluatedCoverage = preferredCoverageRatio(evaluated);
    const double localCoverage = preferredCoverageRatio(local);
    const double evaluatedCost =
        transformSelectionCost(evaluated, optimizedPrior.dx, optimizedPrior.dy, targetNormalRmsePx);
    const double localCost =
        transformSelectionCost(local, optimizedPrior.dx, optimizedPrior.dy, targetNormalRmsePx);
    const double localCorrectionPx = std::hypot(local.dx - optimizedPrior.dx, local.dy - optimizedPrior.dy);
    const double localCorrectionDeg = std::abs(local.da - optimizedPrior.da);
    const bool catastrophicEvaluatedNormal =
        std::isfinite(evaluatedNormal) &&
        evaluatedNormal > std::max(0.35, targetNormalRmsePx + 0.15);
    const bool coverageAcceptable =
        localCoverage + 0.03 >= evaluatedCoverage ||
        (catastrophicEvaluatedNormal && localCoverage >= 0.50);

    const double maxLocalCorrectionPx = catastrophicEvaluatedNormal ? 18.0 : 8.0;
    const double maxLocalCorrectionDeg = catastrophicEvaluatedNormal ? 0.10 : 0.04;
    const bool boundedCorrection =
        localCorrectionPx <= maxLocalCorrectionPx &&
        localCorrectionDeg <= maxLocalCorrectionDeg;
    const bool meaningfulImprovement =
        coverageAcceptable &&
        ((std::isfinite(localNormal) && std::isfinite(evaluatedNormal) && localNormal + 0.01 < evaluatedNormal) ||
         localCost < evaluatedCost * 0.97 ||
         (std::isfinite(evaluatedNormal) && evaluatedNormal > 0.30 && localNormal < evaluatedNormal));

    if (!boundedCorrection || !meaningfulImprovement) {
        if (!local.candidateDiagnostics.empty()) {
            evaluated.candidateDiagnostics = std::move(local.candidateDiagnostics);
        }
        return evaluated;
    }

    outSearchRangeX = localSearchRangeX;
    outSearchRangeY = localSearchRangeY;
    usedLocalRefinement = true;
    return local;
}

std::size_t countStepsAboveRmseThreshold(const std::vector<StitchStepRecord>& steps,
                                         const double thresholdPx)
{
    std::size_t count = 0;
    for (const StitchStepRecord& step : steps) {
        const double rmse = preferredNormalRmse(step.transform);
        if (std::isfinite(rmse) && rmse > thresholdPx) {
            ++count;
        }
    }
    return count;
}

std::size_t countNewBadSteps(const std::vector<StitchStepRecord>& before,
                             const std::vector<StitchStepRecord>& after,
                             const double thresholdPx)
{
    const std::size_t count = std::min(before.size(), after.size());
    std::size_t newBadSteps = 0;
    for (std::size_t stepIndex = 0; stepIndex < count; ++stepIndex) {
        const double beforeRmse = preferredNormalRmse(before[stepIndex].transform);
        const double afterRmse = preferredNormalRmse(after[stepIndex].transform);
        if (!std::isfinite(beforeRmse) || !std::isfinite(afterRmse)) {
            continue;
        }
        if (beforeRmse <= thresholdPx && afterRmse > thresholdPx) {
            ++newBadSteps;
        }
    }
    return newBadSteps;
}

std::size_t countProtectedGoodStepRegressions(const std::vector<StitchStepRecord>& before,
                                              const std::vector<StitchStepRecord>& after)
{
    const std::size_t count = std::min(before.size(), after.size());
    std::size_t regressions = 0;
    for (std::size_t stepIndex = 0; stepIndex < count; ++stepIndex) {
        const double beforeRmse = preferredNormalRmse(before[stepIndex].transform);
        const double afterRmse = preferredNormalRmse(after[stepIndex].transform);
        if (!std::isfinite(beforeRmse) || !std::isfinite(afterRmse)) {
            continue;
        }
        if (beforeRmse <= 0.18 &&
            afterRmse > std::max(0.25, beforeRmse + 0.04)) {
            ++regressions;
        }
    }
    return regressions;
}

bool shouldAcceptGlobalFeedbackStep(const StitchStepRecord& rawStep,
                                    const TransformResult& candidateTransform,
                                    const TransformResult& optimizedPrior)
{
    if (!candidateTransform.hasCandidate()) {
        return false;
    }

    const double rawNormal = preferredNormalRmse(rawStep.transform);
    const double candidateNormal = preferredNormalRmse(candidateTransform);
    const double rawCoverage = preferredCoverageRatio(rawStep.transform);
    const double candidateCoverage = preferredCoverageRatio(candidateTransform);
    const double rawCost =
        transformSelectionCost(rawStep.transform, optimizedPrior.dx, optimizedPrior.dy, rawNormal);
    const double candidateCost =
        transformSelectionCost(candidateTransform, optimizedPrior.dx, optimizedPrior.dy, rawNormal);
    const double transformDeltaPx =
        std::hypot(candidateTransform.dx - rawStep.transform.dx,
                   candidateTransform.dy - rawStep.transform.dy);
    const double transformDeltaDeg = std::abs(candidateTransform.da - rawStep.transform.da);
    const bool rawStepProtected =
        std::isfinite(rawNormal) && rawNormal <= 0.18;
    const bool severeProtectedRegression =
        rawStepProtected &&
        std::isfinite(candidateNormal) &&
        candidateNormal > std::max(0.25, rawNormal + 0.04);
    const bool strongStableLocalImprovement =
        rawStepProtected &&
        std::isfinite(candidateNormal) &&
        candidateNormal + 0.03 < rawNormal &&
        candidateCoverage >= 0.50;

    if (severeProtectedRegression) {
        return false;
    }
    if (strongStableLocalImprovement) {
        return true;
    }

    if (std::isfinite(rawNormal) &&
        std::isfinite(candidateNormal) &&
        candidateNormal > rawNormal + 0.02 &&
        candidateCoverage <= rawCoverage + 0.02) {
        return false;
    }

    if (candidateCost > rawCost * 1.05 &&
        candidateCoverage + 0.02 < rawCoverage &&
        transformDeltaPx > 3.0) {
        return false;
    }

    if (transformDeltaPx > 18.0 || transformDeltaDeg > 0.12) {
        return std::isfinite(rawNormal) &&
               std::isfinite(candidateNormal) &&
               candidateNormal + 0.02 < rawNormal &&
               candidateCoverage + 0.02 >= rawCoverage;
    }

    return true;
}

std::pair<double, std::size_t> worstStepRmseSummary(const std::vector<StitchStepRecord>& steps)
{
    double worstRmse = -1.0;
    std::size_t worstStepIndex = 0;
    for (const StitchStepRecord& step : steps) {
        const double rmse = preferredNormalRmse(step.transform);
        if (!std::isfinite(rmse)) {
            continue;
        }
        if (rmse > worstRmse) {
            worstRmse = rmse;
            worstStepIndex = step.stepIndex;
        }
    }
    return {worstRmse, worstStepIndex};
}

std::pair<double, double> endpointStepRmseSummary(const std::vector<StitchStepRecord>& steps)
{
    if (steps.empty()) {
        return {-1.0, -1.0};
    }

    const double firstRmse = preferredNormalRmse(steps.front().transform);
    const double lastRmse = preferredNormalRmse(steps.back().transform);
    return {firstRmse, lastRmse};
}

struct GlobalPoseCorrectionSummary {
    double maxDeltaDx{0.0};
    double maxDeltaDy{0.0};
    double maxDeltaAngleDeg{0.0};

    bool meaningfulCorrection() const
    {
        return maxDeltaDx >= 0.5 || maxDeltaDy >= 0.5 || maxDeltaAngleDeg >= 0.01;
    }
};

GlobalPoseCorrectionSummary summarizeGlobalPoseCorrections(const std::vector<cv::Mat>& before,
                                                           const std::vector<cv::Mat>& after)
{
    GlobalPoseCorrectionSummary summary;
    const std::size_t count = std::min(before.size(), after.size());
    for (std::size_t index = 0; index < count; ++index) {
        summary.maxDeltaDx = std::max(
            summary.maxDeltaDx,
            std::abs(after[index].at<double>(0, 2) - before[index].at<double>(0, 2)));
        summary.maxDeltaDy = std::max(
            summary.maxDeltaDy,
            std::abs(after[index].at<double>(1, 2) - before[index].at<double>(1, 2)));
        const double beforeAngleDeg =
            std::atan2(before[index].at<double>(1, 0), before[index].at<double>(0, 0)) * 180.0 / CV_PI;
        const double afterAngleDeg =
            std::atan2(after[index].at<double>(1, 0), after[index].at<double>(0, 0)) * 180.0 / CV_PI;
        summary.maxDeltaAngleDeg =
            std::max(summary.maxDeltaAngleDeg, std::abs(afterAngleDeg - beforeAngleDeg));
    }
    return summary;
}

struct GlobalFeedbackRefinementSummary {
    std::vector<StitchStepRecord> steps;
    std::size_t localRefinedStepCount{0};
    double maxLocalRefineCorrectionPx{0.0};
};

GlobalFeedbackRefinementSummary refineStepsWithGlobalPosePrior(
    const std::vector<cv::Mat>& images,
    const std::vector<EdgeVariants>& edgesPrepared,
    const std::vector<cv::Mat>& optimized,
    const std::vector<StitchStepRecord>& rawSteps,
    const StitchPipelineConfig& config)
{
    GlobalFeedbackRefinementSummary summary;
    summary.steps = rawSteps;

    for (std::size_t stepIndex = 0; stepIndex < summary.steps.size(); ++stepIndex) {
        const StitchStepRecord& rawStep = rawSteps[stepIndex];
        if (rawStep.referenceImageIndex >= images.size() ||
            rawStep.targetImageIndex >= images.size() ||
            rawStep.referenceImageIndex >= edgesPrepared.size() ||
            rawStep.targetImageIndex >= edgesPrepared.size()) {
            continue;
        }

        const cv::Point2d center(images[rawStep.referenceImageIndex].cols / 2.0,
                                 images[rawStep.referenceImageIndex].rows / 2.0);
        const TransformResult optimizedPrior =
            extractRelativeTransform(optimized[rawStep.referenceImageIndex],
                                     optimized[rawStep.targetImageIndex],
                                     center,
                                     rawStep.motionAxis);

        double refinedSearchRangeX = 0.0;
        double refinedSearchRangeY = 0.0;
        bool usedLocalRefinement = false;
        TransformResult refinedTransform =
            refineStepWithGlobalPosePrior(images[rawStep.referenceImageIndex],
                                          edgesPrepared[rawStep.referenceImageIndex],
                                          edgesPrepared[rawStep.targetImageIndex],
                                          rawStep,
                                          config,
                                          optimizedPrior,
                                          refinedSearchRangeX,
                                          refinedSearchRangeY,
                                          usedLocalRefinement);

        if (!shouldAcceptGlobalFeedbackStep(rawStep, refinedTransform, optimizedPrior)) {
            continue;
        }

        summary.steps[stepIndex].transform = std::move(refinedTransform);
        if (usedLocalRefinement) {
            summary.steps[stepIndex].searchRangeX = refinedSearchRangeX;
            summary.steps[stepIndex].searchRangeY = refinedSearchRangeY;
            ++summary.localRefinedStepCount;
        }

        summary.maxLocalRefineCorrectionPx = std::max(
            summary.maxLocalRefineCorrectionPx,
            std::hypot(summary.steps[stepIndex].transform.dx - optimizedPrior.dx,
                       summary.steps[stepIndex].transform.dy - optimizedPrior.dy));
    }

    return summary;
}

struct GlobalFeedbackEvaluationSummary {
    std::pair<double, std::size_t> worstBefore{-1.0, 0};
    std::pair<double, std::size_t> worstAfter{-1.0, 0};
    std::size_t badStepCountBefore{0};
    std::size_t badStepCountAfter{0};
    std::size_t newBadStepCount{0};
    std::size_t protectedRegressionCount{0};
    std::pair<double, double> endpointBefore{-1.0, -1.0};
    std::pair<double, double> endpointAfter{-1.0, -1.0};
    double endpointBeforeSum{-1.0};
    double endpointAfterSum{-1.0};
    bool endpointImproved{false};
    bool improvedBadStepCount{false};
    bool improvedWorstStep{false};
    bool regressedBadStepCount{false};
    bool severeWorstStepRegression{false};
    bool accepted{false};
};

GlobalFeedbackEvaluationSummary evaluateGlobalFeedbackOutcome(
    const std::vector<StitchStepRecord>& rawSteps,
    const std::vector<StitchStepRecord>& refinedSteps)
{
    GlobalFeedbackEvaluationSummary summary;
    summary.worstBefore = worstStepRmseSummary(rawSteps);
    summary.worstAfter = worstStepRmseSummary(refinedSteps);
    summary.badStepCountBefore = countStepsAboveRmseThreshold(rawSteps, 0.25);
    summary.badStepCountAfter = countStepsAboveRmseThreshold(refinedSteps, 0.25);
    summary.newBadStepCount = countNewBadSteps(rawSteps, refinedSteps, 0.25);
    summary.protectedRegressionCount = countProtectedGoodStepRegressions(rawSteps, refinedSteps);
    summary.endpointBefore = endpointStepRmseSummary(rawSteps);
    summary.endpointAfter = endpointStepRmseSummary(refinedSteps);
    summary.endpointBeforeSum = summary.endpointBefore.first + summary.endpointBefore.second;
    summary.endpointAfterSum = summary.endpointAfter.first + summary.endpointAfter.second;

    summary.endpointImproved =
        std::isfinite(summary.endpointBeforeSum) &&
        std::isfinite(summary.endpointAfterSum) &&
        summary.endpointAfter.first <= summary.endpointBefore.first + 0.005 &&
        summary.endpointAfter.second <= summary.endpointBefore.second + 0.005 &&
        summary.endpointAfterSum + std::max(5e-5, summary.endpointBeforeSum * 0.00025) <
            summary.endpointBeforeSum;
    summary.improvedBadStepCount = summary.badStepCountAfter < summary.badStepCountBefore;
    summary.improvedWorstStep =
        summary.worstBefore.first >= 0.0 &&
        summary.worstAfter.first >= 0.0 &&
        summary.worstAfter.first + 0.002 < summary.worstBefore.first;
    summary.regressedBadStepCount = summary.badStepCountAfter > summary.badStepCountBefore;
    summary.severeWorstStepRegression =
        summary.worstBefore.first >= 0.0 &&
        summary.worstAfter.first >= 0.0 &&
        summary.worstAfter.first > std::max(0.25, summary.worstBefore.first + 0.05) &&
        summary.worstAfter.first > summary.worstBefore.first * 1.5;
    summary.accepted =
        (summary.worstBefore.first < 0.0 ||
         summary.improvedBadStepCount ||
         summary.improvedWorstStep ||
         summary.endpointImproved) &&
        !summary.regressedBadStepCount &&
        !summary.severeWorstStepRegression &&
        summary.newBadStepCount == 0 &&
        summary.protectedRegressionCount == 0;

    return summary;
}

std::vector<cv::Mat> buildGlobalTransformsFromSteps(const std::vector<cv::Mat>& images,
                                                    const std::vector<StitchStepRecord>& steps)
{
    std::vector<cv::Mat> globalTransforms;
    globalTransforms.reserve(steps.size() + 1);
    globalTransforms.push_back(cv::Mat::eye(3, 3, CV_64F));

    for (const StitchStepRecord& step : steps) {
        if (step.referenceImageIndex >= images.size()) {
            return {};
        }

        const cv::Point2d center(images[step.referenceImageIndex].cols / 2.0,
                                 images[step.referenceImageIndex].rows / 2.0);
        const cv::Mat relative = buildRelativeMatrix(center, step.transform);
        globalTransforms.push_back(globalTransforms.back() * relative);
    }

    return globalTransforms;
}

bool rebuildPanoramaFromTransforms(const std::vector<cv::Mat>& images,
                                   const std::vector<cv::Mat>& globalTransforms,
                                   cv::Mat& canvas,
                                   std::vector<cv::Mat>& shiftedTransforms)
{
    if (images.empty() || globalTransforms.size() != images.size()) {
        return false;
    }

    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        if (images[imageIndex].empty() || globalTransforms[imageIndex].empty()) {
            return false;
        }

        const double maxImageX = std::max(0, images[imageIndex].cols - 1);
        const double maxImageY = std::max(0, images[imageIndex].rows - 1);
        const std::array<cv::Point2d, 4> corners = {
            cv::Point2d(0.0, 0.0),
            cv::Point2d(maxImageX, 0.0),
            cv::Point2d(maxImageX, maxImageY),
            cv::Point2d(0.0, maxImageY)
        };

        for (const cv::Point2d& corner : corners) {
            const cv::Point2d mapped = transformPointLocal(globalTransforms[imageIndex], corner);
            if (!std::isfinite(mapped.x) || !std::isfinite(mapped.y)) {
                return false;
            }
            minX = std::min(minX, mapped.x);
            minY = std::min(minY, mapped.y);
            maxX = std::max(maxX, mapped.x);
            maxY = std::max(maxY, mapped.y);
        }
    }

    if (!std::isfinite(minX) || !std::isfinite(minY) ||
        !std::isfinite(maxX) || !std::isfinite(maxY)) {
        return false;
    }

    constexpr int kPaddingPx = 32;
    const int canvasWidth =
        std::max(1, static_cast<int>(std::ceil(maxX - minX + 1.0)) + 2 * kPaddingPx);
    const int canvasHeight =
        std::max(1, static_cast<int>(std::ceil(maxY - minY + 1.0)) + 2 * kPaddingPx);

    cv::Mat offset = cv::Mat::eye(3, 3, CV_64F);
    offset.at<double>(0, 2) = -minX + kPaddingPx;
    offset.at<double>(1, 2) = -minY + kPaddingPx;

    canvas = cv::Mat(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    shiftedTransforms.clear();
    shiftedTransforms.reserve(globalTransforms.size());

    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        cv::Mat shifted = offset * globalTransforms[imageIndex];
        shiftedTransforms.push_back(shifted);

        cv::Mat warped;
        cv::warpAffine(images[imageIndex],
                       warped,
                       shifted.rowRange(0, 2),
                       canvas.size(),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(255, 255, 255));
        cv::min(canvas, warped, canvas);
    }

    return true;
}

double primaryShiftValue(const TransformResult& transform, const AlignmentAxis axis)
{
    return axis == AlignmentAxis::X ? transform.dx : transform.dy;
}

void scalePrimaryShift(TransformResult& transform, const AlignmentAxis axis, const double factor)
{
    if (axis == AlignmentAxis::X) {
        transform.dx *= factor;
    } else {
        transform.dy *= factor;
    }
}

struct DesignScaleFeedbackCandidate {
    bool ok{false};
    double appliedScaleFactor{1.0};
    std::vector<StitchStepRecord> steps;
    cv::Mat canvas;
    std::vector<cv::Mat> imageTransforms;
    pinjie::cad_design::DesignAlignmentResult design;
    GlobalFeedbackEvaluationSummary stepEvaluation;
};

std::vector<double> buildDesignScaleFeedbackCandidates(const double estimatedScaleFactor)
{
    std::vector<double> candidates{1.0};
    if (!std::isfinite(estimatedScaleFactor)) {
        return candidates;
    }

    constexpr double kMaxAbsScaleDelta = 5.0e-3;
    const auto appendCandidate = [&](const double value) {
        if (!std::isfinite(value) || std::abs(value - 1.0) > kMaxAbsScaleDelta) {
            return;
        }
        for (const double existing : candidates) {
            if (std::abs(existing - value) <= 5e-7) {
                return;
            }
        }
        candidates.push_back(value);
    };

    appendCandidate(estimatedScaleFactor);
    const double delta = estimatedScaleFactor - 1.0;
    if (std::abs(delta) < 2.5e-4) {
        return candidates;
    }

    for (const double ratio : std::array<double, 5>{0.5, 0.75, 1.0, 1.25, 1.5}) {
        appendCandidate(1.0 + delta * ratio);
    }

    for (const double offset : std::array<double, 4>{5.0e-4, 1.0e-3, -5.0e-4, -1.0e-3}) {
        appendCandidate(estimatedScaleFactor + offset);
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

bool isAcceptableDesignScaleFeedbackStepOutcome(const GlobalFeedbackEvaluationSummary& stepEvaluation)
{
    if (stepEvaluation.protectedRegressionCount > 4) {
        return false;
    }
    if (stepEvaluation.newBadStepCount > 5) {
        return false;
    }
    if (stepEvaluation.badStepCountAfter > stepEvaluation.badStepCountBefore + 5) {
        return false;
    }
    if (stepEvaluation.worstAfter.first >= 0.0 &&
        stepEvaluation.worstAfter.first >
            std::max(0.80, std::max(0.45, stepEvaluation.worstBefore.first + 0.70))) {
        return false;
    }
    if (stepEvaluation.worstBefore.first >= 0.0 &&
        stepEvaluation.worstAfter.first >= 0.0 &&
        stepEvaluation.worstAfter.first > std::max(0.12, stepEvaluation.worstBefore.first + 0.015)) {
        return false;
    }
    return true;
}

DesignScaleFeedbackCandidate evaluateDesignScaleFeedbackCandidate(
    const std::vector<cv::Mat>& images,
    const std::vector<EdgeVariants>& edgesPrepared,
    const StitchPipelineConfig& config,
    const std::vector<StitchStepRecord>& rawSteps,
    const double scaleFactor)
{
    DesignScaleFeedbackCandidate candidate;
    candidate.appliedScaleFactor = scaleFactor;
    if (rawSteps.empty()) {
        return candidate;
    }

    candidate.steps = rawSteps;
    std::size_t updatedStepCount = 0;
    for (std::size_t stepIndex = 0; stepIndex < candidate.steps.size(); ++stepIndex) {
        const StitchStepRecord& rawStep = rawSteps[stepIndex];
        if (rawStep.referenceImageIndex >= images.size() ||
            rawStep.targetImageIndex >= images.size() ||
            rawStep.referenceImageIndex >= edgesPrepared.size() ||
            rawStep.targetImageIndex >= edgesPrepared.size()) {
            return candidate;
        }

        TransformResult scaledTransform = rawStep.transform;
        scalePrimaryShift(scaledTransform, rawStep.motionAxis, scaleFactor);

        const cv::Point2d center(images[rawStep.referenceImageIndex].cols / 2.0,
                                 images[rawStep.referenceImageIndex].rows / 2.0);
        const MotionPriorDirection direction = inferStepDirectionConstraint(rawStep, scaledTransform);
        TransformResult evaluated =
            buildEvaluatedPriorTransform(edgesPrepared[rawStep.referenceImageIndex],
                                         edgesPrepared[rawStep.targetImageIndex],
                                         center,
                                         scaledTransform.dx,
                                         scaledTransform.dy,
                                         scaledTransform.da,
                                         direction,
                                         config.tangentResidualCostWeight,
                                         config.tangentCorrelationCostWeight);
        if (!evaluated.hasCandidate()) {
            return candidate;
        }

        evaluated.axis = rawStep.motionAxis;
        if (evaluated.direction == "PRIOR") {
            evaluated.direction = motionDirectionLabel(direction);
        }

        candidate.steps[stepIndex].transform = std::move(evaluated);
        candidate.steps[stepIndex].selectionMode = rawStep.selectionMode + "+design_scale_feedback";
        ++updatedStepCount;
    }

    if (updatedStepCount != candidate.steps.size()) {
        return candidate;
    }

    candidate.stepEvaluation = evaluateGlobalFeedbackOutcome(rawSteps, candidate.steps);
    if (!isAcceptableDesignScaleFeedbackStepOutcome(candidate.stepEvaluation)) {
        return candidate;
    }
    const std::size_t targetCountBefore = countStepsAboveRmseThreshold(rawSteps, 0.10);
    const std::size_t targetCountAfter = countStepsAboveRmseThreshold(candidate.steps, 0.10);
    if (targetCountAfter > targetCountBefore) {
        return candidate;
    }

    const std::vector<cv::Mat> correctedGlobals = buildGlobalTransformsFromSteps(images, candidate.steps);
    if (correctedGlobals.empty() ||
        !rebuildPanoramaFromTransforms(
            images, correctedGlobals, candidate.canvas, candidate.imageTransforms)) {
        return candidate;
    }

    candidate.design =
        pinjie::cad_design::compareMeasuredProfileToDesign(edgesPrepared, candidate.imageTransforms, config);
    if (!candidate.design.ok) {
        return candidate;
    }

    candidate.ok = true;
    return candidate;
}

bool isBetterDesignScaleFeedbackCandidate(const DesignScaleFeedbackCandidate& challenger,
                                          const DesignScaleFeedbackCandidate& incumbent)
{
    if (!challenger.ok) {
        return false;
    }
    if (!incumbent.ok) {
        return true;
    }

    const double challengerWorst = challenger.stepEvaluation.worstAfter.first;
    const double incumbentWorst = incumbent.stepEvaluation.worstAfter.first;
    if (std::isfinite(challengerWorst) && std::isfinite(incumbentWorst)) {
        if (challengerWorst + 0.005 < incumbentWorst) {
            return true;
        }
        if (incumbentWorst + 0.005 < challengerWorst) {
            return false;
        }
    }

    constexpr double kProfilePriorityUm = 0.05;
    constexpr double kNormalPriorityUm = 0.05;
    const double challengerProfile = challenger.design.summary.profileStats.rmseUm;
    const double incumbentProfile = incumbent.design.summary.profileStats.rmseUm;
    if (challengerProfile + kProfilePriorityUm < incumbentProfile) {
        return true;
    }
    if (incumbentProfile + kProfilePriorityUm < challengerProfile) {
        return false;
    }

    const double challengerNormal = challenger.design.summary.normalStats.rmseUm;
    const double incumbentNormal = incumbent.design.summary.normalStats.rmseUm;
    if (challengerNormal + kNormalPriorityUm < incumbentNormal) {
        return true;
    }
    if (incumbentNormal + kNormalPriorityUm < challengerNormal) {
        return false;
    }

    const double challengerResidualScale =
        std::abs(challenger.design.summary.axialScaleFactor - 1.0);
    const double incumbentResidualScale =
        std::abs(incumbent.design.summary.axialScaleFactor - 1.0);
    if (challengerResidualScale + 1e-5 < incumbentResidualScale) {
        return true;
    }
    if (incumbentResidualScale + 1e-5 < challengerResidualScale) {
        return false;
    }

    return std::abs(challenger.appliedScaleFactor - 1.0) <
           std::abs(incumbent.appliedScaleFactor - 1.0);
}

bool applyDesignDrivenPrimaryScaleFeedback(const std::vector<cv::Mat>& images,
                                           const std::vector<EdgeVariants>& edgesPrepared,
                                           const StitchPipelineConfig& config,
                                           StitchingResult& result,
                                           const StitchCallbacks& callbacks)
{
    if (images.size() < 2 ||
        edgesPrepared.size() != images.size() ||
        result.steps.empty() ||
        result.imageTransforms.size() != images.size()) {
        return false;
    }

    const pinjie::cad_design::DesignAlignmentResult beforeDesign =
        pinjie::cad_design::compareMeasuredProfileToDesign(edgesPrepared, result.imageTransforms, config);
    if (!beforeDesign.ok) {
        return false;
    }

    const double estimatedScaleFactor = beforeDesign.summary.axialScaleFactor;
    if (!std::isfinite(estimatedScaleFactor) ||
        std::abs(estimatedScaleFactor - 1.0) < 5.0e-4 ||
        std::abs(estimatedScaleFactor - 1.0) > 5.0e-3) {
        return false;
    }

    const std::vector<double> candidateScaleFactors =
        buildDesignScaleFeedbackCandidates(estimatedScaleFactor);
    if (candidateScaleFactors.size() <= 1) {
        return false;
    }

    DesignScaleFeedbackCandidate bestCandidate;
    for (const double candidateScaleFactor : candidateScaleFactors) {
        if (std::abs(candidateScaleFactor - 1.0) < 1e-7) {
            continue;
        }

        DesignScaleFeedbackCandidate candidate =
            evaluateDesignScaleFeedbackCandidate(
                images, edgesPrepared, config, result.steps, candidateScaleFactor);
        if (!candidate.ok) {
            emitLog(callbacks,
                    "    [Design-scale feedback] candidate rejected before global compare: scale=" +
                        std::to_string(candidateScaleFactor));
            continue;
        }

        emitLog(callbacks,
                "    [Design-scale feedback] candidate scale=" +
                    std::to_string(candidateScaleFactor) +
                    ", profile RMS=" +
                    std::to_string(candidate.design.summary.profileStats.rmseUm) +
                    " um, normal RMSE=" +
                    std::to_string(candidate.design.summary.normalStats.rmseUm) +
                    " um, residual scale=" +
                    std::to_string(candidate.design.summary.axialScaleFactor) +
                    ", worst-step=" +
                    std::to_string(candidate.stepEvaluation.worstAfter.first) +
                    " px");

        if (isBetterDesignScaleFeedbackCandidate(candidate, bestCandidate)) {
            bestCandidate = std::move(candidate);
        }
    }

    if (!bestCandidate.ok) {
        return false;
    }

    constexpr double kMinMeaningfulImprovementUm = 0.10;
    const bool profileImproved =
        bestCandidate.design.summary.profileStats.rmseUm + kMinMeaningfulImprovementUm <
        beforeDesign.summary.profileStats.rmseUm;
    const bool normalImproved =
        bestCandidate.design.summary.normalStats.rmseUm + kMinMeaningfulImprovementUm <
        beforeDesign.summary.normalStats.rmseUm;
    if (!profileImproved && !normalImproved) {
        emitLog(callbacks,
                "    [Design-scale feedback] rejected: no meaningful global improvement, scale=" +
                    std::to_string(bestCandidate.appliedScaleFactor) +
                    ", profile RMS " +
                    std::to_string(beforeDesign.summary.profileStats.rmseUm) +
                    " -> " +
                    std::to_string(bestCandidate.design.summary.profileStats.rmseUm) +
                    " um, normal RMSE " +
                    std::to_string(beforeDesign.summary.normalStats.rmseUm) +
                    " -> " +
                    std::to_string(bestCandidate.design.summary.normalStats.rmseUm) + " um");
        return false;
    }

    result.steps = std::move(bestCandidate.steps);
    result.canvas = std::move(bestCandidate.canvas);
    result.imageTransforms = std::move(bestCandidate.imageTransforms);
    if (!result.imageTransforms.empty()) {
        result.globalTransform = result.imageTransforms.back().clone();
    }

    emitLog(callbacks,
            "    [Design-scale feedback] accepted: scale=" +
                std::to_string(bestCandidate.appliedScaleFactor) +
                ", estimated residual scale " +
                std::to_string(beforeDesign.summary.axialScaleFactor) +
                " -> " +
                std::to_string(bestCandidate.design.summary.axialScaleFactor) +
                ", profile RMS " +
                std::to_string(beforeDesign.summary.profileStats.rmseUm) +
                " -> " +
                std::to_string(bestCandidate.design.summary.profileStats.rmseUm) +
                " um, normal RMSE " +
                std::to_string(beforeDesign.summary.normalStats.rmseUm) +
                " -> " +
                std::to_string(bestCandidate.design.summary.normalStats.rmseUm) +
                " um, worst-step " +
                std::to_string(bestCandidate.stepEvaluation.worstBefore.first) +
                " -> " +
                std::to_string(bestCandidate.stepEvaluation.worstAfter.first) + " px");
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
    std::vector<TransformResult> reliableTransformHistory;
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
        const auto stepBegin = std::chrono::steady_clock::now();

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
            reliableTransformHistory.clear();
        }
        if (i < config.stepTranslationPriorsPx.size()) {
            approxShiftX = config.stepTranslationPriorsPx[i].x;
            approxShiftY = config.stepTranslationPriorsPx[i].y;
            approxInitX = approxShiftX;
            approxInitY = approxShiftY;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
            hasLastReliableTransform = false;
            reliableTransformHistory.clear();
        }
        const TrajectoryPriorEstimate trajectoryPrior =
            estimateTrajectoryPrior(reliableTransformHistory, stepDirection);
        const bool useTrajectoryLocalFirstPass = trajectoryPrior.ok && hasReliableMotionPrior;
        double pairPriorX = useTrajectoryLocalFirstPass ? trajectoryPrior.dx : approxShiftX;
        double pairPriorY = useTrajectoryLocalFirstPass ? trajectoryPrior.dy : approxShiftY;
        const double pairPriorAngle = useTrajectoryLocalFirstPass ? trajectoryPrior.angleDeg : approxAngleDeg;
        const bool hasFixedDirectionPrior = stepDirection != MotionPriorDirection::Auto;
        ImageCorrelationShiftEstimate imageCorrelationEstimate;
        bool useImageCorrelationLocalFirstPass = false;
        double imageLocalPrimaryHalfRangePx = 0.0;
        double imageLocalPerpHalfRangePx = 0.0;
        if (!useTrajectoryLocalFirstPass && hasFixedDirectionPrior) {
            const bool primaryIsX = directionUsesPrimaryX(stepDirection);
            const double nominalPrimaryShiftPx = primaryIsX ? approxInitX : approxInitY;
            const double primaryPriorShiftPx = primaryIsX ? pairPriorX : pairPriorY;
            const double secondaryPriorShiftPx = primaryIsX ? pairPriorY : pairPriorX;
            imageCorrelationEstimate =
                estimateImageCorrelationShift(images[i],
                                              images[i + 1],
                                              stepDirection,
                                              primaryPriorShiftPx,
                                              secondaryPriorShiftPx,
                                              nominalPrimaryShiftPx);
            if (imageCorrelationEstimate.ok) {
                const double imagePriorDeltaPx =
                    std::hypot(imageCorrelationEstimate.dx - pairPriorX,
                               imageCorrelationEstimate.dy - pairPriorY);
                if (imageCorrelationEstimate.score >= 0.35 &&
                    imagePriorDeltaPx <= std::max(90.0, config.baseSearchRange * 0.85)) {
                    pairPriorX = imageCorrelationEstimate.dx;
                    pairPriorY = imageCorrelationEstimate.dy;
                    useImageCorrelationLocalFirstPass = true;
                    const double imagePrimaryShiftPx = primaryIsX ? pairPriorX : pairPriorY;
                    const double imagePerpShiftPx = primaryIsX ? pairPriorY : pairPriorX;
                    imageLocalPrimaryHalfRangePx =
                        std::clamp(std::abs(imagePrimaryShiftPx - nominalPrimaryShiftPx) + 90.0, 90.0, 140.0);
                    imageLocalPerpHalfRangePx =
                        std::clamp(std::abs(imagePerpShiftPx) + 18.0, 20.0, 48.0);
                }
            }
        }
        const double targetNormalRmsePx =
            estimateRecentStableRmsePx(reliableTransformHistory,
                                       hasLastReliableTransform ? &lastReliableTransform : nullptr);
        double pairRotationMinDeg = config.rotationSearchMinDeg;
        double pairRotationMaxDeg = config.rotationSearchMaxDeg;
        if (useTrajectoryLocalFirstPass) {
            pairRotationMinDeg =
                std::max(config.rotationSearchMinDeg, pairPriorAngle - trajectoryPrior.angleWindowDeg);
            pairRotationMaxDeg =
                std::min(config.rotationSearchMaxDeg, pairPriorAngle + trajectoryPrior.angleWindowDeg);
            if (pairRotationMinDeg > pairRotationMaxDeg) {
                pairRotationMinDeg = config.rotationSearchMinDeg;
                pairRotationMaxDeg = config.rotationSearchMaxDeg;
            }
        } else if (useImageCorrelationLocalFirstPass) {
            pairRotationMinDeg = std::max(config.rotationSearchMinDeg, pairPriorAngle - 0.15);
            pairRotationMaxDeg = std::min(config.rotationSearchMaxDeg, pairPriorAngle + 0.15);
        }

        if (useImageCorrelationLocalFirstPass) {
            std::ostringstream initialImagePriorLog;
            initialImagePriorLog << "    [Info] Initial pass seeded by raw-image coarse shift dx="
                                 << pairPriorX << ", dy=" << pairPriorY
                                 << ", corr=" << imageCorrelationEstimate.score << ".";
            emitLog(callbacks, initialImagePriorLog.str());
        }

        const auto initialMatchBegin = std::chrono::steady_clock::now();
        TransformResult transform = matchOnePair(previousEdges, nextEdges, center,
                                                 pairPriorX, pairPriorY,
                                                 pairPriorAngle,
                                                 hasReliableMotionPrior || useTrajectoryLocalFirstPass ||
                                                     useImageCorrelationLocalFirstPass,
                                                 config.baseSearchRange,
                                                 stepDirection,
                                                 pairRotationMinDeg,
                                                 pairRotationMaxDeg,
                                                 config.rotationSearchStepDeg,
                                                 config.tangentResidualCostWeight,
                                                 config.tangentCorrelationCostWeight,
                                                 useTrajectoryLocalFirstPass || useImageCorrelationLocalFirstPass,
                                                 useTrajectoryLocalFirstPass ? trajectoryPrior.primaryWindowPx
                                                                            : imageLocalPrimaryHalfRangePx,
                                                 useTrajectoryLocalFirstPass ? trajectoryPrior.perpWindowPx
                                                                            : imageLocalPerpHalfRangePx,
                                                 searchRangeX, searchRangeY);
        const auto initialMatchEnd = std::chrono::steady_clock::now();
        const double initialMatchSeconds =
            std::chrono::duration<double>(initialMatchEnd - initialMatchBegin).count();
        if (!transform.profilingSummary.empty()) {
            emitLog(callbacks,
                    "    [Profile] initial matchOnePair runtime = " +
                        std::to_string(initialMatchSeconds) + " s | " +
                        transform.profilingSummary);
        }

        bool useQualityLocalRescan = false;
        bool useCandidateReselect = false;
        bool useWideSearchRescue = false;
        bool useUnfilteredEdgeRescue = false;
        bool useTrajectoryPriorRescue = false;
        bool useImageCorrelationRescue = false;
        bool useEndpointImageGuidedRefine = false;
        bool usePriorImageGuidedRefine = false;
        bool usePriorNormalGridRefine = false;
        bool usePriorAffineShearRefine = false;
        bool useLastStepTailTrimRefine = false;
        bool useLastStepReferencePriorHalfOverlapProbe = false;
        bool candidateLastStepReferencePriorHalfOverlapProbe = false;
        bool useLastStepReferencePriorHalfOverlapPullback = false;
        bool useLastStepNormalGridRefine = false;
        bool useLastStepAffineShearRefine = false;
        bool usedMotionPriorFallback = false;
        const auto normalRmseFor = [](const TransformResult& result) {
            const ResidualStatistics& normal =
                result.metrics.normalInlier.valid() ? result.metrics.normalInlier : result.metrics.normalAll;
            return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
        };
        double currentNormal = 0.0;
        double currentCoverage = 0.0;
        double currentSelectionCost = 0.0;
        bool suspiciousTransform = false;
        const auto refreshTransformAssessment = [&]() {
            currentNormal = normalRmseFor(transform);
            currentCoverage = preferredCoverageRatio(transform);
            currentSelectionCost = transformSelectionCost(transform, pairPriorX, pairPriorY, targetNormalRmsePx);
            suspiciousTransform = isSuspiciousTransform(transform, pairPriorX, pairPriorY) ||
                                  touchesRotationBoundary(transform.da, pairRotationMinDeg, pairRotationMaxDeg);
        };
        refreshTransformAssessment();
        const bool shouldTryWideSearchRescue =
            suspiciousTransform &&
            (config.baseSearchRange < 1000.0 ||
             std::max(std::abs(config.rotationSearchMinDeg), std::abs(config.rotationSearchMaxDeg)) < 1.0 - 1e-9);
        if (shouldTryWideSearchRescue) {
            double wideSearchRangeX = 0.0;
            double wideSearchRangeY = 0.0;
            const double wideBaseSearchRange =
                hasFixedDirectionPrior ? std::max(config.baseSearchRange, 220.0)
                                       : std::max(config.baseSearchRange, 3000.0);
            const double wideRotationMinDeg =
                hasFixedDirectionPrior ? config.rotationSearchMinDeg
                                       : std::min(config.rotationSearchMinDeg, -1.0);
            const double wideRotationMaxDeg =
                hasFixedDirectionPrior ? config.rotationSearchMaxDeg
                                       : std::max(config.rotationSearchMaxDeg, 1.0);
            const double wideRotationStepDeg =
                hasFixedDirectionPrior ? config.rotationSearchStepDeg
                                       : std::max(config.rotationSearchStepDeg, 0.05);
            TransformResult wideTransform =
                matchOnePair(previousEdges, nextEdges, center,
                             pairPriorX, pairPriorY,
                             pairPriorAngle, hasReliableMotionPrior || useTrajectoryLocalFirstPass,
                             wideBaseSearchRange,
                             stepDirection,
                             wideRotationMinDeg,
                             wideRotationMaxDeg,
                             wideRotationStepDeg,
                             config.tangentResidualCostWeight,
                             config.tangentCorrelationCostWeight,
                             false,
                             0.0,
                             0.0,
                             wideSearchRangeX, wideSearchRangeY);
            const double wideNormal = normalRmseFor(wideTransform);
            const double wideCoverage = preferredCoverageRatio(wideTransform);
            const double wideSelectionCost =
                transformSelectionCost(wideTransform, pairPriorX, pairPriorY, targetNormalRmsePx);
            const double currentContinuity =
                continuityDistanceToPrior(transform, pairPriorX, pairPriorY);
            const double wideContinuity =
                continuityDistanceToPrior(wideTransform, pairPriorX, pairPriorY);
            const bool continuityGuardPass =
                !useTrajectoryLocalFirstPass ||
                wideContinuity <= std::max(currentContinuity + 8.0,
                                           trajectoryPrior.primaryWindowPx * 1.35);
            if (continuityGuardPass &&
                wideCoverage + 0.04 >= currentCoverage &&
                (wideNormal + 0.02 < currentNormal ||
                 wideSelectionCost < currentSelectionCost * 0.90 ||
                 wideNormal < currentNormal * 0.70)) {
                transform = wideTransform;
                searchRangeX = wideSearchRangeX;
                searchRangeY = wideSearchRangeY;
                useWideSearchRescue = true;
                refreshTransformAssessment();
            } else {
                for (auto diagnostic : wideTransform.candidateDiagnostics) {
                    diagnostic.direction = "wide:" + diagnostic.direction;
                    transform.candidateDiagnostics.push_back(std::move(diagnostic));
                }
            }
        }
        const bool needsQualityRescan =
            suspiciousTransform ||
            std::abs(transform.da) > std::max(std::abs(config.rotationSearchMinDeg),
                                              std::abs(config.rotationSearchMaxDeg)) + 1e-9;
        if (needsQualityRescan && (hasLastReliableTransform || trajectoryPrior.ok)) {
            double localSearchRangeX = 0.0;
            double localSearchRangeY = 0.0;
            const double localPriorX = hasLastReliableTransform ? lastReliableTransform.dx : pairPriorX;
            const double localPriorY = hasLastReliableTransform ? lastReliableTransform.dy : pairPriorY;
            const double localPriorAngle = hasLastReliableTransform ? lastReliableTransform.da : pairPriorAngle;
            const double localRotationHalfRangeDeg =
                trajectoryPrior.ok ? std::clamp(trajectoryPrior.angleWindowDeg * 1.35, 0.05, 0.12) : 0.10;
            const double localRotationMinDeg =
                std::max(config.rotationSearchMinDeg, localPriorAngle - localRotationHalfRangeDeg);
            const double localRotationMaxDeg =
                std::min(config.rotationSearchMaxDeg, localPriorAngle + localRotationHalfRangeDeg);
            const double localRotationStepDeg = std::min(config.rotationSearchStepDeg, 0.01);
            const double localPrimaryHalfRangePx =
                trajectoryPrior.ok ? std::clamp(trajectoryPrior.primaryWindowPx * 1.6, 18.0, 32.0)
                                   : (hasLastReliableTransform ? 28.0 : 36.0);
            const double localPerpHalfRangePx =
                trajectoryPrior.ok ? std::clamp(trajectoryPrior.perpWindowPx * 1.8, 6.0, 12.0)
                                   : (hasLastReliableTransform ? 10.0 : 14.0);
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
                             localPrimaryHalfRangePx,
                             localPerpHalfRangePx,
                             localSearchRangeX, localSearchRangeY);
            const double localNormal = normalRmseFor(localTransform);
            const double localCoverage = preferredCoverageRatio(localTransform);
            const double localSelectionCost =
                transformSelectionCost(localTransform, localPriorX, localPriorY, targetNormalRmsePx);
            if (localCoverage + 0.04 >= currentCoverage &&
                (localNormal + 0.015 < currentNormal ||
                 localSelectionCost < currentSelectionCost * 0.96)) {
                transform = localTransform;
                searchRangeX = localSearchRangeX;
                searchRangeY = localSearchRangeY;
                useQualityLocalRescan = true;
                refreshTransformAssessment();
            } else {
                for (auto diagnostic : localTransform.candidateDiagnostics) {
                    diagnostic.direction = "local:" + diagnostic.direction;
                    transform.candidateDiagnostics.push_back(std::move(diagnostic));
                }
            }
        }
        if (suspiciousTransform && trajectoryPrior.ok) {
            double trajectorySearchRangeX = 0.0;
            double trajectorySearchRangeY = 0.0;
            double trajectoryRotationMinDeg =
                std::max(config.rotationSearchMinDeg, trajectoryPrior.angleDeg - trajectoryPrior.angleWindowDeg);
            double trajectoryRotationMaxDeg =
                std::min(config.rotationSearchMaxDeg, trajectoryPrior.angleDeg + trajectoryPrior.angleWindowDeg);
            if (trajectoryRotationMinDeg > trajectoryRotationMaxDeg) {
                trajectoryRotationMinDeg = config.rotationSearchMinDeg;
                trajectoryRotationMaxDeg = config.rotationSearchMaxDeg;
            }

            TransformResult trajectoryTransform =
                matchOnePair(previousEdges, nextEdges, center,
                             trajectoryPrior.dx, trajectoryPrior.dy,
                             trajectoryPrior.angleDeg, true,
                             config.baseSearchRange,
                             stepDirection,
                             trajectoryRotationMinDeg,
                             trajectoryRotationMaxDeg,
                             std::min(config.rotationSearchStepDeg, 0.01),
                             config.tangentResidualCostWeight,
                             config.tangentCorrelationCostWeight,
                             true,
                             trajectoryPrior.primaryWindowPx,
                             trajectoryPrior.perpWindowPx,
                             trajectorySearchRangeX, trajectorySearchRangeY);

            const double trajectoryNormal = normalRmseFor(trajectoryTransform);
            const double trajectoryCoverage = preferredCoverageRatio(trajectoryTransform);
            const double trajectorySelectionCost =
                transformSelectionCost(trajectoryTransform, trajectoryPrior.dx, trajectoryPrior.dy, targetNormalRmsePx);
            const double currentContinuity =
                continuityDistanceToPrior(transform, trajectoryPrior.dx, trajectoryPrior.dy);
            const double trajectoryContinuity =
                continuityDistanceToPrior(trajectoryTransform, trajectoryPrior.dx, trajectoryPrior.dy);
            const bool currentBoundaryHit =
                touchesRotationBoundary(transform.da, pairRotationMinDeg, pairRotationMaxDeg);

            if (trajectoryTransform.hasCandidate() &&
                std::isfinite(trajectoryNormal) &&
                trajectoryCoverage + 0.02 >= currentCoverage &&
                (trajectoryNormal + 0.015 < currentNormal ||
                 trajectorySelectionCost < currentSelectionCost * 0.85 ||
                 (currentBoundaryHit &&
                  trajectoryNormal < currentNormal * 0.90 &&
                  trajectoryContinuity + 10.0 < currentContinuity) ||
                 (currentContinuity > trajectoryPrior.primaryWindowPx &&
                  trajectoryContinuity + 12.0 < currentContinuity &&
                  trajectoryNormal <= std::max(0.25, currentNormal * 1.02)))) {
                transform = trajectoryTransform;
                searchRangeX = trajectorySearchRangeX;
                searchRangeY = trajectorySearchRangeY;
                useTrajectoryPriorRescue = true;
                refreshTransformAssessment();
            }
        }
        if (suspiciousTransform && previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
            const EdgeVariants previousUnfilteredEdges = buildMatchingEdgeView(previousEdges, true);
            const EdgeVariants nextUnfilteredEdges = buildMatchingEdgeView(nextEdges, true);
            double rescueSearchRangeX = 0.0;
            double rescueSearchRangeY = 0.0;
            const double rescueBaseSearchRange =
                shouldTryWideSearchRescue
                    ? (hasFixedDirectionPrior ? std::max(config.baseSearchRange, 220.0)
                                              : std::max(config.baseSearchRange, 3000.0))
                    : config.baseSearchRange;
            const double rescueRotationMinDeg =
                shouldTryWideSearchRescue && !hasFixedDirectionPrior
                    ? std::min(config.rotationSearchMinDeg, -1.0)
                    : config.rotationSearchMinDeg;
            const double rescueRotationMaxDeg =
                shouldTryWideSearchRescue && !hasFixedDirectionPrior
                    ? std::max(config.rotationSearchMaxDeg, 1.0)
                    : config.rotationSearchMaxDeg;
            const double rescueRotationStepDeg =
                shouldTryWideSearchRescue && !hasFixedDirectionPrior
                    ? std::max(config.rotationSearchStepDeg, 0.05)
                    : config.rotationSearchStepDeg;
            TransformResult unfilteredTransform =
                matchOnePair(previousUnfilteredEdges, nextUnfilteredEdges, center,
                             pairPriorX, pairPriorY,
                             pairPriorAngle, hasReliableMotionPrior || useTrajectoryLocalFirstPass,
                             rescueBaseSearchRange,
                             stepDirection,
                             rescueRotationMinDeg,
                             rescueRotationMaxDeg,
                             rescueRotationStepDeg,
                             config.tangentResidualCostWeight,
                             config.tangentCorrelationCostWeight,
                             false,
                             0.0,
                             0.0,
                             rescueSearchRangeX, rescueSearchRangeY);
            const double unfilteredNormal = normalRmseFor(unfilteredTransform);
            const double unfilteredCoverage = preferredCoverageRatio(unfilteredTransform);
            const double unfilteredSelectionCost =
                transformSelectionCost(unfilteredTransform, pairPriorX, pairPriorY, targetNormalRmsePx);
            if (unfilteredCoverage + 0.07 >= currentCoverage &&
                (unfilteredNormal + 0.015 < currentNormal ||
                 unfilteredSelectionCost < currentSelectionCost * 0.95 ||
                 unfilteredNormal < currentNormal * 0.85)) {
                transform = unfilteredTransform;
                searchRangeX = rescueSearchRangeX;
                searchRangeY = rescueSearchRangeY;
                useUnfilteredEdgeRescue = true;
                refreshTransformAssessment();
            } else {
                for (auto diagnostic : unfilteredTransform.candidateDiagnostics) {
                    diagnostic.direction = "unfiltered:" + diagnostic.direction;
                    transform.candidateDiagnostics.push_back(std::move(diagnostic));
                }
            }
        }
        if ((config.enableBadStepCandidateReselect || suspiciousTransform) &&
            reselectBadStepCandidate(transform, pairPriorX, pairPriorY, targetNormalRmsePx)) {
            useCandidateReselect = true;
            refreshTransformAssessment();
        }
        if (suspiciousTransform) {
            const bool primaryIsX = directionUsesPrimaryX(stepDirection);
            const double nominalPrimaryShiftPx = primaryIsX ? approxInitX : approxInitY;
            const double primaryPriorShiftPx = primaryIsX ? pairPriorX : pairPriorY;
            const double secondaryPriorShiftPx = primaryIsX ? pairPriorY : pairPriorX;
            imageCorrelationEstimate =
                estimateImageCorrelationShift(images[i],
                                              images[i + 1],
                                              stepDirection,
                                              primaryPriorShiftPx,
                                              secondaryPriorShiftPx,
                                              nominalPrimaryShiftPx);
            if (imageCorrelationEstimate.ok) {
                const double imageDeltaToCurrent =
                    continuityDistanceToPrior(transform, imageCorrelationEstimate.dx, imageCorrelationEstimate.dy);
                if (imageDeltaToCurrent > 28.0) {
                    double imageSearchRangeX = 0.0;
                    double imageSearchRangeY = 0.0;
                    const double imageRotationHalfRangeDeg = 0.15;
                    const double imageRotationMinDeg =
                        std::max(config.rotationSearchMinDeg, pairPriorAngle - imageRotationHalfRangeDeg);
                    const double imageRotationMaxDeg =
                        std::min(config.rotationSearchMaxDeg, pairPriorAngle + imageRotationHalfRangeDeg);
                    TransformResult imageTransform =
                        matchOnePair(previousEdges, nextEdges, center,
                                     imageCorrelationEstimate.dx, imageCorrelationEstimate.dy,
                                     pairPriorAngle,
                                     false,
                                     config.baseSearchRange,
                                     stepDirection,
                                     imageRotationMinDeg,
                                     imageRotationMaxDeg,
                                     std::min(config.rotationSearchStepDeg, 0.01),
                                     config.tangentResidualCostWeight,
                                     config.tangentCorrelationCostWeight,
                                     true,
                                     120.0,
                                     60.0,
                                     imageSearchRangeX, imageSearchRangeY);
                    const double imageNormal = normalRmseFor(imageTransform);
                    const double imageCoverage = preferredCoverageRatio(imageTransform);
                    const double imageSelectionCost =
                        transformSelectionCost(imageTransform,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    if (imageTransform.hasCandidate() &&
                        imageCoverage + 0.02 >= currentCoverage &&
                        (imageNormal + 0.02 < currentNormal ||
                         imageSelectionCost < currentSelectionCost * 0.92 ||
                         currentNormal > 0.45)) {
                        transform = imageTransform;
                        searchRangeX = imageSearchRangeX;
                        searchRangeY = imageSearchRangeY;
                        useImageCorrelationRescue = true;
                        refreshTransformAssessment();
                    }
                }
            }
        }
        const bool isFirstStep = i == 0;
        const bool isLastStep = i + 2 == images.size();
        const bool isEndpointStep = isFirstStep || isLastStep;
        constexpr double kMidStepPriorImageRefineThresholdPx = 0.11;
        constexpr double kMidStepPriorGridRefineThresholdPx = 0.18;
        constexpr double kMidStepPriorAffineRefineThresholdPx = 0.26;
        if (!isEndpointStep &&
            currentNormal > kMidStepPriorImageRefineThresholdPx &&
            (suspiciousTransform ||
             currentNormal > std::max(0.13, targetNormalRmsePx + 0.025)) &&
            (hasLastReliableTransform || trajectoryPrior.ok || imageCorrelationEstimate.ok)) {
            ImageCorrelationShiftEstimate refinementGuide = imageCorrelationEstimate;
            if (!refinementGuide.ok) {
                refinementGuide.ok = true;
                refinementGuide.dx =
                    trajectoryPrior.ok ? trajectoryPrior.dx
                                       : (hasLastReliableTransform ? lastReliableTransform.dx : pairPriorX);
                refinementGuide.dy =
                    trajectoryPrior.ok ? trajectoryPrior.dy
                                       : (hasLastReliableTransform ? lastReliableTransform.dy : pairPriorY);
                refinementGuide.score = 1.0;
            }

            const auto preferRefinedCandidate =
                [&](TransformResult preferred, TransformResult alternative) {
                    const double preferredNormal = normalRmseFor(preferred);
                    const double alternativeNormal = normalRmseFor(alternative);
                    if (!alternative.hasCandidate() || !std::isfinite(alternativeNormal)) {
                        return preferred;
                    }
                    if (!preferred.hasCandidate() || !std::isfinite(preferredNormal)) {
                        return alternative;
                    }
                    const double preferredCoverage = preferredCoverageRatio(preferred);
                    const double alternativeCoverage = preferredCoverageRatio(alternative);
                    const double preferredCost =
                        transformSelectionCost(preferred,
                                               refinementGuide.dx,
                                               refinementGuide.dy,
                                               targetNormalRmsePx);
                    const double alternativeCost =
                        transformSelectionCost(alternative,
                                               refinementGuide.dx,
                                               refinementGuide.dy,
                                               targetNormalRmsePx);
                    if (alternativeNormal + 0.001 < preferredNormal ||
                        (std::abs(alternativeNormal - preferredNormal) <= 0.001 &&
                         (alternativeCoverage > preferredCoverage + 0.005 ||
                          alternativeCost < preferredCost * 0.995))) {
                        return alternative;
                    }
                    return preferred;
                };

            const auto maybeAcceptPriorRefine =
                [&](TransformResult candidate, bool& acceptedFlag) {
                    const double candidateNormal = normalRmseFor(candidate);
                    if (!candidate.hasCandidate() || !std::isfinite(candidateNormal)) {
                        return;
                    }
                    const double baseCorrelation = preferredTangentCorrelation(transform);
                    const double candidateCorrelation = preferredTangentCorrelation(candidate);
                    if (candidateCorrelation + 0.02 < baseCorrelation) {
                        return;
                    }
                    const double baseCoverage = preferredCoverageRatio(transform);
                    const double candidateCoverage = preferredCoverageRatio(candidate);
                    if (candidateCoverage + 0.05 < baseCoverage && candidateCoverage < 0.60) {
                        return;
                    }
                    const double candidateCost =
                        transformSelectionCost(candidate,
                                               refinementGuide.dx,
                                               refinementGuide.dy,
                                               targetNormalRmsePx);
                    if (candidateNormal + 0.002 < currentNormal ||
                        candidateCost < currentSelectionCost * 0.99 ||
                        candidateNormal < 0.1) {
                        transform = std::move(candidate);
                        acceptedFlag = true;
                        refreshTransformAssessment();
                    }
                };

            if (imageCorrelationEstimate.ok &&
                currentNormal > kMidStepPriorImageRefineThresholdPx) {
                const bool useAggressivePriorImageSearch =
                    currentNormal > 0.16;
                TransformResult priorImageRefined =
                    refineEndpointStepWithImageGuidedNormalSearch(previousEdges,
                                                                  nextEdges,
                                                                  center,
                                                                  stepDirection,
                                                                  config,
                                                                  transform,
                                                                  imageCorrelationEstimate,
                                                                  targetNormalRmsePx,
                                                                  false,
                                                                  useAggressivePriorImageSearch);
                if (currentNormal > 0.15 &&
                    previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                    TransformResult priorImageUnfilteredRefined =
                        refineEndpointStepWithImageGuidedNormalSearch(previousEdges,
                                                                      nextEdges,
                                                                      center,
                                                                      stepDirection,
                                                                      config,
                                                                      transform,
                                                                      imageCorrelationEstimate,
                                                                      targetNormalRmsePx,
                                                                      true,
                                                                      useAggressivePriorImageSearch);
                    priorImageRefined =
                        preferRefinedCandidate(std::move(priorImageRefined),
                                               std::move(priorImageUnfilteredRefined));
                }
                maybeAcceptPriorRefine(std::move(priorImageRefined), usePriorImageGuidedRefine);
            }

            if (currentNormal > kMidStepPriorGridRefineThresholdPx) {
                TransformResult priorGridRefined =
                    refineLastEndpointStepWithDirectNormalGridSearch(previousEdges,
                                                                     nextEdges,
                                                                     center,
                                                                     stepDirection,
                                                                     config,
                                                                     transform,
                                                                     refinementGuide,
                                                                     targetNormalRmsePx,
                                                                     false);
                if (currentNormal > 0.22 &&
                    previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                    TransformResult priorGridUnfilteredRefined =
                        refineLastEndpointStepWithDirectNormalGridSearch(previousEdges,
                                                                         nextEdges,
                                                                         center,
                                                                         stepDirection,
                                                                         config,
                                                                         transform,
                                                                         refinementGuide,
                                                                         targetNormalRmsePx,
                                                                         true);
                    priorGridRefined =
                        preferRefinedCandidate(std::move(priorGridRefined),
                                               std::move(priorGridUnfilteredRefined));
                }
                maybeAcceptPriorRefine(std::move(priorGridRefined), usePriorNormalGridRefine);
            }

            if (currentNormal > kMidStepPriorAffineRefineThresholdPx) {
                TransformResult priorAffineRefined =
                    refineLastEndpointStepWithAffineShearSearch(previousEdges,
                                                                nextEdges,
                                                                center,
                                                                stepDirection,
                                                                config,
                                                                transform,
                                                                refinementGuide,
                                                                targetNormalRmsePx,
                                                                false);
                if (currentNormal > 0.22 &&
                    previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                    TransformResult priorAffineUnfilteredRefined =
                        refineLastEndpointStepWithAffineShearSearch(previousEdges,
                                                                    nextEdges,
                                                                    center,
                                                                    stepDirection,
                                                                    config,
                                                                    transform,
                                                                    refinementGuide,
                                                                    targetNormalRmsePx,
                                                                    true);
                    priorAffineRefined =
                        preferRefinedCandidate(std::move(priorAffineRefined),
                                               std::move(priorAffineUnfilteredRefined));
                }
                maybeAcceptPriorRefine(std::move(priorAffineRefined), usePriorAffineShearRefine);
            }
        }
        const double endpointImageRefineThresholdPx =
            useImageCorrelationLocalFirstPass ? 0.14 : 0.10;
        if (isEndpointStep &&
            imageCorrelationEstimate.ok &&
            currentNormal > endpointImageRefineThresholdPx) {
            const bool useAggressiveEndpointSearch =
                isLastStep && !config.endpointProbeFastMode;
            TransformResult endpointRefined =
                refineEndpointStepWithImageGuidedNormalSearch(previousEdges,
                                                              nextEdges,
                                                              center,
                                                              stepDirection,
                                                              config,
                                                              transform,
                                                              imageCorrelationEstimate,
                                                              targetNormalRmsePx,
                                                              false,
                                                              useAggressiveEndpointSearch);
            const double filteredEndpointNormal = normalRmseFor(endpointRefined);
            if (!config.endpointProbeFastMode &&
                previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered() &&
                (!std::isfinite(filteredEndpointNormal) ||
                 filteredEndpointNormal > 0.12 ||
                 currentNormal > 0.14)) {
                TransformResult endpointUnfilteredRefined =
                    refineEndpointStepWithImageGuidedNormalSearch(previousEdges,
                                                                  nextEdges,
                                                                  center,
                                                                  stepDirection,
                                                                  config,
                                                                  transform,
                                                                  imageCorrelationEstimate,
                                                                  targetNormalRmsePx,
                                                                  true,
                                                                  useAggressiveEndpointSearch);
                const double filteredNormal = normalRmseFor(endpointRefined);
                const double unfilteredNormal = normalRmseFor(endpointUnfilteredRefined);
                const double filteredCoverage = preferredCoverageRatio(endpointRefined);
                const double unfilteredCoverage = preferredCoverageRatio(endpointUnfilteredRefined);
                const double filteredCost =
                    transformSelectionCost(endpointRefined,
                                           imageCorrelationEstimate.dx,
                                           imageCorrelationEstimate.dy,
                                           targetNormalRmsePx);
                const double unfilteredCost =
                    transformSelectionCost(endpointUnfilteredRefined,
                                           imageCorrelationEstimate.dx,
                                           imageCorrelationEstimate.dy,
                                           targetNormalRmsePx);
                if (endpointUnfilteredRefined.hasCandidate() &&
                    std::isfinite(unfilteredNormal) &&
                    (!std::isfinite(filteredNormal) ||
                     unfilteredNormal + 0.0015 < filteredNormal ||
                     (std::abs(unfilteredNormal - filteredNormal) <= 0.0015 &&
                      (unfilteredCoverage > filteredCoverage + 0.01 ||
                       unfilteredCost < filteredCost * 0.995)))) {
                    endpointRefined = std::move(endpointUnfilteredRefined);
                }
            }
            if (isLastStep) {
                TransformResult endpointTailTrimRefined =
                    refineLastEndpointStepWithTailTrimSearch(previousEdges,
                                                             nextEdges,
                                                             center,
                                                             stepDirection,
                                                             config,
                                                             endpointRefined,
                                                             imageCorrelationEstimate,
                                                             targetNormalRmsePx,
                                                             false);
                if (!config.endpointProbeFastMode &&
                    previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                    TransformResult endpointTailTrimUnfilteredRefined =
                        refineLastEndpointStepWithTailTrimSearch(previousEdges,
                                                                 nextEdges,
                                                                 center,
                                                                 stepDirection,
                                                                 config,
                                                                 endpointRefined,
                                                                 imageCorrelationEstimate,
                                                                 targetNormalRmsePx,
                                                                 true);
                    const double filteredTailTrimNormal = normalRmseFor(endpointTailTrimRefined);
                    const double unfilteredTailTrimNormal = normalRmseFor(endpointTailTrimUnfilteredRefined);
                    const double filteredTailTrimCoverage = preferredCoverageRatio(endpointTailTrimRefined);
                    const double unfilteredTailTrimCoverage = preferredCoverageRatio(endpointTailTrimUnfilteredRefined);
                    const double filteredTailTrimCost =
                        transformSelectionCost(endpointTailTrimRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    const double unfilteredTailTrimCost =
                        transformSelectionCost(endpointTailTrimUnfilteredRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    if (endpointTailTrimUnfilteredRefined.hasCandidate() &&
                        std::isfinite(unfilteredTailTrimNormal) &&
                        (!std::isfinite(filteredTailTrimNormal) ||
                         unfilteredTailTrimNormal + 0.0015 < filteredTailTrimNormal ||
                         (std::abs(unfilteredTailTrimNormal - filteredTailTrimNormal) <= 0.0015 &&
                          (unfilteredTailTrimCoverage > filteredTailTrimCoverage + 0.01 ||
                           unfilteredTailTrimCost < filteredTailTrimCost * 0.995)))) {
                        endpointTailTrimRefined = std::move(endpointTailTrimUnfilteredRefined);
                    }
                }

                const double endpointBaseNormal = normalRmseFor(endpointRefined);
                const double endpointBaseCoverage = preferredCoverageRatio(endpointRefined);
                const double endpointBaseCorrelation = preferredTangentCorrelation(endpointRefined);
                const double endpointBaseCost =
                    transformSelectionCost(endpointRefined,
                                           imageCorrelationEstimate.dx,
                                           imageCorrelationEstimate.dy,
                                           targetNormalRmsePx);
                const double tailTrimNormal = normalRmseFor(endpointTailTrimRefined);
                const double tailTrimCoverage = preferredCoverageRatio(endpointTailTrimRefined);
                const double tailTrimCorrelation = preferredTangentCorrelation(endpointTailTrimRefined);
                const double tailTrimCost =
                    transformSelectionCost(endpointTailTrimRefined,
                                           imageCorrelationEstimate.dx,
                                           imageCorrelationEstimate.dy,
                                           targetNormalRmsePx);
                std::ostringstream tailTrimLog;
                tailTrimLog << "    [Info] Last-step tail-trim candidate normal="
                            << tailTrimNormal
                            << ", coverage=" << tailTrimCoverage
                            << ", corr=" << tailTrimCorrelation
                            << ", cost=" << tailTrimCost << ".";
                emitLog(callbacks, tailTrimLog.str());
                if (endpointTailTrimRefined.hasCandidate() &&
                    std::isfinite(tailTrimNormal) &&
                    tailTrimCorrelation + 0.02 >= endpointBaseCorrelation &&
                    (tailTrimCoverage >= 0.66 ||
                     tailTrimCoverage + 0.05 >= endpointBaseCoverage ||
                     (tailTrimCoverage + 0.04 >= endpointBaseCoverage &&
                      tailTrimNormal + 0.012 < endpointBaseNormal)) &&
                    (tailTrimNormal + 0.0015 < endpointBaseNormal ||
                     tailTrimCost < endpointBaseCost * 0.99 ||
                     tailTrimNormal < 0.1)) {
                    endpointRefined = std::move(endpointTailTrimRefined);
                    useLastStepTailTrimRefine = true;
                }

                if (config.enableLastStepReferencePriorHalfOverlapProbe) {
                    const TransformResult referencePriorTransform =
                        hasLastReliableTransform ? lastReliableTransform
                                                 : makeTranslationPriorFallback(pairPriorX,
                                                                                pairPriorY,
                                                                                pairPriorAngle,
                                                                                stepDirection);
                    double probeSearchRangeX = 0.0;
                    double probeSearchRangeY = 0.0;
                    double probeTrimFraction = 0.0;
                    TransformResult referenceHalfOverlapProbe =
                        refineLastEndpointStepWithReferencePriorHalfOverlapProbe(previousEdges,
                                                                                nextEdges,
                                                                                center,
                                                                                stepDirection,
                                                                                config,
                                                                                referencePriorTransform,
                                                                                probeSearchRangeX,
                                                                                probeSearchRangeY,
                                                                                &probeTrimFraction);
                    const double probeNormal = normalRmseFor(referenceHalfOverlapProbe);
                    const double probeCoverage = preferredCoverageRatio(referenceHalfOverlapProbe);
                    const double probeCorrelation = preferredTangentCorrelation(referenceHalfOverlapProbe);
                    const double probeCost =
                        transformSelectionCost(referenceHalfOverlapProbe,
                                               referencePriorTransform.dx,
                                               referencePriorTransform.dy,
                                               targetNormalRmsePx);
                    const double referenceBaseNormal = normalRmseFor(endpointRefined);
                    const double referenceBaseCorrelation = preferredTangentCorrelation(endpointRefined);
                    std::ostringstream referenceProbeLog;
                    referenceProbeLog << "    [Info] Last-step prev-prior visible-overlap candidate normal="
                                      << probeNormal
                                      << ", coverage=" << probeCoverage
                                      << ", corr=" << probeCorrelation
                                      << ", cost=" << probeCost
                                      << ", trim=" << probeTrimFraction
                                      << ", transform=(" << referenceHalfOverlapProbe.dx
                                      << "," << referenceHalfOverlapProbe.dy
                                      << "," << referenceHalfOverlapProbe.da << ")"
                                      << ", prior=(" << referencePriorTransform.dx
                                      << "," << referencePriorTransform.dy
                                      << "," << referencePriorTransform.da << ")"
                                      << ", window=(" << probeSearchRangeX
                                      << "," << probeSearchRangeY << ").";
                    emitLog(callbacks, referenceProbeLog.str());
                    const bool probeAcceptedByTrimmedRegionDriver =
                        referenceHalfOverlapProbe.hasCandidate() &&
                        std::isfinite(probeNormal) &&
                        probeCorrelation + 0.01 >= referenceBaseCorrelation &&
                        probeCoverage >= 0.20 &&
                        (probeNormal + 0.002 < referenceBaseNormal || probeNormal < 0.1);
                    if (referenceHalfOverlapProbe.hasCandidate() &&
                        probeAcceptedByTrimmedRegionDriver) {
                        endpointRefined = std::move(referenceHalfOverlapProbe);
                        searchRangeX = std::max(searchRangeX, probeSearchRangeX);
                        searchRangeY = std::max(searchRangeY, probeSearchRangeY);
                        candidateLastStepReferencePriorHalfOverlapProbe = true;
                        TransformResult referenceHalfOverlapFullAnchor;
                        TransformResult referenceHalfOverlapPullbackRaw;
                        TransformResult referenceHalfOverlapPullback =
                            refineLastEndpointStepWithReferencePriorHalfOverlapSoftPullback(
                                previousEdges,
                                nextEdges,
                                center,
                                stepDirection,
                                config,
                                endpointRefined,
                                imageCorrelationEstimate,
                                targetNormalRmsePx,
                                &referenceHalfOverlapFullAnchor,
                                &referenceHalfOverlapPullbackRaw);
                        if (referenceHalfOverlapFullAnchor.hasCandidate()) {
                            const double fullAnchorNormal =
                                normalRmseFor(referenceHalfOverlapFullAnchor);
                            const double fullAnchorCoverage =
                                preferredCoverageRatio(referenceHalfOverlapFullAnchor);
                            const double fullAnchorCorrelation =
                                preferredTangentCorrelation(referenceHalfOverlapFullAnchor);
                            const double fullAnchorCost =
                                transformSelectionCost(referenceHalfOverlapFullAnchor,
                                                       imageCorrelationEstimate.dx,
                                                       imageCorrelationEstimate.dy,
                                                       targetNormalRmsePx);
                            std::ostringstream fullAnchorLog;
                            fullAnchorLog
                                << "    [Info] Last-step prev-prior visible-overlap full-anchor normal="
                                << fullAnchorNormal
                                << ", coverage=" << fullAnchorCoverage
                                << ", corr=" << fullAnchorCorrelation
                                << ", cost=" << fullAnchorCost
                                << ", trim=" << probeTrimFraction
                                << ", transform=(" << referenceHalfOverlapFullAnchor.dx
                                << "," << referenceHalfOverlapFullAnchor.dy
                                << "," << referenceHalfOverlapFullAnchor.da << ").";
                            emitLog(callbacks, fullAnchorLog.str());
                        }
                        if (referenceHalfOverlapPullbackRaw.hasCandidate()) {
                            const double pullbackRawNormal =
                                normalRmseFor(referenceHalfOverlapPullbackRaw);
                            const double pullbackRawCoverage =
                                preferredCoverageRatio(referenceHalfOverlapPullbackRaw);
                            const double pullbackRawCorrelation =
                                preferredTangentCorrelation(referenceHalfOverlapPullbackRaw);
                            const double pullbackRawCost =
                                transformSelectionCost(referenceHalfOverlapPullbackRaw,
                                                       imageCorrelationEstimate.dx,
                                                       imageCorrelationEstimate.dy,
                                                       targetNormalRmsePx);
                            std::ostringstream pullbackRawLog;
                            pullbackRawLog
                                << "    [Info] Last-step prev-prior visible-overlap pullback-raw normal="
                                << pullbackRawNormal
                                << ", coverage=" << pullbackRawCoverage
                                << ", corr=" << pullbackRawCorrelation
                                << ", cost=" << pullbackRawCost
                                << ", trim=" << probeTrimFraction
                                << ", transform=(" << referenceHalfOverlapPullbackRaw.dx
                                << "," << referenceHalfOverlapPullbackRaw.dy
                                << "," << referenceHalfOverlapPullbackRaw.da << ").";
                            emitLog(callbacks, pullbackRawLog.str());
                        }
                        const bool pullbackAccepted =
                            referenceHalfOverlapPullback.hasCandidate() &&
                            (std::abs(referenceHalfOverlapPullback.dx - endpointRefined.dx) > 1e-6 ||
                             std::abs(referenceHalfOverlapPullback.dy - endpointRefined.dy) > 1e-6 ||
                             std::abs(referenceHalfOverlapPullback.da - endpointRefined.da) > 1e-9);
                        if (pullbackAccepted) {
                            const double pullbackNormal = normalRmseFor(referenceHalfOverlapPullback);
                            const double pullbackCoverage =
                                preferredCoverageRatio(referenceHalfOverlapPullback);
                            const double pullbackCorrelation =
                                preferredTangentCorrelation(referenceHalfOverlapPullback);
                            const double pullbackCost =
                                transformSelectionCost(referenceHalfOverlapPullback,
                                                       imageCorrelationEstimate.dx,
                                                       imageCorrelationEstimate.dy,
                                                       targetNormalRmsePx);
                            std::ostringstream pullbackLog;
                            pullbackLog
                                << "    [Info] Last-step prev-prior visible-overlap pullback candidate normal="
                                << pullbackNormal
                                << ", coverage=" << pullbackCoverage
                                << ", corr=" << pullbackCorrelation
                                << ", cost=" << pullbackCost
                                << ", trim=" << probeTrimFraction
                                << ", transform=(" << referenceHalfOverlapPullback.dx
                                << "," << referenceHalfOverlapPullback.dy
                                << "," << referenceHalfOverlapPullback.da << ").";
                            emitLog(callbacks, pullbackLog.str());
                            endpointRefined = std::move(referenceHalfOverlapPullback);
                            useLastStepReferencePriorHalfOverlapPullback = true;
                        }
                    }
                }

                if (!config.endpointProbeFastMode) {
                    TransformResult endpointGridRefined =
                        refineLastEndpointStepWithDirectNormalGridSearch(previousEdges,
                                                                         nextEdges,
                                                                         center,
                                                                         stepDirection,
                                                                         config,
                                                                         endpointRefined,
                                                                         imageCorrelationEstimate,
                                                                         targetNormalRmsePx,
                                                                         false);
                    if (previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                        TransformResult endpointGridUnfilteredRefined =
                            refineLastEndpointStepWithDirectNormalGridSearch(previousEdges,
                                                                             nextEdges,
                                                                             center,
                                                                             stepDirection,
                                                                             config,
                                                                             endpointRefined,
                                                                             imageCorrelationEstimate,
                                                                             targetNormalRmsePx,
                                                                             true);
                        const double filteredGridNormal = normalRmseFor(endpointGridRefined);
                        const double unfilteredGridNormal = normalRmseFor(endpointGridUnfilteredRefined);
                        const double filteredGridCoverage = preferredCoverageRatio(endpointGridRefined);
                        const double unfilteredGridCoverage = preferredCoverageRatio(endpointGridUnfilteredRefined);
                        const double filteredGridCost =
                            transformSelectionCost(endpointGridRefined,
                                                   imageCorrelationEstimate.dx,
                                                   imageCorrelationEstimate.dy,
                                                   targetNormalRmsePx);
                        const double unfilteredGridCost =
                            transformSelectionCost(endpointGridUnfilteredRefined,
                                                   imageCorrelationEstimate.dx,
                                                   imageCorrelationEstimate.dy,
                                                   targetNormalRmsePx);
                        if (endpointGridUnfilteredRefined.hasCandidate() &&
                            std::isfinite(unfilteredGridNormal) &&
                            (!std::isfinite(filteredGridNormal) ||
                             unfilteredGridNormal + 0.001 < filteredGridNormal ||
                             (std::abs(unfilteredGridNormal - filteredGridNormal) <= 0.001 &&
                              (unfilteredGridCoverage > filteredGridCoverage + 0.005 ||
                               unfilteredGridCost < filteredGridCost * 0.995)))) {
                            endpointGridRefined = std::move(endpointGridUnfilteredRefined);
                        }
                    }

                    const double endpointGridBaseNormal = normalRmseFor(endpointRefined);
                    const double endpointGridBaseCoverage = preferredCoverageRatio(endpointRefined);
                    const double endpointGridBaseCorrelation = preferredTangentCorrelation(endpointRefined);
                    const double endpointGridBaseCost =
                        transformSelectionCost(endpointRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    const double gridNormal = normalRmseFor(endpointGridRefined);
                    const double gridCoverage = preferredCoverageRatio(endpointGridRefined);
                    const double gridCorrelation = preferredTangentCorrelation(endpointGridRefined);
                    const double gridCost =
                        transformSelectionCost(endpointGridRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    std::ostringstream gridRefineLog;
                    gridRefineLog << "    [Info] Last-step direct-grid candidate normal="
                                  << gridNormal
                                  << ", coverage=" << gridCoverage
                                  << ", corr=" << gridCorrelation
                                  << ", cost=" << gridCost << ".";
                    emitLog(callbacks, gridRefineLog.str());
                    if (endpointGridRefined.hasCandidate() &&
                        std::isfinite(gridNormal) &&
                        gridCorrelation + 0.03 >= endpointGridBaseCorrelation &&
                        (gridCoverage >= 0.66 || gridCoverage + 0.06 >= endpointGridBaseCoverage) &&
                        (gridNormal + 0.001 < endpointGridBaseNormal ||
                         gridCost < endpointGridBaseCost * 0.99 ||
                         gridNormal < 0.1)) {
                        endpointRefined = std::move(endpointGridRefined);
                        useLastStepNormalGridRefine = true;
                    }

                    TransformResult endpointAffineRefined =
                        refineLastEndpointStepWithAffineShearSearch(previousEdges,
                                                                    nextEdges,
                                                                    center,
                                                                    stepDirection,
                                                                    config,
                                                                    endpointRefined,
                                                                    imageCorrelationEstimate,
                                                                    targetNormalRmsePx,
                                                                    false);
                    if (previousEdges.hasUnfiltered() && nextEdges.hasUnfiltered()) {
                        TransformResult endpointAffineUnfilteredRefined =
                            refineLastEndpointStepWithAffineShearSearch(previousEdges,
                                                                        nextEdges,
                                                                        center,
                                                                        stepDirection,
                                                                        config,
                                                                        endpointRefined,
                                                                        imageCorrelationEstimate,
                                                                        targetNormalRmsePx,
                                                                        true);
                        const double filteredAffineNormal = normalRmseFor(endpointAffineRefined);
                        const double unfilteredAffineNormal = normalRmseFor(endpointAffineUnfilteredRefined);
                        const double filteredAffineCoverage = preferredCoverageRatio(endpointAffineRefined);
                        const double unfilteredAffineCoverage = preferredCoverageRatio(endpointAffineUnfilteredRefined);
                        const double filteredAffineCost =
                            transformSelectionCost(endpointAffineRefined,
                                                   imageCorrelationEstimate.dx,
                                                   imageCorrelationEstimate.dy,
                                                   targetNormalRmsePx);
                        const double unfilteredAffineCost =
                            transformSelectionCost(endpointAffineUnfilteredRefined,
                                                   imageCorrelationEstimate.dx,
                                                   imageCorrelationEstimate.dy,
                                                   targetNormalRmsePx);
                        if (endpointAffineUnfilteredRefined.hasCandidate() &&
                            std::isfinite(unfilteredAffineNormal) &&
                            (!std::isfinite(filteredAffineNormal) ||
                             unfilteredAffineNormal + 0.001 < filteredAffineNormal ||
                             (std::abs(unfilteredAffineNormal - filteredAffineNormal) <= 0.001 &&
                              (unfilteredAffineCoverage > filteredAffineCoverage + 0.005 ||
                               unfilteredAffineCost < filteredAffineCost * 0.995)))) {
                            endpointAffineRefined = std::move(endpointAffineUnfilteredRefined);
                        }
                    }

                    const double endpointAffineBaseNormal = normalRmseFor(endpointRefined);
                    const double endpointAffineBaseCoverage = preferredCoverageRatio(endpointRefined);
                    const double endpointAffineBaseCorrelation = preferredTangentCorrelation(endpointRefined);
                    const double endpointAffineBaseCost =
                        transformSelectionCost(endpointRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    const double affineNormal = normalRmseFor(endpointAffineRefined);
                    const double affineCoverage = preferredCoverageRatio(endpointAffineRefined);
                    const double affineCorrelation = preferredTangentCorrelation(endpointAffineRefined);
                    const double affineCost =
                        transformSelectionCost(endpointAffineRefined,
                                               imageCorrelationEstimate.dx,
                                               imageCorrelationEstimate.dy,
                                               targetNormalRmsePx);
                    std::ostringstream affineRefineLog;
                    affineRefineLog << "    [Info] Last-step affine-shear candidate normal="
                                    << affineNormal
                                    << ", coverage=" << affineCoverage
                                    << ", corr=" << affineCorrelation
                                    << ", cost=" << affineCost
                                    << ", shear=" << endpointAffineRefined.shearPrimaryToNormal << ".";
                    emitLog(callbacks, affineRefineLog.str());
                    if (endpointAffineRefined.hasCandidate() &&
                        std::isfinite(affineNormal) &&
                        affineCorrelation + 0.03 >= endpointAffineBaseCorrelation &&
                        (affineCoverage >= 0.66 || affineCoverage + 0.06 >= endpointAffineBaseCoverage) &&
                        (affineNormal + 0.001 < endpointAffineBaseNormal ||
                         affineCost < endpointAffineBaseCost * 0.99 ||
                         affineNormal < 0.1)) {
                        endpointRefined = std::move(endpointAffineRefined);
                        useLastStepAffineShearRefine = true;
                    }
                }
            }

            const double endpointNormal = normalRmseFor(endpointRefined);
            const double endpointCoverage = preferredCoverageRatio(endpointRefined);
            const double endpointCorrelation = preferredTangentCorrelation(endpointRefined);
            const double endpointCost =
                transformSelectionCost(endpointRefined,
                                       imageCorrelationEstimate.dx,
                                       imageCorrelationEstimate.dy,
                                       targetNormalRmsePx);
            const bool endpointAcceptedByDefaultGate =
                endpointRefined.hasCandidate() &&
                std::isfinite(endpointNormal) &&
                endpointCoverage + 0.01 >= currentCoverage &&
                endpointCorrelation + 0.01 >= preferredTangentCorrelation(transform) &&
                (endpointNormal + 0.001 < currentNormal ||
                 endpointCost < currentSelectionCost * 0.99 ||
                 endpointNormal < 0.1);
            const bool endpointAcceptedByRef50Driver =
                candidateLastStepReferencePriorHalfOverlapProbe &&
                endpointRefined.hasCandidate() &&
                std::isfinite(endpointNormal) &&
                endpointCorrelation + 0.01 >= preferredTangentCorrelation(transform) &&
                endpointCoverage >= 0.20 &&
                (endpointNormal + 0.002 < currentNormal || endpointNormal < 0.1);
            if (endpointAcceptedByDefaultGate || endpointAcceptedByRef50Driver) {
                transform = std::move(endpointRefined);
                useLastStepReferencePriorHalfOverlapProbe =
                    candidateLastStepReferencePriorHalfOverlapProbe;
                useLastStepReferencePriorHalfOverlapPullback =
                    useLastStepReferencePriorHalfOverlapProbe &&
                    useLastStepReferencePriorHalfOverlapPullback;
                useEndpointImageGuidedRefine =
                    !candidateLastStepReferencePriorHalfOverlapProbe;
                refreshTransformAssessment();
            }
        }
        const double trajectoryContinuity =
            trajectoryPrior.ok ? continuityDistanceToPrior(transform, trajectoryPrior.dx, trajectoryPrior.dy) : 0.0;
        const bool shouldForceTrajectoryPriorFallback =
            trajectoryPrior.ok &&
            suspiciousTransform &&
            currentNormal > 0.45 &&
            trajectoryContinuity > std::max(60.0, trajectoryPrior.primaryWindowPx * 1.8);
        if (shouldUseTranslationPriorFallback(transform, config) || shouldForceTrajectoryPriorFallback) {
            const double fallbackPriorX = trajectoryPrior.ok ? trajectoryPrior.dx : pairPriorX;
            const double fallbackPriorY = trajectoryPrior.ok ? trajectoryPrior.dy : pairPriorY;
            const double fallbackPriorAngle = trajectoryPrior.ok ? trajectoryPrior.angleDeg : pairPriorAngle;
            TransformResult bestFallback = makeTranslationPriorFallback(fallbackPriorX,
                                                                        fallbackPriorY,
                                                                        fallbackPriorAngle,
                                                                        stepDirection);
            double bestFallbackCoverage = -1.0;
            double bestFallbackCost = std::numeric_limits<double>::infinity();
            const double fallbackReferenceX = trajectoryPrior.ok ? trajectoryPrior.dx : pairPriorX;
            const double fallbackReferenceY = trajectoryPrior.ok ? trajectoryPrior.dy : pairPriorY;
            const auto considerFallbackCandidate =
                [&](double candidateDx, double candidateDy, double candidateAngleDeg) {
                    TransformResult candidate =
                        buildEvaluatedPriorTransform(previousEdges,
                                                    nextEdges,
                                                    center,
                                                    candidateDx,
                                                    candidateDy,
                                                    candidateAngleDeg,
                                                    stepDirection,
                                                    config.tangentResidualCostWeight,
                                                    config.tangentCorrelationCostWeight);
                    const double candidateCoverage = preferredCoverageRatio(candidate);
                    const double candidateNormal = preferredNormalRmse(candidate);
                    if (!std::isfinite(candidateNormal) || candidateCoverage < 0.55) {
                        return;
                    }
                    const double candidateCost =
                        transformSelectionCost(candidate,
                                               fallbackReferenceX,
                                               fallbackReferenceY,
                                               targetNormalRmsePx);
                    if (candidateCost + 1e-9 < bestFallbackCost ||
                        (std::abs(candidateCost - bestFallbackCost) <= 1e-9 &&
                         candidateCoverage > bestFallbackCoverage + 0.01)) {
                        bestFallback = std::move(candidate);
                        bestFallbackCoverage = candidateCoverage;
                        bestFallbackCost = candidateCost;
                    }
                };

            considerFallbackCandidate(fallbackPriorX, fallbackPriorY, fallbackPriorAngle);
            if (hasLastReliableTransform) {
                considerFallbackCandidate(lastReliableTransform.dx,
                                          lastReliableTransform.dy,
                                          lastReliableTransform.da);
            }
            if (imageCorrelationEstimate.ok) {
                considerFallbackCandidate(imageCorrelationEstimate.dx,
                                          imageCorrelationEstimate.dy,
                                          pairPriorAngle);
            }
            transform = std::move(bestFallback);
            searchRangeX = 0.0;
            searchRangeY = 0.0;
            usedMotionPriorFallback = true;
        }

        StitchStepRecord step;
        step.stepIndex = i + 1;
        step.referenceImageIndex = i;
        step.targetImageIndex = i + 1;
        step.motionAxis = directionUsesPrimaryX(stepDirection) ? AlignmentAxis::X : AlignmentAxis::Y;
        step.primaryImageSpanPx =
            step.motionAxis == AlignmentAxis::X ? static_cast<double>(images[i].cols)
                                                : static_cast<double>(images[i].rows);
        step.searchRangeX = searchRangeX;
        step.searchRangeY = searchRangeY;
        step.hasNominalPrior = true;
        step.nominalPriorDx = approxInitX;
        step.nominalPriorDy = approxInitY;
        step.hasPairPrior = true;
        step.pairPriorDx = pairPriorX;
        step.pairPriorDy = pairPriorY;
        step.pairPriorAngleDeg = pairPriorAngle;
        step.hasTrajectoryPrior = trajectoryPrior.ok;
        step.trajectoryPriorDx = trajectoryPrior.dx;
        step.trajectoryPriorDy = trajectoryPrior.dy;
        step.trajectoryPriorAngleDeg = trajectoryPrior.angleDeg;
        step.hasImageCorrelationEstimate = imageCorrelationEstimate.ok;
        step.imageCorrelationDx = imageCorrelationEstimate.dx;
        step.imageCorrelationDy = imageCorrelationEstimate.dy;
        step.imageCorrelationScore = imageCorrelationEstimate.score;
        step.usedWideSearchRescue = useWideSearchRescue;
        step.usedQualityLocalRescan = useQualityLocalRescan;
        step.usedCandidateReselect = useCandidateReselect;
            step.usedUnfilteredEdgeRescue = useUnfilteredEdgeRescue;
            step.usedTrajectoryPriorRescue = useTrajectoryPriorRescue;
            step.usedMotionPriorFallback = usedMotionPriorFallback;
            step.usedForcedTrajectoryPriorFallback = shouldForceTrajectoryPriorFallback && !useEndpointImageGuidedRefine;
        if (usedMotionPriorFallback) {
            step.selectionMode = shouldForceTrajectoryPriorFallback
                                     ? "trajectory_prior_clamp"
                                     : "translation_prior_fallback";
        } else if (useTrajectoryPriorRescue) {
            step.selectionMode = "trajectory_prior_rescue";
        } else if (useLastStepAffineShearRefine) {
            step.selectionMode = "endpoint_affine_shear_refine";
        } else if (useLastStepNormalGridRefine) {
            step.selectionMode = "endpoint_normal_grid_refine";
        } else if (useLastStepReferencePriorHalfOverlapPullback) {
            step.selectionMode = "endpoint_prev_prior_visible_overlap_pullback";
        } else if (useLastStepReferencePriorHalfOverlapProbe) {
            step.selectionMode = "endpoint_prev_prior_visible_overlap_probe";
        } else if (useLastStepTailTrimRefine) {
            step.selectionMode = "endpoint_tail_trim_refine";
        } else if (useEndpointImageGuidedRefine) {
            step.selectionMode = "endpoint_image_guided_refine";
        } else if (usePriorAffineShearRefine) {
            step.selectionMode = "prior_affine_shear_refine";
        } else if (usePriorNormalGridRefine) {
            step.selectionMode = "prior_normal_grid_refine";
        } else if (usePriorImageGuidedRefine) {
            step.selectionMode = "prior_image_guided_refine";
        } else if (useImageCorrelationRescue) {
            step.selectionMode = "image_correlation_rescue";
        } else if (useUnfilteredEdgeRescue) {
            step.selectionMode = "unfiltered_edge_rescue";
        } else if (useWideSearchRescue) {
            step.selectionMode = "wide_search_rescue";
        } else if (useQualityLocalRescan) {
            step.selectionMode = "local_prior_rescan";
        } else if (useCandidateReselect) {
            step.selectionMode = "candidate_reselect";
        }
        step.transform = transform;
        result.steps.push_back(step);
        if (useWideSearchRescue) {
            emitLog(callbacks, "    [Info] Suspicious step triggered wide-search rescue and adopted the lower-RMSE result.");
        }
        if (useUnfilteredEdgeRescue) {
            emitLog(callbacks, "    [Info] Suspicious step retried with unfiltered edge points and adopted the lower-RMSE result.");
        }
        if (useTrajectoryPriorRescue) {
            emitLog(callbacks, "    [Info] Suspicious step was pulled back by stable trajectory-prior rescue.");
        }
        if (imageCorrelationEstimate.ok && suspiciousTransform) {
            std::ostringstream imageEstimateLog;
            imageEstimateLog << "    [Info] Raw-image coarse shift suggests dx="
                             << imageCorrelationEstimate.dx
                             << ", dy=" << imageCorrelationEstimate.dy
                             << ", corr=" << imageCorrelationEstimate.score << ".";
            emitLog(callbacks, imageEstimateLog.str());
        }
        if (useImageCorrelationRescue) {
            emitLog(callbacks, "    [Info] Suspicious step retried around the raw-image coarse shift estimate.");
        }
        if (useLastStepTailTrimRefine) {
            emitLog(callbacks, "    [Info] Last endpoint step accepted a tail-trimmed image-guided refinement.");
        }
        if (useLastStepReferencePriorHalfOverlapProbe &&
            !useLastStepReferencePriorHalfOverlapPullback) {
            emitLog(callbacks, "    [Info] Last endpoint step accepted the previous-prior visible-overlap probe result.");
        }
        if (useLastStepReferencePriorHalfOverlapPullback) {
            emitLog(callbacks, "    [Info] Last endpoint step accepted the previous-prior visible-overlap probe with a full-view soft pullback.");
        }
        if (useLastStepNormalGridRefine) {
            emitLog(callbacks, "    [Info] Last endpoint step accepted a direct normal-RMSE grid refinement.");
        }
        if (useLastStepAffineShearRefine) {
            emitLog(callbacks, "    [Info] Last endpoint step accepted a tiny affine-shear refinement.");
        }
        if (useEndpointImageGuidedRefine) {
            emitLog(callbacks, "    [Info] Endpoint step accepted an image-guided local RMSE refinement.");
        }
        if (usePriorImageGuidedRefine) {
            emitLog(callbacks, "    [Info] Suspicious step accepted a prior/image-guided narrow normal-RMSE refinement.");
        }
        if (usePriorNormalGridRefine) {
            emitLog(callbacks, "    [Info] Suspicious step accepted a prior-guided direct normal grid refinement.");
        }
        if (usePriorAffineShearRefine) {
            emitLog(callbacks, "    [Info] Suspicious step accepted a prior-guided tiny affine-shear refinement.");
        }

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
                step);
            callbacks.onImage("debug_step", step.stepIndex, totalSteps, debugImage);
        }

        applyTransformAndBlend(images[i + 1], result.canvas, result.globalTransform, center, transform);
        result.imageTransforms.push_back(result.globalTransform.clone());
        previousEdges = nextEdges;

        // 这里用法向 RMSE 判断下一步初值是否可信，避免切向代价权重改变 score 尺度后误触发重置。
        const ResidualStatistics& normalForReliability =
            transform.metrics.normalInlier.valid() ? transform.metrics.normalInlier : transform.metrics.normalAll;
        const bool reliableMatch = normalForReliability.valid()
                                       ? normalForReliability.rmse < 0.16
                                       : transform.score < 0.45;
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
            if (reliableMatch) {
                reliableTransformHistory.push_back(transform);
                if (reliableTransformHistory.size() > 6) {
                    reliableTransformHistory.erase(reliableTransformHistory.begin());
                }
            }
        } else {
            approxShiftX = approxInitX;
            approxShiftY = approxInitY;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
            emitLog(callbacks, "    [警告] 匹配得分过高，已将近似位移重置为初始值。");
        }
        previousDirection = stepDirection;
        const auto stepEnd = std::chrono::steady_clock::now();
        const double stepSeconds = std::chrono::duration<double>(stepEnd - stepBegin).count();
        if (!transform.profilingSummary.empty()) {
            emitLog(callbacks, "    [Profile] " + transform.profilingSummary);
        }
        emitLog(callbacks,
                "    [Timing] stitch step " + std::to_string(i + 1) +
                    " runtime = " + std::to_string(stepSeconds) + " s");
    }

    // --- 位姿图全局优化 ---
    // 用邻边 + 跳边约束优化所有图像的全局位姿，减少顺序拼接的误差积累
    if (result.imageTransforms.size() >= 2 && !result.steps.empty()) {
        auto poseEdges = buildAdjacentEdges(result.steps);
        const MeasuredPoseEdgeBuildResult skip2Measured =
            buildMeasuredSkipEdges(images, edgesPrepared, result.imageTransforms, config, 1, 0.25);
        poseEdges.insert(poseEdges.end(),
                         skip2Measured.edges.begin(),
                         skip2Measured.edges.end());
        const MeasuredPoseEdgeBuildResult skip3Measured =
            buildMeasuredSkipEdges(images, edgesPrepared, result.imageTransforms, config, 2, 0.10);
        poseEdges.insert(poseEdges.end(),
                         skip3Measured.edges.begin(),
                         skip3Measured.edges.end());

        const std::vector<cv::Mat> optimized =
            optimizePoseGraph(result.imageTransforms, poseEdges, 0.5, 30);

        if (optimized.size() == result.imageTransforms.size()) {
            const GlobalPoseCorrectionSummary poseCorrection =
                summarizeGlobalPoseCorrections(result.imageTransforms, optimized);
            const double maxDeltaDx = poseCorrection.maxDeltaDx;
            const double maxDeltaDy = poseCorrection.maxDeltaDy;
            const double maxDeltaAngleDeg = poseCorrection.maxDeltaAngleDeg;
            const std::vector<StitchStepRecord> rawSteps = result.steps;

            const bool hasIndependentSkipEdges =
                skip2Measured.acceptedCount + skip3Measured.acceptedCount > 0;
            if (!hasIndependentSkipEdges || !poseCorrection.meaningfulCorrection()) {
                emitLog(callbacks,
                        "    [位姿图] 跳过反写：独立跳边约束不足或全局修正不显著"
                        "（skip2=" + std::to_string(skip2Measured.acceptedCount) + "/" +
                        std::to_string(skip2Measured.attemptedCount) +
                        "，skip3=" + std::to_string(skip3Measured.acceptedCount) + "/" +
                        std::to_string(skip3Measured.attemptedCount) +
                        "，maxΔ=(" + std::to_string(maxDeltaDx) + " px, " +
                        std::to_string(maxDeltaDy) + " px, " +
                        std::to_string(maxDeltaAngleDeg) + " deg)）");
            } else {
                GlobalFeedbackRefinementSummary refinement =
                    refineStepsWithGlobalPosePrior(images, edgesPrepared, optimized, rawSteps, config);
                emitLog(callbacks,
                        "    [位姿图] 优化完成：skip2=" +
                            std::to_string(skip2Measured.acceptedCount) + "/" +
                            std::to_string(skip2Measured.attemptedCount) +
                            "，skip3=" + std::to_string(skip3Measured.acceptedCount) + "/" +
                            std::to_string(skip3Measured.attemptedCount) +
                            "，maxΔ=(" + std::to_string(maxDeltaDx) + " px, " +
                            std::to_string(maxDeltaDy) + " px, " +
                            std::to_string(maxDeltaAngleDeg) + " deg)");

                const GlobalFeedbackEvaluationSummary feedbackEvaluation =
                    evaluateGlobalFeedbackOutcome(rawSteps, refinement.steps);
                std::vector<StitchStepRecord>& refinedSteps = refinement.steps;
                const auto& worstBefore = feedbackEvaluation.worstBefore;
                const auto& worstAfter = feedbackEvaluation.worstAfter;
                const std::size_t badStepCountBefore = feedbackEvaluation.badStepCountBefore;
                const std::size_t badStepCountAfter = feedbackEvaluation.badStepCountAfter;
                const double endpointBeforeSum = feedbackEvaluation.endpointBeforeSum;
                const double endpointAfterSum = feedbackEvaluation.endpointAfterSum;
                const std::size_t localRefinedStepCount = refinement.localRefinedStepCount;
                const double maxLocalRefineCorrectionPx = refinement.maxLocalRefineCorrectionPx;

                if (!feedbackEvaluation.accepted) {
                    emitLog(callbacks,
                            "    [位姿图] 反写已回退：worst-step RMSE " +
                                std::to_string(feedbackEvaluation.worstBefore.first) + " -> " +
                                std::to_string(feedbackEvaluation.worstAfter.first) +
                                " px，坏步数 " + std::to_string(badStepCountBefore) + " -> " +
                                std::to_string(badStepCountAfter) +
                                "，端点RMSE " + std::to_string(endpointBeforeSum) + " -> " +
                                std::to_string(endpointAfterSum) +
                                "，局部重评估步数=" + std::to_string(localRefinedStepCount));
                } else {
                    const std::vector<cv::Mat> rebuiltGlobals =
                        buildGlobalTransformsFromSteps(images, refinedSteps);
                    cv::Mat rebuiltCanvas;
                    std::vector<cv::Mat> rebuiltImageTransforms;
                    if (!rebuiltGlobals.empty() &&
                        rebuildPanoramaFromTransforms(images,
                                                      rebuiltGlobals,
                                                      rebuiltCanvas,
                                                      rebuiltImageTransforms)) {
                        result.steps = std::move(refinedSteps);
                        result.canvas = std::move(rebuiltCanvas);
                        result.imageTransforms = std::move(rebuiltImageTransforms);
                        result.globalTransform = result.imageTransforms.back().clone();
                        emitLog(callbacks,
                                "    [位姿图] 已反写到单步并重评估：worst-step RMSE " +
                                    std::to_string(worstBefore.first) + " -> " +
                                    std::to_string(worstAfter.first) +
                                    " px，最差步 " + std::to_string(worstBefore.second) + " -> " +
                                    std::to_string(worstAfter.second) +
                                    "，坏步数 " + std::to_string(badStepCountBefore) + " -> " +
                                    std::to_string(badStepCountAfter) +
                                    "，端点RMSE " + std::to_string(endpointBeforeSum) + " -> " +
                                    std::to_string(endpointAfterSum) +
                                    "，局部重评估步数=" + std::to_string(localRefinedStepCount) +
                                    "，最大局部修正=" + std::to_string(maxLocalRefineCorrectionPx) + " px");
                    } else {
                        emitLog(callbacks,
                                "    [位姿图] 反写已回退：重建全景失败，保留原始单步结果。");
                    }
                }
            }
        }
    }

    applyDesignDrivenPrimaryScaleFeedback(images, edgesPrepared, config, result, callbacks);

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
