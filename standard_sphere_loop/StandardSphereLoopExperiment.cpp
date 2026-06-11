#include "StandardSphereLoopExperiment.h"

#include "stitch/Alignment.h"
#include "stitch/GeometryUtils.h"
#include "stitch/Pipeline.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace pinjie::standard_sphere_loop {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kVisualizationPaddingPx = 80;
constexpr int kVisualizationMaxSidePx = 2400;

double finiteOrZero(double value)
{
    return std::isfinite(value) ? value : 0.0;
}

std::string csvEscape(const std::string& text)
{
    if (text.find_first_of(",\"\n\r") == std::string::npos) {
        return text;
    }

    std::string escaped = "\"";
    for (char ch : text) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

double medianOf(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
        median = 0.5 * (median + values[mid - 1]);
    }
    return median;
}

double meanOf(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

struct WeightedPoint {
    cv::Point2d point{};
    double weight{1.0};
};

double saneWeight(double weight)
{
    return std::clamp(std::isfinite(weight) ? weight : 1.0, 0.01, 10.0);
}

std::vector<WeightedPoint> makeWeightedPoints(const std::vector<cv::Point2d>& points)
{
    std::vector<WeightedPoint> weighted;
    weighted.reserve(points.size());
    for (const cv::Point2d& point : points) {
        weighted.push_back({point, 1.0});
    }
    return weighted;
}

std::vector<WeightedPoint> makeWeightedPoints(const stitch::EdgeVariants& edge)
{
    std::vector<WeightedPoint> weighted;
    weighted.reserve(edge.raw.size());
    const bool hasWeights = edge.rawQualityWeights.size() == edge.raw.size();
    for (std::size_t i = 0; i < edge.raw.size(); ++i) {
        weighted.push_back({edge.raw[i], hasWeights ? saneWeight(edge.rawQualityWeights[i]) : 1.0});
    }
    return weighted;
}

bool fitCircleLeastSquaresWeighted(const std::vector<WeightedPoint>& points, CircleFitReport& out)
{
    out = {};
    out.pointCount = static_cast<int>(points.size());
    if (points.size() < 3) {
        return false;
    }

    cv::Mat A(static_cast<int>(points.size()), 3, CV_64F);
    cv::Mat b(static_cast<int>(points.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const WeightedPoint& weightedPoint = points[static_cast<std::size_t>(i)];
        const cv::Point2d& p = weightedPoint.point;
        const double w = std::sqrt(saneWeight(weightedPoint.weight));
        A.at<double>(i, 0) = p.x * w;
        A.at<double>(i, 1) = p.y * w;
        A.at<double>(i, 2) = w;
        b.at<double>(i, 0) = -(p.x * p.x + p.y * p.y) * w;
    }

    cv::Mat solution;
    if (!cv::solve(A, b, solution, cv::DECOMP_SVD)) {
        return false;
    }

    const double a = solution.at<double>(0, 0);
    const double bcoef = solution.at<double>(1, 0);
    const double c = solution.at<double>(2, 0);
    const double radiusSquared = (a * a + bcoef * bcoef) * 0.25 - c;
    if (!(radiusSquared > 0.0) || !std::isfinite(radiusSquared)) {
        return false;
    }

    out.center = cv::Point2d(-0.5 * a, -0.5 * bcoef);
    out.radiusPx = std::sqrt(radiusSquared);
    out.ok = std::isfinite(out.center.x) && std::isfinite(out.center.y) && std::isfinite(out.radiusPx);
    return out.ok;
}

CircleFitReport fitCircleRobustWeighted(const std::vector<WeightedPoint>& points)
{
    CircleFitReport fit;
    if (!fitCircleLeastSquaresWeighted(points, fit)) {
        return fit;
    }

    std::vector<WeightedPoint> inliers = points;
    for (int iter = 0; iter < 3; ++iter) {
        std::vector<double> residuals;
        residuals.reserve(inliers.size());
        for (const WeightedPoint& weightedPoint : inliers) {
            residuals.push_back(std::abs(cv::norm(weightedPoint.point - fit.center) - fit.radiusPx));
        }

        const double median = medianOf(residuals);
        std::vector<double> deviations;
        deviations.reserve(residuals.size());
        for (double residual : residuals) {
            deviations.push_back(std::abs(residual - median));
        }

        const double mad = medianOf(deviations);
        const double robustScale = std::max(0.05, 1.4826 * mad);
        const double threshold = std::max(2.5, median + 3.0 * robustScale);

        std::vector<WeightedPoint> next;
        next.reserve(inliers.size());
        for (std::size_t i = 0; i < inliers.size(); ++i) {
            if (residuals[i] <= threshold) {
                next.push_back(inliers[i]);
            }
        }

        if (next.size() < 20 || next.size() == inliers.size()) {
            break;
        }

        CircleFitReport refined;
        if (!fitCircleLeastSquaresWeighted(next, refined)) {
            break;
        }
        fit = refined;
        inliers = std::move(next);
    }

    double sumSquared = 0.0;
    double sumAbs = 0.0;
    double sumWeights = 0.0;
    double maxAbs = 0.0;
    std::vector<double> absResiduals;
    absResiduals.reserve(inliers.size());
    for (const WeightedPoint& weightedPoint : inliers) {
        const double weight = saneWeight(weightedPoint.weight);
        const double residual = std::abs(cv::norm(weightedPoint.point - fit.center) - fit.radiusPx);
        sumSquared += weight * residual * residual;
        sumAbs += weight * residual;
        sumWeights += weight;
        maxAbs = std::max(maxAbs, residual);
        absResiduals.push_back(residual);
    }

    fit.ok = true;
    fit.pointCount = static_cast<int>(inliers.size());
    fit.rmsePx = std::sqrt(sumSquared / std::max(1e-9, sumWeights));
    fit.meanAbsPx = sumAbs / std::max(1e-9, sumWeights);
    fit.medianAbsPx = medianOf(absResiduals);
    std::vector<double> deviations;
    deviations.reserve(absResiduals.size());
    for (double residual : absResiduals) {
        deviations.push_back(std::abs(residual - fit.medianAbsPx));
    }
    fit.madPx = medianOf(deviations);
    fit.sigmaMadPx = 1.4826 * fit.madPx;
    fit.maxAbsPx = maxAbs;
    return fit;
}

CircleFitReport fitCircleRobust(const std::vector<cv::Point2d>& points)
{
    return fitCircleRobustWeighted(makeWeightedPoints(points));
}

CircleFitReport fitCircleRobust(const stitch::EdgeVariants& edge)
{
    CircleFitReport fit = fitCircleRobustWeighted(makeWeightedPoints(edge));
    fit.qualityWeightMean = meanOf(edge.rawQualityWeights);
    fit.confidenceMean = meanOf(edge.rawConfidences);
    fit.gradientMean = meanOf(edge.rawGradients);
    return fit;
}

double resolveShiftRatio(const stitch::StitchPipelineConfig& config)
{
    if (std::isfinite(config.expectedOverlapRatio) &&
        config.expectedOverlapRatio >= 0.0 &&
        config.expectedOverlapRatio <= 1.0) {
        return std::clamp(1.0 - config.expectedOverlapRatio, 0.0, 1.0);
    }
    return std::max(0.0, config.approxShiftRatio);
}

void initializeApproxShift(const cv::Mat& image,
                           const stitch::StitchPipelineConfig& config,
                           double& approxShiftX,
                           double& approxShiftY)
{
    const double shiftRatio = resolveShiftRatio(config);
    approxShiftX = 0.0;
    approxShiftY = 0.0;

    switch (config.directionConstraint) {
    case stitch::MotionPriorDirection::XPositive:
        approxShiftX = image.cols * shiftRatio;
        break;
    case stitch::MotionPriorDirection::XNegative:
        approxShiftX = -image.cols * shiftRatio;
        break;
    case stitch::MotionPriorDirection::YPositive:
        approxShiftY = image.rows * shiftRatio;
        break;
    case stitch::MotionPriorDirection::YNegative:
        approxShiftY = -image.rows * shiftRatio;
        break;
    case stitch::MotionPriorDirection::Auto:
    default:
        approxShiftX = image.cols * shiftRatio;
        approxShiftY = image.rows * shiftRatio;
        break;
    }
}

std::string directionLabel(stitch::MotionPriorDirection direction)
{
    switch (direction) {
    case stitch::MotionPriorDirection::XPositive:
        return "X+";
    case stitch::MotionPriorDirection::YPositive:
        return "Y+";
    case stitch::MotionPriorDirection::XNegative:
        return "X-";
    case stitch::MotionPriorDirection::YNegative:
        return "Y-";
    case stitch::MotionPriorDirection::Auto:
    default:
        return "Auto";
    }
}

struct MotionScheduleEntry {
    stitch::MotionPriorDirection direction{stitch::MotionPriorDirection::Auto};
    std::size_t segmentIndex{0};
};

double horizontalStepPx(const cv::Mat& image, const StandardSphereLoopConfig& config)
{
    if (config.pixelSizeMm > 0.0 && config.horizontalStepMm > 0.0) {
        return config.horizontalStepMm / config.pixelSizeMm;
    }
    if (config.horizontalFieldOfViewMm > 0.0 && config.horizontalStepMm > 0.0) {
        return image.cols * (config.horizontalStepMm / config.horizontalFieldOfViewMm);
    }
    return image.cols * resolveShiftRatio(config.pipelineConfig);
}

double verticalStepPx(const cv::Mat& image, const StandardSphereLoopConfig& config)
{
    if (config.pixelSizeMm > 0.0 && config.verticalStepMm > 0.0) {
        return config.verticalStepMm / config.pixelSizeMm;
    }
    if (config.verticalFieldOfViewMm > 0.0 && config.verticalStepMm > 0.0) {
        return image.rows * (config.verticalStepMm / config.verticalFieldOfViewMm);
    }
    return image.rows * resolveShiftRatio(config.pipelineConfig);
}

double effectivePixelSizeMm(const cv::Mat& image, const StandardSphereLoopConfig& config)
{
    if (config.pixelSizeMm > 0.0) {
        return config.pixelSizeMm;
    }

    std::vector<double> axisScales;
    if (config.horizontalFieldOfViewMm > 0.0 && image.cols > 0) {
        axisScales.push_back(config.horizontalFieldOfViewMm / static_cast<double>(image.cols));
    }
    if (config.verticalFieldOfViewMm > 0.0 && image.rows > 0) {
        axisScales.push_back(config.verticalFieldOfViewMm / static_cast<double>(image.rows));
    }
    if (axisScales.empty()) {
        return 0.0;
    }
    return std::accumulate(axisScales.begin(), axisScales.end(), 0.0) /
           static_cast<double>(axisScales.size());
}

cv::Point2d expectedStepPxForDirection(const cv::Mat& image,
                                       const StandardSphereLoopConfig& config,
                                       stitch::MotionPriorDirection direction)
{
    const double x = horizontalStepPx(image, config);
    const double y = verticalStepPx(image, config);
    switch (direction) {
    case stitch::MotionPriorDirection::XPositive:
        return {x, 0.0};
    case stitch::MotionPriorDirection::YPositive:
        return {0.0, y};
    case stitch::MotionPriorDirection::XNegative:
        return {-x, 0.0};
    case stitch::MotionPriorDirection::YNegative:
        return {0.0, -y};
    case stitch::MotionPriorDirection::Auto:
    default:
        return {0.0, 0.0};
    }
}

cv::Point2d expectedStepMmForDirection(const StandardSphereLoopConfig& config,
                                       stitch::MotionPriorDirection direction)
{
    switch (direction) {
    case stitch::MotionPriorDirection::XPositive:
        return {config.horizontalStepMm, 0.0};
    case stitch::MotionPriorDirection::YPositive:
        return {0.0, config.verticalStepMm};
    case stitch::MotionPriorDirection::XNegative:
        return {-config.horizontalStepMm, 0.0};
    case stitch::MotionPriorDirection::YNegative:
        return {0.0, -config.verticalStepMm};
    case stitch::MotionPriorDirection::Auto:
    default:
        return {0.0, 0.0};
    }
}

std::vector<int> autoCalculateSegmentCounts(std::size_t pairCount)
{
    std::vector<int> counts(4, 0);
    if (pairCount == 0) {
        return counts;
    }

    const std::size_t halfPair = pairCount / 2;
    const std::size_t xPairs = halfPair / 2;
    const std::size_t yPairs = halfPair - xPairs;

    counts[0] = static_cast<int>(xPairs);       // X+
    counts[1] = static_cast<int>(yPairs);       // Y+
    counts[2] = static_cast<int>(xPairs);       // X-
    counts[3] = static_cast<int>(yPairs);       // Y-

    std::size_t allocated = static_cast<std::size_t>(counts[0] + counts[1] + counts[2] + counts[3]);
    std::size_t remaining = pairCount - allocated;

    // Distribute any remaining pairs, keeping X+ == X- and Y+ == Y- balanced.
    // Add to X pair first, then Y pair.
    while (remaining > 0) {
        if (remaining >= 2) {
            ++counts[0];
            ++counts[2];
            remaining -= 2;
        } else {
            ++counts[1];
            --remaining;
        }
    }

    return counts;
}

std::vector<MotionScheduleEntry> buildMotionSchedule(std::size_t pairCount,
                                                     const StandardSphereLoopConfig& config)
{
    std::vector<MotionScheduleEntry> schedule;
    schedule.reserve(pairCount);

    if (!config.useClockwiseLoopPath) {
        for (std::size_t i = 0; i < pairCount; ++i) {
            schedule.push_back({config.pipelineConfig.directionConstraint, 1});
        }
        return schedule;
    }

    static const std::array<stitch::MotionPriorDirection, 4> kDefaultDirectionCycle = {
        stitch::MotionPriorDirection::XPositive,
        stitch::MotionPriorDirection::YPositive,
        stitch::MotionPriorDirection::XNegative,
        stitch::MotionPriorDirection::YNegative
    };
    std::vector<stitch::MotionPriorDirection> directionCycle =
        config.clockwiseSegmentDirections;
    if (directionCycle.empty()) {
        directionCycle.assign(kDefaultDirectionCycle.begin(), kDefaultDirectionCycle.end());
    }

    std::vector<int> counts = config.clockwiseSegmentPairCounts;
    int configuredTotal = 0;
    for (int count : counts) {
        configuredTotal += std::max(0, count);
    }

    if (counts.empty() || configuredTotal <= 0) {
        counts = autoCalculateSegmentCounts(pairCount);
        configuredTotal = 0;
        for (int count : counts) {
            configuredTotal += count;
        }
    }

    // Use exact counts if they sum to pairCount; otherwise auto-calculate.
    if (configuredTotal != static_cast<int>(pairCount)) {
        counts = autoCalculateSegmentCounts(pairCount);
    }

    std::size_t segmentIndex = 1;
    for (std::size_t seg = 0; seg < counts.size() && schedule.size() < pairCount; ++seg) {
        const auto direction = directionCycle[seg % directionCycle.size()];
        const int segCount = std::max(0, counts[seg]);
        for (int step = 0; step < segCount && schedule.size() < pairCount; ++step) {
            schedule.push_back({direction, segmentIndex});
        }
        ++segmentIndex;
    }

    // Pad any remaining slots with the last direction and segment index.
    const auto lastDirection = counts.empty()
                                   ? directionCycle[0]
                                   : directionCycle[(counts.size() - 1) % directionCycle.size()];
    const std::size_t lastSegIdx = segmentIndex > 0 ? segmentIndex - 1 : 1;
    while (schedule.size() < pairCount) {
        schedule.push_back({lastDirection, lastSegIdx});
    }

    return schedule;
}

const stitch::ResidualStatistics& preferredNormalStats(const stitch::TransformResult& transform)
{
    return transform.metrics.normalInlier.valid() ? transform.metrics.normalInlier
                                                  : transform.metrics.normalAll;
}

bool isReliablePair(const stitch::TransformResult& transform)
{
    const stitch::ResidualStatistics& normal = preferredNormalStats(transform);
    return normal.valid() ? normal.rmse < std::sqrt(0.5) : transform.score < 0.5;
}

cv::Point2d rotateAround(const cv::Point2d& point,
                         const cv::Point2d& anchor,
                         double angleDeg);
cv::Mat relativeMatrix(const PairLoopRecord& record);
cv::Point2d transformPoint(const cv::Mat& matrix, const cv::Point2d& point);
double rotationDegFromMatrix(const cv::Mat& matrix);
void evaluateWithGlobalTransforms(StandardSphereLoopResult& result,
                                  const std::vector<cv::Mat>& images,
                                  const std::vector<stitch::EdgeVariants>& edges);

bool needsStandardSphereGeometryRescue(const stitch::TransformResult& transform)
{
    const stitch::ResidualStatistics& normal = preferredNormalStats(transform);
    const stitch::ResidualStatistics& tangent =
        transform.metrics.tangentInlier.valid() ? transform.metrics.tangentInlier
                                                : transform.metrics.tangentAll;
    const double normalRmse = normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
    const double tangentRmse = tangent.valid() ? tangent.rmse : 0.0;
    return normalRmse > 0.7 || tangentRmse > 1.5;
}

void recomputePairAccumulations(std::vector<PairLoopRecord>& records,
                                double evaluationPixelSizeMm)
{
    cv::Point2d cumulativeMeasuredPx(0.0, 0.0);
    cv::Point2d cumulativeExpectedPx(0.0, 0.0);
    for (PairLoopRecord& record : records) {
        const cv::Point2d measuredStep(record.transform.dx, record.transform.dy);
        cumulativeMeasuredPx += measuredStep;
        cumulativeExpectedPx += record.expectedStepPx;
        record.measuredStepMm =
            evaluationPixelSizeMm > 0.0 ? cv::Point2d(measuredStep.x * evaluationPixelSizeMm,
                                                       measuredStep.y * evaluationPixelSizeMm)
                                        : cv::Point2d(0.0, 0.0);
        record.cumulativeMeasuredPx = cumulativeMeasuredPx;
        record.cumulativeExpectedPx = cumulativeExpectedPx;
        record.cumulativeErrorPx = cumulativeMeasuredPx - cumulativeExpectedPx;
        record.localTranslationErrorPx = cv::norm(measuredStep - record.expectedStepPx);
        record.localTranslationErrorMm =
            evaluationPixelSizeMm > 0.0 ? record.localTranslationErrorPx * evaluationPixelSizeMm
                                        : 0.0;
        record.cumulativeTranslationErrorPx = cv::norm(record.cumulativeErrorPx);
        record.cumulativeTranslationErrorMm =
            evaluationPixelSizeMm > 0.0 ? record.cumulativeTranslationErrorPx * evaluationPixelSizeMm
                                        : 0.0;
    }
}

void applyStandardSphereBadStepGeometryRescue(std::vector<PairLoopRecord>& records,
                                              const std::vector<CircleFitReport>& circles,
                                              StandardSphereLoopResult& result)
{
    for (PairLoopRecord& record : records) {
        if (!needsStandardSphereGeometryRescue(record.transform) ||
            record.referenceIndex >= circles.size() ||
            record.targetIndex >= circles.size() ||
            !circles[record.referenceIndex].ok ||
            !circles[record.targetIndex].ok) {
            continue;
        }

        stitch::TransformResult rescued = record.transform;
        rescued.da = std::clamp(rescued.da, -0.2, 0.2);
        const cv::Point2d rotatedTargetCenter =
            rotateAround(circles[record.targetIndex].center, record.rotationCenter, rescued.da);
        const cv::Point2d rescuedShift = circles[record.referenceIndex].center - rotatedTargetCenter;
        const cv::Point2d previousShift(record.transform.dx, record.transform.dy);
        const double correctionPx = cv::norm(rescuedShift - previousShift);
        if (correctionPx > 3.0) {
            continue;
        }

        constexpr double kSoftGeometryBlend = 0.5;
        const cv::Point2d blendedShift =
            previousShift + kSoftGeometryBlend * (rescuedShift - previousShift);
        rescued.dx = blendedShift.x;
        rescued.dy = blendedShift.y;
        record.transform = rescued;

        ++result.sphereBadStepGeometryRescueCount;
        result.sphereBadStepGeometryRescueMaxCorrectionPx =
            std::max(result.sphereBadStepGeometryRescueMaxCorrectionPx,
                     cv::norm(blendedShift - previousShift));
    }
}

stitch::TransformResult transformFromCandidate(const stitch::AlignmentCandidateDiagnostic& candidate,
                                               const stitch::TransformResult& source)
{
    stitch::TransformResult transform = source;
    transform.dx = candidate.dx;
    transform.dy = candidate.dy;
    transform.da = candidate.da;
    transform.score = candidate.score;
    transform.normalMatchCost = candidate.normalMatchCost;
    transform.tangentResidualMatchCost = candidate.tangentResidualMatchCost;
    transform.tangentCorrelationMatchCost = candidate.tangentCorrelationMatchCost;
    transform.directionPenaltyMatchCost = candidate.directionPenaltyMatchCost;
    transform.axis = candidate.axis;
    transform.direction = candidate.direction;
    transform.metrics = candidate.metrics;
    return transform;
}

std::vector<cv::Mat> buildTransformsFromRecords(const std::vector<PairLoopRecord>& records,
                                                const cv::Mat& firstTransform,
                                                std::size_t imageCount)
{
    std::vector<cv::Mat> transforms;
    if (imageCount == 0) {
        return transforms;
    }

    transforms.reserve(imageCount);
    transforms.push_back(firstTransform.empty() ? cv::Mat::eye(3, 3, CV_64F) : firstTransform.clone());
    for (std::size_t i = 0; i < records.size() && transforms.size() < imageCount; ++i) {
        transforms.push_back(transforms.back() * relativeMatrix(records[i]));
    }
    while (transforms.size() < imageCount) {
        transforms.push_back(transforms.back().clone());
    }
    return transforms;
}

CircleFitReport fitStitchedCircleForTransforms(const std::vector<cv::Mat>& transforms,
                                               const std::vector<stitch::EdgeVariants>& edges)
{
    std::vector<WeightedPoint> stitchedPoints;
    std::size_t total = 0;
    for (std::size_t i = 0; i < edges.size() && i < transforms.size(); ++i) {
        total += edges[i].raw.size();
    }
    stitchedPoints.reserve(total);

    for (std::size_t i = 0; i < edges.size() && i < transforms.size(); ++i) {
        const std::vector<WeightedPoint> weightedPoints = makeWeightedPoints(edges[i]);
        const std::size_t maxPerImage = 800;
        const std::size_t step = weightedPoints.size() > maxPerImage
                                     ? std::max<std::size_t>(1, weightedPoints.size() / maxPerImage)
                                     : 1;
        for (std::size_t j = 0; j < weightedPoints.size(); j += step) {
            const WeightedPoint& point = weightedPoints[j];
            stitchedPoints.push_back({transformPoint(transforms[i], point.point), point.weight});
        }
    }
    return fitCircleRobustWeighted(stitchedPoints);
}

double globalConsistencyCost(const std::vector<PairLoopRecord>& records,
                             const std::vector<cv::Mat>& transforms,
                             const std::vector<stitch::EdgeVariants>& edges)
{
    if (records.empty() || transforms.size() < 2) {
        return std::numeric_limits<double>::infinity();
    }

    cv::Point2d expectedClosure(0.0, 0.0);
    for (const PairLoopRecord& record : records) {
        expectedClosure += record.expectedStepPx;
    }

    const cv::Mat closure = transforms.front().inv() * transforms.back();
    const cv::Point2d measuredClosure(closure.at<double>(0, 2), closure.at<double>(1, 2));
    const double closureResidualPx = cv::norm(measuredClosure - expectedClosure);

    std::vector<SegmentLoopRecord> segments;
    for (const PairLoopRecord& record : records) {
        if (segments.empty() || segments.back().segmentIndex != record.segmentIndex) {
            SegmentLoopRecord segment;
            segment.segmentIndex = record.segmentIndex;
            segments.push_back(segment);
        }
        SegmentLoopRecord& segment = segments.back();
        ++segment.pairCount;
        segment.measuredDeltaPx += cv::Point2d(record.transform.dx, record.transform.dy);
        segment.expectedDeltaPx += record.expectedStepPx;
    }
    double segmentMeanSq = 0.0;
    for (const SegmentLoopRecord& segment : segments) {
        segmentMeanSq += cv::norm(segment.measuredDeltaPx - segment.expectedDeltaPx) *
                         cv::norm(segment.measuredDeltaPx - segment.expectedDeltaPx);
    }
    if (!segments.empty()) {
        segmentMeanSq /= static_cast<double>(segments.size());
    }

    double badPairPenalty = 0.0;
    for (const PairLoopRecord& record : records) {
        const stitch::ResidualStatistics& normal = preferredNormalStats(record.transform);
        const double normalRmse = normal.valid() ? normal.rmse : 10.0;
        const double excess = std::max(0.0, normalRmse - 0.7);
        badPairPenalty += excess * excess;
    }

    return 0.02 * closureResidualPx * closureResidualPx +
           0.005 * segmentMeanSq +
           0.30 * badPairPenalty;
}

void applyGlobalConsistencyReregistration(std::vector<PairLoopRecord>& records,
                                          const std::vector<stitch::EdgeVariants>& edges,
                                          const cv::Mat& firstTransform,
                                          std::size_t imageCount,
                                          double evaluationPixelSizeMm,
                                          StandardSphereLoopResult& result)
{
    if (records.empty() || imageCount < 2) {
        return;
    }

    std::vector<cv::Mat> bestTransforms =
        buildTransformsFromRecords(records, firstTransform, imageCount);
    double bestCost = globalConsistencyCost(records, bestTransforms, edges);
    const double initialCost = bestCost;

    for (int iter = 0; iter < 2; ++iter) {
        bool improved = false;
        for (std::size_t i = 0; i < records.size(); ++i) {
            PairLoopRecord& record = records[i];
            if (record.transform.candidateDiagnostics.empty()) {
                continue;
            }

            const stitch::TransformResult originalTransform = record.transform;
            stitch::TransformResult bestTransformForStep = originalTransform;
            double bestStepCost = bestCost;

            const stitch::ResidualStatistics& currentNormal = preferredNormalStats(originalTransform);
            const double currentNormalRmse = currentNormal.valid() ? currentNormal.rmse : 10.0;
            const bool stepIsWorthTesting =
                currentNormalRmse > 0.35 ||
                record.localTranslationErrorPx > 32.0 ||
                std::abs(originalTransform.da) > 0.08;
            if (!stepIsWorthTesting) {
                continue;
            }

            int tested = 0;
            for (const stitch::AlignmentCandidateDiagnostic& candidate : record.transform.candidateDiagnostics) {
                if (tested++ >= 12) {
                    break;
                }
                if (record.expectedDirection != "Auto" &&
                    candidate.direction.find(record.expectedDirection) == std::string::npos) {
                    continue;
                }
                const stitch::ResidualStatistics& candidateNormal =
                    candidate.metrics.normalInlier.valid() ? candidate.metrics.normalInlier
                                                           : candidate.metrics.normalAll;
                if (!candidateNormal.valid() || candidateNormal.rmse > 3.0) {
                    continue;
                }
                if (std::abs(candidate.da) > 0.2) {
                    continue;
                }

                record.transform = transformFromCandidate(candidate, originalTransform);
                std::vector<cv::Mat> trialTransforms =
                    buildTransformsFromRecords(records, firstTransform, imageCount);
                const double trialCost = globalConsistencyCost(records, trialTransforms, edges);
                if (trialCost + 1e-6 < bestStepCost) {
                    bestStepCost = trialCost;
                    bestTransformForStep = record.transform;
                }
            }

            record.transform = bestTransformForStep;
            if (bestStepCost < bestCost * 0.995) {
                bestCost = bestStepCost;
                bestTransforms = buildTransformsFromRecords(records, firstTransform, imageCount);
                ++result.globalConsistencyReregistrationCount;
                improved = true;
            } else {
                record.transform = originalTransform;
            }
        }
        if (!improved) {
            break;
        }
    }

    if (result.globalConsistencyReregistrationCount > 0) {
        result.globalConsistencyReregistrationImprovement = initialCost - bestCost;
        recomputePairAccumulations(records, evaluationPixelSizeMm);
    }
}

void refreshPairMetrics(PairLoopRecord& record,
                        const std::vector<stitch::EdgeVariants>& edges)
{
    if (record.referenceIndex >= edges.size() || record.targetIndex >= edges.size()) {
        return;
    }

    stitch::TransformResult& transform = record.transform;
    std::vector<cv::Point2d> rotatedTarget;
    stitch::rotatePoints(edges[record.targetIndex].raw,
                         rotatedTarget,
                         transform.da,
                         record.rotationCenter);

    if (transform.axis == stitch::AlignmentAxis::Y) {
        stitch::sortContourByY(rotatedTarget);
        stitch::populateAlignmentMetrics(edges[record.referenceIndex].y_sorted,
                                         rotatedTarget,
                                         transform);
    } else {
        transform.axis = stitch::AlignmentAxis::X;
        stitch::sortContourByX(rotatedTarget);
        stitch::populateAlignmentMetrics(edges[record.referenceIndex].x_sorted,
                                         rotatedTarget,
                                         transform);
    }

    const stitch::ResidualStatistics& normal = preferredNormalStats(transform);
    const stitch::ResidualStatistics& tangent =
        transform.metrics.tangentInlier.valid() ? transform.metrics.tangentInlier
                                                : transform.metrics.tangentAll;
    const double normalRmse = normal.valid() ? normal.rmse : 10.0;
    const double tangentRmse = tangent.valid() ? tangent.rmse : 0.0;
    transform.normalMatchCost = normalRmse * normalRmse;
    transform.tangentResidualMatchCost = 0.05 * tangentRmse * tangentRmse;
    transform.tangentCorrelationMatchCost =
        0.25 * std::max(0.0, 1.0 - transform.metrics.tangentCorrInlier);
    transform.score = transform.normalMatchCost +
                      transform.tangentResidualMatchCost +
                      transform.tangentCorrelationMatchCost +
                      transform.directionPenaltyMatchCost;
}

double pairQualityCost(const PairLoopRecord& record)
{
    const stitch::ResidualStatistics& normal = preferredNormalStats(record.transform);
    const stitch::ResidualStatistics& tangent =
        record.transform.metrics.tangentInlier.valid() ? record.transform.metrics.tangentInlier
                                                       : record.transform.metrics.tangentAll;
    const double normalRmse = normal.valid() ? normal.rmse : 10.0;
    const double tangentRmse = tangent.valid() ? tangent.rmse : 0.0;
    const double localPrior = std::clamp(record.localTranslationErrorPx / 40.0, 0.0, 4.0);
    return normalRmse * normalRmse +
           0.02 * tangentRmse * tangentRmse +
           0.04 * localPrior * localPrior;
}

double recordsCircleRmse(const std::vector<PairLoopRecord>& records,
                         const std::vector<stitch::EdgeVariants>& edges,
                         const cv::Mat& firstTransform,
                         std::size_t imageCount)
{
    const std::vector<cv::Mat> transforms = buildTransformsFromRecords(records, firstTransform, imageCount);
    const CircleFitReport fit = fitStitchedCircleForTransforms(transforms, edges);
    return fit.ok ? fit.rmsePx : std::numeric_limits<double>::infinity();
}

void applyBadStepLocalRefinement(std::vector<PairLoopRecord>& records,
                                 const std::vector<stitch::EdgeVariants>& edges,
                                 const cv::Mat& firstTransform,
                                 std::size_t imageCount,
                                 double evaluationPixelSizeMm,
                                 StandardSphereLoopResult& result)
{
    if (records.empty() || imageCount < 2 || edges.empty()) {
        return;
    }

    double bestGlobalCost = globalConsistencyCost(records,
                                                  buildTransformsFromRecords(records, firstTransform, imageCount),
                                                  edges);
    double bestCircleRmse = recordsCircleRmse(records, edges, firstTransform, imageCount);
    const double initialCombinedCost = bestGlobalCost + 0.35 * bestCircleRmse * bestCircleRmse;

    for (int pass = 0; pass < 2; ++pass) {
        bool acceptedInPass = false;
        for (std::size_t i = 0; i < records.size(); ++i) {
            PairLoopRecord& record = records[i];
            const stitch::ResidualStatistics& currentNormal = preferredNormalStats(record.transform);
            const double currentNormalRmse = currentNormal.valid() ? currentNormal.rmse : 10.0;
            if (currentNormalRmse < 0.70 && record.localTranslationErrorPx < 34.0) {
                continue;
            }

            const PairLoopRecord originalRecord = record;
            const double originalPairCost = pairQualityCost(originalRecord);
            PairLoopRecord bestRecord = originalRecord;
            double bestStepCombinedCost =
                bestGlobalCost + 0.35 * bestCircleRmse * bestCircleRmse + 0.25 * originalPairCost;

            std::vector<stitch::TransformResult> seeds;
            seeds.push_back(originalRecord.transform);
            for (const stitch::AlignmentCandidateDiagnostic& diagnostic :
                 originalRecord.transform.candidateDiagnostics) {
                if (originalRecord.expectedDirection != "Auto" &&
                    diagnostic.direction.find(originalRecord.expectedDirection) == std::string::npos) {
                    continue;
                }
                if (std::abs(diagnostic.da) > 0.2) {
                    continue;
                }
                seeds.push_back(transformFromCandidate(diagnostic, originalRecord.transform));
                if (seeds.size() >= 8) {
                    break;
                }
            }

            for (const stitch::TransformResult& seed : seeds) {
                const std::array<double, 5> angleOffsets{{-0.04, -0.02, 0.0, 0.02, 0.04}};
                const std::array<double, 5> primaryOffsets{{-6.0, -3.0, 0.0, 3.0, 6.0}};
                const std::array<double, 7> perpOffsets{{-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0}};

                for (double daOffset : angleOffsets) {
                    const double trialAngle = std::clamp(seed.da + daOffset, -0.2, 0.2);
                    for (double primaryOffset : primaryOffsets) {
                        for (double perpOffset : perpOffsets) {
                            PairLoopRecord trialRecord = originalRecord;
                            trialRecord.transform = seed;
                            trialRecord.transform.da = trialAngle;
                            if (trialRecord.transform.axis == stitch::AlignmentAxis::Y) {
                                trialRecord.transform.dy = seed.dy + primaryOffset;
                                trialRecord.transform.dx = seed.dx + perpOffset;
                            } else {
                                trialRecord.transform.dx = seed.dx + primaryOffset;
                                trialRecord.transform.dy = seed.dy + perpOffset;
                            }
                            trialRecord.transform.candidateDiagnostics =
                                originalRecord.transform.candidateDiagnostics;
                            refreshPairMetrics(trialRecord, edges);

                            std::vector<PairLoopRecord> trialRecords = records;
                            trialRecords[i] = trialRecord;
                            recomputePairAccumulations(trialRecords, evaluationPixelSizeMm);

                            const double trialGlobalCost =
                                globalConsistencyCost(trialRecords,
                                                      buildTransformsFromRecords(trialRecords,
                                                                                firstTransform,
                                                                                imageCount),
                                                      edges);
                            const double trialCircleRmse =
                                recordsCircleRmse(trialRecords, edges, firstTransform, imageCount);
                            const double trialPairCost = pairQualityCost(trialRecords[i]);
                            const double combinedCost =
                                trialGlobalCost + 0.35 * trialCircleRmse * trialCircleRmse +
                                0.25 * trialPairCost;

                            const stitch::ResidualStatistics& trialNormal =
                                preferredNormalStats(trialRecords[i].transform);
                            const double trialNormalRmse =
                                trialNormal.valid() ? trialNormal.rmse : 10.0;
                            const bool pairImproved = trialNormalRmse < currentNormalRmse * 0.98;
                            const bool circleNotWorse = trialCircleRmse <= bestCircleRmse + 0.03;
                            if (pairImproved && circleNotWorse &&
                                combinedCost + 1e-6 < bestStepCombinedCost) {
                                bestStepCombinedCost = combinedCost;
                                bestRecord = trialRecords[i];
                            }
                        }
                    }
                }
            }

            if (bestStepCombinedCost + 1e-6 <
                bestGlobalCost + 0.35 * bestCircleRmse * bestCircleRmse + 0.25 * originalPairCost) {
                const cv::Point2d before(originalRecord.transform.dx, originalRecord.transform.dy);
                const cv::Point2d after(bestRecord.transform.dx, bestRecord.transform.dy);
                record = bestRecord;
                recomputePairAccumulations(records, evaluationPixelSizeMm);
                bestGlobalCost =
                    globalConsistencyCost(records,
                                          buildTransformsFromRecords(records, firstTransform, imageCount),
                                          edges);
                bestCircleRmse = recordsCircleRmse(records, edges, firstTransform, imageCount);
                result.badStepLocalRefinementMaxCorrectionPx =
                    std::max(result.badStepLocalRefinementMaxCorrectionPx, cv::norm(after - before));
                ++result.badStepLocalRefinementCount;
                acceptedInPass = true;
            } else {
                record = originalRecord;
            }
        }
        if (!acceptedInPass) {
            break;
        }
    }

    if (result.badStepLocalRefinementCount > 0) {
        const double finalCombinedCost = bestGlobalCost + 0.35 * bestCircleRmse * bestCircleRmse;
        result.badStepLocalRefinementCostImprovement =
            std::max(0.0, initialCombinedCost - finalCombinedCost);
        recomputePairAccumulations(records, evaluationPixelSizeMm);
    }
}

cv::Point2d closureResidualForRecords(const std::vector<PairLoopRecord>& records,
                                      const cv::Mat& firstTransform,
                                      std::size_t imageCount)
{
    const std::vector<cv::Mat> transforms = buildTransformsFromRecords(records, firstTransform, imageCount);
    if (records.empty() || transforms.size() < 2) {
        return {};
    }

    cv::Point2d expectedClosure(0.0, 0.0);
    for (const PairLoopRecord& record : records) {
        expectedClosure += record.expectedStepPx;
    }
    const cv::Mat closure = transforms.front().inv() * transforms.back();
    const cv::Point2d measuredClosure(closure.at<double>(0, 2), closure.at<double>(1, 2));
    return measuredClosure - expectedClosure;
}

void applySoftGlobalDriftOptimization(std::vector<PairLoopRecord>& records,
                                      const std::vector<stitch::EdgeVariants>& edges,
                                      const cv::Mat& firstTransform,
                                      std::size_t imageCount,
                                      double evaluationPixelSizeMm,
                                      StandardSphereLoopResult& result)
{
    if (records.empty() || imageCount < 2) {
        return;
    }

    std::vector<cv::Mat> transforms = buildTransformsFromRecords(records, firstTransform, imageCount);
    double bestCost = globalConsistencyCost(records, transforms, edges);
    const double initialCost = bestCost;

    for (int iter = 0; iter < 6; ++iter) {
        const cv::Point2d residual = closureResidualForRecords(records, firstTransform, imageCount);
        if (cv::norm(residual) < 0.1) {
            break;
        }

        std::vector<double> weights;
        weights.reserve(records.size());
        double weightSum = 0.0;
        for (const PairLoopRecord& record : records) {
            const stitch::ResidualStatistics& normal = preferredNormalStats(record.transform);
            const double normalRmse = normal.valid() ? normal.rmse : 1.0;
            const double localErrorWeight = std::clamp(record.localTranslationErrorPx / 35.0, 0.25, 3.0);
            double circleBias = 1.0;
            if (record.targetIndex < result.perImageCircles.size() &&
                result.perImageCircles[record.targetIndex].ok) {
                const CircleFitReport& targetCircle = result.perImageCircles[record.targetIndex];
                const double circleRmse = std::max(0.0, targetCircle.rmsePx);
                const double pointBias =
                    1.0 + std::clamp(500.0 / std::max(1, targetCircle.pointCount), 0.0, 1.0);
                circleBias = std::clamp((0.5 + 3.0 * circleRmse) * pointBias, 0.8, 2.5);
            }
            const double weight =
                std::clamp((0.35 + normalRmse * normalRmse + localErrorWeight) * circleBias,
                           0.2,
                           8.0);
            weights.push_back(weight);
            weightSum += weight;
        }
        if (weightSum <= 1e-9) {
            break;
        }

        std::vector<PairLoopRecord> trial = records;
        double maxStepCorrection = 0.0;
        double totalCorrection = 0.0;
        constexpr double kAlpha = 0.65;
        constexpr double kMaxStepCorrectionPx = 2.5;
        for (std::size_t i = 0; i < trial.size(); ++i) {
            cv::Point2d correction = -kAlpha * residual * (weights[i] / weightSum);
            const double correctionNorm = cv::norm(correction);
            if (correctionNorm > kMaxStepCorrectionPx) {
                correction *= kMaxStepCorrectionPx / correctionNorm;
            }
            trial[i].transform.dx += correction.x;
            trial[i].transform.dy += correction.y;
            maxStepCorrection = std::max(maxStepCorrection, cv::norm(correction));
            totalCorrection += cv::norm(correction);
        }

        const std::vector<cv::Mat> trialTransforms = buildTransformsFromRecords(trial, firstTransform, imageCount);
        const double trialCost = globalConsistencyCost(trial, trialTransforms, edges);
        const cv::Point2d trialResidual = closureResidualForRecords(trial, firstTransform, imageCount);
        if (trialCost < bestCost && cv::norm(trialResidual) < cv::norm(residual)) {
            records = std::move(trial);
            bestCost = trialCost;
            ++result.softGlobalDriftOptimizationIterations;
            result.softGlobalDriftOptimizationMaxStepCorrectionPx =
                std::max(result.softGlobalDriftOptimizationMaxStepCorrectionPx, maxStepCorrection);
            result.softGlobalDriftOptimizationTotalCorrectionPx += totalCorrection;
        } else {
            break;
        }
    }

    if (result.softGlobalDriftOptimizationIterations > 0) {
        result.softGlobalDriftOptimizationCostImprovement = initialCost - bestCost;
        recomputePairAccumulations(records, evaluationPixelSizeMm);
    }
}

cv::Mat anchoredRigidCorrectionMatrix(const cv::Point2d& anchor,
                                      double dx,
                                      double dy,
                                      double angleDeg)
{
    const double angleRad = angleDeg * kPi / 180.0;
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    matrix.at<double>(0, 0) = c;
    matrix.at<double>(0, 1) = -s;
    matrix.at<double>(1, 0) = s;
    matrix.at<double>(1, 1) = c;
    matrix.at<double>(0, 2) = anchor.x - c * anchor.x + s * anchor.y + dx;
    matrix.at<double>(1, 2) = anchor.y - s * anchor.x - c * anchor.y + dy;
    return matrix;
}

double poseConsistencyObjective(const StandardSphereLoopResult& candidate,
                                double referenceCircleRmse,
                                double targetClosurePx,
                                double circleRmseTolerancePx)
{
    const double closureExcess =
        std::max(0.0, candidate.closureResidualTranslationPx - targetClosurePx);
    const double centerExcess =
        std::max(0.0, candidate.closureCenterDriftPx - targetClosurePx);
    const double probeExcess =
        std::max(0.0, candidate.closureCornerRmsPx - 2.0 * targetClosurePx);
    const double circleExcess =
        std::max(0.0, candidate.stitchedCircle.rmsePx -
                          referenceCircleRmse -
                          circleRmseTolerancePx);

    return candidate.closureResidualTranslationPx +
           0.35 * candidate.closureCenterDriftPx +
           0.05 * candidate.closureCornerRmsPx +
           8.0 * closureExcess * closureExcess +
           1.5 * centerExcess * centerExcess +
           0.15 * probeExcess * probeExcess +
           120.0 * circleExcess * circleExcess;
}

double standardSphereBranchSelectionCost(const StandardSphereLoopResult& candidate,
                                         const StandardSphereLoopConfig& config)
{
    const double targetClosurePx = std::max(0.05, config.poseConsistencyTargetClosurePx);
    const double closureExcess =
        std::max(0.0, candidate.closureResidualTranslationPx - targetClosurePx);
    const double centerExcess =
        std::max(0.0, candidate.closureCenterDriftPx - targetClosurePx);
    const double probeExcess =
        std::max(0.0, candidate.closureCornerRmsPx - 2.0 * targetClosurePx);

    return 1000.0 * closureExcess * closureExcess +
           120.0 * centerExcess * centerExcess +
           20.0 * probeExcess * probeExcess +
           8.0 * candidate.closureResidualTranslationPx +
           2.0 * candidate.stitchedCircle.rmsePx +
           0.5 * candidate.closureCenterDriftPx;
}

std::vector<cv::Mat> cloneTransforms(const std::vector<cv::Mat>& transforms)
{
    std::vector<cv::Mat> cloned;
    cloned.reserve(transforms.size());
    for (const cv::Mat& transform : transforms) {
        cloned.push_back(transform.clone());
    }
    return cloned;
}

void applySoftGlobalTransformDrift(StandardSphereLoopResult& result,
                                   const std::vector<cv::Mat>& images,
                                   const std::vector<stitch::EdgeVariants>& edges,
                                   const StandardSphereLoopConfig& config)
{
    if (result.globalTransforms.size() < 2) {
        return;
    }

    const cv::Mat closure = result.globalTransforms.front().inv() * result.globalTransforms.back();
    const cv::Point2d measuredClosure(closure.at<double>(0, 2), closure.at<double>(1, 2));
    const cv::Point2d residual = measuredClosure - result.expectedClosurePx;
    const double residualNorm = cv::norm(residual);
    const double closureRotationDeg = rotationDegFromMatrix(closure);
    const double targetClosurePx = std::max(0.05, config.poseConsistencyTargetClosurePx);
    if (residualNorm < targetClosurePx && std::abs(closureRotationDeg) < 0.01) {
        return;
    }

    const double referenceCircleRmse = result.stitchedCircle.rmsePx;
    const double circleTolerance =
        std::max(std::max(0.0, config.poseConsistencyMaxCircleRmseIncreasePx),
                 0.08 * referenceCircleRmse);
    const double maxFraction =
        std::clamp(config.poseConsistencyMaxCorrectionFraction, 0.0, 1.0);
    if (maxFraction <= 0.0 || !std::isfinite(referenceCircleRmse)) {
        return;
    }

    const cv::Mat firstTransform = result.globalTransforms.front().clone();
    const cv::Mat invFirstTransform = firstTransform.inv();
    const cv::Point2d firstCenter(images.empty()
                                      ? 0.0
                                      : static_cast<double>(images.front().cols) * 0.5,
                                  images.empty()
                                      ? 0.0
                                      : static_cast<double>(images.front().rows) * 0.5);
    const double denom = static_cast<double>(result.globalTransforms.size() - 1);

    const double baseObjective =
        poseConsistencyObjective(result, referenceCircleRmse, targetClosurePx, circleTolerance);
    StandardSphereLoopResult best = result;
    best.globalTransforms = cloneTransforms(result.globalTransforms);
    double bestObjective = baseObjective;
    double bestFraction = 0.0;

    std::vector<double> fractions = {
        maxFraction,
        std::min(maxFraction, 0.95),
        std::min(maxFraction, 0.925),
        std::min(maxFraction, 0.90),
        std::min(maxFraction, 0.875),
        std::min(maxFraction, 0.85),
        std::min(maxFraction, 0.80),
        std::min(maxFraction, 0.75),
        std::min(maxFraction, 0.70),
        std::min(maxFraction, 0.60)
    };
    std::sort(fractions.begin(), fractions.end(), std::greater<double>());
    fractions.erase(std::unique(fractions.begin(), fractions.end()), fractions.end());

    const std::array<cv::Point2d, 2> anchors = {
        cv::Point2d(0.0, 0.0),
        firstCenter
    };

    for (const cv::Point2d& anchor : anchors) {
        for (double fraction : fractions) {
            if (fraction <= 0.0) {
                continue;
            }

            StandardSphereLoopResult trial = result;
            trial.globalTransforms = cloneTransforms(result.globalTransforms);
            for (std::size_t i = 1; i < trial.globalTransforms.size(); ++i) {
                const double t = static_cast<double>(i) / denom;
                const cv::Mat localCorrection =
                    anchoredRigidCorrectionMatrix(anchor,
                                                  -fraction * t * residual.x,
                                                  -fraction * t * residual.y,
                                                  -fraction * t * closureRotationDeg);
                const cv::Mat worldCorrection =
                    firstTransform * localCorrection * invFirstTransform;
                trial.globalTransforms[i] = worldCorrection * trial.globalTransforms[i];
            }

            evaluateWithGlobalTransforms(trial, images, edges);
            const double circleDelta = trial.stitchedCircle.rmsePx - referenceCircleRmse;
            if (circleDelta > circleTolerance + 1e-9) {
                continue;
            }
            if (trial.closureResidualTranslationPx > result.closureResidualTranslationPx + 1e-9 ||
                trial.closureCenterDriftPx > result.closureCenterDriftPx + 1e-9) {
                continue;
            }

            const double objective =
                poseConsistencyObjective(trial, referenceCircleRmse, targetClosurePx, circleTolerance) +
                2.50 * std::max(0.0, circleDelta) +
                0.02 * fraction * fraction * residualNorm * residualNorm +
                0.10 * fraction * fraction * closureRotationDeg * closureRotationDeg;
            const bool reachesTarget =
                trial.closureResidualTranslationPx <= targetClosurePx &&
                trial.closureCenterDriftPx <= targetClosurePx;
            const bool bestReachesTarget =
                best.closureResidualTranslationPx <= targetClosurePx &&
                best.closureCenterDriftPx <= targetClosurePx;
            if ((reachesTarget && !bestReachesTarget) ||
                (reachesTarget == bestReachesTarget && objective + 1e-9 < bestObjective)) {
                best = trial;
                bestObjective = objective;
                bestFraction = fraction;
            }
        }
    }

    if (bestFraction <= 0.0 ||
        best.closureResidualTranslationPx >= result.closureResidualTranslationPx - 1e-9) {
        return;
    }

    const double finalCircleDelta = best.stitchedCircle.rmsePx - referenceCircleRmse;
    const double finalObjective =
        poseConsistencyObjective(best, referenceCircleRmse, targetClosurePx, circleTolerance);
    result = std::move(best);
    result.softGlobalDriftOptimizationIterations = 1;
    result.softGlobalDriftOptimizationAcceptedFraction = bestFraction;
    result.softGlobalDriftOptimizationMaxStepCorrectionPx =
        bestFraction * residualNorm / denom;
    result.softGlobalDriftOptimizationMaxRotationCorrectionDeg =
        bestFraction * std::abs(closureRotationDeg);
    result.softGlobalDriftOptimizationTotalCorrectionPx =
        bestFraction * residualNorm;
    result.softGlobalDriftOptimizationCostImprovement =
        std::max(0.0, baseObjective - finalObjective);
    result.softGlobalDriftOptimizationCircleRmseChangePx = finalCircleDelta;
}

cv::Mat relativeMatrix(const PairLoopRecord& record)
{
    cv::Mat rotation = cv::getRotationMatrix2D(record.rotationCenter, record.transform.da, 1.0);
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    rotation.copyTo(matrix(cv::Rect(0, 0, 3, 2)));
    matrix.at<double>(0, 2) += record.transform.dx;
    matrix.at<double>(1, 2) += record.transform.dy;
    return matrix;
}

cv::Point2d transformPoint(const cv::Mat& matrix, const cv::Point2d& point)
{
    const double x = matrix.at<double>(0, 0) * point.x +
                     matrix.at<double>(0, 1) * point.y +
                     matrix.at<double>(0, 2);
    const double y = matrix.at<double>(1, 0) * point.x +
                     matrix.at<double>(1, 1) * point.y +
                     matrix.at<double>(1, 2);
    return {x, y};
}

cv::Scalar paletteColor(std::size_t index)
{
    static const std::array<cv::Scalar, 12> kPalette = {
        cv::Scalar(66, 135, 245),
        cv::Scalar(35, 170, 85),
        cv::Scalar(235, 138, 52),
        cv::Scalar(205, 80, 170),
        cv::Scalar(55, 180, 210),
        cv::Scalar(225, 80, 70),
        cv::Scalar(150, 120, 235),
        cv::Scalar(95, 160, 45),
        cv::Scalar(230, 180, 55),
        cv::Scalar(70, 110, 170),
        cv::Scalar(180, 95, 45),
        cv::Scalar(80, 150, 150)
    };
    return kPalette[index % kPalette.size()];
}

cv::Mat cropWhiteMargin(const cv::Mat& image, int margin = 10, int threshold = 250)
{
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat mask;
    cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(mask, nonZeroPoints);
    if (nonZeroPoints.empty()) {
        return image.clone();
    }

    cv::Rect bounds = cv::boundingRect(nonZeroPoints);
    bounds.x = std::max(0, bounds.x - margin);
    bounds.y = std::max(0, bounds.y - margin);
    bounds.width = std::min(image.cols - bounds.x, bounds.width + margin * 2);
    bounds.height = std::min(image.rows - bounds.y, bounds.height + margin * 2);
    return image(bounds).clone();
}

void drawInfoBox(cv::Mat& canvas,
                 const std::string& text,
                 const cv::Point& anchor,
                 double fontScale,
                 const cv::Scalar& textColor,
                 const cv::Scalar& fillColor = cv::Scalar(252, 252, 252),
                 const cv::Scalar& borderColor = cv::Scalar(218, 218, 218),
                 int thickness = 1)
{
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    const int paddingX = 10;
    const int paddingY = 8;
    cv::Rect box(anchor.x,
                 anchor.y - textSize.height - paddingY,
                 textSize.width + paddingX * 2,
                 textSize.height + paddingY * 2 + baseline);
    box &= cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (box.width <= 0 || box.height <= 0) {
        return;
    }

    cv::rectangle(canvas, box, fillColor, cv::FILLED, cv::LINE_AA);
    cv::rectangle(canvas, box, borderColor, 1, cv::LINE_AA);
    const cv::Point textOrigin(box.x + paddingX, box.y + paddingY + textSize.height);
    cv::putText(canvas, text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
}

void updateBounds(cv::Rect2d& bounds, const cv::Point2d& point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        return;
    }

    if (bounds.width < 0.0 || bounds.height < 0.0) {
        bounds = cv::Rect2d(point.x, point.y, 0.0, 0.0);
        return;
    }

    const double minX = std::min(bounds.x, point.x);
    const double minY = std::min(bounds.y, point.y);
    const double maxX = std::max(bounds.x + bounds.width, point.x);
    const double maxY = std::max(bounds.y + bounds.height, point.y);
    bounds = cv::Rect2d(minX, minY, maxX - minX, maxY - minY);
}

double rotationDegFromMatrix(const cv::Mat& matrix)
{
    return std::atan2(matrix.at<double>(1, 0), matrix.at<double>(0, 0)) * 180.0 / kPi;
}

cv::Mat rigidMatrix(double dx, double dy, double angleDeg)
{
    const double angleRad = angleDeg * kPi / 180.0;
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    matrix.at<double>(0, 0) = c;
    matrix.at<double>(0, 1) = -s;
    matrix.at<double>(0, 2) = dx;
    matrix.at<double>(1, 0) = s;
    matrix.at<double>(1, 1) = c;
    matrix.at<double>(1, 2) = dy;
    return matrix;
}

struct CirclePoseCorrection {
    double dx{0.0};
    double dy{0.0};
    double angleDeg{0.0};
};

struct CircleOptimizationStats {
    bool applied{false};
    int iterations{0};
    double meanCorrectionPx{0.0};
    double maxCorrectionPx{0.0};
    double maxCorrectionRotationDeg{0.0};
};

struct CircleOptimizationImageData {
    cv::Point2d anchor{};
    std::vector<WeightedPoint> basePoints;
};

cv::Point2d rotateAround(const cv::Point2d& point,
                         const cv::Point2d& anchor,
                         double angleDeg)
{
    const double angleRad = angleDeg * kPi / 180.0;
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    const double x = point.x - anchor.x;
    const double y = point.y - anchor.y;
    return {anchor.x + c * x - s * y, anchor.y + s * x + c * y};
}

cv::Point2d applyCircleCorrection(const cv::Point2d& point,
                                  const cv::Point2d& anchor,
                                  const CirclePoseCorrection& correction)
{
    cv::Point2d corrected = rotateAround(point, anchor, correction.angleDeg);
    corrected.x += correction.dx;
    corrected.y += correction.dy;
    return corrected;
}

cv::Mat centeredRigidCorrectionMatrix(const cv::Point2d& anchor,
                                      const CirclePoseCorrection& correction)
{
    const double angleRad = correction.angleDeg * kPi / 180.0;
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    matrix.at<double>(0, 0) = c;
    matrix.at<double>(0, 1) = -s;
    matrix.at<double>(1, 0) = s;
    matrix.at<double>(1, 1) = c;
    matrix.at<double>(0, 2) = anchor.x - c * anchor.x + s * anchor.y + correction.dx;
    matrix.at<double>(1, 2) = anchor.y - s * anchor.x - c * anchor.y + correction.dy;
    return matrix;
}

std::vector<WeightedPoint> samplePoints(const std::vector<WeightedPoint>& points,
                                        std::size_t maxCount)
{
    if (points.size() <= maxCount || maxCount == 0) {
        return points;
    }

    std::vector<WeightedPoint> sampled;
    sampled.reserve(maxCount);
    const double step = static_cast<double>(points.size()) / static_cast<double>(maxCount);
    for (std::size_t i = 0; i < maxCount; ++i) {
        const auto index = std::min<std::size_t>(
            points.size() - 1,
            static_cast<std::size_t>(std::floor(static_cast<double>(i) * step)));
        sampled.push_back(points[index]);
    }
    return sampled;
}

CircleFitReport fitCorrectedGlobalCircle(const std::vector<CircleOptimizationImageData>& data,
                                         const std::vector<CirclePoseCorrection>& corrections)
{
    std::vector<WeightedPoint> points;
    std::size_t total = 0;
    for (const CircleOptimizationImageData& imageData : data) {
        total += imageData.basePoints.size();
    }
    points.reserve(total);

    for (std::size_t i = 0; i < data.size() && i < corrections.size(); ++i) {
        for (const WeightedPoint& point : data[i].basePoints) {
            points.push_back({applyCircleCorrection(point.point, data[i].anchor, corrections[i]), point.weight});
        }
    }
    return fitCircleRobustWeighted(points);
}

double circleImageCost(const CircleOptimizationImageData& data,
                       const CirclePoseCorrection& correction,
                       const CircleFitReport& circle,
                       const StandardSphereLoopConfig& config)
{
    if (!circle.ok || data.basePoints.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double sumSquared = 0.0;
    double sumWeights = 0.0;
    for (const WeightedPoint& point : data.basePoints) {
        const double weight = saneWeight(point.weight);
        const cv::Point2d corrected = applyCircleCorrection(point.point, data.anchor, correction);
        const double residual = cv::norm(corrected - circle.center) - circle.radiusPx;
        sumSquared += weight * residual * residual;
        sumWeights += weight;
    }

    const double meanSquared = sumSquared / std::max(1e-9, sumWeights);
    const double translationPenalty =
        config.circleOptimizationTranslationRegularization *
        (correction.dx * correction.dx + correction.dy * correction.dy);
    const double rotationPenalty =
        config.circleOptimizationRotationRegularization *
        correction.angleDeg * correction.angleDeg;
    return meanSquared + translationPenalty + rotationPenalty;
}

CirclePoseCorrection clampCircleCorrection(CirclePoseCorrection correction,
                                           const StandardSphereLoopConfig& config)
{
    const double maxTranslation = std::max(0.0, config.circleOptimizationMaxTranslationPx);
    const double maxRotation = std::max(0.0, config.circleOptimizationMaxRotationDeg);
    correction.dx = std::clamp(correction.dx, -maxTranslation, maxTranslation);
    correction.dy = std::clamp(correction.dy, -maxTranslation, maxTranslation);
    correction.angleDeg = std::clamp(correction.angleDeg, -maxRotation, maxRotation);
    return correction;
}

void tryCircleCorrectionStep(const CircleOptimizationImageData& data,
                             const CircleFitReport& circle,
                             const StandardSphereLoopConfig& config,
                             CirclePoseCorrection& best,
                             double& bestCost,
                             int parameterIndex,
                             double delta)
{
    CirclePoseCorrection candidate = best;
    if (parameterIndex == 0) {
        candidate.dx += delta;
    } else if (parameterIndex == 1) {
        candidate.dy += delta;
    } else {
        candidate.angleDeg += delta;
    }
    candidate = clampCircleCorrection(candidate, config);
    const double cost = circleImageCost(data, candidate, circle, config);
    if (cost < bestCost) {
        best = candidate;
        bestCost = cost;
    }
}

CircleOptimizationStats optimizeTransformsByCircleConsistency(std::vector<cv::Mat>& transforms,
                                                              const std::vector<cv::Mat>& images,
                                                              const std::vector<stitch::EdgeVariants>& edges,
                                                              const StandardSphereLoopConfig& config)
{
    CircleOptimizationStats stats;
    if (transforms.size() < 2 || edges.size() < transforms.size()) {
        return stats;
    }

    std::vector<CircleOptimizationImageData> data;
    data.reserve(transforms.size());
    constexpr std::size_t kMaxSampledPointsPerImage = 1200;
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        CircleOptimizationImageData imageData;
        const double w = i < images.size() ? static_cast<double>(images[i].cols) : 0.0;
        const double h = i < images.size() ? static_cast<double>(images[i].rows) : 0.0;
        imageData.anchor = transformPoint(transforms[i], cv::Point2d(w * 0.5, h * 0.5));
        const std::vector<WeightedPoint> sampled =
            samplePoints(makeWeightedPoints(edges[i]), kMaxSampledPointsPerImage);
        imageData.basePoints.reserve(sampled.size());
        for (const WeightedPoint& point : sampled) {
            imageData.basePoints.push_back({transformPoint(transforms[i], point.point), point.weight});
        }
        data.push_back(std::move(imageData));
    }

    std::vector<CirclePoseCorrection> corrections(transforms.size());
    const int maxIterations = std::max(0, config.circleOptimizationMaxIterations);
    const std::array<double, 4> translationSteps = {1.0, 0.5, 0.25, 0.1};
    const std::array<double, 3> rotationSteps = {0.02, 0.01, 0.005};

    for (int iter = 0; iter < maxIterations; ++iter) {
        const CircleFitReport circle = fitCorrectedGlobalCircle(data, corrections);
        if (!circle.ok) {
            break;
        }

        bool improved = false;
        for (std::size_t i = 1; i < data.size(); ++i) {
            CirclePoseCorrection best = corrections[i];
            double bestCost = circleImageCost(data[i], best, circle, config);

            for (double step : translationSteps) {
                const CirclePoseCorrection before = best;
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 0, step);
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 0, -step);
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 1, step);
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 1, -step);
                improved = improved ||
                           std::abs(best.dx - before.dx) > 1e-9 ||
                           std::abs(best.dy - before.dy) > 1e-9 ||
                           std::abs(best.angleDeg - before.angleDeg) > 1e-9;
            }
            for (double step : rotationSteps) {
                const CirclePoseCorrection before = best;
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 2, step);
                tryCircleCorrectionStep(data[i], circle, config, best, bestCost, 2, -step);
                improved = improved ||
                           std::abs(best.dx - before.dx) > 1e-9 ||
                           std::abs(best.dy - before.dy) > 1e-9 ||
                           std::abs(best.angleDeg - before.angleDeg) > 1e-9;
            }
            corrections[i] = best;
        }

        stats.iterations = iter + 1;
        if (!improved) {
            break;
        }
    }

    double sumCorrection = 0.0;
    for (std::size_t i = 1; i < transforms.size(); ++i) {
        const double correctionPx = std::hypot(corrections[i].dx, corrections[i].dy);
        sumCorrection += correctionPx;
        stats.maxCorrectionPx = std::max(stats.maxCorrectionPx, correctionPx);
        stats.maxCorrectionRotationDeg =
            std::max(stats.maxCorrectionRotationDeg, std::abs(corrections[i].angleDeg));
        if (correctionPx > 1e-9 || std::abs(corrections[i].angleDeg) > 1e-9) {
            stats.applied = true;
        }
        transforms[i] = centeredRigidCorrectionMatrix(data[i].anchor, corrections[i]) * transforms[i];
    }
    stats.meanCorrectionPx = transforms.size() > 1
                                 ? sumCorrection / static_cast<double>(transforms.size() - 1)
                                 : 0.0;
    return stats;
}

void appendMetric(std::ostringstream& stream,
                  const std::string& name,
                  double value,
                  const std::string& unit,
                  const std::string& note = {})
{
    stream << name << ","
           << std::setprecision(12) << finiteOrZero(value) << ","
           << unit << ","
           << csvEscape(note) << "\n";
}

void evaluateWithGlobalTransforms(StandardSphereLoopResult& result,
                                  const std::vector<cv::Mat>& images,
                                  const std::vector<stitch::EdgeVariants>& edges)
{
    if (images.empty() || edges.empty() || result.globalTransforms.size() < images.size()) {
        return;
    }

    const cv::Mat invFirst = result.globalTransforms.front().inv();
    result.closureMatrix = invFirst * result.globalTransforms.back();
    result.measuredClosurePx = cv::Point2d(result.closureMatrix.at<double>(0, 2),
                                           result.closureMatrix.at<double>(1, 2));
    result.closureResidualPx = result.measuredClosurePx - result.expectedClosurePx;
    result.closureTranslationPx = std::hypot(result.measuredClosurePx.x, result.measuredClosurePx.y);
    result.closureResidualTranslationPx = cv::norm(result.closureResidualPx);
    result.closureResidualTranslationMm =
        result.evaluationPixelSizeMm > 0.0 ? result.closureResidualTranslationPx *
                                                 result.evaluationPixelSizeMm
                                            : 0.0;
    result.closureRotationDeg = rotationDegFromMatrix(result.closureMatrix);

    const cv::Point2d firstCenter(images.front().cols / 2.0, images.front().rows / 2.0);
    result.closureCenterDriftPx = cv::norm(transformPoint(result.closureMatrix, firstCenter) - firstCenter);

    const double width = static_cast<double>(images.front().cols);
    const double height = static_cast<double>(images.front().rows);
    const std::array<cv::Point2d, 5> probes = {
        cv::Point2d(0.0, 0.0),
        cv::Point2d(width, 0.0),
        cv::Point2d(width, height),
        cv::Point2d(0.0, height),
        firstCenter
    };
    double probeSquared = 0.0;
    double probeMax = 0.0;
    for (const cv::Point2d& probe : probes) {
        const double drift = cv::norm(transformPoint(result.closureMatrix, probe) - probe);
        probeSquared += drift * drift;
        probeMax = std::max(probeMax, drift);
    }
    result.closureCornerRmsPx = std::sqrt(probeSquared / static_cast<double>(probes.size()));
    result.closureCornerMaxPx = probeMax;

    result.perImageCircles.clear();
    result.perImageCircles.reserve(edges.size());
    std::vector<WeightedPoint> stitchedPoints;
    for (std::size_t i = 0; i < edges.size() && i < result.globalTransforms.size(); ++i) {
        result.perImageCircles.push_back(fitCircleRobust(edges[i]));
        stitchedPoints.reserve(stitchedPoints.size() + edges[i].raw.size());
        const std::vector<WeightedPoint> weightedPoints = makeWeightedPoints(edges[i]);
        for (const WeightedPoint& point : weightedPoints) {
            stitchedPoints.push_back({transformPoint(result.globalTransforms[i], point.point), point.weight});
        }
    }
    result.stitchedCircle = fitCircleRobustWeighted(stitchedPoints);
}

} // namespace

StandardSphereLoopResult runStandardSphereLoopExperiment(const std::vector<cv::Mat>& images,
                                                        const std::vector<stitch::EdgeVariants>& edges,
                                                        const StandardSphereLoopConfig& config)
{
    StandardSphereLoopResult result;
    if (images.size() < 2 || edges.size() != images.size()) {
        result.message = "standard sphere loop experiment needs at least two preprocessed images";
        return result;
    }
    result.evaluationPixelSizeMm = effectivePixelSizeMm(images.front(), config);

    const std::vector<MotionScheduleEntry> motionSchedule =
        buildMotionSchedule(images.size(), config);

    double approxShiftX = 0.0;
    double approxShiftY = 0.0;
    double approxAngleDeg = 0.0;
    bool hasReliableMotionPrior = false;
    stitch::MotionPriorDirection previousDirection = stitch::MotionPriorDirection::Auto;

    result.pairRecords.reserve(images.size());
    cv::Point2d cumulativeMeasuredPx(0.0, 0.0);
    cv::Point2d cumulativeExpectedPx(0.0, 0.0);
    for (std::size_t i = 0; i < images.size(); ++i) {
        const std::size_t targetIndex = (i + 1) % images.size();
        const stitch::MotionPriorDirection expectedDirection =
            i < motionSchedule.size() ? motionSchedule[i].direction : config.pipelineConfig.directionConstraint;
        const cv::Point2d expectedStepPx =
            expectedStepPxForDirection(images[i], config, expectedDirection);
        const cv::Point2d expectedStepMm =
            expectedStepMmForDirection(config, expectedDirection);
        if (i == 0 || expectedDirection != previousDirection || !hasReliableMotionPrior) {
            approxShiftX = expectedStepPx.x;
            approxShiftY = expectedStepPx.y;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
        }

        const cv::Point2d center(images[i].cols / 2.0, images[i].rows / 2.0);
        double searchRangeX = 0.0;
        double searchRangeY = 0.0;

        stitch::TransformResult transform =
            stitch::matchOnePair(edges[i],
                                 edges[targetIndex],
                                 center,
                                 approxShiftX,
                                 approxShiftY,
                                 approxAngleDeg,
                                 hasReliableMotionPrior,
                                 config.pipelineConfig.baseSearchRange,
                                 expectedDirection,
                                 config.pipelineConfig.rotationSearchMinDeg,
                                 config.pipelineConfig.rotationSearchMaxDeg,
                                 config.pipelineConfig.rotationSearchStepDeg,
                                 config.pipelineConfig.tangentResidualCostWeight,
                                 config.pipelineConfig.tangentCorrelationCostWeight,
                                 false,
                                 0.0,
                                 0.0,
                                 searchRangeX,
                                 searchRangeY);

        PairLoopRecord record;
        record.stepIndex = i + 1;
        record.referenceIndex = i;
        record.targetIndex = targetIndex;
        record.segmentIndex = i < motionSchedule.size() ? motionSchedule[i].segmentIndex : 0;
        record.expectedDirection = directionLabel(expectedDirection);
        record.rotationCenter = center;
        record.expectedStepPx = expectedStepPx;
        record.expectedStepMm = expectedStepMm;
        record.measuredStepMm =
            result.evaluationPixelSizeMm > 0.0 ? cv::Point2d(transform.dx * result.evaluationPixelSizeMm,
                                                             transform.dy * result.evaluationPixelSizeMm)
                                               : cv::Point2d(0.0, 0.0);
        cumulativeMeasuredPx += cv::Point2d(transform.dx, transform.dy);
        cumulativeExpectedPx += expectedStepPx;
        record.cumulativeMeasuredPx = cumulativeMeasuredPx;
        record.cumulativeExpectedPx = cumulativeExpectedPx;
        record.cumulativeErrorPx = cumulativeMeasuredPx - cumulativeExpectedPx;
        record.localTranslationErrorPx =
            cv::norm(cv::Point2d(transform.dx, transform.dy) - expectedStepPx);
        record.localTranslationErrorMm =
            result.evaluationPixelSizeMm > 0.0 ? record.localTranslationErrorPx * result.evaluationPixelSizeMm
                                                : 0.0;
        record.cumulativeTranslationErrorPx = cv::norm(record.cumulativeErrorPx);
        record.cumulativeTranslationErrorMm =
            result.evaluationPixelSizeMm > 0.0 ? record.cumulativeTranslationErrorPx *
                                                     result.evaluationPixelSizeMm
                                                : 0.0;
        record.searchRangeX = searchRangeX;
        record.searchRangeY = searchRangeY;
        record.transform = std::move(transform);
        result.pairRecords.push_back(std::move(record));

        const stitch::TransformResult& accepted = result.pairRecords.back().transform;
        if (isReliablePair(accepted)) {
            constexpr double kPriorAlpha = 0.7;
            approxShiftX = kPriorAlpha * accepted.dx + (1.0 - kPriorAlpha) * approxShiftX;
            approxShiftY = kPriorAlpha * accepted.dy + (1.0 - kPriorAlpha) * approxShiftY;
            approxAngleDeg = accepted.da;
            hasReliableMotionPrior = true;
        } else {
            approxShiftX = expectedStepPx.x;
            approxShiftY = expectedStepPx.y;
            approxAngleDeg = 0.0;
            hasReliableMotionPrior = false;
        }
        previousDirection = expectedDirection;
    }

    result.segmentRecords.clear();
    for (const PairLoopRecord& record : result.pairRecords) {
        if (result.segmentRecords.empty() ||
            result.segmentRecords.back().segmentIndex != record.segmentIndex) {
            SegmentLoopRecord segment;
            segment.segmentIndex = record.segmentIndex;
            segment.expectedDirection = record.expectedDirection;
            segment.startStep = record.stepIndex;
            segment.endStep = record.stepIndex;
            result.segmentRecords.push_back(segment);
        }

        SegmentLoopRecord& segment = result.segmentRecords.back();
        segment.endStep = record.stepIndex;
        ++segment.pairCount;
        segment.measuredDeltaPx += cv::Point2d(record.transform.dx, record.transform.dy);
        segment.expectedDeltaPx += record.expectedStepPx;
        segment.residualDeltaPx = segment.measuredDeltaPx - segment.expectedDeltaPx;
        segment.residualTranslationPx = cv::norm(segment.residualDeltaPx);
        segment.residualTranslationMm =
            result.evaluationPixelSizeMm > 0.0 ? segment.residualTranslationPx *
                                                     result.evaluationPixelSizeMm
                                                : 0.0;
    }

    result.globalTransforms.clear();
    result.globalTransforms.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
        result.globalTransforms.push_back(identity.clone());
    }
    result.expectedGlobalTranslationsPx.assign(images.size(), cv::Point2d(0.0, 0.0));
    for (std::size_t i = 0; i + 1 < images.size(); ++i) {
        result.globalTransforms[i + 1] = result.globalTransforms[i] * relativeMatrix(result.pairRecords[i]);
        result.expectedGlobalTranslationsPx[i + 1] = result.pairRecords[i].cumulativeExpectedPx;
    }

    result.expectedClosurePx = result.pairRecords.back().cumulativeExpectedPx;
    evaluateWithGlobalTransforms(result, images, edges);

    result.ok = true;
    return result;
}

StandardSphereLoopResult evaluateStandardSphereStitchingResult(const std::vector<cv::Mat>& images,
                                                               const std::vector<stitch::EdgeVariants>& edges,
                                                               const stitch::StitchingResult& stitching,
                                                               const StandardSphereLoopConfig& config)
{
    StandardSphereLoopResult result;
    if (images.size() < 2 || edges.size() != images.size()) {
        result.message = "standard sphere stitching evaluation needs at least two preprocessed images";
        return result;
    }
    if (stitching.imageTransforms.size() < images.size()) {
        result.message = "stitching pipeline did not return one global transform per image";
        return result;
    }

    result.evaluationPixelSizeMm = effectivePixelSizeMm(images.front(), config);
    const std::vector<MotionScheduleEntry> motionSchedule =
        buildMotionSchedule(stitching.steps.size(), config);
    std::vector<CircleFitReport> sourceCircles;
    sourceCircles.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        sourceCircles.push_back(fitCircleRobust(edge));
    }

    result.pairRecords.reserve(stitching.steps.size());
    cv::Point2d cumulativeMeasuredPx(0.0, 0.0);
    cv::Point2d cumulativeExpectedPx(0.0, 0.0);
    for (std::size_t i = 0; i < stitching.steps.size(); ++i) {
        const stitch::StitchStepRecord& step = stitching.steps[i];
        if (step.referenceImageIndex >= images.size() || step.targetImageIndex >= images.size()) {
            continue;
        }

        const stitch::MotionPriorDirection expectedDirection =
            i < motionSchedule.size() ? motionSchedule[i].direction : config.pipelineConfig.directionConstraint;
        const cv::Point2d expectedStepPx =
            expectedStepPxForDirection(images[step.referenceImageIndex], config, expectedDirection);
        const cv::Point2d expectedStepMm =
            expectedStepMmForDirection(config, expectedDirection);
        const cv::Point2d measuredStep(step.transform.dx, step.transform.dy);

        PairLoopRecord record;
        record.stepIndex = step.stepIndex;
        record.referenceIndex = step.referenceImageIndex;
        record.targetIndex = step.targetImageIndex;
        record.segmentIndex = i < motionSchedule.size() ? motionSchedule[i].segmentIndex : 0;
        record.expectedDirection = directionLabel(expectedDirection);
        record.rotationCenter = cv::Point2d(images[step.referenceImageIndex].cols / 2.0,
                                            images[step.referenceImageIndex].rows / 2.0);
        record.expectedStepPx = expectedStepPx;
        record.expectedStepMm = expectedStepMm;
        record.measuredStepMm =
            result.evaluationPixelSizeMm > 0.0 ? cv::Point2d(measuredStep.x * result.evaluationPixelSizeMm,
                                                             measuredStep.y * result.evaluationPixelSizeMm)
                                               : cv::Point2d(0.0, 0.0);
        cumulativeMeasuredPx += measuredStep;
        cumulativeExpectedPx += expectedStepPx;
        record.cumulativeMeasuredPx = cumulativeMeasuredPx;
        record.cumulativeExpectedPx = cumulativeExpectedPx;
        record.cumulativeErrorPx = cumulativeMeasuredPx - cumulativeExpectedPx;
        record.localTranslationErrorPx = cv::norm(measuredStep - expectedStepPx);
        record.localTranslationErrorMm =
            result.evaluationPixelSizeMm > 0.0 ? record.localTranslationErrorPx * result.evaluationPixelSizeMm
                                                : 0.0;
        record.cumulativeTranslationErrorPx = cv::norm(record.cumulativeErrorPx);
        record.cumulativeTranslationErrorMm =
            result.evaluationPixelSizeMm > 0.0 ? record.cumulativeTranslationErrorPx *
                                                     result.evaluationPixelSizeMm
                                                : 0.0;
        record.searchRangeX = step.searchRangeX;
        record.searchRangeY = step.searchRangeY;
        record.transform = step.transform;
        result.pairRecords.push_back(std::move(record));
    }

    if (config.enableSphereBadStepGeometryRescue) {
        applyStandardSphereBadStepGeometryRescue(result.pairRecords, sourceCircles, result);
    }
    if (config.enableGlobalConsistencyReregistration) {
        applyGlobalConsistencyReregistration(result.pairRecords,
                                             edges,
                                             stitching.imageTransforms.front(),
                                             images.size(),
                                             result.evaluationPixelSizeMm,
                                             result);
    }
    if (config.enableBadStepLocalRefinement) {
        applyBadStepLocalRefinement(result.pairRecords,
                                    edges,
                                    stitching.imageTransforms.front(),
                                    images.size(),
                                    result.evaluationPixelSizeMm,
                                    result);
    }
    if (false && config.enableSoftGlobalDriftOptimization) {
        applySoftGlobalDriftOptimization(result.pairRecords,
                                         edges,
                                         stitching.imageTransforms.front(),
                                         images.size(),
                                         result.evaluationPixelSizeMm,
                                         result);
    }
    if (result.sphereBadStepGeometryRescueCount > 0 ||
        result.globalConsistencyReregistrationCount > 0 ||
        result.badStepLocalRefinementCount > 0 ||
        result.softGlobalDriftOptimizationIterations > 0) {
        recomputePairAccumulations(result.pairRecords, result.evaluationPixelSizeMm);
    }

    result.segmentRecords.clear();
    for (const PairLoopRecord& record : result.pairRecords) {
        if (result.segmentRecords.empty() ||
            result.segmentRecords.back().segmentIndex != record.segmentIndex) {
            SegmentLoopRecord segment;
            segment.segmentIndex = record.segmentIndex;
            segment.expectedDirection = record.expectedDirection;
            segment.startStep = record.stepIndex;
            segment.endStep = record.stepIndex;
            result.segmentRecords.push_back(segment);
        }

        SegmentLoopRecord& segment = result.segmentRecords.back();
        segment.endStep = record.stepIndex;
        ++segment.pairCount;
        segment.measuredDeltaPx += cv::Point2d(record.transform.dx, record.transform.dy);
        segment.expectedDeltaPx += record.expectedStepPx;
        segment.residualDeltaPx = segment.measuredDeltaPx - segment.expectedDeltaPx;
        segment.residualTranslationPx = cv::norm(segment.residualDeltaPx);
        segment.residualTranslationMm =
            result.evaluationPixelSizeMm > 0.0 ? segment.residualTranslationPx *
                                                     result.evaluationPixelSizeMm
                                                : 0.0;
    }

    result.globalTransforms.clear();
    result.globalTransforms.reserve(images.size());
    if (result.sphereBadStepGeometryRescueCount > 0 ||
        result.globalConsistencyReregistrationCount > 0 ||
        result.badStepLocalRefinementCount > 0 ||
        result.softGlobalDriftOptimizationIterations > 0) {
        result.globalTransforms.push_back(stitching.imageTransforms.front().clone());
        for (std::size_t i = 0; i < result.pairRecords.size() && i + 1 < images.size(); ++i) {
            result.globalTransforms.push_back(result.globalTransforms.back() * relativeMatrix(result.pairRecords[i]));
        }
        while (result.globalTransforms.size() < images.size()) {
            result.globalTransforms.push_back(stitching.imageTransforms[result.globalTransforms.size()].clone());
        }
    } else {
        for (std::size_t i = 0; i < images.size(); ++i) {
            result.globalTransforms.push_back(stitching.imageTransforms[i].clone());
        }
    }

    result.expectedGlobalTranslationsPx.assign(images.size(), cv::Point2d(0.0, 0.0));
    for (std::size_t i = 0; i < result.pairRecords.size() && i + 1 < result.expectedGlobalTranslationsPx.size(); ++i) {
        result.expectedGlobalTranslationsPx[i + 1] = result.pairRecords[i].cumulativeExpectedPx;
    }
    result.expectedClosurePx = result.pairRecords.empty()
                                   ? cv::Point2d(0.0, 0.0)
                                   : result.pairRecords.back().cumulativeExpectedPx;

    evaluateWithGlobalTransforms(result, images, edges);
    result.preOptimizationClosureTranslationPx = result.closureTranslationPx;
    result.preOptimizationClosureRotationDeg = result.closureRotationDeg;
    result.preOptimizationStitchedCircleRmsePx = result.stitchedCircle.rmsePx;
    result.preOptimizationStitchedCircleMeanAbsPx = result.stitchedCircle.meanAbsPx;

    StandardSphereLoopResult bestBranch = result;
    bestBranch.globalTransforms = cloneTransforms(result.globalTransforms);
    if (config.enableSoftGlobalDriftOptimization) {
        applySoftGlobalTransformDrift(bestBranch, images, edges, config);
        evaluateWithGlobalTransforms(bestBranch, images, edges);
    }

    if (config.enableGlobalLoopClosureOptimization) {
        std::vector<cv::Mat> trialTransforms = cloneTransforms(result.globalTransforms);
        const double beforeCircleRmse = result.stitchedCircle.rmsePx;
        const double beforeClosure = result.closureResidualTranslationPx;
        const double beforeRotation = std::abs(result.closureRotationDeg);
        const CircleOptimizationStats stats =
            optimizeTransformsByCircleConsistency(trialTransforms, images, edges, config);

        StandardSphereLoopResult circleBranch = result;
        circleBranch.globalTransforms = std::move(trialTransforms);
        evaluateWithGlobalTransforms(circleBranch, images, edges);

        const bool circleImproved =
            circleBranch.stitchedCircle.ok &&
            circleBranch.stitchedCircle.rmsePx + 1e-6 < beforeCircleRmse;
        const bool closureGuard =
            circleBranch.closureResidualTranslationPx <= std::max(25.0, 3.0 * beforeClosure);
        const bool rotationGuard =
            std::abs(circleBranch.closureRotationDeg) <= std::max(0.5, beforeRotation + 0.35);

        if (stats.applied && circleImproved && closureGuard && rotationGuard) {
            circleBranch.globalLoopClosureOptimized = true;
            circleBranch.circleOptimizationIterations = stats.iterations;
            circleBranch.circleOptimizationMeanCorrectionPx = stats.meanCorrectionPx;
            circleBranch.circleOptimizationMaxCorrectionPx = stats.maxCorrectionPx;
            circleBranch.circleOptimizationMaxCorrectionRotationDeg = stats.maxCorrectionRotationDeg;
            if (config.enableSoftGlobalDriftOptimization) {
                applySoftGlobalTransformDrift(circleBranch, images, edges, config);
                evaluateWithGlobalTransforms(circleBranch, images, edges);
            }
            if (standardSphereBranchSelectionCost(circleBranch, config) + 1e-9 <
                standardSphereBranchSelectionCost(bestBranch, config)) {
                bestBranch = std::move(circleBranch);
            }
        }
    }

    if (bestBranch.globalTransforms.size() >= 2 &&
        std::isfinite(bestBranch.stitchedCircle.rmsePx)) {
        StandardSphereLoopResult polished = bestBranch;
        polished.globalTransforms = cloneTransforms(bestBranch.globalTransforms);

        StandardSphereLoopConfig polishConfig = config;
        polishConfig.circleOptimizationMaxTranslationPx =
            std::min(config.circleOptimizationMaxTranslationPx, 2.0);
        polishConfig.circleOptimizationMaxRotationDeg =
            std::min(config.circleOptimizationMaxRotationDeg, 0.05);
        polishConfig.circleOptimizationMaxIterations =
            std::min(std::max(1, config.circleOptimizationMaxIterations), 3);

        const CircleOptimizationStats polishStats =
            optimizeTransformsByCircleConsistency(polished.globalTransforms, images, edges, polishConfig);
        if (polishStats.applied) {
            evaluateWithGlobalTransforms(polished, images, edges);
            const bool closureGuard =
                polished.closureResidualTranslationPx <=
                    std::max(1.0, bestBranch.closureResidualTranslationPx + 0.20);
            const bool circleGuard =
                polished.stitchedCircle.ok &&
                polished.stitchedCircle.rmsePx + 1e-6 <
                    bestBranch.stitchedCircle.rmsePx - 0.10;
            const bool driftGuard =
                polished.closureCenterDriftPx <= bestBranch.closureCenterDriftPx + 1.0;
            if (closureGuard && circleGuard && driftGuard &&
                standardSphereBranchSelectionCost(polished, config) + 1e-9 <
                    standardSphereBranchSelectionCost(bestBranch, config)) {
                polished.globalLoopClosureOptimized = bestBranch.globalLoopClosureOptimized ||
                                                     polishStats.applied;
                polished.circleOptimizationIterations += polishStats.iterations;
                polished.circleOptimizationMeanCorrectionPx =
                    std::max(polished.circleOptimizationMeanCorrectionPx,
                             polishStats.meanCorrectionPx);
                polished.circleOptimizationMaxCorrectionPx =
                    std::max(polished.circleOptimizationMaxCorrectionPx,
                             polishStats.maxCorrectionPx);
                polished.circleOptimizationMaxCorrectionRotationDeg =
                    std::max(polished.circleOptimizationMaxCorrectionRotationDeg,
                             polishStats.maxCorrectionRotationDeg);
                bestBranch = std::move(polished);
            }
        }
    }

    result = std::move(bestBranch);
    result.ok = true;
    return result;
}

std::vector<stitch::MotionPriorDirection> buildClockwiseStepDirections(std::size_t pairCount,
                                                                       const StandardSphereLoopConfig& config)
{
    const std::vector<MotionScheduleEntry> schedule = buildMotionSchedule(pairCount, config);
    std::vector<stitch::MotionPriorDirection> directions;
    directions.reserve(schedule.size());
    for (const MotionScheduleEntry& entry : schedule) {
        directions.push_back(entry.direction);
    }
    return directions;
}

std::vector<cv::Point2d> buildCircleCenterTranslationPriors(const std::vector<stitch::EdgeVariants>& edges)
{
    std::vector<CircleFitReport> fits;
    fits.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        fits.push_back(fitCircleRobust(edge));
    }

    std::vector<cv::Point2d> priors;
    if (fits.size() < 2) {
        return priors;
    }
    priors.reserve(fits.size() - 1);
    for (std::size_t i = 0; i + 1 < fits.size(); ++i) {
        if (fits[i].ok && fits[i + 1].ok) {
            priors.push_back(fits[i].center - fits[i + 1].center);
        } else {
            priors.push_back(cv::Point2d(0.0, 0.0));
        }
    }
    return priors;
}

std::string buildPairCsv(const StandardSphereLoopResult& result)
{
    std::ostringstream stream;
    stream << "step,reference_index,target_index,segment,direction,expected_dx_px,expected_dy_px,"
              "expected_dx_mm,expected_dy_mm,dx_px,dy_px,dx_mm,dy_mm,local_error_px,local_error_mm,"
              "cumulative_expected_x_px,cumulative_expected_y_px,"
              "cumulative_measured_x_px,cumulative_measured_y_px,cumulative_error_px,cumulative_error_mm,"
              "da_deg,score,"
              "search_range_x_px,search_range_y_px,"
              "overlap_count,inlier_count,inlier_ratio,trimmed_overlap_count,trimmed_overlap_ratio,"
              "normal_rmse_px,normal_all_rmse_px,normal_inlier_rmse_px,normal_trimmed_rmse_px,"
              "normal_mean_abs_px,normal_p95_abs_px,trimmed_normal_mean_abs_px,trimmed_normal_p95_abs_px,"
              "tangent_rmse_px,tangent_corr_inlier\n";
    for (const PairLoopRecord& record : result.pairRecords) {
        const stitch::TransformResult& t = record.transform;
        const stitch::ResidualStatistics& normal = preferredNormalStats(t);
        const stitch::ResidualStatistics& normalAll = t.metrics.normalAll;
        const stitch::ResidualStatistics& normalInlier = t.metrics.normalInlier;
        const stitch::ResidualStatistics& normalTrimmed = t.metrics.normalTrimmed;
        const stitch::ResidualStatistics& tangent =
            t.metrics.tangentInlier.valid() ? t.metrics.tangentInlier : t.metrics.tangentAll;
        stream << record.stepIndex << ","
               << record.referenceIndex << ","
               << record.targetIndex << ","
               << record.segmentIndex << ","
               << record.expectedDirection << ","
               << record.expectedStepPx.x << ","
               << record.expectedStepPx.y << ","
               << record.expectedStepMm.x << ","
               << record.expectedStepMm.y << ","
               << t.dx << ","
               << t.dy << ","
               << record.measuredStepMm.x << ","
               << record.measuredStepMm.y << ","
               << record.localTranslationErrorPx << ","
               << record.localTranslationErrorMm << ","
               << record.cumulativeExpectedPx.x << ","
               << record.cumulativeExpectedPx.y << ","
               << record.cumulativeMeasuredPx.x << ","
               << record.cumulativeMeasuredPx.y << ","
               << record.cumulativeTranslationErrorPx << ","
               << record.cumulativeTranslationErrorMm << ","
               << t.da << ","
               << t.score << ","
               << record.searchRangeX << ","
               << record.searchRangeY << ","
               << t.metrics.overlapCount << ","
               << t.metrics.inlierCount << ","
               << t.metrics.inlierRatio << ","
               << t.metrics.trimmedOverlapCount << ","
               << t.metrics.trimmedOverlapRatio << ","
               << normal.rmse << ","
               << normalAll.rmse << ","
               << normalInlier.rmse << ","
               << normalTrimmed.rmse << ","
               << normal.meanAbs << ","
               << normal.p95Abs << ","
               << normalTrimmed.meanAbs << ","
               << normalTrimmed.p95Abs << ","
               << tangent.rmse << ","
               << t.metrics.tangentCorrInlier << "\n";
    }
    return stream.str();
}

std::string buildBadStepDiagnosticsCsv(const StandardSphereLoopResult& result,
                                       const std::vector<stitch::EdgeVariants>& edges)
{
    const std::vector<cv::Point2d> circlePriors = buildCircleCenterTranslationPriors(edges);

    auto appendTransformDiagnostics = [](std::ostringstream& stream,
                                         const PairLoopRecord& record,
                                         const stitch::TransformResult& transform,
                                         const std::string& role,
                                         int rank,
                                         const cv::Point2d& circlePrior,
                                         const cv::Point2d& selectedShift,
                                         double trialClosureResidualPx,
                                         double trialGlobalCost) {
        const stitch::ResidualStatistics& normalPreferred =
            transform.metrics.normalInlier.valid() ? transform.metrics.normalInlier : transform.metrics.normalAll;
        const stitch::ResidualStatistics& tangentPreferred =
            transform.metrics.tangentInlier.valid() ? transform.metrics.tangentInlier : transform.metrics.tangentAll;
        const double normalRmse = normalPreferred.valid() ? normalPreferred.rmse : 0.0;
        const double tangentRmse = tangentPreferred.valid() ? tangentPreferred.rmse : 0.0;
        const cv::Point2d shift(transform.dx, transform.dy);
        const cv::Point2d expectedDelta = shift - record.expectedStepPx;
        const cv::Point2d circleDelta = shift - circlePrior;
        const cv::Point2d selectedDelta = shift - selectedShift;

        stream << record.stepIndex << ","
               << record.referenceIndex << ","
               << record.targetIndex << ","
               << record.expectedDirection << ","
               << role << ","
               << rank << ","
               << transform.direction << ","
               << (transform.axis == stitch::AlignmentAxis::X ? "X" : "Y") << ","
               << transform.dx << ","
               << transform.dy << ","
               << transform.da << ","
               << transform.score << ","
               << transform.normalMatchCost << ","
               << transform.tangentResidualMatchCost << ","
               << transform.tangentCorrelationMatchCost << ","
               << transform.directionPenaltyMatchCost << ","
               << transform.metrics.overlapCount << ","
               << transform.metrics.inlierCount << ","
               << transform.metrics.inlierRatio << ","
               << transform.metrics.trimmedOverlapCount << ","
               << transform.metrics.trimmedOverlapRatio << ","
               << transform.metrics.normalAll.rmse << ","
               << transform.metrics.normalInlier.rmse << ","
               << transform.metrics.normalTrimmed.rmse << ","
               << normalRmse << ","
               << transform.metrics.tangentAll.rmse << ","
               << transform.metrics.tangentInlier.rmse << ","
               << transform.metrics.tangentTrimmed.rmse << ","
               << tangentRmse << ","
               << transform.metrics.tangentCorrAll << ","
               << transform.metrics.tangentCorrInlier << ","
               << transform.metrics.tangentCorrTrimmed << ","
               << record.expectedStepPx.x << ","
               << record.expectedStepPx.y << ","
               << expectedDelta.x << ","
               << expectedDelta.y << ","
               << cv::norm(expectedDelta) << ","
               << circlePrior.x << ","
               << circlePrior.y << ","
               << circleDelta.x << ","
               << circleDelta.y << ","
               << cv::norm(circleDelta) << ","
               << selectedDelta.x << ","
               << selectedDelta.y << ","
               << cv::norm(selectedDelta) << ","
               << trialClosureResidualPx << ","
               << trialGlobalCost << "\n";
    };

    std::ostringstream stream;
    stream << "step,reference_index,target_index,expected_direction,role,rank,direction,axis,"
              "dx_px,dy_px,da_deg,score,normal_match_cost,tangent_residual_match_cost,"
              "tangent_correlation_match_cost,direction_penalty_match_cost,"
              "overlap_count,inlier_count,inlier_ratio,trimmed_overlap_count,trimmed_overlap_ratio,"
              "normal_all_rmse_px,normal_inlier_rmse_px,normal_trimmed_rmse_px,normal_preferred_rmse_px,"
              "tangent_all_rmse_px,tangent_inlier_rmse_px,tangent_trimmed_rmse_px,tangent_preferred_rmse_px,"
              "tangent_corr_all,tangent_corr_inlier,tangent_corr_trimmed,"
              "expected_dx_px,expected_dy_px,delta_expected_dx_px,delta_expected_dy_px,delta_expected_norm_px,"
              "circle_prior_dx_px,circle_prior_dy_px,delta_circle_dx_px,delta_circle_dy_px,delta_circle_norm_px,"
              "delta_selected_dx_px,delta_selected_dy_px,delta_selected_norm_px,"
              "trial_closure_residual_px,trial_global_consistency_cost\n";

    const cv::Mat firstTransform =
        result.globalTransforms.empty() ? cv::Mat::eye(3, 3, CV_64F) : result.globalTransforms.front();
    const std::size_t imageCount = result.globalTransforms.empty()
                                       ? result.pairRecords.size() + 1
                                       : result.globalTransforms.size();
    for (const PairLoopRecord& record : result.pairRecords) {
        const cv::Point2d circlePrior =
            (record.stepIndex > 0 && record.stepIndex - 1 < circlePriors.size())
                ? circlePriors[record.stepIndex - 1]
                : record.expectedStepPx;
        const cv::Point2d selectedShift(record.transform.dx, record.transform.dy);
        const cv::Point2d selectedClosure =
            closureResidualForRecords(result.pairRecords, firstTransform, imageCount);
        const double selectedGlobalCost =
            globalConsistencyCost(result.pairRecords,
                                  buildTransformsFromRecords(result.pairRecords, firstTransform, imageCount),
                                  edges);
        appendTransformDiagnostics(stream,
                                   record,
                                   record.transform,
                                   "selected",
                                   -1,
                                   circlePrior,
                                   selectedShift,
                                   cv::norm(selectedClosure),
                                   selectedGlobalCost);

        int rank = 0;
        for (const stitch::AlignmentCandidateDiagnostic& diagnostic : record.transform.candidateDiagnostics) {
            stitch::TransformResult candidate = transformFromCandidate(diagnostic, record.transform);
            std::vector<PairLoopRecord> trialRecords = result.pairRecords;
            const std::size_t recordIndex = record.stepIndex > 0 ? record.stepIndex - 1 : 0;
            double trialClosureNorm = 0.0;
            double trialGlobalCost = 0.0;
            if (recordIndex < trialRecords.size()) {
                trialRecords[recordIndex].transform = candidate;
                const cv::Point2d trialClosure =
                    closureResidualForRecords(trialRecords, firstTransform, imageCount);
                trialClosureNorm = cv::norm(trialClosure);
                trialGlobalCost =
                    globalConsistencyCost(trialRecords,
                                          buildTransformsFromRecords(trialRecords, firstTransform, imageCount),
                                          edges);
            }
            appendTransformDiagnostics(stream,
                                       record,
                                       candidate,
                                       "candidate",
                                       rank,
                                       circlePrior,
                                       selectedShift,
                                       trialClosureNorm,
                                       trialGlobalCost);
            ++rank;
        }
    }

    return stream.str();
}

std::string buildCircleCsv(const StandardSphereLoopResult& result,
                           const std::vector<std::string>& imagePaths,
                           double pixelSizeMm,
                           double sphereDiameterMm)
{
    std::ostringstream stream;
    stream << "image_index,image_path,ok,point_count,center_x_px,center_y_px,radius_px,diameter_px,"
              "rmse_px,mean_abs_px,median_abs_px,mad_px,sigma_mad_px,max_abs_px,"
              "quality_weight_mean,confidence_mean,gradient_mean,diameter_mm,diameter_error_mm\n";
    for (std::size_t i = 0; i < result.perImageCircles.size(); ++i) {
        const CircleFitReport& fit = result.perImageCircles[i];
        const double diameterPx = fit.radiusPx * 2.0;
        const double diameterMm = pixelSizeMm > 0.0 ? diameterPx * pixelSizeMm : 0.0;
        const double diameterErrorMm =
            (pixelSizeMm > 0.0 && sphereDiameterMm > 0.0) ? diameterMm - sphereDiameterMm : 0.0;
        stream << i << ","
               << csvEscape(i < imagePaths.size() ? imagePaths[i] : std::string{}) << ","
               << (fit.ok ? 1 : 0) << ","
               << fit.pointCount << ","
               << fit.center.x << ","
               << fit.center.y << ","
               << fit.radiusPx << ","
               << diameterPx << ","
               << fit.rmsePx << ","
               << fit.meanAbsPx << ","
               << fit.medianAbsPx << ","
               << fit.madPx << ","
               << fit.sigmaMadPx << ","
               << fit.maxAbsPx << ","
               << fit.qualityWeightMean << ","
               << fit.confidenceMean << ","
               << fit.gradientMean << ","
               << diameterMm << ","
               << diameterErrorMm << "\n";
    }
    return stream.str();
}

std::string buildFieldBiasCsv(const StandardSphereLoopResult& result,
                              const std::vector<stitch::EdgeVariants>& edges,
                              const std::vector<cv::Mat>& images)
{
    struct CellStats {
        int count{0};
        double sumResidual{0.0};
        double sumResidualSq{0.0};
        double sumAbsResidual{0.0};
        double sumNx{0.0};
        double sumNy{0.0};
        double sumX{0.0};
        double sumY{0.0};
        double sumWeight{0.0};
        double sumConfidence{0.0};
        double sumGradient{0.0};
    };

    std::ostringstream stream;
    stream << "image_index,grid_x,grid_y,count,center_x_px,center_y_px,"
              "mean_residual_px,rmse_px,mean_abs_residual_px,"
              "mean_nx,mean_ny,mean_point_x_px,mean_point_y_px,"
              "mean_quality_weight,mean_confidence,mean_gradient\n";

    constexpr int kGrid = 3;
    for (std::size_t imageIndex = 0; imageIndex < edges.size() &&
                                      imageIndex < images.size() &&
                                      imageIndex < result.perImageCircles.size();
         ++imageIndex) {
        const CircleFitReport& circle = result.perImageCircles[imageIndex];
        if (!circle.ok || circle.radiusPx <= 0.0 || edges[imageIndex].raw.empty() ||
            images[imageIndex].empty()) {
            continue;
        }

        std::array<CellStats, kGrid * kGrid> cells{};
        const bool hasWeights =
            edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
        const bool hasConfidences =
            edges[imageIndex].rawConfidences.size() == edges[imageIndex].raw.size();
        const bool hasGradients =
            edges[imageIndex].rawGradients.size() == edges[imageIndex].raw.size();

        const double width = static_cast<double>(std::max(1, images[imageIndex].cols));
        const double height = static_cast<double>(std::max(1, images[imageIndex].rows));

        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            const cv::Point2d& point = edges[imageIndex].raw[pointIndex];
            const cv::Point2d radial = point - circle.center;
            const double radius = cv::norm(radial);
            if (radius <= 1e-9 || !std::isfinite(radius)) {
                continue;
            }

            int gridX = static_cast<int>(std::floor(point.x / width * kGrid));
            int gridY = static_cast<int>(std::floor(point.y / height * kGrid));
            gridX = std::clamp(gridX, 0, kGrid - 1);
            gridY = std::clamp(gridY, 0, kGrid - 1);

            CellStats& cell = cells[static_cast<std::size_t>(gridY * kGrid + gridX)];
            const double residual = radius - circle.radiusPx;
            const cv::Point2d normal = radial * (1.0 / radius);
            const double weight = hasWeights ? saneWeight(edges[imageIndex].rawQualityWeights[pointIndex]) : 1.0;
            const double confidence = hasConfidences ? edges[imageIndex].rawConfidences[pointIndex] : 0.0;
            const double gradient = hasGradients ? edges[imageIndex].rawGradients[pointIndex] : 0.0;

            ++cell.count;
            cell.sumResidual += residual;
            cell.sumResidualSq += residual * residual;
            cell.sumAbsResidual += std::abs(residual);
            cell.sumNx += normal.x;
            cell.sumNy += normal.y;
            cell.sumX += point.x;
            cell.sumY += point.y;
            cell.sumWeight += weight;
            cell.sumConfidence += confidence;
            cell.sumGradient += gradient;
        }

        for (int gridY = 0; gridY < kGrid; ++gridY) {
            for (int gridX = 0; gridX < kGrid; ++gridX) {
                const CellStats& cell = cells[static_cast<std::size_t>(gridY * kGrid + gridX)];
                if (cell.count == 0) {
                    continue;
                }
                const double invCount = 1.0 / static_cast<double>(cell.count);
                stream << imageIndex << ","
                       << gridX << ","
                       << gridY << ","
                       << cell.count << ","
                       << circle.center.x << ","
                       << circle.center.y << ","
                       << cell.sumResidual * invCount << ","
                       << std::sqrt(cell.sumResidualSq * invCount) << ","
                       << cell.sumAbsResidual * invCount << ","
                       << cell.sumNx * invCount << ","
                       << cell.sumNy * invCount << ","
                       << cell.sumX * invCount << ","
                       << cell.sumY * invCount << ","
                       << cell.sumWeight * invCount << ","
                       << cell.sumConfidence * invCount << ","
                       << cell.sumGradient * invCount << "\n";
            }
        }
    }

    return stream.str();
}

std::string buildFieldBiasCompensationCsv(const StandardSphereLoopResult& result,
                                          const std::vector<stitch::EdgeVariants>& edges,
                                          const std::vector<cv::Mat>& images)
{
    struct FieldBiasCell {
        int count{0};
        double sumResidual{0.0};
        double sumResidualSq{0.0};
        double sumWeight{0.0};
        double meanResidual{0.0};
        double rmseResidual{0.0};
        double gain{0.0};
    };

    struct FieldBiasModel {
        std::array<FieldBiasCell, 9> cells{};
        double maxAbsBias{0.0};
        double meanAbsBias{0.0};
    };

    auto cellIndexFor = [](const cv::Point2d& point, const cv::Size& size) {
        const double width = static_cast<double>(std::max(1, size.width));
        const double height = static_cast<double>(std::max(1, size.height));
        int gridX = static_cast<int>(std::floor(point.x / width * 3.0));
        int gridY = static_cast<int>(std::floor(point.y / height * 3.0));
        gridX = std::clamp(gridX, 0, 2);
        gridY = std::clamp(gridY, 0, 2);
        return gridY * 3 + gridX;
    };

    auto buildFieldBiasModel = [&](const std::vector<stitch::EdgeVariants>& allEdges,
                                   const std::vector<CircleFitReport>& circles,
                                   const std::vector<cv::Mat>& allImages) {
        FieldBiasModel model;
        for (std::size_t imageIndex = 0;
             imageIndex < allEdges.size() &&
             imageIndex < circles.size() &&
             imageIndex < allImages.size();
             ++imageIndex) {
            const CircleFitReport& circle = circles[imageIndex];
            if (!circle.ok || circle.radiusPx <= 0.0) {
                continue;
            }

            const bool hasWeights =
                allEdges[imageIndex].rawQualityWeights.size() == allEdges[imageIndex].raw.size();
            for (std::size_t pointIndex = 0; pointIndex < allEdges[imageIndex].raw.size(); ++pointIndex) {
                const cv::Point2d& point = allEdges[imageIndex].raw[pointIndex];
                const cv::Point2d radial = point - circle.center;
                const double radius = cv::norm(radial);
                if (radius <= 1e-9 || !std::isfinite(radius)) {
                    continue;
                }
                const double residual = radius - circle.radiusPx;
                const double weight = hasWeights ? saneWeight(allEdges[imageIndex].rawQualityWeights[pointIndex])
                                                 : 1.0;
                FieldBiasCell& cell = model.cells[static_cast<std::size_t>(
                    cellIndexFor(point, allImages[imageIndex].size()))];
                cell.count += 1;
                cell.sumResidual += weight * std::clamp(residual, -3.0, 3.0);
                cell.sumResidualSq += weight * residual * residual;
                cell.sumWeight += weight;
            }
        }

        for (FieldBiasCell& cell : model.cells) {
            if (cell.sumWeight > 1e-9) {
                cell.meanResidual = cell.sumResidual / cell.sumWeight;
                const double variance =
                    std::max(0.0, cell.sumResidualSq / cell.sumWeight - cell.meanResidual * cell.meanResidual);
                cell.rmseResidual = std::sqrt(variance);
                const double reliability = std::clamp(1.0 - cell.rmseResidual / 0.8, 0.0, 1.0);
                const double support = std::clamp(static_cast<double>(cell.count) / 200.0, 0.0, 1.0);
                cell.gain = 0.85 * reliability * support;
                model.maxAbsBias = std::max(model.maxAbsBias, std::abs(cell.meanResidual));
                model.meanAbsBias += std::abs(cell.meanResidual);
            }
        }
        model.meanAbsBias /= static_cast<double>(model.cells.size());
        return model;
    };

    auto applyBias = [&](const cv::Point2d& point,
                         const CircleFitReport& circle,
                         const FieldBiasModel& model,
                         const cv::Size& size) {
        const double radius = cv::norm(point - circle.center);
        if (radius <= 1e-9 || !std::isfinite(radius)) {
            return point;
        }
        const int idx = cellIndexFor(point, size);
        const FieldBiasCell& cell = model.cells[static_cast<std::size_t>(idx)];
        if (cell.count < 80 || cell.gain <= 1e-9) {
            return point;
        }
        const double bias = std::clamp(cell.meanResidual, -2.0, 2.0);
        const cv::Point2d normal = (point - circle.center) * (1.0 / radius);
        return point - cell.gain * bias * normal;
    };

    const FieldBiasModel model = buildFieldBiasModel(edges, result.perImageCircles, images);

    std::vector<CircleFitReport> correctedPerImage;
    correctedPerImage.reserve(result.perImageCircles.size());
    std::vector<WeightedPoint> correctedStitchedPoints;
    for (std::size_t imageIndex = 0;
         imageIndex < edges.size() &&
         imageIndex < images.size() &&
         imageIndex < result.perImageCircles.size() &&
         imageIndex < result.globalTransforms.size();
         ++imageIndex) {
        const CircleFitReport& rawCircle = result.perImageCircles[imageIndex];
        if (!rawCircle.ok) {
            correctedPerImage.push_back({});
            continue;
        }

        const std::vector<WeightedPoint> weightedPoints = makeWeightedPoints(edges[imageIndex]);
        std::vector<WeightedPoint> correctedPoints;
        correctedPoints.reserve(weightedPoints.size());
        for (const WeightedPoint& point : weightedPoints) {
            const cv::Point2d corrected = applyBias(point.point, rawCircle, model, images[imageIndex].size());
            correctedPoints.push_back({corrected, point.weight});
            correctedStitchedPoints.push_back({transformPoint(result.globalTransforms[imageIndex], corrected),
                                              point.weight});
        }
        correctedPerImage.push_back(fitCircleRobustWeighted(correctedPoints));
    }
    const CircleFitReport correctedStitched = fitCircleRobustWeighted(correctedStitchedPoints);

    std::ostringstream stream;
    stream << "scope,metric,value,unit\n";
    auto append = [&](const std::string& scope, const std::string& metric, double value, const std::string& unit) {
        stream << csvEscape(scope) << "," << csvEscape(metric) << "," << std::setprecision(12)
               << finiteOrZero(value) << "," << unit << "\n";
    };

    append("global", "cell_count", 9.0, "count");
    append("global", "mean_abs_bias", model.meanAbsBias, "px");
    append("global", "max_abs_bias", model.maxAbsBias, "px");
    double meanGain = 0.0;
    double maxGain = 0.0;
    double meanCellRmse = 0.0;
    for (const auto& cell : model.cells) {
        meanGain += cell.gain;
        maxGain = std::max(maxGain, cell.gain);
        meanCellRmse += cell.rmseResidual;
    }
    meanGain /= static_cast<double>(model.cells.size());
    meanCellRmse /= static_cast<double>(model.cells.size());
    append("global", "mean_cell_gain", meanGain, "");
    append("global", "max_cell_gain", maxGain, "");
    append("global", "mean_cell_rmse", meanCellRmse, "px");
    append("global", "raw_stitched_rmse", result.stitchedCircle.rmsePx, "px");
    append("global", "corrected_stitched_rmse", correctedStitched.rmsePx, "px");
    append("global", "raw_stitched_mean_abs", result.stitchedCircle.meanAbsPx, "px");
    append("global", "corrected_stitched_mean_abs", correctedStitched.meanAbsPx, "px");
    append("global", "raw_stitched_point_count", static_cast<double>(result.stitchedCircle.pointCount), "count");
    append("global", "corrected_stitched_point_count", static_cast<double>(correctedStitched.pointCount), "count");

    for (std::size_t i = 0; i < correctedPerImage.size() && i < result.perImageCircles.size(); ++i) {
        const CircleFitReport& raw = result.perImageCircles[i];
        const CircleFitReport& corrected = correctedPerImage[i];
        append("image_" + std::to_string(i),
               "raw_rmse",
               raw.rmsePx,
               "px");
        append("image_" + std::to_string(i),
               "corrected_rmse",
               corrected.rmsePx,
               "px");
        append("image_" + std::to_string(i),
               "raw_mean_abs",
               raw.meanAbsPx,
               "px");
        append("image_" + std::to_string(i),
               "corrected_mean_abs",
               corrected.meanAbsPx,
               "px");
    }

    return stream.str();
}

std::string buildTransformCsv(const StandardSphereLoopResult& result)
{
    std::ostringstream stream;
    stream << "image_index,m00,m01,m02,m10,m11,m12,m20,m21,m22,"
              "expected_x_px,expected_y_px,translation_error_x_px,translation_error_y_px,"
              "translation_error_px\n";
    const cv::Mat invFirst = result.globalTransforms.empty()
                                 ? cv::Mat()
                                 : result.globalTransforms.front().inv();
    for (std::size_t i = 0; i < result.globalTransforms.size(); ++i) {
        const cv::Mat& m = result.globalTransforms[i];
        const cv::Mat relative = invFirst.empty() ? m : invFirst * m;
        const cv::Point2d expected =
            i < result.expectedGlobalTranslationsPx.size() ? result.expectedGlobalTranslationsPx[i]
                                                           : cv::Point2d(0.0, 0.0);
        const cv::Point2d measured(relative.at<double>(0, 2), relative.at<double>(1, 2));
        const cv::Point2d error = measured - expected;
        stream << i;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                stream << "," << relative.at<double>(r, c);
            }
        }
        stream << "," << expected.x
               << "," << expected.y
               << "," << error.x
               << "," << error.y
               << "," << cv::norm(error)
               << "\n";
    }
    return stream.str();
}

std::string buildSegmentCsv(const StandardSphereLoopResult& result)
{
    std::ostringstream stream;
    stream << "segment,direction,start_step,end_step,pair_count,expected_dx_px,expected_dy_px,"
              "measured_dx_px,measured_dy_px,residual_dx_px,residual_dy_px,residual_translation_px,"
              "residual_translation_mm\n";
    for (const SegmentLoopRecord& segment : result.segmentRecords) {
        stream << segment.segmentIndex << ","
               << segment.expectedDirection << ","
               << segment.startStep << ","
               << segment.endStep << ","
               << segment.pairCount << ","
               << segment.expectedDeltaPx.x << ","
               << segment.expectedDeltaPx.y << ","
               << segment.measuredDeltaPx.x << ","
               << segment.measuredDeltaPx.y << ","
               << segment.residualDeltaPx.x << ","
               << segment.residualDeltaPx.y << ","
               << segment.residualTranslationPx << ","
               << segment.residualTranslationMm << "\n";
    }
    return stream.str();
}

std::string buildSummaryCsv(const StandardSphereLoopResult& result,
                            const std::vector<stitch::EdgeVariants>& edges,
                            const std::vector<cv::Mat>& images,
                            double pixelSizeMm,
                            double sphereDiameterMm)
{
    std::ostringstream stream;
    stream << "metric,value,unit,note\n";
    appendMetric(stream, "image_count", static_cast<double>(result.globalTransforms.size()), "count");
    appendMetric(stream, "pair_count", static_cast<double>(result.pairRecords.size()), "count");
    appendMetric(stream, "segment_count", static_cast<double>(result.segmentRecords.size()), "count");
    appendMetric(stream,
                 "global_circle_consistency_optimized",
                 result.circleOptimizationIterations > 0 ? 1.0 : 0.0,
                 "bool",
                 "standard sphere 2D circle-consistency refinement; does not force loop closure to zero");
    appendMetric(stream,
                 "pre_optimization_closure_translation",
                 result.preOptimizationClosureTranslationPx,
                 "px",
                 "raw workpiece pipeline closure before sphere-specific 2D refinement");
    appendMetric(stream,
                 "pre_optimization_closure_rotation",
                 result.preOptimizationClosureRotationDeg,
                 "deg",
                 "raw workpiece pipeline closure before sphere-specific 2D refinement");
    appendMetric(stream,
                 "pre_optimization_stitched_circle_rmse",
                 result.preOptimizationStitchedCircleRmsePx,
                 "px");
    appendMetric(stream,
                 "pre_optimization_stitched_circle_mean_abs",
                 result.preOptimizationStitchedCircleMeanAbsPx,
                 "px");
    appendMetric(stream,
                 "circle_optimization_iterations",
                 static_cast<double>(result.circleOptimizationIterations),
                 "count");
    appendMetric(stream,
                 "circle_optimization_mean_correction",
                 result.circleOptimizationMeanCorrectionPx,
                 "px");
    appendMetric(stream,
                 "circle_optimization_max_correction",
                 result.circleOptimizationMaxCorrectionPx,
                 "px");
    appendMetric(stream,
                 "circle_optimization_max_rotation_correction",
                 result.circleOptimizationMaxCorrectionRotationDeg,
                 "deg");
    appendMetric(stream,
                 "sphere_bad_step_geometry_rescue_count",
                 static_cast<double>(result.sphereBadStepGeometryRescueCount),
                 "count",
                 "bad pair transforms replaced by standard-sphere circle-center geometry");
    appendMetric(stream,
                 "sphere_bad_step_geometry_rescue_max_correction",
                 result.sphereBadStepGeometryRescueMaxCorrectionPx,
                 "px");
    appendMetric(stream,
                 "global_consistency_reregistration_count",
                 static_cast<double>(result.globalConsistencyReregistrationCount),
                 "count",
                 "candidate transforms accepted only when whole-loop consistency cost decreases");
    appendMetric(stream,
                 "global_consistency_reregistration_cost_improvement",
                 result.globalConsistencyReregistrationImprovement,
                 "cost");
    appendMetric(stream,
                 "bad_step_local_refinement_count",
                 static_cast<double>(result.badStepLocalRefinementCount),
                 "count",
                 "bad steps accepted after local dx/dy/angle refinement with circle-consistency guard");
    appendMetric(stream,
                 "bad_step_local_refinement_cost_improvement",
                 result.badStepLocalRefinementCostImprovement,
                 "cost");
    appendMetric(stream,
                 "bad_step_local_refinement_max_correction",
                 result.badStepLocalRefinementMaxCorrectionPx,
                 "px");
    appendMetric(stream,
                 "soft_global_drift_optimization_iterations",
                 static_cast<double>(result.softGlobalDriftOptimizationIterations),
                 "count",
                 "standard sphere pose-consistency correction; bounded and accepted only if circle RMSE is preserved");
    appendMetric(stream,
                 "soft_global_drift_optimization_accepted_fraction",
                 result.softGlobalDriftOptimizationAcceptedFraction,
                 "",
                 "fraction of measured residual pose correction accepted by validation");
    appendMetric(stream,
                 "soft_global_drift_optimization_max_step_correction",
                 result.softGlobalDriftOptimizationMaxStepCorrectionPx,
                 "px");
    appendMetric(stream,
                 "soft_global_drift_optimization_max_rotation_correction",
                 result.softGlobalDriftOptimizationMaxRotationCorrectionDeg,
                 "deg");
    appendMetric(stream,
                 "soft_global_drift_optimization_total_correction",
                 result.softGlobalDriftOptimizationTotalCorrectionPx,
                 "px");
    appendMetric(stream,
                 "soft_global_drift_optimization_circle_rmse_change",
                 result.softGlobalDriftOptimizationCircleRmseChangePx,
                 "px");
    appendMetric(stream,
                 "soft_global_drift_optimization_cost_improvement",
                 result.softGlobalDriftOptimizationCostImprovement,
                 "cost");
    appendMetric(stream,
                 "evaluation_pixel_size",
                 result.evaluationPixelSizeMm,
                 "mm/px",
                 pixelSizeMm > 0.0 ? "from --pixel-size" : "derived from configured field of view");
    appendMetric(stream, "measured_closure_x", result.measuredClosurePx.x, "px");
    appendMetric(stream, "measured_closure_y", result.measuredClosurePx.y, "px");
    appendMetric(stream, "expected_closure_x", result.expectedClosurePx.x, "px");
    appendMetric(stream, "expected_closure_y", result.expectedClosurePx.y, "px");
    appendMetric(stream, "closure_translation", result.closureTranslationPx, "px");
    appendMetric(stream,
                 "closure_translation_error_vs_expected",
                 result.closureResidualTranslationPx,
                 "px",
                 "measured closure vector minus configured clockwise step model");
    appendMetric(stream, "closure_rotation", result.closureRotationDeg, "deg");
    appendMetric(stream, "closure_center_drift", result.closureCenterDriftPx, "px");
    appendMetric(stream, "closure_probe_rms", result.closureCornerRmsPx, "px", "four image corners plus center");
    appendMetric(stream, "closure_probe_max", result.closureCornerMaxPx, "px", "four image corners plus center");

    const double evaluationScaleMm = pixelSizeMm > 0.0 ? pixelSizeMm : 0.0;
    if (evaluationScaleMm > 0.0) {
        appendMetric(stream,
                     "pre_optimization_closure_translation",
                     result.preOptimizationClosureTranslationPx * evaluationScaleMm,
                     "mm");
        appendMetric(stream,
                     "pre_optimization_stitched_circle_rmse",
                     result.preOptimizationStitchedCircleRmsePx * evaluationScaleMm,
                     "mm");
        appendMetric(stream,
                     "circle_optimization_mean_correction",
                     result.circleOptimizationMeanCorrectionPx * evaluationScaleMm,
                     "mm");
        appendMetric(stream,
                     "circle_optimization_max_correction",
                     result.circleOptimizationMaxCorrectionPx * evaluationScaleMm,
                     "mm");
        appendMetric(stream, "measured_closure_x", result.measuredClosurePx.x * evaluationScaleMm, "mm");
        appendMetric(stream, "measured_closure_y", result.measuredClosurePx.y * evaluationScaleMm, "mm");
        appendMetric(stream, "expected_closure_x", result.expectedClosurePx.x * evaluationScaleMm, "mm");
        appendMetric(stream, "expected_closure_y", result.expectedClosurePx.y * evaluationScaleMm, "mm");
        appendMetric(stream, "closure_translation", result.closureTranslationPx * evaluationScaleMm, "mm");
        appendMetric(stream,
                     "closure_translation_error_vs_expected",
                     result.closureResidualTranslationMm,
                     "mm",
                     "measured closure vector minus configured clockwise step model");
        appendMetric(stream, "closure_center_drift", result.closureCenterDriftPx * evaluationScaleMm, "mm");
        appendMetric(stream, "closure_probe_rms", result.closureCornerRmsPx * evaluationScaleMm, "mm");
        appendMetric(stream, "closure_probe_max", result.closureCornerMaxPx * evaluationScaleMm, "mm");
    }

    std::vector<double> radii;
    std::vector<double> rmses;
    std::vector<double> meanAbsValues;
    double pooledNoiseSquared = 0.0;
    double pooledMeanAbsNumerator = 0.0;
    int pooledNoiseCount = 0;
    for (const CircleFitReport& fit : result.perImageCircles) {
        if (fit.ok) {
            radii.push_back(fit.radiusPx);
            rmses.push_back(fit.rmsePx);
            meanAbsValues.push_back(fit.meanAbsPx);
            pooledNoiseSquared += fit.rmsePx * fit.rmsePx * static_cast<double>(fit.pointCount);
            pooledMeanAbsNumerator += fit.meanAbsPx * static_cast<double>(fit.pointCount);
            pooledNoiseCount += fit.pointCount;
        }
    }

    if (!radii.empty()) {
        const double meanRadius = std::accumulate(radii.begin(), radii.end(), 0.0) /
                                  static_cast<double>(radii.size());
        double variance = 0.0;
        for (double radius : radii) {
            const double delta = radius - meanRadius;
            variance += delta * delta;
        }
        variance /= static_cast<double>(radii.size());

        const auto [minIt, maxIt] = std::minmax_element(radii.begin(), radii.end());
        const double meanRmse = std::accumulate(rmses.begin(), rmses.end(), 0.0) /
                                static_cast<double>(rmses.size());
        double rmseVariance = 0.0;
        for (double rmse : rmses) {
            const double delta = rmse - meanRmse;
            rmseVariance += delta * delta;
        }
        rmseVariance /= static_cast<double>(rmses.size());
        const double meanAbs = std::accumulate(meanAbsValues.begin(), meanAbsValues.end(), 0.0) /
                               static_cast<double>(meanAbsValues.size());
        const double pooledRmse = pooledNoiseCount > 0
                                      ? std::sqrt(pooledNoiseSquared /
                                                  static_cast<double>(pooledNoiseCount))
                                      : 0.0;
        const double pooledMeanAbs = pooledNoiseCount > 0
                                         ? pooledMeanAbsNumerator /
                                               static_cast<double>(pooledNoiseCount)
                                         : 0.0;

        appendMetric(stream, "per_image_circle_valid_count", static_cast<double>(radii.size()), "count");
        appendMetric(stream, "per_image_radius_mean", meanRadius, "px");
        appendMetric(stream, "per_image_radius_stddev", std::sqrt(variance), "px");
        appendMetric(stream, "per_image_radius_min", *minIt, "px");
        appendMetric(stream, "per_image_radius_max", *maxIt, "px");
        appendMetric(stream, "per_image_circle_rmse_mean", meanRmse, "px");
        appendMetric(stream, "per_image_circle_rmse_stddev", std::sqrt(rmseVariance), "px");
        appendMetric(stream, "random_noise_per_image_rmse_mean", meanRmse, "px");
        appendMetric(stream, "random_noise_per_image_mean_abs_mean", meanAbs, "px");
        appendMetric(stream, "random_noise_pooled_rmse", pooledRmse, "px");
        appendMetric(stream, "random_noise_pooled_mean_abs", pooledMeanAbs, "px");

        if (pixelSizeMm > 0.0) {
            const double meanDiameterMm = meanRadius * 2.0 * pixelSizeMm;
            appendMetric(stream, "per_image_diameter_mean", meanDiameterMm, "mm");
            appendMetric(stream, "random_noise_per_image_rmse_mean", meanRmse * pixelSizeMm, "mm");
            appendMetric(stream, "random_noise_pooled_rmse", pooledRmse * pixelSizeMm, "mm");
            if (sphereDiameterMm > 0.0) {
                appendMetric(stream,
                             "per_image_diameter_mean_error",
                             meanDiameterMm - sphereDiameterMm,
                             "mm",
                             "mean measured diameter minus standard sphere diameter");
            }
        }
    }

    if (result.stitchedCircle.ok) {
        appendMetric(stream, "stitched_circle_point_count", result.stitchedCircle.pointCount, "count");
        appendMetric(stream, "stitched_circle_radius", result.stitchedCircle.radiusPx, "px");
        appendMetric(stream, "stitched_circle_diameter", result.stitchedCircle.radiusPx * 2.0, "px");
        appendMetric(stream, "stitched_circle_rmse", result.stitchedCircle.rmsePx, "px");
        appendMetric(stream, "stitched_circle_mean_abs", result.stitchedCircle.meanAbsPx, "px");
        appendMetric(stream, "stitched_circle_median_abs", result.stitchedCircle.medianAbsPx, "px");
        appendMetric(stream, "stitched_circle_mad", result.stitchedCircle.madPx, "px");
        appendMetric(stream, "stitched_circle_sigma_mad", result.stitchedCircle.sigmaMadPx, "px");
        appendMetric(stream, "stitched_circle_max_abs", result.stitchedCircle.maxAbsPx, "px");

        if (pixelSizeMm > 0.0) {
            const double stitchedDiameterMm = result.stitchedCircle.radiusPx * 2.0 * pixelSizeMm;
            appendMetric(stream, "stitched_circle_diameter", stitchedDiameterMm, "mm");
            appendMetric(stream, "stitched_circle_rmse", result.stitchedCircle.rmsePx * pixelSizeMm, "mm");
            appendMetric(stream, "stitched_circle_mean_abs", result.stitchedCircle.meanAbsPx * pixelSizeMm, "mm");
            appendMetric(stream, "stitched_circle_median_abs", result.stitchedCircle.medianAbsPx * pixelSizeMm, "mm");
            appendMetric(stream, "stitched_circle_sigma_mad", result.stitchedCircle.sigmaMadPx * pixelSizeMm, "mm");
            appendMetric(stream, "stitched_circle_max_abs", result.stitchedCircle.maxAbsPx * pixelSizeMm, "mm");
            if (sphereDiameterMm > 0.0) {
                appendMetric(stream,
                             "stitched_circle_diameter_error",
                             stitchedDiameterMm - sphereDiameterMm,
                             "mm",
                             "stitched diameter minus standard sphere diameter");
            }
        }
    }

    const std::string compensationCsv = buildFieldBiasCompensationCsv(result, edges, images);
    std::vector<std::pair<std::string, double>> compensationMetrics;
    {
        std::istringstream input(compensationCsv);
        std::string line;
        bool isHeader = true;
        while (std::getline(input, line)) {
            if (isHeader) {
                isHeader = false;
                continue;
            }
            if (line.empty()) {
                continue;
            }
            std::array<std::string, 4> cols{};
            std::size_t col = 0;
            std::size_t start = 0;
            while (col < cols.size()) {
                const std::size_t pos = line.find(',', start);
                cols[col++] = line.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
                if (pos == std::string::npos) {
                    break;
                }
                start = pos + 1;
            }
            if (cols[0] != "global" || cols[1].empty() || cols[2].empty()) {
                continue;
            }
            try {
                compensationMetrics.push_back({cols[1], std::stod(cols[2])});
            } catch (...) {
            }
        }
    }

    const auto compValue = [&](const std::string& key) {
        for (const auto& item : compensationMetrics) {
            if (item.first == key) {
                return item.second;
            }
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    appendMetric(stream, "field_bias_mean_abs_bias", compValue("mean_abs_bias"), "px");
    appendMetric(stream, "field_bias_max_abs_bias", compValue("max_abs_bias"), "px");
    appendMetric(stream, "field_bias_mean_cell_gain", compValue("mean_cell_gain"), "");
    appendMetric(stream, "field_bias_max_cell_gain", compValue("max_cell_gain"), "");
    appendMetric(stream, "field_bias_mean_cell_rmse", compValue("mean_cell_rmse"), "px");
    appendMetric(stream, "field_bias_raw_stitched_rmse", compValue("raw_stitched_rmse"), "px");
    appendMetric(stream, "field_bias_corrected_stitched_rmse", compValue("corrected_stitched_rmse"), "px");
    appendMetric(stream, "field_bias_raw_stitched_mean_abs", compValue("raw_stitched_mean_abs"), "px");
    appendMetric(stream, "field_bias_corrected_stitched_mean_abs", compValue("corrected_stitched_mean_abs"), "px");
    appendMetric(stream, "field_bias_raw_stitched_point_count", compValue("raw_stitched_point_count"), "count");
    appendMetric(stream, "field_bias_corrected_stitched_point_count", compValue("corrected_stitched_point_count"), "count");

    return stream.str();
}

cv::Mat buildStitchedVisualization(const StandardSphereLoopResult& result,
                                   const std::vector<stitch::EdgeVariants>& edges,
                                   const std::vector<cv::Mat>& images)
{
    if (edges.empty() || result.globalTransforms.size() < edges.size()) {
        return {};
    }

    cv::Rect2d bounds(0.0, 0.0, -1.0, -1.0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        for (const cv::Point2d& point : edges[i].raw) {
            updateBounds(bounds, transformPoint(result.globalTransforms[i], point));
        }

        if (i < images.size() && !images[i].empty()) {
            const double w = static_cast<double>(images[i].cols - 1);
            const double h = static_cast<double>(images[i].rows - 1);
            updateBounds(bounds, transformPoint(result.globalTransforms[i], cv::Point2d(0.0, 0.0)));
            updateBounds(bounds, transformPoint(result.globalTransforms[i], cv::Point2d(w, 0.0)));
            updateBounds(bounds, transformPoint(result.globalTransforms[i], cv::Point2d(w, h)));
            updateBounds(bounds, transformPoint(result.globalTransforms[i], cv::Point2d(0.0, h)));
        }
    }

    if (result.stitchedCircle.ok) {
        updateBounds(bounds, result.stitchedCircle.center -
                                cv::Point2d(result.stitchedCircle.radiusPx, result.stitchedCircle.radiusPx));
        updateBounds(bounds, result.stitchedCircle.center +
                                cv::Point2d(result.stitchedCircle.radiusPx, result.stitchedCircle.radiusPx));
    }

    if (bounds.width <= 0.0 || bounds.height <= 0.0) {
        return {};
    }

    const double paddedWidth = bounds.width + 2.0 * kVisualizationPaddingPx;
    const double paddedHeight = bounds.height + 2.0 * kVisualizationPaddingPx;
    const double scale = std::min(1.0,
                                  static_cast<double>(kVisualizationMaxSidePx) /
                                      std::max(paddedWidth, paddedHeight));
    const int canvasWidth = std::max(1, static_cast<int>(std::ceil(paddedWidth * scale)));
    const int canvasHeight = std::max(1, static_cast<int>(std::ceil(paddedHeight * scale)));
    cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar(250, 250, 250));

    const auto toCanvas = [&](const cv::Point2d& point) {
        return cv::Point(static_cast<int>(std::lround((point.x - bounds.x + kVisualizationPaddingPx) * scale)),
                         static_cast<int>(std::lround((point.y - bounds.y + kVisualizationPaddingPx) * scale)));
    };

    for (std::size_t i = 0; i < images.size() && i < result.globalTransforms.size(); ++i) {
        if (images[i].empty()) {
            continue;
        }
        const double w = static_cast<double>(images[i].cols - 1);
        const double h = static_cast<double>(images[i].rows - 1);
        std::vector<cv::Point> corners = {
            toCanvas(transformPoint(result.globalTransforms[i], cv::Point2d(0.0, 0.0))),
            toCanvas(transformPoint(result.globalTransforms[i], cv::Point2d(w, 0.0))),
            toCanvas(transformPoint(result.globalTransforms[i], cv::Point2d(w, h))),
            toCanvas(transformPoint(result.globalTransforms[i], cv::Point2d(0.0, h)))
        };
        const cv::Scalar color = paletteColor(i);
        cv::polylines(canvas, corners, true, color, 1, cv::LINE_AA);
    }

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const cv::Scalar color = paletteColor(i);
        const int pointRadius = scale >= 0.35 ? 2 : 1;
        for (const cv::Point2d& point : edges[i].raw) {
            cv::circle(canvas, toCanvas(transformPoint(result.globalTransforms[i], point)),
                       pointRadius, color, -1, cv::LINE_AA);
        }
    }

    if (result.stitchedCircle.ok) {
        const cv::Point center = toCanvas(result.stitchedCircle.center);
        const int radius = std::max(1, static_cast<int>(std::lround(result.stitchedCircle.radiusPx * scale)));
        cv::circle(canvas, center, radius, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
        cv::drawMarker(canvas, center, cv::Scalar(0, 0, 0), cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    }

    if (!result.closureMatrix.empty() && !images.empty() && !images.front().empty()) {
        const cv::Point2d firstCenter(images.front().cols / 2.0, images.front().rows / 2.0);
        const cv::Point2d closureCenter = transformPoint(result.closureMatrix, firstCenter);
        cv::arrowedLine(canvas,
                        toCanvas(firstCenter),
                        toCanvas(closureCenter),
                        cv::Scalar(40, 40, 220),
                        2,
                        cv::LINE_AA,
                        0,
                        0.03);
    }

    drawInfoBox(canvas, "Standard-sphere stitched contour", cv::Point(18, 34), 0.62, cv::Scalar(42, 42, 42));
    std::ostringstream summary;
    summary << edges.size() << " views | circle RMSE "
            << std::fixed << std::setprecision(3) << result.stitchedCircle.rmsePx << " px";
    if (!result.closureMatrix.empty()) {
        summary << " | red arrow: closure drift";
    }
    drawInfoBox(canvas, summary.str(), cv::Point(18, 62), 0.46, cv::Scalar(72, 72, 72));

    std::ostringstream closureInfo;
    closureInfo << "closure drift " << std::fixed << std::setprecision(3) << result.closureTranslationPx << " px"
                << " | black circle: fitted global circle";
    drawInfoBox(canvas, closureInfo.str(), cv::Point(18, 90), 0.44, cv::Scalar(72, 72, 72));

    std::ostringstream scaleInfo;
    scaleInfo << "Display scale " << std::fixed << std::setprecision(3) << scale;
    drawInfoBox(canvas, scaleInfo.str(), cv::Point(18, canvas.rows - 18), 0.44, cv::Scalar(72, 72, 72));

    return cropWhiteMargin(canvas, 8);
}

} // namespace pinjie::standard_sphere_loop
