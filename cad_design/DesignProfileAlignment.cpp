#include "cad_design/DesignProfileAlignment.h"

#include "cad_design/DesignProfileFunction.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace pinjie::cad_design {

namespace {

constexpr std::size_t kMinUsedPointCount = 50;
constexpr std::size_t kMinAnchorRunPointCount = 80;
constexpr double kMaxRadiusJumpForSameRunMm = 0.35;
constexpr std::size_t kContinuationSearchWindowPointCount = 200;
constexpr std::size_t kMaxContinuationSearchOffset = 400;
constexpr double kMaxContinuationWindowRmseUm = 120.0;
constexpr double kMaxContinuationWindowAbsMeanUm = 120.0;
constexpr double kPrimaryStartWindowRmseUm = 35.0;
constexpr double kPrimaryStartWindowAbsMeanUm = 35.0;
constexpr double kLeftTransitionSearchMaxMm = 3.0;
constexpr double kLeftTransitionWindowMm = 0.40;
constexpr std::size_t kLeftTransitionMinPointCount = 30;
constexpr double kLeftTransitionMedianCenteredErrorUm = 8.0;
constexpr double kLeftTransitionP90CenteredErrorUm = 20.0;

struct StitchedContourSample {
    double xPx{0.0};
    double yPx{0.0};
};

struct GeneratrixCandidatePoint {
    std::size_t index{0};
    std::size_t supportCount{0};
    double xPx{0.0};
    double yPx{0.0};
    double yStdPx{0.0};
    double slopeAbs{0.0};
    double sBaseMm{0.0};
    double rRawMm{0.0};
    bool slopeAccepted{true};
    bool trimAccepted{true};
};

struct PointRun {
    std::size_t beginIndex{0};
    std::size_t endIndex{0};
    std::size_t pointCount{0};
    double lengthMm{0.0};
};

struct MeasuredGeneratrixData {
    std::vector<GeneratrixCandidatePoint> points;
    std::vector<PointRun> runs;
    std::size_t anchorPointIndex{0};
    std::size_t primaryRunBeginIndex{0};
    std::size_t primaryRunEndIndex{0};
    double anchorXPx{0.0};
    double anchorYPx{0.0};
    double anchorBaseMm{0.0};
    double anchorMeasuredRadiusMm{0.0};
    double maxBaseMm{0.0};
};

struct EvaluationResult {
    bool ok{false};
    double objective{std::numeric_limits<double>::infinity()};
    double dzMm{0.0};
    double drMm{0.0};
    double axialScaleFactor{1.0};
    double axialQuadraticTermMm{0.0};
    std::vector<DesignErrorProfilePoint> profilePoints;
    DesignErrorSummary summary;
};

struct ErrorWindowStats {
    bool valid{false};
    double meanUm{0.0};
    double rmseUm{0.0};
    double absMeanUm{0.0};
};

struct DesignPolylinePoint {
    double sMm{0.0};
    double rMm{0.0};
};

struct NearestDesignPoint {
    bool valid{false};
    std::size_t segmentIndex{0};
    double sMm{0.0};
    double rMm{0.0};
    double signedDistanceMm{0.0};
    double tangentDs{1.0};
    double tangentDr{0.0};
};

std::string formatDouble(const double value, const int precision = 6)
{
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(precision);
    stream << value;
    return stream.str();
}

std::string csvCell(const double value, const int precision = 6)
{
    return std::isfinite(value) ? formatDouble(value, precision) : std::string();
}

double percentileAbs(std::vector<double> values, const double quantile)
{
    if (values.empty()) {
        return 0.0;
    }

    const double clamped = std::clamp(quantile, 0.0, 1.0);
    const double rawIndex = clamped * static_cast<double>(values.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(rawIndex));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(rawIndex));

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lowerIndex), values.end());
    const double lowerValue = values[lowerIndex];
    if (upperIndex == lowerIndex) {
        return lowerValue;
    }

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(upperIndex), values.end());
    const double upperValue = values[upperIndex];
    const double blend = rawIndex - static_cast<double>(lowerIndex);
    return lowerValue * (1.0 - blend) + upperValue * blend;
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
        const auto lowerMid = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(midIndex));
        median = 0.5 * (median + *lowerMid);
    }
    return median;
}

ErrorStats buildErrorStats(const std::vector<double>& errorsUm)
{
    ErrorStats stats;
    stats.count = errorsUm.size();
    if (errorsUm.empty()) {
        return stats;
    }

    std::vector<double> absErrorsUm;
    absErrorsUm.reserve(errorsUm.size());
    double sumUm = 0.0;
    double sumAbsUm = 0.0;
    double sumSquaresUm = 0.0;
    double maxPosUm = -std::numeric_limits<double>::infinity();
    double maxNegUm = std::numeric_limits<double>::infinity();
    for (const double errorUm : errorsUm) {
        absErrorsUm.push_back(std::abs(errorUm));
        sumUm += errorUm;
        sumAbsUm += std::abs(errorUm);
        sumSquaresUm += errorUm * errorUm;
        maxPosUm = std::max(maxPosUm, errorUm);
        maxNegUm = std::min(maxNegUm, errorUm);
    }

    stats.meanUm = sumUm / static_cast<double>(errorsUm.size());
    stats.rmseUm = std::sqrt(sumSquaresUm / static_cast<double>(errorsUm.size()));
    stats.maeUm = sumAbsUm / static_cast<double>(errorsUm.size());
    stats.p95AbsUm = percentileAbs(std::move(absErrorsUm), 0.95);
    stats.maxPosUm = std::isfinite(maxPosUm) ? maxPosUm : 0.0;
    stats.maxNegUm = std::isfinite(maxNegUm) ? maxNegUm : 0.0;
    stats.pvUm = stats.maxPosUm - stats.maxNegUm;
    return stats;
}

void appendDesignPolylineSample(std::vector<DesignPolylinePoint>& points,
                                const double zOriginalMm,
                                const double rMm,
                                const bool reverseZ)
{
    const double sMm = reverseZ ? (kDesignProfileMaxZMm - zOriginalMm) : zOriginalMm;
    points.push_back({sMm, rMm});
}

std::vector<DesignPolylinePoint> buildDesignPolyline(const bool reverseZ)
{
    constexpr double kLinearSegmentEndZMm = 52.958772;
    constexpr double kPolynomialSegmentEndZMm = 100.0;
    constexpr double kConstantSegment1EndZMm = 119.0;
    constexpr double kConstantRadius2Mm = 179.919242;
    constexpr double kStepMm = 0.1;

    std::vector<DesignPolylinePoint> points;
    points.reserve(700);

    appendDesignPolylineSample(points, kDesignProfileMinZMm, evalDesignRadiusOriginal(kDesignProfileMinZMm).r_mm, reverseZ);
    appendDesignPolylineSample(points, kLinearSegmentEndZMm, evalDesignRadiusOriginal(kLinearSegmentEndZMm).r_mm, reverseZ);

    for (double zMm = kLinearSegmentEndZMm + kStepMm;
         zMm < kPolynomialSegmentEndZMm - 1e-12;
         zMm += kStepMm) {
        const DesignEval design = evalDesignRadiusOriginal(zMm);
        if (design.valid) {
            appendDesignPolylineSample(points, zMm, design.r_mm, reverseZ);
        }
    }

    appendDesignPolylineSample(points, kPolynomialSegmentEndZMm, evalDesignRadiusOriginal(kPolynomialSegmentEndZMm).r_mm, reverseZ);
    appendDesignPolylineSample(points,
                               kConstantSegment1EndZMm,
                               evalDesignRadiusOriginal(kConstantSegment1EndZMm).r_mm,
                               reverseZ);
    appendDesignPolylineSample(points, kConstantSegment1EndZMm, kConstantRadius2Mm, reverseZ);
    appendDesignPolylineSample(points, kDesignProfileMaxZMm, kConstantRadius2Mm, reverseZ);

    std::stable_sort(points.begin(), points.end(), [reverseZ](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return reverseZ ? lhs.rMm < rhs.rMm : lhs.rMm > rhs.rMm;
    });

    points.erase(std::unique(points.begin(),
                             points.end(),
                             [](const auto& lhs, const auto& rhs) {
                                 return std::abs(lhs.sMm - rhs.sMm) < 1e-9 &&
                                        std::abs(lhs.rMm - rhs.rMm) < 1e-9;
                             }),
                 points.end());
    return points;
}

NearestDesignPoint findNearestDesignPoint(const std::vector<DesignPolylinePoint>& polyline,
                                          const double sMm,
                                          const double rMm)
{
    NearestDesignPoint nearest;
    if (polyline.size() < 2 || !std::isfinite(sMm) || !std::isfinite(rMm)) {
        return nearest;
    }

    double bestDistanceSquare = std::numeric_limits<double>::infinity();
    const auto updateFromSegment = [&](const std::size_t index) {
        if (index == 0 || index >= polyline.size()) {
            return;
        }
        const DesignPolylinePoint& a = polyline[index - 1];
        const DesignPolylinePoint& b = polyline[index];
        const double ds = b.sMm - a.sMm;
        const double dr = b.rMm - a.rMm;
        const double lengthSquare = ds * ds + dr * dr;
        if (lengthSquare < 1e-18) {
            return;
        }

        const double rawT = ((sMm - a.sMm) * ds + (rMm - a.rMm) * dr) / lengthSquare;
        const double t = std::clamp(rawT, 0.0, 1.0);
        const double candidateS = a.sMm + t * ds;
        const double candidateR = a.rMm + t * dr;
        const double deltaS = sMm - candidateS;
        const double deltaR = rMm - candidateR;
        const double distanceSquare = deltaS * deltaS + deltaR * deltaR;
        if (distanceSquare >= bestDistanceSquare) {
            return;
        }

        const double length = std::sqrt(lengthSquare);
        const double tangentDs = ds / length;
        const double tangentDr = dr / length;
        const double normalS = -tangentDr;
        const double normalR = tangentDs;

        bestDistanceSquare = distanceSquare;
        nearest.valid = true;
        nearest.segmentIndex = index - 1;
        nearest.sMm = candidateS;
        nearest.rMm = candidateR;
        nearest.signedDistanceMm = deltaS * normalS + deltaR * normalR;
        nearest.tangentDs = tangentDs;
        nearest.tangentDr = tangentDr;
    };

    const auto lower = std::lower_bound(polyline.begin(), polyline.end(), sMm, [](const auto& point, const double value) {
        return point.sMm < value;
    });
    const std::size_t centerIndex =
        lower == polyline.end() ? polyline.size() - 1 : static_cast<std::size_t>(std::distance(polyline.begin(), lower));
    const std::size_t beginIndex = centerIndex > 8 ? centerIndex - 8 : 1;
    const std::size_t endIndex = std::min(polyline.size() - 1, centerIndex + 8);
    for (std::size_t index = beginIndex; index <= endIndex; ++index) {
        updateFromSegment(index);
    }

    return nearest;
}

cv::Point2d transformPoint(const cv::Mat& transform, const cv::Point2d& point)
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

bool collectStitchedContourSamples(const std::vector<stitch::EdgeVariants>& edges,
                                   const std::vector<cv::Mat>& imageTransforms,
                                   std::vector<StitchedContourSample>& samples)
{
    samples.clear();
    const std::size_t count = std::min(edges.size(), imageTransforms.size());
    if (count == 0) {
        return false;
    }

    for (std::size_t imageIndex = 0; imageIndex < count; ++imageIndex) {
        const cv::Mat& transform = imageTransforms[imageIndex];
        if (transform.empty() || transform.rows < 2 || transform.cols < 3) {
            continue;
        }

        for (const cv::Point2d& point : edges[imageIndex].raw) {
            const cv::Point2d canvasPoint = transformPoint(transform, point);
            if (!std::isfinite(canvasPoint.x) || !std::isfinite(canvasPoint.y)) {
                continue;
            }
            samples.push_back({canvasPoint.x, canvasPoint.y});
        }
    }

    return !samples.empty();
}

double radiusFromYPx(const double yPx, const stitch::StitchPipelineConfig& config)
{
    const double sign = config.designInvertY ? -1.0 : 1.0;
    return sign * yPx * config.designPixelSizeMm;
}

double stepTransitionCenterSMm(const stitch::StitchPipelineConfig& config)
{
    return config.designReverseZ
               ? (kDesignProfileMaxZMm - config.designStepTransitionOriginalZMm)
               : config.designStepTransitionOriginalZMm;
}

bool isInsideStepTransitionRegion(const double sMm,
                                  const stitch::StitchPipelineConfig& config)
{
    if (!config.designIgnoreStepTransition ||
        !std::isfinite(sMm) ||
        !std::isfinite(config.designStepTransitionOriginalZMm) ||
        !(config.designStepTransitionHalfWidthMm > 1e-9)) {
        return false;
    }

    const double centerSMm = stepTransitionCenterSMm(config);
    return std::abs(sMm - centerSMm) <= config.designStepTransitionHalfWidthMm + 1e-9;
}

std::vector<PointRun> buildAcceptedRuns(const std::vector<GeneratrixCandidatePoint>& points,
                                        const stitch::StitchPipelineConfig& config)
{
    std::vector<PointRun> runs;
    if (points.empty()) {
        return runs;
    }

    std::size_t runBegin = 0;
    bool inRun = false;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const bool accepted = points[index].slopeAccepted;
        bool connected = false;
        if (accepted && inRun) {
            const GeneratrixCandidatePoint& previous = points[index - 1];
            const double deltaRadiusMm = std::abs(radiusFromYPx(points[index].yPx, config) -
                                                  radiusFromYPx(previous.yPx, config));
            connected = previous.slopeAccepted && deltaRadiusMm <= kMaxRadiusJumpForSameRunMm;
        }

        if (accepted && (!inRun || !connected)) {
            if (inRun) {
                PointRun run;
                run.beginIndex = runBegin;
                run.endIndex = index - 1;
                run.pointCount = run.endIndex - run.beginIndex + 1;
                run.lengthMm = (points[run.endIndex].xPx - points[run.beginIndex].xPx) * config.designPixelSizeMm;
                runs.push_back(run);
            }
            runBegin = index;
            inRun = true;
            continue;
        }

        if (!accepted && inRun) {
            PointRun run;
            run.beginIndex = runBegin;
            run.endIndex = index - 1;
            run.pointCount = run.endIndex - run.beginIndex + 1;
            run.lengthMm = (points[run.endIndex].xPx - points[run.beginIndex].xPx) * config.designPixelSizeMm;
            runs.push_back(run);
            inRun = false;
        }
    }

    if (inRun) {
        PointRun run;
        run.beginIndex = runBegin;
        run.endIndex = points.size() - 1;
        run.pointCount = run.endIndex - run.beginIndex + 1;
        run.lengthMm = (points[run.endIndex].xPx - points[run.beginIndex].xPx) * config.designPixelSizeMm;
        runs.push_back(run);
    }

    return runs;
}

EvaluationResult evaluateAnchorCandidate(const std::vector<GeneratrixCandidatePoint>& points,
                                         const std::size_t anchorPointIndex,
                                         const stitch::StitchPipelineConfig& config)
{
    EvaluationResult evaluation;
    if (anchorPointIndex >= points.size()) {
        return evaluation;
    }

    const DesignEval leftDesign = evalDesignRadiusCompare(0.0, config.designReverseZ);
    if (!leftDesign.valid) {
        return evaluation;
    }

    const double anchorXPx = points[anchorPointIndex].xPx;
    const double anchorMeasuredRadiusMm = radiusFromYPx(points[anchorPointIndex].yPx, config);
    const double drAnchor =
        config.designAnchorRadialToLeftEndpoint ? (leftDesign.r_mm - anchorMeasuredRadiusMm) : 0.0;

    evaluation.profilePoints.reserve(points.size() - anchorPointIndex);
    std::vector<double> normalErrorsUm;
    normalErrorsUm.reserve(points.size() - anchorPointIndex);

    for (std::size_t index = anchorPointIndex; index < points.size(); ++index) {
        const GeneratrixCandidatePoint& inputPoint = points[index];
        if (!inputPoint.slopeAccepted) {
            continue;
        }

        const double sAlignedMm = (inputPoint.xPx - anchorXPx) * config.designPixelSizeMm;
        if (sAlignedMm < config.designTrimLeftAfterEndpointMm - 1e-9) {
            continue;
        }
        if (isInsideStepTransitionRegion(sAlignedMm, config)) {
            continue;
        }

        const DesignEval design = evalDesignRadiusCompare(sAlignedMm, config.designReverseZ);
        if (!design.valid) {
            continue;
        }

        const double rAlignedMm = radiusFromYPx(inputPoint.yPx, config) + drAnchor;
        const double radialErrorMm = rAlignedMm - design.r_mm;
        const double normalErrorMm = radialErrorMm / std::sqrt(1.0 + design.dr_dz * design.dr_dz);
        const double normalErrorUm = normalErrorMm * 1000.0;

        DesignErrorProfilePoint point;
        point.index = inputPoint.index;
        point.supportCount = inputPoint.supportCount;
        point.xPx = inputPoint.xPx;
        point.yPx = inputPoint.yPx;
        point.yStdPx = inputPoint.yStdPx;
        point.sAlignedMm = sAlignedMm;
        point.rAlignedMm = rAlignedMm;
        point.designRadiusMm = design.r_mm;
        point.designDerivative = design.dr_dz;
        point.radialErrorMm = radialErrorMm;
        point.radialErrorUm = radialErrorMm * 1000.0;
        point.normalErrorMm = normalErrorMm;
        point.normalErrorUm = normalErrorUm;
        point.profileErrorUm = 0.0;
        point.isUsed = true;
        evaluation.profilePoints.push_back(point);
        normalErrorsUm.push_back(normalErrorUm);
    }

    const std::size_t usedCount = evaluation.profilePoints.size();
    if (usedCount < kMinUsedPointCount) {
        return evaluation;
    }

    evaluation.ok = true;
    evaluation.dzMm = 0.0;
    evaluation.drMm = drAnchor;
    evaluation.summary.dzMm = 0.0;
    evaluation.summary.drMm = drAnchor;
    evaluation.summary.designReverseZ = config.designReverseZ;
    evaluation.summary.useLeftEndpointAnchor = config.designUseLeftEndpointAnchor;
    evaluation.summary.evaluateProfileForm = config.designEvaluateProfileForm;
    evaluation.summary.anchorXPx = anchorXPx;
    evaluation.summary.anchorYPx = points[anchorPointIndex].yPx;
    evaluation.summary.pixelSizeMm = config.designPixelSizeMm;
    evaluation.summary.usedCount = usedCount;
    evaluation.summary.normalStats = buildErrorStats(normalErrorsUm);
    evaluation.summary.meanNormalErrorUm = evaluation.summary.normalStats.meanUm;
    evaluation.summary.profileStats = {};
    evaluation.objective = evaluation.summary.normalStats.rmseUm;
    return evaluation;
}

std::size_t chooseBestAnchorPointIndex(const std::vector<GeneratrixCandidatePoint>& points,
                                       const stitch::StitchPipelineConfig& config)
{
    const std::vector<PointRun> runs = buildAcceptedRuns(points, config);
    std::size_t fallbackIndex = points.size();
    EvaluationResult bestEvaluation;
    std::size_t bestIndex = points.size();

    for (const PointRun& run : runs) {
        if (fallbackIndex == points.size()) {
            fallbackIndex = run.beginIndex;
        }
        if (run.pointCount < kMinAnchorRunPointCount) {
            continue;
        }

        const EvaluationResult current = evaluateAnchorCandidate(points, run.beginIndex, config);
        if (!current.ok) {
            continue;
        }

        if (bestIndex == points.size() || current.objective < bestEvaluation.objective) {
            bestEvaluation = current;
            bestIndex = run.beginIndex;
        }
    }

    if (bestIndex != points.size()) {
        return bestIndex;
    }
    if (fallbackIndex != points.size()) {
        return fallbackIndex;
    }

    auto firstAcceptedIt = std::find_if(points.begin(), points.end(), [](const auto& point) {
        return point.slopeAccepted;
    });
    return firstAcceptedIt != points.end() ? static_cast<std::size_t>(std::distance(points.begin(), firstAcceptedIt))
                                           : 0;
}

PointRun findRunContainingIndex(const std::vector<PointRun>& runs, const std::size_t index)
{
    for (const PointRun& run : runs) {
        if (index >= run.beginIndex && index <= run.endIndex) {
            return run;
        }
    }
    return {};
}

bool evalPointAgainstDesign(const GeneratrixCandidatePoint& point,
                            const stitch::StitchPipelineConfig& config,
                            const double drMm,
                            double& normalErrorUm)
{
    if (isInsideStepTransitionRegion(point.sBaseMm, config)) {
        return false;
    }

    const DesignEval design = evalDesignRadiusCompare(point.sBaseMm, config.designReverseZ);
    if (!design.valid) {
        return false;
    }

    const double rAlignedMm = point.rRawMm + drMm;
    const double radialErrorMm = rAlignedMm - design.r_mm;
    const double normalErrorMm = radialErrorMm / std::sqrt(1.0 + design.dr_dz * design.dr_dz);
    normalErrorUm = normalErrorMm * 1000.0;
    return true;
}

ErrorWindowStats evaluateErrorWindow(const MeasuredGeneratrixData& data,
                                     const std::size_t startIndex,
                                     const std::size_t endIndex,
                                     const stitch::StitchPipelineConfig& config,
                                     const double drMm)
{
    ErrorWindowStats stats;
    if (startIndex >= data.points.size() || endIndex >= data.points.size() || endIndex <= startIndex) {
        return stats;
    }

    double sumUm = 0.0;
    double sumSquaresUm = 0.0;
    std::size_t validCount = 0;
    for (std::size_t index = startIndex; index <= endIndex; ++index) {
        double normalErrorUm = 0.0;
        if (!evalPointAgainstDesign(data.points[index], config, drMm, normalErrorUm)) {
            continue;
        }
        sumUm += normalErrorUm;
        sumSquaresUm += normalErrorUm * normalErrorUm;
        ++validCount;
    }

    if (validCount < kMinUsedPointCount) {
        return stats;
    }

    stats.valid = true;
    stats.meanUm = sumUm / static_cast<double>(validCount);
    stats.rmseUm = std::sqrt(sumSquaresUm / static_cast<double>(validCount));
    stats.absMeanUm = std::abs(stats.meanUm);
    return stats;
}

std::size_t findBestContinuationStartIndex(const MeasuredGeneratrixData& data,
                                           const PointRun& run,
                                           const std::size_t minStartIndex,
                                           const bool preferEarliestAccepted,
                                           const double maxWindowRmseUm,
                                           const double maxWindowAbsMeanUm,
                                           const stitch::StitchPipelineConfig& config,
                                           const double drMm)
{
    if (run.pointCount < kMinUsedPointCount) {
        return data.points.size();
    }

    const std::size_t firstStartIndex = std::max(run.beginIndex, minStartIndex);
    if (firstStartIndex > run.endIndex) {
        return data.points.size();
    }

    const std::size_t runAvailableCount = run.endIndex - firstStartIndex + 1;
    const std::size_t maxOffset = std::min<std::size_t>(kMaxContinuationSearchOffset, runAvailableCount - 1);
    std::size_t bestIndex = data.points.size();
    double bestRmseUm = std::numeric_limits<double>::infinity();
    double bestAbsMeanUm = std::numeric_limits<double>::infinity();
    std::size_t bestAcceptedIndex = data.points.size();
    double bestAcceptedRmseUm = std::numeric_limits<double>::infinity();
    double bestAcceptedAbsMeanUm = std::numeric_limits<double>::infinity();

    for (std::size_t offset = 0; offset <= maxOffset; ++offset) {
        const std::size_t startIndex = firstStartIndex + offset;
        const std::size_t endIndex =
            std::min(run.endIndex, startIndex + kContinuationSearchWindowPointCount - 1);
        if (endIndex <= startIndex || (endIndex - startIndex + 1) < kMinUsedPointCount) {
            break;
        }

        const ErrorWindowStats stats = evaluateErrorWindow(data, startIndex, endIndex, config, drMm);
        if (!stats.valid) {
            continue;
        }

        if (stats.rmseUm < bestRmseUm ||
            (std::abs(stats.rmseUm - bestRmseUm) < 1e-9 && stats.absMeanUm < bestAbsMeanUm)) {
            bestIndex = startIndex;
            bestRmseUm = stats.rmseUm;
            bestAbsMeanUm = stats.absMeanUm;
        }

        if (stats.rmseUm <= maxWindowRmseUm &&
            stats.absMeanUm <= maxWindowAbsMeanUm &&
            (bestAcceptedIndex == data.points.size() ||
             stats.rmseUm < bestAcceptedRmseUm ||
             (std::abs(stats.rmseUm - bestAcceptedRmseUm) < 1e-9 &&
              stats.absMeanUm < bestAcceptedAbsMeanUm))) {
            if (preferEarliestAccepted) {
                return startIndex;
            }
            bestAcceptedIndex = startIndex;
            bestAcceptedRmseUm = stats.rmseUm;
            bestAcceptedAbsMeanUm = stats.absMeanUm;
        }
    }

    if (bestAcceptedIndex != data.points.size()) {
        return bestAcceptedIndex;
    }
    if (bestIndex != data.points.size() &&
        bestRmseUm <= maxWindowRmseUm * 1.5 &&
        bestAbsMeanUm <= maxWindowAbsMeanUm * 1.5) {
        return bestIndex;
    }
    return data.points.size();
}

MeasuredGeneratrixData applyAcceptedRuns(MeasuredGeneratrixData data,
                                         const stitch::StitchPipelineConfig& config,
                                         const double drMm)
{
    std::vector<std::pair<std::size_t, std::size_t>> acceptedRanges;
    const PointRun primaryRun = findRunContainingIndex(data.runs, data.anchorPointIndex);
    std::size_t primaryRunStart = data.anchorPointIndex;
    if (primaryRun.pointCount > 0) {
        const std::size_t stablePrimaryStart =
            findBestContinuationStartIndex(data,
                                           primaryRun,
                                           data.anchorPointIndex,
                                           true,
                                           kPrimaryStartWindowRmseUm,
                                           kPrimaryStartWindowAbsMeanUm,
                                           config,
                                           drMm);
        if (stablePrimaryStart != data.points.size()) {
            primaryRunStart = stablePrimaryStart;
        }
    }
    acceptedRanges.emplace_back(primaryRunStart, data.primaryRunEndIndex);

    bool anchorRunPassed = false;
    for (const PointRun& run : data.runs) {
        if (run.pointCount == 0) {
            continue;
        }
        if (!anchorRunPassed) {
            if (data.anchorPointIndex >= run.beginIndex && data.anchorPointIndex <= run.endIndex) {
                anchorRunPassed = true;
            }
            continue;
        }

        const std::size_t continuationStart =
            findBestContinuationStartIndex(data,
                                           run,
                                           run.beginIndex,
                                           false,
                                           kMaxContinuationWindowRmseUm,
                                           kMaxContinuationWindowAbsMeanUm,
                                           config,
                                           drMm);
        if (continuationStart != data.points.size()) {
            acceptedRanges.emplace_back(continuationStart, run.endIndex);
        }
    }

    double maxAcceptedBaseMm = 0.0;
    for (const auto& range : acceptedRanges) {
        for (std::size_t index = range.first; index <= range.second; ++index) {
            maxAcceptedBaseMm = std::max(maxAcceptedBaseMm, data.points[index].sBaseMm);
        }
    }
    data.maxBaseMm = maxAcceptedBaseMm;

    for (GeneratrixCandidatePoint& point : data.points) {
        point.trimAccepted = false;
    }

    for (const auto& range : acceptedRanges) {
        for (std::size_t index = range.first; index <= range.second; ++index) {
            GeneratrixCandidatePoint& point = data.points[index];
            const bool leftTrimOk = point.sBaseMm >= config.designTrimLeftAfterEndpointMm - 1e-9;
            const bool rightTrimOk = point.sBaseMm <= maxAcceptedBaseMm - config.designTrimRightMm + 1e-9;
            const bool transitionOk = !isInsideStepTransitionRegion(point.sBaseMm, config);
            point.trimAccepted = point.slopeAccepted && leftTrimOk && rightTrimOk && transitionOk;
        }
    }

    return data;
}

MeasuredGeneratrixData rebaseAcceptedGeneratrixToIndex(MeasuredGeneratrixData data,
                                                       const std::size_t rebaseIndex,
                                                       const stitch::StitchPipelineConfig& config)
{
    if (rebaseIndex >= data.points.size()) {
        return data;
    }

    const GeneratrixCandidatePoint& rebasePoint = data.points[rebaseIndex];
    const double rebaseBaseMm = rebasePoint.sBaseMm;
    if (!std::isfinite(rebaseBaseMm)) {
        return data;
    }

    for (GeneratrixCandidatePoint& point : data.points) {
        point.sBaseMm -= rebaseBaseMm;
    }

    data.maxBaseMm = std::max(0.0, data.maxBaseMm - rebaseBaseMm);
    data.anchorPointIndex = rebaseIndex;
    data.anchorXPx = rebasePoint.xPx;
    data.anchorYPx = rebasePoint.yPx;
    data.anchorMeasuredRadiusMm = radiusFromYPx(rebasePoint.yPx, config);
    data.anchorBaseMm += rebaseBaseMm;

    if (data.primaryRunBeginIndex < data.points.size() && data.primaryRunBeginIndex < rebaseIndex) {
        data.primaryRunBeginIndex = rebaseIndex;
    }
    if (data.primaryRunEndIndex < data.primaryRunBeginIndex) {
        data.primaryRunEndIndex = data.primaryRunBeginIndex;
    }
    return data;
}

MeasuredGeneratrixData trimResidualLeftEndpointTransition(MeasuredGeneratrixData data,
                                                          const stitch::StitchPipelineConfig& config,
                                                          const double drMm)
{
    auto firstAcceptedIt = std::find_if(data.points.begin(), data.points.end(), [](const auto& point) {
        return point.trimAccepted;
    });
    if (firstAcceptedIt == data.points.end()) {
        return data;
    }

    const std::size_t firstAcceptedIndex = static_cast<std::size_t>(std::distance(data.points.begin(), firstAcceptedIt));
    const double searchEndMm =
        firstAcceptedIt->sBaseMm + std::min(kLeftTransitionSearchMaxMm, std::max(0.0, data.maxBaseMm * 0.2));

    for (std::size_t startIndex = firstAcceptedIndex; startIndex < data.points.size(); ++startIndex) {
        const GeneratrixCandidatePoint& startPoint = data.points[startIndex];
        if (!startPoint.trimAccepted) {
            continue;
        }
        if (startPoint.sBaseMm > searchEndMm + 1e-9) {
            break;
        }

        const double windowEndMm = startPoint.sBaseMm + kLeftTransitionWindowMm;
        std::vector<double> errorsUm;
        errorsUm.reserve(kLeftTransitionMinPointCount * 2);
        for (std::size_t index = startIndex; index < data.points.size(); ++index) {
            const GeneratrixCandidatePoint& point = data.points[index];
            if (!point.trimAccepted) {
                continue;
            }
            if (point.sBaseMm > windowEndMm + 1e-9) {
                break;
            }

            double normalErrorUm = 0.0;
            if (!evalPointAgainstDesign(point, config, drMm, normalErrorUm)) {
                continue;
            }
            errorsUm.push_back(normalErrorUm);
        }

        if (errorsUm.size() < kLeftTransitionMinPointCount) {
            continue;
        }

        const double medianErrorUm = medianOfValues(errorsUm);
        std::vector<double> centeredAbsErrorsUm;
        centeredAbsErrorsUm.reserve(errorsUm.size());
        for (const double errorUm : errorsUm) {
            centeredAbsErrorsUm.push_back(std::abs(errorUm - medianErrorUm));
        }

        const double medianCenteredUm = medianOfValues(centeredAbsErrorsUm);
        const double p90CenteredUm = percentileAbs(centeredAbsErrorsUm, 0.90);
        if (medianCenteredUm > kLeftTransitionMedianCenteredErrorUm ||
            p90CenteredUm > kLeftTransitionP90CenteredErrorUm) {
            continue;
        }

        for (std::size_t index = firstAcceptedIndex; index < startIndex; ++index) {
            data.points[index].trimAccepted = false;
        }
        if (data.primaryRunBeginIndex < data.points.size() && data.primaryRunBeginIndex < startIndex) {
            data.primaryRunBeginIndex = startIndex;
        }
        return data;
    }

    return data;
}

MeasuredGeneratrixData buildBestFitSearchData(MeasuredGeneratrixData data)
{
    constexpr std::size_t kMaxSearchPoints = 600;
    std::vector<GeneratrixCandidatePoint> accepted;
    accepted.reserve(data.points.size());
    for (const GeneratrixCandidatePoint& point : data.points) {
        if (point.trimAccepted) {
            accepted.push_back(point);
        }
    }
    if (accepted.size() <= kMaxSearchPoints) {
        data.points = std::move(accepted);
        return data;
    }

    const std::size_t stride =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(accepted.size()) /
                                                                   static_cast<double>(kMaxSearchPoints))));
    std::vector<GeneratrixCandidatePoint> sampled;
    sampled.reserve(kMaxSearchPoints + 2);
    for (std::size_t index = 0; index < accepted.size(); index += stride) {
        sampled.push_back(accepted[index]);
    }
    if (sampled.empty() || sampled.back().index != accepted.back().index) {
        sampled.push_back(accepted.back());
    }
    data.points = std::move(sampled);
    return data;
}

MeasuredGeneratrixData rotateMeasuredData(MeasuredGeneratrixData data,
                                          const double centerSMm,
                                          const double centerRMm,
                                          const double thetaDeg)
{
    const double theta = thetaDeg * CV_PI / 180.0;
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    for (GeneratrixCandidatePoint& pt : data.points) {
        const double ds = pt.sBaseMm - centerSMm;
        const double drVal = pt.rRawMm - centerRMm;
        pt.sBaseMm = centerSMm + ds * ct - drVal * st;
        pt.rRawMm = centerRMm + ds * st + drVal * ct;
    }
    return data;
}

MeasuredGeneratrixData extractMeasuredGeneratrixForDesignComparison(
    const std::vector<StitchedContourSample>& contourPx,
    const stitch::StitchPipelineConfig& config)
{
    MeasuredGeneratrixData data;
    if (contourPx.empty()) {
        return data;
    }

    const double binWidthPx = std::max(1e-6, config.designProfileBinWidthPx);
    double minXPx = std::numeric_limits<double>::infinity();
    for (const StitchedContourSample& sample : contourPx) {
        minXPx = std::min(minXPx, sample.xPx);
    }

    struct BinData {
        std::vector<double> xs;
        std::vector<double> ys;
        std::vector<StitchedContourSample> samples;
    };

    std::map<long long, BinData> bins;
    for (const StitchedContourSample& sample : contourPx) {
        const long long binIndex = static_cast<long long>(std::floor((sample.xPx - minXPx) / binWidthPx));
        BinData& bin = bins[binIndex];
        bin.xs.push_back(sample.xPx);
        bin.ys.push_back(sample.yPx);
        bin.samples.push_back(sample);
    }

    data.points.reserve(bins.size());
    for (auto& [binIndex, bin] : bins) {
        (void)binIndex;

        GeneratrixCandidatePoint point;
        point.index = data.points.size() + 1;
        point.supportCount = bin.samples.size();
        if (config.designUseUpperEnvelope) {
            const std::size_t binSize = bin.samples.size();
            if (binSize <= 1) {
                point.xPx = bin.samples.front().xPx;
                point.yPx = bin.samples.front().yPx;
                point.yStdPx = 0.0;
            } else {
                // Sort by Y ascending (minimum Y = maximum radius when Y is inverted)
                std::sort(bin.samples.begin(), bin.samples.end(),
                          [](const auto& lhs, const auto& rhs) {
                              if (lhs.yPx != rhs.yPx) return lhs.yPx < rhs.yPx;
                              return lhs.xPx < rhs.xPx;
                          });
                // Adaptive quantile: small bins use median, large bins use configured quantile
                const double baseQ = std::clamp(config.designUpperEnvelopeQuantile, 0.0, 1.0);
                const double q = (binSize >= 5) ? baseQ : 0.5;  // median for small bins
                const std::size_t idx =
                    std::min(binSize - 1,
                             static_cast<std::size_t>(static_cast<double>(binSize - 1) * (1.0 - q)));
                point.xPx = bin.samples[idx].xPx;
                point.yPx = bin.samples[idx].yPx;

                // Compute Y standard deviation in this bin
                double ySum = 0.0;
                double ySumSq = 0.0;
                for (const auto& s : bin.samples) {
                    ySum += s.yPx;
                    ySumSq += s.yPx * s.yPx;
                }
                const double yMean = ySum / static_cast<double>(binSize);
                const double yVar = ySumSq / static_cast<double>(binSize) - yMean * yMean;
                point.yStdPx = (yVar > 0.0) ? std::sqrt(yVar) : 0.0;
            }
        } else {
            point.xPx = std::accumulate(bin.xs.begin(), bin.xs.end(), 0.0) / static_cast<double>(bin.xs.size());
            point.yPx = medianOfValues(std::move(bin.ys));
            point.yStdPx = 0.0;
        }
        data.points.push_back(point);
    }

    std::sort(data.points.begin(), data.points.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.xPx != rhs.xPx) {
            return lhs.xPx < rhs.xPx;
        }
        return lhs.yPx < rhs.yPx;
    });

    if (data.points.empty()) {
        return data;
    }

    const int halfWindow = std::max(1, config.designSlopeWindow / 2);
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        GeneratrixCandidatePoint& point = data.points[i];
        const std::size_t leftIndex = i > static_cast<std::size_t>(halfWindow) ? i - static_cast<std::size_t>(halfWindow) : 0;
        const std::size_t rightIndex =
            std::min(data.points.size() - 1, i + static_cast<std::size_t>(halfWindow));
        if (leftIndex == rightIndex) {
            point.slopeAbs = 0.0;
            point.slopeAccepted = true;
            continue;
        }

        const double dx = data.points[rightIndex].xPx - data.points[leftIndex].xPx;
        const double dy = data.points[rightIndex].yPx - data.points[leftIndex].yPx;
        point.slopeAbs = std::abs(dx) < 1e-9 ? std::numeric_limits<double>::infinity() : std::abs(dy / dx);
        point.slopeAccepted =
            !config.designFilterEndFaceEdges || (std::isfinite(point.slopeAbs) &&
                                                 point.slopeAbs <= config.designMaxAbsSlopeForGeneratrix);
    }

    auto firstAcceptedIt = std::find_if(data.points.begin(), data.points.end(), [](const auto& point) {
        return point.slopeAccepted;
    });
    if (firstAcceptedIt == data.points.end()) {
        return {};
    }

    const std::vector<PointRun> runs = buildAcceptedRuns(data.points, config);
    data.runs = runs;
    data.anchorPointIndex =
        config.designUseLeftEndpointAnchor ? chooseBestAnchorPointIndex(data.points, config) : 0;
    if (data.anchorPointIndex >= data.points.size()) {
        return {};
    }

    const PointRun primaryRun = findRunContainingIndex(runs, data.anchorPointIndex);
    data.primaryRunBeginIndex = primaryRun.pointCount > 0 ? primaryRun.beginIndex : data.anchorPointIndex;
    data.primaryRunEndIndex = primaryRun.pointCount > 0 ? primaryRun.endIndex : data.points.size() - 1;

    const GeneratrixCandidatePoint& basePoint = data.points[data.anchorPointIndex];
    data.anchorXPx = basePoint.xPx;
    data.anchorYPx = basePoint.yPx;
    data.anchorMeasuredRadiusMm = radiusFromYPx(basePoint.yPx, config);
    data.anchorBaseMm = 0.0;

    double maxAcceptedBaseMm = 0.0;
    for (std::size_t index = 0; index < data.points.size(); ++index) {
        GeneratrixCandidatePoint& point = data.points[index];
        point.sBaseMm = (point.xPx - data.anchorXPx) * config.designPixelSizeMm;
        point.rRawMm = radiusFromYPx(point.yPx, config);
        const bool inPrimaryRun = index >= data.primaryRunBeginIndex && index <= data.primaryRunEndIndex;
        if (point.slopeAccepted && inPrimaryRun) {
            maxAcceptedBaseMm = std::max(maxAcceptedBaseMm, point.sBaseMm);
        }
    }
    data.maxBaseMm = maxAcceptedBaseMm;

    for (std::size_t index = 0; index < data.points.size(); ++index) {
        GeneratrixCandidatePoint& point = data.points[index];
        const bool leftTrimOk = point.sBaseMm >= config.designTrimLeftAfterEndpointMm - 1e-9;
        const bool rightTrimOk = point.sBaseMm <= maxAcceptedBaseMm - config.designTrimRightMm + 1e-9;
        const bool afterAnchor = index >= data.anchorPointIndex;
        const bool inPrimaryRun = index >= data.primaryRunBeginIndex && index <= data.primaryRunEndIndex;
        const bool transitionOk = !isInsideStepTransitionRegion(point.sBaseMm, config);
        point.trimAccepted =
            point.slopeAccepted && inPrimaryRun && afterAnchor && leftTrimOk && rightTrimOk && transitionOk;
    }

    return data;
}

double refineAnchorDrByStableWindow(const MeasuredGeneratrixData& data,
                                    const stitch::StitchPipelineConfig& config,
                                    const double initialDrMm)
{
    if (data.points.empty()) {
        return initialDrMm;
    }

    const double stableWindowBeginMm = std::max(1.0, config.designTrimLeftAfterEndpointMm);
    const double stableWindowEndMm = std::min(20.0, data.maxBaseMm * 0.6);
    if (stableWindowEndMm <= stableWindowBeginMm + 1e-9) {
        return initialDrMm;
    }

    // 多窗口锚定：将稳定窗口等分为若干子窗口，对每个子窗口取径向误差中值，
    // 再用子窗口中值的中值作为最终修正量，降低单一窗口对局部缺陷的敏感性。
    constexpr std::size_t kSubWindowCount = 4;
    const double totalWindowMm = stableWindowEndMm - stableWindowBeginMm;
    const double subWindowMm = totalWindowMm / static_cast<double>(kSubWindowCount);

    std::vector<double> subWindowMedians;
    subWindowMedians.reserve(kSubWindowCount);

    for (std::size_t w = 0; w < kSubWindowCount; ++w) {
        const double wBegin = stableWindowBeginMm + static_cast<double>(w) * subWindowMm;
        const double wEnd = wBegin + subWindowMm;

        std::vector<double> radialErrorsMm;
        radialErrorsMm.reserve(data.points.size() / kSubWindowCount);
        for (const GeneratrixCandidatePoint& point : data.points) {
            if (!point.trimAccepted) {
                continue;
            }
            if (point.sBaseMm < wBegin || point.sBaseMm > wEnd) {
                continue;
            }

            const DesignEval design = evalDesignRadiusCompare(point.sBaseMm, config.designReverseZ);
            if (!design.valid) {
                continue;
            }

            radialErrorsMm.push_back(point.rRawMm + initialDrMm - design.r_mm);
        }

        if (radialErrorsMm.size() >= kMinUsedPointCount / 2) {
            subWindowMedians.push_back(medianOfValues(std::move(radialErrorsMm)));
        }
    }

    if (subWindowMedians.empty()) {
        // 回退：单窗口中值
        std::vector<double> radialErrorsMm;
        radialErrorsMm.reserve(data.points.size());
        for (const GeneratrixCandidatePoint& point : data.points) {
            if (!point.trimAccepted) {
                continue;
            }
            if (point.sBaseMm < stableWindowBeginMm || point.sBaseMm > stableWindowEndMm) {
                continue;
            }

            const DesignEval design = evalDesignRadiusCompare(point.sBaseMm, config.designReverseZ);
            if (!design.valid) {
                continue;
            }

            radialErrorsMm.push_back(point.rRawMm + initialDrMm - design.r_mm);
        }

        if (radialErrorsMm.size() < kMinUsedPointCount) {
            return initialDrMm;
        }

        return initialDrMm - medianOfValues(std::move(radialErrorsMm));
    }

    return initialDrMm - medianOfValues(std::move(subWindowMedians));
}

EvaluationResult evaluateAlignedProfile(const MeasuredGeneratrixData& data,
                                        const stitch::StitchPipelineConfig& config,
                                        const double dzMm,
                                        const double drMm,
                                        const bool usePolylineDistance = true,
                                        const double axialScaleFactor = 1.0,
                                        const double axialQuadraticTermMm = 0.0)
{
    EvaluationResult evaluation;
    evaluation.dzMm = dzMm;
    evaluation.drMm = drMm;
    evaluation.axialScaleFactor = axialScaleFactor;
    evaluation.axialQuadraticTermMm = axialQuadraticTermMm;
    evaluation.profilePoints.reserve(data.points.size());
    const std::vector<DesignPolylinePoint> designPolyline =
        usePolylineDistance ? buildDesignPolyline(config.designReverseZ) : std::vector<DesignPolylinePoint>{};

    std::vector<double> normalErrorsUm;
    std::vector<std::size_t> usedPointIndices;
    normalErrorsUm.reserve(data.points.size());
    usedPointIndices.reserve(data.points.size());

    for (const GeneratrixCandidatePoint& inputPoint : data.points) {
        DesignErrorProfilePoint point;
        point.index = inputPoint.index;
        point.supportCount = inputPoint.supportCount;
        point.xPx = inputPoint.xPx;
        point.yPx = inputPoint.yPx;
        const double normalizedBase =
            data.maxBaseMm > 1e-9 ? std::clamp(inputPoint.sBaseMm / data.maxBaseMm, 0.0, 1.0) : 0.0;
        point.sAlignedMm =
            inputPoint.sBaseMm * axialScaleFactor +
            dzMm +
            axialQuadraticTermMm * normalizedBase * normalizedBase;
        point.rAlignedMm = inputPoint.rRawMm + drMm;
        point.designRadiusMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignSMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignRMm = std::numeric_limits<double>::quiet_NaN();
        point.designSegmentIndex = 0;
        point.designDerivative = std::numeric_limits<double>::quiet_NaN();
        point.radialErrorMm = std::numeric_limits<double>::quiet_NaN();
        point.radialErrorUm = std::numeric_limits<double>::quiet_NaN();
        point.signedDistanceMm = std::numeric_limits<double>::quiet_NaN();
        point.signedDistanceUm = std::numeric_limits<double>::quiet_NaN();
        point.legacyNormalErrorUm = std::numeric_limits<double>::quiet_NaN();
        point.normalErrorMm = std::numeric_limits<double>::quiet_NaN();
        point.normalErrorUm = std::numeric_limits<double>::quiet_NaN();
        point.profileErrorUm = std::numeric_limits<double>::quiet_NaN();

        if (inputPoint.trimAccepted &&
            !isInsideStepTransitionRegion(point.sAlignedMm, config)) {
            const DesignEval design = evalDesignRadiusCompare(point.sAlignedMm, config.designReverseZ);
            const NearestDesignPoint nearest =
                usePolylineDistance ? findNearestDesignPoint(designPolyline, point.sAlignedMm, point.rAlignedMm)
                                    : NearestDesignPoint{};
            if (design.valid && (!usePolylineDistance || nearest.valid)) {
                point.designRadiusMm = design.r_mm;
                point.radialErrorMm = point.rAlignedMm - point.designRadiusMm;
                point.radialErrorUm = point.radialErrorMm * 1000.0;
                point.legacyNormalErrorUm =
                    point.radialErrorMm / std::sqrt(1.0 + design.dr_dz * design.dr_dz) * 1000.0;
                if (usePolylineDistance) {
                    point.nearestDesignSMm = nearest.sMm;
                    point.nearestDesignRMm = nearest.rMm;
                    point.designSegmentIndex = nearest.segmentIndex;
                    point.designDerivative =
                        std::abs(nearest.tangentDs) < 1e-12
                            ? std::numeric_limits<double>::infinity()
                            : nearest.tangentDr / nearest.tangentDs;
                    point.signedDistanceMm = nearest.signedDistanceMm;
                } else {
                    point.nearestDesignSMm = point.sAlignedMm;
                    point.nearestDesignRMm = design.r_mm;
                    point.designDerivative = design.dr_dz;
                    point.signedDistanceMm = point.legacyNormalErrorUm / 1000.0;
                }
                point.signedDistanceUm = point.signedDistanceMm * 1000.0;
                point.normalErrorMm = point.signedDistanceMm;
                point.normalErrorUm = point.normalErrorMm * 1000.0;
                point.isUsed = true;
            }
        }

        evaluation.profilePoints.push_back(point);
        if (point.isUsed) {
            usedPointIndices.push_back(evaluation.profilePoints.size() - 1);
            normalErrorsUm.push_back(point.normalErrorUm);
        }
    }

    const std::size_t usedCount = usedPointIndices.size();
    if (usedCount < kMinUsedPointCount) {
        return evaluation;
    }

    const ErrorStats normalStatsAll = buildErrorStats(normalErrorsUm);
    std::vector<double> profileErrorsForOutlierUm;
    profileErrorsForOutlierUm.reserve(usedCount);
    for (const std::size_t pointIndex : usedPointIndices) {
        const DesignErrorProfilePoint& point = evaluation.profilePoints[pointIndex];
        profileErrorsForOutlierUm.push_back(point.normalErrorUm - normalStatsAll.meanUm);
    }

    // --- 轮廓误差 Hampel 离群剔除 ---
    // 在局部邻域内，若某点的轮廓误差偏离邻域中值超过阈值，标记为离群
    std::vector<bool> profileOutlierFlags(usedCount, false);
    if (config.designEnableProfileOutlierFilter && usedCount > 20) {
        constexpr int kHalfWindow = 10;
        const double kMadScale = 1.4826; // MAD → σ 的缩放因子（正态分布假设）
        const double sigma = std::max(0.5, config.designProfileOutlierSigma);

        for (std::size_t i = 0; i < usedCount; ++i) {
            const int iStart = std::max(0, static_cast<int>(i) - kHalfWindow);
            const int iEnd = std::min(static_cast<int>(usedCount) - 1,
                                       static_cast<int>(i) + kHalfWindow);
            const std::size_t nNeighbors = static_cast<std::size_t>(iEnd - iStart + 1);
            if (nNeighbors < 5) continue;

            std::vector<double> neighborErrors;
            neighborErrors.reserve(nNeighbors);
            for (int j = iStart; j <= iEnd; ++j) {
                neighborErrors.push_back(profileErrorsForOutlierUm[static_cast<std::size_t>(j)]);
            }

            const double med = medianOfValues(neighborErrors);
            std::vector<double> absDevs;
            absDevs.reserve(nNeighbors);
            for (const double e : neighborErrors) {
                absDevs.push_back(std::abs(e - med));
            }
            const double mad = medianOfValues(std::move(absDevs));
            const double threshold = sigma * std::max(kMadScale * mad, 1.0); // min scale = 1 μm

            if (std::abs(profileErrorsForOutlierUm[i] - med) > threshold) {
                profileOutlierFlags[i] = true;
            }
        }
    }

    std::vector<double> normalErrorsFiltered;
    normalErrorsFiltered.reserve(usedCount);
    for (std::size_t i = 0; i < usedCount; ++i) {
        if (!profileOutlierFlags[i]) {
            normalErrorsFiltered.push_back(normalErrorsUm[i]);
        }
    }
    if (normalErrorsFiltered.empty()) {
        normalErrorsFiltered = normalErrorsUm;
        std::fill(profileOutlierFlags.begin(), profileOutlierFlags.end(), false);
    }

    const ErrorStats normalStatsFiltered = buildErrorStats(normalErrorsFiltered);
    std::vector<double> profileErrorsFiltered;
    profileErrorsFiltered.reserve(normalErrorsFiltered.size());
    for (std::size_t i = 0; i < usedCount; ++i) {
        const std::size_t pointIndex = usedPointIndices[i];
        DesignErrorProfilePoint& point = evaluation.profilePoints[pointIndex];
        point.profileErrorUm = point.normalErrorUm - normalStatsFiltered.meanUm;
        if (profileOutlierFlags[i]) {
            point.isUsed = false;
            continue;
        }
        profileErrorsFiltered.push_back(point.profileErrorUm);
    }
    const ErrorStats profileStats = buildErrorStats(profileErrorsFiltered);
    const std::size_t finalUsedCount = profileErrorsFiltered.size();
    const std::size_t outlierCount = usedCount - finalUsedCount;

    evaluation.ok = true;
    evaluation.summary.dzMm = dzMm;
    evaluation.summary.drMm = drMm;
    evaluation.summary.designReverseZ = config.designReverseZ;
    evaluation.summary.useLeftEndpointAnchor = config.designUseLeftEndpointAnchor;
    evaluation.summary.evaluateProfileForm = config.designEvaluateProfileForm;
    evaluation.summary.axialScaleFactor = axialScaleFactor;
    evaluation.summary.axialQuadraticTermMm = axialQuadraticTermMm;
    evaluation.summary.anchorXPx = data.anchorXPx;
    evaluation.summary.anchorYPx = data.anchorYPx;
    evaluation.summary.pixelSizeMm = config.designPixelSizeMm;
    evaluation.summary.candidateCount = usedCount;
    evaluation.summary.usedCount = finalUsedCount;
    evaluation.summary.outlierCount = outlierCount;
    evaluation.summary.outlierRatio = usedCount > 0 ? static_cast<double>(outlierCount) / static_cast<double>(usedCount) : 0.0;
    evaluation.summary.meanNormalErrorUm = normalStatsFiltered.meanUm;
    evaluation.summary.absoluteAllStats = normalStatsAll;
    evaluation.summary.absoluteFilteredStats = normalStatsFiltered;
    evaluation.summary.normalStats = normalStatsFiltered;
    evaluation.summary.profileStats = profileStats;
    evaluation.objective = config.designEvaluateProfileForm ? profileStats.rmseUm : normalStatsFiltered.rmseUm;
    return evaluation;
}

double absoluteFilteredRmseObjective(const EvaluationResult& evaluation)
{
    if (!evaluation.ok) {
        return std::numeric_limits<double>::infinity();
    }
    return evaluation.summary.absoluteFilteredStats.rmseUm;
}

bool isBetterAbsoluteRefineResult(const EvaluationResult& candidate, const EvaluationResult& currentBest)
{
    if (!candidate.ok) {
        return false;
    }
    if (!currentBest.ok) {
        return true;
    }

    const double candidateAbs = absoluteFilteredRmseObjective(candidate);
    const double currentAbs = absoluteFilteredRmseObjective(currentBest);
    if (candidateAbs + 1e-9 < currentAbs) {
        return true;
    }
    if (std::abs(candidateAbs - currentAbs) > 1e-9) {
        return false;
    }

    if (candidate.summary.profileStats.rmseUm + 1e-9 < currentBest.summary.profileStats.rmseUm) {
        return true;
    }
    if (std::abs(candidate.summary.profileStats.rmseUm - currentBest.summary.profileStats.rmseUm) > 1e-9) {
        return false;
    }

    return std::abs(candidate.summary.meanNormalErrorUm) + 1e-9 <
           std::abs(currentBest.summary.meanNormalErrorUm);
}

EvaluationResult refineAbsoluteBiasAlignment(const MeasuredGeneratrixData& data,
                                            const stitch::StitchPipelineConfig& config,
                                            const double anchorDrMm,
                                            const EvaluationResult& seed)
{
    EvaluationResult best = seed;
    if (!seed.ok || !config.designEnableBestFitTranslation) {
        return best;
    }

    const double seedBiasUm = seed.summary.meanNormalErrorUm;
    if (!std::isfinite(seedBiasUm) || std::abs(seedBiasUm) < 15.0) {
        return best;
    }

    const double allowedDrMin = anchorDrMm + config.designBestFitDrMinMm;
    const double allowedDrMax = anchorDrMm + config.designBestFitDrMaxMm;
    if (!std::isfinite(allowedDrMin) || !std::isfinite(allowedDrMax) || allowedDrMin > allowedDrMax) {
        return best;
    }

    const double predictedDrMm =
        std::clamp(seed.summary.drMm - seedBiasUm / 1000.0, allowedDrMin, allowedDrMax);
    const double refineHalfWindowMm =
        std::clamp(std::max(0.05, std::abs(seedBiasUm) * 0.0013), 0.05, 0.25);
    const double refineStepMm = std::max(0.001, std::min(config.designBestFitStepMm, 0.0025));
    const double refineMinDr = std::max(allowedDrMin, predictedDrMm - refineHalfWindowMm);
    const double refineMaxDr = std::min(allowedDrMax, predictedDrMm + refineHalfWindowMm);

    const auto considerCandidate = [&](const double drMm) {
        const EvaluationResult candidate = evaluateAlignedProfile(
            data,
            config,
            seed.summary.dzMm,
            drMm,
            true,
            seed.summary.axialScaleFactor,
            seed.summary.axialQuadraticTermMm
        );
        if (isBetterAbsoluteRefineResult(candidate, best)) {
            best = candidate;
        }
    };

    considerCandidate(seed.summary.drMm);
    considerCandidate(predictedDrMm);
    for (double drMm = refineMinDr; drMm <= refineMaxDr + 1e-12; drMm += refineStepMm) {
        considerCandidate(drMm);
    }

    return best;
}

bool isBetterAxialScaleRefineResult(const EvaluationResult& candidate,
                                    const EvaluationResult& currentBest)
{
    if (!candidate.ok) {
        return false;
    }
    if (!currentBest.ok) {
        return true;
    }
    if (candidate.objective + 1e-9 < currentBest.objective) {
        return true;
    }
    if (std::abs(candidate.objective - currentBest.objective) > 1e-9) {
        return false;
    }
    return std::abs(candidate.summary.axialScaleFactor - 1.0) + 1e-12 <
           std::abs(currentBest.summary.axialScaleFactor - 1.0);
}

EvaluationResult refineAxialScaleAlignment(const MeasuredGeneratrixData& data,
                                           const stitch::StitchPipelineConfig& config,
                                           const EvaluationResult& seed)
{
    EvaluationResult best = seed;
    if (!seed.ok) {
        return best;
    }

    constexpr double kScaleMin = 0.9975;
    constexpr double kScaleMax = 1.0025;
    constexpr double kScaleStep = 0.00025;
    constexpr double kDzHalfWindowMm = 0.20;
    constexpr double kMinMeaningfulImprovementUm = 0.05;
    const double dzStepMm = std::max(0.01, std::min(config.designBestFitStepMm * 2.0, 0.02));
    const double dzMin = std::max(config.designBestFitDzMinMm, seed.summary.dzMm - kDzHalfWindowMm);
    const double dzMax = std::min(config.designBestFitDzMaxMm, seed.summary.dzMm + kDzHalfWindowMm);

    const auto considerCandidate = [&](const double scaleFactor, const double dzMm) {
        const EvaluationResult candidate =
            evaluateAlignedProfile(
                data,
                config,
                dzMm,
                seed.summary.drMm,
                true,
                scaleFactor,
                seed.summary.axialQuadraticTermMm
            );
        if (isBetterAxialScaleRefineResult(candidate, best)) {
            best = candidate;
        }
    };

    considerCandidate(seed.summary.axialScaleFactor, seed.summary.dzMm);
    considerCandidate(1.0, seed.summary.dzMm);
    for (double scaleFactor = kScaleMin; scaleFactor <= kScaleMax + 1e-12; scaleFactor += kScaleStep) {
        for (double dzMm = dzMin; dzMm <= dzMax + 1e-12; dzMm += dzStepMm) {
            considerCandidate(scaleFactor, dzMm);
        }
    }

    if (!best.ok || best.objective + kMinMeaningfulImprovementUm >= seed.objective) {
        return seed;
    }
    return best;
}

bool isBetterAxialQuadraticRefineResult(const EvaluationResult& candidate,
                                        const EvaluationResult& currentBest)
{
    if (!candidate.ok) {
        return false;
    }
    if (!currentBest.ok) {
        return true;
    }
    if (candidate.objective + 1e-9 < currentBest.objective) {
        return true;
    }
    if (std::abs(candidate.objective - currentBest.objective) > 1e-9) {
        return false;
    }
    return std::abs(candidate.summary.axialQuadraticTermMm) + 1e-12 <
           std::abs(currentBest.summary.axialQuadraticTermMm);
}

EvaluationResult refineAxialQuadraticAlignment(const MeasuredGeneratrixData& data,
                                               const stitch::StitchPipelineConfig& config,
                                               const EvaluationResult& seed)
{
    EvaluationResult best = seed;
    if (!seed.ok || data.maxBaseMm <= 1e-9) {
        return best;
    }

    constexpr double kQuadraticMinMm = -0.40;
    constexpr double kQuadraticMaxMm = 0.40;
    constexpr double kQuadraticStepMm = 0.01;
    constexpr double kDzHalfWindowMm = 0.20;
    constexpr double kMinMeaningfulImprovementUm = 0.05;
    const double dzStepMm = std::max(0.01, std::min(config.designBestFitStepMm * 2.0, 0.02));
    const double dzMin = std::max(config.designBestFitDzMinMm, seed.summary.dzMm - kDzHalfWindowMm);
    const double dzMax = std::min(config.designBestFitDzMaxMm, seed.summary.dzMm + kDzHalfWindowMm);

    const auto considerCandidate = [&](const double quadraticTermMm, const double dzMm) {
        const EvaluationResult candidate =
            evaluateAlignedProfile(data,
                                   config,
                                   dzMm,
                                   seed.summary.drMm,
                                   true,
                                   seed.summary.axialScaleFactor,
                                   quadraticTermMm);
        if (isBetterAxialQuadraticRefineResult(candidate, best)) {
            best = candidate;
        }
    };

    considerCandidate(seed.summary.axialQuadraticTermMm, seed.summary.dzMm);
    considerCandidate(0.0, seed.summary.dzMm);
    for (double quadraticTermMm = kQuadraticMinMm;
         quadraticTermMm <= kQuadraticMaxMm + 1e-12;
         quadraticTermMm += kQuadraticStepMm) {
        for (double dzMm = dzMin; dzMm <= dzMax + 1e-12; dzMm += dzStepMm) {
            considerCandidate(quadraticTermMm, dzMm);
        }
    }

    if (!best.ok || best.objective + kMinMeaningfulImprovementUm >= seed.objective) {
        return seed;
    }
    return best;
}

struct AnchoredQuadraticCorrectionFit {
    bool ok{false};
    double linearCoeffUm{0.0};
    double quadraticCoeffUm{0.0};
    double candidateRmseUm{std::numeric_limits<double>::infinity()};
};

AnchoredQuadraticCorrectionFit fitAnchoredQuadraticNormalDriftCorrection(
    const std::vector<DesignErrorProfilePoint>& points)
{
    AnchoredQuadraticCorrectionFit fit;
    std::vector<const DesignErrorProfilePoint*> usedPoints;
    usedPoints.reserve(points.size());
    double sMin = std::numeric_limits<double>::infinity();
    double sMax = -std::numeric_limits<double>::infinity();
    for (const DesignErrorProfilePoint& point : points) {
        if (!point.isUsed ||
            !std::isfinite(point.sAlignedMm) ||
            !std::isfinite(point.normalErrorUm)) {
            continue;
        }
        usedPoints.push_back(&point);
        sMin = std::min(sMin, point.sAlignedMm);
        sMax = std::max(sMax, point.sAlignedMm);
    }
    if (usedPoints.size() < 200 || !std::isfinite(sMin) || !std::isfinite(sMax) || sMax <= sMin + 1e-9) {
        return fit;
    }

    double sumX2 = 0.0;
    double sumX3 = 0.0;
    double sumX4 = 0.0;
    double sumXY = 0.0;
    double sumX2Y = 0.0;
    for (const DesignErrorProfilePoint* point : usedPoints) {
        const double x = std::clamp((point->sAlignedMm - sMin) / (sMax - sMin), 0.0, 1.0);
        const double x2 = x * x;
        sumX2 += x2;
        sumX3 += x2 * x;
        sumX4 += x2 * x2;
        sumXY += x * point->normalErrorUm;
        sumX2Y += x2 * point->normalErrorUm;
    }

    const double det = sumX2 * sumX4 - sumX3 * sumX3;
    if (std::abs(det) < 1e-9) {
        return fit;
    }

    fit.linearCoeffUm = (sumXY * sumX4 - sumX2Y * sumX3) / det;
    fit.quadraticCoeffUm = (sumX2 * sumX2Y - sumX3 * sumXY) / det;

    double sse = 0.0;
    for (const DesignErrorProfilePoint* point : usedPoints) {
        const double x = std::clamp((point->sAlignedMm - sMin) / (sMax - sMin), 0.0, 1.0);
        const double correctionUm = fit.linearCoeffUm * x + fit.quadraticCoeffUm * x * x;
        const double correctedUm = point->normalErrorUm - correctionUm;
        sse += correctedUm * correctedUm;
    }
    fit.candidateRmseUm = std::sqrt(sse / static_cast<double>(usedPoints.size()));
    fit.ok = true;
    return fit;
}

void applyLowFrequencyNormalDriftCorrection(EvaluationResult& evaluation)
{
    if (!evaluation.ok) {
        return;
    }

    const AnchoredQuadraticCorrectionFit fit =
        fitAnchoredQuadraticNormalDriftCorrection(evaluation.profilePoints);
    if (!fit.ok) {
        return;
    }

    constexpr double kMinMeaningfulImprovementUm = 0.10;
    constexpr double kMaxTailCorrectionUm = 120.0;
    const double baselineRmseUm = evaluation.summary.normalStats.rmseUm;
    const double tailCorrectionUm = fit.linearCoeffUm + fit.quadraticCoeffUm;
    if (!std::isfinite(baselineRmseUm) ||
        !std::isfinite(fit.candidateRmseUm) ||
        fit.candidateRmseUm + kMinMeaningfulImprovementUm >= baselineRmseUm ||
        std::abs(tailCorrectionUm) > kMaxTailCorrectionUm) {
        return;
    }

    double sMin = std::numeric_limits<double>::infinity();
    double sMax = -std::numeric_limits<double>::infinity();
    for (const DesignErrorProfilePoint& point : evaluation.profilePoints) {
        if (!point.isUsed ||
            !std::isfinite(point.sAlignedMm) ||
            !std::isfinite(point.normalErrorUm)) {
            continue;
        }
        sMin = std::min(sMin, point.sAlignedMm);
        sMax = std::max(sMax, point.sAlignedMm);
    }
    if (!std::isfinite(sMin) || !std::isfinite(sMax) || sMax <= sMin + 1e-9) {
        return;
    }

    std::vector<double> correctedAll;
    std::vector<double> correctedUsed;
    correctedAll.reserve(evaluation.profilePoints.size());
    correctedUsed.reserve(evaluation.summary.usedCount);

    for (DesignErrorProfilePoint& point : evaluation.profilePoints) {
        if (!std::isfinite(point.normalErrorUm) || !std::isfinite(point.sAlignedMm)) {
            continue;
        }
        const double x = std::clamp((point.sAlignedMm - sMin) / (sMax - sMin), 0.0, 1.0);
        const double correctionUm = fit.linearCoeffUm * x + fit.quadraticCoeffUm * x * x;
        point.normalErrorUm -= correctionUm;
        point.normalErrorMm = point.normalErrorUm / 1000.0;
        correctedAll.push_back(point.normalErrorUm);
        if (point.isUsed) {
            correctedUsed.push_back(point.normalErrorUm);
        }
    }

    if (correctedUsed.size() < kMinUsedPointCount) {
        return;
    }

    const ErrorStats correctedAllStats = buildErrorStats(correctedAll);
    const ErrorStats correctedUsedStats = buildErrorStats(correctedUsed);
    std::vector<double> correctedProfileErrors;
    correctedProfileErrors.reserve(correctedUsed.size());
    for (DesignErrorProfilePoint& point : evaluation.profilePoints) {
        if (!point.isUsed || !std::isfinite(point.normalErrorUm)) {
            continue;
        }
        point.profileErrorUm = point.normalErrorUm - correctedUsedStats.meanUm;
        correctedProfileErrors.push_back(point.profileErrorUm);
    }
    const ErrorStats correctedProfileStats = buildErrorStats(correctedProfileErrors);

    evaluation.summary.meanNormalErrorUm = correctedUsedStats.meanUm;
    evaluation.summary.absoluteAllStats = correctedAllStats;
    evaluation.summary.absoluteFilteredStats = correctedUsedStats;
    evaluation.summary.normalStats = correctedUsedStats;
    evaluation.summary.profileStats = correctedProfileStats;
    evaluation.objective =
        evaluation.summary.evaluateProfileForm ? correctedProfileStats.rmseUm : correctedUsedStats.rmseUm;
}

EvaluationResult runBestFitIfEnabled(const MeasuredGeneratrixData& data,
                                     const stitch::StitchPipelineConfig& config,
                                     const double anchorDrMm)
{
    const MeasuredGeneratrixData searchData = buildBestFitSearchData(data);
    const auto evaluateAt = [&](const double dzMm, const double drMm) {
        return evaluateAlignedProfile(searchData, config, dzMm, drMm, false);
    };

    const double fineStepMm = std::max(1e-6, config.designBestFitStepMm);
    const double coarseStepMm = fineStepMm * 8.0;

    // 平移搜索 lambda
    const auto searchTranslation = [&](double& bestDzMm, double& bestDrMm,
                                        double& bestObjective, EvaluationResult& best,
                                        const bool useCoarseFine) {
        if (useCoarseFine) {
            // --- 粗搜索 ---
            for (double dzMm = config.designBestFitDzMinMm;
                 dzMm <= config.designBestFitDzMaxMm + 1e-12; dzMm += coarseStepMm) {
                for (double deltaDrMm = config.designBestFitDrMinMm;
                     deltaDrMm <= config.designBestFitDrMaxMm + 1e-12; deltaDrMm += coarseStepMm) {
                    const double drMm = anchorDrMm + deltaDrMm;
                    const EvaluationResult current = evaluateAt(dzMm, drMm);
                    if (!current.ok) continue;
                    if (current.objective < bestObjective) {
                        bestObjective = current.objective;
                        bestDzMm = dzMm; bestDrMm = drMm;
                        best = current;
                    }
                }
            }

            // --- 精搜索 ---
            const double fineHalfWindow = coarseStepMm * 1.2;
            const double dzFineMin = std::max(config.designBestFitDzMinMm, bestDzMm - fineHalfWindow);
            const double dzFineMax = std::min(config.designBestFitDzMaxMm, bestDzMm + fineHalfWindow);
            const double drFineMin = std::max(anchorDrMm + config.designBestFitDrMinMm,
                                               bestDrMm - fineHalfWindow);
            const double drFineMax = std::min(anchorDrMm + config.designBestFitDrMaxMm,
                                               bestDrMm + fineHalfWindow);

            for (double dzMm = dzFineMin; dzMm <= dzFineMax + 1e-12; dzMm += fineStepMm) {
                for (double drMm = drFineMin; drMm <= drFineMax + 1e-12; drMm += fineStepMm) {
                    const EvaluationResult current = evaluateAt(dzMm, drMm);
                    if (!current.ok) continue;
                    if (current.objective < bestObjective) {
                        bestObjective = current.objective;
                        bestDzMm = dzMm; bestDrMm = drMm;
                        best = current;
                    }
                }
            }

            // --- 亚网格精化：2D 抛物线插值 ---
            {
                const double hz = fineStepMm;
                const double hr = fineStepMm;
                double c[3][3];
                for (int iz = -1; iz <= 1; ++iz) {
                    for (int ir = -1; ir <= 1; ++ir) {
                        const EvaluationResult cur =
                            evaluateAt(bestDzMm + iz * hz, bestDrMm + ir * hr);
                        c[iz + 1][ir + 1] = cur.ok ? cur.objective : 1e9;
                    }
                }

                const auto parabolic1D = [](double ym1, double y0, double yp1, double step) -> double {
                    const double denom = 2.0 * (ym1 + yp1 - 2.0 * y0);
                    if (denom <= 1e-12) return 0.0;
                    return std::clamp(step * (ym1 - yp1) / denom, -step, step);
                };

                const double deltaDz0 = parabolic1D(c[0][1], c[1][1], c[2][1], hz);
                const double deltaDr0 = parabolic1D(c[1][0], c[1][1], c[1][2], hr);

                if (std::abs(deltaDz0) > 1e-9 || std::abs(deltaDr0) > 1e-9) {
                    const EvaluationResult refined =
                        evaluateAt(bestDzMm + deltaDz0, bestDrMm + deltaDr0);
                    if (refined.ok && refined.objective < bestObjective) {
                        bestObjective = refined.objective;
                        bestDzMm = bestDzMm + deltaDz0;
                        bestDrMm = bestDrMm + deltaDr0;
                        best = refined;
                    }
                }
            }
        }
    };

    // --- 无旋转时的平移搜索 ---
    EvaluationResult best = evaluateAt(0.0, anchorDrMm);
    if (!config.designEnableBestFitTranslation) {
        return evaluateAlignedProfile(data, config, 0.0, anchorDrMm, true);
    }

    double bestDzMm = 0.0;
    double bestDrMm = anchorDrMm;
    double bestObjective = best.ok ? best.objective : std::numeric_limits<double>::infinity();
    double bestThetaDeg = 0.0;
    double rotationCenterSMm = 0.0;
    double rotationCenterRMm = 0.0;

    searchTranslation(bestDzMm, bestDrMm, bestObjective, best, true);

    // --- 旋转搜索 ---
    double bestThetaForSummary = 0.0;
    if (config.designEnableBestFitRotation) {
        // 计算母线质心（旋转中心）
        std::size_t rotCount = 0;
        for (const GeneratrixCandidatePoint& pt : data.points) {
            if (!pt.trimAccepted) continue;
            rotationCenterSMm += pt.sBaseMm;
            rotationCenterRMm += pt.rRawMm;
            ++rotCount;
        }
        if (rotCount > 0) {
            rotationCenterSMm /= static_cast<double>(rotCount);
            rotationCenterRMm /= static_cast<double>(rotCount);
        }

        const double dThetaStep = std::max(0.005, config.designBestFitRotationStepDeg);

        for (double dThetaDeg = config.designBestFitRotationMinDeg;
             dThetaDeg <= config.designBestFitRotationMaxDeg + 1e-12; dThetaDeg += dThetaStep) {
            if (std::abs(dThetaDeg) < 1e-9) continue; // 已包含在 dTheta=0 中

            // 旋转后重新评估
            MeasuredGeneratrixData rotatedData =
                rotateMeasuredData(searchData, rotationCenterSMm, rotationCenterRMm, dThetaDeg);

            double dzLocal = 0.0;
            double drLocal = anchorDrMm;
            double objLocal = std::numeric_limits<double>::infinity();
            EvaluationResult bestLocal;

            const auto evaluateRotated = [&](double dzMm, double drMm) {
                return evaluateAlignedProfile(rotatedData, config, dzMm, drMm, false);
            };

            // 宽窗口精搜索（±1.5 mm dz, ±0.2 mm dr）
            const double rotDzHalfWindow = 1.5;
            const double rotDrHalfWindow = 0.2;
            const double rotDzMin = std::max(config.designBestFitDzMinMm, -rotDzHalfWindow);
            const double rotDzMax = std::min(config.designBestFitDzMaxMm, rotDzHalfWindow);
            const double rotDrMin = std::max(anchorDrMm + config.designBestFitDrMinMm,
                                              anchorDrMm - rotDrHalfWindow);
            const double rotDrMax = std::min(anchorDrMm + config.designBestFitDrMaxMm,
                                              anchorDrMm + rotDrHalfWindow);

            const double rotStep = coarseStepMm; // 0.04 mm for speed

            // 粗搜索
            for (double dzMm = rotDzMin; dzMm <= rotDzMax + 1e-12; dzMm += rotStep) {
                for (double drMm = rotDrMin; drMm <= rotDrMax + 1e-12; drMm += rotStep) {
                    const EvaluationResult current = evaluateRotated(dzMm, drMm);
                    if (!current.ok) continue;
                    if (current.objective < objLocal) {
                        objLocal = current.objective;
                        dzLocal = dzMm; drLocal = drMm;
                        bestLocal = current;
                    }
                }
            }

            // 精细搜索
            const double rotFineHalf = rotStep * 1.5;
            for (double dzMm = std::max(rotDzMin, dzLocal - rotFineHalf);
                 dzMm <= std::min(rotDzMax, dzLocal + rotFineHalf) + 1e-12; dzMm += fineStepMm) {
                for (double drMm = std::max(rotDrMin, drLocal - rotFineHalf);
                     drMm <= std::min(rotDrMax, drLocal + rotFineHalf) + 1e-12; drMm += fineStepMm) {
                    const EvaluationResult current = evaluateRotated(dzMm, drMm);
                    if (!current.ok) continue;
                    if (current.objective < objLocal) {
                        objLocal = current.objective;
                        dzLocal = dzMm; drLocal = drMm;
                        bestLocal = current;
                    }
                }
            }

            if (objLocal < bestObjective) {
                bestObjective = objLocal;
                best = bestLocal;
                bestDzMm = dzLocal;
                bestDrMm = drLocal;
                bestThetaDeg = dThetaDeg;
                bestThetaForSummary = dThetaDeg;
            }
        }
    }

    MeasuredGeneratrixData finalData = data;
    if (std::abs(bestThetaForSummary) > 1e-12) {
        finalData = rotateMeasuredData(finalData, rotationCenterSMm, rotationCenterRMm, bestThetaForSummary);
    }
    EvaluationResult finalEvaluation = evaluateAlignedProfile(finalData, config, bestDzMm, bestDrMm, true);
    if (!finalEvaluation.ok) {
        best.summary.dThetaDeg = bestThetaForSummary;
        return best;
    }

    finalEvaluation = refineAxialScaleAlignment(finalData, config, finalEvaluation);
    finalEvaluation.summary.dThetaDeg = bestThetaForSummary;
    finalEvaluation = refineAxialQuadraticAlignment(finalData, config, finalEvaluation);
    finalEvaluation.summary.dThetaDeg = bestThetaForSummary;
    EvaluationResult refinedEvaluation =
        refineAbsoluteBiasAlignment(finalData, config, anchorDrMm, finalEvaluation);
    refinedEvaluation.summary.dThetaDeg = bestThetaForSummary;
    refinedEvaluation.summary.preRefineMeanNormalErrorUm = finalEvaluation.summary.meanNormalErrorUm;
    refinedEvaluation.summary.preRefineAbsoluteFilteredRmseUm =
        finalEvaluation.summary.absoluteFilteredStats.rmseUm;
    refinedEvaluation.summary.absoluteBiasCorrectionUm =
        (refinedEvaluation.summary.drMm - finalEvaluation.summary.drMm) * 1000.0;
    refinedEvaluation.summary.appliedAbsoluteBiasRefine =
        std::abs(refinedEvaluation.summary.absoluteBiasCorrectionUm) > 1e-6;
    applyLowFrequencyNormalDriftCorrection(refinedEvaluation);
    return refinedEvaluation;
}

std::string buildProfileCsv(const std::vector<DesignErrorProfilePoint>& points)
{
    std::ostringstream stream;
    stream << "index,x_px,y_px,y_std_px,s_aligned_mm,r_aligned_mm,r_design_mm,nearest_design_s_mm,"
              "nearest_design_r_mm,design_segment_index,dr_design_ds,radial_error_mm,radial_error_um,"
              "signed_distance_mm,signed_distance_um,legacy_normal_error_um,normal_error_mm,normal_error_um,"
              "profile_error_um,is_used\n";

    for (const DesignErrorProfilePoint& point : points) {
        stream << point.index << ","
               << csvCell(point.xPx, 6) << ","
               << csvCell(point.yPx, 6) << ","
               << csvCell(point.yStdPx, 6) << ","
               << csvCell(point.sAlignedMm, 6) << ","
               << csvCell(point.rAlignedMm, 6) << ","
               << csvCell(point.designRadiusMm, 6) << ","
               << csvCell(point.nearestDesignSMm, 6) << ","
               << csvCell(point.nearestDesignRMm, 6) << ","
               << point.designSegmentIndex << ","
               << csvCell(point.designDerivative, 6) << ","
               << csvCell(point.radialErrorMm, 6) << ","
               << csvCell(point.radialErrorUm, 6) << ","
               << csvCell(point.signedDistanceMm, 6) << ","
               << csvCell(point.signedDistanceUm, 6) << ","
               << csvCell(point.legacyNormalErrorUm, 6) << ","
               << csvCell(point.normalErrorMm, 6) << ","
               << csvCell(point.normalErrorUm, 6) << ","
               << csvCell(point.profileErrorUm, 6) << ","
               << (point.isUsed ? 1 : 0) << "\n";
    }

    return stream.str();
}

std::string buildSummaryCsv(const DesignErrorSummary& summary)
{
    std::ostringstream stream;
    stream << "dz_mm,dr_mm,dtheta_deg,axial_scale_factor,axial_quadratic_term_mm,applied_absolute_bias_refine,absolute_bias_correction_um,"
              "pre_refine_mean_normal_error_um,pre_refine_absolute_filtered_rmse_um,"
              "design_reverse_z,use_left_endpoint_anchor,design_evaluate_profile_form,anchor_x_px,anchor_y_px,"
              "pixel_size_mm,candidate_count,used_count,outlier_count,outlier_ratio,mean_normal_error_um,normal_rmse_um,normal_mae_um,normal_p95_abs_um,"
              "normal_max_pos_um,normal_max_neg_um,normal_pv_um,profile_rms_um,profile_mae_um,profile_p95_abs_um,"
              "profile_max_pos_um,profile_max_neg_um,profile_pv_um,absolute_all_mean_um,absolute_all_rmse_um,"
              "absolute_all_mae_um,absolute_all_p95_abs_um,absolute_all_max_pos_um,absolute_all_max_neg_um,"
              "absolute_all_pv_um,absolute_filtered_mean_um,absolute_filtered_rmse_um,absolute_filtered_mae_um,"
              "absolute_filtered_p95_abs_um,absolute_filtered_max_pos_um,absolute_filtered_max_neg_um,"
              "absolute_filtered_pv_um\n";
    stream << csvCell(summary.dzMm, 6) << ","
           << csvCell(summary.drMm, 6) << ","
           << csvCell(summary.dThetaDeg, 6) << ","
           << csvCell(summary.axialScaleFactor, 6) << ","
           << csvCell(summary.axialQuadraticTermMm, 6) << ","
           << (summary.appliedAbsoluteBiasRefine ? 1 : 0) << ","
           << csvCell(summary.absoluteBiasCorrectionUm, 6) << ","
           << csvCell(summary.preRefineMeanNormalErrorUm, 6) << ","
           << csvCell(summary.preRefineAbsoluteFilteredRmseUm, 6) << ","
           << (summary.designReverseZ ? 1 : 0) << ","
           << (summary.useLeftEndpointAnchor ? 1 : 0) << ","
           << (summary.evaluateProfileForm ? 1 : 0) << ","
           << csvCell(summary.anchorXPx, 6) << ","
           << csvCell(summary.anchorYPx, 6) << ","
           << csvCell(summary.pixelSizeMm, 6) << ","
           << summary.candidateCount << ","
           << summary.usedCount << ","
           << summary.outlierCount << ","
           << csvCell(summary.outlierRatio, 6) << ","
           << csvCell(summary.meanNormalErrorUm, 6) << ","
           << csvCell(summary.normalStats.rmseUm, 6) << ","
           << csvCell(summary.normalStats.maeUm, 6) << ","
           << csvCell(summary.normalStats.p95AbsUm, 6) << ","
           << csvCell(summary.normalStats.maxPosUm, 6) << ","
           << csvCell(summary.normalStats.maxNegUm, 6) << ","
           << csvCell(summary.normalStats.pvUm, 6) << ","
           << csvCell(summary.profileStats.rmseUm, 6) << ","
           << csvCell(summary.profileStats.maeUm, 6) << ","
           << csvCell(summary.profileStats.p95AbsUm, 6) << ","
           << csvCell(summary.profileStats.maxPosUm, 6) << ","
           << csvCell(summary.profileStats.maxNegUm, 6) << ","
           << csvCell(summary.profileStats.pvUm, 6) << ","
           << csvCell(summary.absoluteAllStats.meanUm, 6) << ","
           << csvCell(summary.absoluteAllStats.rmseUm, 6) << ","
           << csvCell(summary.absoluteAllStats.maeUm, 6) << ","
           << csvCell(summary.absoluteAllStats.p95AbsUm, 6) << ","
           << csvCell(summary.absoluteAllStats.maxPosUm, 6) << ","
           << csvCell(summary.absoluteAllStats.maxNegUm, 6) << ","
           << csvCell(summary.absoluteAllStats.pvUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.meanUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.rmseUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.maeUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.p95AbsUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.maxPosUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.maxNegUm, 6) << ","
           << csvCell(summary.absoluteFilteredStats.pvUm, 6) << "\n";
    return stream.str();
}

cv::Point mapPointToRect(const cv::Rect& rect,
                         const double x,
                         const double y,
                         const double xMin,
                         const double xMax,
                         const double yMin,
                         const double yMax)
{
    const double xRange = std::max(1e-9, xMax - xMin);
    const double yRange = std::max(1e-9, yMax - yMin);
    const double xNorm = (x - xMin) / xRange;
    const double yNorm = (y - yMin) / yRange;
    const int px = rect.x + static_cast<int>(std::round(xNorm * rect.width));
    const int py = rect.y + rect.height - static_cast<int>(std::round(yNorm * rect.height));
    return {px, py};
}

void drawPolyline(cv::Mat& canvas,
                  const cv::Rect& rect,
                  const std::vector<cv::Point2d>& series,
                  const cv::Scalar& color,
                  const double xMin,
                  const double xMax,
                  const double yMin,
                  const double yMax,
                  const int thickness)
{
    if (series.size() < 2) {
        return;
    }

    std::vector<cv::Point> points;
    points.reserve(series.size());
    for (const cv::Point2d& point : series) {
        points.push_back(mapPointToRect(rect, point.x, point.y, xMin, xMax, yMin, yMax));
    }

    for (std::size_t index = 1; index < points.size(); ++index) {
        cv::line(canvas, points[index - 1], points[index], color, thickness, cv::LINE_AA);
    }
}

void drawScatter(cv::Mat& canvas,
                 const cv::Rect& rect,
                 const std::vector<cv::Point2d>& series,
                 const cv::Scalar& color,
                 const double xMin,
                 const double xMax,
                 const double yMin,
                 const double yMax,
                 const int radius)
{
    for (const cv::Point2d& point : series) {
        cv::circle(canvas, mapPointToRect(rect, point.x, point.y, xMin, xMax, yMin, yMax), radius, color, -1,
                   cv::LINE_AA);
    }
}

void drawInfoBox(cv::Mat& canvas,
                 const std::string& text,
                 const cv::Point& anchor,
                 const double fontScale,
                 const cv::Scalar& textColor,
                 const cv::Scalar& fillColor = cv::Scalar(252, 252, 252),
                 const cv::Scalar& borderColor = cv::Scalar(220, 220, 220),
                 const int thickness = 1)
{
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    const int padX = 10;
    const int padY = 8;
    cv::Rect box(anchor.x,
                 anchor.y - textSize.height - padY,
                 textSize.width + padX * 2,
                 textSize.height + padY * 2 + baseline);
    box &= cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (box.width <= 0 || box.height <= 0) {
        return;
    }

    cv::rectangle(canvas, box, fillColor, cv::FILLED, cv::LINE_AA);
    cv::rectangle(canvas, box, borderColor, 1, cv::LINE_AA);
    cv::putText(canvas,
                text,
                cv::Point(box.x + padX, box.y + padY + textSize.height),
                cv::FONT_HERSHEY_SIMPLEX,
                fontScale,
                textColor,
                thickness,
                cv::LINE_AA);
}

void drawPlotGrid(cv::Mat& canvas,
                  const cv::Rect& rect,
                  const double xMin,
                  const double xMax,
                  const double yMin,
                  const double yMax,
                  const int xTicks,
                  const int yTicks)
{
    cv::rectangle(canvas, rect, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    for (int tick = 0; tick <= yTicks; ++tick) {
        const double ratio = static_cast<double>(tick) / static_cast<double>(yTicks);
        const int y = rect.y + rect.height - static_cast<int>(std::lround(ratio * rect.height));
        cv::line(canvas, cv::Point(rect.x, y), cv::Point(rect.x + rect.width, y),
                 cv::Scalar(238, 238, 238), 1, cv::LINE_AA);

        std::ostringstream label;
        label << std::fixed << std::setprecision(2) << (yMin + ratio * (yMax - yMin));
        cv::putText(canvas, label.str(), cv::Point(rect.x - 68, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(92, 92, 92), 1, cv::LINE_AA);
    }

    for (int tick = 0; tick <= xTicks; ++tick) {
        const double ratio = static_cast<double>(tick) / static_cast<double>(xTicks);
        const int x = rect.x + static_cast<int>(std::lround(ratio * rect.width));
        cv::line(canvas, cv::Point(x, rect.y), cv::Point(x, rect.y + rect.height),
                 cv::Scalar(244, 244, 244), 1, cv::LINE_AA);

        std::ostringstream label;
        label << std::fixed << std::setprecision(1) << (xMin + ratio * (xMax - xMin));
        cv::putText(canvas, label.str(), cv::Point(x - 18, rect.y + rect.height + 24), cv::FONT_HERSHEY_SIMPLEX,
                    0.44, cv::Scalar(92, 92, 92), 1, cv::LINE_AA);
    }
}

} // namespace

DesignAlignmentResult compareMeasuredProfileToDesign(const std::vector<stitch::EdgeVariants>& edges,
                                                     const std::vector<cv::Mat>& imageTransforms,
                                                     const stitch::StitchPipelineConfig& config)
{
    DesignAlignmentResult result;
    if (!config.enableDesignComparison) {
        result.message = "design profile comparison disabled by configuration.";
        return result;
    }

    std::vector<StitchedContourSample> samples;
    if (!collectStitchedContourSamples(edges, imageTransforms, samples)) {
        result.message = "design profile alignment failed: stitched contour profile is empty.";
        return result;
    }

    const MeasuredGeneratrixData generatrix = extractMeasuredGeneratrixForDesignComparison(samples, config);
    if (generatrix.points.empty()) {
        result.message = "design profile alignment failed: no valid measured generatrix after envelope extraction.";
        return result;
    }

    const std::size_t trimmedCount = std::count_if(generatrix.points.begin(), generatrix.points.end(), [](const auto& point) {
        return point.trimAccepted;
    });
    if (trimmedCount < kMinUsedPointCount) {
        result.message =
            "design profile alignment failed: too few measured generatrix points after end-face filtering.";
        return result;
    }

    const DesignEval leftDesign = evalDesignRadiusCompare(0.0, config.designReverseZ);
    if (!leftDesign.valid) {
        result.message = "design profile alignment failed: design left endpoint is outside valid domain.";
        return result;
    }

    double anchorDrMm =
        config.designAnchorRadialToLeftEndpoint ? (leftDesign.r_mm - generatrix.anchorMeasuredRadiusMm) : 0.0;
    anchorDrMm = refineAnchorDrByStableWindow(generatrix, config, anchorDrMm);
    MeasuredGeneratrixData selectedGeneratrix = applyAcceptedRuns(generatrix, config, anchorDrMm);
    selectedGeneratrix = trimResidualLeftEndpointTransition(selectedGeneratrix, config, anchorDrMm);
    const EvaluationResult evaluation = runBestFitIfEnabled(selectedGeneratrix, config, anchorDrMm);
    if (!evaluation.ok) {
        result.message =
            "design profile alignment failed: too few valid overlap samples after anchoring and domain filtering.";
        return result;
    }

    result.ok = true;
    result.summary = evaluation.summary;
    result.profilePoints = evaluation.profilePoints;
    result.profileCsvText = buildProfileCsv(result.profilePoints);
    result.summaryCsvText = buildSummaryCsv(result.summary);
    result.message = "design profile alignment completed.";
    return result;
}

cv::Mat buildDesignComparisonPlot(const DesignAlignmentResult& result)
{
    const int width = 1440;
    const int height = 920;
    const cv::Rect profileRect(104, 100, width - 180, 420);
    const cv::Rect errorRect(104, 586, width - 180, 214);
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    drawInfoBox(canvas, "Measured-design generatrix comparison", cv::Point(34, 34), 0.66, cv::Scalar(38, 38, 38));

    std::vector<cv::Point2d> usedMeasuredSeries;
    std::vector<cv::Point2d> excludedMeasuredSeries;
    std::vector<cv::Point2d> designSeries;
    std::vector<cv::Point2d> profileErrorSeries;
    std::vector<cv::Point2d> normalErrorSeries;
    usedMeasuredSeries.reserve(result.profilePoints.size());
    excludedMeasuredSeries.reserve(result.profilePoints.size());
    profileErrorSeries.reserve(result.profilePoints.size());
    normalErrorSeries.reserve(result.profilePoints.size());

    double minS = std::numeric_limits<double>::infinity();
    double maxS = -std::numeric_limits<double>::infinity();
    double minRadius = std::numeric_limits<double>::infinity();
    double maxRadius = -std::numeric_limits<double>::infinity();
    double maxAbsErrorUm = 1.0;

    for (const DesignErrorProfilePoint& point : result.profilePoints) {
        const cv::Point2d measuredPoint(point.sAlignedMm, point.rAlignedMm);
        if (point.isUsed) {
            usedMeasuredSeries.push_back(measuredPoint);
            if (std::isfinite(point.designRadiusMm)) {
                if (std::isfinite(point.profileErrorUm)) {
                    profileErrorSeries.emplace_back(point.sAlignedMm, point.profileErrorUm);
                    maxAbsErrorUm = std::max(maxAbsErrorUm, std::abs(point.profileErrorUm));
                }
                if (std::isfinite(point.normalErrorUm)) {
                    normalErrorSeries.emplace_back(point.sAlignedMm, point.normalErrorUm);
                    maxAbsErrorUm = std::max(maxAbsErrorUm, std::abs(point.normalErrorUm));
                }
                minRadius = std::min(minRadius, std::min(point.rAlignedMm, point.designRadiusMm));
                maxRadius = std::max(maxRadius, std::max(point.rAlignedMm, point.designRadiusMm));
            }
        } else {
            excludedMeasuredSeries.push_back(measuredPoint);
            minRadius = std::min(minRadius, point.rAlignedMm);
            maxRadius = std::max(maxRadius, point.rAlignedMm);
        }
        minS = std::min(minS, point.sAlignedMm);
        maxS = std::max(maxS, point.sAlignedMm);
    }

    if (usedMeasuredSeries.empty()) {
        cv::putText(canvas,
                    "No valid design comparison samples",
                    cv::Point(profileRect.x + 40, profileRect.y + 80),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(90, 90, 90),
                    2,
                    cv::LINE_AA);
        return canvas;
    }

    const double sBegin = std::max(0.0, minS);
    const double sEnd = std::max(sBegin, maxS);
    const std::vector<DesignPolylinePoint> designPolyline = buildDesignPolyline(result.summary.designReverseZ);
    for (const DesignPolylinePoint& design : designPolyline) {
        if (design.sMm < sBegin - 1e-9 || design.sMm > sEnd + 1e-9) {
            continue;
        }
        designSeries.emplace_back(design.sMm, design.rMm);
        minRadius = std::min(minRadius, design.rMm);
        maxRadius = std::max(maxRadius, design.rMm);
    }

    if (!std::isfinite(minRadius) || !std::isfinite(maxRadius)) {
        minRadius = -1.0;
        maxRadius = 1.0;
    }
    minRadius -= 1.0;
    maxRadius += 1.0;

    const cv::Scalar usedColor(44, 111, 220);
    const cv::Scalar excludedColor(190, 190, 190);
    const cv::Scalar designColor(36, 157, 81);
    const cv::Scalar profileErrorColor(30, 158, 138);
    const cv::Scalar normalErrorColor(199, 67, 67);
    const cv::Scalar zeroColor(160, 160, 160);
    const cv::Scalar anchorColor(55, 75, 220);

    drawPlotGrid(canvas, profileRect, minS, maxS, minRadius, maxRadius, 5, 5);
    drawPlotGrid(canvas, errorRect, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm, 5, 4);

    drawScatter(canvas, profileRect, excludedMeasuredSeries, excludedColor, minS, maxS, minRadius, maxRadius, 3);
    drawPolyline(canvas, profileRect, usedMeasuredSeries, usedColor, minS, maxS, minRadius, maxRadius, 2);
    drawPolyline(canvas, profileRect, designSeries, designColor, minS, maxS, minRadius, maxRadius, 2);
    drawPolyline(canvas, errorRect, profileErrorSeries, profileErrorColor, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm, 2);
    drawPolyline(canvas, errorRect, normalErrorSeries, normalErrorColor, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm, 1);

    const cv::Point zeroBegin = mapPointToRect(errorRect, minS, 0.0, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
    const cv::Point zeroEnd = mapPointToRect(errorRect, maxS, 0.0, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
    cv::line(canvas, zeroBegin, zeroEnd, zeroColor, 1, cv::LINE_AA);

    if (result.summary.profileStats.p95AbsUm > 0.0) {
        const double p95 = result.summary.profileStats.p95AbsUm;
        const cv::Point p95BeginPos = mapPointToRect(errorRect, minS, p95, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
        const cv::Point p95EndPos = mapPointToRect(errorRect, maxS, p95, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
        const cv::Point p95BeginNeg = mapPointToRect(errorRect, minS, -p95, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
        const cv::Point p95EndNeg = mapPointToRect(errorRect, maxS, -p95, minS, maxS, -maxAbsErrorUm, maxAbsErrorUm);
        cv::line(canvas, p95BeginPos, p95EndPos, cv::Scalar(208, 208, 208), 1, cv::LINE_AA);
        cv::line(canvas, p95BeginNeg, p95EndNeg, cv::Scalar(208, 208, 208), 1, cv::LINE_AA);
    }

    const auto anchorIt = std::find_if(result.profilePoints.begin(), result.profilePoints.end(), [&result](const auto& point) {
        return std::abs(point.xPx - result.summary.anchorXPx) < 1e-6 &&
               std::abs(point.yPx - result.summary.anchorYPx) < 1e-6;
    });
    if (anchorIt != result.profilePoints.end()) {
        const cv::Point anchorPoint =
            mapPointToRect(profileRect, anchorIt->sAlignedMm, anchorIt->rAlignedMm, minS, maxS, minRadius, maxRadius);
        cv::drawMarker(canvas, anchorPoint, anchorColor, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    }

    cv::putText(canvas, "Radius (mm)", cv::Point(20, profileRect.y - 16), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    cv::putText(canvas, "Error (um)", cv::Point(20, errorRect.y - 16), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    cv::putText(canvas, "Comparison coordinate s (mm)",
                cv::Point(profileRect.x + profileRect.width / 2 - 110, height - 28),
                cv::FONT_HERSHEY_SIMPLEX,
                0.64,
                cv::Scalar(90, 90, 90),
                1,
                cv::LINE_AA);

    const int legendX = width - 338;
    const int legendY = 62;
    cv::line(canvas, cv::Point(legendX, legendY), cv::Point(legendX + 34, legendY), usedColor, 3, cv::LINE_AA);
    cv::putText(canvas, "Measured contour", cv::Point(legendX + 46, legendY + 5), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(legendX, legendY + 24), cv::Point(legendX + 34, legendY + 24), designColor, 3, cv::LINE_AA);
    cv::putText(canvas, "Design contour", cv::Point(legendX + 46, legendY + 29), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(legendX, legendY + 48), cv::Point(legendX + 34, legendY + 48), profileErrorColor, 3, cv::LINE_AA);
    cv::putText(canvas, "Profile-form error", cv::Point(legendX + 46, legendY + 53), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(legendX, legendY + 72), cv::Point(legendX + 34, legendY + 72), normalErrorColor, 2, cv::LINE_AA);
    cv::putText(canvas, "Normal error", cv::Point(legendX + 46, legendY + 77), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::circle(canvas, cv::Point(legendX + 17, legendY + 96), 4, excludedColor, -1, cv::LINE_AA);
    cv::putText(canvas, "Excluded samples", cv::Point(legendX + 46, legendY + 101), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::drawMarker(canvas, cv::Point(legendX + 17, legendY + 120), anchorColor, cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
    cv::putText(canvas, "Anchor sample", cv::Point(legendX + 46, legendY + 125), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

    std::ostringstream info;
    info.setf(std::ios::fixed);
    info.precision(3);
    info << "reverse_z=" << (result.summary.designReverseZ ? 1 : 0)
         << "  dz=" << result.summary.dzMm << " mm"
         << "  dr=" << result.summary.drMm << " mm"
         << "  abs_rmse=" << result.summary.absoluteFilteredStats.rmseUm << " um"
         << "  offset=" << result.summary.meanNormalErrorUm << " um"
         << "  profile_rms=" << result.summary.profileStats.rmseUm << " um"
         << "  profile_p95=" << result.summary.profileStats.p95AbsUm << " um"
         << "  N=" << result.summary.usedCount;
    drawInfoBox(canvas, info.str(), cv::Point(34, 844), 0.48, cv::Scalar(60, 60, 60));

    if (result.summary.appliedAbsoluteBiasRefine) {
        std::ostringstream refineInfo;
        refineInfo.setf(std::ios::fixed);
        refineInfo.precision(3);
        refineInfo << "bias refine " << result.summary.preRefineMeanNormalErrorUm
                   << " -> " << result.summary.meanNormalErrorUm
                   << " um  (dr correction " << result.summary.absoluteBiasCorrectionUm << " um)";
        drawInfoBox(canvas, refineInfo.str(), cv::Point(34, 874), 0.45, cv::Scalar(86, 86, 86));
    }

    return canvas;
}

} // namespace pinjie::cad_design
