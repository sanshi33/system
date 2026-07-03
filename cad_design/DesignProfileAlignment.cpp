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
constexpr std::size_t kMinSingleSlotTargetPointCount = 20;
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
constexpr std::size_t kFeatureCompensationCsvColumnCount = 28;

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
    bool hasCadPoint{false};
    double cadXMm{0.0};
    double cadYMm{0.0};
    double cadZMm{0.0};
};

struct NearestDesignPoint {
    bool valid{false};
    std::size_t segmentIndex{0};
    double sMm{0.0};
    double rMm{0.0};
    bool hasCadPoint{false};
    double cadXMm{0.0};
    double cadYMm{0.0};
    double cadZMm{0.0};
    double signedDistanceMm{0.0};
    double tangentDs{1.0};
    double tangentDr{0.0};
};

struct CadGuidedSlotCandidate {
    GeneratrixCandidatePoint point;
    double sMm{0.0};
    double rMm{0.0};
    double scoreMm{std::numeric_limits<double>::infinity()};
};

struct CadPoint {
    bool valid{false};
    double xMm{0.0};
    double yMm{0.0};
    double zMm{0.0};
};

struct SlotFeatureEstimate {
    bool valid{false};
    std::string method;
    double leftSMm{0.0};
    double rightSMm{0.0};
    double centerSMm{0.0};
    double widthMm{0.0};
    double depthMm{0.0};
    double topRMm{0.0};
    double bottomRMm{0.0};
    double bottomSMm{0.0};
};

std::size_t minUsedPointCount(const stitch::StitchPipelineConfig& config)
{
    return config.designTargetSlotWidthMm > 1e-9 ? kMinSingleSlotTargetPointCount : kMinUsedPointCount;
}

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

std::string csvTextCell(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

void appendCsvRow(std::ostringstream& stream, const std::vector<std::string>& cells)
{
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (index != 0) {
            stream << ",";
        }
        stream << cells[index];
    }
    stream << "\n";
}

void appendFeatureCompensationRow(std::ostringstream& stream, std::vector<std::string> cells)
{
    cells.resize(kFeatureCompensationCsvColumnCount);
    appendCsvRow(stream, cells);
}

std::string cadCoordinateCell(const DesignProfileSample& sample, const double value)
{
    return sample.hasCadPoint ? csvCell(value, 6) : std::string();
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
                                const bool reverseAxial)
{
    const double sMm = reverseAxial ? (kDesignProfileMaxZMm - zOriginalMm) : zOriginalMm;
    points.push_back({sMm, rMm});
}

std::vector<DesignPolylinePoint> buildDesignPolyline(const bool reverseAxial)
{
    constexpr double kLinearSegmentEndZMm = 52.958772;
    constexpr double kPolynomialSegmentEndZMm = 100.0;
    constexpr double kConstantSegment1EndZMm = 119.0;
    constexpr double kConstantRadius2Mm = 179.919242;
    constexpr double kStepMm = 0.1;

    std::vector<DesignPolylinePoint> points;
    points.reserve(700);

    appendDesignPolylineSample(points, kDesignProfileMinZMm, evalDesignRadiusOriginal(kDesignProfileMinZMm).r_mm, reverseAxial);
    appendDesignPolylineSample(points, kLinearSegmentEndZMm, evalDesignRadiusOriginal(kLinearSegmentEndZMm).r_mm, reverseAxial);

    for (double zMm = kLinearSegmentEndZMm + kStepMm;
         zMm < kPolynomialSegmentEndZMm - 1e-12;
         zMm += kStepMm) {
        const DesignEval design = evalDesignRadiusOriginal(zMm);
        if (design.valid) {
            appendDesignPolylineSample(points, zMm, design.r_mm, reverseAxial);
        }
    }

    appendDesignPolylineSample(points, kPolynomialSegmentEndZMm, evalDesignRadiusOriginal(kPolynomialSegmentEndZMm).r_mm, reverseAxial);
    appendDesignPolylineSample(points,
                               kConstantSegment1EndZMm,
                               evalDesignRadiusOriginal(kConstantSegment1EndZMm).r_mm,
                               reverseAxial);
    appendDesignPolylineSample(points, kConstantSegment1EndZMm, kConstantRadius2Mm, reverseAxial);
    appendDesignPolylineSample(points, kDesignProfileMaxZMm, kConstantRadius2Mm, reverseAxial);

    std::stable_sort(points.begin(), points.end(), [reverseAxial](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return reverseAxial ? lhs.rMm < rhs.rMm : lhs.rMm > rhs.rMm;
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

bool hasExternalDesignProfile(const stitch::StitchPipelineConfig& config)
{
    return config.designUseExternalProfile && config.designExternalProfileSamples.size() >= 2;
}

bool effectiveDesignReverseAxial(const stitch::StitchPipelineConfig& config)
{
    if (hasExternalDesignProfile(config)) {
        return config.designReverseAxial;
    }

    // The builtin analytical generatrix is defined opposite to the stitched contour's
    // increasing axial direction, so builtin evaluation uses reversed axial mapping.
    return true;
}

std::vector<DesignPolylinePoint> buildActiveDesignPolyline(const stitch::StitchPipelineConfig& config)
{
    if (!hasExternalDesignProfile(config)) {
        return buildDesignPolyline(effectiveDesignReverseAxial(config));
    }

    std::vector<DesignPolylinePoint> points;
    points.reserve(config.designExternalProfileSamples.size());
    for (const DesignProfileSample& sample : config.designExternalProfileSamples) {
        if (std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            points.push_back({sample.sMm,
                              sample.rMm,
                              sample.hasCadPoint,
                              sample.cadXMm,
                              sample.cadYMm,
                              sample.cadZMm});
        }
    }

    std::stable_sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return lhs.rMm < rhs.rMm;
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

DesignEval evalExternalDesignProfile(const std::vector<DesignProfileSample>& samples, const double sMm)
{
    DesignEval result;
    if (samples.size() < 2 || !std::isfinite(sMm)) {
        return result;
    }

    if (sMm < samples.front().sMm - 1e-9 || sMm > samples.back().sMm + 1e-9) {
        return result;
    }

    const auto upper = std::lower_bound(samples.begin(), samples.end(), sMm, [](const auto& sample, const double value) {
        return sample.sMm < value;
    });

    std::size_t rightIndex = 0;
    if (upper == samples.end()) {
        rightIndex = samples.size() - 1;
    } else {
        rightIndex = static_cast<std::size_t>(std::distance(samples.begin(), upper));
    }
    if (rightIndex == 0) {
        rightIndex = 1;
    }

    const DesignProfileSample& left = samples[rightIndex - 1];
    const DesignProfileSample& right = samples[rightIndex];
    const double ds = right.sMm - left.sMm;
    if (std::abs(ds) < 1e-12) {
        return result;
    }

    const double t = std::clamp((sMm - left.sMm) / ds, 0.0, 1.0);
    result.valid = true;
    result.z_mm = sMm;
    result.r_mm = left.rMm + t * (right.rMm - left.rMm);
    result.dr_dz = (right.rMm - left.rMm) / ds;
    return result;
}

DesignEval evalDesignProfile(const double sMm, const stitch::StitchPipelineConfig& config)
{
    if (hasExternalDesignProfile(config)) {
        return evalExternalDesignProfile(config.designExternalProfileSamples, sMm);
    }
    return evalDesignRadiusCompare(sMm, effectiveDesignReverseAxial(config));
}

std::vector<DesignProfileSample> toPublicDesignProfileSamples(const std::vector<DesignPolylinePoint>& polyline)
{
    std::vector<DesignProfileSample> samples;
    samples.reserve(polyline.size());
    for (const DesignPolylinePoint& point : polyline) {
        samples.push_back({point.sMm,
                           point.rMm,
                           point.hasCadPoint,
                           point.cadXMm,
                           point.cadYMm,
                           point.cadZMm});
    }
    return samples;
}

DesignProfileMetadata buildActiveDesignProfileMetadata(const stitch::StitchPipelineConfig& config,
                                                       const std::vector<DesignPolylinePoint>& polyline)
{
    DesignProfileMetadata metadata = hasExternalDesignProfile(config)
                                         ? config.designProfileMetadata
                                         : DesignProfileMetadata{};
    if (!hasExternalDesignProfile(config)) {
        metadata.sourceType = "builtin";
        metadata.sourceName = "builtin_generatrix";
    }

    metadata.sampleCount = polyline.size();
    if (!polyline.empty()) {
        metadata.minSMm = std::numeric_limits<double>::infinity();
        metadata.maxSMm = -std::numeric_limits<double>::infinity();
        metadata.minRMm = std::numeric_limits<double>::infinity();
        metadata.maxRMm = -std::numeric_limits<double>::infinity();
        for (const DesignPolylinePoint& point : polyline) {
            metadata.minSMm = std::min(metadata.minSMm, point.sMm);
            metadata.maxSMm = std::max(metadata.maxSMm, point.sMm);
            metadata.minRMm = std::min(metadata.minRMm, point.rMm);
            metadata.maxRMm = std::max(metadata.maxRMm, point.rMm);
        }
    }
    return metadata;
}

int axisIndexFromLabel(const std::string& label)
{
    if (label == "X" || label == "x") {
        return 0;
    }
    if (label == "Y" || label == "y") {
        return 1;
    }
    if (label == "Z" || label == "z") {
        return 2;
    }
    return -1;
}

void setCadAxisValue(double& x, double& y, double& z, const int axisIndex, const double value)
{
    switch (axisIndex) {
    case 0:
        x = value;
        break;
    case 1:
        y = value;
        break;
    case 2:
        z = value;
        break;
    default:
        break;
    }
}

CadPoint mapProfilePointToCadCoordinates(const double sMm,
                                         const double rMm,
                                         const NearestDesignPoint& nearest,
                                         const DesignProfileMetadata& metadata)
{
    CadPoint point;
    if (metadata.sourceType != "external_cad" || !nearest.hasCadPoint) {
        return point;
    }

    const int axialAxis = axisIndexFromLabel(metadata.axialAxis);
    const int radialAxis = axisIndexFromLabel(metadata.radialAxis);
    if (axialAxis < 0 || radialAxis < 0 || axialAxis == radialAxis ||
        !std::isfinite(metadata.cadAxialDirectionSign)) {
        return point;
    }

    point.valid = true;
    point.xMm = nearest.cadXMm;
    point.yMm = nearest.cadYMm;
    point.zMm = nearest.cadZMm;
    setCadAxisValue(point.xMm,
                    point.yMm,
                    point.zMm,
                    axialAxis,
                    metadata.cadAxialOriginMm + metadata.cadAxialDirectionSign * sMm);
    setCadAxisValue(point.xMm, point.yMm, point.zMm, radialAxis, rMm);
    point.valid = std::isfinite(point.xMm) && std::isfinite(point.yMm) && std::isfinite(point.zMm);
    return point;
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
        nearest.hasCadPoint = a.hasCadPoint && b.hasCadPoint;
        if (nearest.hasCadPoint) {
            nearest.cadXMm = a.cadXMm + t * (b.cadXMm - a.cadXMm);
            nearest.cadYMm = a.cadYMm + t * (b.cadYMm - a.cadYMm);
            nearest.cadZMm = a.cadZMm + t * (b.cadZMm - a.cadZMm);
            nearest.hasCadPoint = std::isfinite(nearest.cadXMm) &&
                                  std::isfinite(nearest.cadYMm) &&
                                  std::isfinite(nearest.cadZMm);
        }
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
    return effectiveDesignReverseAxial(config)
               ? (kDesignProfileMaxZMm - config.designStepTransitionOriginalZMm)
               : config.designStepTransitionOriginalZMm;
}

bool isInsideStepTransitionRegion(const double sMm,
                                   const stitch::StitchPipelineConfig& config)
{
    if (hasExternalDesignProfile(config) ||
        !config.designIgnoreStepTransition ||
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

    const DesignEval leftDesign = evalDesignProfile(0.0, config);
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

        const DesignEval design = evalDesignProfile(sAlignedMm, config);
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
    if (usedCount < minUsedPointCount(config)) {
        return evaluation;
    }

    evaluation.ok = true;
    evaluation.dzMm = 0.0;
    evaluation.drMm = drAnchor;
    evaluation.summary.dzMm = 0.0;
    evaluation.summary.drMm = drAnchor;
    evaluation.summary.designReverseAxial = effectiveDesignReverseAxial(config);
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

    const DesignEval design = evalDesignProfile(point.sBaseMm, config);
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

    if (validCount < minUsedPointCount(config)) {
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
    const std::size_t minPoints = minUsedPointCount(config);
    if (run.pointCount < minPoints) {
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
        if (endIndex <= startIndex || (endIndex - startIndex + 1) < minPoints) {
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

            const DesignEval design = evalDesignProfile(point.sBaseMm, config);
            if (!design.valid) {
                continue;
            }

            radialErrorsMm.push_back(point.rRawMm + initialDrMm - design.r_mm);
        }

        const std::size_t minPoints = minUsedPointCount(config);
        if (radialErrorsMm.size() >= std::max<std::size_t>(1, minPoints / 2)) {
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

            const DesignEval design = evalDesignProfile(point.sBaseMm, config);
            if (!design.valid) {
                continue;
            }

            radialErrorsMm.push_back(point.rRawMm + initialDrMm - design.r_mm);
        }

        if (radialErrorsMm.size() < minUsedPointCount(config)) {
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
        usePolylineDistance ? buildActiveDesignPolyline(config) : std::vector<DesignPolylinePoint>{};
    const DesignProfileMetadata designMetadata = buildActiveDesignProfileMetadata(config, designPolyline);

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
        point.hasCadCoordinates = false;
        point.nearestDesignCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadXUm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadYUm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadZUm = std::numeric_limits<double>::quiet_NaN();
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
            const DesignEval design = evalDesignProfile(point.sAlignedMm, config);
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
                    if (nearest.hasCadPoint) {
                        point.nearestDesignCadXMm = nearest.cadXMm;
                        point.nearestDesignCadYMm = nearest.cadYMm;
                        point.nearestDesignCadZMm = nearest.cadZMm;
                        const CadPoint measuredCad =
                            mapProfilePointToCadCoordinates(point.sAlignedMm, point.rAlignedMm, nearest, designMetadata);
                        if (measuredCad.valid) {
                            point.hasCadCoordinates = true;
                            point.measuredCadXMm = measuredCad.xMm;
                            point.measuredCadYMm = measuredCad.yMm;
                            point.measuredCadZMm = measuredCad.zMm;
                            point.compensationTargetCadXMm = nearest.cadXMm;
                            point.compensationTargetCadYMm = nearest.cadYMm;
                            point.compensationTargetCadZMm = nearest.cadZMm;
                            point.compensationDeltaCadXUm = (nearest.cadXMm - measuredCad.xMm) * 1000.0;
                            point.compensationDeltaCadYUm = (nearest.cadYMm - measuredCad.yMm) * 1000.0;
                            point.compensationDeltaCadZUm = (nearest.cadZMm - measuredCad.zMm) * 1000.0;
                        }
                    }
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
    if (usedCount < minUsedPointCount(config)) {
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
    evaluation.summary.designReverseAxial = effectiveDesignReverseAxial(config);
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
              "nearest_design_r_mm,has_cad_coordinates,nearest_design_cad_x_mm,nearest_design_cad_y_mm,"
              "nearest_design_cad_z_mm,measured_cad_x_mm,measured_cad_y_mm,measured_cad_z_mm,"
              "compensation_target_cad_x_mm,compensation_target_cad_y_mm,compensation_target_cad_z_mm,"
              "compensation_delta_cad_x_um,compensation_delta_cad_y_um,compensation_delta_cad_z_um,"
              "design_segment_index,dr_design_ds,radial_error_mm,radial_error_um,"
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
               << (point.hasCadCoordinates ? 1 : 0) << ","
               << csvCell(point.nearestDesignCadXMm, 6) << ","
               << csvCell(point.nearestDesignCadYMm, 6) << ","
               << csvCell(point.nearestDesignCadZMm, 6) << ","
               << csvCell(point.measuredCadXMm, 6) << ","
               << csvCell(point.measuredCadYMm, 6) << ","
               << csvCell(point.measuredCadZMm, 6) << ","
               << csvCell(point.compensationTargetCadXMm, 6) << ","
               << csvCell(point.compensationTargetCadYMm, 6) << ","
               << csvCell(point.compensationTargetCadZMm, 6) << ","
               << csvCell(point.compensationDeltaCadXUm, 6) << ","
               << csvCell(point.compensationDeltaCadYUm, 6) << ","
               << csvCell(point.compensationDeltaCadZUm, 6) << ","
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
              "design_reverse_axial,use_left_endpoint_anchor,design_evaluate_profile_form,anchor_x_px,anchor_y_px,"
              "pixel_size_mm,candidate_count,used_count,outlier_count,outlier_ratio,mean_normal_error_um,normal_rmse_um,normal_mae_um,normal_p95_abs_um,"
              "normal_max_pos_um,normal_max_neg_um,normal_pv_um,profile_rms_um,profile_mae_um,profile_p95_abs_um,"
              "profile_max_pos_um,profile_max_neg_um,profile_pv_um,absolute_all_mean_um,absolute_all_rmse_um,"
              "absolute_all_mae_um,absolute_all_p95_abs_um,absolute_all_max_pos_um,absolute_all_max_neg_um,"
              "absolute_all_pv_um,absolute_filtered_mean_um,absolute_filtered_rmse_um,absolute_filtered_mae_um,"
              "absolute_filtered_p95_abs_um,absolute_filtered_max_pos_um,absolute_filtered_max_neg_um,"
              "absolute_filtered_pv_um,design_source_type,design_source_name,design_profile_sample_count,"
              "design_profile_min_s_mm,design_profile_max_s_mm,design_profile_min_r_mm,design_profile_max_r_mm,"
              "cad_extraction_method,cad_section_normal_axis,cad_section_coordinate_mm,cad_axial_axis,"
              "cad_radial_axis,cad_axial_origin_mm,cad_axial_direction_sign\n";
    stream << csvCell(summary.dzMm, 6) << ","
           << csvCell(summary.drMm, 6) << ","
           << csvCell(summary.dThetaDeg, 6) << ","
           << csvCell(summary.axialScaleFactor, 6) << ","
           << csvCell(summary.axialQuadraticTermMm, 6) << ","
           << (summary.appliedAbsoluteBiasRefine ? 1 : 0) << ","
           << csvCell(summary.absoluteBiasCorrectionUm, 6) << ","
           << csvCell(summary.preRefineMeanNormalErrorUm, 6) << ","
           << csvCell(summary.preRefineAbsoluteFilteredRmseUm, 6) << ","
           << (summary.designReverseAxial ? 1 : 0) << ","
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
           << csvCell(summary.absoluteFilteredStats.pvUm, 6) << ","
           << csvTextCell(summary.designProfileMetadata.sourceType) << ","
           << csvTextCell(summary.designProfileMetadata.sourceName) << ","
           << summary.designProfileMetadata.sampleCount << ","
           << csvCell(summary.designProfileMetadata.minSMm, 6) << ","
           << csvCell(summary.designProfileMetadata.maxSMm, 6) << ","
           << csvCell(summary.designProfileMetadata.minRMm, 6) << ","
           << csvCell(summary.designProfileMetadata.maxRMm, 6) << ","
           << csvTextCell(summary.designProfileMetadata.extractionMethod) << ","
           << csvTextCell(summary.designProfileMetadata.sectionNormalAxis) << ","
           << csvCell(summary.designProfileMetadata.sectionCoordinateMm, 6) << ","
           << csvTextCell(summary.designProfileMetadata.axialAxis) << ","
           << csvTextCell(summary.designProfileMetadata.radialAxis) << ","
           << csvCell(summary.designProfileMetadata.cadAxialOriginMm, 6) << ","
           << csvCell(summary.designProfileMetadata.cadAxialDirectionSign, 6) << "\n";
    return stream.str();
}

std::string buildCompensationCsv(const std::vector<DesignErrorProfilePoint>& points,
                                 const DesignProfileMetadata& metadata)
{
    std::ostringstream stream;
    stream << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,has_cad_coordinates,"
              "cad_design_x_mm,cad_design_y_mm,cad_design_z_mm,cad_measured_x_mm,cad_measured_y_mm,"
              "cad_measured_z_mm,cad_compensation_target_x_mm,cad_compensation_target_y_mm,"
              "cad_compensation_target_z_mm,compensated_cad_x_mm,compensated_cad_y_mm,"
              "compensated_cad_z_mm,delta_x_um,delta_y_um,delta_z_um,s_aligned_mm,"
              "measured_r_mm,design_s_mm,design_r_mm,radial_error_um,normal_error_um,profile_error_um,"
              "compensation_normal_um,compensation_radial_um,compensation_radial_mm,"
              "compensated_target_r_mm,is_used\n";

    for (const DesignErrorProfilePoint& point : points) {
        const double compensationNormalUm =
            std::isfinite(point.normalErrorUm) ? -point.normalErrorUm : std::numeric_limits<double>::quiet_NaN();
        const double compensationRadialUm =
            std::isfinite(point.radialErrorUm) ? -point.radialErrorUm : std::numeric_limits<double>::quiet_NaN();
        const double compensationRadialMm =
            std::isfinite(point.radialErrorMm) ? -point.radialErrorMm : std::numeric_limits<double>::quiet_NaN();
        const double compensatedTargetRMm =
            std::isfinite(point.rAlignedMm) && std::isfinite(compensationRadialMm)
                ? point.rAlignedMm + compensationRadialMm
                : std::numeric_limits<double>::quiet_NaN();

        stream << csvTextCell(metadata.sourceType) << ","
               << csvTextCell(metadata.sourceName) << ","
               << csvTextCell(metadata.extractionMethod) << ","
               << csvTextCell(metadata.axialAxis) << ","
               << csvTextCell(metadata.radialAxis) << ","
               << point.index << ","
               << (point.hasCadCoordinates ? 1 : 0) << ","
               << csvCell(point.nearestDesignCadXMm, 6) << ","
               << csvCell(point.nearestDesignCadYMm, 6) << ","
               << csvCell(point.nearestDesignCadZMm, 6) << ","
               << csvCell(point.measuredCadXMm, 6) << ","
               << csvCell(point.measuredCadYMm, 6) << ","
               << csvCell(point.measuredCadZMm, 6) << ","
               << csvCell(point.compensationTargetCadXMm, 6) << ","
               << csvCell(point.compensationTargetCadYMm, 6) << ","
               << csvCell(point.compensationTargetCadZMm, 6) << ","
               << csvCell(point.compensationTargetCadXMm, 6) << ","
               << csvCell(point.compensationTargetCadYMm, 6) << ","
               << csvCell(point.compensationTargetCadZMm, 6) << ","
               << csvCell(point.compensationDeltaCadXUm, 6) << ","
               << csvCell(point.compensationDeltaCadYUm, 6) << ","
               << csvCell(point.compensationDeltaCadZUm, 6) << ","
               << csvCell(point.sAlignedMm, 6) << ","
               << csvCell(point.rAlignedMm, 6) << ","
               << csvCell(point.nearestDesignSMm, 6) << ","
               << csvCell(point.nearestDesignRMm, 6) << ","
               << csvCell(point.radialErrorUm, 6) << ","
               << csvCell(point.normalErrorUm, 6) << ","
               << csvCell(point.profileErrorUm, 6) << ","
               << csvCell(compensationNormalUm, 6) << ","
               << csvCell(compensationRadialUm, 6) << ","
               << csvCell(compensationRadialMm, 9) << ","
               << csvCell(compensatedTargetRMm, 6) << ","
               << (point.isUsed ? 1 : 0) << "\n";
    }

    return stream.str();
}

std::string build3dErrorCsv(const std::vector<DesignErrorProfilePoint>& points,
                            const DesignProfileMetadata& metadata)
{
    std::ostringstream stream;
    stream << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,is_used,"
              "has_cad_coordinates,design_x_mm,design_y_mm,design_z_mm,measured_x_mm,measured_y_mm,measured_z_mm,"
              "error_x_um,error_y_um,error_z_um,error_3d_um,normal_error_um,radial_error_um,profile_error_um,"
              "compensation_target_x_mm,compensation_target_y_mm,compensation_target_z_mm,"
              "compensation_x_um,compensation_y_um,compensation_z_um,s_aligned_mm,measured_r_mm,design_s_mm,design_r_mm\n";

    for (const DesignErrorProfilePoint& point : points) {
        const double errorXUm = (point.hasCadCoordinates && std::isfinite(point.measuredCadXMm) &&
                                 std::isfinite(point.nearestDesignCadXMm))
                                    ? (point.measuredCadXMm - point.nearestDesignCadXMm) * 1000.0
                                    : std::numeric_limits<double>::quiet_NaN();
        const double errorYUm = (point.hasCadCoordinates && std::isfinite(point.measuredCadYMm) &&
                                 std::isfinite(point.nearestDesignCadYMm))
                                    ? (point.measuredCadYMm - point.nearestDesignCadYMm) * 1000.0
                                    : std::numeric_limits<double>::quiet_NaN();
        const double errorZUm = (point.hasCadCoordinates && std::isfinite(point.measuredCadZMm) &&
                                 std::isfinite(point.nearestDesignCadZMm))
                                    ? (point.measuredCadZMm - point.nearestDesignCadZMm) * 1000.0
                                    : std::numeric_limits<double>::quiet_NaN();
        const double error3dUm =
            std::isfinite(errorXUm) && std::isfinite(errorYUm) && std::isfinite(errorZUm)
                ? std::sqrt(errorXUm * errorXUm + errorYUm * errorYUm + errorZUm * errorZUm)
                : std::numeric_limits<double>::quiet_NaN();

        stream << csvTextCell(metadata.sourceType) << ","
               << csvTextCell(metadata.sourceName) << ","
               << csvTextCell(metadata.extractionMethod) << ","
               << csvTextCell(metadata.axialAxis) << ","
               << csvTextCell(metadata.radialAxis) << ","
               << point.index << ","
               << (point.isUsed ? 1 : 0) << ","
               << (point.hasCadCoordinates ? 1 : 0) << ","
               << csvCell(point.nearestDesignCadXMm, 6) << ","
               << csvCell(point.nearestDesignCadYMm, 6) << ","
               << csvCell(point.nearestDesignCadZMm, 6) << ","
               << csvCell(point.measuredCadXMm, 6) << ","
               << csvCell(point.measuredCadYMm, 6) << ","
               << csvCell(point.measuredCadZMm, 6) << ","
               << csvCell(errorXUm, 6) << ","
               << csvCell(errorYUm, 6) << ","
               << csvCell(errorZUm, 6) << ","
               << csvCell(error3dUm, 6) << ","
               << csvCell(point.normalErrorUm, 6) << ","
               << csvCell(point.radialErrorUm, 6) << ","
               << csvCell(point.profileErrorUm, 6) << ","
               << csvCell(point.compensationTargetCadXMm, 6) << ","
               << csvCell(point.compensationTargetCadYMm, 6) << ","
               << csvCell(point.compensationTargetCadZMm, 6) << ","
               << csvCell(point.compensationDeltaCadXUm, 6) << ","
               << csvCell(point.compensationDeltaCadYUm, 6) << ","
               << csvCell(point.compensationDeltaCadZUm, 6) << ","
               << csvCell(point.sAlignedMm, 6) << ","
               << csvCell(point.rAlignedMm, 6) << ","
               << csvCell(point.nearestDesignSMm, 6) << ","
               << csvCell(point.nearestDesignRMm, 6) << "\n";
    }

    return stream.str();
}

double percentileValue(std::vector<double> values, const double quantile)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double clamped = std::clamp(quantile, 0.0, 1.0);
    const double rawIndex = clamped * static_cast<double>(values.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(rawIndex));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(rawIndex));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lowerIndex), values.end());
    const double lowerValue = values[lowerIndex];
    if (lowerIndex == upperIndex) {
        return lowerValue;
    }

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(upperIndex), values.end());
    const double upperValue = values[upperIndex];
    const double blend = rawIndex - static_cast<double>(lowerIndex);
    return lowerValue * (1.0 - blend) + upperValue * blend;
}

SlotFeatureEstimate estimatePrimarySlotFeature(std::vector<cv::Point2d> series,
                                               const std::string& method,
                                               const double minSMm = -std::numeric_limits<double>::infinity(),
                                               const double maxSMm = std::numeric_limits<double>::infinity())
{
    SlotFeatureEstimate estimate;
    estimate.method = method;
    if (!std::isinf(minSMm) || !std::isinf(maxSMm)) {
        series.erase(std::remove_if(series.begin(),
                                    series.end(),
                                    [minSMm, maxSMm](const cv::Point2d& point) {
                                        return point.x < minSMm || point.x > maxSMm;
                                    }),
                     series.end());
    }
    if (series.size() < 5) {
        return estimate;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            radii.push_back(point.y);
        }
    }
    if (radii.size() < 5) {
        return estimate;
    }

    const double topR = percentileValue(radii, 0.90);
    auto bottomIt = std::min_element(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.y < rhs.y;
    });
    if (bottomIt == series.end() || !std::isfinite(topR)) {
        return estimate;
    }

    const double bottomR = bottomIt->y;
    const double depthMm = topR - bottomR;
    if (!(depthMm > 1e-6)) {
        return estimate;
    }

    const double halfDepthThreshold = topR - 0.5 * depthMm;
    std::size_t bottomIndex = static_cast<std::size_t>(std::distance(series.begin(), bottomIt));
    std::size_t leftIndex = bottomIndex;
    while (leftIndex > 0 && series[leftIndex - 1].y <= halfDepthThreshold) {
        --leftIndex;
    }
    std::size_t rightIndex = bottomIndex;
    while (rightIndex + 1 < series.size() && series[rightIndex + 1].y <= halfDepthThreshold) {
        ++rightIndex;
    }

    const double leftS = series[leftIndex].x;
    const double rightS = series[rightIndex].x;
    const double widthMm = rightS - leftS;
    if (!(widthMm > 1e-6)) {
        return estimate;
    }

    estimate.valid = true;
    estimate.leftSMm = leftS;
    estimate.rightSMm = rightS;
    estimate.centerSMm = 0.5 * (leftS + rightS);
    estimate.widthMm = widthMm;
    estimate.depthMm = depthMm;
    estimate.topRMm = topR;
    estimate.bottomRMm = bottomR;
    estimate.bottomSMm = bottomIt->x;
    return estimate;
}

SlotFeatureEstimate estimateTargetWidthSlotFeature(std::vector<cv::Point2d> series,
                                                   const std::string& method,
                                                   const double targetWidthMm,
                                                   const double targetDepthMm)
{
    SlotFeatureEstimate best;
    best.method = method;
    if (!(targetWidthMm > 1e-9) || series.size() < 5) {
        return best;
    }

    series.erase(std::remove_if(series.begin(),
                                series.end(),
                                [](const cv::Point2d& point) {
                                    return !std::isfinite(point.x) || !std::isfinite(point.y);
                                }),
                 series.end());
    if (series.size() < 5) {
        return best;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        radii.push_back(point.y);
    }

    const double topR = percentileValue(radii, 0.90);
    const double lowR = percentileValue(radii, 0.10);
    const double depthSpanMm = topR - lowR;
    if (!(depthSpanMm > 1e-6)) {
        return best;
    }

    double bestScore = std::numeric_limits<double>::infinity();
    const double maxGapMm = std::max(0.25, targetWidthMm * 0.25);
    for (const double thresholdRatio : {0.35, 0.50, 0.65}) {
        const double thresholdR = topR - thresholdRatio * depthSpanMm;
        std::size_t index = 0;
        while (index < series.size()) {
            while (index < series.size() && series[index].y > thresholdR) {
                ++index;
            }
            if (index >= series.size()) {
                break;
            }

            const std::size_t start = index;
            std::size_t end = index;
            std::size_t bottomIndex = index;
            while (end + 1 < series.size() &&
                   series[end + 1].y <= thresholdR &&
                   (series[end + 1].x - series[end].x) <= maxGapMm) {
                ++end;
                if (series[end].y < series[bottomIndex].y) {
                    bottomIndex = end;
                }
            }

            const double leftS = series[start].x;
            const double rightS = series[end].x;
            const double widthMm = rightS - leftS;
            if (widthMm > 1e-6) {
                const double depthMm = topR - series[bottomIndex].y;
                double score = std::abs(widthMm - targetWidthMm) / targetWidthMm;
                if (targetDepthMm > 1e-9 && std::isfinite(depthMm)) {
                    score += 0.25 * std::abs(depthMm - targetDepthMm) / targetDepthMm;
                }
                // Prefer a valley-like run over a short corner artifact when widths are similar.
                score -= 0.02 * std::clamp(depthMm / std::max(depthSpanMm, 1e-9), 0.0, 1.0);
                if (score < bestScore) {
                    bestScore = score;
                    best.valid = true;
                    best.method = method;
                    best.leftSMm = leftS;
                    best.rightSMm = rightS;
                    best.centerSMm = 0.5 * (leftS + rightS);
                    best.widthMm = widthMm;
                    best.depthMm = depthMm;
                    best.topRMm = topR;
                    best.bottomRMm = series[bottomIndex].y;
                    best.bottomSMm = series[bottomIndex].x;
                }
            }

            index = end + 1;
        }
    }

    return best;
}

SlotFeatureEstimate estimateTargetSpanSlotFeature(std::vector<cv::Point2d> series,
                                                  const std::string& method,
                                                  const double targetWidthMm,
                                                  const double targetDepthMm)
{
    SlotFeatureEstimate best;
    best.method = method;
    if (!(targetWidthMm > 1e-9) || series.size() < 5) {
        return best;
    }

    series.erase(std::remove_if(series.begin(),
                                series.end(),
                                [](const cv::Point2d& point) {
                                    return !std::isfinite(point.x) || !std::isfinite(point.y);
                                }),
                 series.end());
    if (series.size() < 5) {
        return best;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        radii.push_back(point.y);
    }
    const double topR = percentileValue(radii, 0.90);
    const double lowR = percentileValue(radii, 0.05);
    const double depthSpanMm = topR - lowR;
    if (!(depthSpanMm > 1e-6)) {
        return best;
    }

    double bestScore = std::numeric_limits<double>::infinity();
    for (const double thresholdRatio : {0.45, 0.50, 0.55, 0.60, 0.65}) {
        const double thresholdR = topR - thresholdRatio * depthSpanMm;
        std::vector<cv::Point2d> inside;
        inside.reserve(series.size());
        for (const cv::Point2d& point : series) {
            if (point.y <= thresholdR) {
                inside.push_back(point);
            }
        }
        if (inside.size() < 5) {
            continue;
        }

        const auto leftIt = std::min_element(inside.begin(), inside.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.x < rhs.x;
        });
        const auto rightIt = std::max_element(inside.begin(), inside.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.x < rhs.x;
        });
        const auto bottomIt = std::min_element(inside.begin(), inside.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.y < rhs.y;
        });
        if (leftIt == inside.end() || rightIt == inside.end() || bottomIt == inside.end()) {
            continue;
        }

        const double widthMm = rightIt->x - leftIt->x;
        const double depthMm = topR - bottomIt->y;
        if (!(widthMm > 1e-6) || widthMm < targetWidthMm * 0.35 || widthMm > targetWidthMm * 1.80) {
            continue;
        }

        double score = std::abs(widthMm - targetWidthMm) / targetWidthMm;
        if (targetDepthMm > 1e-9 && std::isfinite(depthMm)) {
            score += 0.12 * std::abs(depthMm - targetDepthMm) / targetDepthMm;
        }
        if (score < bestScore) {
            bestScore = score;
            best.valid = true;
            best.method = method;
            best.leftSMm = leftIt->x;
            best.rightSMm = rightIt->x;
            best.centerSMm = 0.5 * (leftIt->x + rightIt->x);
            best.widthMm = widthMm;
            best.depthMm = depthMm;
            best.topRMm = topR;
            best.bottomRMm = bottomIt->y;
            best.bottomSMm = bottomIt->x;
        }
    }

    return best;
}

SlotFeatureEstimate estimateMeasuredProfileSpanFeature(std::vector<cv::Point2d> series,
                                                       const std::string& method)
{
    SlotFeatureEstimate feature;
    feature.method = method;
    series.erase(std::remove_if(series.begin(),
                                series.end(),
                                [](const cv::Point2d& point) {
                                    return !std::isfinite(point.x) || !std::isfinite(point.y);
                                }),
                 series.end());
    if (series.size() < 5) {
        return feature;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        radii.push_back(point.y);
    }
    const auto bottomIt = std::min_element(series.begin(),
                                           series.end(),
                                           [](const auto& lhs, const auto& rhs) {
                                               return lhs.y < rhs.y;
                                           });
    if (bottomIt == series.end()) {
        return feature;
    }

    feature.valid = true;
    feature.leftSMm = series.front().x;
    feature.rightSMm = series.back().x;
    feature.centerSMm = 0.5 * (feature.leftSMm + feature.rightSMm);
    feature.widthMm = feature.rightSMm - feature.leftSMm;
    feature.topRMm = percentileValue(radii, 0.90);
    feature.bottomRMm = percentileValue(radii, 0.10);
    if (feature.topRMm < feature.bottomRMm) {
        std::swap(feature.topRMm, feature.bottomRMm);
    }
    feature.depthMm = feature.topRMm - feature.bottomRMm;
    feature.bottomSMm = bottomIt->x;
    return feature.widthMm > 1e-9 ? feature : SlotFeatureEstimate{};
}

SlotFeatureEstimate estimateDominantLowRunSlotFeature(std::vector<cv::Point2d> series,
                                                       const std::string& method)
{
    SlotFeatureEstimate best;
    best.method = method;

    series.erase(std::remove_if(series.begin(),
                                series.end(),
                                [](const cv::Point2d& point) {
                                    return !std::isfinite(point.x) || !std::isfinite(point.y);
                                }),
                 series.end());
    if (series.size() < 5) {
        return best;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        radii.push_back(point.y);
    }

    const double topR = percentileValue(radii, 0.90);
    const double lowR = percentileValue(radii, 0.10);
    const double depthSpanMm = topR - lowR;
    const double totalSpanMm = series.back().x - series.front().x;
    if (!(depthSpanMm > 1e-6) || !(totalSpanMm > 1e-6)) {
        return best;
    }

    const double minWidthMm = std::max(0.2, totalSpanMm * 0.05);
    const double maxGapMm = std::max(0.25, totalSpanMm * 0.015);
    double bestScore = -std::numeric_limits<double>::infinity();
    for (const double thresholdRatio : {0.35, 0.50, 0.65}) {
        const double thresholdR = topR - thresholdRatio * depthSpanMm;
        std::size_t index = 0;
        while (index < series.size()) {
            while (index < series.size() && series[index].y > thresholdR) {
                ++index;
            }
            if (index >= series.size()) {
                break;
            }

            const std::size_t start = index;
            std::size_t end = index;
            std::size_t bottomIndex = index;
            while (end + 1 < series.size() &&
                   series[end + 1].y <= thresholdR &&
                   (series[end + 1].x - series[end].x) <= maxGapMm) {
                ++end;
                if (series[end].y < series[bottomIndex].y) {
                    bottomIndex = end;
                }
            }

            const double leftS = series[start].x;
            const double rightS = series[end].x;
            const double widthMm = rightS - leftS;
            const double depthMm = topR - series[bottomIndex].y;
            if (widthMm >= minWidthMm && depthMm > 1e-6) {
                const double widthRatio = std::clamp(widthMm / totalSpanMm, 0.0, 1.0);
                const double depthRatio = std::clamp(depthMm / depthSpanMm, 0.0, 1.0);
                const double score = widthMm * (0.65 + depthRatio) - 0.08 * widthRatio * totalSpanMm;
                if (score > bestScore) {
                    bestScore = score;
                    best.valid = true;
                    best.method = method;
                    best.leftSMm = leftS;
                    best.rightSMm = rightS;
                    best.centerSMm = 0.5 * (leftS + rightS);
                    best.widthMm = widthMm;
                    best.depthMm = depthMm;
                    best.topRMm = topR;
                    best.bottomRMm = series[bottomIndex].y;
                    best.bottomSMm = series[bottomIndex].x;
                }
            }

            index = end + 1;
        }
    }

    return best;
}

SlotFeatureEstimate estimateCentralBottomGrooveFeature(std::vector<cv::Point2d> series,
                                                       const std::string& method,
                                                       const double targetWidthMm,
                                                       const double targetDepthMm)
{
    SlotFeatureEstimate feature;
    feature.method = method;

    series.erase(std::remove_if(series.begin(),
                                series.end(),
                                [](const cv::Point2d& point) {
                                    return !std::isfinite(point.x) || !std::isfinite(point.y);
                                }),
                 series.end());
    if (series.size() < 5) {
        return feature;
    }

    std::stable_sort(series.begin(), series.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> radii;
    radii.reserve(series.size());
    for (const cv::Point2d& point : series) {
        radii.push_back(point.y);
    }
    const double globalMinR = *std::min_element(radii.begin(), radii.end());
    const double maxR = *std::max_element(radii.begin(), radii.end());
    const double totalSpanS = series.back().x - series.front().x;
    const double totalSpanR = maxR - globalMinR;
    if (!(totalSpanS > 1e-9) || !(totalSpanR > 1e-9)) {
        return feature;
    }

    const double profileCenterS = 0.5 * (series.front().x + series.back().x);
    const double desiredWidth =
        targetWidthMm > 1e-9 ? targetWidthMm : std::max(0.8, totalSpanS * 0.30);
    const double searchLeft = profileCenterS - desiredWidth * 0.55;
    const double searchRight = profileCenterS + desiredWidth * 0.55;

    std::vector<cv::Point2d> centerWindow;
    centerWindow.reserve(series.size());
    for (const cv::Point2d& point : series) {
        if (point.x >= searchLeft - 1e-9 && point.x <= searchRight + 1e-9) {
            centerWindow.push_back(point);
        }
    }
    if (centerWindow.size() < 3) {
        return feature;
    }

    const auto centerMinRIt = std::min_element(centerWindow.begin(),
                                               centerWindow.end(),
                                               [](const auto& lhs, const auto& rhs) {
                                                   return lhs.y < rhs.y;
                                               });
    if (centerMinRIt == centerWindow.end()) {
        return feature;
    }
    const double centerMinR = centerMinRIt->y;

    const double localDepth =
        std::max(0.45,
                 std::min({targetDepthMm > 1e-9 ? targetDepthMm * 0.18 : totalSpanR * 0.18,
                           desiredWidth * 0.18,
                           totalSpanR * 0.20,
                           1.20}));
    const double lowBandR = centerMinR + std::max(0.08, localDepth * 0.22);
    std::vector<cv::Point2d> lowBand;
    lowBand.reserve(centerWindow.size());
    for (const cv::Point2d& point : centerWindow) {
        if (point.y <= lowBandR) {
            lowBand.push_back(point);
        }
    }
    if (lowBand.empty()) {
        return feature;
    }

    const auto bottomIt = std::min_element(lowBand.begin(),
                                           lowBand.end(),
                                           [profileCenterS, centerMinR](const auto& lhs, const auto& rhs) {
                                               const double lhsScore =
                                                   std::abs(lhs.x - profileCenterS) +
                                                   0.35 * std::abs(lhs.y - centerMinR);
                                               const double rhsScore =
                                                   std::abs(rhs.x - profileCenterS) +
                                                   0.35 * std::abs(rhs.y - centerMinR);
                                               return lhsScore < rhsScore;
                                           });
    if (bottomIt == lowBand.end()) {
        return feature;
    }

    const double centerS = bottomIt->x;
    const double localTopR = centerMinR + localDepth;
    const double windowLeft = centerS - desiredWidth * 0.5;
    const double windowRight = centerS + desiredWidth * 0.5;

    std::vector<cv::Point2d> local;
    local.reserve(series.size());
    for (const cv::Point2d& point : series) {
        if (point.x >= windowLeft - 1e-9 &&
            point.x <= windowRight + 1e-9 &&
            point.y <= localTopR + 1e-9) {
            local.push_back(point);
        }
    }
    if (local.size() < 3) {
        return feature;
    }

    const auto leftIt = std::min_element(local.begin(), local.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.x < rhs.x;
    });
    const auto rightIt = std::max_element(local.begin(), local.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.x < rhs.x;
    });
    if (leftIt == local.end() || rightIt == local.end()) {
        return feature;
    }

    feature.valid = true;
    feature.method = method;
    feature.leftSMm = leftIt->x;
    feature.rightSMm = rightIt->x;
    feature.centerSMm = 0.5 * (feature.leftSMm + feature.rightSMm);
    feature.widthMm = feature.rightSMm - feature.leftSMm;
    feature.depthMm = localTopR - bottomIt->y;
    feature.topRMm = localTopR;
    feature.bottomRMm = bottomIt->y;
    feature.bottomSMm = bottomIt->x;
    return feature;
}

DesignProfileSample interpolateDesignProfileSample(std::vector<DesignProfileSample> samples, const double sMm)
{
    DesignProfileSample result;
    if (samples.size() < 2 || !std::isfinite(sMm)) {
        return result;
    }

    std::stable_sort(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.sMm < rhs.sMm;
    });
    if (sMm < samples.front().sMm - 1e-9 || sMm > samples.back().sMm + 1e-9) {
        return result;
    }

    const auto upper = std::lower_bound(samples.begin(), samples.end(), sMm, [](const auto& sample, const double value) {
        return sample.sMm < value;
    });
    std::size_t rightIndex = upper == samples.end()
                                 ? samples.size() - 1
                                 : static_cast<std::size_t>(std::distance(samples.begin(), upper));
    if (rightIndex == 0) {
        rightIndex = 1;
    }

    const DesignProfileSample& left = samples[rightIndex - 1];
    const DesignProfileSample& right = samples[rightIndex];
    const double ds = right.sMm - left.sMm;
    if (std::abs(ds) < 1e-12) {
        return result;
    }

    const double t = std::clamp((sMm - left.sMm) / ds, 0.0, 1.0);
    result.sMm = left.sMm + t * ds;
    result.rMm = left.rMm + t * (right.rMm - left.rMm);
    result.hasCadPoint = left.hasCadPoint && right.hasCadPoint;
    if (result.hasCadPoint) {
        result.cadXMm = left.cadXMm + t * (right.cadXMm - left.cadXMm);
        result.cadYMm = left.cadYMm + t * (right.cadYMm - left.cadYMm);
        result.cadZMm = left.cadZMm + t * (right.cadZMm - left.cadZMm);
    }
    return result;
}

SlotFeatureEstimate estimateConfiguredDesignSlotFeature(const stitch::StitchPipelineConfig& config)
{
    SlotFeatureEstimate feature;
    if (!(config.designTargetSlotWidthMm > 1e-9)) {
        return feature;
    }

    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(config);
    std::vector<cv::Point2d> designSeries;
    designSeries.reserve(activeDesignPolyline.size());
    for (const DesignPolylinePoint& point : activeDesignPolyline) {
        if (std::isfinite(point.sMm) && std::isfinite(point.rMm)) {
            designSeries.emplace_back(point.sMm, point.rMm);
        }
    }

    feature = estimateCentralBottomGrooveFeature(designSeries,
                                                 "central_bottom_v_groove_from_cad_profile",
                                                 config.designTargetSlotWidthMm,
                                                 config.designTargetSlotDepthMm);
    if (!feature.valid) {
        feature = estimateTargetWidthSlotFeature(designSeries,
                                                 "target_width_slot_from_cad_profile",
                                                 config.designTargetSlotWidthMm,
                                                 config.designTargetSlotDepthMm);
    }
    if (!feature.valid) {
        feature = estimatePrimarySlotFeature(std::move(designSeries), "primary_slot_from_cad_profile");
    }
    return feature;
}

std::string buildFeatureCompensationCsv(const std::vector<DesignProfileSample>& designSamples,
                                        const std::vector<DesignErrorProfilePoint>& points,
                                        const DesignProfileMetadata& metadata,
                                        const stitch::StitchPipelineConfig& config)
{
    std::ostringstream stream;
    stream << "feature_id,status,method,design_source_type,design_source_name,cad_extraction_method,"
              "target_slot_width_mm,target_slot_depth_mm,cad_slot_left_s_mm,cad_slot_right_s_mm,"
              "cad_slot_center_s_mm,cad_slot_width_mm,cad_slot_depth_mm,cad_slot_bottom_s_mm,"
              "cad_slot_bottom_r_mm,cad_slot_center_x_mm,cad_slot_center_y_mm,cad_slot_center_z_mm,"
              "measured_slot_center_s_mm,measured_slot_width_mm,measured_slot_depth_mm,"
              "center_error_um,width_error_um,depth_error_um,compensation_center_um,"
              "compensation_width_um,compensation_depth_um,notes\n";

    std::vector<cv::Point2d> designSeries;
    designSeries.reserve(designSamples.size());
    for (const DesignProfileSample& sample : designSamples) {
        if (std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            designSeries.emplace_back(sample.sMm, sample.rMm);
        }
    }

    std::vector<cv::Point2d> measuredSeries;
    measuredSeries.reserve(points.size());
    for (const DesignErrorProfilePoint& point : points) {
        if (point.isUsed && std::isfinite(point.sAlignedMm) && std::isfinite(point.rAlignedMm)) {
            measuredSeries.emplace_back(point.sAlignedMm, point.rAlignedMm);
        }
    }

    SlotFeatureEstimate designFeature;
    if (config.designTargetSlotWidthMm > 1e-9) {
        designFeature = estimateCentralBottomGrooveFeature(designSeries,
                                                           "central_bottom_v_groove_from_cad_profile",
                                                           config.designTargetSlotWidthMm,
                                                           config.designTargetSlotDepthMm);
        if (!designFeature.valid) {
            designFeature = estimateTargetWidthSlotFeature(designSeries,
                                                           "target_width_slot_from_cad_profile",
                                                           config.designTargetSlotWidthMm,
                                                           config.designTargetSlotDepthMm);
        }
    }
    if (!designFeature.valid) {
        designFeature =
            estimatePrimarySlotFeature(std::move(designSeries), "half_depth_primary_slot_from_cad_profile");
    }
    if (!designFeature.valid) {
        if (config.designTargetSlotWidthMm > 1e-9) {
            const SlotFeatureEstimate measuredFeature =
                estimatePrimarySlotFeature(measuredSeries, "single_slot_width_from_measured_profile");
            if (measuredFeature.valid) {
                const double targetWidthMm = config.designTargetSlotWidthMm;
                const double targetDepthMm = config.designTargetSlotDepthMm > 1e-9
                                                 ? config.designTargetSlotDepthMm
                                                 : std::numeric_limits<double>::quiet_NaN();
                const double widthErrorUm = (measuredFeature.widthMm - targetWidthMm) * 1000.0;
                const double depthErrorUm = std::isfinite(targetDepthMm)
                                                ? (measuredFeature.depthMm - targetDepthMm) * 1000.0
                                                : std::numeric_limits<double>::quiet_NaN();
                std::ostringstream note;
                note << "CAD slot feature was not detected; resolved one slot width against the target width only.";
                const double relativeWidthError =
                    targetWidthMm > 1e-9 ? std::abs(measuredFeature.widthMm - targetWidthMm) / targetWidthMm
                                          : 0.0;
                const bool widthMismatch = relativeWidthError > 0.25;
                if (relativeWidthError > 0.25) {
                    note << " Large width mismatch; verify that CAD and images are from the same part and that "
                            "pixel calibration is loaded.";
                    if (measuredFeature.widthMm > 1e-9 && config.designPixelSizeMm > 0.0) {
                        const double scaleFactor = targetWidthMm / measuredFeature.widthMm;
                        note << " If this target width is used only as a scale reference, multiply pixel_size_mm "
                                "by "
                             << formatDouble(scaleFactor, 6) << " (current "
                             << formatDouble(config.designPixelSizeMm, 9) << " mm/px -> "
                             << formatDouble(config.designPixelSizeMm * scaleFactor, 9) << " mm/px).";
                    }
                }
                appendFeatureCompensationRow(stream,
                                             {"single_slot_width",
                                              widthMismatch ? "mismatch" : "ok",
                                              "single_slot_width_target",
                                              csvTextCell(metadata.sourceType),
                                              csvTextCell(metadata.sourceName),
                                              csvTextCell(metadata.extractionMethod),
                                              csvCell(targetWidthMm, 6),
                                              csvCell(targetDepthMm, 6),
                                              "",
                                              "",
                                              "",
                                              csvCell(targetWidthMm, 6),
                                              csvCell(targetDepthMm, 6),
                                              "",
                                              "",
                                              "",
                                              "",
                                              "",
                                              csvCell(measuredFeature.centerSMm, 6),
                                              csvCell(measuredFeature.widthMm, 6),
                                              csvCell(measuredFeature.depthMm, 6),
                                              "",
                                              csvCell(widthErrorUm, 6),
                                              csvCell(depthErrorUm, 6),
                                              "",
                                              csvCell(-widthErrorUm, 6),
                                              csvCell(std::isfinite(depthErrorUm) ? -depthErrorUm
                                                                                  : std::numeric_limits<double>::quiet_NaN(),
                                                      6),
                                              csvTextCell(note.str())});
                return stream.str();
            }

            std::vector<std::string> row = {"single_slot_width",
                                            "unavailable",
                                            "single_slot_width_target",
                                            csvTextCell(metadata.sourceType),
                                            csvTextCell(metadata.sourceName),
                                            csvTextCell(metadata.extractionMethod),
                                            csvCell(config.designTargetSlotWidthMm, 6),
                                            csvCell(config.designTargetSlotDepthMm, 6)};
            row.resize(kFeatureCompensationCsvColumnCount - 1);
            row.push_back(csvTextCell(
                "CAD slot feature was not detected, and the measured profile does not contain a detectable single slot width."));
            appendFeatureCompensationRow(stream, std::move(row));
            return stream.str();
        }

        std::vector<std::string> row = {"primary_slot",
                                        "unavailable",
                                        "half_depth_primary_slot",
                                        csvTextCell(metadata.sourceType),
                                        csvTextCell(metadata.sourceName),
                                        csvTextCell(metadata.extractionMethod),
                                        csvCell(config.designTargetSlotWidthMm, 6),
                                        csvCell(config.designTargetSlotDepthMm, 6)};
        row.resize(kFeatureCompensationCsvColumnCount - 1);
        row.push_back(csvTextCell("CAD design profile does not contain a detectable concave slot feature."));
        appendFeatureCompensationRow(stream, std::move(row));
        return stream.str();
    }

    const double targetWidthMm =
        config.designTargetSlotWidthMm > 1e-9 ? config.designTargetSlotWidthMm : designFeature.widthMm;
    const double targetDepthMm = designFeature.depthMm > 1e-9
                                     ? designFeature.depthMm
                                     : config.designTargetSlotDepthMm;
    const double windowPaddingMm = std::max(1.0, designFeature.widthMm);
    std::vector<cv::Point2d> measuredWindowSeries;
    measuredWindowSeries.reserve(measuredSeries.size());
    for (const cv::Point2d& point : measuredSeries) {
        if (point.x >= designFeature.leftSMm - windowPaddingMm &&
            point.x <= designFeature.rightSMm + windowPaddingMm) {
            measuredWindowSeries.push_back(point);
        }
    }

    SlotFeatureEstimate measuredFeature;
    if (config.designUseCentralSlotImageRoi && targetWidthMm > 1e-9) {
        measuredFeature = estimateMeasuredProfileSpanFeature(measuredWindowSeries,
                                                             "direct_roi_span_from_measured_profile");
        if (!measuredFeature.valid) {
            measuredFeature = estimateMeasuredProfileSpanFeature(measuredSeries,
                                                                 "direct_roi_span_from_measured_profile_global");
        }
    }
    if (!measuredFeature.valid && targetWidthMm > 1e-9) {
        measuredFeature = estimateTargetWidthSlotFeature(measuredWindowSeries,
                                                         "target_width_slot_from_measured_profile",
                                                         targetWidthMm,
                                                         targetDepthMm);
    }
    if (!measuredFeature.valid) {
        measuredFeature =
            estimatePrimarySlotFeature(measuredWindowSeries,
                                       "half_depth_primary_slot_from_measured_profile",
                                       designFeature.leftSMm - windowPaddingMm,
                                       designFeature.rightSMm + windowPaddingMm);
    }
    if (!measuredFeature.valid) {
        if (targetWidthMm > 1e-9) {
            measuredFeature = estimateTargetWidthSlotFeature(measuredSeries,
                                                             "target_width_slot_from_measured_profile_global",
                                                             targetWidthMm,
                                                             targetDepthMm);
        }
    }
    if (!measuredFeature.valid) {
        measuredFeature =
            estimateDominantLowRunSlotFeature(measuredSeries,
                                              "dominant_low_run_slot_from_measured_profile_global");
    }
    if (!measuredFeature.valid) {
        measuredFeature =
            estimatePrimarySlotFeature(std::move(measuredSeries),
                                       "half_depth_primary_slot_from_measured_profile_global");
    }
    if (targetWidthMm > 1e-9) {
        const SlotFeatureEstimate spanFeature =
            estimateTargetSpanSlotFeature(measuredWindowSeries,
                                          "target_span_slot_from_measured_profile",
                                          targetWidthMm,
                                          targetDepthMm);
        const double currentWidthError =
            measuredFeature.valid ? std::abs(measuredFeature.widthMm - targetWidthMm)
                                  : std::numeric_limits<double>::infinity();
        const double spanWidthError =
            spanFeature.valid ? std::abs(spanFeature.widthMm - targetWidthMm)
                              : std::numeric_limits<double>::infinity();
        if (spanFeature.valid && spanWidthError < currentWidthError) {
            measuredFeature = spanFeature;
        }
    }
    if (config.localSlotImageMode && targetWidthMm > 1e-9) {
        const double localScale =
            std::isfinite(config.localSlotPixelSizeScale) && config.localSlotPixelSizeScale > 0.0
                ? config.localSlotPixelSizeScale
                : 1.0;
        const double calibratedWidthMm =
            (config.localSlotBottomWidthMm > 1e-9 ? config.localSlotBottomWidthMm : targetWidthMm) *
            localScale;
        if (calibratedWidthMm > 1e-9) {
            if (!measuredFeature.valid) {
                measuredFeature = designFeature;
                measuredFeature.method = "local_slot_bottom_width_calibrated";
            } else {
                measuredFeature.method = "local_slot_bottom_width_calibrated";
            }
            measuredFeature.widthMm = calibratedWidthMm;
            if (!std::isfinite(measuredFeature.centerSMm)) {
                measuredFeature.centerSMm = designFeature.centerSMm;
            }
            measuredFeature.leftSMm = measuredFeature.centerSMm - 0.5 * calibratedWidthMm;
            measuredFeature.rightSMm = measuredFeature.centerSMm + 0.5 * calibratedWidthMm;
        }
    }

    const DesignProfileSample centerSample =
        interpolateDesignProfileSample(designSamples, designFeature.centerSMm);

    if (!measuredFeature.valid) {
        std::vector<std::string> row = {"primary_slot",
                                        "unavailable",
                                        "half_depth_primary_slot",
                                        csvTextCell(metadata.sourceType),
                                        csvTextCell(metadata.sourceName),
                                        csvTextCell(metadata.extractionMethod),
                                        csvCell(targetWidthMm, 6),
                                        csvCell(targetDepthMm, 6),
                                        csvCell(designFeature.leftSMm, 6),
                                        csvCell(designFeature.rightSMm, 6),
                                        csvCell(designFeature.centerSMm, 6),
                                        csvCell(designFeature.widthMm, 6),
                                        csvCell(designFeature.depthMm, 6),
                                        csvCell(designFeature.bottomSMm, 6),
                                        csvCell(designFeature.bottomRMm, 6),
                                        cadCoordinateCell(centerSample, centerSample.cadXMm),
                                        cadCoordinateCell(centerSample, centerSample.cadYMm),
                                        cadCoordinateCell(centerSample, centerSample.cadZMm)};
        row.resize(kFeatureCompensationCsvColumnCount - 1);
        row.push_back(csvTextCell("Measured profile does not contain a detectable corresponding slot feature."));
        appendFeatureCompensationRow(stream, std::move(row));
        return stream.str();
    }

    const double centerErrorUm = (measuredFeature.centerSMm - designFeature.centerSMm) * 1000.0;
    const double widthErrorUm = (measuredFeature.widthMm - targetWidthMm) * 1000.0;
    const double depthErrorUm = (measuredFeature.depthMm - targetDepthMm) * 1000.0;
    const double relativeWidthError =
        targetWidthMm > 1e-9 ? std::abs(measuredFeature.widthMm - targetWidthMm) / targetWidthMm : 0.0;
    const bool widthMismatch = relativeWidthError > 0.25;
    std::ostringstream note;
    if (widthMismatch) {
        note << "Mismatch: measured slot width differs from target/CAD width by "
             << formatDouble(relativeWidthError * 100.0, 2)
             << "%. This compensation is diagnostic only; verify that CAD, image ROI, and calibration match.";
    } else if (config.localSlotImageMode) {
        note << "Local slot photo mode: slot width is calibrated by the configured bottom width; CAD XYZ compensation remains based on the imported CAD coordinate frame.";
    } else {
        note << "Positive compensation means move/expand/deepen toward the target slot feature.";
    }

    appendFeatureCompensationRow(stream,
                                 {"primary_slot",
                                  widthMismatch ? "mismatch" : "ok",
                                  measuredFeature.method,
                                  csvTextCell(metadata.sourceType),
                                  csvTextCell(metadata.sourceName),
                                  csvTextCell(metadata.extractionMethod),
                                  csvCell(targetWidthMm, 6),
                                  csvCell(targetDepthMm, 6),
                                  csvCell(designFeature.leftSMm, 6),
                                  csvCell(designFeature.rightSMm, 6),
                                  csvCell(designFeature.centerSMm, 6),
                                  csvCell(designFeature.widthMm, 6),
                                  csvCell(designFeature.depthMm, 6),
                                  csvCell(designFeature.bottomSMm, 6),
                                  csvCell(designFeature.bottomRMm, 6),
                                  cadCoordinateCell(centerSample, centerSample.cadXMm),
                                  cadCoordinateCell(centerSample, centerSample.cadYMm),
                                  cadCoordinateCell(centerSample, centerSample.cadZMm),
                                  csvCell(measuredFeature.centerSMm, 6),
                                  csvCell(measuredFeature.widthMm, 6),
                                  csvCell(measuredFeature.depthMm, 6),
                                  csvCell(centerErrorUm, 6),
                                  csvCell(widthErrorUm, 6),
                                  csvCell(depthErrorUm, 6),
                                  csvCell(-centerErrorUm, 6),
                                  csvCell(-widthErrorUm, 6),
                                  csvCell(-depthErrorUm, 6),
                                  csvTextCell(note.str())});
    return stream.str();
}

std::size_t applySingleSlotCurveOutlierFilter(std::vector<DesignErrorProfilePoint>& points,
                                              const stitch::StitchPipelineConfig& config)
{
    if (!config.designEnableProfileOutlierFilter || points.size() < 9) {
        return 0;
    }

    std::vector<std::size_t> usedIndices;
    usedIndices.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const DesignErrorProfilePoint& point = points[index];
        if (point.isUsed &&
            std::isfinite(point.sAlignedMm) &&
            std::isfinite(point.rAlignedMm) &&
            std::isfinite(point.radialErrorUm)) {
            usedIndices.push_back(index);
        }
    }
    if (usedIndices.size() < 9) {
        return 0;
    }

    constexpr int kHalfWindow = 6;
    constexpr double kMadScale = 1.4826;
    constexpr double kMinLocalScaleUm = 8.0;
    constexpr double kMinOutlierAbsUm = 35.0;
    constexpr double kMinOutlierDeltaUm = 45.0;
    const double sigma = std::clamp(config.designProfileOutlierSigma, 2.0, 3.0);

    std::vector<bool> outlierFlags(usedIndices.size(), false);
    for (std::size_t localIndex = 0; localIndex < usedIndices.size(); ++localIndex) {
        const int begin = std::max(0, static_cast<int>(localIndex) - kHalfWindow);
        const int end = std::min(static_cast<int>(usedIndices.size()) - 1,
                                 static_cast<int>(localIndex) + kHalfWindow);
        std::vector<double> neighborErrorsUm;
        neighborErrorsUm.reserve(static_cast<std::size_t>(end - begin + 1));
        for (int neighbor = begin; neighbor <= end; ++neighbor) {
            if (neighbor == static_cast<int>(localIndex)) {
                continue;
            }
            const double radialErrorUm =
                points[usedIndices[static_cast<std::size_t>(neighbor)]].radialErrorUm;
            if (std::isfinite(radialErrorUm)) {
                neighborErrorsUm.push_back(radialErrorUm);
            }
        }
        if (neighborErrorsUm.size() < 5) {
            continue;
        }

        const double medianErrorUm = medianOfValues(neighborErrorsUm);
        std::vector<double> absDeviationsUm;
        absDeviationsUm.reserve(neighborErrorsUm.size());
        for (const double errorUm : neighborErrorsUm) {
            absDeviationsUm.push_back(std::abs(errorUm - medianErrorUm));
        }
        const double madUm = medianOfValues(std::move(absDeviationsUm));
        const double thresholdUm =
            std::max(kMinOutlierDeltaUm, sigma * std::max(kMadScale * madUm, kMinLocalScaleUm));
        const double currentErrorUm = points[usedIndices[localIndex]].radialErrorUm;
        if (std::abs(currentErrorUm) > kMinOutlierAbsUm &&
            std::abs(currentErrorUm - medianErrorUm) > thresholdUm) {
            outlierFlags[localIndex] = true;
        }
    }

    const std::size_t outlierCount =
        static_cast<std::size_t>(std::count(outlierFlags.begin(), outlierFlags.end(), true));
    if (outlierCount == 0 || usedIndices.size() - outlierCount < 5) {
        return 0;
    }

    for (std::size_t localIndex = 0; localIndex < usedIndices.size(); ++localIndex) {
        if (outlierFlags[localIndex]) {
            points[usedIndices[localIndex]].isUsed = false;
        }
    }
    return outlierCount;
}

bool applySingleSlotRoi(DesignAlignmentResult& result, const stitch::StitchPipelineConfig& config)
{
    if (!(config.designTargetSlotWidthMm > 1e-9) || result.profilePoints.size() < 5) {
        return false;
    }

    const auto collectWindow = [&result](const double minSMm, const double maxSMm) {
        std::vector<DesignErrorProfilePoint> points;
        points.reserve(result.profilePoints.size());
        for (const DesignErrorProfilePoint& point : result.profilePoints) {
            if (std::isfinite(point.sAlignedMm) &&
                point.sAlignedMm >= minSMm &&
                point.sAlignedMm <= maxSMm) {
                points.push_back(point);
            }
        }
        return points;
    };

    std::vector<cv::Point2d> measuredSeries;
    measuredSeries.reserve(result.profilePoints.size());
    for (const DesignErrorProfilePoint& point : result.profilePoints) {
        if (point.isUsed && std::isfinite(point.sAlignedMm) && std::isfinite(point.rAlignedMm)) {
            measuredSeries.emplace_back(point.sAlignedMm, point.rAlignedMm);
        }
    }

    std::vector<DesignErrorProfilePoint> filteredPoints;
    SlotFeatureEstimate feature;
    bool usedCadTargetWindow = false;
    double roiPaddingMm = 0.0;
    double roiMinSMm = std::numeric_limits<double>::quiet_NaN();
    double roiMaxSMm = std::numeric_limits<double>::quiet_NaN();

    const SlotFeatureEstimate designFeature = estimateConfiguredDesignSlotFeature(config);
    if (designFeature.valid) {
        for (const double paddingScale : {0.20, 0.35, 0.50, 1.00}) {
            roiPaddingMm = std::max({config.designTargetSlotWidthMm * paddingScale,
                                     designFeature.widthMm * paddingScale,
                                     config.designPixelSizeMm * 2.0,
                                     0.10});
            roiMinSMm = designFeature.leftSMm - roiPaddingMm;
            roiMaxSMm = designFeature.rightSMm + roiPaddingMm;
            filteredPoints = collectWindow(roiMinSMm, roiMaxSMm);
            if (filteredPoints.size() >= 5) {
                feature = designFeature;
                usedCadTargetWindow = true;
                break;
            }
        }
    }

    if (!usedCadTargetWindow && config.designTargetSlotWidthMm > 1e-9) {
        feature = estimateTargetWidthSlotFeature(measuredSeries,
                                                 "single_slot_roi_from_target_width",
                                                 config.designTargetSlotWidthMm,
                                                 config.designTargetSlotDepthMm);
    }
    if (!usedCadTargetWindow && !feature.valid) {
        feature = estimatePrimarySlotFeature(std::move(measuredSeries), "single_slot_roi_from_measured_profile");
    }
    if (!feature.valid) {
        return false;
    }

    if (!usedCadTargetWindow) {
        roiMinSMm = feature.leftSMm;
        roiMaxSMm = feature.rightSMm;
        for (const double paddingScale : {0.20, 0.50, 1.00, 2.00}) {
            roiPaddingMm = std::max({config.designTargetSlotWidthMm * paddingScale,
                                     feature.widthMm * paddingScale,
                                     config.designPixelSizeMm,
                                     1e-6});
            roiMinSMm = feature.leftSMm - roiPaddingMm;
            roiMaxSMm = feature.rightSMm + roiPaddingMm;
            filteredPoints = collectWindow(roiMinSMm, roiMaxSMm);
            if (filteredPoints.size() >= 5) {
                break;
            }
        }
    }
    if (filteredPoints.size() < 5) {
        filteredPoints = result.profilePoints;
        const double centerSMm = feature.centerSMm;
        std::stable_sort(filteredPoints.begin(), filteredPoints.end(), [centerSMm](const auto& lhs, const auto& rhs) {
            return std::abs(lhs.sAlignedMm - centerSMm) < std::abs(rhs.sAlignedMm - centerSMm);
        });
        filteredPoints.resize(std::min<std::size_t>(filteredPoints.size(), 24));
        std::stable_sort(filteredPoints.begin(), filteredPoints.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.sAlignedMm < rhs.sAlignedMm;
        });
        if (filteredPoints.size() < 5) {
            return false;
        }
        roiMinSMm = filteredPoints.front().sAlignedMm;
        roiMaxSMm = filteredPoints.back().sAlignedMm;
    }

    result.profilePoints = std::move(filteredPoints);
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        point.isUsed =
            std::isfinite(point.sAlignedMm) &&
            std::isfinite(point.rAlignedMm) &&
            std::isfinite(point.normalErrorUm);
    }

    std::vector<double> normalErrorsUm;
    std::vector<double> profileErrorsUm;
    normalErrorsUm.reserve(result.profilePoints.size());
    profileErrorsUm.reserve(result.profilePoints.size());
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        if (point.isUsed && std::isfinite(point.normalErrorUm)) {
            normalErrorsUm.push_back(point.normalErrorUm);
        }
    }

    const ErrorStats normalStats = buildErrorStats(normalErrorsUm);
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        if (!point.isUsed || !std::isfinite(point.normalErrorUm)) {
            continue;
        }
        point.profileErrorUm = point.normalErrorUm - normalStats.meanUm;
        if (std::isfinite(point.profileErrorUm)) {
            profileErrorsUm.push_back(point.profileErrorUm);
        }
    }
    const ErrorStats profileStats =
        profileErrorsUm.empty() ? normalStats : buildErrorStats(profileErrorsUm);
    const std::size_t usedCount = static_cast<std::size_t>(
        std::count_if(result.profilePoints.begin(),
                      result.profilePoints.end(),
                      [](const auto& point) {
                          return point.isUsed;
                      }));
    result.summary.candidateCount = result.profilePoints.size();
    result.summary.usedCount = usedCount;
    result.summary.outlierCount = result.summary.candidateCount - result.summary.usedCount;
    result.summary.outlierRatio = result.summary.candidateCount > 0
                                      ? static_cast<double>(result.summary.outlierCount) /
                                            static_cast<double>(result.summary.candidateCount)
                                      : 0.0;
    result.summary.normalStats = normalStats;
    result.summary.profileStats = profileStats;
    result.summary.absoluteAllStats = normalStats;
    result.summary.absoluteFilteredStats = normalStats;
    result.summary.meanNormalErrorUm = normalStats.meanUm;

    std::ostringstream message;
    message << result.message << " Single-slot ROI applied: s=["
            << formatDouble(roiMinSMm, 6) << ", " << formatDouble(roiMaxSMm, 6)
            << "] mm, "
            << (usedCadTargetWindow ? "cad_target_width=" : "measured_width=")
            << formatDouble(feature.widthMm, 6) << " mm.";
    result.message = message.str();
    return true;
}

std::size_t countUsedCadCoordinatePoints(const std::vector<DesignErrorProfilePoint>& points)
{
    return static_cast<std::size_t>(std::count_if(points.begin(), points.end(), [](const auto& point) {
        return point.isUsed && point.hasCadCoordinates;
    }));
}

void refreshSummaryStatsFromProfilePoints(DesignAlignmentResult& result)
{
    std::vector<double> normalErrorsUm;
    normalErrorsUm.reserve(result.profilePoints.size());
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        if (point.isUsed && std::isfinite(point.normalErrorUm)) {
            normalErrorsUm.push_back(point.normalErrorUm);
        }
    }

    const ErrorStats normalStats = buildErrorStats(normalErrorsUm);
    std::vector<double> profileErrorsUm;
    profileErrorsUm.reserve(normalErrorsUm.size());
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        if (!point.isUsed || !std::isfinite(point.normalErrorUm)) {
            continue;
        }
        point.profileErrorUm = point.normalErrorUm - normalStats.meanUm;
        if (std::isfinite(point.profileErrorUm)) {
            profileErrorsUm.push_back(point.profileErrorUm);
        }
    }

    result.summary.candidateCount = result.profilePoints.size();
    result.summary.usedCount = static_cast<std::size_t>(std::count_if(result.profilePoints.begin(),
                                                                      result.profilePoints.end(),
                                                                      [](const auto& point) {
                                                                          return point.isUsed;
                                                                      }));
    result.summary.outlierCount = result.summary.candidateCount - result.summary.usedCount;
    result.summary.outlierRatio = result.summary.candidateCount > 0
                                      ? static_cast<double>(result.summary.outlierCount) /
                                            static_cast<double>(result.summary.candidateCount)
                                      : 0.0;
    result.summary.normalStats = normalStats;
    result.summary.profileStats = profileErrorsUm.empty() ? normalStats : buildErrorStats(profileErrorsUm);
    result.summary.absoluteAllStats = normalStats;
    result.summary.absoluteFilteredStats = normalStats;
    result.summary.meanNormalErrorUm = normalStats.meanUm;
}

bool applySingleSlotCadFrameMapping(DesignAlignmentResult& result,
                                    const stitch::StitchPipelineConfig& config,
                                    const std::vector<DesignPolylinePoint>& activeDesignPolyline,
                                    const DesignProfileMetadata& metadata)
{
    if (!(config.designTargetSlotWidthMm > 1e-9) ||
        metadata.sourceType != "external_cad" ||
        activeDesignPolyline.size() < 2 ||
        result.profilePoints.size() < 5 ||
        countUsedCadCoordinatePoints(result.profilePoints) > 0) {
        return false;
    }

    std::vector<cv::Point2d> measuredSeries;
    measuredSeries.reserve(result.profilePoints.size());
    for (const DesignErrorProfilePoint& point : result.profilePoints) {
        if (point.isUsed && std::isfinite(point.sAlignedMm) && std::isfinite(point.rAlignedMm)) {
            measuredSeries.emplace_back(point.sAlignedMm, point.rAlignedMm);
        }
    }
    SlotFeatureEstimate measuredFeature;
    if (config.designTargetSlotWidthMm > 1e-9) {
        measuredFeature = estimateTargetWidthSlotFeature(measuredSeries,
                                                         "single_slot_roi_for_cad_frame_mapping_target_width",
                                                         config.designTargetSlotWidthMm,
                                                         config.designTargetSlotDepthMm);
    }
    if (!measuredFeature.valid) {
        measuredFeature =
            estimatePrimarySlotFeature(std::move(measuredSeries), "single_slot_roi_for_cad_frame_mapping");
    }
    if (!measuredFeature.valid) {
        return false;
    }

    std::vector<cv::Point2d> designSeries;
    designSeries.reserve(activeDesignPolyline.size());
    for (const DesignPolylinePoint& point : activeDesignPolyline) {
        if (std::isfinite(point.sMm) && std::isfinite(point.rMm)) {
            designSeries.emplace_back(point.sMm, point.rMm);
        }
    }
    const SlotFeatureEstimate designFeature =
        estimateTargetWidthSlotFeature(std::move(designSeries),
                                       "target_width_slot_from_cad_profile",
                                       config.designTargetSlotWidthMm,
                                       config.designTargetSlotDepthMm);
    if (!designFeature.valid) {
        return false;
    }

    const double targetDepthMm = designFeature.depthMm > 1e-9
                                     ? designFeature.depthMm
                                     : config.designTargetSlotDepthMm;
    const double radialScale =
        measuredFeature.depthMm > 1e-9 && targetDepthMm > 1e-9
            ? targetDepthMm / measuredFeature.depthMm
            : 1.0;
    const double sPaddingMm = std::max(0.5, config.designTargetSlotWidthMm * 0.5);
    const double rPaddingMm = std::max(0.25, targetDepthMm * 0.25);
    const double minMappedSMm = designFeature.leftSMm - sPaddingMm;
    const double maxMappedSMm = designFeature.rightSMm + sPaddingMm;
    const double minMappedRMm = designFeature.bottomRMm - rPaddingMm;
    const double maxMappedRMm = designFeature.topRMm + rPaddingMm;
    const double compensationMatchLimitMm =
        std::max({config.designPixelSizeMm * 8.0,
                  config.designTargetSlotWidthMm * 0.08,
                  0.38});

    std::size_t mappedCount = 0;
    for (DesignErrorProfilePoint& point : result.profilePoints) {
        point.hasCadCoordinates = false;
        point.nearestDesignCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.nearestDesignCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.measuredCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadXMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadYMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationTargetCadZMm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadXUm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadYUm = std::numeric_limits<double>::quiet_NaN();
        point.compensationDeltaCadZUm = std::numeric_limits<double>::quiet_NaN();

        if (!point.isUsed || !std::isfinite(point.sAlignedMm) || !std::isfinite(point.rAlignedMm)) {
            continue;
        }

        const double mappedSMm = std::clamp(designFeature.leftSMm +
                                                (point.sAlignedMm - measuredFeature.leftSMm),
                                            minMappedSMm,
                                            maxMappedSMm);
        const double mappedRMm = std::clamp(designFeature.bottomRMm +
                                                (point.rAlignedMm - measuredFeature.bottomRMm) * radialScale,
                                            minMappedRMm,
                                            maxMappedRMm);
        const NearestDesignPoint nearest = findNearestDesignPoint(activeDesignPolyline, mappedSMm, mappedRMm);
        if (!nearest.valid || !nearest.hasCadPoint) {
            point.isUsed = false;
            continue;
        }

        const CadPoint measuredCad = mapProfilePointToCadCoordinates(mappedSMm, mappedRMm, nearest, metadata);
        if (!measuredCad.valid) {
            point.isUsed = false;
            continue;
        }

        point.sAlignedMm = mappedSMm;
        point.rAlignedMm = mappedRMm;
        point.designRadiusMm = nearest.rMm;
        point.nearestDesignSMm = nearest.sMm;
        point.nearestDesignRMm = nearest.rMm;
        point.hasCadCoordinates = true;
        point.nearestDesignCadXMm = nearest.cadXMm;
        point.nearestDesignCadYMm = nearest.cadYMm;
        point.nearestDesignCadZMm = nearest.cadZMm;
        point.measuredCadXMm = measuredCad.xMm;
        point.measuredCadYMm = measuredCad.yMm;
        point.measuredCadZMm = measuredCad.zMm;
        point.compensationTargetCadXMm = nearest.cadXMm;
        point.compensationTargetCadYMm = nearest.cadYMm;
        point.compensationTargetCadZMm = nearest.cadZMm;
        point.compensationDeltaCadXUm = (nearest.cadXMm - measuredCad.xMm) * 1000.0;
        point.compensationDeltaCadYUm = (nearest.cadYMm - measuredCad.yMm) * 1000.0;
        point.compensationDeltaCadZUm = (nearest.cadZMm - measuredCad.zMm) * 1000.0;
        point.designSegmentIndex = nearest.segmentIndex;
        point.designDerivative = std::abs(nearest.tangentDs) < 1e-12
                                      ? std::numeric_limits<double>::infinity()
                                      : nearest.tangentDr / nearest.tangentDs;
        point.radialErrorMm = mappedRMm - nearest.rMm;
        point.radialErrorUm = point.radialErrorMm * 1000.0;
        point.signedDistanceMm = nearest.signedDistanceMm;
        point.signedDistanceUm = point.signedDistanceMm * 1000.0;
        point.normalErrorMm = point.signedDistanceMm;
        point.normalErrorUm = point.signedDistanceUm;
        point.legacyNormalErrorUm = point.normalErrorUm;
        point.isUsed = std::hypot(mappedSMm - nearest.sMm, mappedRMm - nearest.rMm) <= compensationMatchLimitMm;
        ++mappedCount;
    }

    if (mappedCount < 5) {
        return false;
    }

    const std::size_t slotCurveOutlierCount = applySingleSlotCurveOutlierFilter(result.profilePoints, config);
    refreshSummaryStatsFromProfilePoints(result);
    std::ostringstream message;
    message << result.message
            << " CAD-frame single-slot mapping applied: cad_slot_s=["
            << formatDouble(designFeature.leftSMm, 6) << ", "
            << formatDouble(designFeature.rightSMm, 6) << "] mm, cad_slot_width="
            << formatDouble(designFeature.widthMm, 6) << " mm, slot_curve_outliers="
            << slotCurveOutlierCount << ".";
    result.message = message.str();
    return true;
}

stitch::StitchPipelineConfig effectiveSingleSlotTargetConfig(const stitch::StitchPipelineConfig& config,
                                                             std::string& note)
{
    note.clear();
    if (config.designTargetSlotWidthMm > 1e-9) {
        return config;
    }

    std::vector<cv::Point2d> designSeries;
    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(config);
    designSeries.reserve(activeDesignPolyline.size());
    for (const DesignPolylinePoint& point : activeDesignPolyline) {
        if (std::isfinite(point.sMm) && std::isfinite(point.rMm)) {
            designSeries.emplace_back(point.sMm, point.rMm);
        }
    }

    SlotFeatureEstimate inferredFeature =
        estimateDominantLowRunSlotFeature(designSeries, "inferred_single_slot_low_run_from_cad_profile");
    if (!inferredFeature.valid) {
        inferredFeature =
            estimatePrimarySlotFeature(std::move(designSeries), "inferred_single_slot_from_cad_profile");
    }
    if (!inferredFeature.valid || !(inferredFeature.widthMm > 1e-9)) {
        return config;
    }

    stitch::StitchPipelineConfig effectiveConfig = config;
    effectiveConfig.designTargetSlotWidthMm = inferredFeature.widthMm;
    if (!(effectiveConfig.designTargetSlotDepthMm > 1e-9) &&
        inferredFeature.depthMm > 1e-9) {
        effectiveConfig.designTargetSlotDepthMm = inferredFeature.depthMm;
    }

    std::ostringstream stream;
    stream << "Inferred single-slot target from CAD profile: width="
           << formatDouble(effectiveConfig.designTargetSlotWidthMm, 6)
           << " mm";
    if (effectiveConfig.designTargetSlotDepthMm > 1e-9) {
        stream << ", depth=" << formatDouble(effectiveConfig.designTargetSlotDepthMm, 6)
               << " mm";
    }
    stream << ".";
    note = stream.str();
    return effectiveConfig;
}

MeasuredGeneratrixData buildCadGuidedSingleSlotBottomGeneratrix(
    const std::vector<StitchedContourSample>& samples,
    const MeasuredGeneratrixData& referenceGeneratrix,
    const stitch::StitchPipelineConfig& config)
{
    MeasuredGeneratrixData data;
    if (samples.empty() ||
        referenceGeneratrix.points.empty() ||
        !(config.designTargetSlotWidthMm > 1e-9) ||
        !(config.designPixelSizeMm > 1e-12) ||
        !hasExternalDesignProfile(config) ||
        !std::isfinite(referenceGeneratrix.anchorXPx) ||
        !std::isfinite(referenceGeneratrix.anchorYPx)) {
        return data;
    }

    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(config);
    if (activeDesignPolyline.size() < 2) {
        return data;
    }

    const SlotFeatureEstimate designFeature = estimateConfiguredDesignSlotFeature(config);
    if (!designFeature.valid || !(designFeature.depthMm > 1e-9)) {
        return data;
    }

    const DesignEval leftDesign = evalDesignProfile(0.0, config);
    const double anchorRadiusMm = radiusFromYPx(referenceGeneratrix.anchorYPx, config);
    const double anchorDrMm =
        leftDesign.valid && config.designAnchorRadialToLeftEndpoint ? (leftDesign.r_mm - anchorRadiusMm) : 0.0;

    const double sPaddingMm = std::max({config.designTargetSlotWidthMm * 0.18,
                                        config.designPixelSizeMm * 4.0,
                                        0.18});
    const double rPaddingMm = std::max({designFeature.depthMm * 1.05,
                                        config.designPixelSizeMm * 10.0,
                                        0.80});
    const double minSMm = designFeature.leftSMm - sPaddingMm;
    const double maxSMm = designFeature.rightSMm + sPaddingMm;
    const double minRMm = designFeature.bottomRMm - rPaddingMm;
    const double maxRMm = designFeature.topRMm + rPaddingMm;
    double minSampleXPx = std::numeric_limits<double>::infinity();
    double maxSampleXPx = -std::numeric_limits<double>::infinity();
    for (const StitchedContourSample& sample : samples) {
        if (!std::isfinite(sample.xPx)) {
            continue;
        }
        minSampleXPx = std::min(minSampleXPx, sample.xPx);
        maxSampleXPx = std::max(maxSampleXPx, sample.xPx);
    }

    std::vector<double> anchorXPxCandidates;
    const auto addAnchorCandidate = [&anchorXPxCandidates](const double anchorXPx) {
        if (!std::isfinite(anchorXPx)) {
            return;
        }
        const auto alreadyAdded = std::any_of(anchorXPxCandidates.begin(),
                                              anchorXPxCandidates.end(),
                                              [anchorXPx](const double existing) {
                                                  return std::abs(existing - anchorXPx) < 0.5;
                                              });
        if (!alreadyAdded) {
            anchorXPxCandidates.push_back(anchorXPx);
        }
    };

    addAnchorCandidate(referenceGeneratrix.anchorXPx);
    if (std::isfinite(minSampleXPx) && std::isfinite(maxSampleXPx)) {
        addAnchorCandidate(minSampleXPx);
        addAnchorCandidate(minSampleXPx - designFeature.leftSMm / config.designPixelSizeMm);
        addAnchorCandidate(maxSampleXPx - designFeature.rightSMm / config.designPixelSizeMm);
        addAnchorCandidate(0.5 * (minSampleXPx + maxSampleXPx) -
                           designFeature.centerSMm / config.designPixelSizeMm);
        if (std::isfinite(config.designProfileMetadata.maxSMm) &&
            config.designProfileMetadata.maxSMm > 1e-9) {
            addAnchorCandidate(maxSampleXPx -
                               config.designProfileMetadata.maxSMm / config.designPixelSizeMm);
        }
    }

    std::vector<CadGuidedSlotCandidate> acceptedCandidates;
    std::vector<CadGuidedSlotCandidate> referenceAnchorCandidates;
    double bestAnchorXPx = referenceGeneratrix.anchorXPx;
    double bestAnchorScore = -std::numeric_limits<double>::infinity();
    for (const double anchorXPxCandidate : anchorXPxCandidates) {
        for (const double distanceLimitMm : {0.18, 0.32, 0.55, 0.85, 1.20, 2.00, 3.20, 5.00, 7.00}) {
            std::vector<CadGuidedSlotCandidate> candidates;
            candidates.reserve(samples.size());
            for (const StitchedContourSample& sample : samples) {
                if (!std::isfinite(sample.xPx) || !std::isfinite(sample.yPx)) {
                    continue;
                }

                const double sMm = (sample.xPx - anchorXPxCandidate) * config.designPixelSizeMm;
                if (!std::isfinite(sMm) || sMm < minSMm || sMm > maxSMm) {
                    continue;
                }

                const double rMm = radiusFromYPx(sample.yPx, config) + anchorDrMm;
                if (!std::isfinite(rMm) || rMm < minRMm || rMm > maxRMm) {
                    continue;
                }

                const NearestDesignPoint nearest = findNearestDesignPoint(activeDesignPolyline, sMm, rMm);
                if (!nearest.valid ||
                    nearest.sMm < minSMm ||
                    nearest.sMm > maxSMm ||
                    nearest.rMm < minRMm ||
                    nearest.rMm > maxRMm) {
                    continue;
                }

                const double euclideanDistanceMm = std::hypot(sMm - nearest.sMm, rMm - nearest.rMm);
                const double normalDistanceMm = std::abs(nearest.signedDistanceMm);
                const double scoreMm = std::max(euclideanDistanceMm, normalDistanceMm);
                if (!std::isfinite(scoreMm) || scoreMm > distanceLimitMm) {
                    continue;
                }

                CadGuidedSlotCandidate candidate;
                candidate.sMm = sMm;
                candidate.rMm = rMm;
                candidate.scoreMm = scoreMm;
                candidate.point.index = 0;
                candidate.point.supportCount = 1;
                candidate.point.xPx = sample.xPx;
                candidate.point.yPx = sample.yPx;
                candidate.point.yStdPx = 0.0;
                candidate.point.slopeAbs = 0.0;
                candidate.point.sBaseMm = sMm;
                candidate.point.rRawMm = radiusFromYPx(sample.yPx, config);
                candidate.point.slopeAccepted = true;
                candidate.point.trimAccepted = true;
                candidates.push_back(candidate);
            }
            if (candidates.empty()) {
                continue;
            }
            if (std::abs(anchorXPxCandidate - referenceGeneratrix.anchorXPx) < 0.5 &&
                candidates.size() >= kMinSingleSlotTargetPointCount &&
                candidates.size() > referenceAnchorCandidates.size()) {
                referenceAnchorCandidates = candidates;
            }

            double acceptedMinSMm = std::numeric_limits<double>::infinity();
            double acceptedMaxSMm = -std::numeric_limits<double>::infinity();
            double averageScoreMm = 0.0;
            for (const CadGuidedSlotCandidate& candidate : candidates) {
                acceptedMinSMm = std::min(acceptedMinSMm, candidate.sMm);
                acceptedMaxSMm = std::max(acceptedMaxSMm, candidate.sMm);
                averageScoreMm += candidate.scoreMm;
            }
            averageScoreMm /= static_cast<double>(candidates.size());
            const double spanCoverage =
                designFeature.widthMm > 1e-9
                    ? std::clamp((acceptedMaxSMm - acceptedMinSMm) / designFeature.widthMm, 0.0, 1.0)
                    : 0.0;
            const double anchorScore =
                static_cast<double>(candidates.size()) +
                120.0 * spanCoverage -
                8.0 * averageScoreMm -
                2.0 * distanceLimitMm;
            if (anchorScore > bestAnchorScore) {
                bestAnchorScore = anchorScore;
                bestAnchorXPx = anchorXPxCandidate;
                acceptedCandidates = std::move(candidates);
            }
        }
    }
    if (referenceAnchorCandidates.size() >= kMinSingleSlotTargetPointCount) {
        acceptedCandidates = std::move(referenceAnchorCandidates);
        bestAnchorXPx = referenceGeneratrix.anchorXPx;
    }

    if (acceptedCandidates.size() < kMinSingleSlotTargetPointCount) {
        return data;
    }

    std::stable_sort(acceptedCandidates.begin(),
                     acceptedCandidates.end(),
                     [](const CadGuidedSlotCandidate& lhs, const CadGuidedSlotCandidate& rhs) {
                         if (lhs.sMm != rhs.sMm) {
                             return lhs.sMm < rhs.sMm;
                         }
                         return lhs.rMm < rhs.rMm;
                     });

    data.points.reserve(acceptedCandidates.size());
    double maxBaseMm = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < acceptedCandidates.size(); ++index) {
        GeneratrixCandidatePoint point = acceptedCandidates[index].point;
        point.index = index + 1;
        maxBaseMm = std::max(maxBaseMm, point.sBaseMm);
        data.points.push_back(point);
    }

    data.anchorPointIndex = 0;
    data.primaryRunBeginIndex = 0;
    data.primaryRunEndIndex = data.points.empty() ? 0 : data.points.size() - 1;
    data.anchorXPx = bestAnchorXPx;
    data.anchorYPx = referenceGeneratrix.anchorYPx;
    data.anchorMeasuredRadiusMm = anchorRadiusMm;
    data.anchorBaseMm = 0.0;
    data.maxBaseMm = std::isfinite(maxBaseMm) ? maxBaseMm : 0.0;
    if (!data.points.empty()) {
        PointRun run;
        run.beginIndex = 0;
        run.endIndex = data.points.size() - 1;
        run.pointCount = data.points.size();
        run.lengthMm = data.points.back().sBaseMm - data.points.front().sBaseMm;
        data.runs.push_back(run);
    }

    return data;
}

DesignAlignmentResult buildDirectCentralSlotRoiResult(const MeasuredGeneratrixData& generatrix,
                                                      const stitch::StitchPipelineConfig& config,
                                                      const std::string& message)
{
    DesignAlignmentResult result;
    std::string inferredTargetNote;
    const stitch::StitchPipelineConfig effectiveConfig =
        effectiveSingleSlotTargetConfig(config, inferredTargetNote);
    if (!effectiveConfig.designUseCentralSlotImageRoi ||
        !hasExternalDesignProfile(effectiveConfig) ||
        !(effectiveConfig.designTargetSlotWidthMm > 1e-9) ||
        !(effectiveConfig.designPixelSizeMm > 1e-12) ||
        generatrix.points.size() < 5) {
        result.message = message + " Direct ROI mode skipped: central slot ROI is not fully configured.";
        return result;
    }

    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(effectiveConfig);
    if (activeDesignPolyline.size() < 2) {
        result.message = message + " Direct ROI mode failed: CAD design polyline is empty.";
        return result;
    }

    const SlotFeatureEstimate designFeature = estimateConfiguredDesignSlotFeature(effectiveConfig);
    if (!designFeature.valid || !(designFeature.widthMm > 1e-9)) {
        result.message = message + " Direct ROI mode failed: CAD local slot feature was not found.";
        return result;
    }

    std::vector<const GeneratrixCandidatePoint*> sourcePoints;
    sourcePoints.reserve(generatrix.points.size());
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(generatrix.points.size());
    ys.reserve(generatrix.points.size());
    for (const GeneratrixCandidatePoint& point : generatrix.points) {
        if (std::isfinite(point.xPx) && std::isfinite(point.yPx)) {
            sourcePoints.push_back(&point);
            xs.push_back(point.xPx);
            ys.push_back(point.yPx);
        }
    }
    if (sourcePoints.size() < 5) {
        result.message = message + " Direct ROI mode failed: fewer than five ROI edge points are available.";
        return result;
    }

    std::stable_sort(sourcePoints.begin(),
                     sourcePoints.end(),
                     [](const auto* lhs, const auto* rhs) {
                         if (lhs->xPx != rhs->xPx) {
                             return lhs->xPx < rhs->xPx;
                         }
                         return lhs->yPx < rhs->yPx;
                     });

    const double imageLeftXPx = percentileValue(xs, 0.01);
    const double imageRightXPx = percentileValue(xs, 0.99);
    const double imageTopYPx = percentileValue(ys, 0.05);
    const double imageBottomYPx = percentileValue(ys, 0.95);
    if (!(imageRightXPx - imageLeftXPx > 1e-6) ||
        !(imageBottomYPx - imageTopYPx > 1e-6)) {
        result.message = message + " Direct ROI mode failed: ROI edge span is degenerate.";
        return result;
    }

    const double designTopRMm = std::max(designFeature.topRMm, designFeature.bottomRMm);
    const double designBottomRMm = std::min(designFeature.topRMm, designFeature.bottomRMm);
    const double designDepthMm = designTopRMm - designBottomRMm;
    if (!(designDepthMm > 1e-9)) {
        result.message = message + " Direct ROI mode failed: CAD local slot depth is degenerate.";
        return result;
    }

    const double sPaddingMm = std::max({effectiveConfig.designTargetSlotWidthMm * 0.15,
                                        effectiveConfig.designPixelSizeMm * 4.0,
                                        0.15});
    const double rPaddingMm = std::max({designDepthMm * 0.20,
                                        effectiveConfig.designPixelSizeMm * 6.0,
                                        0.20});
    std::vector<DesignPolylinePoint> slotDesignPolyline;
    slotDesignPolyline.reserve(activeDesignPolyline.size());
    for (const DesignPolylinePoint& point : activeDesignPolyline) {
        if (point.sMm >= designFeature.leftSMm - sPaddingMm &&
            point.sMm <= designFeature.rightSMm + sPaddingMm &&
            point.rMm >= designBottomRMm - rPaddingMm &&
            point.rMm <= designTopRMm + rPaddingMm) {
            slotDesignPolyline.push_back(point);
        }
    }
    if (slotDesignPolyline.size() < 2) {
        slotDesignPolyline = activeDesignPolyline;
    }

    result.ok = true;
    result.message = inferredTargetNote.empty() ? message : (message + " " + inferredTargetNote);
    result.summary.dzMm = 0.0;
    result.summary.drMm = 0.0;
    result.summary.designReverseAxial = effectiveDesignReverseAxial(effectiveConfig);
    result.summary.useLeftEndpointAnchor = false;
    result.summary.evaluateProfileForm = effectiveConfig.designEvaluateProfileForm;
    result.summary.anchorXPx = imageLeftXPx;
    result.summary.anchorYPx = imageTopYPx;
    result.summary.pixelSizeMm = effectiveConfig.designPixelSizeMm;
    result.summary.designProfileMetadata = buildActiveDesignProfileMetadata(effectiveConfig, activeDesignPolyline);
    result.designProfileSamples = toPublicDesignProfileSamples(activeDesignPolyline);

    const DesignProfileMetadata& metadata = result.summary.designProfileMetadata;
    result.profilePoints.reserve(sourcePoints.size());
    std::size_t outputIndex = 0;
    for (const GeneratrixCandidatePoint* inputPoint : sourcePoints) {
        const double xNorm = std::clamp((inputPoint->xPx - imageLeftXPx) /
                                            (imageRightXPx - imageLeftXPx),
                                        0.0,
                                        1.0);
        const double yNorm = std::clamp((inputPoint->yPx - imageTopYPx) /
                                            (imageBottomYPx - imageTopYPx),
                                        0.0,
                                        1.0);
        const double mappedSMm = designFeature.leftSMm + xNorm * designFeature.widthMm;
        const double mappedRMm = designTopRMm - yNorm * designDepthMm;
        const NearestDesignPoint nearest = findNearestDesignPoint(slotDesignPolyline, mappedSMm, mappedRMm);
        if (!nearest.valid || !nearest.hasCadPoint) {
            continue;
        }
        const CadPoint measuredCad = mapProfilePointToCadCoordinates(mappedSMm, mappedRMm, nearest, metadata);
        if (!measuredCad.valid) {
            continue;
        }

        DesignErrorProfilePoint point;
        point.index = outputIndex++;
        point.supportCount = inputPoint->supportCount;
        point.xPx = inputPoint->xPx;
        point.yPx = inputPoint->yPx;
        point.yStdPx = inputPoint->yStdPx;
        point.sAlignedMm = mappedSMm;
        point.rAlignedMm = mappedRMm;
        point.designRadiusMm = nearest.rMm;
        point.nearestDesignSMm = nearest.sMm;
        point.nearestDesignRMm = nearest.rMm;
        point.hasCadCoordinates = true;
        point.nearestDesignCadXMm = nearest.cadXMm;
        point.nearestDesignCadYMm = nearest.cadYMm;
        point.nearestDesignCadZMm = nearest.cadZMm;
        point.measuredCadXMm = measuredCad.xMm;
        point.measuredCadYMm = measuredCad.yMm;
        point.measuredCadZMm = measuredCad.zMm;
        point.compensationTargetCadXMm = nearest.cadXMm;
        point.compensationTargetCadYMm = nearest.cadYMm;
        point.compensationTargetCadZMm = nearest.cadZMm;
        point.compensationDeltaCadXUm = (nearest.cadXMm - measuredCad.xMm) * 1000.0;
        point.compensationDeltaCadYUm = (nearest.cadYMm - measuredCad.yMm) * 1000.0;
        point.compensationDeltaCadZUm = (nearest.cadZMm - measuredCad.zMm) * 1000.0;
        point.designSegmentIndex = nearest.segmentIndex;
        point.designDerivative = std::abs(nearest.tangentDs) < 1e-12
                                     ? std::numeric_limits<double>::infinity()
                                     : nearest.tangentDr / nearest.tangentDs;
        point.radialErrorMm = mappedRMm - nearest.rMm;
        point.radialErrorUm = point.radialErrorMm * 1000.0;
        point.signedDistanceMm = nearest.signedDistanceMm;
        point.signedDistanceUm = point.signedDistanceMm * 1000.0;
        point.normalErrorMm = point.signedDistanceMm;
        point.normalErrorUm = point.signedDistanceUm;
        point.legacyNormalErrorUm = point.normalErrorUm;
        point.profileErrorUm = point.normalErrorUm;
        point.isUsed = true;
        result.profilePoints.push_back(point);
    }

    if (result.profilePoints.size() < 5) {
        result = {};
        result.message = message + " Direct ROI mode failed: fewer than five CAD-mapped ROI points remained.";
        return result;
    }

    const std::size_t slotCurveOutlierCount = applySingleSlotCurveOutlierFilter(result.profilePoints, effectiveConfig);
    refreshSummaryStatsFromProfilePoints(result);
    result.summary.designProfileMetadata = metadata;
    result.profileCsvText = buildProfileCsv(result.profilePoints);
    result.summaryCsvText = buildSummaryCsv(result.summary);
    result.error3dCsvText = build3dErrorCsv(result.profilePoints, metadata);
    result.compensationCsvText = buildCompensationCsv(result.profilePoints, metadata);
    result.featureCompensationCsvText = buildFeatureCompensationCsv(result.designProfileSamples,
                                                                    result.profilePoints,
                                                                    metadata,
                                                                    effectiveConfig);

    std::ostringstream finalMessage;
    finalMessage << result.message
                 << " Direct central ROI slot mapping applied: image_x=["
                 << formatDouble(imageLeftXPx, 3) << ", "
                 << formatDouble(imageRightXPx, 3) << "] px -> cad_slot_s=["
                 << formatDouble(designFeature.leftSMm, 6) << ", "
                 << formatDouble(designFeature.rightSMm, 6) << "] mm, used_points="
                 << result.summary.usedCount << ", slot_curve_outliers="
                 << slotCurveOutlierCount << ".";
    result.message = finalMessage.str();
    return result;
}

DesignAlignmentResult buildSingleSlotTargetFallbackResult(const MeasuredGeneratrixData& generatrix,
                                                          const stitch::StitchPipelineConfig& config,
                                                          const std::string& message)
{
    DesignAlignmentResult result;
    std::string inferredTargetNote;
    const stitch::StitchPipelineConfig effectiveConfig =
        effectiveSingleSlotTargetConfig(config, inferredTargetNote);
    if (!(effectiveConfig.designTargetSlotWidthMm > 1e-9)) {
        result.message =
            message + " Single-slot mode skipped: target slot width is not configured and could not be inferred from CAD.";
        return result;
    }

    std::vector<const GeneratrixCandidatePoint*> selectedPoints;
    selectedPoints.reserve(generatrix.points.size());
    for (const GeneratrixCandidatePoint& point : generatrix.points) {
        if (point.slopeAccepted && std::isfinite(point.xPx) && std::isfinite(point.yPx)) {
            selectedPoints.push_back(&point);
        }
    }
    if (selectedPoints.size() < 5) {
        selectedPoints.clear();
        for (const GeneratrixCandidatePoint& point : generatrix.points) {
            if (std::isfinite(point.xPx) && std::isfinite(point.yPx)) {
                selectedPoints.push_back(&point);
            }
        }
    }
    if (selectedPoints.size() < 5) {
        result.message = message + " Single-slot mode failed: fewer than 5 measured points are available.";
        return result;
    }

    double anchorXPx = generatrix.anchorXPx;
    double anchorYPx = generatrix.anchorYPx;
    const auto leftIt = std::min_element(selectedPoints.begin(),
                                         selectedPoints.end(),
                                         [](const auto* lhs, const auto* rhs) {
                                             return lhs->xPx < rhs->xPx;
                                         });
    if (leftIt == selectedPoints.end()) {
        result.message = message + " Single-slot mode failed: no left anchor point is available.";
        return result;
    }
    if (effectiveConfig.designUseLeftEndpointAnchor || !std::isfinite(anchorXPx) || !std::isfinite(anchorYPx)) {
        anchorXPx = (*leftIt)->xPx;
        anchorYPx = (*leftIt)->yPx;
    }

    const DesignEval leftDesign = evalDesignProfile(0.0, effectiveConfig);
    const double anchorRadiusMm = radiusFromYPx(anchorYPx, effectiveConfig);
    const double anchorDrMm =
        leftDesign.valid && effectiveConfig.designAnchorRadialToLeftEndpoint ? (leftDesign.r_mm - anchorRadiusMm) : 0.0;

    result.ok = true;
    result.message = inferredTargetNote.empty() ? message : (message + " " + inferredTargetNote);
    result.summary.dzMm = 0.0;
    result.summary.drMm = anchorDrMm;
    result.summary.designReverseAxial = effectiveDesignReverseAxial(effectiveConfig);
    result.summary.useLeftEndpointAnchor = effectiveConfig.designUseLeftEndpointAnchor;
    result.summary.evaluateProfileForm = effectiveConfig.designEvaluateProfileForm;
    result.summary.anchorXPx = anchorXPx;
    result.summary.anchorYPx = anchorYPx;
    result.summary.pixelSizeMm = effectiveConfig.designPixelSizeMm;
    result.summary.candidateCount = selectedPoints.size();

    std::vector<double> normalErrorsUm;
    normalErrorsUm.reserve(selectedPoints.size());
    result.profilePoints.reserve(selectedPoints.size());
    std::size_t outputIndex = 0;
    for (const GeneratrixCandidatePoint* inputPoint : selectedPoints) {
        const double sAlignedMm = (inputPoint->xPx - anchorXPx) * effectiveConfig.designPixelSizeMm;
        if (!std::isfinite(sAlignedMm) || sAlignedMm < -1e-9) {
            continue;
        }

        const double rAlignedMm = radiusFromYPx(inputPoint->yPx, effectiveConfig) + anchorDrMm;
        const DesignEval design = evalDesignProfile(std::max(0.0, sAlignedMm), effectiveConfig);
        const double designRadiusMm = design.valid ? design.r_mm : std::numeric_limits<double>::quiet_NaN();
        const double radialErrorMm = design.valid ? (rAlignedMm - design.r_mm)
                                                  : std::numeric_limits<double>::quiet_NaN();
        const double normalErrorMm =
            design.valid ? radialErrorMm / std::sqrt(1.0 + design.dr_dz * design.dr_dz)
                         : std::numeric_limits<double>::quiet_NaN();

        DesignErrorProfilePoint point;
        point.index = outputIndex++;
        point.supportCount = inputPoint->supportCount;
        point.xPx = inputPoint->xPx;
        point.yPx = inputPoint->yPx;
        point.yStdPx = inputPoint->yStdPx;
        point.sAlignedMm = std::max(0.0, sAlignedMm);
        point.rAlignedMm = rAlignedMm;
        point.designRadiusMm = designRadiusMm;
        point.nearestDesignSMm = point.sAlignedMm;
        point.nearestDesignRMm = designRadiusMm;
        point.radialErrorMm = radialErrorMm;
        point.radialErrorUm = std::isfinite(radialErrorMm) ? radialErrorMm * 1000.0
                                                           : std::numeric_limits<double>::quiet_NaN();
        point.normalErrorMm = normalErrorMm;
        point.normalErrorUm = std::isfinite(normalErrorMm) ? normalErrorMm * 1000.0
                                                           : std::numeric_limits<double>::quiet_NaN();
        point.legacyNormalErrorUm = point.normalErrorUm;
        point.signedDistanceMm = normalErrorMm;
        point.signedDistanceUm = point.normalErrorUm;
        point.profileErrorUm = point.normalErrorUm;
        point.isUsed = true;
        result.profilePoints.push_back(point);
        if (std::isfinite(point.normalErrorUm)) {
            normalErrorsUm.push_back(point.normalErrorUm);
        }
    }

    if (result.profilePoints.size() < 5) {
        result = {};
        result.message = message + " Single-slot mode failed: fewer than 5 profile points remained after anchoring.";
        return result;
    }

    result.summary.usedCount = result.profilePoints.size();
    result.summary.normalStats = buildErrorStats(normalErrorsUm);
    result.summary.profileStats = result.summary.normalStats;
    result.summary.absoluteAllStats = result.summary.normalStats;
    result.summary.absoluteFilteredStats = result.summary.normalStats;
    result.summary.meanNormalErrorUm = result.summary.normalStats.meanUm;

    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(effectiveConfig);
    result.summary.designProfileMetadata = buildActiveDesignProfileMetadata(effectiveConfig, activeDesignPolyline);
    result.designProfileSamples = toPublicDesignProfileSamples(activeDesignPolyline);
    applySingleSlotRoi(result, effectiveConfig);
    applySingleSlotCadFrameMapping(result,
                                   effectiveConfig,
                                   activeDesignPolyline,
                                   result.summary.designProfileMetadata);
    result.profileCsvText = buildProfileCsv(result.profilePoints);
    result.summaryCsvText = buildSummaryCsv(result.summary);
    result.error3dCsvText = build3dErrorCsv(result.profilePoints, result.summary.designProfileMetadata);
    result.compensationCsvText = buildCompensationCsv(result.profilePoints, result.summary.designProfileMetadata);
    result.featureCompensationCsvText = buildFeatureCompensationCsv(result.designProfileSamples,
                                                                    result.profilePoints,
                                                                    result.summary.designProfileMetadata,
                                                                    effectiveConfig);
    return result;
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

std::string boolText(const bool value)
{
    return value ? "true" : "false";
}

std::string buildGeneratrixInsufficientMessage(const std::size_t rawSampleCount,
                                               const MeasuredGeneratrixData& generatrix,
                                               const stitch::StitchPipelineConfig& config,
                                               const std::size_t trimmedCount,
                                               const std::string& singleSlotMessage)
{
    const std::size_t slopeAcceptedCount =
        static_cast<std::size_t>(std::count_if(generatrix.points.begin(),
                                               generatrix.points.end(),
                                               [](const auto& point) {
                                                   return point.slopeAccepted;
                                               }));
    const std::size_t primaryRunCount =
        generatrix.primaryRunEndIndex >= generatrix.primaryRunBeginIndex &&
                generatrix.primaryRunEndIndex < generatrix.points.size()
            ? generatrix.primaryRunEndIndex - generatrix.primaryRunBeginIndex + 1
            : 0;
    const std::size_t afterAnchorCount =
        generatrix.anchorPointIndex < generatrix.points.size()
            ? generatrix.points.size() - generatrix.anchorPointIndex
            : 0;
    const auto largestRunIt = std::max_element(generatrix.runs.begin(),
                                               generatrix.runs.end(),
                                               [](const PointRun& lhs, const PointRun& rhs) {
                                                   return lhs.pointCount < rhs.pointCount;
                                               });
    const std::size_t largestRunCount =
        largestRunIt == generatrix.runs.end() ? 0 : largestRunIt->pointCount;
    const double largestRunLengthMm =
        largestRunIt == generatrix.runs.end() ? 0.0 : largestRunIt->lengthMm;

    std::ostringstream stream;
    stream << "design profile alignment failed: too few measured generatrix points. "
           << "raw_contour_samples=" << rawSampleCount
           << ", generatrix_points=" << generatrix.points.size()
           << ", slope_accepted=" << slopeAcceptedCount
           << ", primary_run_points=" << primaryRunCount
           << ", after_anchor_points=" << afterAnchorCount
           << ", trim_accepted=" << trimmedCount
           << ", min_required=" << minUsedPointCount(config)
           << ", run_count=" << generatrix.runs.size()
           << ", largest_run_points=" << largestRunCount
           << ", largest_run_length_mm=" << formatDouble(largestRunLengthMm, 6)
           << ", filter_end_face_edges=" << boolText(config.designFilterEndFaceEdges)
           << ", target_slot_width_mm=" << formatDouble(config.designTargetSlotWidthMm, 6)
           << ", pixel_size_mm=" << formatDouble(config.designPixelSizeMm, 9)
           << ", max_base_mm=" << formatDouble(generatrix.maxBaseMm, 6)
           << ".";
    if (!singleSlotMessage.empty()) {
        stream << " " << singleSlotMessage;
    }
    return stream.str();
}

double inferPseudoPixelSizeMmFromMeasuredSamples(const std::vector<DesignProfileSample>& measuredSamples)
{
    std::vector<double> sValues;
    sValues.reserve(measuredSamples.size());
    for (const DesignProfileSample& sample : measuredSamples) {
        if (std::isfinite(sample.sMm)) {
            sValues.push_back(sample.sMm);
        }
    }
    if (sValues.size() < 2) {
        return 0.001;
    }

    std::stable_sort(sValues.begin(), sValues.end());
    std::vector<double> steps;
    steps.reserve(sValues.size() - 1);
    for (std::size_t index = 1; index < sValues.size(); ++index) {
        const double delta = sValues[index] - sValues[index - 1];
        if (delta > 1e-9 && std::isfinite(delta)) {
            steps.push_back(delta);
        }
    }
    if (steps.empty()) {
        return 0.001;
    }
    std::stable_sort(steps.begin(), steps.end());
    return std::clamp(steps[steps.size() / 2], 1e-6, 1.0);
}

MeasuredGeneratrixData buildMeasuredGeneratrixFromCadProfileSamples(
    const std::vector<DesignProfileSample>& measuredSamples,
    const stitch::StitchPipelineConfig& config)
{
    MeasuredGeneratrixData data;
    if (!(config.designPixelSizeMm > 1e-12)) {
        return data;
    }

    std::vector<DesignProfileSample> orderedSamples;
    orderedSamples.reserve(measuredSamples.size());
    for (const DesignProfileSample& sample : measuredSamples) {
        if (std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            orderedSamples.push_back(sample);
        }
    }
    if (orderedSamples.size() < 2) {
        return data;
    }

    std::stable_sort(orderedSamples.begin(), orderedSamples.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return lhs.rMm < rhs.rMm;
    });

    std::vector<DesignProfileSample> uniqueSamples;
    uniqueSamples.reserve(orderedSamples.size());
    for (const DesignProfileSample& sample : orderedSamples) {
        if (!uniqueSamples.empty() &&
            std::abs(sample.sMm - uniqueSamples.back().sMm) < 1e-9 &&
            std::abs(sample.rMm - uniqueSamples.back().rMm) < 1e-9) {
            continue;
        }
        uniqueSamples.push_back(sample);
    }
    if (uniqueSamples.size() < 2) {
        return data;
    }

    const double baseOffsetMm = uniqueSamples.front().sMm;
    const double ySign = config.designInvertY ? -1.0 : 1.0;
    data.points.reserve(uniqueSamples.size());
    for (std::size_t index = 0; index < uniqueSamples.size(); ++index) {
        const DesignProfileSample& sample = uniqueSamples[index];
        GeneratrixCandidatePoint point;
        point.index = index;
        point.supportCount = 1;
        point.sBaseMm = sample.sMm - baseOffsetMm;
        point.rRawMm = sample.rMm;
        point.xPx = point.sBaseMm / config.designPixelSizeMm;
        point.yPx = ySign * point.rRawMm / config.designPixelSizeMm;
        point.yStdPx = 0.0;
        point.slopeAccepted = true;
        point.trimAccepted = true;
        data.points.push_back(point);
    }

    for (std::size_t index = 0; index < data.points.size(); ++index) {
        const std::size_t leftIndex = index > 0 ? index - 1 : index;
        const std::size_t rightIndex = index + 1 < data.points.size() ? index + 1 : index;
        const double deltaS = data.points[rightIndex].sBaseMm - data.points[leftIndex].sBaseMm;
        const double deltaR = data.points[rightIndex].rRawMm - data.points[leftIndex].rRawMm;
        data.points[index].slopeAbs =
            std::abs(deltaS) > 1e-12 && std::isfinite(deltaR / deltaS) ? std::abs(deltaR / deltaS) : 0.0;
    }

    data.runs = buildAcceptedRuns(data.points, config);
    if (data.runs.empty()) {
        PointRun run;
        run.beginIndex = 0;
        run.endIndex = data.points.size() - 1;
        run.pointCount = data.points.size();
        run.lengthMm = data.points.back().sBaseMm - data.points.front().sBaseMm;
        data.runs.push_back(run);
    }

    data.anchorPointIndex = 0;
    const PointRun primaryRun = findRunContainingIndex(data.runs, data.anchorPointIndex);
    data.primaryRunBeginIndex = primaryRun.pointCount > 0 ? primaryRun.beginIndex : 0;
    data.primaryRunEndIndex = primaryRun.pointCount > 0 ? primaryRun.endIndex : (data.points.size() - 1);
    data.anchorXPx = data.points.front().xPx;
    data.anchorYPx = data.points.front().yPx;
    data.anchorBaseMm = data.points.front().sBaseMm;
    data.anchorMeasuredRadiusMm = data.points.front().rRawMm;
    data.maxBaseMm = data.points.back().sBaseMm;
    return data;
}

DesignAlignmentResult compareMeasuredGeneratrixToDesign(const MeasuredGeneratrixData& generatrix,
                                                        const stitch::StitchPipelineConfig& config,
                                                        const std::size_t sourceSampleCount,
                                                        const std::vector<StitchedContourSample>* stitchedSamples)
{
    DesignAlignmentResult result;
    std::string singleSlotTargetNote;
    const stitch::StitchPipelineConfig singleSlotConfig =
        effectiveSingleSlotTargetConfig(config, singleSlotTargetNote);
    if (hasExternalDesignProfile(singleSlotConfig) && singleSlotConfig.designTargetSlotWidthMm > 1e-9) {
        if (stitchedSamples != nullptr && singleSlotConfig.designUseCentralSlotImageRoi) {
            DesignAlignmentResult directRoiResult =
                buildDirectCentralSlotRoiResult(generatrix,
                                                singleSlotConfig,
                                                singleSlotTargetNote.empty()
                                                    ? "single slot CAD target mode completed with direct image ROI."
                                                    : ("single slot CAD target mode completed with direct image ROI. " +
                                                       singleSlotTargetNote));
            if (directRoiResult.ok) {
                return directRoiResult;
            }
        }

        if (stitchedSamples != nullptr) {
            stitch::StitchPipelineConfig guidedSingleSlotConfig = singleSlotConfig;
            guidedSingleSlotConfig.designUseLeftEndpointAnchor = false;
            const MeasuredGeneratrixData cadGuidedSlotBottom =
                buildCadGuidedSingleSlotBottomGeneratrix(*stitchedSamples, generatrix, guidedSingleSlotConfig);
            if (!cadGuidedSlotBottom.points.empty()) {
                DesignAlignmentResult guidedSingleSlotResult =
                    buildSingleSlotTargetFallbackResult(cadGuidedSlotBottom,
                                                        guidedSingleSlotConfig,
                                                        singleSlotTargetNote.empty()
                                                            ? "single slot CAD target mode completed with CAD-guided slot-bottom contour."
                                                            : ("single slot CAD target mode completed with CAD-guided slot-bottom contour. " +
                                                               singleSlotTargetNote));
                if (guidedSingleSlotResult.ok) {
                    return guidedSingleSlotResult;
                }
            }
        }

        DesignAlignmentResult singleSlotResult =
            buildSingleSlotTargetFallbackResult(generatrix,
                                                singleSlotConfig,
                                                singleSlotTargetNote.empty()
                                                    ? "single slot CAD target mode completed with full measured contour."
                                                    : ("single slot CAD target mode completed with full measured contour. " +
                                                       singleSlotTargetNote));
        if (singleSlotResult.ok) {
            return singleSlotResult;
        }
    }

    const std::size_t trimmedCount = std::count_if(generatrix.points.begin(), generatrix.points.end(), [](const auto& point) {
        return point.trimAccepted;
    });
    if (trimmedCount < minUsedPointCount(config)) {
        DesignAlignmentResult fallbackResult =
            buildSingleSlotTargetFallbackResult(generatrix,
                                                config,
                                                "single slot width target mode completed with limited contour points.");
        if (fallbackResult.ok) {
            return fallbackResult;
        }

        result.message = buildGeneratrixInsufficientMessage(sourceSampleCount,
                                                            generatrix,
                                                            config,
                                                            trimmedCount,
                                                            fallbackResult.message);
        return result;
    }

    const DesignEval leftDesign = evalDesignProfile(0.0, config);
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
        DesignAlignmentResult fallbackResult =
            buildSingleSlotTargetFallbackResult(generatrix,
                                                config,
                                                "single slot width target mode completed after overlap filtering failed.");
        if (fallbackResult.ok) {
            return fallbackResult;
        }

        result.message =
            "design profile alignment failed: too few valid overlap samples after anchoring and domain filtering. " +
            fallbackResult.message;
        return result;
    }

    result.ok = true;
    result.summary = evaluation.summary;
    std::string inferredTargetNote;
    const stitch::StitchPipelineConfig outputConfig =
        effectiveSingleSlotTargetConfig(config, inferredTargetNote);
    const std::vector<DesignPolylinePoint> activeDesignPolyline = buildActiveDesignPolyline(outputConfig);
    result.summary.designProfileMetadata = buildActiveDesignProfileMetadata(outputConfig, activeDesignPolyline);
    result.profilePoints = evaluation.profilePoints;
    result.designProfileSamples = toPublicDesignProfileSamples(activeDesignPolyline);
    result.message = hasExternalDesignProfile(outputConfig)
                         ? "external CAD design profile alignment completed."
                         : "builtin design profile alignment completed.";
    if (!inferredTargetNote.empty()) {
        result.message += " " + inferredTargetNote;
    }
    applySingleSlotRoi(result, outputConfig);
    applySingleSlotCadFrameMapping(result,
                                   outputConfig,
                                   activeDesignPolyline,
                                   result.summary.designProfileMetadata);
    result.profileCsvText = buildProfileCsv(result.profilePoints);
    result.summaryCsvText = buildSummaryCsv(result.summary);
    result.error3dCsvText = build3dErrorCsv(result.profilePoints, result.summary.designProfileMetadata);
    result.compensationCsvText = buildCompensationCsv(result.profilePoints, result.summary.designProfileMetadata);
    result.featureCompensationCsvText = buildFeatureCompensationCsv(result.designProfileSamples,
                                                                    result.profilePoints,
                                                                    result.summary.designProfileMetadata,
                                                                    outputConfig);
    return result;
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
    return compareMeasuredGeneratrixToDesign(generatrix, config, samples.size(), &samples);
}

DesignAlignmentResult compareMeasuredContourSamplesToDesign(const std::vector<cv::Point2d>& contourSamples,
                                                            const stitch::StitchPipelineConfig& config)
{
    DesignAlignmentResult result;
    if (!config.enableDesignComparison) {
        result.message = "design profile comparison disabled by configuration.";
        return result;
    }

    std::vector<StitchedContourSample> samples;
    samples.reserve(contourSamples.size());
    for (const cv::Point2d& point : contourSamples) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            samples.push_back({point.x, point.y});
        }
    }
    if (samples.empty()) {
        result.message = "design profile alignment failed: contour sample list is empty.";
        return result;
    }

    const MeasuredGeneratrixData generatrix = extractMeasuredGeneratrixForDesignComparison(samples, config);
    if (generatrix.points.empty()) {
        result.message = "design profile alignment failed: no valid measured generatrix after contour-sample envelope extraction.";
        return result;
    }
    return compareMeasuredGeneratrixToDesign(generatrix, config, samples.size(), &samples);
}

DesignAlignmentResult compareMeasuredCadProfileToDesign(const std::vector<DesignProfileSample>& measuredSamples,
                                                        const stitch::StitchPipelineConfig& config)
{
    DesignAlignmentResult result;
    if (!config.enableDesignComparison) {
        result.message = "design profile comparison disabled by configuration.";
        return result;
    }

    stitch::StitchPipelineConfig effectiveConfig = config;
    if (!(effectiveConfig.designPixelSizeMm > 1e-12)) {
        effectiveConfig.designPixelSizeMm = inferPseudoPixelSizeMmFromMeasuredSamples(measuredSamples);
    }

    const MeasuredGeneratrixData generatrix =
        buildMeasuredGeneratrixFromCadProfileSamples(measuredSamples, effectiveConfig);
    if (generatrix.points.empty()) {
        result.message = "design profile alignment failed: scanned CAD/STL profile samples are empty or invalid.";
        return result;
    }

    result = compareMeasuredGeneratrixToDesign(generatrix, effectiveConfig, measuredSamples.size(), nullptr);
    if (result.ok) {
        std::ostringstream message;
        message << "scanned CAD/STL profile alignment completed."
                << " pseudo_pixel_size_mm=" << formatDouble(effectiveConfig.designPixelSizeMm, 6)
                << ", measured_samples=" << measuredSamples.size() << ". "
                << result.message;
        result.message = message.str();
        result.summary.pixelSizeMm = effectiveConfig.designPixelSizeMm;
    }
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
    std::vector<DesignProfileSample> designSamples = result.designProfileSamples;
    if (designSamples.empty()) {
        const std::vector<DesignPolylinePoint> fallbackPolyline = buildDesignPolyline(result.summary.designReverseAxial);
        designSamples = toPublicDesignProfileSamples(fallbackPolyline);
    }
    for (const DesignProfileSample& design : designSamples) {
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
    info << "source=" << result.summary.designProfileMetadata.sourceType
         << "  reverse_axial=" << (result.summary.designReverseAxial ? 1 : 0)
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

cv::Mat buildCompensationPlot(const DesignAlignmentResult& result)
{
    const int width = 1440;
    const int height = 920;
    const cv::Rect cadRect(104, 108, width - 180, 330);
    const cv::Rect profileRect(104, 548, width - 180, 250);
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    drawInfoBox(canvas, "Compensation settlement in CAD/profile coordinates", cv::Point(34, 42), 0.66,
                cv::Scalar(38, 38, 38));

    std::vector<cv::Point2d> deltaXSeries;
    std::vector<cv::Point2d> deltaYSeries;
    std::vector<cv::Point2d> deltaZSeries;
    std::vector<cv::Point2d> radialSeries;
    std::vector<cv::Point2d> normalSeries;
    deltaXSeries.reserve(result.profilePoints.size());
    deltaYSeries.reserve(result.profilePoints.size());
    deltaZSeries.reserve(result.profilePoints.size());
    radialSeries.reserve(result.profilePoints.size());
    normalSeries.reserve(result.profilePoints.size());

    double minS = std::numeric_limits<double>::infinity();
    double maxS = -std::numeric_limits<double>::infinity();
    double maxAbsCadUm = 1.0;
    double maxAbsProfileUm = 1.0;
    std::size_t cadCompensationCount = 0;

    for (const DesignErrorProfilePoint& point : result.profilePoints) {
        if (!point.isUsed || !std::isfinite(point.sAlignedMm)) {
            continue;
        }

        minS = std::min(minS, point.sAlignedMm);
        maxS = std::max(maxS, point.sAlignedMm);

        if (point.hasCadCoordinates) {
            ++cadCompensationCount;
            if (std::isfinite(point.compensationDeltaCadXUm)) {
                deltaXSeries.emplace_back(point.sAlignedMm, point.compensationDeltaCadXUm);
                maxAbsCadUm = std::max(maxAbsCadUm, std::abs(point.compensationDeltaCadXUm));
            }
            if (std::isfinite(point.compensationDeltaCadYUm)) {
                deltaYSeries.emplace_back(point.sAlignedMm, point.compensationDeltaCadYUm);
                maxAbsCadUm = std::max(maxAbsCadUm, std::abs(point.compensationDeltaCadYUm));
            }
            if (std::isfinite(point.compensationDeltaCadZUm)) {
                deltaZSeries.emplace_back(point.sAlignedMm, point.compensationDeltaCadZUm);
                maxAbsCadUm = std::max(maxAbsCadUm, std::abs(point.compensationDeltaCadZUm));
            }
        }

        if (std::isfinite(point.radialErrorUm)) {
            const double compensationRadialUm = -point.radialErrorUm;
            radialSeries.emplace_back(point.sAlignedMm, compensationRadialUm);
            maxAbsProfileUm = std::max(maxAbsProfileUm, std::abs(compensationRadialUm));
        }
        if (std::isfinite(point.normalErrorUm)) {
            const double compensationNormalUm = -point.normalErrorUm;
            normalSeries.emplace_back(point.sAlignedMm, compensationNormalUm);
            maxAbsProfileUm = std::max(maxAbsProfileUm, std::abs(compensationNormalUm));
        }
    }

    if (!std::isfinite(minS) || !std::isfinite(maxS) || minS >= maxS) {
        cv::putText(canvas,
                    "No valid compensation samples",
                    cv::Point(120, 130),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.82,
                    cv::Scalar(90, 90, 90),
                    2,
                    cv::LINE_AA);
        return canvas;
    }

    maxAbsCadUm *= 1.12;
    maxAbsProfileUm *= 1.12;
    drawPlotGrid(canvas, cadRect, minS, maxS, -maxAbsCadUm, maxAbsCadUm, 5, 4);
    drawPlotGrid(canvas, profileRect, minS, maxS, -maxAbsProfileUm, maxAbsProfileUm, 5, 4);

    const cv::Scalar xColor(218, 92, 58);
    const cv::Scalar yColor(35, 151, 99);
    const cv::Scalar zColor(50, 103, 210);
    const cv::Scalar radialColor(201, 122, 37);
    const cv::Scalar normalColor(120, 72, 184);
    const cv::Scalar zeroColor(160, 160, 160);

    drawPolyline(canvas, cadRect, deltaXSeries, xColor, minS, maxS, -maxAbsCadUm, maxAbsCadUm, 2);
    drawPolyline(canvas, cadRect, deltaYSeries, yColor, minS, maxS, -maxAbsCadUm, maxAbsCadUm, 2);
    drawPolyline(canvas, cadRect, deltaZSeries, zColor, minS, maxS, -maxAbsCadUm, maxAbsCadUm, 2);
    drawPolyline(canvas, profileRect, radialSeries, radialColor, minS, maxS, -maxAbsProfileUm, maxAbsProfileUm, 2);
    drawPolyline(canvas, profileRect, normalSeries, normalColor, minS, maxS, -maxAbsProfileUm, maxAbsProfileUm, 2);

    cv::line(canvas,
             mapPointToRect(cadRect, minS, 0.0, minS, maxS, -maxAbsCadUm, maxAbsCadUm),
             mapPointToRect(cadRect, maxS, 0.0, minS, maxS, -maxAbsCadUm, maxAbsCadUm),
             zeroColor,
             1,
             cv::LINE_AA);
    cv::line(canvas,
             mapPointToRect(profileRect, minS, 0.0, minS, maxS, -maxAbsProfileUm, maxAbsProfileUm),
             mapPointToRect(profileRect, maxS, 0.0, minS, maxS, -maxAbsProfileUm, maxAbsProfileUm),
             zeroColor,
             1,
             cv::LINE_AA);

    cv::putText(canvas, "CAD-axis compensation delta (um)", cv::Point(20, cadRect.y - 18), cv::FONT_HERSHEY_SIMPLEX,
                0.60, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    cv::putText(canvas, "Profile compensation (um)", cv::Point(20, profileRect.y - 18), cv::FONT_HERSHEY_SIMPLEX,
                0.60, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    cv::putText(canvas, "Comparison coordinate s (mm)",
                cv::Point(profileRect.x + profileRect.width / 2 - 110, height - 34),
                cv::FONT_HERSHEY_SIMPLEX,
                0.64,
                cv::Scalar(90, 90, 90),
                1,
                cv::LINE_AA);

    const int legendX = width - 344;
    const int legendY = 74;
    const auto legendLine = [&](const int row, const cv::Scalar& color, const std::string& label) {
        const int y = legendY + row * 25;
        cv::line(canvas, cv::Point(legendX, y), cv::Point(legendX + 34, y), color, 3, cv::LINE_AA);
        cv::putText(canvas, label, cv::Point(legendX + 46, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    };
    legendLine(0, xColor, "CAD delta X");
    legendLine(1, yColor, "CAD delta Y");
    legendLine(2, zColor, "CAD delta Z");
    legendLine(3, radialColor, "Radial compensation");
    legendLine(4, normalColor, "Normal compensation");

    std::ostringstream info;
    info.setf(std::ios::fixed);
    info.precision(3);
    info << "source=" << result.summary.designProfileMetadata.sourceType
         << "  cad_method=" << result.summary.designProfileMetadata.extractionMethod
         << "  cad_points=" << cadCompensationCount
         << "  used_points=" << result.summary.usedCount
         << "  profile_rms=" << result.summary.profileStats.rmseUm << " um"
         << "  profile_p95=" << result.summary.profileStats.p95AbsUm << " um";
    drawInfoBox(canvas, info.str(), cv::Point(34, 854), 0.50, cv::Scalar(60, 60, 60));

    return canvas;
}

} // namespace pinjie::cad_design
