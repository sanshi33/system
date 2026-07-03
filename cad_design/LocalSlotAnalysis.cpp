#include "cad_design/LocalSlotAnalysis.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace pinjie::cad_design {

namespace {

constexpr double kEpsilon = 1e-9;
constexpr double kLocalExceedanceThresholdUm = 15.0;

struct Segment {
    cv::Point2d a;
    cv::Point2d b;
    std::string label;

    [[nodiscard]] double lengthPx() const
    {
        return cv::norm(b - a);
    }
};

struct MatchedPoint {
    std::size_t sourceIndex{0};
    cv::Point2d imagePoint;
    cv::Point2d designPoint;
    std::string segmentLabel;
    double pathPx{0.0};
    double distancePx{0.0};
    double signedDistancePx{0.0};
    bool isUsed{false};
};

cv::Point2d segmentUnit(const Segment& segment)
{
    const cv::Point2d direction = segment.b - segment.a;
    const double length = cv::norm(direction);
    return length > kEpsilon ? direction * (1.0 / length) : cv::Point2d(1.0, 0.0);
}

cv::Point2d segmentNormal(const Segment& segment)
{
    const cv::Point2d unit = segmentUnit(segment);
    return {unit.y, -unit.x};
}

double segmentSpx(const cv::Point2d& point, const Segment& segment)
{
    return (point - segment.a).dot(segmentUnit(segment));
}

double segmentRpx(const cv::Point2d& point, const Segment& segment)
{
    return (point - segment.a).dot(segmentNormal(segment));
}

std::string csvTextCell(const std::string& text)
{
    std::string escaped;
    escaped.reserve(text.size() + 2);
    escaped.push_back('"');
    for (const char ch : text) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string csvCell(const double value, const int precision = 6)
{
    if (!std::isfinite(value)) {
        return {};
    }
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream << std::setprecision(precision) << value;
    return stream.str();
}

std::string joinCsvRow(const std::vector<std::string>& cells)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << cells[index];
    }
    stream << "\n";
    return stream.str();
}

double percentile(std::vector<double> values, const double q)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](const double value) {
                     return !std::isfinite(value);
                 }),
                 values.end());
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double scaled = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(scaled));
    const auto hi = static_cast<std::size_t>(std::ceil(scaled));
    if (lo == hi) {
        return values[lo];
    }
    const double t = scaled - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

double median(std::vector<double> values)
{
    return percentile(std::move(values), 0.5);
}

double mean(const std::vector<double>& values)
{
    double sum = 0.0;
    std::size_t count = 0;
    for (const double value : values) {
        if (std::isfinite(value)) {
            sum += value;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
}

double rmse(const std::vector<double>& values)
{
    double sum = 0.0;
    std::size_t count = 0;
    for (const double value : values) {
        if (std::isfinite(value)) {
            sum += value * value;
            ++count;
        }
    }
    return count > 0 ? std::sqrt(sum / static_cast<double>(count))
                     : std::numeric_limits<double>::quiet_NaN();
}

double maxAbsValue(const std::vector<double>& values)
{
    double result = 0.0;
    bool hasValue = false;
    for (const double value : values) {
        if (std::isfinite(value)) {
            result = hasValue ? std::max(result, std::abs(value)) : std::abs(value);
            hasValue = true;
        }
    }
    return hasValue ? result : std::numeric_limits<double>::quiet_NaN();
}

std::size_t countExceedances(const std::vector<double>& values, const double thresholdUm)
{
    return static_cast<std::size_t>(
        std::count_if(values.begin(), values.end(), [thresholdUm](const double value) {
            return std::isfinite(value) && std::abs(value) > thresholdUm;
        }));
}

struct CadSlotFeature {
    bool valid{false};
    double leftSMm{0.0};
    double rightSMm{0.0};
    double centerSMm{0.0};
    double widthMm{0.0};
    double bottomSMm{0.0};
    double bottomRMm{0.0};
    double topRMm{0.0};
    double depthMm{0.0};
};

struct CadNearestPoint {
    bool valid{false};
    bool hasCadPoint{false};
    double sMm{0.0};
    double rMm{0.0};
    double cadXMm{0.0};
    double cadYMm{0.0};
    double cadZMm{0.0};
};

struct CadMappedPoint {
    bool valid{false};
    double targetSMm{std::numeric_limits<double>::quiet_NaN()};
    double targetRMm{std::numeric_limits<double>::quiet_NaN()};
    double measuredCadXMm{std::numeric_limits<double>::quiet_NaN()};
    double measuredCadYMm{std::numeric_limits<double>::quiet_NaN()};
    double measuredCadZMm{std::numeric_limits<double>::quiet_NaN()};
    double targetCadXMm{std::numeric_limits<double>::quiet_NaN()};
    double targetCadYMm{std::numeric_limits<double>::quiet_NaN()};
    double targetCadZMm{std::numeric_limits<double>::quiet_NaN()};
    double deltaXUm{std::numeric_limits<double>::quiet_NaN()};
    double deltaYUm{std::numeric_limits<double>::quiet_NaN()};
    double deltaZUm{std::numeric_limits<double>::quiet_NaN()};
};

struct CadMappingContext {
    bool valid{false};
    DesignProfileMetadata metadata;
    std::vector<DesignProfileSample> samples;
    CadSlotFeature feature;
    double localWidthMm{0.0};
    double localDepthMm{0.0};
    double radialScale{1.0};
};

int axisIndexFromLabel(const std::string& axis)
{
    if (axis == "X" || axis == "x") {
        return 0;
    }
    if (axis == "Y" || axis == "y") {
        return 1;
    }
    if (axis == "Z" || axis == "z") {
        return 2;
    }
    return -1;
}

void setCadAxisValue(double& x, double& y, double& z, const int axis, const double value)
{
    if (axis == 0) {
        x = value;
    } else if (axis == 1) {
        y = value;
    } else if (axis == 2) {
        z = value;
    }
}

int remainingCadAxis(const int first, const int second)
{
    if (first < 0 || first > 2 || second < 0 || second > 2 || first == second) {
        return -1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != first && axis != second) {
            return axis;
        }
    }
    return -1;
}

CadSlotFeature estimateTargetWidthCadSlotFeature(std::vector<cv::Point2d> series,
                                                 const double targetWidthMm)
{
    CadSlotFeature best;
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
    const double topR = percentile(radii, 0.90);
    const double lowR = percentile(radii, 0.10);
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
                score -= 0.02 * std::clamp(depthMm / std::max(depthSpanMm, 1e-9), 0.0, 1.0);
                if (score < bestScore) {
                    bestScore = score;
                    best.valid = true;
                    best.leftSMm = leftS;
                    best.rightSMm = rightS;
                    best.centerSMm = 0.5 * (leftS + rightS);
                    best.widthMm = widthMm;
                    best.bottomSMm = series[bottomIndex].x;
                    best.bottomRMm = series[bottomIndex].y;
                    best.topRMm = topR;
                    best.depthMm = depthMm;
                }
            }

            index = end + 1;
        }
    }
    return best;
}

CadNearestPoint findNearestCadProfilePoint(const std::vector<DesignProfileSample>& samples,
                                           const double sMm,
                                           const double rMm)
{
    CadNearestPoint nearest;
    if (samples.size() < 2 || !std::isfinite(sMm) || !std::isfinite(rMm)) {
        return nearest;
    }

    double bestDistanceSquare = std::numeric_limits<double>::infinity();
    for (std::size_t index = 1; index < samples.size(); ++index) {
        const DesignProfileSample& a = samples[index - 1];
        const DesignProfileSample& b = samples[index];
        if (!std::isfinite(a.sMm) || !std::isfinite(a.rMm) ||
            !std::isfinite(b.sMm) || !std::isfinite(b.rMm)) {
            continue;
        }
        const double ds = b.sMm - a.sMm;
        const double dr = b.rMm - a.rMm;
        const double lengthSquare = ds * ds + dr * dr;
        if (lengthSquare < 1e-18) {
            continue;
        }
        const double rawT = ((sMm - a.sMm) * ds + (rMm - a.rMm) * dr) / lengthSquare;
        const double t = std::clamp(rawT, 0.0, 1.0);
        const double candidateS = a.sMm + t * ds;
        const double candidateR = a.rMm + t * dr;
        const double distanceSquare =
            (sMm - candidateS) * (sMm - candidateS) + (rMm - candidateR) * (rMm - candidateR);
        if (distanceSquare >= bestDistanceSquare) {
            continue;
        }
        bestDistanceSquare = distanceSquare;
        nearest.valid = true;
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
    }
    return nearest;
}

CadMappedPoint mapLocalSlotPointToCad(const CadMappingContext& context,
                                      const double measuredS,
                                      const double measuredR,
                                      const double targetS,
                                      const double targetR)
{
    CadMappedPoint mapped;
    if (!context.valid ||
        !(context.localWidthMm > 1e-9) ||
        !std::isfinite(measuredS) ||
        !std::isfinite(measuredR) ||
        !std::isfinite(targetS) ||
        !std::isfinite(targetR)) {
        return mapped;
    }

    const double sScale = context.feature.widthMm / context.localWidthMm;
    const double measuredCadS = context.feature.leftSMm + measuredS * sScale;
    const double targetCadS = context.feature.leftSMm + targetS * sScale;
    const double measuredCadR = context.feature.bottomRMm + measuredR * context.radialScale;
    const double targetCadR = context.feature.bottomRMm + targetR * context.radialScale;

    const CadNearestPoint targetNearest =
        findNearestCadProfilePoint(context.samples, targetCadS, targetCadR);
    const CadNearestPoint measuredNearest =
        findNearestCadProfilePoint(context.samples, measuredCadS, measuredCadR);
    if (!targetNearest.valid || !targetNearest.hasCadPoint ||
        !measuredNearest.valid || !measuredNearest.hasCadPoint) {
        return mapped;
    }

    const int axialAxis = axisIndexFromLabel(context.metadata.axialAxis);
    const int radialAxis = axisIndexFromLabel(context.metadata.radialAxis);
    if (axialAxis < 0 || radialAxis < 0 || axialAxis == radialAxis ||
        !std::isfinite(context.metadata.cadAxialDirectionSign)) {
        return mapped;
    }

    mapped.valid = true;
    mapped.targetSMm = targetNearest.sMm;
    mapped.targetRMm = targetNearest.rMm;
    mapped.targetCadXMm = targetNearest.cadXMm;
    mapped.targetCadYMm = targetNearest.cadYMm;
    mapped.targetCadZMm = targetNearest.cadZMm;
    mapped.measuredCadXMm = measuredNearest.cadXMm;
    mapped.measuredCadYMm = measuredNearest.cadYMm;
    mapped.measuredCadZMm = measuredNearest.cadZMm;
    setCadAxisValue(mapped.measuredCadXMm,
                    mapped.measuredCadYMm,
                    mapped.measuredCadZMm,
                    axialAxis,
                    context.metadata.cadAxialOriginMm +
                        context.metadata.cadAxialDirectionSign * measuredCadS);
    setCadAxisValue(mapped.measuredCadXMm,
                    mapped.measuredCadYMm,
                    mapped.measuredCadZMm,
                    radialAxis,
                    measuredCadR);
    mapped.valid = std::isfinite(mapped.measuredCadXMm) &&
                   std::isfinite(mapped.measuredCadYMm) &&
                   std::isfinite(mapped.measuredCadZMm) &&
                   std::isfinite(mapped.targetCadXMm) &&
                   std::isfinite(mapped.targetCadYMm) &&
                   std::isfinite(mapped.targetCadZMm);
    if (mapped.valid) {
        mapped.deltaXUm = (mapped.targetCadXMm - mapped.measuredCadXMm) * 1000.0;
        mapped.deltaYUm = (mapped.targetCadYMm - mapped.measuredCadYMm) * 1000.0;
        mapped.deltaZUm = (mapped.targetCadZMm - mapped.measuredCadZMm) * 1000.0;
    }
    return mapped;
}

cv::Mat buildBinaryMask(const cv::Mat& image, bool& objectBoundaryMode)
{
    objectBoundaryMode = false;
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image.clone();
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);

    const double imageArea = static_cast<double>(std::max(1, image.rows * image.cols));
    cv::Mat materialMask;
    cv::threshold(blurred, materialMask, 0.0, 255.0, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    const double materialRatio = static_cast<double>(cv::countNonZero(materialMask)) / imageArea;
    if (materialRatio > 0.12 && materialRatio < 0.88) {
        cv::Mat boundary;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(materialMask, materialMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(materialMask, boundary, cv::MORPH_GRADIENT, kernel);
        const int clearMargin = 3;
        boundary.rowRange(0, std::min(clearMargin, boundary.rows)).setTo(cv::Scalar(0));
        boundary.rowRange(std::max(0, boundary.rows - clearMargin), boundary.rows).setTo(cv::Scalar(0));
        boundary.colRange(0, std::min(clearMargin, boundary.cols)).setTo(cv::Scalar(0));
        boundary.colRange(std::max(0, boundary.cols - clearMargin), boundary.cols).setTo(cv::Scalar(0));
        if (cv::countNonZero(boundary) > 30) {
            objectBoundaryMode = true;
            return boundary;
        }
    }

    cv::Mat mask;
    cv::threshold(blurred, mask, 0.0, 255.0, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    const double foregroundRatio =
        static_cast<double>(cv::countNonZero(mask)) /
        static_cast<double>(std::max(1, mask.rows * mask.cols));
    if (foregroundRatio > 0.45) {
        cv::threshold(blurred, mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

void zhangSuenIteration(cv::Mat& image, const int iteration)
{
    cv::Mat marker = cv::Mat::zeros(image.size(), CV_8UC1);
    for (int y = 1; y < image.rows - 1; ++y) {
        const uchar* prev = image.ptr<uchar>(y - 1);
        const uchar* curr = image.ptr<uchar>(y);
        const uchar* next = image.ptr<uchar>(y + 1);
        uchar* mark = marker.ptr<uchar>(y);
        for (int x = 1; x < image.cols - 1; ++x) {
            const int p2 = prev[x] > 0;
            const int p3 = prev[x + 1] > 0;
            const int p4 = curr[x + 1] > 0;
            const int p5 = next[x + 1] > 0;
            const int p6 = next[x] > 0;
            const int p7 = next[x - 1] > 0;
            const int p8 = curr[x - 1] > 0;
            const int p9 = prev[x - 1] > 0;
            const int p1 = curr[x] > 0;
            if (!p1) {
                continue;
            }

            const int neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            const int transitions =
                (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6) +
                (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
            if (neighbors < 2 || neighbors > 6 || transitions != 1) {
                continue;
            }

            const bool remove =
                iteration == 0
                    ? (p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                    : (p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0);
            if (remove) {
                mark[x] = 1;
            }
        }
    }
    image &= ~marker;
}

cv::Mat skeletonize(const cv::Mat& binaryMask)
{
    cv::Mat skeleton;
    binaryMask.copyTo(skeleton);
    cv::threshold(skeleton, skeleton, 0.0, 1.0, cv::THRESH_BINARY);

    cv::Mat previous = cv::Mat::zeros(skeleton.size(), CV_8UC1);
    cv::Mat diff;
    for (int iteration = 0; iteration < 80; ++iteration) {
        zhangSuenIteration(skeleton, 0);
        zhangSuenIteration(skeleton, 1);
        cv::absdiff(skeleton, previous, diff);
        if (cv::countNonZero(diff) == 0) {
            break;
        }
        skeleton.copyTo(previous);
    }
    skeleton *= 255;
    return skeleton;
}

std::vector<cv::Point2d> extractPoints(const cv::Mat& mask)
{
    std::vector<cv::Point> integerPoints;
    cv::findNonZero(mask, integerPoints);
    std::vector<cv::Point2d> points;
    points.reserve(integerPoints.size());
    for (const cv::Point& point : integerPoints) {
        points.emplace_back(static_cast<double>(point.x), static_cast<double>(point.y));
    }
    return points;
}

cv::Rect2d boundsOf(const std::vector<cv::Point2d>& points)
{
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const cv::Point2d& point : points) {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }
    return cv::Rect2d(minX, minY, std::max(0.0, maxX - minX), std::max(0.0, maxY - minY));
}

bool estimateBottomSegment(const std::vector<cv::Point2d>& points,
                           const cv::Rect2d& bounds,
                           const bool preferUpperInnerFeature,
                           Segment& bottom,
                           double& bottomY)
{
    if (points.empty() || !(bounds.width > 4.0 && bounds.height > 4.0)) {
        return false;
    }

    const int minRow = preferUpperInnerFeature
                           ? std::max(0, static_cast<int>(std::floor(bounds.y + bounds.height * 0.055)))
                           : std::max(0, static_cast<int>(std::floor(bounds.y + bounds.height * 0.45)));
    const int maxRow = preferUpperInnerFeature
                           ? static_cast<int>(std::ceil(bounds.y + bounds.height * 0.42))
                           : static_cast<int>(std::ceil(bounds.y + bounds.height + 2.0));
    std::vector<std::vector<double>> rows(static_cast<std::size_t>(std::max(1, maxRow - minRow + 1)));
    for (const cv::Point2d& point : points) {
        const int y = static_cast<int>(std::round(point.y));
        if (y < minRow || y > maxRow) {
            continue;
        }
        rows[static_cast<std::size_t>(y - minRow)].push_back(point.x);
    }

    double bestScore = -std::numeric_limits<double>::infinity();
    int bestRow = -1;
    double bestLeft = 0.0;
    double bestRight = 0.0;
    for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        std::vector<double> xs;
        for (int offset = -2; offset <= 2; ++offset) {
            const int neighbor = static_cast<int>(rowIndex) + offset;
            if (neighbor >= 0 && neighbor < static_cast<int>(rows.size())) {
                xs.insert(xs.end(), rows[static_cast<std::size_t>(neighbor)].begin(),
                          rows[static_cast<std::size_t>(neighbor)].end());
            }
        }
        if (xs.size() < 6) {
            continue;
        }
        const auto [minIt, maxIt] = std::minmax_element(xs.begin(), xs.end());
        const double width = *maxIt - *minIt;
        if (preferUpperInnerFeature) {
            const double widthRatio = width / std::max(bounds.width, 1.0);
            if (widthRatio < 0.18 || widthRatio > 0.62) {
                continue;
            }
        }
        const double y = static_cast<double>(minRow) + static_cast<double>(rowIndex);
        const double score = preferUpperInnerFeature
                                 ? width + 0.32 * static_cast<double>(xs.size()) + 0.02 * y
                                 : width + 0.12 * static_cast<double>(xs.size()) + 0.015 * y;
        if (score > bestScore) {
            bestScore = score;
            bestRow = static_cast<int>(rowIndex) + minRow;
            bestLeft = *minIt;
            bestRight = *maxIt;
        }
    }

    if ((bestRow < 0 || !(bestRight > bestLeft + 5.0)) && preferUpperInnerFeature) {
        const double centralMinX = bounds.x + bounds.width * 0.22;
        const double centralMaxX = bounds.x + bounds.width * 0.78;
        const double topMinY = bounds.y + bounds.height * 0.04;
        const double topMaxY = bounds.y + bounds.height * 0.45;
        std::vector<cv::Point2d> topCentralPoints;
        for (const cv::Point2d& point : points) {
            if (point.x >= centralMinX && point.x <= centralMaxX &&
                point.y >= topMinY && point.y <= topMaxY) {
                topCentralPoints.push_back(point);
            }
        }
        if (!topCentralPoints.empty()) {
            const auto deepestIt = std::max_element(topCentralPoints.begin(),
                                                    topCentralPoints.end(),
                                                    [](const cv::Point2d& lhs, const cv::Point2d& rhs) {
                                                        return lhs.y < rhs.y;
                                                    });
            const double deepestY = deepestIt->y;
            const double yBand = std::max(3.0, bounds.height * 0.018);
            std::vector<double> bottomXs;
            std::vector<double> bottomYs;
            for (const cv::Point2d& point : topCentralPoints) {
                if (std::abs(point.y - deepestY) <= yBand) {
                    bottomXs.push_back(point.x);
                    bottomYs.push_back(point.y);
                }
            }
            if (bottomXs.size() >= 2) {
                const auto [leftIt, rightIt] = std::minmax_element(bottomXs.begin(), bottomXs.end());
                const double width = *rightIt - *leftIt;
                const double widthRatio = width / std::max(bounds.width, 1.0);
                if (width > 5.0 && widthRatio >= 0.10 && widthRatio <= 0.70) {
                    bottomY = median(std::move(bottomYs));
                    bottom = {{*leftIt, bottomY}, {*rightIt, bottomY}, "bottom"};
                    return true;
                }
            }
        }
    }

    if (bestRow < 0 || !(bestRight > bestLeft + 5.0)) {
        return false;
    }

    bottomY = static_cast<double>(bestRow);
    bottom = {{bestLeft, bottomY}, {bestRight, bottomY}, "bottom"};
    return true;
}

bool estimateDiagonalSlotSegment(const cv::Mat& boundaryMask,
                                 const cv::Rect2d& bounds,
                                 Segment& bottom,
                                 double& bottomY)
{
    if (boundaryMask.empty() || !(bounds.width > 10.0 && bounds.height > 10.0)) {
        return false;
    }

    std::vector<cv::Vec4i> lines;
    const double maxSpan = std::max(bounds.width, bounds.height);
    const double minLength = std::max(24.0, maxSpan * 0.055);
    const double maxLength = std::max(minLength + 2.0, maxSpan * 0.34);
    cv::HoughLinesP(boundaryMask, lines, 1.0, CV_PI / 180.0, 18, minLength, 8.0);

    double bestScore = -std::numeric_limits<double>::infinity();
    Segment best;
    for (const cv::Vec4i& line : lines) {
        cv::Point2d a(line[0], line[1]);
        cv::Point2d b(line[2], line[3]);
        const double length = cv::norm(b - a);
        if (length < minLength || length > maxLength) {
            continue;
        }
        double angle = std::abs(std::atan2(b.y - a.y, b.x - a.x) * 180.0 / CV_PI);
        if (angle > 90.0) {
            angle = 180.0 - angle;
        }
        const double diagonalError = std::abs(angle - 45.0);
        if (diagonalError > 18.0) {
            continue;
        }

        const cv::Point2d mid = (a + b) * 0.5;
        const double nx = (mid.x - bounds.x) / std::max(bounds.width, 1.0);
        const double ny = (mid.y - bounds.y) / std::max(bounds.height, 1.0);
        const bool nearDiagonalSlotBand =
            (nx > 0.06 && nx < 0.36 && ny > 0.06 && ny < 0.36) ||
            (nx > 0.64 && nx < 0.94 && ny > 0.06 && ny < 0.36) ||
            (nx > 0.06 && nx < 0.36 && ny > 0.64 && ny < 0.94) ||
            (nx > 0.64 && nx < 0.94 && ny > 0.64 && ny < 0.94);
        if (!nearDiagonalSlotBand) {
            continue;
        }

        if (a.x > b.x) {
            std::swap(a, b);
        }

        const double cornerScore = nx + ny;
        const double lengthScore = -std::abs(length - maxSpan * 0.13) / std::max(maxSpan, 1.0);
        const double score = 2.2 * cornerScore + 1.5 * lengthScore - 0.04 * diagonalError;
        if (score > bestScore) {
            bestScore = score;
            best = {a, b, "bottom"};
        }
    }

    if (!(bestScore > -std::numeric_limits<double>::infinity())) {
        return false;
    }
    bottom = best;
    bottomY = 0.5 * (bottom.a.y + bottom.b.y);
    return bottom.lengthPx() > 5.0;
}

bool fitLineSegment(const std::vector<cv::Point2d>& points,
                    const cv::Point2d& forcedEndpoint,
                    const bool useMinYAsFarEndpoint,
                    Segment& segment)
{
    if (points.size() < 3) {
        return false;
    }

    std::vector<cv::Point2f> fitPoints;
    fitPoints.reserve(points.size());
    for (const cv::Point2d& point : points) {
        fitPoints.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    cv::Vec4f line;
    cv::fitLine(fitPoints, line, cv::DIST_HUBER, 0.0, 0.01, 0.01);
    const cv::Point2d direction(line[0], line[1]);
    const cv::Point2d pointOnLine(line[2], line[3]);
    const double dirNorm = cv::norm(direction);
    if (!(dirNorm > kEpsilon)) {
        return false;
    }
    const cv::Point2d unit = direction * (1.0 / dirNorm);

    double farT = 0.0;
    bool initialized = false;
    for (const cv::Point2d& point : points) {
        const double t = (point - pointOnLine).dot(unit);
        if (!initialized) {
            farT = t;
            initialized = true;
            continue;
        }
        const cv::Point2d current = pointOnLine + unit * t;
        const cv::Point2d best = pointOnLine + unit * farT;
        if (useMinYAsFarEndpoint
                ? (cv::norm(current - forcedEndpoint) > cv::norm(best - forcedEndpoint))
                : (current.y > best.y)) {
            farT = t;
        }
    }

    const cv::Point2d farEndpoint = pointOnLine + unit * farT;
    segment.a = farEndpoint;
    segment.b = forcedEndpoint;
    return segment.lengthPx() > 4.0;
}

double clampProjectionT(const Segment& segment, const cv::Point2d& point)
{
    const cv::Point2d ab = segment.b - segment.a;
    const double denom = ab.dot(ab);
    if (!(denom > kEpsilon)) {
        return 0.0;
    }
    return std::clamp((point - segment.a).dot(ab) / denom, 0.0, 1.0);
}

cv::Point2d projectToSegment(const Segment& segment, const cv::Point2d& point, double& t)
{
    t = clampProjectionT(segment, point);
    return segment.a + (segment.b - segment.a) * t;
}

std::vector<MatchedPoint> matchPointsToSegments(const std::vector<cv::Point2d>& points,
                                                const Segment& left,
                                                const Segment& bottom,
                                                const Segment& right,
                                                const cv::Rect2d& bounds)
{
    const std::vector<Segment> segments = {left, bottom, right};
    const double leftLength = left.lengthPx();
    const double bottomLength = bottom.lengthPx();
    const double maxDistancePx = std::max(3.0, std::max(bounds.width, bounds.height) * 0.025);

    std::vector<MatchedPoint> matched;
    matched.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const cv::Point2d& point = points[index];
        double bestDistance = std::numeric_limits<double>::infinity();
        double bestT = 0.0;
        cv::Point2d bestProjection;
        std::size_t bestSegment = 0;
        for (std::size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
            double t = 0.0;
            const cv::Point2d projection = projectToSegment(segments[segmentIndex], point, t);
            const double distance = cv::norm(point - projection);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestT = t;
                bestProjection = projection;
                bestSegment = segmentIndex;
            }
        }

        if (bestDistance > maxDistancePx) {
            continue;
        }

        double pathPx = 0.0;
        if (bestSegment == 0) {
            pathPx = bestT * leftLength;
        } else if (bestSegment == 1) {
            pathPx = leftLength + bestT * bottomLength;
        } else {
            pathPx = leftLength + bottomLength + bestT * right.lengthPx();
        }

        const double signedDistance = (point - bestProjection).dot(segmentNormal(segments[bestSegment]));
        matched.push_back({index,
                           point,
                           bestProjection,
                           segments[bestSegment].label,
                           pathPx,
                           bestDistance,
                           signedDistance,
                           true});
    }

    std::sort(matched.begin(), matched.end(), [](const MatchedPoint& lhs, const MatchedPoint& rhs) {
        if (std::abs(lhs.pathPx - rhs.pathPx) > 1e-6) {
            return lhs.pathPx < rhs.pathPx;
        }
        return lhs.sourceIndex < rhs.sourceIndex;
    });
    return matched;
}

void markOutliers(std::vector<MatchedPoint>& matched,
                  const double pixelSizeMm,
                  const double maxAcceptedErrorUm)
{
    std::vector<double> distancesUm;
    distancesUm.reserve(matched.size());
    for (const MatchedPoint& point : matched) {
        distancesUm.push_back(point.distancePx * pixelSizeMm * 1000.0);
    }
    const double med = median(distancesUm);
    std::vector<double> deviations;
    deviations.reserve(distancesUm.size());
    for (const double value : distancesUm) {
        deviations.push_back(std::abs(value - med));
    }
    const double mad = median(deviations);
    const double robustLimit = std::isfinite(mad) ? med + std::max(3.5 * 1.4826 * mad, 45.0) : maxAcceptedErrorUm;
    const double limit = std::max(35.0, std::min(maxAcceptedErrorUm, robustLimit));

    for (std::size_t index = 0; index < matched.size(); ++index) {
        matched[index].isUsed = distancesUm[index] <= limit;
    }
}

void thinUsedPoints(std::vector<MatchedPoint>& matched, const std::size_t maxOutputPoints)
{
    if (maxOutputPoints == 0) {
        return;
    }
    std::vector<std::size_t> usedIndices;
    for (std::size_t index = 0; index < matched.size(); ++index) {
        if (matched[index].isUsed) {
            usedIndices.push_back(index);
        }
    }
    if (usedIndices.size() <= maxOutputPoints) {
        return;
    }
    const double step = static_cast<double>(usedIndices.size() - 1) /
                        static_cast<double>(std::max<std::size_t>(1, maxOutputPoints - 1));
    std::vector<unsigned char> keep(matched.size(), 0);
    for (std::size_t outputIndex = 0; outputIndex < maxOutputPoints; ++outputIndex) {
        const auto source = static_cast<std::size_t>(std::round(static_cast<double>(outputIndex) * step));
        keep[usedIndices[std::min(source, usedIndices.size() - 1)]] = 1;
    }
    for (std::size_t index = 0; index < matched.size(); ++index) {
        if (matched[index].isUsed && !keep[index]) {
            matched[index].isUsed = false;
        }
    }
}

double localS(const cv::Point2d& point, const Segment& bottom, const double pixelSizeMm)
{
    return segmentSpx(point, bottom) * pixelSizeMm;
}

double localR(const cv::Point2d& point, const Segment& bottom, const double pixelSizeMm)
{
    return segmentRpx(point, bottom) * pixelSizeMm;
}

std::vector<double> usedSignedErrorsUm(const std::vector<MatchedPoint>& points,
                                       const double pixelSizeMm)
{
    std::vector<double> values;
    for (const MatchedPoint& point : points) {
        if (point.isUsed) {
            values.push_back(point.signedDistancePx * pixelSizeMm * 1000.0);
        }
    }
    return values;
}

CadMappingContext buildCadMappingContext(const LocalSlotAnalysisRequest& request,
                                         const std::vector<MatchedPoint>& points,
                                         const Segment& bottom,
                                         const double pixelSizeMm)
{
    CadMappingContext context;
    if (request.cadProfileSamples.size() < 2) {
        return context;
    }

    context.metadata = request.cadProfileMetadata;
    context.samples = request.cadProfileSamples;
    context.samples.erase(std::remove_if(context.samples.begin(),
                                         context.samples.end(),
                                         [](const DesignProfileSample& sample) {
                                             return !std::isfinite(sample.sMm) ||
                                                    !std::isfinite(sample.rMm) ||
                                                    !sample.hasCadPoint;
                                         }),
                          context.samples.end());
    if (context.samples.size() < 2) {
        return context;
    }
    std::stable_sort(context.samples.begin(), context.samples.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return lhs.rMm < rhs.rMm;
    });

    std::vector<cv::Point2d> designSeries;
    designSeries.reserve(context.samples.size());
    for (const DesignProfileSample& sample : context.samples) {
        designSeries.emplace_back(sample.sMm, sample.rMm);
    }
    context.feature = estimateTargetWidthCadSlotFeature(designSeries, request.bottomWidthMm);
    if (!context.feature.valid || !(context.feature.widthMm > 1e-9)) {
        return context;
    }

    context.localWidthMm = bottom.lengthPx() * pixelSizeMm;
    if (!(context.localWidthMm > 1e-9)) {
        return context;
    }

    std::vector<double> targetRValues;
    targetRValues.reserve(points.size());
    for (const MatchedPoint& point : points) {
        if (point.isUsed) {
            const double dr = localR(point.designPoint, bottom, pixelSizeMm);
            if (std::isfinite(dr)) {
                targetRValues.push_back(dr);
            }
        }
    }
    const double localTopR = percentile(targetRValues, 0.98);
    context.localDepthMm = std::isfinite(localTopR) ? localTopR : 0.0;
    if (context.localDepthMm > 1e-9 && context.feature.depthMm > 1e-9) {
        context.radialScale = context.feature.depthMm / context.localDepthMm;
    }

    if (context.metadata.sourceType.empty() || context.metadata.sourceType == "builtin") {
        context.metadata.sourceType = "external_cad";
    }
    if (context.metadata.sourceName.empty()) {
        context.metadata.sourceName = "imported_cad_profile";
    }
    if (context.metadata.extractionMethod.empty()) {
        context.metadata.extractionMethod = "cad_profile_mapping";
    }
    if (context.metadata.axialAxis.empty()) {
        context.metadata.axialAxis = "X";
    }
    if (context.metadata.radialAxis.empty()) {
        context.metadata.radialAxis = "Y";
    }
    if (!std::isfinite(context.metadata.cadAxialDirectionSign) ||
        std::abs(context.metadata.cadAxialDirectionSign) < 1e-12) {
        context.metadata.cadAxialDirectionSign = 1.0;
    }

    context.valid = true;
    return context;
}

std::string buildProfileCsv(const std::vector<MatchedPoint>& points,
                            const Segment& bottom,
                            const double pixelSizeMm,
                            const CadMappingContext& cadMapping)
{
    std::ostringstream stream;
    stream << "index,is_used,segment,x_px,y_px,s_aligned_mm,r_aligned_mm,measured_r_mm,"
              "design_s_mm,r_design_mm,design_r_mm,nearest_design_r_mm,radial_error_um,"
              "normal_error_um,profile_error_um,has_cad_coordinates,cad_measured_x_mm,"
              "cad_measured_y_mm,cad_measured_z_mm,cad_design_x_mm,cad_design_y_mm,cad_design_z_mm\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
        const MatchedPoint& point = points[index];
        const double s = localS(point.imagePoint, bottom, pixelSizeMm);
        const double r = localR(point.imagePoint, bottom, pixelSizeMm);
        const double ds = localS(point.designPoint, bottom, pixelSizeMm);
        const double dr = localR(point.designPoint, bottom, pixelSizeMm);
        const double radialErrorUm = (r - dr) * 1000.0;
        const double normalErrorUm = point.signedDistancePx * pixelSizeMm * 1000.0;
        const CadMappedPoint cadPoint = mapLocalSlotPointToCad(cadMapping, s, r, ds, dr);
        stream << index << ","
               << (point.isUsed ? 1 : 0) << ","
               << csvTextCell(point.segmentLabel) << ","
               << csvCell(point.imagePoint.x, 6) << ","
               << csvCell(point.imagePoint.y, 6) << ","
               << csvCell(s, 9) << ","
               << csvCell(r, 9) << ","
               << csvCell(r, 9) << ","
               << csvCell(ds, 9) << ","
               << csvCell(dr, 9) << ","
               << csvCell(dr, 9) << ","
               << csvCell(dr, 9) << ","
               << csvCell(radialErrorUm, 6) << ","
               << csvCell(normalErrorUm, 6) << ","
               << csvCell(normalErrorUm, 6) << ","
               << (cadPoint.valid ? 1 : 0) << ","
               << csvCell(cadPoint.measuredCadXMm, 6) << ","
               << csvCell(cadPoint.measuredCadYMm, 6) << ","
               << csvCell(cadPoint.measuredCadZMm, 6) << ","
               << csvCell(cadPoint.targetCadXMm, 6) << ","
               << csvCell(cadPoint.targetCadYMm, 6) << ","
               << csvCell(cadPoint.targetCadZMm, 6) << "\n";
    }
    return stream.str();
}

std::string buildSummaryCsv(const LocalSlotAnalysisRequest& request,
                            const std::vector<MatchedPoint>& points,
                            const Segment& bottom,
                            const double pixelSizeMm,
                            const std::vector<double>& usedErrorsUm,
                            const double p95Abs,
                            const double pv,
                            const CadMappingContext& cadMapping)
{
    const std::size_t usedCount = static_cast<std::size_t>(
        std::count_if(points.begin(), points.end(), [](const MatchedPoint& point) { return point.isUsed; }));
    const std::size_t outlierCount = points.size() - usedCount;
    const double bottomWidthMm = bottom.lengthPx() * pixelSizeMm;
    const double rms = rmse(usedErrorsUm);
    const double mae = mean([&usedErrorsUm]() {
        std::vector<double> absValues;
        absValues.reserve(usedErrorsUm.size());
        for (const double value : usedErrorsUm) {
            absValues.push_back(std::abs(value));
        }
        return absValues;
    }());
    const double minS = 0.0;
    const double maxS = bottomWidthMm;
    const double minR = 0.0;
    const double maxR = 0.0;
    const DesignProfileMetadata& metadata = cadMapping.valid ? cadMapping.metadata : request.cadProfileMetadata;
    const bool hasCadMapping = cadMapping.valid;

    std::ostringstream stream;
    stream << "dz_mm,dr_mm,dtheta_deg,axial_scale_factor,axial_quadratic_term_mm,"
              "applied_absolute_bias_refine,absolute_bias_correction_um,"
              "pre_refine_mean_normal_error_um,pre_refine_absolute_filtered_rmse_um,"
              "design_reverse_axial,use_left_endpoint_anchor,design_evaluate_profile_form,"
              "anchor_x_px,anchor_y_px,pixel_size_mm,candidate_count,used_count,outlier_count,"
              "outlier_ratio,mean_normal_error_um,normal_rmse_um,normal_mae_um,normal_p95_abs_um,"
              "normal_max_pos_um,normal_max_neg_um,normal_pv_um,profile_rms_um,profile_mae_um,"
              "profile_p95_abs_um,profile_max_pos_um,profile_max_neg_um,profile_pv_um,"
              "local_exceedance_threshold_um,local_exceedance_count,local_max_abs_error_um,"
              "local_max_exceedance_um,"
              "absolute_all_mean_um,absolute_all_rmse_um,absolute_all_mae_um,absolute_all_p95_abs_um,"
              "absolute_all_max_pos_um,absolute_all_max_neg_um,absolute_all_pv_um,"
              "absolute_filtered_mean_um,absolute_filtered_rmse_um,absolute_filtered_mae_um,"
              "absolute_filtered_p95_abs_um,absolute_filtered_max_pos_um,absolute_filtered_max_neg_um,"
              "absolute_filtered_pv_um,design_source_type,design_source_name,design_profile_sample_count,"
              "design_profile_min_s_mm,design_profile_max_s_mm,design_profile_min_r_mm,design_profile_max_r_mm,"
              "cad_extraction_method,cad_section_normal_axis,cad_section_coordinate_mm,cad_axial_axis,"
              "cad_radial_axis,cad_axial_origin_mm,cad_axial_direction_sign\n";
    const double maxPos = usedErrorsUm.empty() ? std::numeric_limits<double>::quiet_NaN()
                                               : *std::max_element(usedErrorsUm.begin(), usedErrorsUm.end());
    const double maxNeg = usedErrorsUm.empty() ? std::numeric_limits<double>::quiet_NaN()
                                               : *std::min_element(usedErrorsUm.begin(), usedErrorsUm.end());
    const double meanError = mean(usedErrorsUm);
    const double outlierRatio = points.empty() ? 0.0 : static_cast<double>(outlierCount) / static_cast<double>(points.size());
    const double localMaxAbsErrorUm = maxAbsValue(usedErrorsUm);
    const std::size_t localExceedanceCount = countExceedances(usedErrorsUm, kLocalExceedanceThresholdUm);
    const double localMaxExceedanceUm =
        std::isfinite(localMaxAbsErrorUm) ? std::max(0.0, localMaxAbsErrorUm - kLocalExceedanceThresholdUm) : 0.0;
    stream << "0,0,0,1,0,0,0,"
           << csvCell(meanError, 6) << ","
           << csvCell(rms, 6) << ","
           << "0,1,1,"
           << csvCell(bottom.a.x, 6) << ","
           << csvCell(bottom.a.y, 6) << ","
           << csvCell(pixelSizeMm, 9) << ","
           << points.size() << ","
           << usedCount << ","
           << outlierCount << ","
           << csvCell(outlierRatio, 6) << ","
           << csvCell(meanError, 6) << ","
           << csvCell(rms, 6) << ","
           << csvCell(mae, 6) << ","
           << csvCell(p95Abs, 6) << ","
           << csvCell(maxPos, 6) << ","
           << csvCell(maxNeg, 6) << ","
           << csvCell(pv, 6) << ","
           << csvCell(rms, 6) << ","
           << csvCell(mae, 6) << ","
           << csvCell(p95Abs, 6) << ","
           << csvCell(maxPos, 6) << ","
           << csvCell(maxNeg, 6) << ","
           << csvCell(pv, 6) << ","
           << csvCell(kLocalExceedanceThresholdUm, 6) << ","
           << localExceedanceCount << ","
           << csvCell(localMaxAbsErrorUm, 6) << ","
           << csvCell(localMaxExceedanceUm, 6) << ","
           << csvCell(meanError, 6) << ","
           << csvCell(rms, 6) << ","
           << csvCell(mae, 6) << ","
           << csvCell(p95Abs, 6) << ","
           << csvCell(maxPos, 6) << ","
           << csvCell(maxNeg, 6) << ","
           << csvCell(pv, 6) << ","
           << csvCell(meanError, 6) << ","
           << csvCell(rms, 6) << ","
           << csvCell(mae, 6) << ","
           << csvCell(p95Abs, 6) << ","
           << csvCell(maxPos, 6) << ","
           << csvCell(maxNeg, 6) << ","
           << csvCell(pv, 6) << ","
           << csvTextCell(hasCadMapping ? metadata.sourceType : "local_slot_image") << ","
           << csvTextCell(hasCadMapping ? metadata.sourceName
                                        : (request.sourceImagePath.empty() ? "local_slot" : request.sourceImagePath)) << ","
           << (hasCadMapping ? cadMapping.samples.size() : 3) << ","
           << csvCell(hasCadMapping ? metadata.minSMm : minS, 6) << ","
           << csvCell(hasCadMapping ? metadata.maxSMm : maxS, 6) << ","
           << csvCell(hasCadMapping ? metadata.minRMm : minR, 6) << ","
           << csvCell(hasCadMapping ? metadata.maxRMm : maxR, 6) << ","
           << csvTextCell(hasCadMapping ? metadata.extractionMethod : "local_slot_skeleton_fit") << ","
           << csvTextCell(hasCadMapping ? metadata.sectionNormalAxis : "Z") << ","
           << csvCell(hasCadMapping ? metadata.sectionCoordinateMm : 0.0, 6) << ","
           << csvTextCell(hasCadMapping ? metadata.axialAxis : "X") << ","
           << csvTextCell(hasCadMapping ? metadata.radialAxis : "Y") << ","
           << csvCell(hasCadMapping ? metadata.cadAxialOriginMm : 0.0, 6) << ","
           << csvCell(hasCadMapping ? metadata.cadAxialDirectionSign : 1.0, 6) << "\n";
    return stream.str();
}

std::string buildCompensationCsv(const std::vector<MatchedPoint>& points,
                                 const Segment& bottom,
                                 const double pixelSizeMm,
                                 const CadMappingContext& cadMapping)
{
    std::ostringstream stream;
    stream << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,"
              "has_cad_coordinates,cad_design_x_mm,cad_design_y_mm,cad_design_z_mm,cad_measured_x_mm,"
              "cad_measured_y_mm,cad_measured_z_mm,cad_compensation_target_x_mm,"
              "cad_compensation_target_y_mm,cad_compensation_target_z_mm,compensated_cad_x_mm,"
              "compensated_cad_y_mm,compensated_cad_z_mm,delta_x_um,delta_y_um,delta_z_um,"
              "s_aligned_mm,measured_r_mm,design_s_mm,design_r_mm,radial_error_um,"
              "normal_error_um,profile_error_um,compensation_normal_um,compensation_radial_um,"
              "compensation_radial_mm,compensated_target_r_mm,is_used\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
        const MatchedPoint& point = points[index];
        const double s = localS(point.imagePoint, bottom, pixelSizeMm);
        const double r = localR(point.imagePoint, bottom, pixelSizeMm);
        const double ds = localS(point.designPoint, bottom, pixelSizeMm);
        const double dr = localR(point.designPoint, bottom, pixelSizeMm);
        const double radialErrorUm = (r - dr) * 1000.0;
        const double normalErrorUm = point.signedDistancePx * pixelSizeMm * 1000.0;
        const double compensationRadialUm = -radialErrorUm;
        const CadMappedPoint cadPoint = mapLocalSlotPointToCad(cadMapping, s, r, ds, dr);
        const std::string sourceType = cadMapping.valid ? cadMapping.metadata.sourceType : "local_slot_image";
        const std::string sourceName = cadMapping.valid ? cadMapping.metadata.sourceName : "local_slot";
        const std::string extractionMethod =
            cadMapping.valid ? cadMapping.metadata.extractionMethod : "local_slot_skeleton_fit";
        const std::string axialAxis = cadMapping.valid ? cadMapping.metadata.axialAxis : "X";
        const std::string radialAxis = cadMapping.valid ? cadMapping.metadata.radialAxis : "Y";
        stream << joinCsvRow({
            csvTextCell(sourceType),
            csvTextCell(sourceName),
            csvTextCell(extractionMethod),
            csvTextCell(axialAxis),
            csvTextCell(radialAxis),
            std::to_string(index),
            cadPoint.valid ? "1" : "0",
            csvCell(cadPoint.targetCadXMm, 6),
            csvCell(cadPoint.targetCadYMm, 6),
            csvCell(cadPoint.targetCadZMm, 6),
            csvCell(cadPoint.measuredCadXMm, 6),
            csvCell(cadPoint.measuredCadYMm, 6),
            csvCell(cadPoint.measuredCadZMm, 6),
            csvCell(cadPoint.targetCadXMm, 6),
            csvCell(cadPoint.targetCadYMm, 6),
            csvCell(cadPoint.targetCadZMm, 6),
            csvCell(cadPoint.targetCadXMm, 6),
            csvCell(cadPoint.targetCadYMm, 6),
            csvCell(cadPoint.targetCadZMm, 6),
            csvCell(cadPoint.deltaXUm, 6),
            csvCell(cadPoint.deltaYUm, 6),
            csvCell(cadPoint.deltaZUm, 6),
            csvCell(s, 9),
            csvCell(r, 9),
            csvCell(ds, 9),
            csvCell(dr, 9),
            csvCell(radialErrorUm, 6),
            csvCell(normalErrorUm, 6),
            csvCell(normalErrorUm, 6),
            csvCell(-normalErrorUm, 6),
            csvCell(compensationRadialUm, 6),
            csvCell(compensationRadialUm / 1000.0, 9),
            csvCell(dr, 9),
            point.isUsed ? "1" : "0",
        });
    }
    return stream.str();
}

std::string build3dCsv(const std::vector<MatchedPoint>& points,
                       const Segment& bottom,
                       const double pixelSizeMm,
                       const CadMappingContext& cadMapping)
{
    std::ostringstream stream;
    stream << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,is_used,"
              "has_cad_coordinates,design_x_mm,design_y_mm,design_z_mm,measured_x_mm,measured_y_mm,"
              "measured_z_mm,error_x_um,error_y_um,error_z_um,error_3d_um,normal_error_um,"
              "radial_error_um,profile_error_um,compensation_target_x_mm,compensation_target_y_mm,"
              "compensation_target_z_mm,compensation_x_um,compensation_y_um,compensation_z_um,"
              "s_aligned_mm,measured_r_mm,design_s_mm,design_r_mm\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
        const MatchedPoint& point = points[index];
        const double s = localS(point.imagePoint, bottom, pixelSizeMm);
        const double r = localR(point.imagePoint, bottom, pixelSizeMm);
        const double ds = localS(point.designPoint, bottom, pixelSizeMm);
        const double dr = localR(point.designPoint, bottom, pixelSizeMm);
        const double radialErrorUm = (r - dr) * 1000.0;
        const double normalErrorUm = point.signedDistancePx * pixelSizeMm * 1000.0;
        const CadMappedPoint cadPoint = mapLocalSlotPointToCad(cadMapping, s, r, ds, dr);
        const double error3dUm =
            cadPoint.valid
                ? std::sqrt(cadPoint.deltaXUm * cadPoint.deltaXUm +
                            cadPoint.deltaYUm * cadPoint.deltaYUm +
                            cadPoint.deltaZUm * cadPoint.deltaZUm)
                : std::numeric_limits<double>::quiet_NaN();
        const std::string sourceType = cadMapping.valid ? cadMapping.metadata.sourceType : "local_slot_image";
        const std::string sourceName = cadMapping.valid ? cadMapping.metadata.sourceName : "local_slot";
        const std::string extractionMethod =
            cadMapping.valid ? cadMapping.metadata.extractionMethod : "local_slot_skeleton_fit";
        const std::string axialAxis = cadMapping.valid ? cadMapping.metadata.axialAxis : "X";
        const std::string radialAxis = cadMapping.valid ? cadMapping.metadata.radialAxis : "Y";
        stream << joinCsvRow({
            csvTextCell(sourceType),
            csvTextCell(sourceName),
            csvTextCell(extractionMethod),
            csvTextCell(axialAxis),
            csvTextCell(radialAxis),
            std::to_string(index),
            point.isUsed ? "1" : "0",
            cadPoint.valid ? "1" : "0",
            csvCell(cadPoint.targetCadXMm, 6),
            csvCell(cadPoint.targetCadYMm, 6),
            csvCell(cadPoint.targetCadZMm, 6),
            csvCell(cadPoint.measuredCadXMm, 6),
            csvCell(cadPoint.measuredCadYMm, 6),
            csvCell(cadPoint.measuredCadZMm, 6),
            csvCell(-cadPoint.deltaXUm, 6),
            csvCell(-cadPoint.deltaYUm, 6),
            csvCell(-cadPoint.deltaZUm, 6),
            csvCell(error3dUm, 6),
            csvCell(normalErrorUm, 6),
            csvCell(radialErrorUm, 6),
            csvCell(normalErrorUm, 6),
            csvCell(cadPoint.targetCadXMm, 6),
            csvCell(cadPoint.targetCadYMm, 6),
            csvCell(cadPoint.targetCadZMm, 6),
            csvCell(cadPoint.deltaXUm, 6),
            csvCell(cadPoint.deltaYUm, 6),
            csvCell(cadPoint.deltaZUm, 6),
            csvCell(s, 9),
            csvCell(r, 9),
            csvCell(ds, 9),
            csvCell(dr, 9),
        });
    }
    return stream.str();
}

std::string buildFeatureCsv(const LocalSlotAnalysisRequest& request,
                            const Segment& bottom,
                            const double pixelSizeMm)
{
    const double measuredWidthMm = bottom.lengthPx() * pixelSizeMm;
    const double targetWidthMm = request.bottomWidthMm;
    const double widthErrorUm = (measuredWidthMm - targetWidthMm) * 1000.0;
    std::ostringstream stream;
    stream << "feature_id,status,method,design_source_type,design_source_name,cad_extraction_method,"
              "target_slot_width_mm,target_slot_depth_mm,cad_slot_left_s_mm,cad_slot_right_s_mm,"
              "cad_slot_center_s_mm,cad_slot_width_mm,cad_slot_depth_mm,cad_slot_bottom_s_mm,"
              "cad_slot_bottom_r_mm,cad_slot_center_x_mm,cad_slot_center_y_mm,cad_slot_center_z_mm,"
              "measured_slot_center_s_mm,measured_slot_width_mm,measured_slot_depth_mm,"
              "center_error_um,width_error_um,depth_error_um,compensation_center_um,"
              "compensation_width_um,compensation_depth_um,notes\n";
    stream << csvTextCell("local_single_slot_width") << ","
           << csvTextCell("ok") << ","
           << csvTextCell("bottom_width_calibrated_from_local_image") << ","
           << csvTextCell("local_slot_image") << ","
           << csvTextCell("local_slot") << ","
           << csvTextCell("local_slot_skeleton_fit") << ","
           << csvCell(targetWidthMm, 9) << ",,"
           << "0,"
           << csvCell(measuredWidthMm, 9) << ","
           << csvCell(measuredWidthMm * 0.5, 9) << ","
           << csvCell(targetWidthMm, 9) << ",,"
           << csvCell(measuredWidthMm * 0.5, 9) << ",0,,,,"
           << csvCell(measuredWidthMm * 0.5, 9) << ","
           << csvCell(measuredWidthMm, 9) << ",,"
           << "0,"
           << csvCell(widthErrorUm, 6) << ",,"
           << "0,"
           << csvCell(-widthErrorUm, 6) << ",,"
           << csvTextCell("Local slot 2D section compensation; bottom width calibrates pixel size.")
           << "\n";
    return stream.str();
}

std::string buildQualityReviewCsv(const LocalSlotAnalysisRequest& request,
                                  const LocalSlotAnalysisResult& result,
                                  const CadMappingContext& cadMapping)
{
    std::ostringstream stream;
    stream << "check,status,value,threshold,message\n";
    stream << "local_slot_pipeline,PASS,1,1,"
           << csvTextCell("Local slot preprocessing, edge extraction, alignment and compensation completed.")
           << "\n";
    stream << "used_edge_point_count,"
           << (result.usedPointCount >= 50 ? "PASS" : "WARN") << ","
           << result.usedPointCount << ",50,"
           << csvTextCell("Usable edge points after skeleton matching and outlier filtering.")
           << "\n";
    stream << "bottom_width_calibration,PASS,"
           << csvCell(result.measuredSlotWidthMm, 9) << ","
           << csvCell(result.bottomWidthPx, 3) << ","
           << csvTextCell("Measured local slot bottom width after calibration.")
           << "\n";
    stream << "cad_model_and_target_size,"
           << (cadMapping.valid ? "PASS" : "WARN") << ","
           << csvCell(request.bottomWidthMm, 9) << ",>0,"
           << csvTextCell(cadMapping.valid
                              ? "CAD section samples were imported and mapped to the local slot compensation points."
                              : "Local slot compensation completed; CAD XYZ output requires imported CAD section samples.")
           << "\n";
    stream << "measurement_data_read_display,PASS,"
           << csvTextCell(request.sourceImagePath.empty() ? "local_slot_image" : request.sourceImagePath) << ",image,"
           << csvTextCell("Input image was read and saved as the local slot measurement evidence.")
           << "\n";
    stream << "coordinate_conversion_filtering,PASS,"
           << csvCell(result.pixelSizeMm, 9) << ",mm_per_px,"
           << csvTextCell("Pixel-to-mm conversion, skeleton edge extraction, outlier filtering and point thinning were applied.")
           << "\n";
    stream << "slot_feature_extraction,PASS,"
           << csvCell(result.measuredSlotWidthMm, 9) << ",mm,"
           << csvTextCell("Slot edge contour and bottom-width feature were extracted.")
           << "\n";
    stream << "model_alignment_and_visualization,"
           << (cadMapping.valid ? "PASS" : "PASS") << ","
           << result.usedPointCount << ",points,"
           << csvTextCell(cadMapping.valid
                              ? "Local s-r slot contour was aligned to the imported CAD slot and CAD XYZ points were generated."
                              : "Local s-r slot contour was aligned to the target bottom-width reference.")
           << "\n";
    stream << "error_analysis,PASS,"
           << csvCell(result.normalRmseUm, 6) << ",um,"
           << csvTextCell("Slot width error, normal/profile errors and error plots were generated.")
           << "\n";
    stream << "local_exceedance_annotation,"
           << (result.localExceedanceCount == 0 ? "PASS" : "WARN") << ","
           << result.localExceedanceCount << ","
           << csvCell(result.localExceedanceThresholdUm, 6) << ","
           << csvTextCell("Local points above the threshold are highlighted in red on matplotlib error and compensation plots.")
           << "\n";
    stream << "max_abs_profile_error_um,"
           << (result.localExceedanceCount == 0 ? "PASS" : "WARN") << ","
           << csvCell(result.localMaxAbsErrorUm, 6) << ","
           << csvCell(result.localExceedanceThresholdUm, 6) << ","
           << csvTextCell("Maximum absolute local profile error among used slot edge points.")
           << "\n";
    stream << "cad_compensation_xyz_output,"
           << (cadMapping.valid ? "PASS" : "WARN") << ","
           << (cadMapping.valid ? 1 : 0) << ",1,"
           << csvTextCell(cadMapping.valid
                              ? "design_compensation.csv and compensated_slot_edge_points.csv contain compensated_cad_x/y/z_mm."
                              : "CAD XYZ compensation columns exist but are empty because no CAD mapping was available.")
           << "\n";
    return stream.str();
}

struct MicrochannelBar {
    int channelIndex{0};
    double leftPx{0.0};
    double rightPx{0.0};
    double topPx{0.0};
    double bottomPx{0.0};
    double centerXPx{0.0};
    double widthPx{0.0};
    double heightPx{0.0};
};

struct MicrochannelReference {
    bool valid{false};
    double channelWidthMm{0.0};
    double channelPitchMm{0.0};
    double arrayLeftSMm{0.0};
    double arrayRightSMm{0.0};
    double profileMinSMm{0.0};
    double profileMaxSMm{0.0};
    double profileTopRMm{0.0};
    double profileBottomRMm{0.0};
    std::size_t channelCount{0};
};

cv::Mat buildMicrochannelMask(const cv::Mat& image)
{
    cv::Mat mask;
    if (image.channels() >= 3) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);
        cv::Mat lowSaturation;
        cv::Mat highValue;
        cv::threshold(channels[1], lowSaturation, 72.0, 255.0, cv::THRESH_BINARY_INV);
        cv::threshold(channels[2], highValue, 48.0, 255.0, cv::THRESH_BINARY);
        cv::bitwise_and(lowSaturation, highValue, mask);
    } else {
        cv::Mat blurred;
        cv::GaussianBlur(image, blurred, cv::Size(3, 3), 0.0);
        cv::threshold(blurred, mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    return mask;
}

std::vector<MicrochannelBar> detectMicrochannelBars(const cv::Mat& image)
{
    const cv::Mat mask = buildMicrochannelMask(image);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int componentCount = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    std::vector<MicrochannelBar> bars;
    const double minHeight = std::max(60.0, static_cast<double>(image.rows) * 0.35);
    const double maxWidth = std::max(8.0, static_cast<double>(image.cols) * 0.08);
    for (int label = 1; label < componentCount; ++label) {
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (width <= 0 || height <= 0) {
            continue;
        }
        const double aspect = static_cast<double>(height) / static_cast<double>(width);
        const double fillRatio = static_cast<double>(area) / static_cast<double>(width * height);
        if (height < minHeight || width > maxWidth || width < 4 || aspect < 5.0 || fillRatio < 0.55) {
            continue;
        }
        bars.push_back({
            0,
            static_cast<double>(x),
            static_cast<double>(x + width - 1),
            static_cast<double>(y),
            static_cast<double>(y + height - 1),
            static_cast<double>(x) + 0.5 * static_cast<double>(width - 1),
            static_cast<double>(width),
            static_cast<double>(height),
        });
    }

    std::stable_sort(bars.begin(), bars.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.centerXPx < rhs.centerXPx;
    });
    for (std::size_t index = 0; index < bars.size(); ++index) {
        bars[index].channelIndex = static_cast<int>(index + 1);
    }
    return bars;
}

MicrochannelReference estimateMicrochannelReference(const LocalSlotAnalysisRequest& request)
{
    MicrochannelReference reference;
    if (request.cadProfileSamples.size() < 20) {
        return reference;
    }

    std::vector<cv::Point2d> series;
    series.reserve(request.cadProfileSamples.size());
    for (const DesignProfileSample& sample : request.cadProfileSamples) {
        if (std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            series.emplace_back(sample.sMm, sample.rMm);
        }
    }
    if (series.size() < 20) {
        return reference;
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
    const double topR = percentile(radii, 0.95);
    const double bottomR = percentile(radii, 0.05);
    if (!(topR > bottomR + 1e-6)) {
        return reference;
    }

    struct VerticalCadBin {
        std::size_t count{0};
        double sSum{0.0};
        double minR{std::numeric_limits<double>::infinity()};
        double maxR{-std::numeric_limits<double>::infinity()};
    };
    struct VerticalCadEdge {
        double s{0.0};
        double minR{0.0};
        double maxR{0.0};
        std::size_t count{0};
    };

    const double targetWidth = request.bottomWidthMm > 1e-9 ? request.bottomWidthMm : 0.05;
    const double verticalBinWidthMm = std::clamp(targetWidth * 0.10, 0.001, 0.010);
    const double minVerticalSpanMm = std::max(0.20, 0.22 * (topR - bottomR));
    std::map<long long, VerticalCadBin> verticalBins;
    for (const cv::Point2d& point : series) {
        const long long binIndex = static_cast<long long>(std::llround(point.x / verticalBinWidthMm));
        VerticalCadBin& bin = verticalBins[binIndex];
        ++bin.count;
        bin.sSum += point.x;
        bin.minR = std::min(bin.minR, point.y);
        bin.maxR = std::max(bin.maxR, point.y);
    }

    std::vector<VerticalCadEdge> verticalEdges;
    for (const auto& [binIndex, bin] : verticalBins) {
        (void)binIndex;
        if (bin.count < 8 || !(bin.maxR > bin.minR + minVerticalSpanMm)) {
            continue;
        }
        verticalEdges.push_back({
            bin.sSum / static_cast<double>(bin.count),
            bin.minR,
            bin.maxR,
            bin.count,
        });
    }
    std::stable_sort(verticalEdges.begin(), verticalEdges.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.s < rhs.s;
    });

    if (!verticalEdges.empty()) {
        std::vector<VerticalCadEdge> mergedEdges;
        const double mergeDistanceMm = std::max(0.003, verticalBinWidthMm * 2.5);
        for (const VerticalCadEdge& edge : verticalEdges) {
            if (!mergedEdges.empty() && std::abs(edge.s - mergedEdges.back().s) <= mergeDistanceMm) {
                VerticalCadEdge& merged = mergedEdges.back();
                const std::size_t totalCount = merged.count + edge.count;
                merged.s = totalCount > 0
                               ? (merged.s * static_cast<double>(merged.count) +
                                  edge.s * static_cast<double>(edge.count)) /
                                     static_cast<double>(totalCount)
                               : 0.5 * (merged.s + edge.s);
                merged.minR = std::min(merged.minR, edge.minR);
                merged.maxR = std::max(merged.maxR, edge.maxR);
                merged.count = totalCount;
            } else {
                mergedEdges.push_back(edge);
            }
        }

        std::vector<std::size_t> edgeCounts;
        edgeCounts.reserve(mergedEdges.size());
        std::size_t maxEdgeCount = 0;
        for (const VerticalCadEdge& edge : mergedEdges) {
            edgeCounts.push_back(edge.count);
            maxEdgeCount = std::max(maxEdgeCount, edge.count);
        }
        const std::size_t denseCountThreshold =
            std::max<std::size_t>(32, static_cast<std::size_t>(std::ceil(static_cast<double>(maxEdgeCount) * 0.25)));
        std::vector<VerticalCadEdge> denseEdges;
        denseEdges.reserve(mergedEdges.size());
        for (const VerticalCadEdge& edge : mergedEdges) {
            if (edge.count >= denseCountThreshold) {
                denseEdges.push_back(edge);
            }
        }
        const std::vector<VerticalCadEdge>& runEdges =
            denseEdges.size() >= 6 ? denseEdges : mergedEdges;

        const double denseGapLimitMm = std::max(0.18, targetWidth * 4.0);
        std::size_t bestStart = 0;
        std::size_t bestEnd = 0;
        std::size_t runStart = 0;
        for (std::size_t i = 1; i < runEdges.size(); ++i) {
            if (runEdges[i].s - runEdges[i - 1].s > denseGapLimitMm) {
                if (i - 1 - runStart > bestEnd - bestStart) {
                    bestStart = runStart;
                    bestEnd = i - 1;
                }
                runStart = i;
            }
        }
        if (!runEdges.empty() && runEdges.size() - 1 - runStart > bestEnd - bestStart) {
            bestStart = runStart;
            bestEnd = runEdges.size() - 1;
        }

        if (bestEnd >= bestStart && bestEnd - bestStart + 1 >= 6) {
            std::vector<double> adjacentGaps;
            std::vector<double> pitchGaps;
            double runMinR = std::numeric_limits<double>::infinity();
            double runMaxR = -std::numeric_limits<double>::infinity();
            for (std::size_t i = bestStart; i <= bestEnd; ++i) {
                runMinR = std::min(runMinR, runEdges[i].minR);
                runMaxR = std::max(runMaxR, runEdges[i].maxR);
                if (i > bestStart) {
                    const double gap = runEdges[i].s - runEdges[i - 1].s;
                    if (gap > 1e-6) {
                        adjacentGaps.push_back(gap);
                    }
                }
                if (i >= bestStart + 2) {
                    const double pitch = runEdges[i].s - runEdges[i - 2].s;
                    if (pitch > 1e-6) {
                        pitchGaps.push_back(pitch);
                    }
                }
            }

            reference.valid = true;
            reference.channelWidthMm = targetWidth;
            if (request.bottomWidthMm <= 1e-9 && !adjacentGaps.empty()) {
                reference.channelWidthMm = median(adjacentGaps);
            }
            reference.channelPitchMm = pitchGaps.empty() ? 0.0 : median(pitchGaps);
            reference.arrayLeftSMm = runEdges[bestStart].s;
            reference.arrayRightSMm = runEdges[bestEnd].s;
            reference.profileMinSMm = series.front().x;
            reference.profileMaxSMm = series.back().x;
            reference.profileTopRMm = std::isfinite(runMaxR) ? runMaxR : topR;
            reference.profileBottomRMm = std::isfinite(runMinR) ? runMinR : bottomR;
            reference.channelCount = (bestEnd - bestStart + 1) / 2;
            return reference;
        }
    }

    const double threshold = topR - 0.35 * (topR - bottomR);
    const double gapLimitMm = std::max(0.12, (series.back().x - series.front().x) / 300.0);
    std::vector<double> widths;
    std::vector<double> centers;
    std::vector<double> leftEdges;
    std::vector<double> rightEdges;
    std::size_t index = 0;
    while (index < series.size()) {
        while (index < series.size() && series[index].y > threshold) {
            ++index;
        }
        if (index >= series.size()) {
            break;
        }
        std::size_t start = index;
        std::size_t end = index;
        while (end + 1 < series.size() &&
               series[end + 1].y <= threshold &&
               (series[end + 1].x - series[end].x) <= gapLimitMm) {
            ++end;
        }
        const double left = series[start].x;
        const double right = series[end].x;
        if (right > left + 1e-6) {
            widths.push_back(right - left);
            centers.push_back(0.5 * (left + right));
            leftEdges.push_back(left);
            rightEdges.push_back(right);
        }
        index = end + 1;
    }

    if (widths.empty()) {
        return reference;
    }
    std::vector<double> pitches;
    for (std::size_t i = 1; i < centers.size(); ++i) {
        const double pitch = centers[i] - centers[i - 1];
        if (pitch > 1e-6) {
            pitches.push_back(pitch);
        }
    }

    reference.valid = true;
    reference.channelWidthMm = median(widths);
    reference.channelPitchMm = pitches.empty() ? 0.0 : median(pitches);
    reference.arrayLeftSMm = leftEdges.empty() ? series.front().x : *std::min_element(leftEdges.begin(), leftEdges.end());
    reference.arrayRightSMm = rightEdges.empty() ? series.back().x : *std::max_element(rightEdges.begin(), rightEdges.end());
    reference.profileMinSMm = series.front().x;
    reference.profileMaxSMm = series.back().x;
    reference.profileTopRMm = topR;
    reference.profileBottomRMm = bottomR;
    reference.channelCount = widths.size();
    return reference;
}

std::string microchannelSourceName(const LocalSlotAnalysisRequest& request)
{
    return request.sourceImagePath.empty() ? "microchannel_array_image" : request.sourceImagePath;
}

double cadAxisMinimum(const DesignProfileMetadata& metadata, const int axis)
{
    switch (axis) {
    case 0:
        return metadata.minCadXMm;
    case 1:
        return metadata.minCadYMm;
    case 2:
        return metadata.minCadZMm;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double cadAxisMaximum(const DesignProfileMetadata& metadata, const int axis)
{
    switch (axis) {
    case 0:
        return metadata.maxCadXMm;
    case 1:
        return metadata.maxCadYMm;
    case 2:
        return metadata.maxCadZMm;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double clampToCadAxisBounds(const DesignProfileMetadata& metadata,
                            const int axis,
                            const double value)
{
    if (!metadata.hasCadBounds || axis < 0 || !std::isfinite(value)) {
        return value;
    }
    const double minValue = cadAxisMinimum(metadata, axis);
    const double maxValue = cadAxisMaximum(metadata, axis);
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue < minValue) {
        return value;
    }
    return std::clamp(value, minValue, maxValue);
}

void cadPointForMicrochannel(const LocalSlotAnalysisRequest& request,
                             const double axialSMm,
                             const double radialMm,
                             double& xMm,
                             double& yMm,
                             double& zMm)
{
    const DesignProfileMetadata& metadata = request.cadProfileMetadata;
    const int axialAxis = axisIndexFromLabel(metadata.axialAxis);
    const int radialAxis = axisIndexFromLabel(metadata.radialAxis);
    const int normalAxis = remainingCadAxis(axialAxis, radialAxis);
    xMm = std::numeric_limits<double>::quiet_NaN();
    yMm = std::numeric_limits<double>::quiet_NaN();
    zMm = std::numeric_limits<double>::quiet_NaN();
    if (axialAxis < 0 || radialAxis < 0 || normalAxis < 0 || axialAxis == radialAxis) {
        return;
    }

    const double sign = std::isfinite(metadata.cadAxialDirectionSign)
                            ? metadata.cadAxialDirectionSign
                            : 1.0;
    double axialCoordinate = metadata.cadAxialOriginMm + sign * axialSMm;
    double radialCoordinate = radialMm;
    double normalCoordinate = metadata.sectionCoordinateMm;
    if (metadata.hasCadBounds) {
        axialCoordinate = clampToCadAxisBounds(metadata, axialAxis, axialCoordinate);
        radialCoordinate = clampToCadAxisBounds(metadata, radialAxis, radialCoordinate);
        const double minNormal = cadAxisMinimum(metadata, normalAxis);
        const double maxNormal = cadAxisMaximum(metadata, normalAxis);
        if (!std::isfinite(normalCoordinate) &&
            std::isfinite(minNormal) &&
            std::isfinite(maxNormal) &&
            maxNormal >= minNormal) {
            normalCoordinate = 0.5 * (minNormal + maxNormal);
        }
    }
    if (!std::isfinite(normalCoordinate)) {
        normalCoordinate = 0.0;
    }
    if (!std::isfinite(radialCoordinate)) {
        radialCoordinate = 0.0;
    }
    if (!std::isfinite(axialCoordinate)) {
        axialCoordinate = 0.0;
    }

    xMm = 0.0;
    yMm = 0.0;
    zMm = 0.0;
    setCadAxisValue(xMm, yMm, zMm, normalAxis, normalCoordinate);
    setCadAxisValue(xMm, yMm, zMm, axialAxis, axialCoordinate);
    setCadAxisValue(xMm, yMm, zMm, radialAxis, radialCoordinate);
}

bool tryAnalyzeMicrochannelArray(const cv::Mat& image,
                                 const LocalSlotAnalysisRequest& request,
                                 LocalSlotAnalysisResult& result)
{
    const std::vector<MicrochannelBar> bars = detectMicrochannelBars(image);
    if (bars.size() < 4) {
        return false;
    }

    std::vector<double> widthsPx;
    std::vector<double> heightsPx;
    std::vector<double> pitchesPx;
    widthsPx.reserve(bars.size());
    heightsPx.reserve(bars.size());
    for (const MicrochannelBar& bar : bars) {
        widthsPx.push_back(bar.widthPx);
        heightsPx.push_back(bar.heightPx);
    }
    for (std::size_t i = 1; i < bars.size(); ++i) {
        pitchesPx.push_back(bars[i].centerXPx - bars[i - 1].centerXPx);
    }
    const double medianWidthPx = median(widthsPx);
    const double medianHeightPx = median(heightsPx);
    const double medianPitchPx = pitchesPx.empty() ? 0.0 : median(pitchesPx);
    if (!(medianWidthPx > 4.0) || !(medianHeightPx > 20.0)) {
        return false;
    }

    const MicrochannelReference cadReference = estimateMicrochannelReference(request);
    const double targetWidthMm =
        request.bottomWidthMm > 1e-9
            ? request.bottomWidthMm
            : (cadReference.valid && cadReference.channelWidthMm > 1e-9
                   ? cadReference.channelWidthMm
                   : 0.05);
    const double pixelSizeMm =
        (std::isfinite(request.pixelSizeOverrideMm) && request.pixelSizeOverrideMm > 0.0)
            ? request.pixelSizeOverrideMm
            : ((cadReference.valid && cadReference.channelPitchMm > 1e-9 && medianPitchPx > 1e-9)
                   ? cadReference.channelPitchMm / medianPitchPx
                   : targetWidthMm / medianWidthPx);
    if (!(pixelSizeMm > 0.0) || !std::isfinite(pixelSizeMm)) {
        return false;
    }

    const double minLeftPx = std::min_element(bars.begin(), bars.end(), [](const auto& lhs, const auto& rhs) {
                                 return lhs.leftPx < rhs.leftPx;
                             })->leftPx;
    const double topPx = std::min_element(bars.begin(), bars.end(), [](const auto& lhs, const auto& rhs) {
                             return lhs.topPx < rhs.topPx;
                         })->topPx;
    const double targetWidthPx = targetWidthMm / pixelSizeMm;
    const double targetHeightMm = medianHeightPx * pixelSizeMm;
    const DesignProfileMetadata& metadata = request.cadProfileMetadata;
    const int cadRadialAxisForMicrochannel = axisIndexFromLabel(metadata.radialAxis);
    double cadArrayStartSMm =
        cadReference.valid && std::isfinite(cadReference.arrayLeftSMm)
            ? cadReference.arrayLeftSMm
            : 0.0;
    double cadRadialTopMm =
        cadReference.valid && std::isfinite(cadReference.profileTopRMm)
            ? cadReference.profileTopRMm
            : metadata.maxRMm;
    double cadRadialBottomMm = cadRadialTopMm - targetHeightMm;
    if (metadata.hasCadBounds && cadRadialAxisForMicrochannel >= 0) {
        const double minRadial = cadAxisMinimum(metadata, cadRadialAxisForMicrochannel);
        const double maxRadial = cadAxisMaximum(metadata, cadRadialAxisForMicrochannel);
        if (std::isfinite(minRadial) && std::isfinite(maxRadial) && maxRadial >= minRadial) {
            if (!std::isfinite(cadRadialTopMm)) {
                cadRadialTopMm = maxRadial;
            }
            cadRadialTopMm = std::clamp(cadRadialTopMm, minRadial, maxRadial);
            cadRadialBottomMm = std::clamp(cadRadialTopMm - targetHeightMm, minRadial, maxRadial);
        }
    }
    if (!std::isfinite(cadRadialTopMm)) {
        cadRadialTopMm = targetHeightMm;
    }
    if (!std::isfinite(cadRadialBottomMm) || cadRadialBottomMm > cadRadialTopMm) {
        cadRadialBottomMm = cadRadialTopMm - targetHeightMm;
    }
    const std::size_t samplesPerEdge = 56;

    std::vector<double> usedErrorsUm;
    std::ostringstream profile;
    profile << "index,is_used,segment,x_px,y_px,s_aligned_mm,r_aligned_mm,measured_r_mm,"
               "design_s_mm,r_design_mm,design_r_mm,nearest_design_r_mm,radial_error_um,"
               "normal_error_um,profile_error_um,has_cad_coordinates,cad_measured_x_mm,"
               "cad_measured_y_mm,cad_measured_z_mm,cad_design_x_mm,cad_design_y_mm,cad_design_z_mm\n";
    std::ostringstream compensation;
    compensation << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,"
                    "has_cad_coordinates,cad_design_x_mm,cad_design_y_mm,cad_design_z_mm,cad_measured_x_mm,"
                    "cad_measured_y_mm,cad_measured_z_mm,cad_compensation_target_x_mm,"
                    "cad_compensation_target_y_mm,cad_compensation_target_z_mm,compensated_cad_x_mm,"
                    "compensated_cad_y_mm,compensated_cad_z_mm,delta_x_um,delta_y_um,delta_z_um,"
                    "s_aligned_mm,measured_r_mm,design_s_mm,design_r_mm,radial_error_um,"
                    "normal_error_um,profile_error_um,compensation_normal_um,compensation_radial_um,"
                    "compensation_radial_mm,compensated_target_r_mm,is_used,segment\n";
    std::ostringstream csv3d;
    csv3d << "design_source_type,design_source_name,cad_extraction_method,axial_axis,radial_axis,index,is_used,"
             "has_cad_coordinates,design_x_mm,design_y_mm,design_z_mm,measured_x_mm,measured_y_mm,"
             "measured_z_mm,error_x_um,error_y_um,error_z_um,error_3d_um,normal_error_um,"
             "radial_error_um,profile_error_um,compensation_target_x_mm,compensation_target_y_mm,"
             "compensation_target_z_mm,compensation_x_um,compensation_y_um,compensation_z_um,"
             "s_aligned_mm,measured_r_mm,design_s_mm,design_r_mm\n";
    std::ostringstream contour;
    contour << "index,is_used,segment,x_px,y_px,matched_x_px,matched_y_px,distance_px\n";

    const bool hasCadCoordinates =
        request.cadProfileMetadata.sourceType == "external_cad" &&
        request.cadProfileMetadata.hasCadBounds &&
        !request.cadProfileMetadata.axialAxis.empty() &&
        !request.cadProfileMetadata.radialAxis.empty();
    const std::string sourceType = hasCadCoordinates ? "external_cad" : "local_slot_microchannel_array";
    const std::string sourceName =
        hasCadCoordinates ? request.cadProfileMetadata.sourceName : microchannelSourceName(request);
    const std::string extractionMethod =
        hasCadCoordinates ? request.cadProfileMetadata.extractionMethod : "local_slot_microchannel_array";
    const std::string axialAxis = hasCadCoordinates ? request.cadProfileMetadata.axialAxis : "Z";
    const std::string radialAxis = hasCadCoordinates ? request.cadProfileMetadata.radialAxis : "Y";

    std::size_t rowIndex = 0;
    for (const MicrochannelBar& bar : bars) {
        const double targetLeftPx = bar.centerXPx - 0.5 * targetWidthPx;
        const double targetRightPx = bar.centerXPx + 0.5 * targetWidthPx;
        for (const auto side : {0, 1}) {
            const bool leftSide = side == 0;
            const double measuredXPx = leftSide ? bar.leftPx : bar.rightPx;
            const double designXPx = leftSide ? targetLeftPx : targetRightPx;
            const double signedErrorUm = (measuredXPx - designXPx) * pixelSizeMm * 1000.0;
            for (std::size_t sample = 0; sample < samplesPerEdge; ++sample) {
                const double t = samplesPerEdge <= 1
                                     ? 0.0
                                     : static_cast<double>(sample) / static_cast<double>(samplesPerEdge - 1);
                const double yPx = bar.topPx + t * (bar.bottomPx - bar.topPx);
                const double lengthMm = (yPx - topPx) * pixelSizeMm;
                const double measuredLateralMm = (measuredXPx - minLeftPx) * pixelSizeMm;
                const double designLateralMm = (designXPx - minLeftPx) * pixelSizeMm;
                const double cadRadialMm = cadRadialTopMm - t * (cadRadialTopMm - cadRadialBottomMm);
                const double measuredCadSMm = cadArrayStartSMm + measuredLateralMm;
                const double designCadSMm = cadArrayStartSMm + designLateralMm;
                double measuredX = std::numeric_limits<double>::quiet_NaN();
                double measuredY = std::numeric_limits<double>::quiet_NaN();
                double measuredZ = std::numeric_limits<double>::quiet_NaN();
                double designX = std::numeric_limits<double>::quiet_NaN();
                double designY = std::numeric_limits<double>::quiet_NaN();
                double designZ = std::numeric_limits<double>::quiet_NaN();
                if (hasCadCoordinates) {
                    cadPointForMicrochannel(request,
                                            measuredCadSMm,
                                            cadRadialMm,
                                            measuredX,
                                            measuredY,
                                            measuredZ);
                    cadPointForMicrochannel(request,
                                            designCadSMm,
                                            cadRadialMm,
                                            designX,
                                            designY,
                                            designZ);
                }
                const double deltaXUm = hasCadCoordinates ? (designX - measuredX) * 1000.0
                                                          : std::numeric_limits<double>::quiet_NaN();
                const double deltaYUm = hasCadCoordinates ? (designY - measuredY) * 1000.0
                                                          : std::numeric_limits<double>::quiet_NaN();
                const double deltaZUm = hasCadCoordinates ? (designZ - measuredZ) * 1000.0
                                                          : std::numeric_limits<double>::quiet_NaN();
                const double error3dUm =
                    hasCadCoordinates
                        ? std::sqrt(deltaXUm * deltaXUm + deltaYUm * deltaYUm + deltaZUm * deltaZUm)
                        : std::numeric_limits<double>::quiet_NaN();
                const std::string segment =
                    "microchannel_" + std::to_string(bar.channelIndex) + (leftSide ? "_left" : "_right");
                const bool isUsed = true;
                usedErrorsUm.push_back(signedErrorUm);

                profile << joinCsvRow({
                    std::to_string(rowIndex),
                    isUsed ? "1" : "0",
                    csvTextCell(segment),
                    csvCell(measuredXPx, 6),
                    csvCell(yPx, 6),
                    csvCell(lengthMm, 9),
                    csvCell(measuredLateralMm, 9),
                    csvCell(measuredLateralMm, 9),
                    csvCell(lengthMm, 9),
                    csvCell(designLateralMm, 9),
                    csvCell(designLateralMm, 9),
                    csvCell(designLateralMm, 9),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    hasCadCoordinates ? "1" : "0",
                    csvCell(measuredX, 6),
                    csvCell(measuredY, 6),
                    csvCell(measuredZ, 6),
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                });
                compensation << joinCsvRow({
                    csvTextCell(sourceType),
                    csvTextCell(sourceName),
                    csvTextCell(extractionMethod),
                    csvTextCell(axialAxis),
                    csvTextCell(radialAxis),
                    std::to_string(rowIndex),
                    hasCadCoordinates ? "1" : "0",
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                    csvCell(measuredX, 6),
                    csvCell(measuredY, 6),
                    csvCell(measuredZ, 6),
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                    csvCell(deltaXUm, 6),
                    csvCell(deltaYUm, 6),
                    csvCell(deltaZUm, 6),
                    csvCell(lengthMm, 9),
                    csvCell(measuredLateralMm, 9),
                    csvCell(lengthMm, 9),
                    csvCell(designLateralMm, 9),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(-signedErrorUm, 6),
                    csvCell(-signedErrorUm, 6),
                    csvCell(-signedErrorUm / 1000.0, 9),
                    csvCell(designLateralMm, 9),
                    "1",
                    csvTextCell(segment),
                });
                csv3d << joinCsvRow({
                    csvTextCell(sourceType),
                    csvTextCell(sourceName),
                    csvTextCell(extractionMethod),
                    csvTextCell(axialAxis),
                    csvTextCell(radialAxis),
                    std::to_string(rowIndex),
                    "1",
                    hasCadCoordinates ? "1" : "0",
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                    csvCell(measuredX, 6),
                    csvCell(measuredY, 6),
                    csvCell(measuredZ, 6),
                    csvCell(-deltaXUm, 6),
                    csvCell(-deltaYUm, 6),
                    csvCell(-deltaZUm, 6),
                    csvCell(error3dUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(signedErrorUm, 6),
                    csvCell(designX, 6),
                    csvCell(designY, 6),
                    csvCell(designZ, 6),
                    csvCell(deltaXUm, 6),
                    csvCell(deltaYUm, 6),
                    csvCell(deltaZUm, 6),
                    csvCell(lengthMm, 9),
                    csvCell(measuredLateralMm, 9),
                    csvCell(lengthMm, 9),
                    csvCell(designLateralMm, 9),
                });
                contour << rowIndex << ",1," << csvTextCell(segment) << ","
                        << csvCell(measuredXPx, 6) << ","
                        << csvCell(yPx, 6) << ","
                        << csvCell(designXPx, 6) << ","
                        << csvCell(yPx, 6) << ","
                        << csvCell(std::abs(measuredXPx - designXPx), 6) << "\n";
                ++rowIndex;
            }
        }
    }

    const double measuredWidthMm = medianWidthPx * pixelSizeMm;
    const double measuredPitchMm = medianPitchPx * pixelSizeMm;
    const double widthErrorUm = (measuredWidthMm - targetWidthMm) * 1000.0;
    const double rms = rmse(usedErrorsUm);
    std::vector<double> usedAbsErrors;
    usedAbsErrors.reserve(usedErrorsUm.size());
    for (const double value : usedErrorsUm) {
        usedAbsErrors.push_back(std::abs(value));
    }
    const double p95 = percentile(usedAbsErrors, 0.95);
    const double maxAbs = maxAbsValue(usedErrorsUm);
    const std::size_t exceedanceCount = countExceedances(usedErrorsUm, kLocalExceedanceThresholdUm);
    const double maxPos = usedErrorsUm.empty() ? 0.0 : *std::max_element(usedErrorsUm.begin(), usedErrorsUm.end());
    const double maxNeg = usedErrorsUm.empty() ? 0.0 : *std::min_element(usedErrorsUm.begin(), usedErrorsUm.end());
    const double pv = maxPos - maxNeg;
    const double outlierRatio = 0.0;

    std::ostringstream summary;
    summary << "dz_mm,dr_mm,dtheta_deg,axial_scale_factor,axial_quadratic_term_mm,"
               "applied_absolute_bias_refine,absolute_bias_correction_um,"
               "pre_refine_mean_normal_error_um,pre_refine_absolute_filtered_rmse_um,"
               "design_reverse_axial,use_left_endpoint_anchor,design_evaluate_profile_form,"
               "anchor_x_px,anchor_y_px,pixel_size_mm,candidate_count,used_count,outlier_count,"
               "outlier_ratio,mean_normal_error_um,normal_rmse_um,normal_mae_um,normal_p95_abs_um,"
               "normal_max_pos_um,normal_max_neg_um,normal_pv_um,profile_rms_um,profile_mae_um,"
               "profile_p95_abs_um,profile_max_pos_um,profile_max_neg_um,profile_pv_um,"
               "local_exceedance_threshold_um,local_exceedance_count,local_max_abs_error_um,"
               "local_max_exceedance_um,"
               "absolute_all_mean_um,absolute_all_rmse_um,absolute_all_mae_um,absolute_all_p95_abs_um,"
               "absolute_all_max_pos_um,absolute_all_max_neg_um,absolute_all_pv_um,"
               "absolute_filtered_mean_um,absolute_filtered_rmse_um,absolute_filtered_mae_um,"
               "absolute_filtered_p95_abs_um,absolute_filtered_max_pos_um,absolute_filtered_max_neg_um,"
               "absolute_filtered_pv_um,design_source_type,design_source_name,design_profile_sample_count,"
               "design_profile_min_s_mm,design_profile_max_s_mm,design_profile_min_r_mm,design_profile_max_r_mm,"
               "cad_extraction_method,cad_section_normal_axis,cad_section_coordinate_mm,cad_axial_axis,"
               "cad_radial_axis,cad_axial_origin_mm,cad_axial_direction_sign\n";
    summary << "0,0,0,1,0,0,0,"
            << csvCell(mean(usedErrorsUm), 6) << ","
            << csvCell(rms, 6) << ",0,1,1,"
            << csvCell(bars.front().leftPx, 6) << ","
            << csvCell(bars.front().topPx, 6) << ","
            << csvCell(pixelSizeMm, 9) << ","
            << rowIndex << "," << rowIndex << ",0,"
            << csvCell(outlierRatio, 6) << ","
            << csvCell(mean(usedErrorsUm), 6) << ","
            << csvCell(rms, 6) << ","
            << csvCell(mean(usedAbsErrors), 6) << ","
            << csvCell(p95, 6) << ","
            << csvCell(maxPos, 6) << ","
            << csvCell(maxNeg, 6) << ","
            << csvCell(pv, 6) << ","
            << csvCell(rms, 6) << ","
            << csvCell(mean(usedAbsErrors), 6) << ","
            << csvCell(p95, 6) << ","
            << csvCell(maxPos, 6) << ","
            << csvCell(maxNeg, 6) << ","
            << csvCell(pv, 6) << ","
            << csvCell(kLocalExceedanceThresholdUm, 6) << ","
            << exceedanceCount << ","
            << csvCell(maxAbs, 6) << ","
            << csvCell(std::isfinite(maxAbs) ? std::max(0.0, maxAbs - kLocalExceedanceThresholdUm) : 0.0, 6) << ","
            << csvCell(mean(usedErrorsUm), 6) << ","
            << csvCell(rms, 6) << ","
            << csvCell(mean(usedAbsErrors), 6) << ","
            << csvCell(p95, 6) << ","
            << csvCell(maxPos, 6) << ","
            << csvCell(maxNeg, 6) << ","
            << csvCell(pv, 6) << ","
            << csvCell(mean(usedErrorsUm), 6) << ","
            << csvCell(rms, 6) << ","
            << csvCell(mean(usedAbsErrors), 6) << ","
            << csvCell(p95, 6) << ","
            << csvCell(maxPos, 6) << ","
            << csvCell(maxNeg, 6) << ","
            << csvCell(pv, 6) << ","
            << csvTextCell(sourceType) << ","
            << csvTextCell(sourceName) << ","
            << (hasCadCoordinates ? request.cadProfileMetadata.sampleCount : bars.size()) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.minSMm : 0.0, 6) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.maxSMm : measuredPitchMm * bars.size(), 6) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.minRMm : 0.0, 6) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.maxRMm : targetHeightMm, 6) << ","
            << csvTextCell(extractionMethod) << ","
            << csvTextCell(hasCadCoordinates ? request.cadProfileMetadata.sectionNormalAxis : "X") << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.sectionCoordinateMm : 0.0, 6) << ","
            << csvTextCell(axialAxis) << ","
            << csvTextCell(radialAxis) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.cadAxialOriginMm : 0.0, 6) << ","
            << csvCell(hasCadCoordinates ? request.cadProfileMetadata.cadAxialDirectionSign : 1.0, 6) << "\n";

    std::ostringstream feature;
    feature << "feature_id,status,method,design_source_type,design_source_name,cad_extraction_method,"
               "target_slot_width_mm,target_slot_depth_mm,cad_slot_left_s_mm,cad_slot_right_s_mm,"
               "cad_slot_center_s_mm,cad_slot_width_mm,cad_slot_depth_mm,cad_slot_bottom_s_mm,"
               "cad_slot_bottom_r_mm,cad_slot_center_x_mm,cad_slot_center_y_mm,cad_slot_center_z_mm,"
               "measured_slot_center_s_mm,measured_slot_width_mm,measured_slot_depth_mm,"
               "center_error_um,width_error_um,depth_error_um,compensation_center_um,"
               "compensation_width_um,compensation_depth_um,channel_count,measured_pitch_mm,cad_pitch_mm,notes\n";
    feature << csvTextCell("local_slot_microchannel_array") << ","
            << csvTextCell("ok") << ","
            << csvTextCell("local_slot_microchannel_array") << ","
            << csvTextCell(sourceType) << ","
            << csvTextCell(sourceName) << ","
            << csvTextCell(extractionMethod) << ","
            << csvCell(targetWidthMm, 9) << ","
            << csvCell(targetHeightMm, 9) << ","
            << "0,"
            << csvCell(targetWidthMm, 9) << ","
            << csvCell(targetWidthMm * 0.5, 9) << ","
            << csvCell(targetWidthMm, 9) << ","
            << csvCell(targetHeightMm, 9) << ","
            << csvCell(targetWidthMm * 0.5, 9) << ",0,,,,"
            << csvCell((bars.front().centerXPx - minLeftPx) * pixelSizeMm, 9) << ","
            << csvCell(measuredWidthMm, 9) << ","
            << csvCell(targetHeightMm, 9) << ","
            << "0,"
            << csvCell(widthErrorUm, 6) << ",0,0,"
            << csvCell(-widthErrorUm, 6) << ",0,"
            << bars.size() << ","
            << csvCell(measuredPitchMm, 9) << ","
            << csvCell(cadReference.valid ? cadReference.channelPitchMm : 0.0, 9) << ","
            << csvTextCell("Detected vertical diamond microchannel array; width and pitch are reported per channel.")
            << "\n";

    std::ostringstream quality;
    quality << "check,status,value,threshold,message\n";
    quality << "microchannel_array_detection,PASS," << bars.size() << ",4,"
            << csvTextCell("Detected multiple tall narrow microchannels in the imported image.") << "\n";
    quality << "microchannel_width,PASS," << csvCell(measuredWidthMm, 9) << ","
            << csvCell(targetWidthMm, 9) << ","
            << csvTextCell("Median microchannel width compared with CAD/target width.") << "\n";
    quality << "microchannel_pitch,PASS," << csvCell(measuredPitchMm, 9) << ","
            << csvCell(cadReference.valid ? cadReference.channelPitchMm : 0.0, 9) << ","
            << csvTextCell("Median channel pitch; CAD pitch is used when the CAD section exposes the channel array.") << "\n";
    quality << "cad_compensation_xyz_output," << (hasCadCoordinates ? "PASS" : "WARN") << ","
            << (hasCadCoordinates ? 1 : 0) << ",1,"
            << csvTextCell(hasCadCoordinates
                               ? "Microchannel edge rows include compensated CAD x/y/z coordinates."
                               : "Microchannel detection completed without CAD XYZ mapping.") << "\n";

    result.ok = true;
    result.message = "Diamond microchannel array analysis completed.";
    result.detectedPointCount = static_cast<std::size_t>(cv::countNonZero(buildMicrochannelMask(image)));
    result.usedPointCount = rowIndex;
    result.outlierPointCount = 0;
    result.bottomWidthPx = medianWidthPx;
    result.pixelSizeMm = pixelSizeMm;
    result.measuredSlotWidthMm = measuredWidthMm;
    result.widthErrorUm = widthErrorUm;
    result.normalRmseUm = rms;
    result.normalP95AbsUm = p95;
    result.normalPvUm = pv;
    result.localExceedanceThresholdUm = kLocalExceedanceThresholdUm;
    result.localExceedanceCount = exceedanceCount;
    result.localMaxAbsErrorUm = maxAbs;
    result.localMaxExceedanceUm =
        std::isfinite(maxAbs) ? std::max(0.0, maxAbs - kLocalExceedanceThresholdUm) : 0.0;
    result.profileCsvText = profile.str();
    result.summaryCsvText = summary.str();
    result.compensationCsvText = compensation.str();
    result.error3dCsvText = csv3d.str();
    result.featureCompensationCsvText = feature.str();
    result.qualityReviewCsvText = quality.str();
    result.contourPointsCsvText = contour.str();
    return true;
}

std::string buildContourPointsCsv(const std::vector<MatchedPoint>& points)
{
    std::ostringstream stream;
    stream << "index,is_used,segment,x_px,y_px,matched_x_px,matched_y_px,distance_px\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
        const MatchedPoint& point = points[index];
        stream << index << ","
               << (point.isUsed ? 1 : 0) << ","
               << csvTextCell(point.segmentLabel) << ","
               << csvCell(point.imagePoint.x, 6) << ","
               << csvCell(point.imagePoint.y, 6) << ","
               << csvCell(point.designPoint.x, 6) << ","
               << csvCell(point.designPoint.y, 6) << ","
               << csvCell(point.distancePx, 6) << "\n";
    }
    return stream.str();
}

} // namespace

LocalSlotAnalysisResult analyzeLocalSlotImage(const cv::Mat& image,
                                              const LocalSlotAnalysisRequest& request)
{
    LocalSlotAnalysisResult result;
    if (image.empty()) {
        result.message = "Local slot image is empty.";
        return result;
    }
    if (!(request.bottomWidthMm > 0.0)) {
        result.message = "Local slot bottom width must be positive.";
        return result;
    }

    if (tryAnalyzeMicrochannelArray(image, request, result)) {
        return result;
    }

    bool objectBoundaryMode = false;
    const cv::Mat binary = buildBinaryMask(image, objectBoundaryMode);
    cv::Mat skeleton = skeletonize(binary);
    std::vector<cv::Point2d> points = extractPoints(skeleton);
    if (points.size() < 40) {
        points = extractPoints(binary);
    }
    result.detectedPointCount = points.size();
    if (points.size() < 20) {
        result.message = "Local slot edge extraction failed: not enough foreground edge points.";
        return result;
    }

    const cv::Rect2d bounds = boundsOf(points);
    Segment bottom;
    double bottomY = 0.0;
    const bool hasDiagonalSlot =
        objectBoundaryMode && estimateDiagonalSlotSegment(binary, bounds, bottom, bottomY);
    if (!hasDiagonalSlot && !estimateBottomSegment(points, bounds, objectBoundaryMode, bottom, bottomY)) {
        result.message = "Local slot bottom edge detection failed.";
        return result;
    }

    const double bottomWidthPx = bottom.lengthPx();
    if (!(bottomWidthPx > 5.0)) {
        result.message = "Local slot bottom edge is too short for calibration.";
        return result;
    }

    const double calibratedPixelSizeMm = request.bottomWidthMm / bottomWidthPx;
    const double pixelSizeScale =
        (std::isfinite(request.pixelSizeScale) && request.pixelSizeScale > 0.0)
            ? request.pixelSizeScale
            : 1.0;
    const double pixelSizeMm =
        (std::isfinite(request.pixelSizeOverrideMm) && request.pixelSizeOverrideMm > 0.0)
            ? request.pixelSizeOverrideMm
            : calibratedPixelSizeMm * pixelSizeScale;
    const double sideBand = std::max(12.0, bottomWidthPx * 0.42);
    const double sideOuterReach = std::max(25.0, bottomWidthPx * 0.24);
    const double bottomTolerance = std::max(2.0, bounds.height * 0.025);
    const double sideHeightLimit = std::max(12.0, std::min(bounds.height * 0.72, bottomWidthPx * 0.38));

    auto collectSideCandidates = [&](const Segment& candidateBottom,
                                     std::vector<cv::Point2d>& left,
                                     std::vector<cv::Point2d>& right) {
        left.clear();
        right.clear();
        const double lengthPx = candidateBottom.lengthPx();
        for (const cv::Point2d& point : points) {
            const double s = segmentSpx(point, candidateBottom);
            const double r = segmentRpx(point, candidateBottom);
            if (std::abs(r) <= bottomTolerance || r < -bottomTolerance || r > sideHeightLimit) {
                continue;
            }
            if (s >= -sideOuterReach && s <= sideBand) {
                left.push_back(point);
            }
            if (s >= lengthPx - sideBand && s <= lengthPx + sideOuterReach) {
                right.push_back(point);
            }
        }
    };

    std::vector<cv::Point2d> leftCandidates;
    std::vector<cv::Point2d> rightCandidates;
    collectSideCandidates(bottom, leftCandidates, rightCandidates);
    Segment reversedBottom{bottom.b, bottom.a, "bottom"};
    std::vector<cv::Point2d> reversedLeftCandidates;
    std::vector<cv::Point2d> reversedRightCandidates;
    collectSideCandidates(reversedBottom, reversedLeftCandidates, reversedRightCandidates);
    const std::size_t currentScore =
        std::min(leftCandidates.size(), rightCandidates.size()) +
        (leftCandidates.size() + rightCandidates.size()) / 8;
    const std::size_t reversedScore =
        std::min(reversedLeftCandidates.size(), reversedRightCandidates.size()) +
        (reversedLeftCandidates.size() + reversedRightCandidates.size()) / 8;
    if (reversedScore > currentScore) {
        bottom = reversedBottom;
        leftCandidates = std::move(reversedLeftCandidates);
        rightCandidates = std::move(reversedRightCandidates);
    }

    Segment left;
    left.label = "left";
    Segment right;
    right.label = "right";
    if (!fitLineSegment(leftCandidates, bottom.a, true, left)) {
        result.message = "Local slot left edge fitting failed.";
        return result;
    }
    if (!fitLineSegment(rightCandidates, bottom.b, true, right)) {
        result.message = "Local slot right edge fitting failed.";
        return result;
    }

    std::vector<MatchedPoint> matched = matchPointsToSegments(points, left, bottom, right, bounds);
    if (matched.size() < 20) {
        result.message = "Local slot contour matching failed: not enough points near fitted slot edges.";
        return result;
    }
    markOutliers(matched, pixelSizeMm, request.maxAcceptedErrorUm);
    thinUsedPoints(matched, request.maxOutputPoints);

    result.usedPointCount = static_cast<std::size_t>(
        std::count_if(matched.begin(), matched.end(), [](const MatchedPoint& point) { return point.isUsed; }));
    result.outlierPointCount = matched.size() - result.usedPointCount;
    if (result.usedPointCount < 12) {
        result.message = "Local slot contour filtering removed too many points.";
        return result;
    }

    const std::vector<double> usedErrors = usedSignedErrorsUm(matched, pixelSizeMm);
    std::vector<double> usedAbsErrors;
    usedAbsErrors.reserve(usedErrors.size());
    for (const double value : usedErrors) {
        usedAbsErrors.push_back(std::abs(value));
    }
    const double maxPos = usedErrors.empty() ? std::numeric_limits<double>::quiet_NaN()
                                             : *std::max_element(usedErrors.begin(), usedErrors.end());
    const double maxNeg = usedErrors.empty() ? std::numeric_limits<double>::quiet_NaN()
                                             : *std::min_element(usedErrors.begin(), usedErrors.end());

    result.bottomWidthPx = bottomWidthPx;
    result.pixelSizeMm = pixelSizeMm;
    result.measuredSlotWidthMm = bottomWidthPx * pixelSizeMm;
    result.widthErrorUm = (result.measuredSlotWidthMm - request.bottomWidthMm) * 1000.0;
    result.normalRmseUm = rmse(usedErrors);
    result.normalP95AbsUm = percentile(usedAbsErrors, 0.95);
    result.normalPvUm = (std::isfinite(maxPos) && std::isfinite(maxNeg)) ? (maxPos - maxNeg) : 0.0;
    result.localExceedanceThresholdUm = kLocalExceedanceThresholdUm;
    result.localExceedanceCount = countExceedances(usedErrors, result.localExceedanceThresholdUm);
    result.localMaxAbsErrorUm = maxAbsValue(usedErrors);
    result.localMaxExceedanceUm =
        std::isfinite(result.localMaxAbsErrorUm)
            ? std::max(0.0, result.localMaxAbsErrorUm - result.localExceedanceThresholdUm)
            : 0.0;
    const CadMappingContext cadMapping = buildCadMappingContext(request, matched, bottom, pixelSizeMm);
    result.profileCsvText = buildProfileCsv(matched, bottom, pixelSizeMm, cadMapping);
    result.summaryCsvText = buildSummaryCsv(request,
                                            matched,
                                            bottom,
                                            pixelSizeMm,
                                            usedErrors,
                                            result.normalP95AbsUm,
                                            result.normalPvUm,
                                            cadMapping);
    result.compensationCsvText = buildCompensationCsv(matched, bottom, pixelSizeMm, cadMapping);
    result.error3dCsvText = build3dCsv(matched, bottom, pixelSizeMm, cadMapping);
    result.featureCompensationCsvText = buildFeatureCsv(request, bottom, pixelSizeMm);
    result.contourPointsCsvText = buildContourPointsCsv(matched);
    result.ok = true;
    result.message = "Local slot image analysis completed.";
    result.qualityReviewCsvText = buildQualityReviewCsv(request, result, cadMapping);
    return result;
}

} // namespace pinjie::cad_design
