#include "DebugVis.h"

#include "GeometryUtils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iomanip>
#include <map>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace stitch {

namespace {

std::string formatDouble(double value, int precision = 6)
{
    std::ostringstream stream;
    stream << std::setprecision(precision) << value;
    return stream.str();
}

void appendResidualStatisticsCsv(std::ostringstream& stream, const ResidualStatistics& stats)
{
    stream << formatDouble(stats.bias) << ","
           << formatDouble(stats.rmse) << ","
           << formatDouble(stats.meanAbs) << ","
           << formatDouble(stats.medianAbs) << ","
           << formatDouble(stats.p95Abs) << ","
           << formatDouble(stats.maxAbs);
}

cv::Mat toBgr(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }

    if (image.channels() == 3) {
        return image.clone();
    }

    cv::Mat converted;
    if (image.channels() == 1) {
        cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
    } else {
        image.convertTo(converted, CV_8UC3);
    }
    return converted;
}

cv::Scalar indexedColor(std::size_t index)
{
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 160, 255),
        cv::Scalar(0, 220, 0),
        cv::Scalar(255, 120, 0),
        cv::Scalar(220, 0, 220),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 180, 180),
        cv::Scalar(120, 80, 255)
    };
    return colors[index % colors.size()];
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

void transformTargetPointsForResult(const std::vector<cv::Point2d>& edgesTarget,
                                    std::vector<cv::Point2d>& transformedTarget,
                                    const cv::Point2d& center,
                                    const TransformResult& result)
{
    if (result.hasCustomRelativeMatrix &&
        !result.customRelativeMatrix.empty() &&
        result.customRelativeMatrix.rows == 3 &&
        result.customRelativeMatrix.cols == 3) {
        transformedTarget.resize(edgesTarget.size());
        for (std::size_t index = 0; index < edgesTarget.size(); ++index) {
            transformedTarget[index] = transformPoint(result.customRelativeMatrix, edgesTarget[index]);
        }
        return;
    }

    rotatePoints(edgesTarget, transformedTarget, result.da, center);
    for (auto& point : transformedTarget) {
        point.x += result.dx;
        point.y += result.dy;
    }
}

double tangentCorrelationValue(const StitchStepRecord& step, bool useInlierCorrelation)
{
    const AlignmentMetrics& metrics = step.transform.metrics;
    if (useInlierCorrelation && metrics.tangentInlier.valid()) {
        return metrics.tangentCorrInlier;
    }
    return metrics.tangentCorrAll;
}

bool isPriorClampedStep(const StitchStepRecord& step)
{
    return step.usedMotionPriorFallback || step.transform.direction == "PRIOR" ||
           step.selectionMode == "trajectory_prior_clamp" ||
           step.selectionMode == "translation_prior_fallback";
}

std::string stepStatusLabel(const StitchStepRecord& step)
{
    if (step.selectionMode == "trajectory_prior_clamp") {
        return "Trajectory-prior clamped";
    }
    if (step.selectionMode == "translation_prior_fallback") {
        return "Translation-prior fallback";
    }
    if (step.selectionMode == "trajectory_prior_rescue") {
        return "Trajectory-prior rescued";
    }
    if (step.selectionMode == "unfiltered_edge_rescue") {
        return "Unfiltered-edge rescued";
    }
    if (step.selectionMode == "wide_search_rescue") {
        return "Wide-search rescued";
    }
    if (step.selectionMode == "local_prior_rescan") {
        return "Local prior rescan";
    }
    if (step.selectionMode == "candidate_reselect") {
        return "Candidate reselected";
    }
    return "Direct match";
}

std::string motionAxisLabel(AlignmentAxis axis)
{
    return axis == AlignmentAxis::X ? "X" : "Y";
}

double motionPrimaryShift(const StitchStepRecord& step, double dx, double dy)
{
    return step.motionAxis == AlignmentAxis::X ? dx : dy;
}

double motionPerpendicularShift(const StitchStepRecord& step, double dx, double dy)
{
    return step.motionAxis == AlignmentAxis::X ? dy : dx;
}

double overlapRatioFromPrimaryShift(const StitchStepRecord& step, double primaryShiftPx)
{
    if (!(step.primaryImageSpanPx > 1e-9)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 - primaryShiftPx / step.primaryImageSpanPx;
}

cv::Scalar stepStatusColor(const StitchStepRecord& step)
{
    if (isPriorClampedStep(step)) {
        return cv::Scalar(70, 105, 205);
    }
    if (step.usedTrajectoryPriorRescue || step.usedWideSearchRescue || step.usedUnfilteredEdgeRescue) {
        return cv::Scalar(64, 138, 214);
    }
    return cv::Scalar(64, 64, 64);
}

struct StitchedContourSample {
    double canvasX{0.0};
    double canvasY{0.0};
    double primary{0.0};
    double secondary{0.0};
};

struct ContourProfilePoint {
    double primary{0.0};
    double secondary{0.0};
    double smoothedSecondary{0.0};
    double fluctuation{0.0};
    std::size_t supportCount{0};
};

struct ContourProfileData {
    bool primaryAxisIsX{true};
    std::vector<ContourProfilePoint> points;
};

using PointSeries = std::vector<cv::Point2d>;

bool collectStitchedContourSamples(const std::vector<EdgeVariants>& edges,
                                   const std::vector<cv::Mat>& imageTransforms,
                                   std::vector<StitchedContourSample>& samples,
                                   bool& primaryAxisIsX);

enum class ContourSeriesMode {
    Secondary,
    SmoothedSecondary,
    Fluctuation
};

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
        median = (median + *lowerMid) * 0.5;
    }
    return median;
}

std::string csvCell(double value, int precision = 6)
{
    return std::isfinite(value) ? formatDouble(value, precision) : std::string();
}

std::vector<double> buildSmoothedSeries(const std::vector<double>& values);

std::vector<double> buildFluctuationSeries(const std::vector<double>& values);

std::size_t chooseSmoothingWindow(std::size_t count)
{
    if (count <= 3) {
        return count;
    }

    std::size_t window = 11;
    if (count > 120) {
        window = 31;
    } else if (count > 60) {
        window = 21;
    } else if (count > 20) {
        window = 11;
    } else {
        window = 5;
    }

    if (window > count) {
        window = count;
    }
    if (window % 2 == 0) {
        if (window > 1) {
            --window;
        } else {
            window = 1;
        }
    }

    return std::max<std::size_t>(1, window);
}

std::vector<double> buildSmoothedSeries(const std::vector<double>& values)
{
    std::vector<double> smoothed(values.size(), 0.0);
    if (values.empty()) {
        return smoothed;
    }

    const std::size_t window = chooseSmoothingWindow(values.size());
    const std::size_t halfWindow = window / 2;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t begin = (i > halfWindow) ? (i - halfWindow) : 0;
        const std::size_t end = std::min(values.size() - 1, i + halfWindow);

        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t j = begin; j <= end; ++j) {
            sum += values[j];
            ++count;
        }
        smoothed[i] = count > 0 ? sum / static_cast<double>(count) : values[i];
    }

    return smoothed;
}

std::vector<double> buildFluctuationSeries(const std::vector<double>& values)
{
    const std::vector<double> smoothed = buildSmoothedSeries(values);
    std::vector<double> fluctuation(values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        fluctuation[i] = values[i] - smoothed[i];
    }
    return fluctuation;
}

bool collectPerImageCanvasPointSeries(const std::vector<EdgeVariants>& edges,
                                      const std::vector<cv::Mat>& imageTransforms,
                                      std::vector<PointSeries>& seriesByImage,
                                      bool& primaryAxisIsX)
{
    std::vector<StitchedContourSample> samples;
    if (!collectStitchedContourSamples(edges, imageTransforms, samples, primaryAxisIsX)) {
        seriesByImage.clear();
        return false;
    }

    const std::size_t count = std::min(edges.size(), imageTransforms.size());
    seriesByImage.assign(count, {});
    for (std::size_t imageIndex = 0; imageIndex < count; ++imageIndex) {
        const cv::Mat& transform = imageTransforms[imageIndex];
        if (transform.empty() || transform.rows < 2 || transform.cols < 3) {
            continue;
        }

        PointSeries& series = seriesByImage[imageIndex];
        series.reserve(edges[imageIndex].raw.size());
        for (const cv::Point2d& point : edges[imageIndex].raw) {
            const cv::Point2d canvasPoint = transformPoint(transform, point);
            if (!std::isfinite(canvasPoint.x) || !std::isfinite(canvasPoint.y)) {
                continue;
            }
            series.push_back(canvasPoint);
        }

        std::sort(series.begin(), series.end(), [primaryAxisIsX](const cv::Point2d& lhs, const cv::Point2d& rhs) {
            const double lhsPrimary = primaryAxisIsX ? lhs.x : lhs.y;
            const double rhsPrimary = primaryAxisIsX ? rhs.x : rhs.y;
            if (lhsPrimary != rhsPrimary) {
                return lhsPrimary < rhsPrimary;
            }

            const double lhsSecondary = primaryAxisIsX ? lhs.y : lhs.x;
            const double rhsSecondary = primaryAxisIsX ? rhs.y : rhs.x;
            return lhsSecondary < rhsSecondary;
        });
    }

    return true;
}

std::vector<cv::Point> buildPolylinePixels(const PointSeries& series,
                                           bool primaryAxisIsX,
                                           double minPrimarySpacingPx = 0.75)
{
    std::vector<cv::Point> polyline;
    polyline.reserve(series.size());

    bool hasPrevious = false;
    double previousPrimary = 0.0;
    cv::Point previousPoint;
    for (const cv::Point2d& point : series) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }

        const double primary = primaryAxisIsX ? point.x : point.y;
        const cv::Point pixel(static_cast<int>(std::lround(point.x)),
                              static_cast<int>(std::lround(point.y)));
        if (hasPrevious) {
            const int dx = pixel.x - previousPoint.x;
            const int dy = pixel.y - previousPoint.y;
            if (std::abs(primary - previousPrimary) < minPrimarySpacingPx &&
                dx * dx + dy * dy <= 1) {
                continue;
            }
        }

        polyline.push_back(pixel);
        previousPrimary = primary;
        previousPoint = pixel;
        hasPrevious = true;
    }

    return polyline;
}

void drawPanoramaEnvelope(cv::Mat& canvas, const cv::Mat& panorama)
{
    if (panorama.empty()) {
        return;
    }

    cv::Mat gray;
    if (panorama.channels() == 1) {
        gray = panorama;
    } else {
        cv::cvtColor(panorama, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat mask;
    cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return;
    }

    auto largest = std::max_element(contours.begin(), contours.end(),
                                    [](const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) {
                                        return cv::contourArea(lhs) < cv::contourArea(rhs);
                                    });
    if (largest == contours.end() || largest->empty()) {
        return;
    }

    cv::drawContours(canvas, contours, static_cast<int>(std::distance(contours.begin(), largest)),
                     cv::Scalar(72, 72, 72), 2, cv::LINE_AA);
}

bool collectStitchedContourSamples(const std::vector<EdgeVariants>& edges,
                                   const std::vector<cv::Mat>& imageTransforms,
                                   std::vector<StitchedContourSample>& samples,
                                   bool& primaryAxisIsX)
{
    samples.clear();
    const std::size_t count = std::min(edges.size(), imageTransforms.size());
    if (count == 0) {
        return false;
    }

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

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

            minX = std::min(minX, canvasPoint.x);
            maxX = std::max(maxX, canvasPoint.x);
            minY = std::min(minY, canvasPoint.y);
            maxY = std::max(maxY, canvasPoint.y);
        }
    }

    if (!std::isfinite(minX) || !std::isfinite(maxX) || !std::isfinite(minY) || !std::isfinite(maxY)) {
        return false;
    }

    primaryAxisIsX = (maxX - minX) >= (maxY - minY);
    samples.reserve(1024);

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

            StitchedContourSample sample;
            sample.canvasX = canvasPoint.x;
            sample.canvasY = canvasPoint.y;
            sample.primary = primaryAxisIsX ? canvasPoint.x : canvasPoint.y;
            sample.secondary = primaryAxisIsX ? canvasPoint.y : canvasPoint.x;
            samples.push_back(sample);
        }
    }

    return !samples.empty();
}

ContourProfileData buildStitchedContourProfileData(const std::vector<EdgeVariants>& edges,
                                                   const std::vector<cv::Mat>& imageTransforms)
{
    ContourProfileData result;
    std::vector<StitchedContourSample> samples;
    if (!collectStitchedContourSamples(edges, imageTransforms, samples, result.primaryAxisIsX)) {
        return result;
    }

    double minPrimary = std::numeric_limits<double>::infinity();
    for (const auto& sample : samples) {
        minPrimary = std::min(minPrimary, sample.primary);
    }

    const double binWidth = 1.0;
    struct BinData {
        std::vector<double> primaries;
        std::vector<double> secondaries;
    };

    std::map<long long, BinData> bins;
    for (const auto& sample : samples) {
        const long long binIndex =
            static_cast<long long>(std::floor((sample.primary - minPrimary) / binWidth));
        BinData& bin = bins[binIndex];
        bin.primaries.push_back(sample.primary);
        bin.secondaries.push_back(sample.secondary);
    }

    result.points.reserve(bins.size());
    for (auto& [binIndex, bin] : bins) {
        ContourProfilePoint point;
        point.primary = std::accumulate(bin.primaries.begin(), bin.primaries.end(), 0.0) /
                        static_cast<double>(bin.primaries.size());
        point.secondary = medianOfValues(bin.secondaries);
        point.supportCount = bin.secondaries.size();
        result.points.push_back(point);
    }

    if (result.points.empty()) {
        return result;
    }

    const std::size_t window = chooseSmoothingWindow(result.points.size());
    const std::size_t halfWindow = window / 2;
    for (std::size_t i = 0; i < result.points.size(); ++i) {
        const std::size_t begin = (i > halfWindow) ? (i - halfWindow) : 0;
        const std::size_t end = std::min(result.points.size() - 1, i + halfWindow);

        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t j = begin; j <= end; ++j) {
            sum += result.points[j].secondary;
            ++count;
        }

        const double smoothed = count > 0 ? sum / static_cast<double>(count) : result.points[i].secondary;
        result.points[i].smoothedSecondary = smoothed;
        result.points[i].fluctuation = result.points[i].secondary - smoothed;
    }

    return result;
}

void drawLineSeries(cv::Mat& canvas,
                    const cv::Rect& plotRect,
                    const std::vector<ContourProfilePoint>& points,
                    const cv::Scalar& color,
                    ContourSeriesMode mode,
                    double yMin,
                    double yMax,
                    double xMin,
                    double xMax)
{
    if (points.size() < 2) {
        return;
    }

    const double xRange = std::max(1e-9, xMax - xMin);
    const double yRange = std::max(1e-9, yMax - yMin);

    cv::Point previous;
    bool hasPrevious = false;
    for (const auto& point : points) {
        double yValue = point.secondary;
        if (mode == ContourSeriesMode::SmoothedSecondary) {
            yValue = point.smoothedSecondary;
        } else if (mode == ContourSeriesMode::Fluctuation) {
            yValue = point.fluctuation;
        }
        const double xRatio = (point.primary - xMin) / xRange;
        const double yRatio = (yValue - yMin) / yRange;
        const int x = plotRect.x + static_cast<int>(std::round(xRatio * plotRect.width));
        const int y = plotRect.y + plotRect.height - static_cast<int>(std::round(yRatio * plotRect.height));
        const cv::Point current(x, y);

        if (hasPrevious) {
            cv::line(canvas, previous, current, color, 2, cv::LINE_AA);
        }
        previous = current;
        hasPrevious = true;
    }
}

void drawProfileScatter(cv::Mat& canvas,
                        const cv::Rect& plotRect,
                        const std::vector<ContourProfilePoint>& points,
                        const cv::Scalar& color,
                    ContourSeriesMode mode,
                    double yMin,
                    double yMax,
                    double xMin,
                        double xMax)
{
    const double xRange = std::max(1e-9, xMax - xMin);
    const double yRange = std::max(1e-9, yMax - yMin);

    for (const auto& point : points) {
        double yValue = point.secondary;
        if (mode == ContourSeriesMode::SmoothedSecondary) {
            yValue = point.smoothedSecondary;
        } else if (mode == ContourSeriesMode::Fluctuation) {
            yValue = point.fluctuation;
        }
        const double xRatio = (point.primary - xMin) / xRange;
        const double yRatio = (yValue - yMin) / yRange;
        const int x = plotRect.x + static_cast<int>(std::round(xRatio * plotRect.width));
        const int y = plotRect.y + plotRect.height - static_cast<int>(std::round(yRatio * plotRect.height));
        cv::circle(canvas, cv::Point(x, y), 3, color, -1, cv::LINE_AA);
    }
}

double quantileOfValues(std::vector<double> values, double q)
{
    if (values.empty()) {
        return 0.0;
    }

    q = std::clamp(q, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
    return values[std::min(index, values.size() - 1)];
}

double quantileOfAbsolute(const std::vector<double>& values, double q)
{
    std::vector<double> absValues;
    absValues.reserve(values.size());
    for (const double value : values) {
        if (std::isfinite(value)) {
            absValues.push_back(std::abs(value));
        }
    }
    return quantileOfValues(std::move(absValues), q);
}

cv::Mat cropWhiteMargin(const cv::Mat& image, int margin = 16, int threshold = 250)
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

    cv::Mat nonWhiteMask;
    cv::threshold(gray, nonWhiteMask, threshold, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(nonWhiteMask, nonZeroPoints);
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

void drawTextBox(cv::Mat& canvas,
                 const std::string& text,
                 const cv::Point& anchor,
                 double fontScale,
                 const cv::Scalar& textColor,
                 const cv::Scalar& fillColor = cv::Scalar(252, 252, 252),
                 const cv::Scalar& borderColor = cv::Scalar(220, 220, 220),
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

} // namespace

cv::Mat buildPreprocessVisualization(const cv::Mat& sourceImage,
                                     const std::vector<cv::Point2d>& filteredEdges,
                                     int imageIndex,
                                     int totalImages,
                                     const std::string& preprocessingMode)
{
    if (sourceImage.empty()) {
        return {};
    }

    cv::Mat original = toBgr(sourceImage);
    cv::Mat overlay = original.clone();

    const int radius = std::max(1, std::min(sourceImage.cols, sourceImage.rows) / 300);
    for (const auto& point : filteredEdges) {
        const int x = static_cast<int>(std::round(point.x));
        const int y = static_cast<int>(std::round(point.y));
        if (x < 0 || x >= overlay.cols || y < 0 || y >= overlay.rows) {
            continue;
        }
        cv::circle(overlay, cv::Point(x, y), radius, cv::Scalar(0, 255, 80), -1, cv::LINE_AA);
    }

    cv::Mat blended;
    cv::addWeighted(original, 0.68, overlay, 0.32, 0.0, blended);

    const int outerPad = 16;
    const int headerHeight = 54;
    const int gap = 18;
    const int canvasWidth = original.cols * 2 + gap + outerPad * 2;
    const int canvasHeight = headerHeight + original.rows + outerPad * 2;
    cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    const cv::Rect leftRect(outerPad, headerHeight, original.cols, original.rows);
    const cv::Rect rightRect(outerPad + original.cols + gap, headerHeight, blended.cols, blended.rows);
    original.copyTo(canvas(leftRect));
    blended.copyTo(canvas(rightRect));

    cv::putText(canvas,
                "Preprocess view",
                cv::Point(outerPad, 28),
                cv::FONT_HERSHEY_SIMPLEX,
                0.74,
                cv::Scalar(38, 38, 38),
                2,
                cv::LINE_AA);

    std::ostringstream info;
    info << "Image " << imageIndex << "/" << totalImages
         << " | edges " << filteredEdges.size()
         << " | " << sourceImage.cols << "x" << sourceImage.rows << " px";
    if (!preprocessingMode.empty()) {
        info << " | " << preprocessingMode;
    }
    cv::putText(canvas,
                info.str(),
                cv::Point(outerPad, 48),
                cv::FONT_HERSHEY_SIMPLEX,
                0.50,
                cv::Scalar(92, 92, 92),
                1,
                cv::LINE_AA);

    drawTextBox(canvas, "Raw field", cv::Point(leftRect.x + 8, leftRect.y + leftRect.height - 8), 0.56,
                cv::Scalar(45, 45, 45));
    std::string overlayLabel = "Filtered edge overlay";
    if (preprocessingMode == "Binary mask edge") {
        overlayLabel = "Binary mask edge overlay";
    } else if (preprocessingMode == "Binary line skeleton") {
        overlayLabel = "Binary skeleton overlay";
    }
    drawTextBox(canvas, overlayLabel,
                cv::Point(rightRect.x + 8, rightRect.y + rightRect.height - 8), 0.56, cv::Scalar(45, 45, 45));

    return cropWhiteMargin(canvas, 8);
}

cv::Mat buildStitchedContourOverlay(const cv::Mat& panorama,
                                    const std::vector<EdgeVariants>& edges,
                                    const std::vector<cv::Mat>& imageTransforms)
{
    if (panorama.empty() || edges.empty() || imageTransforms.empty()) {
        return {};
    }

    bool primaryAxisIsX = true;
    std::vector<PointSeries> seriesByImage;
    if (!collectPerImageCanvasPointSeries(edges, imageTransforms, seriesByImage, primaryAxisIsX)) {
        return {};
    }

    cv::Mat canvas(panorama.rows, panorama.cols, CV_8UC3, cv::Scalar(252, 252, 252));
    drawPanoramaEnvelope(canvas, panorama);

    const std::size_t count = std::min(seriesByImage.size(), imageTransforms.size());
    const int lineThickness = std::max(1, std::min(canvas.cols, canvas.rows) / 1400);
    for (std::size_t imageIndex = 0; imageIndex < count; ++imageIndex) {
        const std::vector<cv::Point> polyline = buildPolylinePixels(seriesByImage[imageIndex], primaryAxisIsX);
        if (polyline.size() < 2) {
            continue;
        }

        cv::polylines(canvas,
                      polyline,
                      false,
                      indexedColor(imageIndex),
                      lineThickness,
                      cv::LINE_AA);
    }

    drawTextBox(canvas,
                "Registered contour overlay",
                cv::Point(18, 34),
                0.62,
                cv::Scalar(40, 40, 40));
    std::ostringstream conciseInfo;
    conciseInfo << count << " local fields | gray: stitched envelope | color: per-field contour";
    drawTextBox(canvas, conciseInfo.str(), cv::Point(18, 64), 0.50, cv::Scalar(70, 70, 70));
    return cropWhiteMargin(canvas, 8);
}

cv::Mat buildStitchedContourProfilePlot(const std::vector<EdgeVariants>& edges,
                                        const std::vector<cv::Mat>& imageTransforms)
{
    const ContourProfileData profile = buildStitchedContourProfileData(edges, imageTransforms);
    if (profile.points.size() < 2) {
        return {};
    }

    {
        const int width = 1400;
        const int height = 860;
        const int left = 118;
        const int right = 42;
        const int top = 52;
        const int bottom = 78;
        const int gap = 58;
        const int panelHeight = (height - top - bottom - gap) / 2;
        const cv::Rect topRect(left, top, width - left - right, panelHeight);
        const cv::Rect bottomRect(left, top + panelHeight + gap, width - left - right, panelHeight);

        cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        const double xMin = profile.points.front().primary;
        const double xMax = profile.points.back().primary;

        std::vector<double> secondaryValues;
        std::vector<double> fluctuationValues;
        secondaryValues.reserve(profile.points.size() * 2);
        fluctuationValues.reserve(profile.points.size());
        for (const auto& point : profile.points) {
            secondaryValues.push_back(point.secondary);
            secondaryValues.push_back(point.smoothedSecondary);
            fluctuationValues.push_back(point.fluctuation);
        }

        double secondaryMin = *std::min_element(secondaryValues.begin(), secondaryValues.end());
        double secondaryMax = *std::max_element(secondaryValues.begin(), secondaryValues.end());
        double secondaryPad = std::max(0.8, (secondaryMax - secondaryMin) * 0.10);
        if (!std::isfinite(secondaryPad)) {
            secondaryPad = 1.0;
        }
        secondaryMin -= secondaryPad;
        secondaryMax += secondaryPad;

        double fluctuationAbs = std::max(0.15, quantileOfAbsolute(fluctuationValues, 0.985) * 1.20);
        if (!std::isfinite(fluctuationAbs) || fluctuationAbs <= 0.0) {
            fluctuationAbs = 1.0;
        }
        const double fluctuationMinPlot = -fluctuationAbs;
        const double fluctuationMaxPlot = fluctuationAbs;

        const cv::Scalar borderColor(214, 214, 214);
        const cv::Scalar gridColor(236, 236, 236);
        const cv::Scalar tickColor(96, 96, 96);
        const cv::Scalar measuredColor(182, 111, 59);
        const cv::Scalar smoothColor(81, 154, 78);
        const cv::Scalar fluctuationColor(182, 67, 67);
        const cv::Scalar pointColor(176, 184, 196);

        auto drawPanelGrid = [&](const cv::Rect& rect,
                                 double yMin,
                                 double yMax,
                                 bool drawXTicks) {
            cv::rectangle(canvas, rect, borderColor, 1, cv::LINE_AA);
            for (int tick = 0; tick <= 4; ++tick) {
                const double ratio = static_cast<double>(tick) / 4.0;
                const int y = rect.y + rect.height -
                              static_cast<int>(std::lround(ratio * rect.height));
                cv::line(canvas,
                         cv::Point(rect.x, y),
                         cv::Point(rect.x + rect.width, y),
                         gridColor,
                         1,
                         cv::LINE_AA);

                std::ostringstream label;
                label << std::fixed << std::setprecision(2) << (yMin + ratio * (yMax - yMin));
                cv::putText(canvas,
                            label.str(),
                            cv::Point(24, y + 5),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.48,
                            tickColor,
                            1,
                            cv::LINE_AA);
            }

            if (drawXTicks) {
                for (int tick = 0; tick <= 5; ++tick) {
                    const double ratio = static_cast<double>(tick) / 5.0;
                    const int x = rect.x + static_cast<int>(std::lround(ratio * rect.width));
                    cv::line(canvas,
                             cv::Point(x, rect.y),
                             cv::Point(x, rect.y + rect.height),
                             gridColor,
                             1,
                             cv::LINE_AA);
                    std::ostringstream label;
                    label << std::fixed << std::setprecision(1) << (xMin + ratio * (xMax - xMin));
                    cv::putText(canvas,
                                label.str(),
                                cv::Point(x - 18, rect.y + rect.height + 24),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.46,
                                tickColor,
                                1,
                                cv::LINE_AA);
                }
            }
        };

        drawPanelGrid(topRect, secondaryMin, secondaryMax, false);
        drawPanelGrid(bottomRect, fluctuationMinPlot, fluctuationMaxPlot, true);

        drawProfileScatter(canvas, topRect, profile.points, pointColor, ContourSeriesMode::Secondary,
                           secondaryMin, secondaryMax, xMin, xMax);
        drawLineSeries(canvas, topRect, profile.points, measuredColor, ContourSeriesMode::Secondary,
                       secondaryMin, secondaryMax, xMin, xMax);
        drawLineSeries(canvas, topRect, profile.points, smoothColor, ContourSeriesMode::SmoothedSecondary,
                       secondaryMin, secondaryMax, xMin, xMax);
        drawLineSeries(canvas, bottomRect, profile.points, fluctuationColor, ContourSeriesMode::Fluctuation,
                       fluctuationMinPlot, fluctuationMaxPlot, xMin, xMax);

        const double yRange = std::max(1e-9, fluctuationMaxPlot - fluctuationMinPlot);
        const int zeroY = bottomRect.y + bottomRect.height -
                          static_cast<int>(std::lround((0.0 - fluctuationMinPlot) / yRange * bottomRect.height));
        cv::line(canvas,
                 cv::Point(bottomRect.x, zeroY),
                 cv::Point(bottomRect.x + bottomRect.width, zeroY),
                 cv::Scalar(130, 130, 130),
                 1,
                 cv::LINE_AA);

        drawTextBox(canvas, "Stitched contour profile", cv::Point(24, 34), 0.64, cv::Scalar(42, 42, 42));
        drawTextBox(canvas,
                    profile.primaryAxisIsX ? "Primary axis: canvas X" : "Primary axis: canvas Y",
                    cv::Point(24, 64),
                    0.48,
                    cv::Scalar(72, 72, 72));

        std::ostringstream supportInfo;
        supportInfo << profile.points.size() << " profile samples";
        drawTextBox(canvas, supportInfo.str(), cv::Point(width - 252, 34), 0.48, cv::Scalar(72, 72, 72));

        drawTextBox(canvas,
                    "Measured contour and smoothed baseline",
                    cv::Point(topRect.x + 10, topRect.y + 28),
                    0.46,
                    cv::Scalar(70, 70, 70));
        drawTextBox(canvas,
                    "Local fluctuation after baseline removal",
                    cv::Point(bottomRect.x + 10, bottomRect.y + 28),
                    0.46,
                    cv::Scalar(70, 70, 70));

        const int legendX = width - 380;
        const int legendY = topRect.y + 26;
        cv::line(canvas, cv::Point(legendX, legendY), cv::Point(legendX + 32, legendY), measuredColor, 2, cv::LINE_AA);
        cv::putText(canvas, "Measured", cv::Point(legendX + 42, legendY + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
        cv::line(canvas, cv::Point(legendX + 150, legendY), cv::Point(legendX + 182, legendY), smoothColor, 2, cv::LINE_AA);
        cv::putText(canvas, "Baseline", cv::Point(legendX + 192, legendY + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
        cv::line(canvas, cv::Point(legendX, legendY + 26), cv::Point(legendX + 32, legendY + 26), fluctuationColor, 2, cv::LINE_AA);
        cv::putText(canvas, "Fluctuation", cv::Point(legendX + 42, legendY + 31),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

        cv::putText(canvas,
                    profile.primaryAxisIsX ? "Canvas Y coordinate (px)" : "Canvas X coordinate (px)",
                    cv::Point(18, topRect.y + topRect.height / 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.56,
                    cv::Scalar(82, 82, 82),
                    1,
                    cv::LINE_AA);
        cv::putText(canvas,
                    "Residual after smoothing (px)",
                    cv::Point(18, bottomRect.y + bottomRect.height / 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.56,
                    cv::Scalar(82, 82, 82),
                    1,
                    cv::LINE_AA);
        cv::putText(canvas,
                    profile.primaryAxisIsX ? "Canvas X coordinate (px)" : "Canvas Y coordinate (px)",
                    cv::Point(bottomRect.x + bottomRect.width / 2 - 105, height - 24),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.60,
                    cv::Scalar(82, 82, 82),
                    1,
                    cv::LINE_AA);

        std::ostringstream fluctuationInfo;
        fluctuationInfo << "Display range: +/- " << std::fixed << std::setprecision(2) << fluctuationAbs << " px";
        drawTextBox(canvas,
                    fluctuationInfo.str(),
                    cv::Point(bottomRect.x + 8, bottomRect.y + 28),
                    0.44,
                    cv::Scalar(72, 72, 72));

        return cropWhiteMargin(canvas, 10);
    }

    const int width = 1600;
    const int height = 980;
    const int left = 110;
    const int right = 60;
    const int top = 90;
    const int bottom = 80;
    const int gap = 80;
    const int panelHeight = (height - top - bottom - gap) / 2;
    const cv::Rect topRect(left, top, width - left - right, panelHeight);
    const cv::Rect bottomRect(left, top + panelHeight + gap, width - left - right, panelHeight);

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::putText(canvas,
                "Stitched contour fluctuation profile",
                cv::Point(32, 44),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(35, 35, 35),
                2,
                cv::LINE_AA);

    const std::string primaryLabel = profile.primaryAxisIsX ? "Primary axis: Canvas X (px)"
                                                             : "Primary axis: Canvas Y (px)";
    const std::string secondaryLabel = profile.primaryAxisIsX ? "Secondary axis: Canvas Y (px)"
                                                               : "Secondary axis: Canvas X (px)";
    cv::putText(canvas,
                primaryLabel + " | " + secondaryLabel,
                cv::Point(34, 72),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(90, 90, 90),
                1,
                cv::LINE_AA);

    const double xMin = profile.points.front().primary;
    const double xMax = profile.points.back().primary;

    double secondaryMin = profile.points.front().secondary;
    double secondaryMax = profile.points.front().secondary;
    double fluctuationMin = profile.points.front().fluctuation;
    double fluctuationMax = profile.points.front().fluctuation;
    for (const auto& point : profile.points) {
        secondaryMin = std::min(secondaryMin, std::min(point.secondary, point.smoothedSecondary));
        secondaryMax = std::max(secondaryMax, std::max(point.secondary, point.smoothedSecondary));
        fluctuationMin = std::min(fluctuationMin, point.fluctuation);
        fluctuationMax = std::max(fluctuationMax, point.fluctuation);
    }

    const double secondaryPad = std::max(1.0, (secondaryMax - secondaryMin) * 0.12);
    secondaryMin -= secondaryPad;
    secondaryMax += secondaryPad;

    const double fluctuationAbs = std::max(std::abs(fluctuationMin), std::abs(fluctuationMax));
    const double fluctuationPad = std::max(0.05, fluctuationAbs * 0.18);
    double fluctuationRange = std::max(0.1, fluctuationAbs + fluctuationPad);
    double fluctuationMinPlot = -fluctuationRange;
    double fluctuationMaxPlot = fluctuationRange;
    if (std::abs(fluctuationMinPlot - fluctuationMaxPlot) < 1e-9) {
        fluctuationMinPlot = -1.0;
        fluctuationMaxPlot = 1.0;
    }

    auto drawGrid = [](cv::Mat& target,
                       const cv::Rect& rect,
                       double yMin,
                       double yMax,
                       const cv::Scalar& axisColor) {
        cv::rectangle(target, rect, cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
        for (int tick = 0; tick <= 4; ++tick) {
            const double ratio = static_cast<double>(tick) / 4.0;
            const int y = rect.y + rect.height - static_cast<int>(std::round(ratio * rect.height));
            cv::line(target, cv::Point(rect.x, y), cv::Point(rect.x + rect.width, y),
                     cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

            std::ostringstream label;
            label << std::fixed << std::setprecision(2)
                  << (yMin + ratio * (yMax - yMin));
            cv::putText(target, label.str(), cv::Point(24, y + 6), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        axisColor, 1, cv::LINE_AA);
        }
    };

    drawGrid(canvas, topRect, secondaryMin, secondaryMax, cv::Scalar(90, 90, 90));
    drawGrid(canvas, bottomRect, fluctuationMinPlot, fluctuationMaxPlot, cv::Scalar(90, 90, 90));

    const cv::Scalar actualColor(40, 90, 240);
    const cv::Scalar smoothColor(220, 70, 70);
    const cv::Scalar fluctuationColor(30, 160, 70);
    const cv::Scalar zeroColor(120, 120, 120);

    drawLineSeries(canvas, topRect, profile.points, actualColor, ContourSeriesMode::Secondary, secondaryMin,
                   secondaryMax, xMin, xMax);
    drawLineSeries(canvas, topRect, profile.points, smoothColor, ContourSeriesMode::SmoothedSecondary, secondaryMin,
                   secondaryMax, xMin, xMax);
    drawLineSeries(canvas, bottomRect, profile.points, fluctuationColor, ContourSeriesMode::Fluctuation,
                   fluctuationMinPlot, fluctuationMaxPlot,
                   xMin, xMax);

    const double xRange = std::max(1e-9, xMax - xMin);
    const double yRange = std::max(1e-9, fluctuationMaxPlot - fluctuationMinPlot);
    const int zeroY = bottomRect.y + bottomRect.height -
                      static_cast<int>(std::round((0.0 - fluctuationMinPlot) / yRange * bottomRect.height));
    cv::line(canvas, cv::Point(bottomRect.x, zeroY), cv::Point(bottomRect.x + bottomRect.width, zeroY),
             zeroColor, 1, cv::LINE_AA);

    const int legendX = width - 440;
    const int legendY = 52;
    cv::line(canvas, cv::Point(legendX, legendY), cv::Point(legendX + 40, legendY), actualColor, 3, cv::LINE_AA);
    cv::putText(canvas, "Actual contour", cv::Point(legendX + 52, legendY + 5), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(legendX, legendY + 26), cv::Point(legendX + 40, legendY + 26), smoothColor, 3,
             cv::LINE_AA);
    cv::putText(canvas, "Smoothed baseline", cv::Point(legendX + 52, legendY + 31), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(legendX, legendY + 52), cv::Point(legendX + 40, legendY + 52), fluctuationColor, 3,
             cv::LINE_AA);
    cv::putText(canvas, "Fluctuation", cv::Point(legendX + 52, legendY + 57), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    cv::putText(canvas, "Contour value", cv::Point(18, topRect.y + topRect.height / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    cv::putText(canvas, "Fluctuation", cv::Point(18, bottomRect.y + bottomRect.height / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    cv::putText(canvas, "Primary coordinate", cv::Point(topRect.x + topRect.width / 2 - 80, height - 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);

    std::ostringstream info;
    info << "Profile points: " << profile.points.size();
    cv::putText(canvas, info.str(), cv::Point(34, height - 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    (void)xRange;
    return canvas;
}

cv::Mat buildTangentCorrelationPlot(const std::vector<StitchStepRecord>& steps,
                                    bool useInlierCorrelation)
{
    const int width = 1180;
    const int height = 640;
    const int left = 88;
    const int right = 36;
    const int top = 72;
    const int bottom = 72;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    const cv::Rect plotRect(left, top, width - left - right, height - top - bottom);
    cv::rectangle(canvas, plotRect, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    if (steps.empty()) {
        drawTextBox(canvas, "No stitching steps available", cv::Point(left + 22, top + 56), 0.62,
                    cv::Scalar(80, 80, 80));
        return canvas;
    }

    std::vector<double> values;
    values.reserve(steps.size());
    for (const auto& step : steps) {
        values.push_back(std::clamp(tangentCorrelationValue(step, useInlierCorrelation), -1.0, 1.0));
    }

    double minValue = *std::min_element(values.begin(), values.end());
    double maxValue = *std::max_element(values.begin(), values.end());
    double yMin = -1.0;
    double yMax = 1.0;
    if (minValue > 0.90 && maxValue > 0.95) {
        yMax = std::min(1.001, maxValue + 0.001);
        yMin = std::max(0.970, minValue - 0.004);
    } else {
        const double span = std::max(0.08, (maxValue - minValue) * 1.25);
        const double center = 0.5 * (minValue + maxValue);
        yMin = std::max(-1.0, center - 0.5 * span);
        yMax = std::min(1.0, center + 0.5 * span);
    }
    const double yRange = std::max(1e-6, yMax - yMin);

    for (int tick = 0; tick <= 4; ++tick) {
        const double ratio = static_cast<double>(tick) / 4.0;
        const double value = yMin + ratio * yRange;
        const int y = plotRect.y + plotRect.height -
                      static_cast<int>(std::lround(ratio * plotRect.height));
        cv::line(canvas, cv::Point(plotRect.x, y), cv::Point(plotRect.x + plotRect.width, y),
                 cv::Scalar(236, 236, 236), 1, cv::LINE_AA);

        std::ostringstream label;
        label << std::fixed << std::setprecision(3) << value;
        cv::putText(canvas, label.str(), cv::Point(24, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(92, 92, 92), 1, cv::LINE_AA);
    }

    const auto worstIt = std::min_element(values.begin(), values.end());
    const std::size_t worstIndex = static_cast<std::size_t>(std::distance(values.begin(), worstIt));
    std::vector<cv::Point> points;
    points.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double xRatio = values.size() == 1 ? 0.5 : static_cast<double>(i) / static_cast<double>(values.size() - 1);
        const double yRatio = (values[i] - yMin) / yRange;
        const int x = plotRect.x + static_cast<int>(std::lround(xRatio * plotRect.width));
        const int y = plotRect.y + plotRect.height -
                      static_cast<int>(std::lround(yRatio * plotRect.height));
        points.emplace_back(x, y);
    }

    const int highlightHalfWidth =
        std::max(6, static_cast<int>(std::lround(static_cast<double>(plotRect.width) /
                                                 std::max<std::size_t>(8, values.size() * 3))));
    int priorClampedCount = 0;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (!isPriorClampedStep(steps[i])) {
            continue;
        }
        ++priorClampedCount;
        const int bandLeft = std::max(plotRect.x, points[i].x - highlightHalfWidth);
        const int bandRight = std::min(plotRect.x + plotRect.width, points[i].x + highlightHalfWidth);
        cv::rectangle(canvas,
                      cv::Rect(bandLeft, plotRect.y, std::max(1, bandRight - bandLeft), plotRect.height),
                      cv::Scalar(246, 246, 254),
                      cv::FILLED,
                      cv::LINE_AA);
    }

    for (int tick = 0; tick <= 4; ++tick) {
        const double ratio = static_cast<double>(tick) / 4.0;
        const double value = yMin + ratio * yRange;
        const int y = plotRect.y + plotRect.height -
                      static_cast<int>(std::lround(ratio * plotRect.height));
        cv::line(canvas, cv::Point(plotRect.x, y), cv::Point(plotRect.x + plotRect.width, y),
                 cv::Scalar(236, 236, 236), 1, cv::LINE_AA);

        std::ostringstream label;
        label << std::fixed << std::setprecision(3) << value;
        cv::putText(canvas, label.str(), cv::Point(24, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(92, 92, 92), 1, cv::LINE_AA);
    }

    for (std::size_t i = 1; i < points.size(); ++i) {
        cv::line(canvas, points[i - 1], points[i], cv::Scalar(50, 74, 168), 2, cv::LINE_AA);
    }

    for (std::size_t i = 0; i < points.size(); ++i) {
        const cv::Scalar pointColor = (i == worstIndex) ? cv::Scalar(64, 67, 199) : stepStatusColor(steps[i]);
        if (isPriorClampedStep(steps[i])) {
            cv::rectangle(canvas,
                          cv::Rect(points[i].x - 5, points[i].y - 5, 10, 10),
                          pointColor,
                          cv::FILLED,
                          cv::LINE_AA);
            cv::rectangle(canvas,
                          cv::Rect(points[i].x - 6, points[i].y - 6, 12, 12),
                          cv::Scalar(255, 255, 255),
                          1,
                          cv::LINE_AA);
        } else {
            cv::circle(canvas, points[i], 5, pointColor, -1, cv::LINE_AA);
        }
        cv::putText(canvas,
                    std::to_string(static_cast<int>(steps[i].stepIndex)),
                    cv::Point(points[i].x - 7, plotRect.y + plotRect.height + 26),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.43,
                    cv::Scalar(95, 95, 95),
                    1,
                    cv::LINE_AA);
    }

    drawTextBox(canvas,
                useInlierCorrelation ? "Tangent correlation of inlier samples" : "Tangent correlation of all samples",
                cv::Point(left, 30),
                0.60,
                cv::Scalar(38, 38, 38));
    if (!values.empty()) {
        std::ostringstream info;
        info << "min step = " << steps[worstIndex].stepIndex << ", corr = "
             << std::fixed << std::setprecision(4) << values[worstIndex];
        if (priorClampedCount > 0) {
            info << " | prior-clamped = " << priorClampedCount;
        }
        drawTextBox(canvas, info.str(), cv::Point(left, 58), 0.48, cv::Scalar(72, 72, 72));
    }

    cv::putText(canvas, "Stitching step", cv::Point(plotRect.x + plotRect.width / 2 - 50, height - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(72, 72, 72), 1, cv::LINE_AA);
    cv::putText(canvas, "Correlation", cv::Point(18, plotRect.y - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(72, 72, 72), 1, cv::LINE_AA);
    return canvas;
}

cv::Mat buildDebugStepVisualization(const cv::Mat& imgRef,
                                    const cv::Mat& imgTarget,
                                    const std::vector<cv::Point2d>& edgesRef,
                                    const std::vector<cv::Point2d>& edgesTarget,
                                    const StitchStepRecord& step)
{
    const TransformResult& result = step.transform;
    const int stepIndex = static_cast<int>(step.stepIndex);
    {
        const int height = imgRef.rows;
        const int padX = std::max(40, imgRef.cols / 6);
        const int padY = std::max(40, imgRef.rows / 6);
        const int canvasWidth = imgRef.cols + imgTarget.cols + 2 * padX;
        const int canvasHeight = height + 2 * padY;
        const int plotHeight = 320;

        cv::Mat overlay(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat refRoi = overlay(cv::Rect(padX, padY, imgRef.cols, imgRef.rows));
        if (imgRef.channels() == 1) {
            cv::cvtColor(imgRef, refRoi, cv::COLOR_GRAY2BGR);
        } else {
            imgRef.copyTo(refRoi);
        }

        const cv::Point2d center(imgTarget.cols / 2.0, imgTarget.rows / 2.0);
        cv::Mat rotation = cv::getRotationMatrix2D(center, result.da, 1.0);
        cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);
        rotation.copyTo(transform(cv::Rect(0, 0, 3, 2)));
        transform.at<double>(0, 2) += result.dx + padX;
        transform.at<double>(1, 2) += result.dy + padY;

        cv::Mat warpedTarget;
        cv::warpAffine(imgTarget, warpedTarget, transform.rowRange(0, 2), cv::Size(canvasWidth, canvasHeight),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        if (warpedTarget.channels() == 1) {
            cv::cvtColor(warpedTarget, warpedTarget, cv::COLOR_GRAY2BGR);
        }
        cv::min(overlay, warpedTarget, overlay);

        const int pointRadius = std::max(2, std::min(imgRef.cols, imgRef.rows) / 900);
        for (const auto& point : edgesRef) {
            const cv::Point2d shifted = point + cv::Point2d(padX, padY);
            cv::circle(overlay, shifted, pointRadius, cv::Scalar(214, 78, 64), -1, cv::LINE_AA);
        }

        std::vector<cv::Point2d> transformedTarget;
        transformTargetPointsForResult(edgesTarget, transformedTarget, center, result);

        std::vector<double> errors;
        std::vector<double> validCoordinates;
        const bool usePrecomputed =
            !result.inlierErrors.empty() && result.inlierErrors.size() == result.inlierCoordinates.size();
        if (usePrecomputed) {
            errors = result.inlierErrors;
            validCoordinates = result.inlierCoordinates;
        }

        for (auto& point : transformedTarget) {
            point.x += padX;
            point.y += padY;

            if (point.x >= 0 && point.x < canvasWidth && point.y >= 0 && point.y < canvasHeight) {
                cv::circle(overlay, point, pointRadius, cv::Scalar(44, 111, 220), -1, cv::LINE_AA);
            }

            if (!usePrecomputed) {
                if (result.axis == AlignmentAxis::X) {
                    double refValue = 0.0;
                    if (getInterpolatedY(edgesRef, point.x, refValue)) {
                        errors.push_back(refValue - point.y);
                        validCoordinates.push_back(point.x);
                    }
                } else {
                    double refValue = 0.0;
                    if (getInterpolatedX(edgesRef, point.y, refValue)) {
                        errors.push_back(refValue - point.x);
                        validCoordinates.push_back(point.y);
                    }
                }
            }
        }

        cv::Mat dashboard(canvasHeight + plotHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));
        overlay.copyTo(dashboard(cv::Rect(0, 0, canvasWidth, canvasHeight)));

        drawTextBox(dashboard,
                    "Adjacent-view contour overlap",
                    cv::Point(18, 30),
                    0.60,
                    cv::Scalar(40, 40, 40));
        drawTextBox(dashboard,
                    stepStatusLabel(step),
                    cv::Point(std::max(18, canvasWidth - 260), 30),
                    0.50,
                    stepStatusColor(step),
                    cv::Scalar(252, 252, 252),
                    cv::Scalar(218, 218, 218));

        const ResidualStatistics& normalDisplay =
            result.metrics.normalInlier.valid() ? result.metrics.normalInlier : result.metrics.normalAll;
        const ResidualStatistics& tangentDisplay =
            result.metrics.tangentInlier.valid() ? result.metrics.tangentInlier : result.metrics.tangentAll;
        const double tangentCorrDisplay =
            result.metrics.tangentInlier.valid() ? result.metrics.tangentCorrInlier : result.metrics.tangentCorrAll;

        std::ostringstream summary;
        summary << "step " << stepIndex
                << " | normal RMSE " << std::fixed << std::setprecision(4) << normalDisplay.rmse << " px"
                << " | tangent RMSE " << tangentDisplay.rmse << " px"
                << " | corr " << tangentCorrDisplay;
        drawTextBox(dashboard, summary.str(), cv::Point(18, 58), 0.48, cv::Scalar(72, 72, 72));

        std::ostringstream summary2;
        summary2 << "inliers " << result.metrics.inlierCount << "/" << result.metrics.overlapCount
                 << " | coverage " << std::setprecision(3) << result.metrics.inlierCoverageRatio
                 << " | dx " << result.dx << " px, dy " << result.dy << " px, da " << result.da << " deg";
        drawTextBox(dashboard, summary2.str(), cv::Point(18, 86), 0.46, cv::Scalar(72, 72, 72));

        std::ostringstream summary3;
        const double actualPrimary = motionPrimaryShift(step, result.dx, result.dy);
        const double actualPerp = motionPerpendicularShift(step, result.dx, result.dy);
        summary3 << "motion " << motionAxisLabel(step.motionAxis)
                 << " | primary " << std::fixed << std::setprecision(1) << actualPrimary
                 << " px | perp " << actualPerp << " px";
        if (step.hasNominalPrior) {
            summary3 << " | nominal "
                     << motionPrimaryShift(step, step.nominalPriorDx, step.nominalPriorDy) << " px";
        }
        if (step.hasTrajectoryPrior) {
            summary3 << " | traj "
                     << motionPrimaryShift(step, step.trajectoryPriorDx, step.trajectoryPriorDy) << " px";
        }
        if (step.hasImageCorrelationEstimate) {
            summary3 << " | raw img "
                     << motionPrimaryShift(step, step.imageCorrelationDx, step.imageCorrelationDy) << " px"
                     << " (corr " << std::setprecision(3) << step.imageCorrelationScore << ")";
        }
        drawTextBox(dashboard, summary3.str(), cv::Point(18, 114), 0.44, cv::Scalar(72, 72, 72));

        const cv::Rect plotRect(78, canvasHeight + 34, canvasWidth - 118, plotHeight - 76);
        cv::rectangle(dashboard, plotRect, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

        if (!errors.empty() && errors.size() == validCoordinates.size()) {
            std::vector<std::pair<double, double>> errorPairs;
            errorPairs.reserve(errors.size());
            for (std::size_t i = 0; i < errors.size(); ++i) {
                errorPairs.emplace_back(validCoordinates[i], errors[i]);
            }
            std::sort(errorPairs.begin(), errorPairs.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

            const double xMin = errorPairs.front().first;
            const double xMax = std::max(xMin + 1.0, errorPairs.back().first);
            const double yAbs = std::max(0.20, quantileOfAbsolute(errors, 0.985) * 1.18);
            const double yMin = -yAbs;
            const double yMax = yAbs;
            const double xRange = std::max(1e-9, xMax - xMin);
            const double yRange = std::max(1e-9, yMax - yMin);

            for (int tick = 0; tick <= 4; ++tick) {
                const double ratio = static_cast<double>(tick) / 4.0;
                const double value = yMin + ratio * yRange;
                const int y = plotRect.y + plotRect.height -
                              static_cast<int>(std::lround(ratio * plotRect.height));
                cv::line(dashboard, cv::Point(plotRect.x, y), cv::Point(plotRect.x + plotRect.width, y),
                         cv::Scalar(236, 236, 236), 1, cv::LINE_AA);
                std::ostringstream label;
                label << std::fixed << std::setprecision(2) << value;
                cv::putText(dashboard, label.str(), cv::Point(24, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                            cv::Scalar(92, 92, 92), 1, cv::LINE_AA);
            }

            const int zeroY = plotRect.y + plotRect.height -
                              static_cast<int>(std::lround((0.0 - yMin) / yRange * plotRect.height));
            cv::line(dashboard, cv::Point(plotRect.x, zeroY), cv::Point(plotRect.x + plotRect.width, zeroY),
                     cv::Scalar(135, 135, 135), 1, cv::LINE_AA);
            if (yMin <= 0.25 && yMax >= 0.25) {
                const int upperY = plotRect.y + plotRect.height -
                                   static_cast<int>(std::lround((0.25 - yMin) / yRange * plotRect.height));
                const int lowerY = plotRect.y + plotRect.height -
                                   static_cast<int>(std::lround((-0.25 - yMin) / yRange * plotRect.height));
                cv::line(dashboard, cv::Point(plotRect.x, upperY), cv::Point(plotRect.x + plotRect.width, upperY),
                         cv::Scalar(214, 214, 230), 1, cv::LINE_AA);
                cv::line(dashboard, cv::Point(plotRect.x, lowerY), cv::Point(plotRect.x + plotRect.width, lowerY),
                         cv::Scalar(214, 214, 230), 1, cv::LINE_AA);
            }

            for (const auto& errorPair : errorPairs) {
                const double xRatio = (errorPair.first - xMin) / xRange;
                const double yRatio = (errorPair.second - yMin) / yRange;
                const int px = plotRect.x + static_cast<int>(std::lround(xRatio * plotRect.width));
                const int py = plotRect.y + plotRect.height -
                               static_cast<int>(std::lround(yRatio * plotRect.height));

                cv::Scalar color(34, 159, 110);
                if (std::abs(errorPair.second) > 0.75 * yAbs) {
                    color = cv::Scalar(74, 85, 217);
                } else if (std::abs(errorPair.second) > 0.45 * yAbs) {
                    color = cv::Scalar(64, 168, 255);
                }

                if (px >= plotRect.x && px <= plotRect.x + plotRect.width &&
                    py >= plotRect.y && py <= plotRect.y + plotRect.height) {
                    cv::circle(dashboard, cv::Point(px, py), 2, color, -1, cv::LINE_AA);
                }
            }

            std::ostringstream axisInfo;
            axisInfo << "residual range ±" << std::fixed << std::setprecision(2) << yAbs << " px";
            drawTextBox(dashboard, axisInfo.str(), cv::Point(plotRect.x + 8, plotRect.y + 24), 0.44,
                        cv::Scalar(92, 92, 92));
            if (isPriorClampedStep(step)) {
                drawTextBox(dashboard,
                            "Local optimum was rejected; this step is reported with a prior-clamped transform.",
                            cv::Point(plotRect.x + 8, plotRect.y + 52),
                            0.40,
                            cv::Scalar(88, 96, 176),
                            cv::Scalar(252, 252, 252),
                            cv::Scalar(224, 224, 236));
            }

            cv::putText(dashboard, "Overlap coordinate (px)",
                        cv::Point(plotRect.x + plotRect.width / 2 - 64, canvasHeight + plotHeight - 18),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.58,
                        cv::Scalar(72, 72, 72),
                        1,
                        cv::LINE_AA);
            cv::putText(dashboard, "Normal residual (px)",
                        cv::Point(18, plotRect.y - 14),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.56,
                        cv::Scalar(72, 72, 72),
                        1,
                        cv::LINE_AA);
        }

        return cropWhiteMargin(dashboard, 14);
    }

    const int height = imgRef.rows;
    const int padX = imgRef.cols / 2;
    const int padY = imgRef.rows / 2;
    const int canvasWidth = imgRef.cols + imgTarget.cols + 2 * padX;
    const int canvasHeight = height + 2 * padY;
    const int plotHeight = 1000;

    cv::Mat dashboard(canvasHeight + plotHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat overlay = cv::Mat::zeros(canvasHeight, canvasWidth, CV_8UC3);
    overlay.setTo(cv::Scalar(255, 255, 255));

    cv::Mat refRoi = overlay(cv::Rect(padX, padY, imgRef.cols, imgRef.rows));
    if (imgRef.channels() == 1) {
        cv::cvtColor(imgRef, refRoi, cv::COLOR_GRAY2BGR);
    } else {
        imgRef.copyTo(refRoi);
    }

    const cv::Point2d center(imgTarget.cols / 2.0, imgTarget.rows / 2.0);
    cv::Mat rotation = cv::getRotationMatrix2D(center, result.da, 1.0);
    cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);
    rotation.copyTo(transform(cv::Rect(0, 0, 3, 2)));
    transform.at<double>(0, 2) += result.dx + padX;
    transform.at<double>(1, 2) += result.dy + padY;

    cv::Mat warpedTarget;
    cv::warpAffine(imgTarget, warpedTarget, transform.rowRange(0, 2), cv::Size(canvasWidth, canvasHeight),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    if (warpedTarget.channels() == 1) {
        cv::cvtColor(warpedTarget, warpedTarget, cv::COLOR_GRAY2BGR);
    }
    cv::min(overlay, warpedTarget, overlay);

    for (const auto& point : edgesRef) {
        const cv::Point2d shifted = point + cv::Point2d(padX, padY);
        cv::circle(overlay, shifted, 5, cv::Scalar(255, 0, 0), -1);
    }

    std::vector<cv::Point2d> transformedTarget;
    transformTargetPointsForResult(edgesTarget, transformedTarget, center, result);

    std::vector<double> errors;
    std::vector<double> validCoordinates;

    const bool usePrecomputed =
        !result.inlierErrors.empty() && result.inlierErrors.size() == result.inlierCoordinates.size();
    if (usePrecomputed) {
        errors = result.inlierErrors;
        validCoordinates = result.inlierCoordinates;
    }

    for (auto& point : transformedTarget) {
        point.x += padX;
        point.y += padY;

        if (point.x >= 0 && point.x < canvasWidth && point.y >= 0 && point.y < canvasHeight) {
            cv::circle(overlay, point, 5, cv::Scalar(0, 0, 255), -1);
        }

        if (!usePrecomputed) {
            if (result.axis == AlignmentAxis::X) {
                double refValue = 0.0;
                if (getInterpolatedY(edgesRef, point.x, refValue)) {
                    const double error = refValue - point.y;
                    errors.push_back(error);
                    validCoordinates.push_back(point.x);
                }
            } else {
                double refValue = 0.0;
                if (getInterpolatedX(edgesRef, point.y, refValue)) {
                    const double error = refValue - point.x;
                    errors.push_back(error);
                    validCoordinates.push_back(point.y);
                }
            }
        }
    }

    overlay.copyTo(dashboard(cv::Rect(0, 0, canvasWidth, canvasHeight)));

    cv::Mat plotRoi = dashboard(cv::Rect(0, canvasHeight, canvasWidth, plotHeight));
    plotRoi.setTo(cv::Scalar(30, 30, 30));

    if (!errors.empty()) {
        const int centerY = plotHeight / 2;
        cv::line(plotRoi, cv::Point(0, centerY), cv::Point(plotRoi.cols, centerY), cv::Scalar(100, 100, 100), 5);

        std::vector<std::pair<double, double>> errorPairs;
        errorPairs.reserve(errors.size());
        for (std::size_t i = 0; i < errors.size() && i < validCoordinates.size(); ++i) {
            errorPairs.emplace_back(validCoordinates[i], errors[i]);
        }

        std::sort(errorPairs.begin(), errorPairs.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        if (!errorPairs.empty()) {
            double xRange = errorPairs.back().first - errorPairs.front().first;
            if (xRange < 1.0) {
                xRange = 1.0;
            }

            const double scaleY = 50.0;
            for (const auto& errorPair : errorPairs) {
                const double xNorm = (errorPair.first - errorPairs.front().first) / xRange;
                const int px = static_cast<int>(xNorm * (plotRoi.cols - 50)) + 25;
                const int py = centerY - static_cast<int>(errorPair.second * scaleY);

                cv::Scalar color(0, 255, 0);
                if (std::abs(errorPair.second) > 0.5) {
                    color = cv::Scalar(0, 0, 255);
                } else if (std::abs(errorPair.second) > 0.2) {
                    color = cv::Scalar(0, 255, 255);
                }

                if (px >= 0 && px < plotRoi.cols && py >= 0 && py < plotRoi.rows) {
                    cv::circle(plotRoi, cv::Point(px, py), 3, color, -1);
                }
            }
        }

        const ResidualStatistics& normalDisplay =
            result.metrics.normalInlier.valid() ? result.metrics.normalInlier : result.metrics.normalAll;
        const ResidualStatistics& tangentDisplay =
            result.metrics.tangentInlier.valid() ? result.metrics.tangentInlier : result.metrics.tangentAll;
        const double tangentCorrDisplay =
            result.metrics.tangentInlier.valid() ? result.metrics.tangentCorrInlier : result.metrics.tangentCorrAll;

        std::ostringstream text;
        text << "步骤 " << stepIndex << " 调试分析";
        cv::putText(plotRoi, text.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    cv::Scalar(255, 255, 255), 5);

        text.str("");
        text.clear();
        text << "法向 RMSE(显示): " << std::setprecision(4) << normalDisplay.rmse << " px";
        cv::putText(plotRoi, text.str(), cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    cv::Scalar(0, 255, 0), 5);

        text.str("");
        text.clear();
        text << "法向 RMSE(全部): " << std::setprecision(4) << result.metrics.normalAll.rmse << " px";
        cv::putText(plotRoi, text.str(), cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 1.1,
                    cv::Scalar(150, 255, 150), 3);

        text.str("");
        text.clear();
        text << "切向 RMSE(显示): " << std::setprecision(4) << tangentDisplay.rmse << " px";
        cv::putText(plotRoi, text.str(), cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(255, 255, 0), 4);

        text.str("");
        text.clear();
        text << "切向相关(显示): " << std::setprecision(4) << tangentCorrDisplay;
        cv::putText(plotRoi, text.str(), cv::Point(20, 190), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(255, 255, 0), 4);

        text.str("");
        text.clear();
        text << "覆盖率(重叠/内点): " << std::setprecision(3) << result.metrics.overlapCoverageRatio
             << " / " << result.metrics.inlierCoverageRatio;
        cv::putText(plotRoi, text.str(), cv::Point(20, 230), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(255, 255, 255), 4);

        text.str("");
        text.clear();
        text << "重叠点/内点: " << result.metrics.overlapCount << "/" << result.metrics.inlierCount
             << " (" << std::setprecision(3) << result.metrics.inlierRatio << ")";
        cv::putText(plotRoi, text.str(), cv::Point(20, 270), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(255, 255, 255), 4);
    }

    cv::Mat dashboardGray;
    cv::cvtColor(dashboard, dashboardGray, cv::COLOR_BGR2GRAY);
    cv::Mat nonWhiteMask;
    cv::threshold(dashboardGray, nonWhiteMask, 250, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(nonWhiteMask, nonZeroPoints);

    if (nonZeroPoints.empty()) {
        return dashboard;
    }

    cv::Rect boundingBox = cv::boundingRect(nonZeroPoints);
    const int margin = 20;
    boundingBox.x = std::max(0, boundingBox.x - margin);
    boundingBox.y = std::max(0, boundingBox.y - margin);
    boundingBox.width = std::min(dashboard.cols - boundingBox.x, boundingBox.width + 2 * margin);
    boundingBox.height = std::min(dashboard.rows - boundingBox.y, boundingBox.height + 2 * margin);
    return dashboard(boundingBox).clone();
}

std::string formatStepSummary(const StitchStepRecord& step)
{
    {
        const TransformResult& transform = step.transform;
        const AlignmentMetrics& metrics = transform.metrics;
        const ResidualStatistics& normalDisplay =
            metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
        const ResidualStatistics& tangentDisplay =
            metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
        const double tangentCorrDisplay =
            metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;

        std::ostringstream summary;
        summary << "    结果：dx=" << transform.dx
                << "，dy=" << transform.dy
                << "，角度=" << transform.da
                << "，法向RMSE(显示)=" << normalDisplay.rmse
                << "，法向RMSE(全部)=" << metrics.normalAll.rmse
                << "，切向RMSE(显示)=" << tangentDisplay.rmse
                << "，切向相关(显示)=" << tangentCorrDisplay
                << "，score=" << transform.score
                << " [法向代价=" << transform.normalMatchCost
                << "，切向RMSE代价=" << transform.tangentResidualMatchCost
                << "，切向相关代价=" << transform.tangentCorrelationMatchCost
                << "，方向惩罚=" << transform.directionPenaltyMatchCost << "]"
                << "，覆盖率(重叠/内点)=" << metrics.overlapCoverageRatio << "/" << metrics.inlierCoverageRatio
                << "，重叠点数=" << metrics.overlapCount
                << "，内点数=" << metrics.inlierCount
                << " [方向=" << transform.direction
                << "，sx=" << step.searchRangeX
                << "，sy=" << step.searchRangeY
                << "，模式=" << step.selectionMode << "]";
        if (isPriorClampedStep(step)) {
            summary << " [PriorClamp]";
        }
        return summary.str();
    }
    const TransformResult& result = step.transform;
    const AlignmentMetrics& metrics = result.metrics;
    const ResidualStatistics& normalDisplay = metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
    const ResidualStatistics& tangentDisplay = metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
    const double tangentCorrDisplay = metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;

    std::ostringstream stream;
    stream << "    结果：dx=" << result.dx
           << "，dy=" << result.dy
           << "，角度=" << result.da
           << "，法向RMSE(显示)=" << normalDisplay.rmse
           << "，法向RMSE(全部)=" << metrics.normalAll.rmse
           << "，切向RMSE(显示)=" << tangentDisplay.rmse
           << "，切向相关(显示)=" << tangentCorrDisplay
           << "，score=" << result.score
           << " [法向代价=" << result.normalMatchCost
           << "，切向RMSE代价=" << result.tangentResidualMatchCost
           << "，切向相关代价=" << result.tangentCorrelationMatchCost
           << "，方向惩罚=" << result.directionPenaltyMatchCost << "]"
           << "，覆盖率(重叠/内点)=" << metrics.overlapCoverageRatio << "/" << metrics.inlierCoverageRatio
           << "，重叠点数=" << metrics.overlapCount
           << "，内点数=" << metrics.inlierCount
           << " [方向=" << result.direction
           << "，sx=" << step.searchRangeX
           << "，sy=" << step.searchRangeY << "]";
    return stream.str();
}

std::string buildContourPointCsv(const std::vector<EdgeVariants>& edges,
                                 const std::vector<cv::Mat>& imageTransforms)
{
    bool primaryAxisIsX = true;
    std::vector<PointSeries> seriesByImage;
    collectPerImageCanvasPointSeries(edges, imageTransforms, seriesByImage, primaryAxisIsX);

    std::ostringstream stream;
    stream << "ImageIndex,PointIndex,PrimaryAxis,SourceX(px),SourceY(px),CanvasX(px),CanvasY(px),"
              "PrimaryCoord(px),SecondaryCoord(px)\n";

    const std::size_t count = std::min(edges.size(), imageTransforms.size());
    for (std::size_t imageIndex = 0; imageIndex < count; ++imageIndex) {
        const cv::Mat& transform = imageTransforms[imageIndex];
        if (transform.empty() || transform.rows < 2 || transform.cols < 3) {
            continue;
        }

        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            const cv::Point2d& sourcePoint = edges[imageIndex].raw[pointIndex];
            const cv::Point2d canvasPoint = transformPoint(transform, sourcePoint);
            stream << (imageIndex + 1) << ","
                   << (pointIndex + 1) << ","
                   << (primaryAxisIsX ? "X" : "Y") << ","
                   << formatDouble(sourcePoint.x) << ","
                   << formatDouble(sourcePoint.y) << ","
                   << formatDouble(canvasPoint.x) << ","
                   << formatDouble(canvasPoint.y) << ","
                   << formatDouble(primaryAxisIsX ? canvasPoint.x : canvasPoint.y) << ","
                   << formatDouble(primaryAxisIsX ? canvasPoint.y : canvasPoint.x) << "\n";
        }
    }

    return stream.str();
}

std::string buildOriginContourOverlayCsv(const std::vector<EdgeVariants>& edges,
                                         const std::vector<cv::Mat>& imageTransforms)
{
    bool primaryAxisIsX = true;
    std::vector<PointSeries> seriesByImage;
    if (!collectPerImageCanvasPointSeries(edges, imageTransforms, seriesByImage, primaryAxisIsX)) {
        return {};
    }

    std::size_t maxPointCount = 0;
    for (const auto& series : seriesByImage) {
        maxPointCount = std::max(maxPointCount, series.size());
    }

    std::ostringstream stream;
    stream << "PointIndex,PrimaryAxis";
    for (std::size_t imageIndex = 0; imageIndex < seriesByImage.size(); ++imageIndex) {
        stream << ",Image" << (imageIndex + 1) << "_X(px)"
               << ",Image" << (imageIndex + 1) << "_Y(px)";
    }
    stream << "\n";

    for (std::size_t pointIndex = 0; pointIndex < maxPointCount; ++pointIndex) {
        stream << (pointIndex + 1) << "," << (primaryAxisIsX ? "X" : "Y");
        for (const auto& series : seriesByImage) {
            if (pointIndex < series.size()) {
                stream << "," << formatDouble(series[pointIndex].x)
                       << "," << formatDouble(series[pointIndex].y);
            } else {
                stream << ",,";
            }
        }
        stream << "\n";
    }

    return stream.str();
}

std::string buildStitchedContourProfileCsv(const std::vector<EdgeVariants>& edges,
                                           const std::vector<cv::Mat>& imageTransforms)
{
    const ContourProfileData profile = buildStitchedContourProfileData(edges, imageTransforms);
    std::ostringstream stream;
    stream << "PointIndex,ProfileAxis,PrimaryCoord(px),SecondaryCoord(px),SmoothedSecondary(px),Fluctuation(px),"
              "SupportCount,CanvasX(px),CanvasY(px)\n";

    for (std::size_t index = 0; index < profile.points.size(); ++index) {
        const auto& point = profile.points[index];
        const double canvasX = profile.primaryAxisIsX ? point.primary : point.secondary;
        const double canvasY = profile.primaryAxisIsX ? point.secondary : point.primary;
        stream << (index + 1) << ","
               << (profile.primaryAxisIsX ? "X" : "Y") << ","
               << formatDouble(point.primary) << ","
               << formatDouble(point.secondary) << ","
               << formatDouble(point.smoothedSecondary) << ","
               << formatDouble(point.fluctuation) << ","
               << point.supportCount << ","
               << formatDouble(canvasX) << ","
               << formatDouble(canvasY) << "\n";
    }

    return stream.str();
}

std::string buildTangentCorrelationStepCsv(const std::vector<StitchStepRecord>& steps)
{
    {
        std::ostringstream stream;
        stream << "Step,ImageA,ImageB,Direction,SelectionMode,PriorClamped,ForcedPriorClamp,"
                  "SearchRangeX(px),SearchRangeY(px),"
                  "MatchScore,NormalMatchCost,TangentResidualMatchCost,TangentCorrelationMatchCost,DirectionPenaltyMatchCost,"
                  "NormalRMSEDisplay(px),TangentRMSEDisplay(px),TangentCorrDisplay,"
                  "TangentCorrAll,TangentCorrInlier,OverlapCoverageRatio,InlierCoverageRatio,InlierRatio\n";

        for (const auto& step : steps) {
            const TransformResult& result = step.transform;
            const AlignmentMetrics& metrics = result.metrics;
            const ResidualStatistics& normalDisplay =
                metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
            const ResidualStatistics& tangentDisplay =
                metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
            const double tangentCorrDisplay =
                metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;

            stream << step.stepIndex << ","
                   << (step.referenceImageIndex + 1) << ","
                   << (step.targetImageIndex + 1) << ","
                   << result.direction << ","
                   << step.selectionMode << ","
                   << (isPriorClampedStep(step) ? 1 : 0) << ","
                   << (step.usedForcedTrajectoryPriorFallback ? 1 : 0) << ","
                   << formatDouble(step.searchRangeX) << ","
                   << formatDouble(step.searchRangeY) << ","
                   << formatDouble(result.score) << ","
                   << formatDouble(result.normalMatchCost) << ","
                   << formatDouble(result.tangentResidualMatchCost) << ","
                   << formatDouble(result.tangentCorrelationMatchCost) << ","
                   << formatDouble(result.directionPenaltyMatchCost) << ","
                   << formatDouble(normalDisplay.rmse) << ","
                   << formatDouble(tangentDisplay.rmse) << ","
                   << formatDouble(tangentCorrDisplay) << ","
                   << formatDouble(metrics.tangentCorrAll) << ","
                   << formatDouble(metrics.tangentCorrInlier) << ","
                   << formatDouble(metrics.overlapCoverageRatio) << ","
                   << formatDouble(metrics.inlierCoverageRatio) << ","
                   << formatDouble(metrics.inlierRatio) << "\n";
        }

        return stream.str();
    }
    std::ostringstream stream;
    stream << "Step,ImageA,ImageB,Direction,SearchRangeX(px),SearchRangeY(px),"
              "MatchScore,NormalMatchCost,TangentResidualMatchCost,TangentCorrelationMatchCost,DirectionPenaltyMatchCost,"
              "NormalRMSEDisplay(px),TangentRMSEDisplay(px),TangentCorrDisplay,"
              "TangentCorrAll,TangentCorrInlier,OverlapCoverageRatio,InlierCoverageRatio,InlierRatio\n";

    for (const auto& step : steps) {
        const TransformResult& result = step.transform;
        const AlignmentMetrics& metrics = result.metrics;
        const ResidualStatistics& normalDisplay = metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
        const ResidualStatistics& tangentDisplay =
            metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
        const double tangentCorrDisplay =
            metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;

        stream << step.stepIndex << ","
               << (step.referenceImageIndex + 1) << ","
               << (step.targetImageIndex + 1) << ","
               << result.direction << ","
               << formatDouble(step.searchRangeX) << ","
               << formatDouble(step.searchRangeY) << ","
               << formatDouble(result.score) << ","
               << formatDouble(result.normalMatchCost) << ","
               << formatDouble(result.tangentResidualMatchCost) << ","
               << formatDouble(result.tangentCorrelationMatchCost) << ","
               << formatDouble(result.directionPenaltyMatchCost) << ","
               << formatDouble(normalDisplay.rmse) << ","
               << formatDouble(tangentDisplay.rmse) << ","
               << formatDouble(tangentCorrDisplay) << ","
               << formatDouble(metrics.tangentCorrAll) << ","
               << formatDouble(metrics.tangentCorrInlier) << ","
               << formatDouble(metrics.overlapCoverageRatio) << ","
               << formatDouble(metrics.inlierCoverageRatio) << ","
               << formatDouble(metrics.inlierRatio) << "\n";
    }

    return stream.str();
}

std::string buildNormalErrorProfileCsv(const std::vector<StitchStepRecord>& steps)
{
    std::ostringstream stream;
    stream << "Step,ImageA,ImageB,SampleIndex,PrimaryCoord(px),NormalInlierError(px)\n";

    for (const auto& step : steps) {
        const std::size_t count = std::min(step.transform.inlierCoordinates.size(), step.transform.inlierErrors.size());
        for (std::size_t sampleIndex = 0; sampleIndex < count; ++sampleIndex) {
            stream << step.stepIndex << ","
                   << (step.referenceImageIndex + 1) << ","
                   << (step.targetImageIndex + 1) << ","
                   << (sampleIndex + 1) << ","
                   << formatDouble(step.transform.inlierCoordinates[sampleIndex]) << ","
                   << formatDouble(step.transform.inlierErrors[sampleIndex]) << "\n";
        }
    }

    return stream.str();
}

std::string buildTangentProfileCompareCsv(const std::vector<StitchStepRecord>& steps)
{
    std::ostringstream stream;
    stream << "Step,ImageA,ImageB,Axis,Direction,SampleIndex,PrimaryCoord(px),"
              "RefSecondary(px),TargetSecondary(px),RefSmoothed(px),TargetSmoothed(px),"
              "RefFluctuation(px),TargetFluctuation(px),FluctuationDelta(px),"
              "NormalError(px),TangentError(px),IsInlier,TangentErrorAbs(px),NormalErrorAbs(px)\n";

    for (const auto& step : steps) {
        const TransformResult& transform = step.transform;
        const std::vector<double> refSmoothed = buildSmoothedSeries(transform.sampleRefSecondaryCoordinates);
        const std::vector<double> targetSmoothed = buildSmoothedSeries(transform.sampleTargetSecondaryCoordinates);
        const std::vector<double> refFluctuation = buildFluctuationSeries(transform.sampleRefSecondaryCoordinates);
        const std::vector<double> targetFluctuation =
            buildFluctuationSeries(transform.sampleTargetSecondaryCoordinates);
        const std::size_t sampleCount = std::min({transform.samplePrimaryCoordinates.size(),
                                                  transform.sampleRefSecondaryCoordinates.size(),
                                                  transform.sampleTargetSecondaryCoordinates.size(),
                                                  refSmoothed.size(),
                                                  targetSmoothed.size(),
                                                  refFluctuation.size(),
                                                  targetFluctuation.size(),
                                                  transform.sampleNormalErrors.size(),
                                                  transform.sampleTangentErrors.size(),
                                                  transform.sampleInlierFlags.size()});
        const char axisLabel = transform.axis == AlignmentAxis::X ? 'X' : 'Y';

        for (std::size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
            stream << step.stepIndex << ","
                   << (step.referenceImageIndex + 1) << ","
                   << (step.targetImageIndex + 1) << ","
                   << axisLabel << ","
                   << transform.direction << ","
                   << (sampleIndex + 1) << ","
                   << formatDouble(transform.samplePrimaryCoordinates[sampleIndex]) << ","
                   << formatDouble(transform.sampleRefSecondaryCoordinates[sampleIndex]) << ","
                   << formatDouble(transform.sampleTargetSecondaryCoordinates[sampleIndex]) << ","
                   << formatDouble(refSmoothed[sampleIndex]) << ","
                   << formatDouble(targetSmoothed[sampleIndex]) << ","
                   << formatDouble(refFluctuation[sampleIndex]) << ","
                   << formatDouble(targetFluctuation[sampleIndex]) << ","
                   << formatDouble(targetFluctuation[sampleIndex] - refFluctuation[sampleIndex]) << ","
                   << formatDouble(transform.sampleNormalErrors[sampleIndex]) << ","
                   << formatDouble(transform.sampleTangentErrors[sampleIndex]) << ","
                   << transform.sampleInlierFlags[sampleIndex] << ","
                   << formatDouble(std::abs(transform.sampleTangentErrors[sampleIndex])) << ","
                   << formatDouble(std::abs(transform.sampleNormalErrors[sampleIndex])) << "\n";
        }
    }

    return stream.str();
}

std::string buildOriginTangentPointMetricsCsv(const std::vector<StitchStepRecord>& steps)
{
    std::vector<std::vector<double>> refFluctuationByStep;
    std::vector<std::vector<double>> targetFluctuationByStep;
    refFluctuationByStep.reserve(steps.size());
    targetFluctuationByStep.reserve(steps.size());

    std::size_t maxSampleCount = 0;
    for (const auto& step : steps) {
        refFluctuationByStep.push_back(buildFluctuationSeries(step.transform.sampleRefSecondaryCoordinates));
        targetFluctuationByStep.push_back(buildFluctuationSeries(step.transform.sampleTargetSecondaryCoordinates));
        const TransformResult& transform = step.transform;
        const std::size_t sampleCount = std::min({transform.samplePrimaryCoordinates.size(),
                                                  refFluctuationByStep.back().size(),
                                                  targetFluctuationByStep.back().size(),
                                                  transform.sampleTangentErrors.size(),
                                                  transform.sampleNormalErrors.size(),
                                                  transform.sampleInlierFlags.size()});
        maxSampleCount = std::max(maxSampleCount, sampleCount);
    }

    std::ostringstream stream;
    stream << "SampleIndex";
    for (const auto& step : steps) {
        stream << ",Step" << step.stepIndex << "_PrimaryCoord(px)"
               << ",Step" << step.stepIndex << "_RefFluctuation(px)"
               << ",Step" << step.stepIndex << "_TargetFluctuation(px)"
               << ",Step" << step.stepIndex << "_FluctuationDelta(px)"
               << ",Step" << step.stepIndex << "_TangentError(px)"
               << ",Step" << step.stepIndex << "_NormalError(px)"
               << ",Step" << step.stepIndex << "_IsInlier";
    }
    stream << "\n";

    for (std::size_t sampleIndex = 0; sampleIndex < maxSampleCount; ++sampleIndex) {
        stream << (sampleIndex + 1);
        for (std::size_t stepIndex = 0; stepIndex < steps.size(); ++stepIndex) {
            const TransformResult& transform = steps[stepIndex].transform;
            const std::size_t sampleCount = std::min({transform.samplePrimaryCoordinates.size(),
                                                      refFluctuationByStep[stepIndex].size(),
                                                      targetFluctuationByStep[stepIndex].size(),
                                                      transform.sampleTangentErrors.size(),
                                                      transform.sampleNormalErrors.size(),
                                                      transform.sampleInlierFlags.size()});
            if (sampleIndex < sampleCount) {
                stream << "," << formatDouble(transform.samplePrimaryCoordinates[sampleIndex])
                       << "," << formatDouble(refFluctuationByStep[stepIndex][sampleIndex])
                       << "," << formatDouble(targetFluctuationByStep[stepIndex][sampleIndex])
                       << "," << formatDouble(targetFluctuationByStep[stepIndex][sampleIndex] -
                                             refFluctuationByStep[stepIndex][sampleIndex])
                       << "," << formatDouble(transform.sampleTangentErrors[sampleIndex])
                       << "," << formatDouble(transform.sampleNormalErrors[sampleIndex])
                       << "," << transform.sampleInlierFlags[sampleIndex];
            } else {
                stream << ",,,,,,";
            }
        }
        stream << "\n";
    }

    return stream.str();
}

std::string buildAlignmentCandidateDiagnosticsCsv(const std::vector<StitchStepRecord>& steps)
{
    std::ostringstream stream;
    stream << "Step,ImageA,ImageB,Rank,Direction,Axis,dx(px),dy(px),Angle(deg),"
              "MatchScore,NormalMatchCost,TangentResidualMatchCost,TangentCorrelationMatchCost,DirectionPenaltyMatchCost,"
              "OverlapCount,OverlapCoverageRatio,InlierCount,InlierRatio,InlierCoverageRatio,"
              "NormalRMSEAll(px),NormalRMSEInlier(px),TangentRMSEAll(px),TangentRMSEInlier(px),"
              "TangentCorrAll,TangentCorrInlier\n";

    for (const auto& step : steps) {
        const auto& diagnostics = step.transform.candidateDiagnostics;
        for (std::size_t i = 0; i < diagnostics.size(); ++i) {
            const AlignmentCandidateDiagnostic& candidate = diagnostics[i];
            const AlignmentMetrics& metrics = candidate.metrics;
            stream << step.stepIndex << ","
                   << (step.referenceImageIndex + 1) << ","
                   << (step.targetImageIndex + 1) << ","
                   << (i + 1) << ","
                   << candidate.direction << ","
                   << (candidate.axis == AlignmentAxis::X ? "X" : "Y") << ","
                   << formatDouble(candidate.dx) << ","
                   << formatDouble(candidate.dy) << ","
                   << formatDouble(candidate.da) << ","
                   << formatDouble(candidate.score) << ","
                   << formatDouble(candidate.normalMatchCost) << ","
                   << formatDouble(candidate.tangentResidualMatchCost) << ","
                   << formatDouble(candidate.tangentCorrelationMatchCost) << ","
                   << formatDouble(candidate.directionPenaltyMatchCost) << ","
                   << metrics.overlapCount << ","
                   << formatDouble(metrics.overlapCoverageRatio) << ","
                   << metrics.inlierCount << ","
                   << formatDouble(metrics.inlierRatio) << ","
                   << formatDouble(metrics.inlierCoverageRatio) << ","
                   << formatDouble(metrics.normalAll.rmse) << ","
                   << formatDouble(metrics.normalInlier.rmse) << ","
                   << formatDouble(metrics.tangentAll.rmse) << ","
                   << formatDouble(metrics.tangentInlier.rmse) << ","
                   << formatDouble(metrics.tangentCorrAll) << ","
                   << formatDouble(metrics.tangentCorrInlier) << "\n";
        }
    }

    return stream.str();
}

std::string buildStitchingCsv(const std::vector<StitchStepRecord>& steps)
{
    {
        std::ostringstream stream;
        stream << "Step,ImageA,ImageB,SelectionMode,PriorClamped,ForcedPriorClamp,"
                  "dx(px),dy(px),Angle(deg),"
                  "MotionAxis,PrimarySpan(px),PrimaryShift(px),PerpShift(px),ActualOverlapRatio,"
                  "NominalPrimaryShift(px),NominalOverlapRatio,PriorPrimaryShift(px),PriorPerpShift(px),"
                  "TrajectoryPrimaryShift(px),TrajectoryPerpShift(px),PrimaryShiftMinusTrajectory(px),PerpShiftMinusTrajectory(px),"
                  "ImageCorrPrimaryShift(px),ImageCorrPerpShift(px),PrimaryShiftMinusImageCorr(px),PerpShiftMinusImageCorr(px),ImageCorrScore,"
                  "MatchScore,NormalMatchCost,TangentResidualMatchCost,TangentCorrelationMatchCost,DirectionPenaltyMatchCost,"
                  "OverlapCount,OverlapSpan(px),OverlapCoverageRatio,"
                  "InlierCount,InlierRatio,InlierSpan(px),InlierCoverageRatio,"
                  "NormalBiasAll(px),NormalRMSEAll(px),NormalMeanAbsAll(px),NormalMedianAbsAll(px),"
                  "NormalP95AbsAll(px),NormalMaxAbsAll(px),"
                  "NormalBiasInlier(px),NormalRMSEInlier(px),NormalMeanAbsInlier(px),NormalMedianAbsInlier(px),"
                  "NormalP95AbsInlier(px),NormalMaxAbsInlier(px),"
                  "TangentBiasAll(px),TangentRMSEAll(px),TangentMeanAbsAll(px),TangentMedianAbsAll(px),"
                  "TangentP95AbsAll(px),TangentMaxAbsAll(px),"
                  "TangentBiasInlier(px),TangentRMSEInlier(px),TangentMeanAbsInlier(px),TangentMedianAbsInlier(px),"
                  "TangentP95AbsInlier(px),TangentMaxAbsInlier(px),"
                  "TangentCorrAll,TangentCorrInlier\n";

        for (const auto& step : steps) {
            const TransformResult& result = step.transform;
            const AlignmentMetrics& metrics = result.metrics;
            const double primaryShiftPx = motionPrimaryShift(step, result.dx, result.dy);
            const double perpShiftPx = motionPerpendicularShift(step, result.dx, result.dy);
            const double actualOverlapRatio = overlapRatioFromPrimaryShift(step, primaryShiftPx);
            const double nominalPrimaryShiftPx =
                step.hasNominalPrior ? motionPrimaryShift(step, step.nominalPriorDx, step.nominalPriorDy)
                                     : std::numeric_limits<double>::quiet_NaN();
            const double nominalOverlapRatio =
                step.hasNominalPrior ? overlapRatioFromPrimaryShift(step, nominalPrimaryShiftPx)
                                     : std::numeric_limits<double>::quiet_NaN();
            const double priorPrimaryShiftPx =
                step.hasPairPrior ? motionPrimaryShift(step, step.pairPriorDx, step.pairPriorDy)
                                  : std::numeric_limits<double>::quiet_NaN();
            const double priorPerpShiftPx =
                step.hasPairPrior ? motionPerpendicularShift(step, step.pairPriorDx, step.pairPriorDy)
                                  : std::numeric_limits<double>::quiet_NaN();
            const double trajectoryPrimaryShiftPx =
                step.hasTrajectoryPrior ? motionPrimaryShift(step, step.trajectoryPriorDx, step.trajectoryPriorDy)
                                        : std::numeric_limits<double>::quiet_NaN();
            const double trajectoryPerpShiftPx =
                step.hasTrajectoryPrior ? motionPerpendicularShift(step, step.trajectoryPriorDx, step.trajectoryPriorDy)
                                        : std::numeric_limits<double>::quiet_NaN();
            const double imageCorrPrimaryShiftPx =
                step.hasImageCorrelationEstimate ? motionPrimaryShift(step, step.imageCorrelationDx, step.imageCorrelationDy)
                                                 : std::numeric_limits<double>::quiet_NaN();
            const double imageCorrPerpShiftPx =
                step.hasImageCorrelationEstimate ? motionPerpendicularShift(step, step.imageCorrelationDx, step.imageCorrelationDy)
                                                 : std::numeric_limits<double>::quiet_NaN();
            stream << step.stepIndex << ","
                   << (step.referenceImageIndex + 1) << ","
                   << (step.targetImageIndex + 1) << ","
                   << step.selectionMode << ","
                   << (isPriorClampedStep(step) ? 1 : 0) << ","
                   << (step.usedForcedTrajectoryPriorFallback ? 1 : 0) << ","
                   << formatDouble(result.dx) << ","
                   << formatDouble(result.dy) << ","
                   << formatDouble(result.da) << ","
                   << motionAxisLabel(step.motionAxis) << ","
                   << formatDouble(step.primaryImageSpanPx) << ","
                   << formatDouble(primaryShiftPx) << ","
                   << formatDouble(perpShiftPx) << ","
                   << formatDouble(actualOverlapRatio) << ","
                   << formatDouble(nominalPrimaryShiftPx) << ","
                   << formatDouble(nominalOverlapRatio) << ","
                   << formatDouble(priorPrimaryShiftPx) << ","
                   << formatDouble(priorPerpShiftPx) << ","
                   << formatDouble(trajectoryPrimaryShiftPx) << ","
                   << formatDouble(trajectoryPerpShiftPx) << ","
                   << formatDouble(primaryShiftPx - trajectoryPrimaryShiftPx) << ","
                   << formatDouble(perpShiftPx - trajectoryPerpShiftPx) << ","
                   << formatDouble(imageCorrPrimaryShiftPx) << ","
                   << formatDouble(imageCorrPerpShiftPx) << ","
                   << formatDouble(primaryShiftPx - imageCorrPrimaryShiftPx) << ","
                   << formatDouble(perpShiftPx - imageCorrPerpShiftPx) << ","
                   << formatDouble(step.hasImageCorrelationEstimate ? step.imageCorrelationScore
                                                                    : std::numeric_limits<double>::quiet_NaN()) << ","
                   << formatDouble(result.score) << ","
                   << formatDouble(result.normalMatchCost) << ","
                   << formatDouble(result.tangentResidualMatchCost) << ","
                   << formatDouble(result.tangentCorrelationMatchCost) << ","
                   << formatDouble(result.directionPenaltyMatchCost) << ","
                   << metrics.overlapCount << ","
                   << formatDouble(metrics.overlapSpan) << ","
                   << formatDouble(metrics.overlapCoverageRatio) << ","
                   << metrics.inlierCount << ","
                   << formatDouble(metrics.inlierRatio) << ","
                   << formatDouble(metrics.inlierSpan) << ","
                   << formatDouble(metrics.inlierCoverageRatio) << ",";
            appendResidualStatisticsCsv(stream, metrics.normalAll);
            stream << ",";
            appendResidualStatisticsCsv(stream, metrics.normalInlier);
            stream << ",";
            appendResidualStatisticsCsv(stream, metrics.tangentAll);
            stream << ",";
            appendResidualStatisticsCsv(stream, metrics.tangentInlier);
            stream << ","
                   << formatDouble(metrics.tangentCorrAll) << ","
                   << formatDouble(metrics.tangentCorrInlier) << "\n";
        }

        return stream.str();
    }
    std::ostringstream stream;
    stream << "Step,ImageA,ImageB,dx(px),dy(px),Angle(deg),"
              "MatchScore,NormalMatchCost,TangentResidualMatchCost,TangentCorrelationMatchCost,DirectionPenaltyMatchCost,"
              "OverlapCount,OverlapSpan(px),OverlapCoverageRatio,"
              "InlierCount,InlierRatio,InlierSpan(px),InlierCoverageRatio,"
              "NormalBiasAll(px),NormalRMSEAll(px),NormalMeanAbsAll(px),NormalMedianAbsAll(px),"
              "NormalP95AbsAll(px),NormalMaxAbsAll(px),"
              "NormalBiasInlier(px),NormalRMSEInlier(px),NormalMeanAbsInlier(px),NormalMedianAbsInlier(px),"
              "NormalP95AbsInlier(px),NormalMaxAbsInlier(px),"
              "TangentBiasAll(px),TangentRMSEAll(px),TangentMeanAbsAll(px),TangentMedianAbsAll(px),"
              "TangentP95AbsAll(px),TangentMaxAbsAll(px),"
              "TangentBiasInlier(px),TangentRMSEInlier(px),TangentMeanAbsInlier(px),TangentMedianAbsInlier(px),"
              "TangentP95AbsInlier(px),TangentMaxAbsInlier(px),"
              "TangentCorrAll,TangentCorrInlier\n";

    for (const auto& step : steps) {
        const TransformResult& result = step.transform;
        const AlignmentMetrics& metrics = result.metrics;
        stream << step.stepIndex << ","
               << (step.referenceImageIndex + 1) << ","
               << (step.targetImageIndex + 1) << ","
               << formatDouble(result.dx) << ","
               << formatDouble(result.dy) << ","
               << formatDouble(result.da) << ","
               << formatDouble(result.score) << ","
               << formatDouble(result.normalMatchCost) << ","
               << formatDouble(result.tangentResidualMatchCost) << ","
               << formatDouble(result.tangentCorrelationMatchCost) << ","
               << formatDouble(result.directionPenaltyMatchCost) << ","
               << metrics.overlapCount << ","
               << formatDouble(metrics.overlapSpan) << ","
               << formatDouble(metrics.overlapCoverageRatio) << ","
               << metrics.inlierCount << ","
               << formatDouble(metrics.inlierRatio) << ","
               << formatDouble(metrics.inlierSpan) << ","
               << formatDouble(metrics.inlierCoverageRatio) << ",";
        appendResidualStatisticsCsv(stream, metrics.normalAll);
        stream << ",";
        appendResidualStatisticsCsv(stream, metrics.normalInlier);
        stream << ",";
        appendResidualStatisticsCsv(stream, metrics.tangentAll);
        stream << ",";
        appendResidualStatisticsCsv(stream, metrics.tangentInlier);
        stream << ","
               << formatDouble(metrics.tangentCorrAll) << ","
               << formatDouble(metrics.tangentCorrInlier) << "\n";
    }

    return stream.str();
}

} // namespace stitch
