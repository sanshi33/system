#include "EdgeProcessing.h"
#include "GeometryUtils.h"

#include "../core/SubpixelEdgeDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace stitch {
using namespace cv;
using namespace std;

namespace {

using SubpixelEdge::SubpixelPoint;

cv::Mat toGray8U(const cv::Mat& image)
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
    if (gray.type() == CV_8UC1) {
        return gray;
    }
    cv::Mat converted;
    gray.convertTo(converted, CV_8U);
    return converted;
}

bool looksLikeCadRenderedSection(const cv::Mat& gray)
{
    if (gray.empty() || gray.type() != CV_8UC1) {
        return false;
    }

    const int step = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(gray.total()) / 120000.0)));
    std::array<int, 16> coarseBins{};
    std::size_t sampleCount = 0;
    std::size_t backgroundCount = 0;
    std::size_t darkCount = 0;
    std::size_t midCount = 0;
    for (int y = 0; y < gray.rows; y += step) {
        const uchar* row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; x += step) {
            const int value = row[x];
            ++sampleCount;
            ++coarseBins[std::clamp(value / 16, 0, 15)];
            if (value >= 245) {
                ++backgroundCount;
            } else if (value <= 70) {
                ++darkCount;
            } else if (value >= 90 && value <= 235) {
                ++midCount;
            }
        }
    }
    if (sampleCount == 0) {
        return false;
    }

    const double backgroundRatio = static_cast<double>(backgroundCount) / static_cast<double>(sampleCount);
    const double darkRatio = static_cast<double>(darkCount) / static_cast<double>(sampleCount);
    const double midRatio = static_cast<double>(midCount) / static_cast<double>(sampleCount);
    const int occupiedBins =
        static_cast<int>(std::count_if(coarseBins.begin(), coarseBins.end(), [](int count) {
            return count > 0;
        }));
    (void)occupiedBins;

    const double foregroundRatio = 1.0 - backgroundRatio;
    return backgroundRatio >= 0.45 &&
           foregroundRatio >= 0.025 &&
           (midRatio >= 0.015 || darkRatio >= 0.006);
}

cv::Mat keepMeaningfulForegroundComponents(const cv::Mat& mask)
{
    if (mask.empty()) {
        return {};
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (count <= 1) {
        return mask.clone();
    }

    const int minArea = std::max(12, static_cast<int>(std::round(static_cast<double>(mask.total()) * 0.00004)));
    cv::Mat filtered(mask.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minArea) {
            continue;
        }
        filtered.setTo(255, labels == label);
    }
    return filtered;
}

cv::Mat buildCadRenderedBinaryMask(const cv::Mat& gray)
{
    if (!looksLikeCadRenderedSection(gray)) {
        return {};
    }

    cv::Mat filledRegion;
    cv::inRange(gray, cv::Scalar(72), cv::Scalar(244), filledRegion);
    const double filledRatio =
        static_cast<double>(cv::countNonZero(filledRegion)) / std::max(1.0, static_cast<double>(filledRegion.total()));

    cv::Mat mask;
    if (filledRatio >= 0.012) {
        mask = filledRegion;
    } else {
        cv::threshold(gray, mask, 244, 255, cv::THRESH_BINARY_INV);
    }

    const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    const cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeKernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, openKernel);
    mask = keepMeaningfulForegroundComponents(mask);

    if (cv::countNonZero(mask) < std::max(20, static_cast<int>(mask.total() / 1000))) {
        return {};
    }
    return mask;
}

void zhangSuenThinningIteration(cv::Mat& image01, int iteration)
{
    cv::Mat marker = cv::Mat::zeros(image01.size(), CV_8UC1);
    for (int y = 1; y < image01.rows - 1; ++y) {
        for (int x = 1; x < image01.cols - 1; ++x) {
            const uchar p1 = image01.at<uchar>(y, x);
            if (p1 == 0) {
                continue;
            }
            const uchar p2 = image01.at<uchar>(y - 1, x);
            const uchar p3 = image01.at<uchar>(y - 1, x + 1);
            const uchar p4 = image01.at<uchar>(y, x + 1);
            const uchar p5 = image01.at<uchar>(y + 1, x + 1);
            const uchar p6 = image01.at<uchar>(y + 1, x);
            const uchar p7 = image01.at<uchar>(y + 1, x - 1);
            const uchar p8 = image01.at<uchar>(y, x - 1);
            const uchar p9 = image01.at<uchar>(y - 1, x - 1);

            const int transitions =
                (p2 == 0 && p3 != 0) + (p3 == 0 && p4 != 0) +
                (p4 == 0 && p5 != 0) + (p5 == 0 && p6 != 0) +
                (p6 == 0 && p7 != 0) + (p7 == 0 && p8 != 0) +
                (p8 == 0 && p9 != 0) + (p9 == 0 && p2 != 0);
            const int neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            if (neighbors < 2 || neighbors > 6 || transitions != 1) {
                continue;
            }

            const bool remove =
                iteration == 0
                    ? (p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                    : (p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0);
            if (remove) {
                marker.at<uchar>(y, x) = 1;
            }
        }
    }
    image01 &= ~marker;
}

void thinBinaryInPlace(cv::Mat& binary)
{
    if (binary.empty()) {
        return;
    }
    cv::threshold(binary, binary, 0, 1, cv::THRESH_BINARY);
    cv::Mat previous = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Mat diff;
    do {
        zhangSuenThinningIteration(binary, 0);
        zhangSuenThinningIteration(binary, 1);
        cv::absdiff(binary, previous, diff);
        binary.copyTo(previous);
    } while (cv::countNonZero(diff) > 0);
    binary *= 255;
}

cv::Mat buildCadRenderedLineSkeleton(const cv::Mat& gray)
{
    if (!looksLikeCadRenderedSection(gray)) {
        return {};
    }

    cv::Mat darkLines;
    cv::inRange(gray, cv::Scalar(0), cv::Scalar(86), darkLines);
    const int minPixels = std::max(20, static_cast<int>(darkLines.total() / 2500));
    if (cv::countNonZero(darkLines) < minPixels) {
        return {};
    }

    const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(darkLines, darkLines, cv::MORPH_CLOSE, closeKernel);
    darkLines = keepMeaningfulForegroundComponents(darkLines);
    thinBinaryInPlace(darkLines);
    if (cv::countNonZero(darkLines) < minPixels) {
        return {};
    }
    return darkLines;
}

std::vector<SubpixelPoint> extractBinaryContourPoints(const cv::Mat& mask)
{
    if (mask.empty()) {
        return {};
    }

    cv::Mat binary;
    if (mask.type() == CV_8UC1) {
        cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    } else {
        cv::Mat gray = toGray8U(mask);
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return {};
    }

    const double imageScale = static_cast<double>(mask.cols + mask.rows);
    const double minPerimeter = std::max(24.0, imageScale * 0.015);
    const int minPoints = 16;
    std::vector<const std::vector<cv::Point>*> keptContours;
    std::size_t totalCandidatePoints = 0;
    for (const auto& contour : contours) {
        if (contour.size() < static_cast<std::size_t>(minPoints)) {
            continue;
        }
        if (cv::arcLength(contour, true) < minPerimeter) {
            continue;
        }
        keptContours.push_back(&contour);
        totalCandidatePoints += contour.size();
    }

    if (totalCandidatePoints == 0) {
        return {};
    }

    const std::size_t maxContourPoints = 12000;
    const std::size_t stride = std::max<std::size_t>(
        1, (totalCandidatePoints + maxContourPoints - 1) / maxContourPoints);
    std::vector<SubpixelPoint> points;
    points.reserve(std::min(totalCandidatePoints, maxContourPoints));

    std::size_t counter = 0;
    for (const auto* contour : keptContours) {
        for (const auto& point : *contour) {
            if ((counter % stride) == 0) {
                points.emplace_back(static_cast<double>(point.x),
                                    static_cast<double>(point.y),
                                    1.0,
                                    255.0);
            }
            ++counter;
        }
    }

    return points;
}

std::vector<SubpixelPoint> extractBinaryPixelPoints(const cv::Mat& binary)
{
    if (binary.empty()) {
        return {};
    }

    std::vector<cv::Point> nonzero;
    cv::findNonZero(binary, nonzero);
    if (nonzero.empty()) {
        return {};
    }

    auto estimateTangent = [&binary](const cv::Point& point) {
        double xx = 0.0;
        double xy = 0.0;
        double yy = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = point.x + dx;
                const int ny = point.y + dy;
                if (nx < 0 || nx >= binary.cols || ny < 0 || ny >= binary.rows) {
                    continue;
                }
                if (binary.at<uchar>(ny, nx) == 0) {
                    continue;
                }
                xx += static_cast<double>(dx * dx);
                xy += static_cast<double>(dx * dy);
                yy += static_cast<double>(dy * dy);
            }
        }

        const double trace = xx + yy;
        if (trace <= 1e-9) {
            return cv::Point2d(1.0, 0.0);
        }

        const double delta = std::sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy);
        const double lambda = 0.5 * (trace + delta);
        cv::Point2d tangent(xy, lambda - xx);
        if (std::abs(tangent.x) + std::abs(tangent.y) < 1e-9) {
            tangent = xx >= yy ? cv::Point2d(1.0, 0.0) : cv::Point2d(0.0, 1.0);
        }
        const double length = std::hypot(tangent.x, tangent.y);
        if (length <= 1e-9) {
            return cv::Point2d(1.0, 0.0);
        }
        return cv::Point2d(tangent.x / length, tangent.y / length);
    };

    const std::size_t maxPoints = 12000;
    const std::size_t minSkeletonPoints = 1000;
    if (nonzero.size() < minSkeletonPoints) {
        std::vector<SubpixelPoint> points;
        points.reserve(minSkeletonPoints);
        for (const cv::Point& point : nonzero) {
            points.emplace_back(static_cast<double>(point.x),
                                static_cast<double>(point.y),
                                1.0,
                                255.0);
        }

        std::size_t needed = minSkeletonPoints - points.size();
        const std::array<double, 8> offsets = {-0.33, 0.33, -0.45, 0.45, -0.20, 0.20, -0.10, 0.10};
        for (double offset : offsets) {
            for (const cv::Point& point : nonzero) {
                if (needed == 0) {
                    return points;
                }
                const cv::Point2d tangent = estimateTangent(point);
                points.emplace_back(static_cast<double>(point.x) + tangent.x * offset,
                                    static_cast<double>(point.y) + tangent.y * offset,
                                    1.0,
                                    255.0);
                --needed;
            }
        }

        return points;
    }

    const std::size_t stride =
        std::max<std::size_t>(1, (nonzero.size() + maxPoints - 1) / maxPoints);
    std::vector<SubpixelPoint> points;
    points.reserve(std::min<std::size_t>(nonzero.size(), maxPoints));
    for (std::size_t index = 0; index < nonzero.size(); index += stride) {
        const cv::Point& point = nonzero[index];
        points.emplace_back(static_cast<double>(point.x),
                            static_cast<double>(point.y),
                            1.0,
                            255.0);
    }
    return points;
}

double clampUnitQuantile(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

double computeQuantile(std::vector<double> values, double q)
{
    if (values.empty()) {
        return 0.0;
    }

    q = clampUnitQuantile(q);
    const std::size_t index =
        static_cast<std::size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

std::vector<SubpixelPoint> collectFinitePoints(const std::vector<SubpixelPoint>& points)
{
    std::vector<SubpixelPoint> filtered;
    filtered.reserve(points.size());

    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.confidence) || !std::isfinite(point.gradient)) {
            continue;
        }
        filtered.push_back(point);
    }

    return filtered;
}

std::vector<SubpixelPoint> detectSubpixelPoints(SubpixelEdgeDetector& det,
                                                const cv::Mat& img,
                                                const EdgeDetectConfig& cfg)
{
    det.setImage(img);
    det.detectCannyEdges(cfg.cannyLow, cfg.cannyHigh);
    det.refineEdgesSubpixel(cfg.subpixWindow, cfg.subpixSigma);
    return collectFinitePoints(det.getSubpixelPoints());
}

std::vector<SubpixelPoint> applyQualityPrefilter(std::vector<SubpixelPoint> points,
                                                 const EdgeDetectConfig& cfg)
{
    if (points.empty()) {
        return points;
    }

    const double confidenceQuantile = clampUnitQuantile(cfg.filterConfidenceQuantile);
    const double gradientQuantile = clampUnitQuantile(cfg.filterGradientQuantile);
    if (confidenceQuantile <= 0.0 && gradientQuantile <= 0.0) {
        return points;
    }

    std::vector<double> confidences;
    std::vector<double> gradients;
    confidences.reserve(points.size());
    gradients.reserve(points.size());

    for (const auto& point : points) {
        confidences.push_back(point.confidence);
        gradients.push_back(point.gradient);
    }

    const double confidenceThreshold =
        confidenceQuantile > 0.0 ? computeQuantile(confidences, confidenceQuantile)
                                 : -std::numeric_limits<double>::infinity();
    const double gradientThreshold =
        gradientQuantile > 0.0 ? computeQuantile(gradients, gradientQuantile)
                               : -std::numeric_limits<double>::infinity();

    std::vector<SubpixelPoint> filtered;
    filtered.reserve(points.size());
    for (const auto& point : points) {
        if (point.confidence < confidenceThreshold || point.gradient < gradientThreshold) {
            continue;
        }
        filtered.push_back(point);
    }

    return filtered;
}

bool fitLocalLineExcludingIndex(const std::vector<SubpixelPoint>& points,
                                int start,
                                int end,
                                int excludedIndex,
                                double& slope,
                                double& intercept)
{
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXX = 0.0;
    double sumXY = 0.0;
    int count = 0;

    for (int index = start; index <= end; ++index) {
        if (index == excludedIndex) {
            continue;
        }

        const double x = points[static_cast<std::size_t>(index)].x;
        const double y = points[static_cast<std::size_t>(index)].y;
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumXY += x * y;
        ++count;
    }

    if (count < 2) {
        return false;
    }

    const double denominator = static_cast<double>(count) * sumXX - sumX * sumX;
    if (std::abs(denominator) < 1e-9) {
        return false;
    }

    slope = (static_cast<double>(count) * sumXY - sumX * sumY) / denominator;
    intercept = (sumY - slope * sumX) / static_cast<double>(count);
    return std::isfinite(slope) && std::isfinite(intercept);
}

std::vector<SubpixelPoint> applyLocalLinearHampel(std::vector<SubpixelPoint> points,
                                                  const EdgeDetectConfig& cfg)
{
    if (points.empty()) {
        return points;
    }

    std::stable_sort(points.begin(), points.end(),
                     [](const SubpixelPoint& lhs, const SubpixelPoint& rhs) {
                         if (lhs.x == rhs.x) {
                             return lhs.y < rhs.y;
                         }
                         return lhs.x < rhs.x;
                     });

    const int radius = std::max(1, cfg.filterLocalLinearWindowRadius);
    const double sigma = std::max(0.5, cfg.filterHampelSigma);
    const double minScale = std::max(1e-6, cfg.filterHampelMinScale);
    if (points.size() < static_cast<std::size_t>(2 * radius + 3)) {
        return points;
    }

    std::vector<char> keep(points.size(), 1);
    std::vector<double> residuals;
    std::vector<double> absDeviations;

    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const int start = std::max(0, i - radius);
        const int end = std::min(static_cast<int>(points.size()) - 1, i + radius);
        if (end - start < 4) {
            continue;
        }

        double slope = 0.0;
        double intercept = 0.0;
        if (!fitLocalLineExcludingIndex(points, start, end, i, slope, intercept)) {
            continue;
        }

        residuals.clear();
        residuals.reserve(static_cast<std::size_t>(end - start));
        for (int index = start; index <= end; ++index) {
            if (index == i) {
                continue;
            }
            const auto& point = points[static_cast<std::size_t>(index)];
            residuals.push_back(point.y - (slope * point.x + intercept));
        }

        if (residuals.size() < 4) {
            continue;
        }

        const double medianResidual = computeQuantile(residuals, 0.5);
        absDeviations.clear();
        absDeviations.reserve(residuals.size());
        for (double residual : residuals) {
            absDeviations.push_back(std::abs(residual - medianResidual));
        }

        const double mad = computeQuantile(absDeviations, 0.5);
        const double scale = std::max(minScale, 1.4826 * mad);
        const auto& centerPoint = points[static_cast<std::size_t>(i)];
        const double centerResidual = centerPoint.y - (slope * centerPoint.x + intercept);

        if (std::abs(centerResidual - medianResidual) > sigma * scale) {
            keep[static_cast<std::size_t>(i)] = 0;
        }
    }

    std::vector<SubpixelPoint> filtered;
    filtered.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(points[i]);
        }
    }

    return filtered;
}

std::vector<cv::Point2d> toCvPoints(const std::vector<SubpixelPoint>& points)
{
    std::vector<cv::Point2d> edges;
    edges.reserve(points.size());
    for (const auto& point : points) {
        edges.emplace_back(point.x, point.y);
    }
    return edges;
}

void fillPointQuality(const std::vector<SubpixelPoint>& points, EdgeVariants& ev)
{
    ev.raw = toCvPoints(points);
    ev.rawConfidences.clear();
    ev.rawGradients.clear();
    ev.rawQualityWeights.clear();
    ev.rawSyntheticFlags.clear();
    ev.rawConfidences.reserve(points.size());
    ev.rawGradients.reserve(points.size());
    ev.rawQualityWeights.reserve(points.size());
    ev.rawSyntheticFlags.reserve(points.size());

    std::vector<double> confidences;
    std::vector<double> gradients;
    confidences.reserve(points.size());
    gradients.reserve(points.size());
    for (const auto& point : points) {
        const double confidence = std::max(0.0, point.confidence);
        const double gradient = std::max(0.0, point.gradient);
        ev.rawConfidences.push_back(confidence);
        ev.rawGradients.push_back(gradient);
        ev.rawSyntheticFlags.push_back(0);
        confidences.push_back(confidence);
        gradients.push_back(gradient);
    }

    const double confidenceMedian = std::max(1e-9, computeQuantile(confidences, 0.5));
    const double gradientMedian = std::max(1e-9, computeQuantile(gradients, 0.5));
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double confidenceRatio = std::clamp(ev.rawConfidences[i] / confidenceMedian, 0.05, 4.0);
        const double gradientRatio = std::clamp(ev.rawGradients[i] / gradientMedian, 0.05, 4.0);
        const double weight = std::sqrt(confidenceRatio * gradientRatio);
        ev.rawQualityWeights.push_back(std::clamp(weight, 0.05, 4.0));
    }
}

void fillOrderedViews(const std::vector<cv::Point2d>& rawPoints,
                      std::vector<cv::Point2d>& xSorted,
                      std::vector<cv::Point2d>& ySorted,
                      std::vector<cv::Point2d>& negXSorted,
                      std::vector<cv::Point2d>& negYSorted)
{
    xSorted = rawPoints;
    ySorted = rawPoints;
    negXSorted = rawPoints;
    negYSorted = rawPoints;

    sortContourByX(xSorted);
    sortContourByY(ySorted);

    for (auto& point : negXSorted) {
        point.x = -point.x;
    }
    sortContourByX(negXSorted);

    for (auto& point : negYSorted) {
        point.y = -point.y;
    }
    sortContourByY(negYSorted);
}

} // namespace

EdgeVariants buildEdgeVariants(SubpixelEdgeDetector& det,
                               const cv::Mat& img,
                               const EdgeDetectConfig& cfg)
{
    const auto stageBegin = std::chrono::steady_clock::now();
    std::vector<SubpixelPoint> finitePoints = detectSubpixelPoints(det, img, cfg);
    const auto subpixelReady = std::chrono::steady_clock::now();
    std::string preprocessingMode = "Subpixel edge detection";
    const auto pointModeReady = subpixelReady;

    std::vector<SubpixelPoint> filteredPoints = finitePoints;
    if (cfg.enablePointFiltering && !filteredPoints.empty()) {
        filteredPoints = applyQualityPrefilter(std::move(filteredPoints), cfg);
        filteredPoints = applyLocalLinearHampel(std::move(filteredPoints), cfg);
    }
    const auto filteringReady = std::chrono::steady_clock::now();

    EdgeVariants ev;
    fillPointQuality(filteredPoints, ev);
    ev.preprocessingMode = preprocessingMode;
    ev.unfiltered_raw = toCvPoints(finitePoints);

    fillOrderedViews(ev.raw, ev.x_sorted, ev.y_sorted, ev.negX_sorted, ev.negY_sorted);
    fillOrderedViews(ev.unfiltered_raw,
                     ev.unfiltered_x_sorted,
                     ev.unfiltered_y_sorted,
                     ev.unfiltered_negX_sorted,
                     ev.unfiltered_negY_sorted);
    const auto orderedReady = std::chrono::steady_clock::now();

    const auto seconds = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    ev.preprocessingMode +=
        " | timing(subpixel=" + std::to_string(seconds(stageBegin, subpixelReady)) +
        "s, mode-switch=" + std::to_string(seconds(subpixelReady, pointModeReady)) +
        "s, filter=" + std::to_string(seconds(pointModeReady, filteringReady)) +
        "s, order=" + std::to_string(seconds(filteringReady, orderedReady)) +
        "s, raw=" + std::to_string(finitePoints.size()) +
        ", kept=" + std::to_string(filteredPoints.size()) + ")";

    return ev;
}

} // namespace stitch
