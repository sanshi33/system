#include "EdgeProcessing.h"
#include "GeometryUtils.h"

#include "../core/SubpixelEdgeDetector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace stitch {
using namespace cv;
using namespace std;

namespace {

using SubpixelEdge::SubpixelPoint;

// 将分位数参数限制在 [0, 1]，同时兼容外部传入的边界值。
double clampUnitQuantile(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

// 使用 nth_element 计算分位数，避免为简单阈值计算做完整排序。
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

// 先剔除坐标、置信度或梯度中出现 NaN / Inf 的异常点。
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

// 先按置信度和梯度分位数做一次轻量预筛，过滤明显弱点。
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

// 在局部窗口内排除当前点后拟合一条直线，用于估计该点的局部参考趋势。
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

// 对按 X 排序后的亚像素点做局部线性 Hampel 判异，只剔除毛刺点，不平滑保留点坐标。
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

// 转成当前拼接流程统一使用的 Point2d 表示。
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

// 汇总亚像素点前置滤波流程：有限值检查 -> 质量预筛 -> Hampel 离群点剔除。
std::vector<cv::Point2d> filterSubpixelEdges(const std::vector<SubpixelPoint>& detectedPoints,
                                             const EdgeDetectConfig& cfg)
{
    std::vector<SubpixelPoint> filtered = collectFinitePoints(detectedPoints);
    if (!cfg.enablePointFiltering || filtered.empty()) {
        return toCvPoints(filtered);
    }

    filtered = applyQualityPrefilter(std::move(filtered), cfg);
    filtered = applyLocalLinearHampel(std::move(filtered), cfg);
    return toCvPoints(filtered);
}

} // namespace

EdgeVariants buildEdgeVariants(SubpixelEdgeDetector &det,
                               const cv::Mat &img,
                               const EdgeDetectConfig &cfg)
{
    det.setImage(img);
    det.detectCannyEdges(cfg.cannyLow, cfg.cannyHigh);
    det.refineEdgesSubpixel(cfg.subpixWindow, cfg.subpixSigma);

    EdgeVariants ev;
    // 这里是“边缘检测之后、拼接之前”的唯一入口，适合放点级清洗。
    std::vector<SubpixelPoint> filtered = collectFinitePoints(det.getSubpixelPoints());
    if (cfg.enablePointFiltering && !filtered.empty()) {
        filtered = applyQualityPrefilter(std::move(filtered), cfg);
        filtered = applyLocalLinearHampel(std::move(filtered), cfg);
    }
    fillPointQuality(filtered, ev);

    ev.x_sorted = ev.raw;
    ev.y_sorted = ev.raw;
    ev.negX_sorted = ev.raw;
    ev.negY_sorted = ev.raw;

    sortContourByX(ev.x_sorted);
    sortContourByY(ev.y_sorted);

    for (auto &p : ev.negX_sorted)
        p.x = -p.x;
    sortContourByX(ev.negX_sorted);

    for (auto &p : ev.negY_sorted)
        p.y = -p.y;
    sortContourByY(ev.negY_sorted);

    return ev;
}

} // namespace stitch
