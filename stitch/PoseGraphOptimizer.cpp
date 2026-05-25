#include "PoseGraphOptimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace stitch {

namespace {

// ---------------------------------------------------------------------------
// 内部：2D 刚体变换 (dx, dy, da_deg) 的合成与求逆
// ---------------------------------------------------------------------------
struct Rigid2D {
    double dx{0.0};
    double dy{0.0};
    double daDeg{0.0}; // 度
};

Rigid2D compose(const Rigid2D& a, const Rigid2D& b)
{
    const double rad = a.daDeg * CV_PI / 180.0;
    const double ct = std::cos(rad);
    const double st = std::sin(rad);
    Rigid2D c;
    c.dx = a.dx + ct * b.dx - st * b.dy;
    c.dy = a.dy + st * b.dx + ct * b.dy;
    c.daDeg = a.daDeg + b.daDeg;
    return c;
}

Rigid2D inverse(const Rigid2D& t)
{
    // 2D 刚体变换的逆：inv(R, t) = (R^T, -R^T * t)
    const double rad = t.daDeg * CV_PI / 180.0;
    const double ct = std::cos(rad);
    const double st = std::sin(rad);
    Rigid2D inv;
    inv.dx = -(ct * t.dx + st * t.dy);
    inv.dy = -(-st * t.dx + ct * t.dy);
    inv.daDeg = -t.daDeg;
    return inv;
}

Rigid2D matToRigid(const cv::Mat& M)
{
    Rigid2D t;
    t.dx = M.at<double>(0, 2);
    t.dy = M.at<double>(1, 2);
    t.daDeg = std::atan2(M.at<double>(1, 0), M.at<double>(0, 0)) * 180.0 / CV_PI;
    return t;
}

cv::Mat rigidToMat(const Rigid2D& t)
{
    const double rad = t.daDeg * CV_PI / 180.0;
    const double ct = std::cos(rad);
    const double st = std::sin(rad);
    cv::Mat M = cv::Mat::eye(3, 3, CV_64F);
    M.at<double>(0, 0) = ct;
    M.at<double>(0, 1) = -st;
    M.at<double>(0, 2) = t.dx;
    M.at<double>(1, 0) = st;
    M.at<double>(1, 1) = ct;
    M.at<double>(1, 2) = t.dy;
    return M;
}

double weightedMedian(std::vector<double> values, std::vector<double> weights)
{
    if (values.empty()) return 0.0;
    // 按值排序，同时重排权重
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        pairs.emplace_back(values[i], weights[i]);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    double totalWeight = 0.0;
    for (const auto& p : pairs) totalWeight += p.second;
    if (totalWeight < 1e-12) return pairs[pairs.size() / 2].first;

    double cum = 0.0;
    for (const auto& p : pairs) {
        cum += p.second;
        if (cum >= totalWeight * 0.5) return p.first;
    }
    return pairs.back().first;
}

} // namespace

// ---------------------------------------------------------------------------
// 主优化函数
// ---------------------------------------------------------------------------
std::vector<cv::Mat> optimizePoseGraph(const std::vector<cv::Mat>& initialGlobalTransforms,
                                       const std::vector<PoseGraphEdge>& edges,
                                       const double regularizationWeight,
                                       const int maxIterations)
{
    const std::size_t N = initialGlobalTransforms.size();
    if (N <= 1 || edges.empty()) {
        return initialGlobalTransforms;
    }

    // 第一幅图像也提取实际矩阵值（含画布放置偏移），后续约束才能一致
    std::vector<Rigid2D> global(N);
    for (std::size_t i = 0; i < N; ++i) {
        global[i] = matToRigid(initialGlobalTransforms[i]);
    }

    // 构建邻接表：每个节点连接哪些边
    struct EdgeRef {
        std::size_t edgeIdx;
        bool isFrom; // true = 此节点是 from，false = 此节点是 to
    };
    std::vector<std::vector<EdgeRef>> adjacency(N);
    for (std::size_t e = 0; e < edges.size(); ++e) {
        const auto& edge = edges[e];
        if (edge.fromIndex < N && edge.toIndex < N && edge.fromIndex != edge.toIndex) {
            adjacency[edge.fromIndex].push_back({e, true});
            adjacency[edge.toIndex].push_back({e, false});
        }
    }

    const auto computeCost = [&]() -> double {
        double total = 0.0;
        for (const auto& edge : edges) {
            if (edge.fromIndex >= N || edge.toIndex >= N) continue;
            const Rigid2D edgeT{edge.dx, edge.dy, edge.da};
            const Rigid2D predicted = compose(global[edge.fromIndex], edgeT);
            const double ex = predicted.dx - global[edge.toIndex].dx;
            const double ey = predicted.dy - global[edge.toIndex].dy;
            const double ea = predicted.daDeg - global[edge.toIndex].daDeg;
            total += edge.weight * (ex * ex + ey * ey + 0.01 * ea * ea);
        }
        return total;
    };

    // 迭代加权松弛
    for (int iter = 0; iter < maxIterations; ++iter) {
        double maxChange = 0.0;

        for (std::size_t i = 1; i < N; ++i) { // 跳过第一幅（固定）
            std::vector<double> dxSuggestions, dySuggestions, daSuggestions;
            std::vector<double> suggestionWeights;

            // 从每条邻边收集"建议"
            for (const auto& ref : adjacency[i]) {
                const auto& edge = edges[ref.edgeIdx];
                const std::size_t otherIdx = ref.isFrom ? edge.toIndex : edge.fromIndex;
                if (otherIdx >= N) continue;

                const Rigid2D edgeT{edge.dx, edge.dy, edge.da};
                Rigid2D suggestion;
                if (ref.isFrom) {
                    // i 是 from: global[to] ≈ global[i] ∘ edgeT
                    // → global[i] ≈ global[to] ∘ inv(edgeT)
                    suggestion = compose(global[otherIdx], inverse(edgeT));
                } else {
                    // i 是 to: global[to] ≈ global[from] ∘ edgeT
                    // → global[i] ≈ global[from] ∘ edgeT
                    suggestion = compose(global[otherIdx], edgeT);
                }
                dxSuggestions.push_back(suggestion.dx);
                dySuggestions.push_back(suggestion.dy);
                daSuggestions.push_back(suggestion.daDeg);
                suggestionWeights.push_back(edge.weight);
            }

            // 正则化建议：保持接近初始值
            const Rigid2D& initT = global[i]; // 这里 initT 实际是上次迭代的值
            // 改为使用真正的初始值
            const Rigid2D trueInit = matToRigid(initialGlobalTransforms[i]);
            dxSuggestions.push_back(trueInit.dx);
            dySuggestions.push_back(trueInit.dy);
            daSuggestions.push_back(trueInit.daDeg);
            suggestionWeights.push_back(regularizationWeight);

            if (suggestionWeights.empty()) continue;

            // 加权平均更新
            const double newDx = weightedMedian(dxSuggestions, suggestionWeights);
            const double newDy = weightedMedian(dySuggestions, suggestionWeights);
            const double newDa = weightedMedian(daSuggestions, suggestionWeights);

            const double change = std::abs(newDx - global[i].dx) +
                                  std::abs(newDy - global[i].dy) +
                                  std::abs(newDa - global[i].daDeg);
            maxChange = std::max(maxChange, change);

            global[i].dx = newDx;
            global[i].dy = newDy;
            global[i].daDeg = newDa;
        }

        if (maxChange < 0.001) break; // 收敛
    }

    const double finalCost = computeCost();

    // 转换回矩阵
    std::vector<cv::Mat> result;
    result.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        result.push_back(rigidToMat(global[i]));
    }

    return result;
}

// ---------------------------------------------------------------------------
// 辅助：构造邻边和跳边约束
// ---------------------------------------------------------------------------
std::vector<PoseGraphEdge> buildAdjacentEdges(const std::vector<StitchStepRecord>& steps)
{
    std::vector<PoseGraphEdge> edges;
    edges.reserve(steps.size());

    for (const auto& step : steps) {
        PoseGraphEdge edge;
        edge.fromIndex = step.referenceImageIndex;
        edge.toIndex = step.targetImageIndex;
        edge.dx = step.transform.dx;
        edge.dy = step.transform.dy;
        edge.da = step.transform.da;

        // 权重基于内点 RMSE（越低越可信）
        const double rmse = step.transform.metrics.normalInlier.valid()
                                ? step.transform.metrics.normalInlier.rmse
                                : step.transform.metrics.normalAll.rmse;
        edge.weight = 1.0 / std::max(0.01, rmse * rmse);
        if (rmse > 0.7) {
            edge.weight *= 0.25;
        }
        if (rmse > 1.5) {
            edge.weight *= 0.05;
        }
        if (std::abs(step.transform.da) > 0.2) {
            edge.weight *= 0.2;
        }
        edge.weight = std::clamp(edge.weight, 1e-4, 100.0);
        edges.push_back(edge);
    }

    return edges;
}

std::vector<PoseGraphEdge> buildSkipEdges(const std::vector<cv::Mat>& globalTransforms,
                                           const std::size_t skipDistance,
                                           const double weight)
{
    std::vector<PoseGraphEdge> edges;
    const std::size_t N = globalTransforms.size();
    if (N <= skipDistance + 1) return edges;

    edges.reserve(N - skipDistance - 1);
    for (std::size_t i = 0; i + skipDistance + 1 < N; ++i) {
        const std::size_t j = i + skipDistance + 1;

        const Rigid2D from = matToRigid(globalTransforms[i]);
        const Rigid2D to = matToRigid(globalTransforms[j]);
        const Rigid2D relative = compose(inverse(from), to);

        PoseGraphEdge edge;
        edge.fromIndex = i;
        edge.toIndex = j;
        edge.dx = relative.dx;
        edge.dy = relative.dy;
        edge.da = relative.daDeg;
        edge.weight = weight; // 跳边权重要低一些
        edges.push_back(edge);
    }

    return edges;
}

} // namespace stitch
