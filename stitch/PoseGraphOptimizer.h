#pragma once

#include "StitchTypes.h"

#include <opencv2/core.hpp>

#include <vector>

namespace stitch {

/**
 * @brief 位姿图的一条边：从 fromIndex 到 toIndex 的相对变换及置信权重。
 */
struct PoseGraphEdge {
    std::size_t fromIndex{0};
    std::size_t toIndex{0};
    double dx{0.0};      // 相对平移 X (px)
    double dy{0.0};      // 相对平移 Y (px)
    double da{0.0};      // 相对旋转 (度)
    double weight{1.0};  // 置信权重 (越高越可信)
};

/**
 * @brief 位姿图优化：最小化所有边的约束不一致性 + 正则项。
 *
 * @param initialGlobalTransforms  顺序拼接得到的初始全局变换 (3x3 仿射矩阵)
 * @param edges                    位姿图边约束
 * @param regularizationWeight     正则化强度（阻止偏离初始值太远）
 * @param maxIterations            最大迭代次数
 * @return 优化后的全局变换矩阵
 */
std::vector<cv::Mat> optimizePoseGraph(const std::vector<cv::Mat>& initialGlobalTransforms,
                                       const std::vector<PoseGraphEdge>& edges,
                                       double regularizationWeight = 0.15,
                                       int maxIterations = 30);

/**
 * @brief 从拼接步骤记录构造相邻边约束。
 */
std::vector<PoseGraphEdge> buildAdjacentEdges(const std::vector<StitchStepRecord>& steps);

/**
 * @brief 为位姿图补充跳边约束（i -> i+2, i -> i+3 ...），
 *        通过复用已有全局变换计算近似的相对位姿。
 */
std::vector<PoseGraphEdge> buildSkipEdges(const std::vector<cv::Mat>& globalTransforms,
                                           std::size_t skipDistance,
                                           double weight);

} // namespace stitch
