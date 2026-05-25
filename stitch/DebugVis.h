#pragma once

#include "StitchTypes.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace stitch {

/**
 * @brief 生成单步拼接的调试可视化图。
 *
 * 该函数只负责返回 `cv::Mat`，不直接落盘。
 * GUI 可以直接显示这张图，CLI 或服务层则可以自行决定是否保存。
 */
cv::Mat buildDebugStepVisualization(const cv::Mat& imgRef,
                                    const cv::Mat& imgTarget,
                                    const std::vector<cv::Point2d>& edgesRef,
                                    const std::vector<cv::Point2d>& edgesTarget,
                                    const TransformResult& result,
                                    int stepIndex);

/**
 * @brief 生成单张图像在预处理阶段的边缘预览图。
 */
cv::Mat buildPreprocessVisualization(const cv::Mat& sourceImage,
                                     const std::vector<cv::Point2d>& filteredEdges,
                                     int imageIndex,
                                     int totalImages);

cv::Mat buildStitchedContourOverlay(const cv::Mat& panorama,
                                    const std::vector<EdgeVariants>& edges,
                                    const std::vector<cv::Mat>& imageTransforms);

cv::Mat buildStitchedContourProfilePlot(const std::vector<EdgeVariants>& edges,
                                        const std::vector<cv::Mat>& imageTransforms);

cv::Mat buildTangentCorrelationPlot(const std::vector<StitchStepRecord>& steps,
                                    bool useInlierCorrelation);

/**
 * @brief 生成单步拼接的文本摘要。
 */
std::string formatStepSummary(const StitchStepRecord& step);

/**
 * @brief 根据步骤记录生成整份 CSV 文本。
 */
std::string buildStitchingCsv(const std::vector<StitchStepRecord>& steps);

std::string buildContourPointCsv(const std::vector<EdgeVariants>& edges,
                                 const std::vector<cv::Mat>& imageTransforms);

std::string buildOriginContourOverlayCsv(const std::vector<EdgeVariants>& edges,
                                         const std::vector<cv::Mat>& imageTransforms);

std::string buildStitchedContourProfileCsv(const std::vector<EdgeVariants>& edges,
                                           const std::vector<cv::Mat>& imageTransforms);

std::string buildTangentCorrelationStepCsv(const std::vector<StitchStepRecord>& steps);

std::string buildNormalErrorProfileCsv(const std::vector<StitchStepRecord>& steps);

std::string buildTangentProfileCompareCsv(const std::vector<StitchStepRecord>& steps);

std::string buildOriginTangentPointMetricsCsv(const std::vector<StitchStepRecord>& steps);

std::string buildAlignmentCandidateDiagnosticsCsv(const std::vector<StitchStepRecord>& steps);

} // namespace stitch
