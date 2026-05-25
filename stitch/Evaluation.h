#pragma once
#include "StitchTypes.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace stitch {

// 对一组点进行圆拟合并计算误差
RealEvalResult evaluateCircleFit(const std::vector<cv::Point2d>& pts,
                                 const std::string& imageName,
                                 const std::string& algoName);

// 在单图上跑某一种 detector 并返回拟合结果
RealEvalResult runDetectionOnImage(const cv::Mat& img,
                                   const std::string& imageName,
                                   const EdgeDetectConfig& cfg);

// 对目录下的所有图片做评估（输出 CSV）
// targetDiameter/pixelScale 为 0 表示不启用对应的物理尺度换算
void runRealBallEvaluation(const std::string& realBallDir,
                           const EdgeDetectConfig& cfg,
                           double targetDiameter = 0.0,
                           double pixelScale = 0.0,
                           bool preferPixelScale = false);

} // namespace stitch
