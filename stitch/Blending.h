#pragma once
#include "StitchTypes.h"
#include <opencv2/core.hpp>

namespace stitch {

// 初始化画布并放置第一张图，输出全局变换矩阵
void initCanvasAndPlaceFirst(const cv::Mat& first,
                             cv::Mat& canvas,
                             cv::Mat& globalTransform);

// 应用一步相对变换（dx/dy/da）并融合到 canvas（同时更新 globalTransform）
void applyTransformAndBlend(const cv::Mat& img_next,
                            cv::Mat& canvas,
                            cv::Mat& globalTransform,
                            const cv::Point2d& center,
                            const TransformResult& step);

// 自动裁剪空白区域
cv::Rect cropCanvasAuto(cv::Mat& canvas);

} // namespace stitch
