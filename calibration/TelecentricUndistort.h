#pragma once

#include "TelecentricCalibIO.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace telecentric {

// 畸变参数定义在像素域主点附近。

/**
 * @brief 对像素点做远心模型去畸变，输入输出都仍是像素坐标。
 *
 * 内部流程为：
 * 1. 用 `intr` 把像素坐标换到物理平面；
 * 2. 反求无畸变平面点；
 * 3. 再投影回像素坐标。
 *
 * 这个接口主要用于：
 * - 拼接后对融合边缘点逐点去畸变
 * - 几何评估时把像素轮廓转换到更一致的测量空间
 *
 * @param in_px 输入的畸变像素点。
 * @param out_px 输出的去畸变像素点。
 * @param p 标定参数，包含 `intr/dist`。
 * @param max_iter 反求无畸变点时的最大迭代次数。
 * @return `true` 表示全部点处理成功，`false` 表示失败。
 */
bool UndistortPointsPxToPx(const std::vector<cv::Point2d>& in_px,
                           std::vector<cv::Point2d>& out_px,
                           const CalibParams& p,
                           int max_iter = 20);

bool BuildUndistortMaps(const cv::Size& image_size,
                        cv::Mat& map_x,
                        cv::Mat& map_y,
                        const CalibParams& p);

/**
 * @brief 对整幅图像执行去畸变，输出尺寸与输入一致。
 *
 * 这个接口主要用于拼接前预处理，即先对原图做 remap，
 * 再进入边缘提取与匹配流程。
 *
 * @param img 输入图像。
 * @param out 输出图像。
 * @param p 标定参数，包含 `intr/dist`。
 * @param interp 重采样插值方式，默认双线性插值。
 * @param border_mode 图像边界填充方式。
 * @return `true` 表示处理成功，`false` 表示失败。
 */
bool UndistortImage(const cv::Mat& img, cv::Mat& out, const CalibParams& p,
                    int interp = cv::INTER_LINEAR,
                    int border_mode = cv::BORDER_CONSTANT);

} // namespace telecentric
