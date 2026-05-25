#pragma once

#include "StitchTypes.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace stitch {

/**
 * @brief 根据文件路径列表加载输入图像。
 *
 * 该接口更适合 GUI，因为文件选择通常来自 `QFileDialog`，
 * 得到的是一组显式路径，而不是固定命名规则的目录。
 */
std::vector<cv::Mat> loadInputImages(const std::vector<std::string>& imagePaths,
                                     const StitchCallbacks& callbacks = {});

/**
 * @brief 对全部输入图像执行边缘预处理。
 */
std::vector<EdgeVariants> preprocessAllImages(const std::vector<cv::Mat>& images,
                                              const EdgeDetectConfig& cfg,
                                              const StitchCallbacks& callbacks = {});

bool saveImageToPath(const std::string& pathUtf8, const cv::Mat& image);

bool writeTextFileToPath(const std::string& pathUtf8, const std::string& content);

} // namespace stitch
