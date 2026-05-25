#pragma once

#include "StitchTypes.h"

#include <opencv2/core.hpp>

#include <vector>

namespace stitch {

/**
 * @brief 纯算法层的拼接入口。
 *
 * 该接口不直接负责：
 * - 终端日志输出
 * - CSV 文件保存
 * - 图像落盘
 *
 * 它只负责：
 * - 根据输入图像和预处理后的轮廓完成拼接
 * - 通过回调报告过程事件
 * - 返回结构化拼接结果
 */
StitchingResult runStitchingPipeline(const std::vector<cv::Mat>& images,
                                     const std::vector<EdgeVariants>& edgesPrepared,
                                     const StitchPipelineConfig& config = {},
                                     const StitchCallbacks& callbacks = {});

} // namespace stitch
