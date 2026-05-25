#pragma once

#include "StitchTypes.h"

namespace stitch {

/**
 * @brief GUI / CLI 共用的一次完整拼接入口。
 *
 * 这个接口会依次完成：
 * - 输入图像加载
 * - 亚像素边缘提取与前置滤波
 * - 顺序拼接主流程执行
 * - 调试图和 CSV 结果生成
 * - 可选的文件保存
 *
 * 上层只需要准备好 `StitchRunRequest` 和 `StitchCallbacks`，
 * 然后在工作线程中调用本函数即可。
 */
StitchRunResult runStitching(const StitchRunRequest& request,
                             StitchRunMode runMode = StitchRunMode::Full,
                             const StitchCallbacks& callbacks = {});

} // namespace stitch
