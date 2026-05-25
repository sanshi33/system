#pragma once

#include <array>
#include <string>

namespace telecentric {

/**
 * @brief 远心标定参数。
 *
 * `intr` 的顺序为 `fx, fy, skew, cx, cy`：
 * - `fx`, `fy`：像素与物理长度比例，单位 `px/mm`
 * - `skew`：轴间斜切项
 * - `cx`, `cy`：主点坐标，单位像素
 *
 * `dist` 的顺序为 `k1, k2, k3, p1, p2`，
 * 畸变模型定义在像素域主点附近。
 */
struct CalibParams {
    std::array<double, 5> intr{};
    std::array<double, 5> dist{};
    std::array<double, 6> image_residual_quad{};
};

/**
 * @brief 从 `telecentric_init.txt` 读取标定参数。
 *
 * 该接口兼容旧工程格式：即使文件中还包含 `image/rvec/tvec` 等行，
 * 这里只会提取当前拼接流程真正需要的 `intr/dist`。
 *
 * @param path 标定初始化文件路径。
 * @param out 输出的标定参数。
 * @param err 失败时返回错误信息，可为空。
 * @return `true` 表示读取成功，`false` 表示失败。
 */
bool LoadFromInitTxt(const std::string& path, CalibParams& out, std::string* err = nullptr);

/**
 * @brief 将标定参数保存为 `telecentric_init.txt` 文本格式。
 *
 * 当前只写入 `intr/dist`，不写每张标定图的外参。
 *
 * @param path 输出文件路径。
 * @param p 要保存的标定参数。
 * @param err 失败时返回错误信息，可为空。
 * @return `true` 表示保存成功，`false` 表示失败。
 */
bool SaveToInitTxt(const std::string& path, const CalibParams& p, std::string* err = nullptr);

} // namespace telecentric
