#pragma once

#ifndef SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG
#define SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG 0
#endif

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace SubpixelEdge {

/**
 * @brief 单个亚像素边缘点。
 *
 * 该结构体既可以作为最终输出点，也可以作为内部调试信息载体：
 * - `x`, `y` 是亚像素坐标
 * - `confidence` 是拟合置信度
 * - `gradient` 是检测点处的梯度强度
 */
struct SubpixelPoint {
    double x{0.0};
    double y{0.0};
    double confidence{1.0};
    double gradient{0.0};

    SubpixelPoint() = default;
    SubpixelPoint(double xValue, double yValue, double confidenceValue = 1.0, double gradientValue = 0.0);

    [[nodiscard]] double distanceTo(const SubpixelPoint& other) const;
};

} // namespace SubpixelEdge

/**
 * @brief 亚像素边缘检测器。
 *
 * 当前工程只保留基于 ERF 的边缘定位方法。
 * 这个类负责：
 * - 接收输入图像
 * - 统一进行灰度化、Canny 配置和亚像素拟合
 * - 输出适合拼接流程使用的亚像素边缘点
 *
 * 设计目标：
 * - 对外保持简单、稳定的 API
 * - 把 ERF 内部实现细节隐藏在 `core/SubpixelEdgeDetector.cpp`
 * - 保持现有工程逻辑与结果不变
 */
class SubpixelEdgeDetector {
public:
    /**
     * @brief 检测参数。
     *
     * 所有参数都围绕“边缘点提取与 ERF 拟合”展开，
     * 不涉及拼接流程本身的搜索范围或融合策略。
     */
    struct Params {
        double canny_low = 50.0;      ///< Canny 低阈值。
        double canny_high = 150.0;    ///< Canny 高阈值。
        int window_size = 7;          ///< 法向采样窗口大小，建议为奇数。
        double presmooth_sigma = 1.0; ///< 梯度计算与采样前的预平滑 sigma。
        bool use_scharr = true;       ///< `true` 时用 Scharr，`false` 时退回 Sobel。
    };

    /**
     * @brief 构造检测器。
     * @param pixelSize 保留的兼容参数，当前拼接流程不直接使用。
     */
    explicit SubpixelEdgeDetector(double pixelSize = 1.0) noexcept;

    /// @brief 覆盖整套检测参数。
    void setParams(const Params& params) noexcept { params_ = params; }

    /// @brief 读取当前检测参数。
    [[nodiscard]] const Params& params() const noexcept { return params_; }

    /**
     * @brief 设置输入图像。
     * @param image 输入图像，可以是灰度图或彩色图。
     *
     * 内部会统一转换成 `CV_8UC1` 灰度图，并清空上一轮结果缓存。
     */
    void setImage(const cv::Mat& image);

    /// @brief 单独更新 Canny 双阈值，常用于运行时快速调参。
    void setCannyThresholds(double low, double high) noexcept;

    /**
     * @brief 兼容性接口。
     *
     * 当前主要在 debug 模式下生成并保存像素级 Canny 边缘图。
     * 正常亚像素流程并不依赖这个函数的输出。
     */
    void detectCannyEdges(double low, double high, int aperture = 3, bool L2 = true);

    /**
     * @brief 执行亚像素 ERF 拟合。
     * @param window_size 可选覆盖窗口大小，传负数表示使用 `params_` 中的设置。
     * @param presmooth_sigma 可选覆盖预平滑 sigma，传负数表示使用 `params_` 中的设置。
     */
    void refineEdgesSubpixel(int window_size = -1, double presmooth_sigma = -1.0);

    /**
     * @brief 一步完成检测。
     * @param image 输入图像。
     * @return 检测到的亚像素边缘坐标。
     */
    [[nodiscard]] std::vector<cv::Point2d> detect(const cv::Mat& image);

    /// @brief 获取亚像素坐标结果。
    [[nodiscard]] const std::vector<cv::Point2d>& getSubpixelEdges() const noexcept { return edges_; }

    /// @brief 获取带置信度与梯度信息的完整结果。
    [[nodiscard]] const std::vector<SubpixelEdge::SubpixelPoint>& getSubpixelPoints() const noexcept { return points_; }

#if SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG
    /// @brief 获取调试用的像素级 Canny 边缘图。
    [[nodiscard]] const cv::Mat& getEdgeImage() const noexcept { return edge_; }
#endif

private:
    static cv::Mat toGray8U(const cv::Mat& image);

private:
    cv::Mat gray_; ///< 当前输入图像的 `CV_8UC1` 灰度缓存。

#if SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG
    cv::Mat edge_; ///< 调试模式下保存的 Canny 边缘图。
#endif

    std::vector<SubpixelEdge::SubpixelPoint> points_; ///< 完整亚像素点结果。
    std::vector<cv::Point2d> edges_;                  ///< 仅坐标版本的亚像素结果。

    double pixelSize_{1.0}; ///< 保留的兼容成员，当前流程不直接参与运算。
    Params params_{};       ///< 当前检测参数。
};
