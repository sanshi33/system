/*==============================================================================
文件：TelecentricCalibrator.h
------------------------------------------------------------------------------
远心相机标定模块（API 层）

它提供一个单函数入口：
  telecentric::RunTelecentricCalibration(opt, out_params, ...)

输入：
  - 标定图片目录
  - 标定板规格：rows/cols/pitch(mm)
  - 过滤与稳健性参数：min_valid_images、roi_radius_px 等

输出：
  - telecentric_init.txt（兼容你原工程格式：intr/dist + 每张图的 rvec/txy）
  - out_params（内存中返回 intr/dist 等）

GUI 会传一个 vis_cb 回调进来，用于：
  - 每张图的圆点阵检测可视化
  - “该图是否被接受/剔除”的原因提示（教学型阅读非常重要）

==============================================================================
*/

#pragma once
#include "TelecentricCalibIO.h"
#include <functional>
#include <string>

namespace cv { class Mat; }

namespace telecentric {

/**
 * @brief 远心标定参数（来自 GUI）
 *
 * 你可以把它理解为“跑一次标定需要的所有输入”。
 */
struct CalibOptions {
    std::string image_folder; ///< 标定图像目录（建议只放标定图）
    std::string init_path = "telecentric_init.txt"; ///< 输出/读取 init 文件

    // ------------------- 标定板几何参数 -------------------
    int rows = 7;          ///< 圆点阵行数
    int cols = 7;          ///< 圆点阵列数
    double pitch_mm = 2.5; ///< 相邻圆心距离（mm）

    // ------------------- 稳健性/过滤参数 -------------------
    /**
     * @brief 最少有效图片数量
     *
     * “有效”指：检测到完整圆点阵，并通过后续筛选（仿射 RMS、椭圆拟合有效等）。
     * 如果有效图片数量 < min_valid_images，则直接认为标定失败。
     */
    int min_valid_images = 3;

    // ------------------- 规范化圆点阵的远心约束 -------------------
    /**
     * @brief 是否启用方形像元约束。
     *
     * 对规范化圆点阵和远心测量场景，通常更希望 fx ≈ fy。
     * 默认启用后，标定会采用各向同性的远心内参模型，避免 fx/fy 自由漂移。
     */
    bool enforce_square_pixels = true;

    /**
     * @brief 是否将主点锁定到图像中心。
     *
     * 在正交/远心平面标定里，主点与每张图的平移项高度耦合。
     * 如果放开主点，往往会出现“重投影误差不大，但物理尺度明显不对”的解。
     */
    bool lock_principal_point_to_image_center = true;

    /**
     * @brief 当不锁主点时，允许主点相对图像中心的最大偏移。
     */
    double principal_point_max_offset_px = 50.0;

    /**
     * @brief 当不锁主点时，对主点回到图像中心的弱先验权重。
     */
    double principal_point_prior_weight = 0.05;

    /**
     * @brief 质量报告中的 fx/fy 差异预警阈值。
     */
    double warn_fx_fy_diff_pct = 2.0;

    /**
     * @brief 质量报告中的主点偏移预警阈值。
     */
    double warn_principal_point_offset_px = 50.0;

    /**
     * @brief 质量报告中的有效图像比例预警阈值。
     */
    double warn_valid_ratio_pct = 60.0;

    /**
     * @brief 仿射 RMS 阈值（像素）
     *
     * 为什么需要这个？
     *   Circle grid 的点序可能旋转/翻转（D4 对称），甚至误检。
     *   我们会对不同点序做仿射拟合，RMS 太大通常意味着点序不对/误检，
     *   该图会被剔除。
     */
    double affine_rms_thresh = 2.0;

    /**
     * @brief 椭圆拟合精修时的 ROI 半径（像素）
     *
     * 这一步用于把 OpenCV findCirclesGrid 的粗中心，精修成亚像素中心。
     * 半径过小：轮廓不完整；半径过大：噪声/其它结构干扰增大。
     */
    int roi_radius_px = 90;

    // ------------------- Optional: image-domain systematic residual compensation -------------------
    bool enable_image_residual_compensation = true;
    double image_residual_prior_sigma_px = 0.02;
    double image_residual_max_coeff_px = 0.2;

    // ------------------- Optional: board point self-calibration -------------------
    bool enable_board_warp_compensation = true;
    double board_warp_prior_sigma_mm = 0.02;
    double board_warp_max_offset_mm = 0.05;
    bool enable_board_point_compensation = true;
    double board_point_prior_sigma_mm = 0.005;
    double board_point_smooth_sigma_mm = 0.003;
    double board_point_max_offset_mm = 0.03;

    // ------------------- Optional: circle local edge-response calibration -------------------
    // Keep false by default so legacy geometric calibration behavior remains unchanged.
    bool enable_edge_response = false;
    int edge_samples_per_circle = 8;
    double edge_profile_half_length = 5.0;
    double edge_profile_step = 0.25;
    int edge_fit_max_iters = 40;
    double edge_fit_tol = 1e-6;
    double edge_sigma_min = 0.2;
    double edge_sigma_max = 6.0;
    double edge_max_fit_rms = 3.0;
    double edge_min_contrast = 8.0;
    double edge_min_gradient = 1.0;
    int edge_min_valid_profiles = 30;
    bool edge_fit_radius_model = true;
    std::string edge_output_txt = "telecentric_edge_response.txt";
    std::string edge_profiles_csv = "circle_edge_profiles.csv";
    std::string edge_summary_csv = "circle_edge_response_summary.csv";
};

struct CalibQualityReport {
    int total_images = 0;
    int valid_images = 0;
    double reproj_rms_px = -1.0;
    double mean_affine_rms_px = -1.0;
    int image_width = 0;
    int image_height = 0;
    double valid_ratio_pct = -1.0;
    double fx_px_per_mm = 0.0;
    double fy_px_per_mm = 0.0;
    double fx_fy_diff_pct = 0.0;
    double cx_px = 0.0;
    double cy_px = 0.0;
    double cx_offset_px = 0.0;
    double cy_offset_px = 0.0;
    std::string intrinsic_model;
    std::string principal_point_policy;
    std::string warning_summary;
    std::string init_path;
    std::string init_hash_hex;
    std::string created_at;
};

std::string DefaultCalibReportPath(const std::string& init_path);
bool SaveCalibQualityReport(const std::string& path,
                            const CalibQualityReport& report,
                            std::string* err = nullptr);
bool LoadCalibQualityReport(const std::string& path,
                            CalibQualityReport& report,
                            std::string* err = nullptr);

// ---------------------- GUI 实时可视化回调 ----------------------
//
// stage 固定为 "detect"；idx/total 以 files 的顺序计数
// accepted=true 表示该图被纳入优化（有效）
// vis：用于展示的可视化图（原图叠加圆点/编号等）
using CalibVisCallback = std::function<void(const std::string& stage,
                                            int idx,
                                            int total,
                                            const std::string& file,
                                            bool accepted,
                                            const cv::Mat& vis)>;
using CalibLogCallback = std::function<void(const std::string& line)>;

/**
 * @brief 运行一次远心标定
 *
 * @param opt      标定参数（目录、标定板规格、过滤阈值等）
 * @param out_params 输出：最终 intr/dist 等
 * @param err      输出：失败原因（可为空）
 * @param vis_cb   输出：每张图的实时可视化（可为空）
 *
 * @return true=成功；false=失败
 */
bool RunTelecentricCalibration(const CalibOptions& opt,
                               CalibParams& out_params,
                               std::string* err = nullptr,
                               CalibVisCallback vis_cb = nullptr,
                               CalibQualityReport* quality_out = nullptr,
                               CalibLogCallback log_cb = nullptr);

} // namespace telecentric
