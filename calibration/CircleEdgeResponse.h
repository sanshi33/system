#pragma once

#include "TelecentricCalibrator.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace telecentric {

struct EdgeResponseStats {
    int total_count = 0;
    int valid_count = 0;
    double mean_fit_rms = 0.0;
    double max_fit_rms = 0.0;
};

struct EdgeResponseResult {
    double sigma0_erf_bg = 0.0;
    double sigma0_erf_integral = 0.0;
    EdgeResponseStats stats_erf_bg;
    EdgeResponseStats stats_erf_integral;
};

bool RunCircleEdgeResponseCalibration(
    const std::vector<std::string>& image_files,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size,
    const CalibOptions& opt,
    EdgeResponseResult& out,
    std::string* err = nullptr);

} // namespace telecentric
