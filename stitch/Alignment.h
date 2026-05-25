#pragma once

#include "StitchTypes.h"
#include <opencv2/core.hpp>
#include <vector>

namespace stitch {

double computeRobustCost(const std::vector<cv::Point2d>& edges1,
                         const std::vector<cv::Point2d>& edges2,
                         double offset_primary,
                         double& out_mean_perp,
                         std::vector<double>* out_inlier_errs = nullptr,
                         std::vector<double>* out_inlier_coords = nullptr,
                         TransformResult::AlignAxis axis = TransformResult::AlignAxis::X);

TransformResult computeGlobalAlignment(const std::vector<cv::Point2d>& edges1,
                                       const std::vector<cv::Point2d>& edges2,
                                       double expected_shift,
                                       double search_range,
                                       double max_perp_threshold,
                                       TransformResult::AlignAxis axis = TransformResult::AlignAxis::X,
                                       double tangent_residual_cost_weight = 0.05,
                                       double tangent_correlation_cost_weight = 0.25);

TransformResult matchOnePair(const EdgeVariants& prev_edges,
                             const EdgeVariants& next_edges,
                             const cv::Point2d& center,
                             double approx_shift,
                             double approx_shift_y,
                             double approx_angle_deg,
                             bool has_reliable_motion_prior,
                             double base_search_range,
                             MotionPriorDirection direction_constraint,
                             double rotation_search_min_deg,
                             double rotation_search_max_deg,
                             double rotation_search_step_deg,
                             double tangent_residual_cost_weight,
                             double tangent_correlation_cost_weight,
                             bool strict_local_prior_window,
                             double local_primary_search_half_range_px,
                             double local_perp_search_half_range_px,
                             double& search_range_x,
                             double& search_range_y);

void populateAlignmentMetrics(const std::vector<cv::Point2d>& ref_edges,
                              const std::vector<cv::Point2d>& rotated_target_edges,
                              TransformResult& res);

} // namespace stitch
