#pragma once

#include "StitchTypes.h"
#include <opencv2/core.hpp>

class SubpixelEdgeDetector;

namespace stitch {

EdgeVariants buildEdgeVariants(SubpixelEdgeDetector &det,
                               const cv::Mat &img,
                               const EdgeDetectConfig &cfg);

} // namespace stitch
