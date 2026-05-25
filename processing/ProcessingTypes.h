#pragma once

#include "stitch/StitchTypes.h"

#include <opencv2/core.hpp>

#include <string>

namespace pinjie {

using EdgeDetectConfig = stitch::EdgeDetectConfig;
using EdgeVariants = stitch::EdgeVariants;

struct ProcessedFrame {
    std::string imagePath;
    cv::Mat sourceImage;
    cv::Mat previewImage;
    EdgeVariants edges;
};

} // namespace pinjie
