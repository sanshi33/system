#pragma once

#include "common/ImageFrameInfo.h"

#include <string>
#include <vector>

namespace pinjie {

enum class CaptureMode {
    Calibration,
    Workpiece
};

struct CaptureSessionInfo {
    CaptureMode mode{CaptureMode::Workpiece};
    std::string sessionName;
    std::string outputDir;
    std::vector<ImageFrameInfo> frames;
};

} // namespace pinjie
