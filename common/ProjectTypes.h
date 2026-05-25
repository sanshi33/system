#pragma once

#include "common/ImageFrameInfo.h"

#include <string>
#include <vector>

namespace pinjie {

struct CalibrationDataset {
    std::string datasetName;
    std::vector<ImageFrameInfo> images;
    std::string calibrationResultPath;
    std::string calibrationCacheDir;
    std::string activeProfileName;
};

struct WorkpieceDataset {
    std::string datasetName;
    std::vector<ImageFrameInfo> images;
    std::string workpieceName;
};

struct ProjectSession {
    std::string projectName;
    std::string projectRootDir;
    std::string resultRootDir;
    std::string latestRunDir;
    CalibrationDataset calibration;
    WorkpieceDataset workpiece;
};

} // namespace pinjie
