#pragma once

#include <cstddef>
#include <string>

namespace pinjie {

struct ImageFrameInfo {
    std::size_t index{0};
    std::string filePath;
    std::string fileName;
    double stageX{0.0};
    double stageY{0.0};
    bool hasStagePosition{false};
    double exposure{0.0};
    double gain{0.0};
    std::size_t stepIndex{0};
    std::string captureTimeIso;
    std::string serialNumber;
    std::string sessionName;
    std::string note;
};

} // namespace pinjie
