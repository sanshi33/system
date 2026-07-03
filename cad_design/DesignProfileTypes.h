#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pinjie::cad_design {

struct DesignProfileSample {
    double sMm{0.0};
    double rMm{0.0};
    bool hasCadPoint{false};
    double cadXMm{0.0};
    double cadYMm{0.0};
    double cadZMm{0.0};
};

struct DesignProfileMetadata {
    std::string sourceType{"builtin"};
    std::string sourceName{"builtin_generatrix"};
    std::string sourcePath;
    std::string extractionMethod{"builtin_function"};
    std::string axialAxis{"X"};
    std::string radialAxis{"Y"};
    std::string sectionNormalAxis;
    double sectionCoordinateMm{0.0};
    double cadAxialOriginMm{0.0};
    double cadAxialDirectionSign{1.0};
    std::size_t sampleCount{0};
    double minSMm{0.0};
    double maxSMm{0.0};
    double minRMm{0.0};
    double maxRMm{0.0};
    bool hasCadBounds{false};
    double minCadXMm{0.0};
    double minCadYMm{0.0};
    double minCadZMm{0.0};
    double maxCadXMm{0.0};
    double maxCadYMm{0.0};
    double maxCadZMm{0.0};
};

} // namespace pinjie::cad_design
