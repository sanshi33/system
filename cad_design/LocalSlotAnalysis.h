#pragma once

#include "cad_design/DesignProfileTypes.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace pinjie::cad_design {

struct LocalSlotAnalysisRequest {
    std::string sourceImagePath;
    double bottomWidthMm{1.527};
    double pixelSizeOverrideMm{0.0};
    double pixelSizeScale{1.001};
    double maxAcceptedErrorUm{250.0};
    std::size_t maxOutputPoints{1600};
    DesignProfileMetadata cadProfileMetadata{};
    std::vector<DesignProfileSample> cadProfileSamples;
};

struct LocalSlotAnalysisResult {
    bool ok{false};
    std::string message;
    std::size_t detectedPointCount{0};
    std::size_t usedPointCount{0};
    std::size_t outlierPointCount{0};
    double bottomWidthPx{0.0};
    double pixelSizeMm{0.0};
    double measuredSlotWidthMm{0.0};
    double widthErrorUm{0.0};
    double normalRmseUm{0.0};
    double normalP95AbsUm{0.0};
    double normalPvUm{0.0};
    double localExceedanceThresholdUm{15.0};
    std::size_t localExceedanceCount{0};
    double localMaxAbsErrorUm{0.0};
    double localMaxExceedanceUm{0.0};
    std::string profileCsvText;
    std::string summaryCsvText;
    std::string compensationCsvText;
    std::string error3dCsvText;
    std::string featureCompensationCsvText;
    std::string qualityReviewCsvText;
    std::string contourPointsCsvText;
};

LocalSlotAnalysisResult analyzeLocalSlotImage(const cv::Mat& image,
                                              const LocalSlotAnalysisRequest& request);

} // namespace pinjie::cad_design
