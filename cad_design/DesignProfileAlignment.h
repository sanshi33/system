#pragma once

#include "cad_design/DesignProfileTypes.h"
#include "stitch/StitchTypes.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace pinjie::cad_design {

struct DesignErrorProfilePoint {
    std::size_t index{0};
    std::size_t supportCount{0};
    double xPx{0.0};
    double yPx{0.0};
    double yStdPx{0.0};
    double sAlignedMm{0.0};
    double rAlignedMm{0.0};
    double designRadiusMm{0.0};
    double nearestDesignSMm{0.0};
    double nearestDesignRMm{0.0};
    bool hasCadCoordinates{false};
    double nearestDesignCadXMm{0.0};
    double nearestDesignCadYMm{0.0};
    double nearestDesignCadZMm{0.0};
    double measuredCadXMm{0.0};
    double measuredCadYMm{0.0};
    double measuredCadZMm{0.0};
    double compensationTargetCadXMm{0.0};
    double compensationTargetCadYMm{0.0};
    double compensationTargetCadZMm{0.0};
    double compensationDeltaCadXUm{0.0};
    double compensationDeltaCadYUm{0.0};
    double compensationDeltaCadZUm{0.0};
    std::size_t designSegmentIndex{0};
    double designDerivative{0.0};
    double radialErrorMm{0.0};
    double radialErrorUm{0.0};
    double signedDistanceMm{0.0};
    double signedDistanceUm{0.0};
    double legacyNormalErrorUm{0.0};
    double normalErrorMm{0.0};
    double normalErrorUm{0.0};
    double profileErrorUm{0.0};
    bool isUsed{false};
};

struct ErrorStats {
    std::size_t count{0};
    double meanUm{0.0};
    double rmseUm{0.0};
    double maeUm{0.0};
    double p95AbsUm{0.0};
    double maxPosUm{0.0};
    double maxNegUm{0.0};
    double pvUm{0.0};
};

struct DesignErrorSummary {
    double dzMm{0.0};
    double drMm{0.0};
    double dThetaDeg{0.0};
    double axialScaleFactor{1.0};
    double axialQuadraticTermMm{0.0};
    bool appliedAbsoluteBiasRefine{false};
    double absoluteBiasCorrectionUm{0.0};
    double preRefineMeanNormalErrorUm{0.0};
    double preRefineAbsoluteFilteredRmseUm{0.0};
    bool designReverseAxial{false};
    bool useLeftEndpointAnchor{true};
    bool evaluateProfileForm{true};
    double anchorXPx{0.0};
    double anchorYPx{0.0};
    double pixelSizeMm{0.0};
    std::size_t candidateCount{0};
    std::size_t usedCount{0};
    std::size_t outlierCount{0};
    double outlierRatio{0.0};
    double meanNormalErrorUm{0.0};
    ErrorStats absoluteAllStats;
    ErrorStats absoluteFilteredStats;
    ErrorStats normalStats;
    ErrorStats profileStats;
    DesignProfileMetadata designProfileMetadata;
};

struct DesignAlignmentResult {
    bool ok{false};
    std::string message;
    DesignErrorSummary summary;
    std::vector<DesignErrorProfilePoint> profilePoints;
    std::vector<DesignProfileSample> designProfileSamples;
    std::string profileCsvText;
    std::string summaryCsvText;
    std::string error3dCsvText;
    std::string compensationCsvText;
    std::string featureCompensationCsvText;
};

DesignAlignmentResult compareMeasuredProfileToDesign(const std::vector<stitch::EdgeVariants>& edges,
                                                     const std::vector<cv::Mat>& imageTransforms,
                                                     const stitch::StitchPipelineConfig& config);

DesignAlignmentResult compareMeasuredContourSamplesToDesign(const std::vector<cv::Point2d>& contourSamples,
                                                            const stitch::StitchPipelineConfig& config);

DesignAlignmentResult compareMeasuredCadProfileToDesign(const std::vector<DesignProfileSample>& measuredSamples,
                                                        const stitch::StitchPipelineConfig& config);

cv::Mat buildDesignComparisonPlot(const DesignAlignmentResult& result);
cv::Mat buildCompensationPlot(const DesignAlignmentResult& result);

} // namespace pinjie::cad_design
