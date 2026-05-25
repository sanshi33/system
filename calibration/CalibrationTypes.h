#pragma once

#include <functional>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace pinjie {

struct CalibrationBoardSpec {
    int rows{7};
    int cols{7};
    double pitchMm{2.5};
    int roiRadiusPx{90};
    int minValidImages{3};
    bool enforceSquarePixels{true};
    bool lockPrincipalPointToImageCenter{true};
    bool enableImageResidualCompensation{true};
    double imageResidualPriorSigmaPx{0.02};
    double imageResidualMaxCoeffPx{0.2};
    bool enableBoardWarpCompensation{true};
    double boardWarpPriorSigmaMm{0.02};
    double boardWarpMaxOffsetMm{0.05};
    bool enableBoardPointCompensation{true};
    double boardPointPriorSigmaMm{0.005};
    double boardPointSmoothSigmaMm{0.003};
    double boardPointMaxOffsetMm{0.03};
};

struct CalibrationQualitySummary {
    int totalImages{0};
    int validImages{0};
    int imageWidth{0};
    int imageHeight{0};
    double validRatioPercent{-1.0};
    double reprojectionRmsPx{-1.0};
    double meanAffineRmsPx{-1.0};
    double fxPixelsPerMm{0.0};
    double fyPixelsPerMm{0.0};
    double fxFyDiffPercent{0.0};
    double principalPointX{0.0};
    double principalPointY{0.0};
    double principalPointOffsetX{0.0};
    double principalPointOffsetY{0.0};
    std::string intrinsicModel;
    std::string principalPointPolicy;
    std::string warningSummary;

    [[nodiscard]] bool hasWarning() const noexcept
    {
        return !warningSummary.empty();
    }
};

struct CalibrationProfile {
    std::string profileName;
    std::string sourcePath;
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
    cv::Mat imageResidualQuadraticCoefficients;
    CalibrationQualitySummary quality;
    bool valid{false};
};

struct CalibrationCachePaths {
    std::string outputDir;
    std::string cacheDir;
    std::string initPath;
    std::string profilePath;
    std::string undistortMapXPath;
    std::string undistortMapYPath;
    std::string qualityReportPath;
    std::string previewImagePath;
};

struct CalibrationResultCache {
    CalibrationProfile profile;
    CalibrationCachePaths paths;
    cv::Mat undistortMapX;
    cv::Mat undistortMapY;
    std::string createdAt;
    bool valid{false};

    [[nodiscard]] bool hasUndistortMaps() const noexcept
    {
        return !undistortMapX.empty() && !undistortMapY.empty();
    }
};

struct CameraCalibrationCallbacks {
    std::function<void(const std::string&)> logMessage;
    std::function<void(int current, int total)> progressChanged;
    std::function<void(int index, int total, const cv::Mat& image)> previewReady;
};

struct CameraCalibrationRequest {
    std::string sessionName;
    std::string imageDirectory;
    std::vector<std::string> imagePaths;
    std::string outputDir;
    std::string cacheDir;
    CalibrationBoardSpec boardSpec{};
    bool preferCachedResult{true};
    bool persistCache{true};
    CameraCalibrationCallbacks callbacks{};
};

struct CameraCalibrationResult {
    bool ok{false};
    bool loadedFromCache{false};
    std::string message;
    CalibrationProfile profile;
    CalibrationResultCache cache;
};

} // namespace pinjie
