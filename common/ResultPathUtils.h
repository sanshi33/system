#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace pinjie {

struct StitchResultPathSet {
    std::filesystem::path projectRootDir;
    std::filesystem::path resultRootDir;
    std::filesystem::path runDir;
    std::filesystem::path panoramaPath;
    std::filesystem::path csvPath;
    std::filesystem::path designErrorProfileCsvPath;
    std::filesystem::path designErrorSummaryCsvPath;
    std::filesystem::path designComparisonOverlayPath;
    std::filesystem::path qualityReviewCsvPath;
    std::filesystem::path alignmentCandidateDiagnosticsCsvPath;
    std::filesystem::path contourPointsCsvPath;
    std::filesystem::path originContourOverlayCsvPath;
    std::filesystem::path stitchedContourProfileCsvPath;
    std::filesystem::path tangentStepCsvPath;
    std::filesystem::path normalErrorProfileCsvPath;
    std::filesystem::path tangentProfileCsvPath;
    std::filesystem::path originTangentPointMetricsCsvPath;
    std::filesystem::path debugDir;
    std::filesystem::path contourOverlayPath;
    std::filesystem::path stitchedContourProfilePlotPath;
    std::filesystem::path tangentCorrelationAllPath;
    std::filesystem::path tangentCorrelationInlierPath;
};

struct CalibrationResultPathSet {
    std::filesystem::path projectRootDir;
    std::filesystem::path resultRootDir;
    std::filesystem::path runDir;
    std::filesystem::path cacheDir;
    std::filesystem::path initPath;
    std::filesystem::path profilePath;
    std::filesystem::path undistortMapXPath;
    std::filesystem::path undistortMapYPath;
    std::filesystem::path qualityReportPath;
    std::filesystem::path previewImagePath;
};

inline std::filesystem::path projectRootPath()
{
#ifdef PINJIE_PROJECT_ROOT
    return std::filesystem::path(PINJIE_PROJECT_ROOT);
#else
    return std::filesystem::current_path();
#endif
}

inline std::string currentTimestampTag()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y%m%d_%H%M%S");
    return stream.str();
}

inline std::string sanitizePathToken(const std::string& value, const std::string& fallback)
{
    std::string sanitized;
    sanitized.reserve(value.size());

    bool previousWasSeparator = false;
    for (unsigned char ch : value) {
        const bool forbidden = ch < 32 || ch == '<' || ch == '>' || ch == ':' || ch == '"' || ch == '/' ||
                               ch == '\\' || ch == '|' || ch == '?' || ch == '*';
        if (!forbidden) {
            sanitized.push_back(static_cast<char>(ch));
            previousWasSeparator = false;
            continue;
        }

        if (!previousWasSeparator) {
            sanitized.push_back('_');
            previousWasSeparator = true;
        }
    }

    while (!sanitized.empty() && (sanitized.front() == ' ' || sanitized.front() == '.')) {
        sanitized.erase(sanitized.begin());
    }
    while (!sanitized.empty() && (sanitized.back() == ' ' || sanitized.back() == '.')) {
        sanitized.pop_back();
    }

    return sanitized.empty() ? fallback : sanitized;
}

inline StitchResultPathSet buildDefaultStitchResultPaths(const std::string& runName,
                                                         const std::string& category,
                                                         bool saveDebugImages)
{
    StitchResultPathSet paths;
    paths.projectRootDir = projectRootPath();
    paths.resultRootDir = paths.projectRootDir / "result";

    const std::string safeCategory = sanitizePathToken(category, "workpiece");
    const std::string safeRunName = sanitizePathToken(runName, "run");
    const std::string runFolderName = safeRunName + "_" + currentTimestampTag();

    paths.runDir = paths.resultRootDir / safeCategory / runFolderName;
    paths.panoramaPath = paths.runDir / "final_panorama.png";
    paths.csvPath = paths.runDir / "stitching_data.csv";
    paths.designErrorProfileCsvPath = paths.runDir / "design_error_profile.csv";
    paths.designErrorSummaryCsvPath = paths.runDir / "design_error_summary.csv";
    paths.designComparisonOverlayPath = paths.runDir / "design_comparison_overlay.png";
    paths.qualityReviewCsvPath = paths.runDir / "quality_review.csv";
    paths.alignmentCandidateDiagnosticsCsvPath = paths.runDir / "alignment_candidate_diagnostics.csv";
    paths.contourPointsCsvPath = paths.runDir / "contour_points.csv";
    paths.originContourOverlayCsvPath = paths.runDir / "origin_contour_overlay_points.csv";
    paths.stitchedContourProfileCsvPath = paths.runDir / "stitched_contour_profile.csv";
    paths.tangentStepCsvPath = paths.runDir / "tangent_correlation_by_step.csv";
    paths.normalErrorProfileCsvPath = paths.runDir / "normal_error_profile.csv";
    paths.tangentProfileCsvPath = paths.runDir / "tangent_profile_compare.csv";
    paths.originTangentPointMetricsCsvPath = paths.runDir / "origin_tangent_point_metrics.csv";
    paths.contourOverlayPath = paths.runDir / "origin_contour_overlay.png";
    paths.stitchedContourProfilePlotPath = paths.runDir / "stitched_contour_profile.png";
    paths.tangentCorrelationAllPath = paths.runDir / "tangent_correlation_all.png";
    paths.tangentCorrelationInlierPath = paths.runDir / "tangent_correlation_inlier.png";
    if (saveDebugImages) {
        paths.debugDir = paths.runDir / "debug";
    }

    return paths;
}

inline bool ensureStitchResultDirectories(const StitchResultPathSet& paths)
{
    std::error_code error;
    std::filesystem::create_directories(paths.runDir, error);
    if (error) {
        return false;
    }

    if (!paths.debugDir.empty()) {
        std::filesystem::create_directories(paths.debugDir, error);
        if (error) {
            return false;
        }
    }

    return true;
}

inline std::filesystem::path resolveResultPathUnderBase(const std::filesystem::path& path,
                                                        const std::filesystem::path& baseDir)
{
    if (path.is_absolute()) {
        return path;
    }

    if (path.empty()) {
        return baseDir;
    }

    return baseDir / path;
}

inline CalibrationResultPathSet buildDefaultCalibrationResultPaths(const std::string& runName)
{
    CalibrationResultPathSet paths;
    paths.projectRootDir = projectRootPath();
    paths.resultRootDir = paths.projectRootDir / "result";

    const std::string safeRunName = sanitizePathToken(runName, "calibration");
    const std::string runFolderName = safeRunName + "_" + currentTimestampTag();

    paths.runDir = paths.resultRootDir / "calibration" / runFolderName;
    paths.cacheDir = paths.runDir / "cache";
    paths.initPath = paths.cacheDir / "telecentric_init.txt";
    paths.profilePath = paths.cacheDir / "camera_profile.yml";
    paths.undistortMapXPath = paths.cacheDir / "undistort_map_x.yml";
    paths.undistortMapYPath = paths.cacheDir / "undistort_map_y.yml";
    paths.qualityReportPath = paths.runDir / "calibration_quality_report.txt";
    paths.previewImagePath = paths.runDir / "calibration_preview.png";
    return paths;
}

inline bool ensureCalibrationResultDirectories(const CalibrationResultPathSet& paths)
{
    std::error_code error;
    std::filesystem::create_directories(paths.runDir, error);
    if (error) {
        return false;
    }

    std::filesystem::create_directories(paths.cacheDir, error);
    return !error;
}

inline std::string genericUtf8String(const std::filesystem::path& path)
{
    return path.generic_u8string();
}

} // namespace pinjie
