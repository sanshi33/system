#include "CalibrationService.h"

#include "TelecentricCalibIO.h"
#include "TelecentricCalibrator.h"
#include "TelecentricUndistort.h"
#include "common/ResultPathUtils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <vector>

namespace pinjie {

namespace {

namespace fs = std::filesystem;

fs::path pathFromUtf8(const std::string& pathUtf8)
{
#if defined(_WIN32)
    return fs::u8path(pathUtf8);
#else
    return fs::path(pathUtf8);
#endif
}

std::string toUtf8String(const fs::path& path)
{
    return path.generic_u8string();
}

bool isSupportedImageExtension(const fs::path& filePath)
{
    std::string extension = filePath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    return extension == ".bmp" || extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
           extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> scanImagePaths(const std::string& imageDirectory)
{
    std::vector<std::string> imagePaths;
    if (imageDirectory.empty()) {
        return imagePaths;
    }

    const fs::path root = pathFromUtf8(imageDirectory);
    std::error_code error;
    if (!fs::exists(root, error) || error) {
        return imagePaths;
    }

    for (const auto& entry : fs::directory_iterator(root, error)) {
        if (error) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!isSupportedImageExtension(entry.path())) {
            continue;
        }
        imagePaths.push_back(toUtf8String(entry.path()));
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

bool loadImageFromPath(const std::string& pathUtf8, const int flags, cv::Mat& image)
{
    image.release();

    std::ifstream stream(pathFromUtf8(pathUtf8), std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    const std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(stream)),
                                            std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        return false;
    }

    image = cv::imdecode(buffer, flags);
    return !image.empty();
}

bool saveImageToPath(const std::string& pathUtf8, const cv::Mat& image)
{
    if (image.empty()) {
        return false;
    }

    const fs::path filePath = pathFromUtf8(pathUtf8);
    if (filePath.has_parent_path()) {
        std::error_code error;
        fs::create_directories(filePath.parent_path(), error);
        if (error) {
            return false;
        }
    }

    std::string extension = filePath.extension().string();
    if (extension.empty()) {
        extension = ".png";
    }

    std::vector<unsigned char> encoded;
    if (!cv::imencode(extension, image, encoded)) {
        return false;
    }

    std::ofstream stream(filePath, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    stream.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    return stream.good();
}

void emitLog(const CameraCalibrationCallbacks& callbacks, const std::string& message)
{
    if (callbacks.logMessage) {
        callbacks.logMessage(message);
    }
}

void emitProgress(const CameraCalibrationCallbacks& callbacks, const int current, const int total)
{
    if (callbacks.progressChanged) {
        callbacks.progressChanged(current, total);
    }
}

void emitPreview(const CameraCalibrationCallbacks& callbacks,
                 const int index,
                 const int total,
                 const cv::Mat& image)
{
    if (callbacks.previewReady) {
        callbacks.previewReady(index, total, image);
    }
}

std::string resolveImageDirectory(const CameraCalibrationRequest& request)
{
    if (!request.imageDirectory.empty()) {
        return request.imageDirectory;
    }

    if (request.imagePaths.empty()) {
        return {};
    }

    const fs::path parent = pathFromUtf8(request.imagePaths.front()).parent_path();
    if (parent.empty()) {
        return {};
    }

    for (const std::string& imagePath : request.imagePaths) {
        if (pathFromUtf8(imagePath).parent_path() != parent) {
            return {};
        }
    }

    return toUtf8String(parent);
}

std::vector<std::string> resolveImagePaths(const CameraCalibrationRequest& request)
{
    if (!request.imagePaths.empty()) {
        return request.imagePaths;
    }
    return scanImagePaths(resolveImageDirectory(request));
}

CalibrationResultPathSet buildPathSet(const std::string& sessionName,
                                      const std::string& outputDir,
                                      const std::string& cacheDir)
{
    CalibrationResultPathSet paths =
        buildDefaultCalibrationResultPaths(sessionName.empty() ? std::string("camera_calibration") : sessionName);

    if (!outputDir.empty()) {
        paths.runDir = pathFromUtf8(outputDir);
    }
    if (!cacheDir.empty()) {
        paths.cacheDir = pathFromUtf8(cacheDir);
    } else if (!outputDir.empty()) {
        paths.cacheDir = paths.runDir / "cache";
    }

    if (!paths.runDir.empty()) {
        paths.resultRootDir = paths.runDir.parent_path();
        if (!paths.resultRootDir.empty()) {
            paths.projectRootDir = paths.resultRootDir.parent_path();
        }
    }

    paths.initPath = paths.cacheDir / "telecentric_init.txt";
    paths.profilePath = paths.cacheDir / "camera_profile.yml";
    paths.undistortMapXPath = paths.cacheDir / "undistort_map_x.yml";
    paths.undistortMapYPath = paths.cacheDir / "undistort_map_y.yml";
    paths.qualityReportPath = paths.runDir / "calibration_quality_report.txt";
    paths.previewImagePath = paths.runDir / "calibration_preview.png";
    return paths;
}

CalibrationResultPathSet buildPathSetForCacheDir(const std::string& cacheDir)
{
    CalibrationResultPathSet paths;
    paths.cacheDir = pathFromUtf8(cacheDir);
    paths.runDir = paths.cacheDir.parent_path();
    paths.resultRootDir = paths.runDir.parent_path();
    if (!paths.resultRootDir.empty()) {
        paths.projectRootDir = paths.resultRootDir.parent_path();
    }
    paths.initPath = paths.cacheDir / "telecentric_init.txt";
    paths.profilePath = paths.cacheDir / "camera_profile.yml";
    paths.undistortMapXPath = paths.cacheDir / "undistort_map_x.yml";
    paths.undistortMapYPath = paths.cacheDir / "undistort_map_y.yml";
    paths.qualityReportPath = paths.runDir / "calibration_quality_report.txt";
    paths.previewImagePath = paths.runDir / "calibration_preview.png";
    return paths;
}

CalibrationCachePaths toCachePaths(const CalibrationResultPathSet& paths)
{
    CalibrationCachePaths cachePaths;
    cachePaths.outputDir = toUtf8String(paths.runDir);
    cachePaths.cacheDir = toUtf8String(paths.cacheDir);
    cachePaths.initPath = toUtf8String(paths.initPath);
    cachePaths.profilePath = toUtf8String(paths.profilePath);
    cachePaths.undistortMapXPath = toUtf8String(paths.undistortMapXPath);
    cachePaths.undistortMapYPath = toUtf8String(paths.undistortMapYPath);
    cachePaths.qualityReportPath = toUtf8String(paths.qualityReportPath);
    cachePaths.previewImagePath = toUtf8String(paths.previewImagePath);
    return cachePaths;
}

CalibrationQualitySummary toQualitySummary(const telecentric::CalibQualityReport& report)
{
    CalibrationQualitySummary summary;
    summary.totalImages = report.total_images;
    summary.validImages = report.valid_images;
    summary.imageWidth = report.image_width;
    summary.imageHeight = report.image_height;
    summary.validRatioPercent = report.valid_ratio_pct;
    summary.reprojectionRmsPx = report.reproj_rms_px;
    summary.meanAffineRmsPx = report.mean_affine_rms_px;
    summary.fxPixelsPerMm = report.fx_px_per_mm;
    summary.fyPixelsPerMm = report.fy_px_per_mm;
    summary.fxFyDiffPercent = report.fx_fy_diff_pct;
    summary.principalPointX = report.cx_px;
    summary.principalPointY = report.cy_px;
    summary.principalPointOffsetX = report.cx_offset_px;
    summary.principalPointOffsetY = report.cy_offset_px;
    summary.intrinsicModel = report.intrinsic_model;
    summary.principalPointPolicy = report.principal_point_policy;
    summary.warningSummary = report.warning_summary;
    return summary;
}

telecentric::CalibQualityReport toTelecentricQualityReport(const CalibrationResultCache& cache)
{
    telecentric::CalibQualityReport report;
    const CalibrationQualitySummary& summary = cache.profile.quality;
    report.total_images = summary.totalImages;
    report.valid_images = summary.validImages;
    report.image_width = summary.imageWidth;
    report.image_height = summary.imageHeight;
    report.valid_ratio_pct = summary.validRatioPercent;
    report.reproj_rms_px = summary.reprojectionRmsPx;
    report.mean_affine_rms_px = summary.meanAffineRmsPx;
    report.fx_px_per_mm = summary.fxPixelsPerMm;
    report.fy_px_per_mm = summary.fyPixelsPerMm;
    report.fx_fy_diff_pct = summary.fxFyDiffPercent;
    report.cx_px = summary.principalPointX;
    report.cy_px = summary.principalPointY;
    report.cx_offset_px = summary.principalPointOffsetX;
    report.cy_offset_px = summary.principalPointOffsetY;
    report.intrinsic_model = summary.intrinsicModel;
    report.principal_point_policy = summary.principalPointPolicy;
    report.warning_summary = summary.warningSummary;
    report.init_path = cache.paths.initPath;
    report.created_at = cache.createdAt;
    return report;
}

CalibrationProfile buildProfile(const std::string& profileName,
                                const std::string& sourcePath,
                                const telecentric::CalibParams& params,
                                const telecentric::CalibQualityReport* quality)
{
    CalibrationProfile profile;
    profile.profileName = profileName;
    profile.sourcePath = sourcePath;
    profile.cameraMatrix =
        (cv::Mat_<double>(3, 3) << params.intr[0], params.intr[2], params.intr[3],
                                   0.0,            params.intr[1], params.intr[4],
                                   0.0,            0.0,            1.0);
    profile.distortionCoefficients =
        (cv::Mat_<double>(1, 5) << params.dist[0], params.dist[1], params.dist[2], params.dist[3], params.dist[4]);
    profile.imageResidualQuadraticCoefficients =
        (cv::Mat_<double>(1, 6) << params.image_residual_quad[0],
                                   params.image_residual_quad[1],
                                   params.image_residual_quad[2],
                                   params.image_residual_quad[3],
                                   params.image_residual_quad[4],
                                   params.image_residual_quad[5]);
    if (quality) {
        profile.quality = toQualitySummary(*quality);
    }
    profile.valid = true;
    return profile;
}

telecentric::CalibParams toTelecentricParams(const CalibrationProfile& profile)
{
    telecentric::CalibParams params;

    cv::Mat cameraMatrix;
    profile.cameraMatrix.convertTo(cameraMatrix, CV_64F);
    if (cameraMatrix.rows == 3 && cameraMatrix.cols == 3) {
        params.intr[0] = cameraMatrix.at<double>(0, 0);
        params.intr[1] = cameraMatrix.at<double>(1, 1);
        params.intr[2] = cameraMatrix.at<double>(0, 1);
        params.intr[3] = cameraMatrix.at<double>(0, 2);
        params.intr[4] = cameraMatrix.at<double>(1, 2);
    }

    if (!profile.distortionCoefficients.empty()) {
        cv::Mat distortionCoefficients;
        profile.distortionCoefficients.reshape(1, 1).convertTo(distortionCoefficients, CV_64F);
        for (int i = 0; i < std::min(5, distortionCoefficients.cols); ++i) {
            params.dist[i] = distortionCoefficients.at<double>(0, i);
        }
    }

    if (!profile.imageResidualQuadraticCoefficients.empty()) {
        cv::Mat imageResidualQuadraticCoefficients;
        profile.imageResidualQuadraticCoefficients.reshape(1, 1).convertTo(imageResidualQuadraticCoefficients, CV_64F);
        for (int i = 0; i < std::min(6, imageResidualQuadraticCoefficients.cols); ++i) {
            params.image_residual_quad[i] = imageResidualQuadraticCoefficients.at<double>(0, i);
        }
    }

    return params;
}

std::string defaultProfileName(const CalibrationResultPathSet& paths, const std::string& sessionName)
{
    if (!sessionName.empty()) {
        return sessionName;
    }

    const std::string runDirName = paths.runDir.filename().u8string();
    return runDirName.empty() ? std::string("camera_calibration") : runDirName;
}

bool saveMatrixFile(const std::string& pathUtf8, const std::string& key, const cv::Mat& matrix)
{
    if (matrix.empty()) {
        return false;
    }

    const fs::path filePath = pathFromUtf8(pathUtf8);
    if (filePath.has_parent_path()) {
        std::error_code error;
        fs::create_directories(filePath.parent_path(), error);
        if (error) {
            return false;
        }
    }

    cv::FileStorage storage(pathUtf8, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!storage.isOpened()) {
        return false;
    }

    storage << key << matrix;
    return true;
}

bool loadMatrixFile(const std::string& pathUtf8, const std::string& key, cv::Mat& matrix)
{
    matrix.release();

    cv::FileStorage storage(pathUtf8, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        return false;
    }

    storage[key] >> matrix;
    return !matrix.empty();
}

bool saveProfileFile(const std::string& pathUtf8, const CalibrationResultCache& cache)
{
    const fs::path filePath = pathFromUtf8(pathUtf8);
    if (filePath.has_parent_path()) {
        std::error_code error;
        fs::create_directories(filePath.parent_path(), error);
        if (error) {
            return false;
        }
    }

    cv::FileStorage storage(pathUtf8, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!storage.isOpened()) {
        return false;
    }

    storage << "profile_name" << cache.profile.profileName;
    storage << "source_path" << cache.profile.sourcePath;
    storage << "created_at" << cache.createdAt;
    storage << "camera_matrix" << cache.profile.cameraMatrix;
    storage << "distortion_coefficients" << cache.profile.distortionCoefficients;
    storage << "image_residual_quadratic_coefficients" << cache.profile.imageResidualQuadraticCoefficients;
    storage << "total_images" << cache.profile.quality.totalImages;
    storage << "valid_images" << cache.profile.quality.validImages;
    storage << "image_width" << cache.profile.quality.imageWidth;
    storage << "image_height" << cache.profile.quality.imageHeight;
    storage << "valid_ratio_percent" << cache.profile.quality.validRatioPercent;
    storage << "reprojection_rms_px" << cache.profile.quality.reprojectionRmsPx;
    storage << "mean_affine_rms_px" << cache.profile.quality.meanAffineRmsPx;
    storage << "fx_pixels_per_mm" << cache.profile.quality.fxPixelsPerMm;
    storage << "fy_pixels_per_mm" << cache.profile.quality.fyPixelsPerMm;
    storage << "fx_fy_diff_percent" << cache.profile.quality.fxFyDiffPercent;
    storage << "principal_point_x" << cache.profile.quality.principalPointX;
    storage << "principal_point_y" << cache.profile.quality.principalPointY;
    storage << "principal_point_offset_x" << cache.profile.quality.principalPointOffsetX;
    storage << "principal_point_offset_y" << cache.profile.quality.principalPointOffsetY;
    storage << "intrinsic_model" << cache.profile.quality.intrinsicModel;
    storage << "principal_point_policy" << cache.profile.quality.principalPointPolicy;
    storage << "warning_summary" << cache.profile.quality.warningSummary;
    return true;
}

bool loadProfileFile(const std::string& pathUtf8, CalibrationProfile& profile, std::string& createdAt)
{
    profile = {};
    createdAt.clear();

    cv::FileStorage storage(pathUtf8, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        return false;
    }

    storage["profile_name"] >> profile.profileName;
    storage["source_path"] >> profile.sourcePath;
    storage["created_at"] >> createdAt;
    storage["camera_matrix"] >> profile.cameraMatrix;
    storage["distortion_coefficients"] >> profile.distortionCoefficients;
    storage["image_residual_quadratic_coefficients"] >> profile.imageResidualQuadraticCoefficients;
    storage["total_images"] >> profile.quality.totalImages;
    storage["valid_images"] >> profile.quality.validImages;
    storage["image_width"] >> profile.quality.imageWidth;
    storage["image_height"] >> profile.quality.imageHeight;
    storage["valid_ratio_percent"] >> profile.quality.validRatioPercent;
    storage["reprojection_rms_px"] >> profile.quality.reprojectionRmsPx;
    storage["mean_affine_rms_px"] >> profile.quality.meanAffineRmsPx;
    storage["fx_pixels_per_mm"] >> profile.quality.fxPixelsPerMm;
    storage["fy_pixels_per_mm"] >> profile.quality.fyPixelsPerMm;
    storage["fx_fy_diff_percent"] >> profile.quality.fxFyDiffPercent;
    storage["principal_point_x"] >> profile.quality.principalPointX;
    storage["principal_point_y"] >> profile.quality.principalPointY;
    storage["principal_point_offset_x"] >> profile.quality.principalPointOffsetX;
    storage["principal_point_offset_y"] >> profile.quality.principalPointOffsetY;
    storage["intrinsic_model"] >> profile.quality.intrinsicModel;
    storage["principal_point_policy"] >> profile.quality.principalPointPolicy;
    storage["warning_summary"] >> profile.quality.warningSummary;

    profile.valid = !profile.cameraMatrix.empty() && !profile.distortionCoefficients.empty();
    return profile.valid;
}

bool savePreviewImage(const CameraCalibrationRequest& request,
                      const telecentric::CalibParams& params,
                      const std::string& previewImagePath)
{
    const std::vector<std::string> imagePaths = resolveImagePaths(request);
    if (imagePaths.empty()) {
        return false;
    }

    cv::Mat image;
    if (!loadImageFromPath(imagePaths.front(), cv::IMREAD_COLOR, image)) {
        return false;
    }

    cv::Mat undistorted;
    if (!telecentric::UndistortImage(image, undistorted, params)) {
        return false;
    }

    return saveImageToPath(previewImagePath, undistorted);
}

cv::Size resolveImageSize(const CameraCalibrationRequest& request,
                          const telecentric::CalibQualityReport& quality)
{
    if (quality.image_width > 0 && quality.image_height > 0) {
        return {quality.image_width, quality.image_height};
    }

    const std::vector<std::string> imagePaths = resolveImagePaths(request);
    if (imagePaths.empty()) {
        return {};
    }

    cv::Mat image;
    if (!loadImageFromPath(imagePaths.front(), cv::IMREAD_GRAYSCALE, image)) {
        return {};
    }

    return image.size();
}

class DefaultCameraCalibrationService final : public CameraCalibrationService {
public:
    CameraCalibrationResult runCalibration(const CameraCalibrationRequest& request) override;
    bool loadCachedResult(const std::string& cacheDir, CalibrationResultCache& cache) override;
    bool saveCachedResult(const CalibrationResultCache& cache, const std::string& cacheDir) override;
};

CameraCalibrationResult DefaultCameraCalibrationService::runCalibration(const CameraCalibrationRequest& request)
{
    CameraCalibrationResult result;

    const CalibrationResultPathSet pathSet = buildPathSet(request.sessionName, request.outputDir, request.cacheDir);
    result.cache.paths = toCachePaths(pathSet);

    if (request.preferCachedResult && loadCachedResult(result.cache.paths.cacheDir, result.cache)) {
        result.ok = true;
        result.loadedFromCache = true;
        result.profile = result.cache.profile;
        result.message = "已加载缓存的标定结果。";
        emitLog(request.callbacks, result.message);
        emitProgress(request.callbacks, 1, 1);
        return result;
    }

    const std::string imageDirectory = resolveImageDirectory(request);
    if (imageDirectory.empty()) {
        result.message = "标定失败：请提供标定图像目录，或传入来自同一目录的图像列表。";
        emitLog(request.callbacks, result.message);
        return result;
    }

    const std::vector<std::string> imagePaths = resolveImagePaths(request);
    if (imagePaths.empty()) {
        result.message = "标定失败：未找到可用的标定图像。";
        emitLog(request.callbacks, result.message);
        return result;
    }

    if (!ensureCalibrationResultDirectories(pathSet)) {
        result.message = "标定失败：无法创建标定结果目录。";
        emitLog(request.callbacks, result.message);
        return result;
    }

    telecentric::CalibOptions options;
    options.image_folder = imageDirectory;
    options.init_path = result.cache.paths.initPath;
    options.rows = request.boardSpec.rows;
    options.cols = request.boardSpec.cols;
    options.pitch_mm = request.boardSpec.pitchMm;
    options.roi_radius_px = request.boardSpec.roiRadiusPx;
    options.min_valid_images = request.boardSpec.minValidImages;
    options.enforce_square_pixels = request.boardSpec.enforceSquarePixels;
    options.lock_principal_point_to_image_center = request.boardSpec.lockPrincipalPointToImageCenter;
    options.enable_image_residual_compensation = request.boardSpec.enableImageResidualCompensation;
    options.image_residual_prior_sigma_px = request.boardSpec.imageResidualPriorSigmaPx;
    options.image_residual_max_coeff_px = request.boardSpec.imageResidualMaxCoeffPx;
    options.enable_board_warp_compensation = request.boardSpec.enableBoardWarpCompensation;
    options.board_warp_prior_sigma_mm = request.boardSpec.boardWarpPriorSigmaMm;
    options.board_warp_max_offset_mm = request.boardSpec.boardWarpMaxOffsetMm;
    options.enable_board_point_compensation = request.boardSpec.enableBoardPointCompensation;
    options.board_point_prior_sigma_mm = request.boardSpec.boardPointPriorSigmaMm;
    options.board_point_smooth_sigma_mm = request.boardSpec.boardPointSmoothSigmaMm;
    options.board_point_max_offset_mm = request.boardSpec.boardPointMaxOffsetMm;

    telecentric::CalibParams params;
    telecentric::CalibQualityReport quality;
    std::string errorMessage;

    auto visCallback = [&](const std::string&,
                           const int index,
                           const int total,
                           const std::string&,
                           const bool,
                           const cv::Mat& image) {
        emitProgress(request.callbacks, index, total);
        emitPreview(request.callbacks, index, total, image);
    };

    auto logCallback = [&](const std::string& line) {
        emitLog(request.callbacks, line);
    };

    if (!telecentric::RunTelecentricCalibration(
            options,
            params,
            &errorMessage,
            telecentric::CalibVisCallback(visCallback),
            &quality,
            telecentric::CalibLogCallback(logCallback))) {
        result.message = errorMessage.empty() ? std::string("标定失败。") : errorMessage;
        emitLog(request.callbacks, result.message);
        return result;
    }

    result.cache.profile = buildProfile(defaultProfileName(pathSet, request.sessionName),
                                        result.cache.paths.initPath,
                                        params,
                                        &quality);
    result.cache.createdAt = quality.created_at;
    result.cache.valid = result.cache.profile.valid;

    const cv::Size imageSize = resolveImageSize(request, quality);
    if (imageSize.width > 0 && imageSize.height > 0) {
        telecentric::BuildUndistortMaps(imageSize, result.cache.undistortMapX, result.cache.undistortMapY, params);
    }

    savePreviewImage(request, params, result.cache.paths.previewImagePath);

    if (request.persistCache && !saveCachedResult(result.cache, result.cache.paths.cacheDir)) {
        result.ok = true;
        result.profile = result.cache.profile;
        result.message = "标定已完成，但扩展缓存文件保存失败。";
        emitLog(request.callbacks, result.message);
        return result;
    }

    result.ok = true;
    result.profile = result.cache.profile;
    result.message = "标定完成。";
    emitLog(request.callbacks, result.message);
    return result;
}

bool DefaultCameraCalibrationService::loadCachedResult(const std::string& cacheDir, CalibrationResultCache& cache)
{
    cache = {};
    if (cacheDir.empty()) {
        return false;
    }

    const CalibrationResultPathSet pathSet = buildPathSetForCacheDir(cacheDir);
    cache.paths = toCachePaths(pathSet);

    std::string createdAt;
    const bool loadedProfile = loadProfileFile(cache.paths.profilePath, cache.profile, createdAt);

    telecentric::CalibQualityReport quality;
    const bool loadedQuality =
        telecentric::LoadCalibQualityReport(cache.paths.qualityReportPath, quality, nullptr);

    if (!loadedProfile) {
        telecentric::CalibParams params;
        std::string errorMessage;
        if (!telecentric::LoadFromInitTxt(cache.paths.initPath, params, &errorMessage)) {
            return false;
        }

        cache.profile = buildProfile(defaultProfileName(pathSet, std::string()),
                                     cache.paths.initPath,
                                     params,
                                     loadedQuality ? &quality : nullptr);
    } else if (loadedQuality) {
        cache.profile.quality = toQualitySummary(quality);
    }

    loadMatrixFile(cache.paths.undistortMapXPath, "map_x", cache.undistortMapX);
    loadMatrixFile(cache.paths.undistortMapYPath, "map_y", cache.undistortMapY);

    if (createdAt.empty() && loadedQuality) {
        createdAt = quality.created_at;
    }
    cache.createdAt = createdAt;
    cache.valid = cache.profile.valid;
    return cache.valid;
}

bool DefaultCameraCalibrationService::saveCachedResult(const CalibrationResultCache& cache, const std::string& cacheDir)
{
    if (!cache.profile.valid) {
        return false;
    }

    const CalibrationResultPathSet pathSet =
        buildPathSetForCacheDir(!cacheDir.empty() ? cacheDir : cache.paths.cacheDir);
    if (!ensureCalibrationResultDirectories(pathSet)) {
        return false;
    }

    CalibrationResultCache normalizedCache = cache;
    normalizedCache.paths = toCachePaths(pathSet);
    normalizedCache.profile.sourcePath = normalizedCache.paths.initPath;

    const telecentric::CalibParams params = toTelecentricParams(normalizedCache.profile);
    std::string errorMessage;
    if (!telecentric::SaveToInitTxt(normalizedCache.paths.initPath, params, &errorMessage)) {
        return false;
    }

    if (!saveProfileFile(normalizedCache.paths.profilePath, normalizedCache)) {
        return false;
    }

    const telecentric::CalibQualityReport report = toTelecentricQualityReport(normalizedCache);
    if ((report.total_images > 0 || report.valid_images > 0 || report.fx_px_per_mm > 0.0) &&
        !telecentric::SaveCalibQualityReport(normalizedCache.paths.qualityReportPath, report, &errorMessage)) {
        return false;
    }

    if (!normalizedCache.undistortMapX.empty() &&
        !saveMatrixFile(normalizedCache.paths.undistortMapXPath, "map_x", normalizedCache.undistortMapX)) {
        return false;
    }
    if (!normalizedCache.undistortMapY.empty() &&
        !saveMatrixFile(normalizedCache.paths.undistortMapYPath, "map_y", normalizedCache.undistortMapY)) {
        return false;
    }

    return true;
}

} // namespace

CameraCalibrationServicePtr createCameraCalibrationService()
{
    return std::make_shared<DefaultCameraCalibrationService>();
}

} // namespace pinjie
