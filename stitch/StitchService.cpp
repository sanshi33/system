#include "StitchService.h"

#include "cad_design/DesignProfileAlignment.h"
#include "DebugVis.h"
#include "IO.h"
#include "Pipeline.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

namespace stitch {

namespace {

bool isCancelled(const StitchCallbacks& callbacks)
{
    return callbacks.isCancelled && callbacks.isCancelled();
}

void emitLog(const StitchCallbacks& callbacks, const std::string& message)
{
    if (callbacks.onLog) {
        callbacks.onLog(message);
    }
}

bool nearlyEqual(double lhs, double rhs, double tolerance = 1e-9)
{
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= tolerance * scale;
}

bool sameEdgeConfig(const EdgeDetectConfig& lhs, const EdgeDetectConfig& rhs)
{
    return nearlyEqual(lhs.cannyLow, rhs.cannyLow) &&
           nearlyEqual(lhs.cannyHigh, rhs.cannyHigh) &&
           lhs.subpixWindow == rhs.subpixWindow &&
           nearlyEqual(lhs.subpixSigma, rhs.subpixSigma) &&
           lhs.minWarnPoints == rhs.minWarnPoints &&
           lhs.enablePointFiltering == rhs.enablePointFiltering &&
           nearlyEqual(lhs.filterConfidenceQuantile, rhs.filterConfidenceQuantile) &&
           nearlyEqual(lhs.filterGradientQuantile, rhs.filterGradientQuantile) &&
           lhs.filterLocalLinearWindowRadius == rhs.filterLocalLinearWindowRadius &&
           nearlyEqual(lhs.filterHampelSigma, rhs.filterHampelSigma) &&
           nearlyEqual(lhs.filterHampelMinScale, rhs.filterHampelMinScale);
}

bool sameDirectionVector(const std::vector<MotionPriorDirection>& lhs,
                         const std::vector<MotionPriorDirection>& rhs)
{
    return lhs == rhs;
}

bool samePointVector(const std::vector<cv::Point2d>& lhs,
                     const std::vector<cv::Point2d>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!nearlyEqual(lhs[i].x, rhs[i].x) || !nearlyEqual(lhs[i].y, rhs[i].y)) {
            return false;
        }
    }
    return true;
}

bool samePipelineConfig(const StitchPipelineConfig& lhs, const StitchPipelineConfig& rhs)
{
    return nearlyEqual(lhs.expectedOverlapRatio, rhs.expectedOverlapRatio) &&
           lhs.directionConstraint == rhs.directionConstraint &&
           sameDirectionVector(lhs.stepDirectionConstraints, rhs.stepDirectionConstraints) &&
           samePointVector(lhs.stepTranslationPriorsPx, rhs.stepTranslationPriorsPx) &&
           nearlyEqual(lhs.baseSearchRange, rhs.baseSearchRange) &&
           nearlyEqual(lhs.approxShiftRatio, rhs.approxShiftRatio) &&
           nearlyEqual(lhs.rotationSearchMinDeg, rhs.rotationSearchMinDeg) &&
           nearlyEqual(lhs.rotationSearchMaxDeg, rhs.rotationSearchMaxDeg) &&
           nearlyEqual(lhs.rotationSearchStepDeg, rhs.rotationSearchStepDeg) &&
           nearlyEqual(lhs.tangentResidualCostWeight, rhs.tangentResidualCostWeight) &&
           nearlyEqual(lhs.tangentCorrelationCostWeight, rhs.tangentCorrelationCostWeight) &&
           lhs.generateDebugVisualization == rhs.generateDebugVisualization &&
           lhs.enableDesignComparison == rhs.enableDesignComparison &&
           lhs.designEvaluateProfileForm == rhs.designEvaluateProfileForm &&
           lhs.designReverseZ == rhs.designReverseZ &&
           lhs.designUseLeftEndpointAnchor == rhs.designUseLeftEndpointAnchor &&
           lhs.designAnchorRadialToLeftEndpoint == rhs.designAnchorRadialToLeftEndpoint &&
           lhs.designEnableBestFitTranslation == rhs.designEnableBestFitTranslation &&
           nearlyEqual(lhs.designBestFitDzMinMm, rhs.designBestFitDzMinMm) &&
           nearlyEqual(lhs.designBestFitDzMaxMm, rhs.designBestFitDzMaxMm) &&
           nearlyEqual(lhs.designBestFitDrMinMm, rhs.designBestFitDrMinMm) &&
           nearlyEqual(lhs.designBestFitDrMaxMm, rhs.designBestFitDrMaxMm) &&
           nearlyEqual(lhs.designBestFitStepMm, rhs.designBestFitStepMm) &&
           lhs.designEnableBestFitRotation == rhs.designEnableBestFitRotation &&
           nearlyEqual(lhs.designBestFitRotationMinDeg, rhs.designBestFitRotationMinDeg) &&
           nearlyEqual(lhs.designBestFitRotationMaxDeg, rhs.designBestFitRotationMaxDeg) &&
           nearlyEqual(lhs.designBestFitRotationStepDeg, rhs.designBestFitRotationStepDeg) &&
           nearlyEqual(lhs.designPixelSizeMm, rhs.designPixelSizeMm) &&
           lhs.designInvertY == rhs.designInvertY &&
           lhs.designUseUpperEnvelope == rhs.designUseUpperEnvelope &&
           nearlyEqual(lhs.designUpperEnvelopeQuantile, rhs.designUpperEnvelopeQuantile) &&
           nearlyEqual(lhs.designProfileBinWidthPx, rhs.designProfileBinWidthPx) &&
           lhs.designFilterEndFaceEdges == rhs.designFilterEndFaceEdges &&
           lhs.designEnableProfileOutlierFilter == rhs.designEnableProfileOutlierFilter &&
           nearlyEqual(lhs.designProfileOutlierSigma, rhs.designProfileOutlierSigma) &&
           nearlyEqual(lhs.designMaxAbsSlopeForGeneratrix, rhs.designMaxAbsSlopeForGeneratrix) &&
           lhs.designSlopeWindow == rhs.designSlopeWindow &&
           nearlyEqual(lhs.designTrimLeftAfterEndpointMm, rhs.designTrimLeftAfterEndpointMm) &&
           nearlyEqual(lhs.designTrimRightMm, rhs.designTrimRightMm);
}

bool isCompleteStitchingCache(const StitchRunCache& cache)
{
    if (!cache.hasStitching() || !cache.hasLoadedImages()) {
        return false;
    }

    if (cache.loadedImages.size() <= 1) {
        return true;
    }

    return cache.stitching.steps.size() + 1 == cache.loadedImages.size();
}

bool canReuseLoadedImages(const std::shared_ptr<const StitchRunCache>& cache,
                          const StitchRunRequest& request)
{
    return cache && cache->imagePaths == request.imagePaths && cache->hasLoadedImages();
}

bool canReusePreprocessedEdges(const std::shared_ptr<const StitchRunCache>& cache,
                               const StitchRunRequest& request)
{
    return canReuseLoadedImages(cache, request) &&
           sameEdgeConfig(cache->edgeConfig, request.edgeConfig) &&
           cache->hasPreprocessedEdges();
}

bool canReuseStitching(const std::shared_ptr<const StitchRunCache>& cache,
                       const StitchRunRequest& request)
{
    return canReusePreprocessedEdges(cache, request) &&
           samePipelineConfig(cache->pipelineConfig, request.pipelineConfig) &&
           isCompleteStitchingCache(*cache);
}

std::string formatReviewValue(const double value)
{
    if (!std::isfinite(value)) {
        return {};
    }
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(6);
    stream << value;
    return stream.str();
}

void appendReviewRow(std::ostringstream& stream,
                     const std::string& check,
                     const bool pass,
                     const double value,
                     const double threshold,
                     const std::string& message)
{
    stream << check << ","
           << (pass ? "PASS" : "WARN") << ","
           << formatReviewValue(value) << ","
           << formatReviewValue(threshold) << ","
           << message << "\n";
}

std::string buildQualityReviewCsv(const pinjie::cad_design::DesignAlignmentResult& designResult,
                                  const StitchingResult& stitching)
{
    std::ostringstream stream;
    stream << "check,status,value,threshold,message\n";

    const auto& summary = designResult.summary;
    appendReviewRow(stream,
                    "absolute_filtered_rmse_um",
                    summary.absoluteFilteredStats.rmseUm <= 80.0,
                    summary.absoluteFilteredStats.rmseUm,
                    80.0,
                    "Filtered absolute signed-distance RMSE for cumulative reconstruction error.");
    appendReviewRow(stream,
                    "form_rmse_um",
                    summary.profileStats.rmseUm <= 30.0,
                    summary.profileStats.rmseUm,
                    30.0,
                    "Demeaned profile-form RMSE after outlier filtering.");
    appendReviewRow(stream,
                    "outlier_ratio",
                    summary.outlierRatio <= 0.03,
                    summary.outlierRatio,
                    0.03,
                    "Profile outlier ratio after Hampel filtering.");
    appendReviewRow(stream,
                    "bestfit_abs_dz_mm",
                    std::abs(summary.dzMm) <= 1.0,
                    std::abs(summary.dzMm),
                    1.0,
                    "Best-fit axial shift; large values mean cumulative drift is being absorbed.");
    appendReviewRow(stream,
                    "bestfit_abs_dtheta_deg",
                    std::abs(summary.dThetaDeg) <= 0.05,
                    std::abs(summary.dThetaDeg),
                    0.05,
                    "Best-fit rotation; large values mean angular drift is being absorbed.");

    double worstNormalRmsePx = 0.0;
    std::size_t worstStep = 0;
    for (const StitchStepRecord& step : stitching.steps) {
        const ResidualStatistics& normal =
            step.transform.metrics.normalInlier.valid() ? step.transform.metrics.normalInlier
                                                        : step.transform.metrics.normalAll;
        if (normal.valid() && normal.rmse > worstNormalRmsePx) {
            worstNormalRmsePx = normal.rmse;
            worstStep = step.stepIndex;
        }
    }

    appendReviewRow(stream,
                    "worst_step_normal_rmse_px",
                    worstNormalRmsePx <= 0.25,
                    worstNormalRmsePx,
                    0.25,
                    "Worst single-step inlier normal RMSE; worst_step=" + std::to_string(worstStep) + ".");
    return stream.str();
}

bool saveImageArtifact(const std::string& path,
                       const cv::Mat& image,
                       const std::string& label,
                       StitchRunResult& runResult,
                       const StitchCallbacks& callbacks);

void emitCachedPreprocessVisualizations(const std::vector<cv::Mat>& images,
                                        const std::vector<EdgeVariants>& edges,
                                        const StitchCallbacks& callbacks)
{
    if (!callbacks.onImage || images.size() != edges.size()) {
        return;
    }

    for (std::size_t i = 0; i < images.size(); ++i) {
        const cv::Mat preview = buildPreprocessVisualization(images[i],
                                                             edges[i].raw,
                                                             static_cast<int>(i + 1),
                                                             static_cast<int>(images.size()));
        callbacks.onImage("preprocess_preview", i + 1, images.size(), preview);
    }
}

void emitCachedRegistrationArtifacts(const std::vector<cv::Mat>& images,
                                     const std::vector<EdgeVariants>& edges,
                                     const StitchingResult& stitching,
                                     const StitchPipelineConfig& config,
                                     const StitchCallbacks& callbacks)
{
    const std::size_t totalSteps = stitching.steps.size();
    for (std::size_t i = 0; i < stitching.steps.size(); ++i) {
        const StitchStepRecord& step = stitching.steps[i];

        if (callbacks.onProgress) {
            callbacks.onProgress("stitch", i + 1, totalSteps);
        }

        if (callbacks.onStepFinished) {
            callbacks.onStepFinished(step);
        }

        if (!config.generateDebugVisualization || !callbacks.onImage) {
            continue;
        }

        if (step.referenceImageIndex >= images.size() ||
            step.targetImageIndex >= images.size() ||
            step.referenceImageIndex >= edges.size() ||
            step.targetImageIndex >= edges.size()) {
            continue;
        }

        const auto& referenceEdges = edges[step.referenceImageIndex].ordered(step.transform.axis);
        const auto& targetEdges = edges[step.targetImageIndex].ordered(step.transform.axis);
        const cv::Mat debugImage = buildDebugStepVisualization(images[step.referenceImageIndex],
                                                               images[step.targetImageIndex],
                                                               referenceEdges,
                                                               targetEdges,
                                                               step.transform,
                                                               static_cast<int>(step.stepIndex));
        callbacks.onImage("debug_step", step.stepIndex, totalSteps, debugImage);
    }
}

bool savePanoramaIfRequested(const StitchRunRequest& request,
                             StitchRunResult& runResult,
                             const StitchCallbacks& callbacks)
{
    if (request.panoramaOutputPath.empty() || runResult.stitching.canvas.empty()) {
        return true;
    }

    if (!saveImageToPath(request.panoramaOutputPath, runResult.stitching.canvas)) {
        runResult.message = "全景图保存失败：" + request.panoramaOutputPath;
        emitLog(callbacks, "[错误] " + runResult.message);
        return false;
    }

    emitLog(callbacks, "[信息] 已保存全景图：" + request.panoramaOutputPath);
    return true;
}

bool saveCsvIfRequested(const StitchRunRequest& request,
                        StitchRunResult& runResult,
                        const StitchCallbacks& callbacks)
{
    if (!request.saveStepSummaryCsv || request.csvOutputPath.empty() || runResult.csvText.empty()) {
        return true;
    }

    if (!writeTextFileToPath(request.csvOutputPath, runResult.csvText)) {
        runResult.message = "CSV 保存失败：" + request.csvOutputPath;
        emitLog(callbacks, "[错误] " + runResult.message);
        return false;
    }

    emitLog(callbacks, "[信息] 已保存拼接数据：" + request.csvOutputPath);
    return true;
}

bool saveAdditionalCsvArtifactsIfRequested(const StitchRunRequest& request,
                                           const std::vector<EdgeVariants>& edges,
                                           StitchRunResult& runResult,
                                           const StitchCallbacks& callbacks)
{
    if (request.saveContourPointsCsv && !request.contourPointsCsvOutputPath.empty()) {
        const std::string contourPointsCsv =
            buildContourPointCsv(edges, runResult.stitching.imageTransforms);
        if (!writeTextFileToPath(request.contourPointsCsvOutputPath, contourPointsCsv)) {
            runResult.message = "轮廓点 CSV 保存失败：" + request.contourPointsCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[信息] 已保存轮廓点 CSV：" + request.contourPointsCsvOutputPath);

        if (!request.originContourOverlayCsvOutputPath.empty()) {
            const std::string originContourOverlayCsv =
                buildOriginContourOverlayCsv(edges, runResult.stitching.imageTransforms);
            if (!writeTextFileToPath(request.originContourOverlayCsvOutputPath, originContourOverlayCsv)) {
                runResult.message = "Origin 轮廓叠加 CSV 保存失败：" + request.originContourOverlayCsvOutputPath;
                emitLog(callbacks, "[错误] " + runResult.message);
                return false;
            }
            emitLog(callbacks, "[信息] 已保存 Origin 轮廓叠加 CSV：" + request.originContourOverlayCsvOutputPath);
        }
    }

    if (request.saveStitchedContourProfileCsv && !request.stitchedContourProfileCsvOutputPath.empty()) {
        const std::string contourProfileCsv =
            buildStitchedContourProfileCsv(edges, runResult.stitching.imageTransforms);
        if (!writeTextFileToPath(request.stitchedContourProfileCsvOutputPath, contourProfileCsv)) {
            runResult.message = "拼接后轮廓剖面 CSV 保存失败：" + request.stitchedContourProfileCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[信息] 已保存拼接后轮廓剖面 CSV：" + request.stitchedContourProfileCsvOutputPath);
    }

    if (request.saveTangentStepCsv && !request.tangentStepCsvOutputPath.empty()) {
        const std::string tangentStepCsv = buildTangentCorrelationStepCsv(runResult.stitching.steps);
        if (!writeTextFileToPath(request.tangentStepCsvOutputPath, tangentStepCsv)) {
            runResult.message = "切向相关 CSV 保存失败：" + request.tangentStepCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[信息] 已保存切向相关 CSV：" + request.tangentStepCsvOutputPath);
    }

    if (request.saveNormalErrorProfileCsv && !request.normalErrorProfileCsvOutputPath.empty()) {
        const std::string normalErrorCsv = buildNormalErrorProfileCsv(runResult.stitching.steps);
        if (!writeTextFileToPath(request.normalErrorProfileCsvOutputPath, normalErrorCsv)) {
            runResult.message = "法向误差剖面 CSV 保存失败：" + request.normalErrorProfileCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[信息] 已保存法向误差剖面 CSV：" + request.normalErrorProfileCsvOutputPath);
    }

    if (request.saveTangentProfileCsv && !request.tangentProfileCsvOutputPath.empty()) {
        const std::string tangentProfileCsv = buildTangentProfileCompareCsv(runResult.stitching.steps);
        if (!writeTextFileToPath(request.tangentProfileCsvOutputPath, tangentProfileCsv)) {
            runResult.message = "轮廓波动分析 CSV 保存失败：" + request.tangentProfileCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[信息] 已保存轮廓波动分析 CSV：" + request.tangentProfileCsvOutputPath);

        if (!request.originTangentPointMetricsCsvOutputPath.empty()) {
            const std::string originTangentMetricsCsv =
                buildOriginTangentPointMetricsCsv(runResult.stitching.steps);
            if (!writeTextFileToPath(request.originTangentPointMetricsCsvOutputPath, originTangentMetricsCsv)) {
                runResult.message =
                    "Origin 轮廓波动 CSV 保存失败：" + request.originTangentPointMetricsCsvOutputPath;
                emitLog(callbacks, "[错误] " + runResult.message);
                return false;
            }
            emitLog(callbacks,
                    "[信息] 已保存 Origin 轮廓波动 CSV：" + request.originTangentPointMetricsCsvOutputPath);
        }
    }

    if (!request.alignmentCandidateDiagnosticsCsvOutputPath.empty()) {
        const std::string candidateDiagnosticsCsv =
            buildAlignmentCandidateDiagnosticsCsv(runResult.stitching.steps);
        if (!writeTextFileToPath(request.alignmentCandidateDiagnosticsCsvOutputPath, candidateDiagnosticsCsv)) {
            runResult.message =
                "Alignment candidate diagnostics CSV 保存失败：" +
                request.alignmentCandidateDiagnosticsCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved alignment candidate diagnostics CSV: " +
                request.alignmentCandidateDiagnosticsCsvOutputPath);
    }

    return true;
}

bool saveDesignComparisonArtifacts(const StitchRunRequest& request,
                                   const std::vector<EdgeVariants>& edges,
                                   StitchRunResult& runResult,
                                   const StitchCallbacks& callbacks)
{
    if (!request.pipelineConfig.enableDesignComparison) {
        return true;
    }

    if (request.designErrorProfileCsvOutputPath.empty() &&
        request.designErrorSummaryCsvOutputPath.empty() &&
        request.designComparisonOverlayOutputPath.empty() &&
        request.qualityReviewCsvOutputPath.empty()) {
        return true;
    }

    const pinjie::cad_design::DesignAlignmentResult designResult =
        pinjie::cad_design::compareMeasuredProfileToDesign(edges, runResult.stitching.imageTransforms, request.pipelineConfig);

    if (!designResult.ok) {
        emitLog(callbacks, "[Warning] Design profile comparison skipped: " + designResult.message);
        return true;
    }

    if (!request.designErrorProfileCsvOutputPath.empty()) {
        if (!writeTextFileToPath(request.designErrorProfileCsvOutputPath, designResult.profileCsvText)) {
            runResult.message =
                "Failed to save design_error_profile.csv: " + request.designErrorProfileCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved design profile error CSV: " + request.designErrorProfileCsvOutputPath);
    }

    if (!request.designErrorSummaryCsvOutputPath.empty()) {
        if (!writeTextFileToPath(request.designErrorSummaryCsvOutputPath, designResult.summaryCsvText)) {
            runResult.message =
                "Failed to save design_error_summary.csv: " + request.designErrorSummaryCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved design profile summary CSV: " + request.designErrorSummaryCsvOutputPath);
    }

    if (!request.qualityReviewCsvOutputPath.empty()) {
        const std::string qualityReviewCsv = buildQualityReviewCsv(designResult, runResult.stitching);
        if (!writeTextFileToPath(request.qualityReviewCsvOutputPath, qualityReviewCsv)) {
            runResult.message =
                "Failed to save quality_review.csv: " + request.qualityReviewCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks, "[Info] Saved automatic quality review CSV: " + request.qualityReviewCsvOutputPath);
    }

    if (!request.designComparisonOverlayOutputPath.empty()) {
        const cv::Mat overlay = pinjie::cad_design::buildDesignComparisonPlot(designResult);
        if (!saveImageArtifact(request.designComparisonOverlayOutputPath,
                               overlay,
                               "Design comparison overlay",
                               runResult,
                               callbacks)) {
            return false;
        }
    }

    return true;
}

bool saveImageArtifact(const std::string& path,
                       const cv::Mat& image,
                       const std::string& label,
                       StitchRunResult& runResult,
                       const StitchCallbacks& callbacks)
{
    if (path.empty() || image.empty()) {
        return true;
    }

    if (!saveImageToPath(path, image)) {
        runResult.message = label + "保存失败：" + path;
        emitLog(callbacks, "[错误] " + runResult.message);
        return false;
    }

    emitLog(callbacks, "[信息] 已保存" + label + "：" + path);
    return true;
}

bool saveVisualArtifactsIfRequested(const StitchRunRequest& request,
                                    const std::vector<EdgeVariants>& edges,
                                    StitchRunResult& runResult,
                                    const StitchCallbacks& callbacks)
{
    const cv::Mat contourOverlay = buildStitchedContourOverlay(runResult.stitching.canvas,
                                                               edges,
                                                               runResult.stitching.imageTransforms);
    if (!saveImageArtifact(request.contourOverlayOutputPath,
                           contourOverlay,
                           "Origin 轮廓叠加图",
                           runResult,
                           callbacks)) {
        return false;
    }

    const cv::Mat contourProfilePlot =
        buildStitchedContourProfilePlot(edges, runResult.stitching.imageTransforms);
    if (!saveImageArtifact(request.stitchedContourProfilePlotOutputPath,
                           contourProfilePlot,
                           "Stitched contour profile",
                           runResult,
                           callbacks)) {
        return false;
    }

    const cv::Mat tangentAllPlot = buildTangentCorrelationPlot(runResult.stitching.steps, false);
    if (!saveImageArtifact(request.tangentCorrelationAllOutputPath,
                           tangentAllPlot,
                           "切向相关总览图",
                           runResult,
                           callbacks)) {
        return false;
    }

    const cv::Mat tangentInlierPlot = buildTangentCorrelationPlot(runResult.stitching.steps, true);
    return saveImageArtifact(request.tangentCorrelationInlierOutputPath,
                             tangentInlierPlot,
                             "切向相关内点图",
                             runResult,
                             callbacks);
}

std::string buildSuccessMessage(const StitchRunResult& runResult)
{
    switch (runResult.mode) {
    case StitchRunMode::Acquisition:
        return "采集模块运行成功，共加载 " + std::to_string(runResult.loadedImageCount) + " 张图像。";
    case StitchRunMode::Processing:
        return "处理模块运行成功，共生成 " +
               std::to_string(runResult.preprocessedImageCount) + " 张预处理预览图。";
    case StitchRunMode::Registration:
        return "配准模块运行成功，共完成 " +
               std::to_string(runResult.stitching.steps.size()) + " 个拼接步骤。";
    case StitchRunMode::Report:
        return "结果模块运行成功。";
    case StitchRunMode::Full:
    default:
        return "完整流程运行成功。";
    }
}

} // namespace

StitchRunResult runStitching(const StitchRunRequest& request,
                             StitchRunMode runMode,
                             const StitchCallbacks& callbacks)
{
    StitchRunResult runResult;
    runResult.mode = runMode;
    auto cache = std::make_shared<StitchRunCache>();
    cache->imagePaths = request.imagePaths;
    cache->edgeConfig = request.edgeConfig;
    cache->pipelineConfig = request.pipelineConfig;
    runResult.cache = cache;

    if (request.imagePaths.empty()) {
        runResult.message = "未提供输入图像。";
        emitLog(callbacks, "[错误] " + runResult.message);
        return runResult;
    }

    if (isCancelled(callbacks)) {
        runResult.message = "运行在开始前被取消。";
        emitLog(callbacks, "[信息] " + runResult.message);
        return runResult;
    }

    StitchCallbacks effectiveCallbacks = callbacks;
    const ImageCallback userImageCallback = callbacks.onImage;

    effectiveCallbacks.onImage = [&](const std::string& stage,
                                     std::size_t index,
                                     std::size_t total,
                                     const cv::Mat& image) {
        if (!request.debugImageOutputDir.empty() && !image.empty()) {
            const std::string imagePath =
                request.debugImageOutputDir + "/" + stage + "_" + std::to_string(index) + ".png";
            if (!saveImageToPath(imagePath, image)) {
                emitLog(callbacks, "[警告] 调试图保存失败：" + imagePath);
            } else {
                emitLog(callbacks, "[信息] 已保存调试图：" + imagePath);
            }
        }

        if (userImageCallback) {
            userImageCallback(stage, index, total, image);
        }
    };

    std::vector<cv::Mat> images;
    if (canReuseLoadedImages(request.previousCache, request)) {
        images = request.previousCache->loadedImages;
        emitLog(callbacks, "[信息] 已复用缓存的采集结果。");
    } else {
        images = loadInputImages(request.imagePaths, effectiveCallbacks);
    }

    cache->loadedImages = images;
    runResult.loadedImageCount = images.size();
    if (images.empty()) {
        runResult.message = isCancelled(callbacks) ? "图像加载阶段已取消。"
                                                   : "输入图像加载失败。";
        return runResult;
    }

    if (runMode == StitchRunMode::Acquisition) {
        runResult.ok = true;
        runResult.message = buildSuccessMessage(runResult);
        return runResult;
    }

    std::vector<EdgeVariants> edges;
    if (canReusePreprocessedEdges(request.previousCache, request)) {
        edges = request.previousCache->preprocessedEdges;
        emitLog(callbacks, "[信息] 已复用缓存的处理结果。");
    } else {
        edges = preprocessAllImages(images, request.edgeConfig, effectiveCallbacks);
    }

    cache->preprocessedEdges = edges;
    runResult.preprocessedImageCount = edges.size();
    if (edges.size() != images.size()) {
        runResult.message = isCancelled(callbacks) ? "预处理阶段已取消。"
                                                   : "边缘预处理失败。";
        emitLog(callbacks, "[错误] " + runResult.message);
        return runResult;
    }

    if (runMode != StitchRunMode::Report) {
        emitCachedPreprocessVisualizations(images, edges, effectiveCallbacks);
    }

    if (runMode == StitchRunMode::Processing) {
        runResult.ok = true;
        runResult.message = buildSuccessMessage(runResult);
        return runResult;
    }

    if (canReuseStitching(request.previousCache, request)) {
        runResult.stitching = request.previousCache->stitching;
        runResult.csvText = request.previousCache->csvText.empty()
                                ? buildStitchingCsv(runResult.stitching.steps)
                                : request.previousCache->csvText;
        emitLog(callbacks, "[信息] 已复用缓存的配准结果。");
        if (runMode != StitchRunMode::Report) {
            emitCachedRegistrationArtifacts(images, edges, runResult.stitching, request.pipelineConfig, effectiveCallbacks);
        }
    } else {
        runResult.stitching = runStitchingPipeline(images, edges, request.pipelineConfig, effectiveCallbacks);
        if (!runResult.stitching.success()) {
            runResult.message = isCancelled(callbacks) ? "配准拼接阶段已取消。"
                                                       : "配准拼接失败。";
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }

        runResult.csvText = buildStitchingCsv(runResult.stitching.steps);
    }

    cache->stitching = runResult.stitching;
    cache->csvText = runResult.csvText;

    if (!savePanoramaIfRequested(request, runResult, callbacks)) {
        return runResult;
    }

    if (!saveCsvIfRequested(request, runResult, callbacks)) {
        return runResult;
    }

    if (!saveAdditionalCsvArtifactsIfRequested(request, edges, runResult, callbacks)) {
        return runResult;
    }

    if (!saveDesignComparisonArtifacts(request, edges, runResult, callbacks)) {
        return runResult;
    }

    if (!saveVisualArtifactsIfRequested(request, edges, runResult, callbacks)) {
        return runResult;
    }

    runResult.ok = true;
    runResult.message = buildSuccessMessage(runResult);
    return runResult;
}

} // namespace stitch
