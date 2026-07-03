#include "StitchService.h"

#include "cad_design/DesignProfileAlignment.h"
#include "cad_design/LocalSlotAnalysis.h"
#include "DebugVis.h"
#include "IO.h"
#include "Pipeline.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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

bool sameDesignProfileMetadata(const pinjie::cad_design::DesignProfileMetadata& lhs,
                               const pinjie::cad_design::DesignProfileMetadata& rhs)
{
    return lhs.sourceType == rhs.sourceType &&
           lhs.sourceName == rhs.sourceName &&
           lhs.sourcePath == rhs.sourcePath &&
           lhs.extractionMethod == rhs.extractionMethod &&
           lhs.axialAxis == rhs.axialAxis &&
           lhs.radialAxis == rhs.radialAxis &&
           lhs.sectionNormalAxis == rhs.sectionNormalAxis &&
           nearlyEqual(lhs.sectionCoordinateMm, rhs.sectionCoordinateMm) &&
           nearlyEqual(lhs.cadAxialOriginMm, rhs.cadAxialOriginMm) &&
           nearlyEqual(lhs.cadAxialDirectionSign, rhs.cadAxialDirectionSign) &&
           lhs.sampleCount == rhs.sampleCount &&
           nearlyEqual(lhs.minSMm, rhs.minSMm) &&
           nearlyEqual(lhs.maxSMm, rhs.maxSMm) &&
           nearlyEqual(lhs.minRMm, rhs.minRMm) &&
           nearlyEqual(lhs.maxRMm, rhs.maxRMm) &&
           lhs.hasCadBounds == rhs.hasCadBounds &&
           nearlyEqual(lhs.minCadXMm, rhs.minCadXMm) &&
           nearlyEqual(lhs.minCadYMm, rhs.minCadYMm) &&
           nearlyEqual(lhs.minCadZMm, rhs.minCadZMm) &&
           nearlyEqual(lhs.maxCadXMm, rhs.maxCadXMm) &&
           nearlyEqual(lhs.maxCadYMm, rhs.maxCadYMm) &&
           nearlyEqual(lhs.maxCadZMm, rhs.maxCadZMm);
}

bool sameDesignProfileSamples(const std::vector<pinjie::cad_design::DesignProfileSample>& lhs,
                              const std::vector<pinjie::cad_design::DesignProfileSample>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!nearlyEqual(lhs[i].sMm, rhs[i].sMm) ||
            !nearlyEqual(lhs[i].rMm, rhs[i].rMm) ||
            lhs[i].hasCadPoint != rhs[i].hasCadPoint ||
            !nearlyEqual(lhs[i].cadXMm, rhs[i].cadXMm) ||
            !nearlyEqual(lhs[i].cadYMm, rhs[i].cadYMm) ||
            !nearlyEqual(lhs[i].cadZMm, rhs[i].cadZMm)) {
            return false;
        }
    }
    return true;
}

double medianOfValues(std::vector<double> values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(),
                                            values.begin() + static_cast<std::ptrdiff_t>(mid));
        median = 0.5 * (median + *lower);
    }
    return median;
}

double motionPrimaryShift(const StitchStepRecord& step, double dx, double dy)
{
    return step.motionAxis == AlignmentAxis::X ? dx : dy;
}

double motionPerpendicularShift(const StitchStepRecord& step, double dx, double dy)
{
    return step.motionAxis == AlignmentAxis::X ? dy : dx;
}

double stepNormalRmsePx(const StitchStepRecord& step)
{
    const ResidualStatistics& stats =
        step.transform.metrics.normalInlier.valid() ? step.transform.metrics.normalInlier
                                                    : step.transform.metrics.normalAll;
    return stats.valid() ? stats.rmse : std::numeric_limits<double>::infinity();
}

bool isStableReferenceStep(const StitchStepRecord& step)
{
    if (step.usedMotionPriorFallback ||
        step.usedWideSearchRescue ||
        step.usedUnfilteredEdgeRescue ||
        step.usedTrajectoryPriorRescue ||
        step.usedCandidateReselect) {
        return false;
    }

    if (step.selectionMode == "image_correlation_rescue" ||
        step.selectionMode == "wide_search_rescue" ||
        step.selectionMode == "translation_prior_fallback" ||
        step.selectionMode == "trajectory_prior_clamp") {
        return false;
    }

    return std::isfinite(stepNormalRmsePx(step)) && stepNormalRmsePx(step) <= 0.25;
}

double stepOverlapRatio(const StitchStepRecord& step, double primaryShiftPx)
{
    if (!(step.primaryImageSpanPx > 1e-9)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 - primaryShiftPx / step.primaryImageSpanPx;
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
           lhs.designReverseAxial == rhs.designReverseAxial &&
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
           nearlyEqual(lhs.designTrimRightMm, rhs.designTrimRightMm) &&
            lhs.designIgnoreStepTransition == rhs.designIgnoreStepTransition &&
            nearlyEqual(lhs.designStepTransitionOriginalZMm, rhs.designStepTransitionOriginalZMm) &&
            nearlyEqual(lhs.designStepTransitionHalfWidthMm, rhs.designStepTransitionHalfWidthMm) &&
            lhs.designUseExternalProfile == rhs.designUseExternalProfile &&
            nearlyEqual(lhs.designTargetSlotWidthMm, rhs.designTargetSlotWidthMm) &&
            nearlyEqual(lhs.designTargetSlotDepthMm, rhs.designTargetSlotDepthMm) &&
            lhs.localSlotImageMode == rhs.localSlotImageMode &&
            nearlyEqual(lhs.localSlotBottomWidthMm, rhs.localSlotBottomWidthMm) &&
            nearlyEqual(lhs.localSlotPixelSizeOverrideMm, rhs.localSlotPixelSizeOverrideMm) &&
            nearlyEqual(lhs.localSlotPixelSizeScale, rhs.localSlotPixelSizeScale) &&
            lhs.localSlotMaxOutputPoints == rhs.localSlotMaxOutputPoints &&
            lhs.designUseCentralSlotImageRoi == rhs.designUseCentralSlotImageRoi &&
            nearlyEqual(lhs.designImageRoiXRatio, rhs.designImageRoiXRatio) &&
            nearlyEqual(lhs.designImageRoiYRatio, rhs.designImageRoiYRatio) &&
            nearlyEqual(lhs.designImageRoiWidthRatio, rhs.designImageRoiWidthRatio) &&
            nearlyEqual(lhs.designImageRoiHeightRatio, rhs.designImageRoiHeightRatio) &&
            sameDesignProfileMetadata(lhs.designProfileMetadata, rhs.designProfileMetadata) &&
            sameDesignProfileSamples(lhs.designExternalProfileSamples, rhs.designExternalProfileSamples);
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

std::string lowerAscii(std::string value)
{
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return value;
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    bool inQuotes = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (ch == '"') {
            if (inQuotes && index + 1 < line.size() && line[index + 1] == '"') {
                cell.push_back('"');
                ++index;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch == ',' && !inQuotes) {
            cells.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    cells.push_back(cell);
    return cells;
}

double parseReviewDouble(const std::string& text)
{
    if (text.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    try {
        return std::stod(text);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

std::string csvValueFor(const std::vector<std::string>& headers,
                        const std::vector<std::string>& cells,
                        const std::string& name)
{
    const auto it = std::find(headers.begin(), headers.end(), name);
    if (it == headers.end()) {
        return {};
    }
    const std::size_t index = static_cast<std::size_t>(it - headers.begin());
    return index < cells.size() ? cells[index] : std::string{};
}

struct FeatureCompensationReview {
    bool available{false};
    std::string status;
    double targetWidthMm{std::numeric_limits<double>::quiet_NaN()};
    double measuredWidthMm{std::numeric_limits<double>::quiet_NaN()};
    double relativeWidthError{std::numeric_limits<double>::quiet_NaN()};
};

FeatureCompensationReview parseFeatureCompensationReview(const std::string& csvText)
{
    FeatureCompensationReview review;
    std::istringstream stream(csvText);
    std::string headerLine;
    if (!std::getline(stream, headerLine)) {
        return review;
    }
    const std::vector<std::string> headers = splitCsvLine(headerLine);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = splitCsvLine(line);
        const std::string featureId = lowerAscii(csvValueFor(headers, cells, "feature_id"));
        const std::string method = lowerAscii(csvValueFor(headers, cells, "method"));
        if (featureId.find("slot") == std::string::npos &&
            method.find("slot") == std::string::npos) {
            continue;
        }

        review.available = true;
        review.status = lowerAscii(csvValueFor(headers, cells, "status"));
        review.targetWidthMm = parseReviewDouble(csvValueFor(headers, cells, "target_slot_width_mm"));
        if (!std::isfinite(review.targetWidthMm)) {
            review.targetWidthMm = parseReviewDouble(csvValueFor(headers, cells, "cad_slot_width_mm"));
        }
        review.measuredWidthMm = parseReviewDouble(csvValueFor(headers, cells, "measured_slot_width_mm"));
        if (std::isfinite(review.targetWidthMm) && review.targetWidthMm > 1e-9 &&
            std::isfinite(review.measuredWidthMm)) {
            review.relativeWidthError =
                std::abs(review.measuredWidthMm - review.targetWidthMm) / review.targetWidthMm;
        }
        return review;
    }
    return review;
}

std::string buildQualityReviewCsv(const pinjie::cad_design::DesignAlignmentResult& designResult,
                                   const StitchingResult& stitching)
{
    std::ostringstream stream;
    stream << "check,status,value,threshold,message\n";

    const auto& summary = designResult.summary;
    appendReviewRow(stream,
                    "external_cad_profile_sample_count",
                    summary.designProfileMetadata.sourceType != "external_cad" ||
                        summary.designProfileMetadata.sampleCount >= 2,
                    static_cast<double>(summary.designProfileMetadata.sampleCount),
                    2.0,
                    "Design source=" + summary.designProfileMetadata.sourceType +
                        "; external CAD comparison requires at least two sampled design-profile points.");
    const std::size_t usedDesignPointCount =
        static_cast<std::size_t>(std::count_if(designResult.profilePoints.begin(),
                                               designResult.profilePoints.end(),
                                               [](const auto& point) {
                                                   return point.isUsed;
                                               }));
    const std::size_t usedCadCoordinateCount =
        static_cast<std::size_t>(std::count_if(designResult.profilePoints.begin(),
                                               designResult.profilePoints.end(),
                                               [](const auto& point) {
                                                   return point.isUsed && point.hasCadCoordinates;
                                               }));
    appendReviewRow(stream,
                    "used_cad_coordinate_point_count",
                    summary.designProfileMetadata.sourceType != "external_cad" ||
                        (usedDesignPointCount > 0 && usedCadCoordinateCount == usedDesignPointCount),
                    static_cast<double>(usedCadCoordinateCount),
                    static_cast<double>(usedDesignPointCount),
                    "External CAD error solving should export CAD-frame X/Y/Z coordinates for every used design-comparison point.");
    const FeatureCompensationReview featureReview =
        parseFeatureCompensationReview(designResult.featureCompensationCsvText);
    const bool slotFeatureConsistent =
        summary.designProfileMetadata.sourceType != "external_cad" ||
        !featureReview.available ||
        (featureReview.status != "mismatch" && featureReview.status != "unavailable" &&
         (!std::isfinite(featureReview.relativeWidthError) || featureReview.relativeWidthError <= 0.25));
    appendReviewRow(stream,
                    "cad_image_slot_feature_consistency",
                    slotFeatureConsistent,
                    featureReview.relativeWidthError,
                    0.25,
                    "CAD and image ROI must describe the same slot before compensation. WARN means diagnostic only.");
    double usedMinS = std::numeric_limits<double>::infinity();
    double usedMaxS = -std::numeric_limits<double>::infinity();
    for (const auto& point : designResult.profilePoints) {
        if (point.isUsed && std::isfinite(point.sAlignedMm)) {
            usedMinS = std::min(usedMinS, point.sAlignedMm);
            usedMaxS = std::max(usedMaxS, point.sAlignedMm);
        }
    }
    const double designSpanMm = summary.designProfileMetadata.maxSMm -
                                summary.designProfileMetadata.minSMm;
    const double usedSpanMm = usedMaxS - usedMinS;
    const double sectionCoverageRatio =
        std::isfinite(usedSpanMm) && usedSpanMm > 0.0 && designSpanMm > 1e-9
            ? usedSpanMm / designSpanMm
            : std::numeric_limits<double>::quiet_NaN();
    appendReviewRow(stream,
                    "cad_image_profile_span_coverage",
                    summary.designProfileMetadata.sourceType != "external_cad" ||
                        (std::isfinite(sectionCoverageRatio) && sectionCoverageRatio >= 0.65),
                    sectionCoverageRatio,
                    0.65,
                    "Detected contour should cover most of the CAD section for full-section alignment. WARN means local ROI only.");
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
                    summary.outlierRatio <= 0.05,
                    summary.outlierRatio,
                    0.05,
                    "Profile outlier ratio after Hampel and slot-edge filtering.");
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
    appendReviewRow(stream,
                    "absolute_bias_refine_correction_um",
                    !summary.appliedAbsoluteBiasRefine || std::abs(summary.absoluteBiasCorrectionUm) <= 180.0,
                    std::abs(summary.absoluteBiasCorrectionUm),
                    180.0,
                    "Final radial bias-refine correction applied after shape best-fit; large values indicate anchor/preset mismatch.");

    double worstNormalRmsePx = 0.0;
    std::size_t worstStep = 0;
    int priorClampedStepCount = 0;
    std::vector<double> stablePrimaryShiftsPx;
    std::vector<double> stableOverlapRatios;
    std::vector<double> nominalPrimaryShiftsPx;
    std::vector<double> nominalOverlapRatios;
    double maxPrimaryTrajectoryJumpPx = 0.0;
    double maxPerpTrajectoryJumpPx = 0.0;
    for (const StitchStepRecord& step : stitching.steps) {
        if (step.usedMotionPriorFallback || step.transform.direction == "PRIOR") {
            ++priorClampedStepCount;
        }
        const ResidualStatistics& normal =
            step.transform.metrics.normalInlier.valid() ? step.transform.metrics.normalInlier
                                                        : step.transform.metrics.normalAll;
        if (normal.valid() && normal.rmse > worstNormalRmsePx) {
            worstNormalRmsePx = normal.rmse;
            worstStep = step.stepIndex;
        }

        const double actualPrimaryShiftPx =
            motionPrimaryShift(step, step.transform.dx, step.transform.dy);
        const double actualPerpShiftPx =
            motionPerpendicularShift(step, step.transform.dx, step.transform.dy);
        if (isStableReferenceStep(step)) {
            stablePrimaryShiftsPx.push_back(actualPrimaryShiftPx);
            const double overlapRatio = stepOverlapRatio(step, actualPrimaryShiftPx);
            if (std::isfinite(overlapRatio)) {
                stableOverlapRatios.push_back(overlapRatio);
            }
        }

        if (step.hasNominalPrior) {
            const double nominalPrimaryShiftPx =
                motionPrimaryShift(step, step.nominalPriorDx, step.nominalPriorDy);
            nominalPrimaryShiftsPx.push_back(nominalPrimaryShiftPx);
            const double overlapRatio = stepOverlapRatio(step, nominalPrimaryShiftPx);
            if (std::isfinite(overlapRatio)) {
                nominalOverlapRatios.push_back(overlapRatio);
            }
        }

        if (step.hasTrajectoryPrior) {
            const double trajectoryPrimaryShiftPx =
                motionPrimaryShift(step, step.trajectoryPriorDx, step.trajectoryPriorDy);
            const double trajectoryPerpShiftPx =
                motionPerpendicularShift(step, step.trajectoryPriorDx, step.trajectoryPriorDy);
            maxPrimaryTrajectoryJumpPx =
                std::max(maxPrimaryTrajectoryJumpPx,
                         std::abs(actualPrimaryShiftPx - trajectoryPrimaryShiftPx));
            maxPerpTrajectoryJumpPx =
                std::max(maxPerpTrajectoryJumpPx,
                         std::abs(actualPerpShiftPx - trajectoryPerpShiftPx));
        }
    }

    appendReviewRow(stream,
                    "worst_step_normal_rmse_px",
                    worstNormalRmsePx <= 0.25,
                    worstNormalRmsePx,
                    0.25,
                    "Worst single-step inlier normal RMSE; worst_step=" + std::to_string(worstStep) + ".");
    appendReviewRow(stream,
                    "prior_clamped_step_count",
                    priorClampedStepCount == 0,
                    static_cast<double>(priorClampedStepCount),
                    0.0,
                    "Steps reported with motion-prior clamp/fallback because local matching was not trustworthy.");
    const double stablePrimaryMedianPx = medianOfValues(stablePrimaryShiftsPx);
    const double nominalPrimaryMedianPx = medianOfValues(nominalPrimaryShiftsPx);
    if (std::isfinite(stablePrimaryMedianPx) && std::isfinite(nominalPrimaryMedianPx)) {
        appendReviewRow(stream,
                        "stable_primary_shift_minus_nominal_px",
                        std::abs(stablePrimaryMedianPx - nominalPrimaryMedianPx) <= 40.0,
                        std::abs(stablePrimaryMedianPx - nominalPrimaryMedianPx),
                        40.0,
                        "Median stable acquisition step differs from the configured overlap prior; large values mean preset overlap is biased.");
    }
    const double stableOverlapMedian = medianOfValues(stableOverlapRatios);
    const double nominalOverlapMedian = medianOfValues(nominalOverlapRatios);
    if (std::isfinite(stableOverlapMedian) && std::isfinite(nominalOverlapMedian)) {
        appendReviewRow(stream,
                        "stable_overlap_minus_nominal_ratio",
                        std::abs(stableOverlapMedian - nominalOverlapMedian) <= 0.01,
                        std::abs(stableOverlapMedian - nominalOverlapMedian),
                        0.01,
                        "Median effective overlap differs from the configured overlap ratio; values above 0.01 usually indicate preset mismatch.");
    }
    appendReviewRow(stream,
                    "max_primary_jump_from_trajectory_px",
                    maxPrimaryTrajectoryJumpPx <= 40.0,
                    maxPrimaryTrajectoryJumpPx,
                    40.0,
                    "Largest primary-axis jump relative to the stable trajectory prior; large values indicate a real acquisition-step change.");
    appendReviewRow(stream,
                    "max_perp_jump_from_trajectory_px",
                    maxPerpTrajectoryJumpPx <= 25.0,
                    maxPerpTrajectoryJumpPx,
                    25.0,
                    "Largest cross-axis jump relative to the stable trajectory prior; large values indicate tilt or cross-feed motion during acquisition.");
    return stream.str();
}

bool saveImageArtifact(const std::string& path,
                       const cv::Mat& image,
                       const std::string& label,
                       StitchRunResult& runResult,
                       const StitchCallbacks& callbacks);

std::string quoteCommandArgument(const std::string& value)
{
    std::string quoted = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            quoted += "\\\"";
        } else {
            quoted += ch;
        }
    }
    quoted += "\"";
    return quoted;
}

std::filesystem::path inferDesignPlotResultDir(const StitchRunRequest& request)
{
    if (!request.resultOutputDir.empty()) {
        return std::filesystem::u8path(request.resultOutputDir);
    }

    const std::vector<std::string> candidates = {
        request.designComparisonOverlayOutputPath,
        request.designCompensationPlotOutputPath,
        request.designErrorProfileCsvOutputPath,
        request.designErrorSummaryCsvOutputPath,
        request.designCompensationCsvOutputPath,
    };
    for (const std::string& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        const std::filesystem::path path = std::filesystem::u8path(candidate);
        if (!path.parent_path().empty()) {
            return path.parent_path();
        }
    }
    return {};
}

bool hasSingleSlotFeatureResult(const pinjie::cad_design::DesignAlignmentResult& designResult)
{
    return designResult.message.find("Single-slot ROI applied") != std::string::npos ||
           designResult.message.find("single slot width target mode") != std::string::npos ||
           designResult.featureCompensationCsvText.find("primary_slot,") != std::string::npos ||
           designResult.featureCompensationCsvText.find("single_slot_width,") != std::string::npos;
}

bool saveMatplotlibDesignPlots(const StitchRunRequest& request,
                               const pinjie::cad_design::DesignAlignmentResult& designResult,
                               StitchRunResult& runResult,
                               const StitchCallbacks& callbacks)
{
    const bool singleSlotContourPreview =
        !request.contourOverlayOutputPath.empty() &&
        (request.pipelineConfig.designTargetSlotWidthMm > 1e-9 || hasSingleSlotFeatureResult(designResult));

    if (request.designComparisonOverlayOutputPath.empty() &&
        request.designCompensationPlotOutputPath.empty() &&
        !singleSlotContourPreview) {
        return true;
    }

    const std::filesystem::path resultDir = inferDesignPlotResultDir(request);
    if (resultDir.empty()) {
        runResult.message = "Matplotlib design plot generation failed: result directory is not configured.";
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    const std::filesystem::path scriptPath =
        std::filesystem::path(PINJIE_PROJECT_ROOT) / "report" / "figure_export" / "gui_design_plots.py";
    if (!std::filesystem::exists(scriptPath)) {
        runResult.message = "Matplotlib design plot script not found: " + scriptPath.generic_string();
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    std::string command = "python " + quoteCommandArgument(scriptPath.generic_string()) +
                          " --result-dir " + quoteCommandArgument(resultDir.generic_string());
    if (!request.designComparisonOverlayOutputPath.empty()) {
        command += " --comparison-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.designComparisonOverlayOutputPath).generic_string());
    }
    if (!request.designCompensationPlotOutputPath.empty()) {
        command += " --compensation-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.designCompensationPlotOutputPath).generic_string());
    }
    if (singleSlotContourPreview) {
        command += " --contour-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.contourOverlayOutputPath).generic_string());
    }
    const std::filesystem::path pointCloudOutputPath = resultDir / "design_3d_point_cloud.png";
    if (!request.design3dErrorCsvOutputPath.empty()) {
        command += " --pointcloud-output " + quoteCommandArgument(pointCloudOutputPath.generic_string());
    }
    command += " --dpi 240";

    emitLog(callbacks, "[Info] Generating design plots with Python matplotlib.");
    const int exitCode = std::system(command.c_str());
    if (exitCode != 0) {
        runResult.message = "Matplotlib design plot generation failed with exit code " +
                            std::to_string(exitCode) + ".";
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    if (!request.designComparisonOverlayOutputPath.empty()) {
        emitLog(callbacks,
                "[Info] Saved matplotlib design comparison plot: " +
                    request.designComparisonOverlayOutputPath);
    }
    if (!request.designCompensationPlotOutputPath.empty()) {
        emitLog(callbacks,
                "[Info] Saved matplotlib design compensation plot: " +
                    request.designCompensationPlotOutputPath);
    }
    if (singleSlotContourPreview) {
        runResult.singleSlotContourPreviewGenerated = true;
        emitLog(callbacks,
                "[Info] Saved matplotlib single-slot contour preview: " +
                    request.contourOverlayOutputPath);
    }
    if (!request.design3dErrorCsvOutputPath.empty()) {
        emitLog(callbacks,
                "[Info] Saved matplotlib 3D point cloud plot: " +
                    pointCloudOutputPath.generic_string());
    }
    return true;
}

bool saveMatplotlibLocalSlotPlots(const StitchRunRequest& request,
                                  StitchRunResult& runResult,
                                  const StitchCallbacks& callbacks)
{
    if (request.designComparisonOverlayOutputPath.empty() &&
        request.designCompensationPlotOutputPath.empty() &&
        request.contourOverlayOutputPath.empty()) {
        return true;
    }

    const std::filesystem::path resultDir = inferDesignPlotResultDir(request);
    if (resultDir.empty()) {
        runResult.message = "Local slot matplotlib plot generation failed: result directory is not configured.";
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    const std::filesystem::path scriptPath =
        std::filesystem::path(PINJIE_PROJECT_ROOT) / "report" / "figure_export" / "gui_design_plots.py";
    if (!std::filesystem::exists(scriptPath)) {
        runResult.message = "Matplotlib design plot script not found: " + scriptPath.generic_string();
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    std::string command = "python " + quoteCommandArgument(scriptPath.generic_string()) +
                          " --result-dir " + quoteCommandArgument(resultDir.generic_string());
    if (!request.contourOverlayOutputPath.empty()) {
        command += " --contour-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.contourOverlayOutputPath).generic_string());
    }
    if (!request.designComparisonOverlayOutputPath.empty()) {
        command += " --comparison-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.designComparisonOverlayOutputPath).generic_string());
    }
    if (!request.designCompensationPlotOutputPath.empty()) {
        command += " --compensation-output " +
                   quoteCommandArgument(std::filesystem::u8path(request.designCompensationPlotOutputPath).generic_string());
    }
    command += " --dpi 260";

    emitLog(callbacks, "[Info] Generating local slot plots with Python matplotlib.");
    const int exitCode = std::system(command.c_str());
    if (exitCode != 0) {
        runResult.message = "Local slot matplotlib plot generation failed with exit code " +
                            std::to_string(exitCode) + ".";
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
    }

    runResult.singleSlotContourPreviewGenerated = true;
    emitLog(callbacks, "[Info] Saved local slot matplotlib plots under: " + resultDir.generic_string());
    return true;
}

std::string buildLocalSlotMetricCsv(const pinjie::cad_design::LocalSlotAnalysisResult& analysis)
{
    std::ostringstream stream;
    stream << "metric,value,unit\n";
    stream << "detected_edge_points," << analysis.detectedPointCount << ",count\n";
    stream << "used_edge_points," << analysis.usedPointCount << ",count\n";
    stream << "filtered_edge_points," << analysis.outlierPointCount << ",count\n";
    stream << "bottom_width_px," << formatReviewValue(analysis.bottomWidthPx) << ",px\n";
    stream << "pixel_size," << formatReviewValue(analysis.pixelSizeMm) << ",mm/px\n";
    stream << "slot_width," << formatReviewValue(analysis.measuredSlotWidthMm) << ",mm\n";
    stream << "slot_width_error," << formatReviewValue(analysis.widthErrorUm) << ",um\n";
    stream << "normal_rmse," << formatReviewValue(analysis.normalRmseUm) << ",um\n";
    stream << "normal_p95_abs," << formatReviewValue(analysis.normalP95AbsUm) << ",um\n";
    stream << "normal_pv," << formatReviewValue(analysis.normalPvUm) << ",um\n";
    return stream.str();
}

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
                                                             static_cast<int>(images.size()),
                                                             edges[i].preprocessingMode);
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
                                                               step);
        callbacks.onImage("debug_step", step.stepIndex, totalSteps, debugImage);
    }
}

bool shouldApplyCentralSlotImageRoi(const StitchPipelineConfig& config)
{
    return config.designUseCentralSlotImageRoi &&
           config.designUseExternalProfile &&
           config.designTargetSlotWidthMm > 1e-9;
}

cv::Rect centralSlotImageRoiRect(const cv::Mat& image, const StitchPipelineConfig& config)
{
    if (image.empty()) {
        return {};
    }

    const double xRatio = std::clamp(config.designImageRoiXRatio, 0.0, 0.95);
    const double yRatio = std::clamp(config.designImageRoiYRatio, 0.0, 0.95);
    const double wRatio = std::clamp(config.designImageRoiWidthRatio, 0.02, 1.0 - xRatio);
    const double hRatio = std::clamp(config.designImageRoiHeightRatio, 0.02, 1.0 - yRatio);

    int x = static_cast<int>(std::lround(xRatio * image.cols));
    int y = static_cast<int>(std::lround(yRatio * image.rows));
    int width = static_cast<int>(std::lround(wRatio * image.cols));
    int height = static_cast<int>(std::lround(hRatio * image.rows));
    width = std::clamp(width, 8, image.cols);
    height = std::clamp(height, 8, image.rows);
    x = std::clamp(x, 0, std::max(0, image.cols - width));
    y = std::clamp(y, 0, std::max(0, image.rows - height));
    return cv::Rect(x, y, width, height) & cv::Rect(0, 0, image.cols, image.rows);
}

std::vector<cv::Mat> cropImagesToCentralSlotRoi(const std::vector<cv::Mat>& images,
                                                const StitchPipelineConfig& config,
                                                const StitchCallbacks& callbacks)
{
    if (!shouldApplyCentralSlotImageRoi(config)) {
        return images;
    }

    std::vector<cv::Mat> cropped;
    cropped.reserve(images.size());
    for (std::size_t index = 0; index < images.size(); ++index) {
        const cv::Mat& image = images[index];
        const cv::Rect roi = centralSlotImageRoiRect(image, config);
        if (roi.width <= 0 || roi.height <= 0) {
            cropped.push_back(image);
            continue;
        }
        cropped.push_back(image(roi).clone());
        emitLog(callbacks,
                "[Info] Applied central slot image ROI to image " +
                    std::to_string(index + 1) + ": x=" + std::to_string(roi.x) +
                    ", y=" + std::to_string(roi.y) +
                    ", w=" + std::to_string(roi.width) +
                    ", h=" + std::to_string(roi.height) + ".");
    }
    return cropped;
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
                                   const StitchRunMode runMode,
                                   const std::vector<EdgeVariants>& edges,
                                   StitchRunResult& runResult,
                                   const StitchCallbacks& callbacks)
{
    if (runMode == StitchRunMode::Registration) {
        emitLog(callbacks,
                "[Info] Registration module completed; CAD error analysis and compensation are reserved for full/report modules.");
        return true;
    }

    if (!request.pipelineConfig.enableDesignComparison) {
        return true;
    }

    if (request.designErrorProfileCsvOutputPath.empty() &&
        request.designErrorSummaryCsvOutputPath.empty() &&
        request.design3dErrorCsvOutputPath.empty() &&
        request.designCompensationCsvOutputPath.empty() &&
        request.designFeatureCompensationCsvOutputPath.empty() &&
        request.designComparisonOverlayOutputPath.empty() &&
        request.designCompensationPlotOutputPath.empty() &&
        request.qualityReviewCsvOutputPath.empty()) {
        return true;
    }

    const pinjie::cad_design::DesignAlignmentResult designResult =
        pinjie::cad_design::compareMeasuredProfileToDesign(edges, runResult.stitching.imageTransforms, request.pipelineConfig);

    if (!designResult.ok) {
        runResult.message = "CAD/设计误差分析失败：" + designResult.message;
        emitLog(callbacks, "[Error] " + runResult.message);
        return false;
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

    if (!request.design3dErrorCsvOutputPath.empty()) {
        if (!writeTextFileToPath(request.design3dErrorCsvOutputPath, designResult.error3dCsvText)) {
            runResult.message =
                "Failed to save design_3d_error_points.csv: " + request.design3dErrorCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved design 3D error points CSV: " + request.design3dErrorCsvOutputPath);
    }

    if (!request.designCompensationCsvOutputPath.empty()) {
        if (!writeTextFileToPath(request.designCompensationCsvOutputPath, designResult.compensationCsvText)) {
            runResult.message =
                "Failed to save design_compensation.csv: " + request.designCompensationCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved design compensation CSV: " + request.designCompensationCsvOutputPath);
    }

    if (!request.designFeatureCompensationCsvOutputPath.empty()) {
        if (!writeTextFileToPath(request.designFeatureCompensationCsvOutputPath,
                                 designResult.featureCompensationCsvText)) {
            runResult.message =
                "Failed to save design_feature_compensation.csv: " +
                request.designFeatureCompensationCsvOutputPath;
            emitLog(callbacks, "[Error] " + runResult.message);
            return false;
        }
        emitLog(callbacks,
                "[Info] Saved design feature compensation CSV: " +
                request.designFeatureCompensationCsvOutputPath);
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

    if (!saveMatplotlibDesignPlots(request, designResult, runResult, callbacks)) {
        return false;
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
    if (request.contourOverlayOutputPath.empty() &&
        request.stitchedContourProfilePlotOutputPath.empty() &&
        request.tangentCorrelationAllOutputPath.empty() &&
        request.tangentCorrelationInlierOutputPath.empty()) {
        return true;
    }

    const bool singleSlotContourPreview =
        (runResult.singleSlotContourPreviewGenerated ||
         request.pipelineConfig.designTargetSlotWidthMm > 1e-9) &&
        !request.contourOverlayOutputPath.empty();
    if (!singleSlotContourPreview) {
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
    } else {
        emitLog(callbacks,
                "[Info] Skipping full-field contour overlay because single-slot ROI contour preview is enabled.");
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
        images = cropImagesToCentralSlotRoi(images, request.pipelineConfig, callbacks);
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

    const bool useLocalSlot2dFallback = request.pipelineConfig.localSlotImageMode;
    if (useLocalSlot2dFallback) {
        if (images.size() != 1) {
            runResult.message =
                "局部槽特征检测要求只输入 1 张已截取好的单槽局部图；当前输入 " +
                std::to_string(images.size()) + " 张。";
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }

        pinjie::cad_design::LocalSlotAnalysisRequest localRequest;
        localRequest.sourceImagePath = request.imagePaths.empty() ? std::string{} : request.imagePaths.front();
        localRequest.bottomWidthMm = request.pipelineConfig.localSlotBottomWidthMm;
        localRequest.pixelSizeOverrideMm = request.pipelineConfig.localSlotPixelSizeOverrideMm;
        localRequest.pixelSizeScale = request.pipelineConfig.localSlotPixelSizeScale;
        localRequest.maxOutputPoints = request.pipelineConfig.localSlotMaxOutputPoints;
        localRequest.cadProfileMetadata = request.pipelineConfig.designProfileMetadata;
        localRequest.cadProfileSamples = request.pipelineConfig.designExternalProfileSamples;

        emitLog(callbacks,
                "[信息] 局部槽特征检测：不执行自动 ROI，不进入拼接，直接按槽底宽度 " +
                    formatReviewValue(localRequest.bottomWidthMm) + " mm 标定，像素当量修正系数 " +
                    formatReviewValue(localRequest.pixelSizeScale) + "。");
        const pinjie::cad_design::LocalSlotAnalysisResult analysis =
            pinjie::cad_design::analyzeLocalSlotImage(images.front(), localRequest);
        if (!analysis.ok) {
            runResult.message = "局部槽检测补偿失败：" + analysis.message;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }

        runResult.preprocessedImageCount = 1;
        runResult.csvText = buildLocalSlotMetricCsv(analysis);
        runResult.stitching.canvas = images.front().clone();
        runResult.stitching.globalTransform = cv::Mat::eye(3, 3, CV_64F);
        runResult.stitching.imageTransforms = {cv::Mat::eye(3, 3, CV_64F)};
        cache->stitching = runResult.stitching;
        cache->csvText = runResult.csvText;

        if (!saveImageArtifact(request.panoramaOutputPath,
                               images.front(),
                               "局部槽输入图像",
                               runResult,
                               callbacks)) {
            return runResult;
        }
        if (!request.csvOutputPath.empty() &&
            !writeTextFileToPath(request.csvOutputPath, runResult.csvText)) {
            runResult.message = "局部槽摘要 CSV 保存失败：" + request.csvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.designErrorProfileCsvOutputPath.empty() &&
            !writeTextFileToPath(request.designErrorProfileCsvOutputPath, analysis.profileCsvText)) {
            runResult.message = "局部槽误差明细 CSV 保存失败：" + request.designErrorProfileCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.designErrorSummaryCsvOutputPath.empty() &&
            !writeTextFileToPath(request.designErrorSummaryCsvOutputPath, analysis.summaryCsvText)) {
            runResult.message = "局部槽误差汇总 CSV 保存失败：" + request.designErrorSummaryCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.design3dErrorCsvOutputPath.empty() &&
            !writeTextFileToPath(request.design3dErrorCsvOutputPath, analysis.error3dCsvText)) {
            runResult.message = "局部槽截面误差坐标 CSV 保存失败：" + request.design3dErrorCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.designCompensationCsvOutputPath.empty() &&
            !writeTextFileToPath(request.designCompensationCsvOutputPath, analysis.compensationCsvText)) {
            runResult.message = "局部槽补偿 CSV 保存失败：" + request.designCompensationCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.designFeatureCompensationCsvOutputPath.empty() &&
            !writeTextFileToPath(request.designFeatureCompensationCsvOutputPath,
                                 analysis.featureCompensationCsvText)) {
            runResult.message = "局部槽特征补偿 CSV 保存失败：" + request.designFeatureCompensationCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.qualityReviewCsvOutputPath.empty() &&
            !writeTextFileToPath(request.qualityReviewCsvOutputPath, analysis.qualityReviewCsvText)) {
            runResult.message = "局部槽质量审查 CSV 保存失败：" + request.qualityReviewCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }
        if (!request.contourPointsCsvOutputPath.empty() &&
            !writeTextFileToPath(request.contourPointsCsvOutputPath, analysis.contourPointsCsvText)) {
            runResult.message = "局部槽轮廓点 CSV 保存失败：" + request.contourPointsCsvOutputPath;
            emitLog(callbacks, "[错误] " + runResult.message);
            return runResult;
        }

        if (!saveMatplotlibLocalSlotPlots(request, runResult, callbacks)) {
            return runResult;
        }

        emitLog(callbacks,
                "[信息] 局部槽检测完成：有效边缘点 " + std::to_string(analysis.usedPointCount) +
                    "，像素当量 " + formatReviewValue(analysis.pixelSizeMm) +
                    " mm/px，槽宽 " + formatReviewValue(analysis.measuredSlotWidthMm) + " mm。");
        runResult.ok = true;
        runResult.message = "局部槽特征检测、误差分析与补偿结算流程运行成功。";
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

    if (!saveDesignComparisonArtifacts(request, runMode, edges, runResult, callbacks)) {
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
