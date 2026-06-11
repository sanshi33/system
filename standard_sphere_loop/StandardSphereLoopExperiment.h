#pragma once

#include "stitch/StitchTypes.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace pinjie::standard_sphere_loop {

struct CircleFitReport {
    bool ok{false};
    int pointCount{0};
    cv::Point2d center{};
    double radiusPx{0.0};
    double rmsePx{0.0};
    double meanAbsPx{0.0};
    double medianAbsPx{0.0};
    double madPx{0.0};
    double sigmaMadPx{0.0};
    double maxAbsPx{0.0};
    double qualityWeightMean{0.0};
    double confidenceMean{0.0};
    double gradientMean{0.0};
};

struct PairLoopRecord {
    std::size_t stepIndex{0};
    std::size_t referenceIndex{0};
    std::size_t targetIndex{0};
    std::size_t segmentIndex{0};
    std::string expectedDirection;
    cv::Point2d rotationCenter{};
    cv::Point2d expectedStepPx{};
    cv::Point2d expectedStepMm{};
    cv::Point2d measuredStepMm{};
    cv::Point2d cumulativeMeasuredPx{};
    cv::Point2d cumulativeExpectedPx{};
    cv::Point2d cumulativeErrorPx{};
    double searchRangeX{0.0};
    double searchRangeY{0.0};
    double localTranslationErrorPx{0.0};
    double localTranslationErrorMm{0.0};
    double cumulativeTranslationErrorPx{0.0};
    double cumulativeTranslationErrorMm{0.0};
    stitch::TransformResult transform;
};

struct SegmentLoopRecord {
    std::size_t segmentIndex{0};
    std::string expectedDirection;
    std::size_t startStep{0};
    std::size_t endStep{0};
    std::size_t pairCount{0};
    cv::Point2d measuredDeltaPx{};
    cv::Point2d expectedDeltaPx{};
    cv::Point2d residualDeltaPx{};
    double residualTranslationPx{0.0};
    double residualTranslationMm{0.0};
};

struct StandardSphereLoopConfig {
    stitch::StitchPipelineConfig pipelineConfig{};
    double pixelSizeMm{0.0};
    double sphereDiameterMm{0.0};
    int sphereDiameterDecimalPlaces{4};
    bool useClockwiseLoopPath{true};
    double horizontalFieldOfViewMm{40.0};
    double verticalFieldOfViewMm{30.0};
    double horizontalStepMm{12.0};
    double verticalStepMm{9.0};
    std::vector<int> clockwiseSegmentPairCounts;
    std::vector<stitch::MotionPriorDirection> clockwiseSegmentDirections;
    bool enableGlobalLoopClosureOptimization{true};
    double circleOptimizationMaxTranslationPx{5.0};
    double circleOptimizationMaxRotationDeg{0.08};
    double circleOptimizationTranslationRegularization{0.02};
    double circleOptimizationRotationRegularization{10.0};
    int circleOptimizationMaxIterations{5};
    bool enableSphereBadStepGeometryRescue{false};
    bool enableGlobalConsistencyReregistration{true};
    bool enableSoftGlobalDriftOptimization{true};
    bool enableBadStepLocalRefinement{true};
    double poseConsistencyTargetClosurePx{0.1};
    double poseConsistencyMaxCircleRmseIncreasePx{2.50};
    double poseConsistencyMaxCorrectionFraction{1.0};
};

struct StandardSphereLoopResult {
    bool ok{false};
    std::string message;
    std::vector<PairLoopRecord> pairRecords;
    std::vector<SegmentLoopRecord> segmentRecords;
    std::vector<cv::Mat> globalTransforms;
    std::vector<cv::Point2d> expectedGlobalTranslationsPx;
    std::vector<CircleFitReport> perImageCircles;
    CircleFitReport stitchedCircle;
    cv::Mat closureMatrix;
    cv::Point2d measuredClosurePx{};
    cv::Point2d expectedClosurePx{};
    cv::Point2d closureResidualPx{};
    double evaluationPixelSizeMm{0.0};
    double closureTranslationPx{0.0};
    double closureResidualTranslationPx{0.0};
    double closureResidualTranslationMm{0.0};
    double closureRotationDeg{0.0};
    double closureCenterDriftPx{0.0};
    double closureCornerRmsPx{0.0};
    double closureCornerMaxPx{0.0};
    bool globalLoopClosureOptimized{false};
    double preOptimizationClosureTranslationPx{0.0};
    double preOptimizationClosureRotationDeg{0.0};
    double preOptimizationStitchedCircleRmsePx{0.0};
    double preOptimizationStitchedCircleMeanAbsPx{0.0};
    double circleOptimizationMeanCorrectionPx{0.0};
    double circleOptimizationMaxCorrectionPx{0.0};
    double circleOptimizationMaxCorrectionRotationDeg{0.0};
    int circleOptimizationIterations{0};
    int sphereBadStepGeometryRescueCount{0};
    double sphereBadStepGeometryRescueMaxCorrectionPx{0.0};
    int globalConsistencyReregistrationCount{0};
    double globalConsistencyReregistrationImprovement{0.0};
    int badStepLocalRefinementCount{0};
    double badStepLocalRefinementCostImprovement{0.0};
    double badStepLocalRefinementMaxCorrectionPx{0.0};
    int softGlobalDriftOptimizationIterations{0};
    double softGlobalDriftOptimizationMaxStepCorrectionPx{0.0};
    double softGlobalDriftOptimizationMaxRotationCorrectionDeg{0.0};
    double softGlobalDriftOptimizationTotalCorrectionPx{0.0};
    double softGlobalDriftOptimizationCostImprovement{0.0};
    double softGlobalDriftOptimizationAcceptedFraction{0.0};
    double softGlobalDriftOptimizationCircleRmseChangePx{0.0};
};

StandardSphereLoopResult runStandardSphereLoopExperiment(const std::vector<cv::Mat>& images,
                                                        const std::vector<stitch::EdgeVariants>& edges,
                                                        const StandardSphereLoopConfig& config);

StandardSphereLoopResult evaluateStandardSphereStitchingResult(const std::vector<cv::Mat>& images,
                                                               const std::vector<stitch::EdgeVariants>& edges,
                                                               const stitch::StitchingResult& stitching,
                                                               const StandardSphereLoopConfig& config);

std::vector<stitch::MotionPriorDirection> buildClockwiseStepDirections(std::size_t pairCount,
                                                                       const StandardSphereLoopConfig& config);

std::vector<cv::Point2d> buildCircleCenterTranslationPriors(const std::vector<stitch::EdgeVariants>& edges);

std::string buildPairCsv(const StandardSphereLoopResult& result);

std::string buildBadStepDiagnosticsCsv(const StandardSphereLoopResult& result,
                                       const std::vector<stitch::EdgeVariants>& edges);

std::string buildCircleCsv(const StandardSphereLoopResult& result,
                           const std::vector<std::string>& imagePaths,
                           double pixelSizeMm,
                           double sphereDiameterMm);

std::string buildFieldBiasCsv(const StandardSphereLoopResult& result,
                              const std::vector<stitch::EdgeVariants>& edges,
                              const std::vector<cv::Mat>& images);

std::string buildFieldBiasCompensationCsv(const StandardSphereLoopResult& result,
                                          const std::vector<stitch::EdgeVariants>& edges,
                                          const std::vector<cv::Mat>& images);

std::string buildTransformCsv(const StandardSphereLoopResult& result);

std::string buildSegmentCsv(const StandardSphereLoopResult& result);

std::string buildSummaryCsv(const StandardSphereLoopResult& result,
                            const std::vector<stitch::EdgeVariants>& edges,
                            const std::vector<cv::Mat>& images,
                            double pixelSizeMm,
                            double sphereDiameterMm);

cv::Mat buildStitchedVisualization(const StandardSphereLoopResult& result,
                                   const std::vector<stitch::EdgeVariants>& edges,
                                   const std::vector<cv::Mat>& images);

} // namespace pinjie::standard_sphere_loop
