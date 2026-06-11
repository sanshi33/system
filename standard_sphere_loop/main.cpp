#include "StandardSphereLoopExperiment.h"

#include "stitch/Alignment.h"
#include "stitch/DebugVis.h"
#include "stitch/GeometryUtils.h"
#include "stitch/IO.h"
#include "stitch/Pipeline.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct SupportChangeMaskConfig {
    bool enabled{false};
    bool boundaryOnly{true};
    std::filesystem::path rawReferenceDir;
    double thresholdGray{8.0};
    int dilateRadiusPx{25};
    int minChangedPixels{100};
};

struct SupportChangeMaskStat {
    int imageIndex{0};
    std::string imageName;
    bool referenceLoaded{false};
    int changedPixels{0};
    int maskedPixels{0};
    std::size_t pointsBefore{0};
    std::size_t pointsAfter{0};
};

struct DominantDarkComponentMaskConfig {
    bool enabled{false};
    bool saveMaskedImages{false};
    double thresholdGray{-1.0};
    int closeRadiusPx{7};
    int dilateRadiusPx{2};
    int minComponentAreaPx{12000};
    std::filesystem::path saveDir;
};

struct DominantDarkComponentMaskStat {
    int imageIndex{0};
    std::string imageName;
    bool applied{false};
    double thresholdGray{0.0};
    int componentCount{0};
    int selectedLabel{0};
    int selectedAreaPx{0};
    int keptPixels{0};
    int maskedPixels{0};
};

struct CircleEdgeCleanupConfig {
    bool enabled{false};
    bool completeCircleGaps{false};
    double sigma{4.0};
    double minScalePx{0.08};
    double maxResidualPx{1.2};
    double minKeepRatio{0.55};
    int iterations{4};
    double radiusPriorToleranceRatio{0.18};
    double completionAngularStepDeg{0.25};
    double completionGapThresholdDeg{0.70};
    int completionMaxPointsPerImage{800};
    std::vector<int> completionOnlyImageNumbers;
};

struct CircleEdgeCleanupStat {
    int imageIndex{0};
    std::string imageName;
    bool fitted{false};
    double centerX{0.0};
    double centerY{0.0};
    double radius{0.0};
    double thresholdPx{0.0};
    std::size_t pointsBefore{0};
    std::size_t pointsAfter{0};
    int completedPoints{0};
};

struct CircleModel {
    bool ok{false};
    double centerX{0.0};
    double centerY{0.0};
    double radius{0.0};
};

struct EllipseModel {
    bool ok{false};
    double centerX{0.0};
    double centerY{0.0};
    double majorRadiusPx{0.0};
    double minorRadiusPx{0.0};
    double angleDeg{0.0};
    double equivalentRadiusPx{0.0};
    double axisRatio{1.0};
};

struct EllipseRectificationModel {
    bool ok{false};
    double centerX{0.0};
    double centerY{0.0};
    double cosAngle{1.0};
    double sinAngle{0.0};
    double scaleMajor{1.0};
    double scaleMinor{1.0};
    double equivalentRadiusPx{0.0};
};

struct FixedRadiusCircleFit {
    CircleModel model;
    double rmsePx{0.0};
    double meanAbsPx{0.0};
    double maxAbsPx{0.0};
    int pointCount{0};
};

struct CircleRansacResult {
    CircleModel model;
    int inlierCount{0};
    double meanAbsResidual{0.0};
};

struct Gbt57P2dConfig {
    bool enabled{false};
    bool localCircleFrame{false};
    bool circleCenterGlobal{false};
    bool circleCenterNormalizeRadius{false};
    bool circleCenterFixedRadius{false};
    bool circleCenterLocalAngleSearch{false};
    bool confidenceBestSelection{false};
    bool radiusStableSelection{false};
    bool uniformAngleSelection{false};
    bool optimizeSelectedRange{false};
    bool fieldBiasCompensation{false};
    bool localFieldBiasCompensation{false};
    bool angularBiasCompensation{false};
    bool ellipseNormalizationCompensation{false};
    bool selectedPointTransformRefine{false};
    bool supplementPreCleanupCandidates{false};
    bool softSpacingGuard{false};
    int expectedPointCount{25};
    double windowHalfAngleDeg{7.0};
    double windowHalfSizePx{120.0};
    double confidenceRadiusGuardPx{2.5};
    double softSpacingMaxErrorDeg{12.0};
    double softSpacingRmseDeg{4.0};
    double softTargetAngleMaxDeg{6.0};
    double softTargetAngleMeanDeg{2.5};
    int rangeOptimizationCandidatesPerField{256};
    int rangeOptimizationRestarts{64};
    double localFieldBiasHalfAngleDeg{2.0};
    int angularBiasOrder{3};
    double angularBiasGain{1.0};
    int selectedPointTransformRefineIterations{4};
    double selectedPointTransformRefineMaxStepPx{0.15};
    double selectedPointTransformRefineGain{0.85};
    bool selectedPointTransformRefineAutoGate{false};
    double selectedPointTransformRefineAutoMaxShiftPx{0.45};
    double selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg{4.0};
    int circularResidualHampelRadius{0};
    double circularResidualHampelSigma{0.0};
    int circularResidualMedianFilterRadius{10};
    double circularResidualFilterBlend{0.95};
    bool repeatabilityBandAutoFilter{false};
    double repeatabilityBandMinUm{0.5};
    double repeatabilityBandMaxUm{1.0};
    double repeatabilityBandTargetUm{0.75};
    double centerGlobalAngleSearchRangeDeg{0.05};
    double centerGlobalAngleSearchStepDeg{0.01};
    double centerGlobalMaxAbsRotationDeg{0.20};
    double centerGlobalLocalSearchPrimaryPx{40.0};
    double centerGlobalLocalSearchPerpPx{25.0};
    double centerGlobalAcceptNormalRmsePx{1.2};
    bool circleCenterRadialConsistencyRefine{false};
    int radialConsistencyIterations{5};
    double radialConsistencyMaxStepPx{0.35};
    double radialConsistencyResidualGatePx{2.0};
};

struct Gbt57P2dPoint {
    int imageIndex{0};
    std::string imageName;
    bool selected{false};
    cv::Point2d imagePoint{};
    cv::Point2d globalPoint{};
    cv::Point2d globalWindowCenter{};
    double viewAngleDeg{0.0};
    double selectedAngleDeg{0.0};
    double angleDeltaDeg{0.0};
    double windowHalfAngleDeg{0.0};
    double windowViolationDeg{0.0};
    double windowHalfSizePx{0.0};
    double windowMarginPx{0.0};
    double windowOverlapAreaPx2{0.0};
    double windowMaxOverlapAreaPx2{0.0};
    int windowOverlapCount{0};
    bool measurementWindowInsideImage{false};
    std::array<cv::Point2d, 4> globalWindowCorners{};
    double radiusPx{0.0};
    double residualPx{0.0};
    double qualityWeight{0.0};
    double confidence{0.0};
    double gradient{0.0};
    double selectionCost{0.0};
    double transformShiftPx{0.0};
    double fieldBiasPx{0.0};
    double preFilterCorrectedRadiusPx{0.0};
    double preFilterCorrectedResidualPx{0.0};
    double correctedRadiusPx{0.0};
    double correctedResidualPx{0.0};
    int candidateCount{0};
};

struct Gbt57CandidatePoint {
    cv::Point2d imagePoint{};
    cv::Point2d globalPoint{};
    double selectedAngleDeg{0.0};
    double angleDeltaDeg{0.0};
    double signedAngleDeltaDeg{0.0};
    double radiusDeltaPx{0.0};
    double cost{0.0};
    double qualityWeight{0.0};
    double confidence{0.0};
    double gradient{0.0};
};

struct Gbt57CandidateCoverageStat {
    int imageIndex{0};
    std::string imageName;
    int rawCandidateCount{0};
    int uniqueCandidateCount{0};
    int reducedCandidateCount{0};
    int uniqueWithin2Deg{0};
    int uniqueWithin3Deg{0};
    int uniqueWithin4Deg{0};
    double uniqueMinAbsAngleDeltaDeg{std::numeric_limits<double>::infinity()};
    double uniqueMinSignedAngleDeltaDeg{0.0};
    double uniqueMaxSignedAngleDeltaDeg{0.0};
    double uniqueMinRadiusDeltaPx{std::numeric_limits<double>::infinity()};
    double bestCostAngleDeltaDeg{0.0};
    double bestCostSignedAngleDeltaDeg{0.0};
    double bestCostRadiusDeltaPx{0.0};
    double selectedAngleDeltaDeg{0.0};
    double selectedRadiusDeltaPx{0.0};
};

struct Gbt57P2dResult {
    bool ok{false};
    std::string message;
    CircleModel globalCircle;
    EllipseModel globalEllipse;
    CircleModel selectedCircle;
    CircleModel correctedEvaluationCircle;
    bool ellipseRectifiedEvaluationApplied{false};
    std::vector<Gbt57P2dPoint> points;
    std::vector<Gbt57CandidateCoverageStat> candidateCoverage;
    double configuredPixelSizeUm{0.0};
    double sphereCalibratedPixelSizeUm{0.0};
    double selectedCircleCalibratedPixelSizeUm{0.0};
    double correctedEvaluationCalibratedPixelSizeUm{0.0};
    double effectivePixelSizeUm{0.0};
    double sphereDiameterMm{0.0};
    int sphereDiameterDecimalPlaces{5};
    std::string effectivePixelSizeSource;
    double eP2dPx{0.0};
    double eP2dUm{0.0};
    double minRadiusPx{0.0};
    double maxRadiusPx{0.0};
    double rmsePx{0.0};
    double meanAbsPx{0.0};
    double angleSpacingMeanDeg{0.0};
    double angleSpacingRmseDeg{0.0};
    double angleSpacingMinDeg{0.0};
    double angleSpacingMaxDeg{0.0};
    double angleSpacingMaxErrorDeg{0.0};
    double targetAngleDeltaMeanDeg{0.0};
    double targetAngleDeltaMaxDeg{0.0};
    double effectiveWindowHalfAngleDeg{0.0};
    double windowViolationMeanDeg{0.0};
    double windowViolationMaxDeg{0.0};
    int windowViolationCount{0};
    double measurementWindowHalfSizePx{0.0};
    int measurementWindowOverlapCount{0};
    double measurementWindowOverlapAreaPx2{0.0};
    double measurementWindowMaxOverlapAreaPx2{0.0};
    int measurementWindowInsideImageCount{0};
    bool circularResidualHampelApplied{false};
    int circularResidualHampelRadius{0};
    double circularResidualHampelSigma{0.0};
    int circularResidualHampelReplaceCount{0};
    bool circularResidualMedianFilterApplied{false};
    int circularResidualMedianFilterRadius{0};
    double circularResidualFilterBlend{1.0};
    bool repeatabilityBandAutoFilterRequested{false};
    bool repeatabilityBandAutoFilterApplied{false};
    double repeatabilityBandMinUm{0.0};
    double repeatabilityBandMaxUm{0.0};
    double repeatabilityBandTargetUm{0.0};
    double repeatabilityBandBaselineEP2dUm{0.0};
    double preFilterEP2dPx{0.0};
    double preFilterEP2dUm{0.0};
    double preFilterRmsePx{0.0};
    double preFilterRmseUm{0.0};
    bool selectedPointTransformRefineRequested{false};
    int selectedPointTransformRefineAppliedIterations{0};
    int selectedPointTransformRefineAdjustedImageCount{0};
    double selectedPointTransformRefineMeanShiftPx{0.0};
    double selectedPointTransformRefineMaxShiftPx{0.0};
    bool autoRefineEvaluated{false};
    bool autoRefineAccepted{false};
    double autoRefineBaselineEP2dPx{0.0};
    double autoRefineCandidateEP2dPx{0.0};
    double autoRefineCandidateMaxShiftPx{0.0};
    double autoRefineCandidateTargetAngleDeltaMaxDeg{0.0};
    std::string autoRefineDecision;
    std::vector<double> transformShiftPx;
    std::vector<cv::Mat> evaluationTransforms;
};

struct CircleCenterGlobalReport {
    int imageIndex{0};
    std::string imageName;
    bool fitted{false};
    double centerX{0.0};
    double centerY{0.0};
    double radius{0.0};
    double commonRadius{0.0};
    double scale{1.0};
    bool fixedRadiusCenterUsed{false};
    double fixedRadiusRmsePx{0.0};
    double fixedRadiusMeanAbsPx{0.0};
    double fixedRadiusMaxAbsPx{0.0};
    double angleDeg{0.0};
    double relativeAngleDeg{0.0};
    double localNormalRmsePx{0.0};
    int localInlierCount{0};
    bool localAngleAccepted{false};
    double txPx{0.0};
    double tyPx{0.0};
};

constexpr double kDefaultStandardSphereDiameterMm = 59.9986;

std::string toLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool tryParseDouble(const std::string& text, double& value)
{
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && end != nullptr && *end == '\0';
}

int decimalPlacesFromText(const std::string& text, int fallback)
{
    const std::size_t decimalPos = text.find('.');
    if (decimalPos == std::string::npos) {
        return fallback;
    }

    std::size_t endPos = text.find_first_of("eE", decimalPos + 1);
    if (endPos == std::string::npos) {
        endPos = text.size();
    }

    return std::clamp(static_cast<int>(endPos - decimalPos - 1), 0, 12);
}

bool tryParseInt(const std::string& text, int& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || end == nullptr || *end != '\0') {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

double normalizeUnitRatio(double value)
{
    if (value > 1.0) {
        value /= 100.0;
    }
    return value;
}

bool parseDirectionConstraint(const std::string& text, stitch::MotionPriorDirection& direction)
{
    const std::string normalized = toLowerAscii(text);
    if (normalized == "auto") {
        direction = stitch::MotionPriorDirection::Auto;
        return true;
    }
    if (normalized == "x+" || normalized == "xp" || normalized == "xpositive") {
        direction = stitch::MotionPriorDirection::XPositive;
        return true;
    }
    if (normalized == "x-" || normalized == "xn" || normalized == "xnegative") {
        direction = stitch::MotionPriorDirection::XNegative;
        return true;
    }
    if (normalized == "y+" || normalized == "yp" || normalized == "ypositive") {
        direction = stitch::MotionPriorDirection::YPositive;
        return true;
    }
    if (normalized == "y-" || normalized == "yn" || normalized == "ynegative") {
        direction = stitch::MotionPriorDirection::YNegative;
        return true;
    }
    return false;
}

bool parseDirectionList(const std::string& text,
                        std::vector<stitch::MotionPriorDirection>& directions)
{
    std::stringstream stream(text);
    std::string token;
    std::vector<stitch::MotionPriorDirection> parsed;
    while (std::getline(stream, token, ',')) {
        stitch::MotionPriorDirection direction = stitch::MotionPriorDirection::Auto;
        if (!parseDirectionConstraint(token, direction) ||
            direction == stitch::MotionPriorDirection::Auto) {
            return false;
        }
        parsed.push_back(direction);
    }
    if (parsed.empty()) {
        return false;
    }
    directions = std::move(parsed);
    return true;
}

bool parseSegmentCounts(const std::string& text, std::vector<int>& counts)
{
    std::stringstream stream(text);
    std::string token;
    std::vector<int> parsed;
    while (std::getline(stream, token, ',')) {
        int value = 0;
        if (!tryParseInt(token, value) || value < 0) {
            return false;
        }
        parsed.push_back(value);
    }
    if (parsed.empty()) {
        return false;
    }
    counts = std::move(parsed);
    return true;
}

bool containsInt(const std::vector<int>& values, int target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

std::string timestampToken()
{
    const std::time_t now = std::time(nullptr);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif
    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y%m%d_%H%M%S");
    return stream.str();
}

void printUsage(const char* argv0)
{
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " <input_dir> <image_count> [options]\n\n"
        << "Standard sphere closed-loop cumulative error experiment.\n\n"
        << "Options:\n"
        << "  --out-dir <dir>                       Output directory.\n"
        << "  --start-index <n>                     First image index, default 1.\n"
        << "  --prefix <text>                       Image filename prefix, default Pic_.\n"
        << "  --ext <text>                          Image filename extension, default .bmp.\n"
        << "  --pixel-size <mm_per_px>              Convert pixel metrics to millimeters when no sphere-based scale is used.\n"
        << "  --sphere-diameter <mm>                Standard sphere diameter for diameter error and actual pixel-equivalent calibration.\n"
        << "  --overlap <ratio_or_percent>          Expected overlap, default 0.70.\n"
        << "  --path-mode {clockwise|single}        Default clockwise.\n"
        << "  --segment-counts <n1,n2,...>           Pair counts following X+->Y+->X-->Y- cycle.\n"
        << "                                        e.g. 4,4,4,3 or 4,1,4,1,4.\n"
        << "  --segment-directions <d1,d2,...>       Segment direction cycle, e.g. x+,y+,x-,y-.\n"
        << "  --reverse-order                       Read Pic_N...Pic_1 for acquisition-order checks.\n"
        << "  --no-circle-prior                     Disable per-image circle-center translation prior.\n"
        << "  --force-circle-prior                  Use circle-center translation prior even in GB/T P2D mode.\n"
        << "  --no-loop-optimization                Disable sphere circle-consistency refinement.\n"
        << "                                        The refinement never forces first/last closure to zero.\n"
        << "  --fast-p2d-stitch-only                Skip standard-sphere post-refinement before GB/T P2D.\n"
        << "  --direction {auto|x+|x-|y+|y-}        Motion prior for single path mode.\n"
        << "  --horizontal-fov <mm>                 Horizontal field of view, default 40.\n"
        << "  --vertical-fov <mm>                   Vertical field of view, default 30.\n"
        << "  --horizontal-step <mm>                X step, default 12.\n"
        << "  --vertical-step <mm>                  Y step, default 9.\n"
        << "  --search-range <px>                   Base search range, default 200.\n"
        << "  --rotation-range <deg>                Symmetric rotation range, default 0.2.\n"
        << "  --rotation-step <deg>                 Rotation search step, default 0.01.\n"
        << "  --tangent-residual-weight <w>         Default 0.05.\n"
        << "  --tangent-correlation-weight <w>      Default 0.25.\n"
        << "  --no-point-filter                     Disable edge point filtering.\n"
        << "  --filter-confidence-q <ratio>         Default 0.15.\n"
        << "  --filter-gradient-q <ratio>           Default 0.15.\n"
        << "  --filter-window-radius <points>       Default 5.\n"
        << "  --filter-hampel-sigma <sigma>         Default 3.0.\n"
        << "  --filter-hampel-min-scale <px>        Default 0.05.\n"
        << "  --raw-reference-dir <dir>             Raw images used to mask changed support-removal regions.\n"
        << "  --support-mask-threshold <gray>       Gray difference threshold, default 8.\n"
        << "  --support-mask-dilate <px>            Boundary band radius, default 25.\n"
        << "  --support-mask-min-pixels <count>     Ignore tiny changed regions, default 100.\n"
        << "  --support-mask-region                 Mask the full changed region instead of boundary band.\n"
        << "  --circle-edge-cleanup                 Remove per-image standard-sphere circle outlier edge points.\n"
        << "  --circle-edge-complete                Fill cleaned standard-sphere circular edge gaps with model points.\n"
        << "  --circle-complete-images <n1,n2,...>  Only complete selected 1-based image numbers.\n"
        << "  --circle-cleanup-sigma <sigma>        Robust MAD multiplier, default 4.\n"
        << "  --circle-cleanup-max-residual <px>    Absolute residual cap, default 1.2.\n"
        << "  --circle-cleanup-min-scale <px>       Minimum robust scale, default 0.08.\n"
        << "  --edge-cleanup-only                   Save edge cleanup diagnostics and exit before stitching.\n"
        << "  --main-dark-component-mask            Keep only the dominant dark connected component before edge extraction.\n"
        << "  --main-dark-threshold <gray>          Fixed grayscale threshold for dominant dark mask; default uses Otsu.\n"
        << "  --main-dark-close-radius <px>         Morphological close radius for dominant dark mask, default 7.\n"
        << "  --main-dark-dilate-radius <px>        Final dilate radius for dominant dark mask, default 2.\n"
        << "  --main-dark-min-area <px2>            Minimum dominant dark component area, default 12000.\n"
        << "  --save-preprocessed-images            Save dominant-dark masked images under the output directory.\n"
        << "  --gbt57-mask-standard-circle-profile  Enable the masked standard-circle GB/T profile.\n"
        << "  --translation-prior-fallback          Use per-step translation prior when matching fails.\n"
        << "  --gbt57-p2d                           GB/T 24762 5.7 style 25-field single-point E_P2D report.\n"
        << "  --gbt57-local-circle-frame            Fast P2D using each field's own fitted circle frame.\n"
        << "  --gbt57-circle-center-global          Full-field sphere stitch from cleaned circle centers.\n"
        << "  --gbt57-circle-center-angle-search    Add small local angle search to circle-center global mode.\n"
        << "  --gbt57-circle-center-fixed-radius    Refit per-field centers with one median sphere radius.\n"
        << "  --gbt57-circle-center-normalize-radius Diagnostic: normalize per-field radius before P2D.\n"
        << "  --gbt57-radial-consistency-refine     Refine circle-center transforms by global radial residuals.\n"
        << "  --gbt57-radial-refine-iterations <n>  Radial consistency refinement iterations, default 5.\n"
        << "  --gbt57-radial-refine-max-step <px>   Max per-image translation per refinement pass, default 0.35.\n"
        << "  --gbt57-center-angle-range <deg>      Local angle search half range, default 0.05.\n"
        << "  --gbt57-center-angle-step <deg>       Local angle search step, default 0.01.\n"
        << "  --gbt57-confidence-best-selection     Select the best confidence/gradient point in each field.\n"
        << "  --gbt57-confidence-radius-guard <px>  Radius guard for confidence selection, default 2.5.\n"
        << "  --gbt57-radius-stable-selection       Prefer points closest to the fitted global radius.\n"
        << "  --gbt57-uniform-angle-selection       Assign 25 preset angles to fields before point selection.\n"
        << "  --gbt57-optimize-selected-range       Locally minimize 25-point Rmax-Rmin within preset windows.\n"
        << "  --gbt57-range-candidates <n>          Candidates kept per field for range optimization, default 256.\n"
        << "  --gbt57-range-restarts <n>            Deterministic restarts for range optimization, default 64.\n"
        << "  --gbt57-field-bias-compensation       Correct selected radii by each field's median radial bias.\n"
        << "  --gbt57-local-field-bias-compensation Correct selected radii by nearby same-field edge bias.\n"
        << "  --gbt57-angular-bias-compensation     Correct selected radii by smooth angular residual model.\n"
        << "  --gbt57-angular-bias-order <n>        Fourier order for angular bias model, default 3.\n"
        << "  --gbt57-angular-bias-gain <v>         Gain for angular bias correction, default 1.\n"
        << "  --gbt57-ellipse-normalize             Normalize selected points with a covariance ellipse model.\n"
        << "  --gbt57-selected-point-refine         Refine per-field transforms using the 25 selected points.\n"
        << "  --gbt57-selected-point-refine-iter <n> Iterations for selected-point transform refinement, default 4.\n"
        << "  --gbt57-selected-point-refine-step <px> Max translation per refinement pass, default 0.15.\n"
        << "  --gbt57-selected-point-refine-gain <v> Gain for selected-point transform refinement, default 0.85.\n"
        << "  --gbt57-moderate-auto-refine-profile  Use v14-style moderate selected-point auto-refine with acceptance gate and fallback.\n"
        << "  --gbt57-auto-refine-max-shift <px>    Accept auto-refine only when per-field max shift stays below this limit, default 0.45.\n"
        << "  --gbt57-auto-refine-max-target-delta <deg> Accept auto-refine only when target-angle max deviation stays below this limit, default 4.\n"
        << "  --gbt57-circular-hampel-radius <n>    Apply circular Hampel outlier limiting on final corrected residuals before median filtering.\n"
        << "  --gbt57-circular-hampel-sigma <v>     Sigma threshold for the circular Hampel limiter.\n"
        << "  --gbt57-circular-median-filter-radius <n> Apply a circular median filter of radius n on final corrected residuals before E_P2D evaluation, default 10.\n"
        << "  --gbt57-circular-filter-blend <v>     Blend ratio from raw corrected residuals to circular-filtered residuals, default 0.95.\n"
        << "  --gbt57-supplement-precleanup-candidates Supplement cleaned candidate pool with points removed by circle-edge cleanup.\n"
        << "  --gbt57-soft-spacing-guard           Use soft spacing regularization in non-uniform window mode.\n"
        << "  --gbt57-soft-spacing-max-error <deg> Soft upper limit for spacing max error, default 12.\n"
        << "  --gbt57-soft-spacing-rmse <deg>      Soft upper limit for spacing RMSE, default 4.\n"
        << "  --gbt57-soft-target-max <deg>        Soft upper limit for per-field target-angle max deviation, default 6.\n"
        << "  --gbt57-soft-target-mean <deg>       Soft upper limit for target-angle mean deviation, default 2.5.\n"
        << "  --gbt57-local-bias-half-angle <deg>   Local bias angular half window, default 2.\n"
        << "  --gbt57-window-half-angle <deg>       Single-point selection angular half window, default 7.\n"
        << "  --gbt57-window-half-size <px>         Reported measurement-window half size, default 120.\n"
        << "  --gui-progress                        Emit GUI-friendly progress/image markers and save process images.\n";
}

std::vector<std::string> collectSequentialImagePaths(const std::filesystem::path& inputDir,
                                                     int imageCount,
                                                     int startIndex,
                                                     const std::string& prefix,
                                                     const std::string& extension)
{
    std::vector<std::string> paths;
    paths.reserve(static_cast<std::size_t>(std::max(imageCount, 0)));
    bool allSequentialFilesExist = true;
    for (int i = 0; i < imageCount; ++i) {
        const std::filesystem::path imagePath =
            inputDir / (prefix + std::to_string(startIndex + i) + extension);
        if (!std::filesystem::exists(imagePath)) {
            allSequentialFilesExist = false;
        }
        paths.push_back(imagePath.u8string());
    }
    if (allSequentialFilesExist || imageCount <= 0) {
        return paths;
    }

    std::vector<std::pair<int, std::filesystem::path>> discoveredPaths;
    const auto hasPrefix = [](const std::string& value, const std::string& textPrefix) {
        return value.size() >= textPrefix.size() &&
               value.compare(0, textPrefix.size(), textPrefix) == 0;
    };
    const auto hasSuffix = [](const std::string& value, const std::string& textSuffix) {
        return value.size() >= textSuffix.size() &&
               value.compare(value.size() - textSuffix.size(),
                             textSuffix.size(),
                             textSuffix) == 0;
    };
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().u8string();
        if (!hasPrefix(filename, prefix) || !hasSuffix(filename, extension)) {
            continue;
        }
        const std::size_t numberStart = prefix.size();
        const std::size_t numberLength =
            filename.size() - prefix.size() - extension.size();
        if (numberLength == 0) {
            continue;
        }
        const std::string numberText = filename.substr(numberStart, numberLength);
        if (!std::all_of(numberText.begin(), numberText.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            })) {
            continue;
        }
        discoveredPaths.emplace_back(std::stoi(numberText), entry.path());
    }
    if (static_cast<int>(discoveredPaths.size()) < imageCount) {
        return paths;
    }
    std::sort(discoveredPaths.begin(),
              discoveredPaths.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first < rhs.first;
                  }
                  return lhs.second.u8string() < rhs.second.u8string();
              });
    paths.clear();
    paths.reserve(static_cast<std::size_t>(imageCount));
    for (int i = 0; i < imageCount; ++i) {
        paths.push_back(discoveredPaths[static_cast<std::size_t>(i)].second.u8string());
    }
    return paths;
}

std::string buildGuiProgressImageFileName(const std::string& stage, const std::size_t index)
{
    std::ostringstream stream;
    stream << stage << "_" << std::setw(3) << std::setfill('0') << index << ".png";
    return stream.str();
}

void emitGuiProgressMarker(const std::string& stage,
                           const std::size_t current,
                           const std::size_t total)
{
    std::cout << "__GUI_PROGRESS__|" << stage << "|" << current << "|" << total << std::endl;
}

void emitGuiImageMarker(const std::string& stage,
                        const std::size_t index,
                        const std::size_t total,
                        const std::string& fileName)
{
    std::cout << "__GUI_IMAGE__|" << stage << "|" << index << "|" << total << "|" << fileName
              << std::endl;
}

cv::Mat toGray8U(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.type() == CV_8UC1) {
        return gray;
    }
    cv::Mat gray8;
    gray.convertTo(gray8, CV_8U);
    return gray8;
}

cv::Point2d transformPointByMatrix(const cv::Mat& matrix, const cv::Point2d& point);

double rotationDegFromMatrix(const cv::Mat& matrix)
{
    if (matrix.empty() || matrix.rows < 2 || matrix.cols < 2) {
        return 0.0;
    }
    return std::atan2(matrix.at<double>(1, 0), matrix.at<double>(0, 0)) * 180.0 / CV_PI;
}

bool measurementWindowInsideImage(const cv::Point2d& imagePoint,
                                  const cv::Size& imageSize,
                                  double windowHalfSizePx,
                                  double* marginOut = nullptr)
{
    const double left = imagePoint.x;
    const double top = imagePoint.y;
    const double right = static_cast<double>(imageSize.width - 1) - imagePoint.x;
    const double bottom = static_cast<double>(imageSize.height - 1) - imagePoint.y;
    const double margin = std::min(std::min(left, right), std::min(top, bottom)) - windowHalfSizePx;
    if (marginOut != nullptr) {
        *marginOut = margin;
    }
    return margin >= 0.0;
}

cv::RotatedRect buildGlobalMeasurementWindow(const cv::Mat& transform,
                                             const cv::Point2d& imagePoint,
                                             double windowHalfSizePx)
{
    const cv::Point2d center = transformPointByMatrix(transform, imagePoint);
    const float side = static_cast<float>(std::max(0.0, 2.0 * windowHalfSizePx));
    return cv::RotatedRect(cv::Point2f(static_cast<float>(center.x),
                                       static_cast<float>(center.y)),
                           cv::Size2f(side, side),
                           static_cast<float>(rotationDegFromMatrix(transform)));
}

std::array<cv::Point2d, 4> rotatedRectCorners(const cv::RotatedRect& rect)
{
    std::array<cv::Point2d, 4> corners{};
    std::array<cv::Point2f, 4> corners32{};
    rect.points(corners32.data());
    for (std::size_t i = 0; i < corners.size(); ++i) {
        corners[i] = cv::Point2d(corners32[i].x, corners32[i].y);
    }
    return corners;
}

double rotatedRectIntersectionArea(const cv::RotatedRect& lhs,
                                   const cv::RotatedRect& rhs)
{
    std::vector<cv::Point2f> polygon;
    const int type = cv::rotatedRectangleIntersection(lhs, rhs, polygon);
    if (type == cv::INTERSECT_NONE || polygon.size() < 3) {
        return 0.0;
    }
    return std::abs(cv::contourArea(polygon));
}

EllipseModel fitEllipseModel(const std::vector<cv::Point2d>& points)
{
    EllipseModel model;
    if (points.size() < 20) {
        return model;
    }

    std::vector<cv::Point2f> samples;
    const std::size_t maxSamples = 6000;
    if (points.size() <= maxSamples) {
        samples.reserve(points.size());
        for (const cv::Point2d& point : points) {
            samples.emplace_back(static_cast<float>(point.x),
                                 static_cast<float>(point.y));
        }
    } else {
        samples.reserve(maxSamples);
        const double step =
            static_cast<double>(points.size() - 1) / static_cast<double>(maxSamples - 1);
        for (std::size_t i = 0; i < maxSamples; ++i) {
            const std::size_t index = std::min(points.size() - 1,
                                               static_cast<std::size_t>(std::llround(i * step)));
            samples.emplace_back(static_cast<float>(points[index].x),
                                 static_cast<float>(points[index].y));
        }
    }

    try {
        const cv::RotatedRect ellipse = cv::fitEllipse(samples);
        const double axisA = 0.5 * static_cast<double>(ellipse.size.width);
        const double axisB = 0.5 * static_cast<double>(ellipse.size.height);
        const double major = std::max(axisA, axisB);
        const double minor = std::min(axisA, axisB);
        if (!(major > 1e-6) || !(minor > 1e-6)) {
            return model;
        }
        model.ok = true;
        model.centerX = ellipse.center.x;
        model.centerY = ellipse.center.y;
        model.majorRadiusPx = major;
        model.minorRadiusPx = minor;
        model.angleDeg = axisA >= axisB ? ellipse.angle : ellipse.angle + 90.0;
        model.equivalentRadiusPx = std::sqrt(major * minor);
        model.axisRatio = major > 1e-9 ? minor / major : 1.0;
    } catch (const cv::Exception&) {
        return model;
    }
    return model;
}

double ellipseEquivalentRadiusPx(const EllipseModel& ellipse,
                                 const cv::Point2d& point)
{
    if (!ellipse.ok || !(ellipse.majorRadiusPx > 0.0) || !(ellipse.minorRadiusPx > 0.0)) {
        return 0.0;
    }
    const double angleRad = ellipse.angleDeg * CV_PI / 180.0;
    const double cosA = std::cos(angleRad);
    const double sinA = std::sin(angleRad);
    const double dx = point.x - ellipse.centerX;
    const double dy = point.y - ellipse.centerY;
    const double u = cosA * dx + sinA * dy;
    const double v = -sinA * dx + cosA * dy;
    const double normalized =
        std::sqrt((u * u) / (ellipse.majorRadiusPx * ellipse.majorRadiusPx) +
                  (v * v) / (ellipse.minorRadiusPx * ellipse.minorRadiusPx));
    return normalized * ellipse.equivalentRadiusPx;
}

EllipseRectificationModel buildEllipseRectificationModel(const EllipseModel& ellipse)
{
    EllipseRectificationModel model;
    if (!ellipse.ok ||
        !(ellipse.majorRadiusPx > 1e-6) ||
        !(ellipse.minorRadiusPx > 1e-6) ||
        !(ellipse.equivalentRadiusPx > 1e-6)) {
        return model;
    }
    const double angleRad = ellipse.angleDeg * CV_PI / 180.0;
    model.ok = true;
    model.centerX = ellipse.centerX;
    model.centerY = ellipse.centerY;
    model.cosAngle = std::cos(angleRad);
    model.sinAngle = std::sin(angleRad);
    model.scaleMajor = ellipse.equivalentRadiusPx / ellipse.majorRadiusPx;
    model.scaleMinor = ellipse.equivalentRadiusPx / ellipse.minorRadiusPx;
    model.equivalentRadiusPx = ellipse.equivalentRadiusPx;
    return model;
}

cv::Point2d rectifyPointByEllipse(const EllipseRectificationModel& ellipse,
                                  const cv::Point2d& point)
{
    if (!ellipse.ok) {
        return point;
    }
    const double dx = point.x - ellipse.centerX;
    const double dy = point.y - ellipse.centerY;
    const double major = ellipse.cosAngle * dx + ellipse.sinAngle * dy;
    const double minor = -ellipse.sinAngle * dx + ellipse.cosAngle * dy;
    return {
        major * ellipse.scaleMajor,
        minor * ellipse.scaleMinor
    };
}

cv::Point2d inverseRectifiedDeltaToRaw(const EllipseRectificationModel& ellipse,
                                       const cv::Point2d& rectifiedDelta)
{
    if (!ellipse.ok) {
        return rectifiedDelta;
    }
    const double majorDelta = rectifiedDelta.x / std::max(1e-9, ellipse.scaleMajor);
    const double minorDelta = rectifiedDelta.y / std::max(1e-9, ellipse.scaleMinor);
    return {
        ellipse.cosAngle * majorDelta - ellipse.sinAngle * minorDelta,
        ellipse.sinAngle * majorDelta + ellipse.cosAngle * minorDelta
    };
}

void rebuildEdgeVariantOrders(stitch::EdgeVariants& edge)
{
    edge.x_sorted = edge.raw;
    edge.y_sorted = edge.raw;
    edge.negX_sorted = edge.raw;
    edge.negY_sorted = edge.raw;

    stitch::sortContourByX(edge.x_sorted);
    stitch::sortContourByY(edge.y_sorted);

    for (auto& point : edge.negX_sorted) {
        point.x = -point.x;
    }
    stitch::sortContourByX(edge.negX_sorted);

    for (auto& point : edge.negY_sorted) {
        point.y = -point.y;
    }
    stitch::sortContourByY(edge.negY_sorted);
}

bool isSyntheticEdgePoint(const stitch::EdgeVariants& edge, std::size_t index)
{
    return edge.rawSyntheticFlags.size() == edge.raw.size() &&
           index < edge.rawSyntheticFlags.size() &&
           edge.rawSyntheticFlags[index] != 0;
}

stitch::EdgeVariants filterSyntheticEdgePoints(const stitch::EdgeVariants& edge)
{
    if (edge.rawSyntheticFlags.size() != edge.raw.size() || edge.raw.empty()) {
        return edge;
    }

    stitch::EdgeVariants filtered;
    const bool hasWeights = edge.rawQualityWeights.size() == edge.raw.size();
    const bool hasConfidences = edge.rawConfidences.size() == edge.raw.size();
    const bool hasGradients = edge.rawGradients.size() == edge.raw.size();

    filtered.raw.reserve(edge.raw.size());
    filtered.rawQualityWeights.reserve(hasWeights ? edge.rawQualityWeights.size() : 0);
    filtered.rawConfidences.reserve(hasConfidences ? edge.rawConfidences.size() : 0);
    filtered.rawGradients.reserve(hasGradients ? edge.rawGradients.size() : 0);
    filtered.rawSyntheticFlags.reserve(edge.raw.size());

    for (std::size_t index = 0; index < edge.raw.size(); ++index) {
        if (isSyntheticEdgePoint(edge, index)) {
            continue;
        }
        filtered.raw.push_back(edge.raw[index]);
        if (hasWeights) {
            filtered.rawQualityWeights.push_back(edge.rawQualityWeights[index]);
        }
        if (hasConfidences) {
            filtered.rawConfidences.push_back(edge.rawConfidences[index]);
        }
        if (hasGradients) {
            filtered.rawGradients.push_back(edge.rawGradients[index]);
        }
        filtered.rawSyntheticFlags.push_back(0);
    }

    if (filtered.raw.empty()) {
        return edge;
    }

    rebuildEdgeVariantOrders(filtered);
    return filtered;
}

double quantileOf(std::vector<double> values, double q)
{
    if (values.empty()) {
        return 0.0;
    }
    q = std::clamp(q, 0.0, 1.0);
    const std::size_t index =
        static_cast<std::size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

CircleModel fitCircleLeastSquares(const std::vector<cv::Point2d>& points,
                                  const std::vector<unsigned char>* keepMask = nullptr)
{
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXX = 0.0;
    double sumYY = 0.0;
    double sumXY = 0.0;
    double sumR2 = 0.0;
    double sumXR2 = 0.0;
    double sumYR2 = 0.0;
    int count = 0;

    for (std::size_t i = 0; i < points.size(); ++i) {
        if (keepMask != nullptr && i < keepMask->size() && (*keepMask)[i] == 0) {
            continue;
        }
        const double x = points[i].x;
        const double y = points[i].y;
        if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
        }
        const double r2 = x * x + y * y;
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumYY += y * y;
        sumXY += x * y;
        sumR2 += r2;
        sumXR2 += x * r2;
        sumYR2 += y * r2;
        ++count;
    }

    if (count < 20) {
        return {};
    }

    const cv::Matx33d normal(sumXX, sumXY, sumX,
                             sumXY, sumYY, sumY,
                             sumX, sumY, static_cast<double>(count));
    const cv::Vec3d rhs(-sumXR2, -sumYR2, -sumR2);
    cv::Vec3d solution;
    if (!cv::solve(normal, rhs, solution, cv::DECOMP_SVD)) {
        return {};
    }

    const double a = solution[0];
    const double b = solution[1];
    const double c = solution[2];
    const double centerX = -0.5 * a;
    const double centerY = -0.5 * b;
    const double radiusSquared = centerX * centerX + centerY * centerY - c;
    if (!std::isfinite(radiusSquared) || radiusSquared <= 0.0) {
        return {};
    }

    CircleModel model;
    model.ok = true;
    model.centerX = centerX;
    model.centerY = centerY;
    model.radius = std::sqrt(radiusSquared);
    return model;
}

CircleModel fitCircleMinRange(const std::vector<cv::Point2d>& points, double* rangeOut = nullptr)
{
    CircleModel fit = fitCircleLeastSquares(points);
    if (!fit.ok) {
        return fit;
    }

    auto evaluateCenter = [&](const cv::Point2d& center,
                              double& range,
                              double& minRadius,
                              double& maxRadius,
                              double& radiusMid) {
        range = std::numeric_limits<double>::infinity();
        minRadius = std::numeric_limits<double>::infinity();
        maxRadius = 0.0;
        int count = 0;
        for (const cv::Point2d& point : points) {
            const double radius = std::hypot(point.x - center.x, point.y - center.y);
            if (!std::isfinite(radius)) {
                return false;
            }
            minRadius = std::min(minRadius, radius);
            maxRadius = std::max(maxRadius, radius);
            ++count;
        }
        if (count < 20) {
            return false;
        }
        range = maxRadius - minRadius;
        radiusMid = 0.5 * (minRadius + maxRadius);
        return std::isfinite(range);
    };

    cv::Point2d center(fit.centerX, fit.centerY);
    double bestRange = std::numeric_limits<double>::infinity();
    double bestRadiusMid = fit.radius;
    double bestMinRadius = 0.0;
    double bestMaxRadius = 0.0;
    if (!evaluateCenter(center, bestRange, bestMinRadius, bestMaxRadius, bestRadiusMid)) {
        return fit;
    }

    const std::array<double, 5> steps{0.20, 0.10, 0.05, 0.02, 0.01};
    for (double step : steps) {
        bool improved = true;
        for (int iteration = 0; iteration < 12 && improved; ++iteration) {
            improved = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const cv::Point2d candidateCenter(center.x + static_cast<double>(dx) * step,
                                                      center.y + static_cast<double>(dy) * step);
                    double candidateRange = 0.0;
                    double candidateMinRadius = 0.0;
                    double candidateMaxRadius = 0.0;
                    double candidateRadiusMid = 0.0;
                    if (!evaluateCenter(candidateCenter,
                                        candidateRange,
                                        candidateMinRadius,
                                        candidateMaxRadius,
                                        candidateRadiusMid)) {
                        continue;
                    }
                    if (candidateRange + 1e-12 < bestRange) {
                        center = candidateCenter;
                        bestRange = candidateRange;
                        bestRadiusMid = candidateRadiusMid;
                        improved = true;
                    }
                }
            }
        }
    }

    fit.centerX = center.x;
    fit.centerY = center.y;
    fit.radius = bestRadiusMid;
    if (rangeOut != nullptr) {
        *rangeOut = bestRange;
    }
    return fit;
}

CircleModel fitCircleFromThreePoints(const cv::Point2d& p1,
                                     const cv::Point2d& p2,
                                     const cv::Point2d& p3)
{
    const double a = p1.x * (p2.y - p3.y) -
                     p1.y * (p2.x - p3.x) +
                     p2.x * p3.y -
                     p3.x * p2.y;
    if (std::abs(a) < 1e-6) {
        return {};
    }

    const double p1Sq = p1.x * p1.x + p1.y * p1.y;
    const double p2Sq = p2.x * p2.x + p2.y * p2.y;
    const double p3Sq = p3.x * p3.x + p3.y * p3.y;
    const double centerX =
        (p1Sq * (p2.y - p3.y) +
         p2Sq * (p3.y - p1.y) +
         p3Sq * (p1.y - p2.y)) /
        (2.0 * a);
    const double centerY =
        (p1Sq * (p3.x - p2.x) +
         p2Sq * (p1.x - p3.x) +
         p3Sq * (p2.x - p1.x)) /
        (2.0 * a);
    const double radius = std::hypot(p1.x - centerX, p1.y - centerY);
    if (!std::isfinite(radius) || radius <= 0.0) {
        return {};
    }

    CircleModel model;
    model.ok = true;
    model.centerX = centerX;
    model.centerY = centerY;
    model.radius = radius;
    return model;
}

std::vector<cv::Point2d> uniformPointSample(const std::vector<cv::Point2d>& points,
                                            std::size_t maxCount)
{
    if (points.size() <= maxCount) {
        return points;
    }
    std::vector<cv::Point2d> sampled;
    sampled.reserve(maxCount);
    for (std::size_t i = 0; i < maxCount; ++i) {
        const std::size_t index =
            std::min(points.size() - 1,
                     static_cast<std::size_t>(
                         std::llround(static_cast<double>(i) *
                                      static_cast<double>(points.size() - 1) /
                                      static_cast<double>(maxCount - 1))));
        sampled.push_back(points[index]);
    }
    return sampled;
}

double normalizeAngleDeg(double angle);
double angleDistanceForwardDeg(double fromDeg, double toDeg);
double angleOfPointDeg(const cv::Point2d& point, const CircleModel& circle);

CircleRansacResult fitCircleRansacWithRadiusPrior(const std::vector<cv::Point2d>& points,
                                                  double radiusPrior,
                                                  double radiusToleranceRatio,
                                                  double inlierThresholdPx)
{
    CircleRansacResult best;
    if (points.size() < 30 || radiusPrior <= 0.0) {
        return best;
    }

    const std::vector<cv::Point2d> samples = uniformPointSample(points, 72);
    if (samples.size() < 30) {
        return best;
    }

    double bestCost = std::numeric_limits<double>::infinity();
    const double minRadius = radiusPrior * std::max(0.20, 1.0 - radiusToleranceRatio);
    const double maxRadius = radiusPrior * (1.0 + radiusToleranceRatio);
    const double scoreThreshold = std::max(1.0, inlierThresholdPx);

    for (std::size_t i = 0; i + 2 < samples.size(); ++i) {
        for (std::size_t j = i + 1; j + 1 < samples.size(); ++j) {
            for (std::size_t k = j + 1; k < samples.size(); ++k) {
                const CircleModel candidate =
                    fitCircleFromThreePoints(samples[i], samples[j], samples[k]);
                if (!candidate.ok ||
                    candidate.radius < minRadius ||
                    candidate.radius > maxRadius) {
                    continue;
                }

                int inliers = 0;
                double sumAbs = 0.0;
                for (const cv::Point2d& point : samples) {
                    const double residual =
                        std::abs(std::hypot(point.x - candidate.centerX,
                                            point.y - candidate.centerY) -
                                 candidate.radius);
                    if (residual <= scoreThreshold) {
                        ++inliers;
                        sumAbs += residual;
                    }
                }
                if (inliers < 12) {
                    continue;
                }

                const double meanAbs = sumAbs / static_cast<double>(inliers);
                const double radiusPenalty =
                    std::abs(candidate.radius - radiusPrior) /
                    std::max(1.0, radiusPrior * radiusToleranceRatio);
                const double cost =
                    -static_cast<double>(inliers) +
                    0.05 * meanAbs +
                    0.50 * radiusPenalty;
                if (cost < bestCost) {
                    bestCost = cost;
                    best.model = candidate;
                    best.inlierCount = inliers;
                    best.meanAbsResidual = meanAbs;
                }
            }
        }
    }

    if (!best.model.ok) {
        return best;
    }

    std::vector<unsigned char> keep(points.size(), 0);
    std::size_t fullInliers = 0;
    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
        const cv::Point2d& point = points[pointIndex];
        const double residual =
            std::abs(std::hypot(point.x - best.model.centerX,
                                point.y - best.model.centerY) -
                     best.model.radius);
        if (residual <= std::max(2.0, inlierThresholdPx * 1.5)) {
            keep[pointIndex] = 1;
            ++fullInliers;
        }
    }
    if (fullInliers >= 30) {
        const CircleModel refined = fitCircleLeastSquares(points, &keep);
        if (refined.ok &&
            refined.radius >= minRadius &&
            refined.radius <= maxRadius) {
            best.model = refined;
            best.inlierCount = static_cast<int>(fullInliers);
        }
    }

    return best;
}

int completeCircleGaps(stitch::EdgeVariants& edge,
                       const CircleModel& circle,
                       const CircleEdgeCleanupConfig& config)
{
    if (!config.completeCircleGaps || !circle.ok || edge.raw.size() < 20) {
        return 0;
    }

    std::vector<double> angles;
    angles.reserve(edge.raw.size());
    for (const cv::Point2d& point : edge.raw) {
        angles.push_back(angleOfPointDeg(point, circle));
    }
    std::sort(angles.begin(), angles.end());
    if (angles.size() < 2) {
        return 0;
    }

    std::vector<cv::Point2d> additions;
    additions.reserve(static_cast<std::size_t>(config.completionMaxPointsPerImage));
    const double stepDeg = std::max(0.05, config.completionAngularStepDeg);
    const double gapThresholdDeg = std::max(stepDeg * 2.0, config.completionGapThresholdDeg);

    for (std::size_t i = 0; i < angles.size(); ++i) {
        const double current = angles[i];
        const double next = (i + 1 < angles.size()) ? angles[i + 1] : angles[0];
        const double gap = angleDistanceForwardDeg(current, next);
        if (gap <= gapThresholdDeg) {
            continue;
        }

        const double fillStart = current + stepDeg;
        const double fillEnd = current + gap - stepDeg;
        for (double angle = fillStart;
             angle <= fillEnd &&
             additions.size() < static_cast<std::size_t>(config.completionMaxPointsPerImage);
             angle += stepDeg) {
            const double rad = normalizeAngleDeg(angle) * CV_PI / 180.0;
            additions.push_back({
                circle.centerX + circle.radius * std::cos(rad),
                circle.centerY + circle.radius * std::sin(rad)
            });
        }
        if (additions.size() >= static_cast<std::size_t>(config.completionMaxPointsPerImage)) {
            break;
        }
    }

    const int added = static_cast<int>(additions.size());
    for (const cv::Point2d& point : additions) {
        edge.raw.push_back(point);
        edge.rawQualityWeights.push_back(0.10);
        edge.rawConfidences.push_back(0.0);
        edge.rawGradients.push_back(0.0);
        edge.rawSyntheticFlags.push_back(1);
    }
    if (added > 0) {
        rebuildEdgeVariantOrders(edge);
    }
    return added;
}

std::string csvEscapeLocal(const std::string& text)
{
    if (text.find_first_of(",\"\n\r") == std::string::npos) {
        return text;
    }
    std::string escaped = "\"";
    for (char ch : text) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

cv::Point2d transformPointByMatrix(const cv::Mat& matrix, const cv::Point2d& point)
{
    return {
        matrix.at<double>(0, 0) * point.x +
            matrix.at<double>(0, 1) * point.y +
            matrix.at<double>(0, 2),
        matrix.at<double>(1, 0) * point.x +
            matrix.at<double>(1, 1) * point.y +
            matrix.at<double>(1, 2)
    };
}

double normalizeAngleDeg(double angle)
{
    while (angle < 0.0) {
        angle += 360.0;
    }
    while (angle >= 360.0) {
        angle -= 360.0;
    }
    return angle;
}

double angleDistanceForwardDeg(double fromDeg, double toDeg)
{
    double delta = normalizeAngleDeg(toDeg) - normalizeAngleDeg(fromDeg);
    if (delta <= 0.0) {
        delta += 360.0;
    }
    return delta;
}

double angleOfPointDeg(const cv::Point2d& point, const CircleModel& circle)
{
    return normalizeAngleDeg(std::atan2(point.y - circle.centerY,
                                        point.x - circle.centerX) *
                             180.0 / CV_PI);
}

double circularDeltaDeg(double a, double b)
{
    double delta = std::fmod(a - b + 540.0, 360.0) - 180.0;
    if (delta < -180.0) {
        delta += 360.0;
    }
    return delta;
}

double circularAbsDeltaDeg(double a, double b)
{
    return std::abs(circularDeltaDeg(a, b));
}

CircleModel fitCircleRobustForP2d(const std::vector<cv::Point2d>& points);

cv::Mat identityTransform()
{
    return cv::Mat::eye(3, 3, CV_64F);
}

FixedRadiusCircleFit fitCircleCenterWithFixedRadius(const std::vector<cv::Point2d>& points,
                                                    const CircleModel& initial,
                                                    double fixedRadius)
{
    FixedRadiusCircleFit fit;
    if (points.size() < 20 || !initial.ok || fixedRadius <= 0.0) {
        return fit;
    }

    cv::Point2d center(initial.centerX, initial.centerY);
    std::vector<unsigned char> keep(points.size(), 1);

    for (int iteration = 0; iteration < 8; ++iteration) {
        cv::Matx22d normal(0.0, 0.0, 0.0, 0.0);
        cv::Vec2d rhs(0.0, 0.0);
        int used = 0;

        for (std::size_t i = 0; i < points.size(); ++i) {
            if (keep[i] == 0) {
                continue;
            }
            const cv::Point2d delta = center - points[i];
            const double distance = std::hypot(delta.x, delta.y);
            if (distance <= 1e-9 || !std::isfinite(distance)) {
                continue;
            }
            const cv::Vec2d jac(delta.x / distance, delta.y / distance);
            const double residual = distance - fixedRadius;
            normal(0, 0) += jac[0] * jac[0];
            normal(0, 1) += jac[0] * jac[1];
            normal(1, 0) += jac[1] * jac[0];
            normal(1, 1) += jac[1] * jac[1];
            rhs[0] += -jac[0] * residual;
            rhs[1] += -jac[1] * residual;
            ++used;
        }

        if (used < 20) {
            return fit;
        }

        cv::Vec2d step;
        if (!cv::solve(normal, rhs, step, cv::DECOMP_SVD)) {
            return fit;
        }
        center.x += step[0];
        center.y += step[1];
        if (std::hypot(step[0], step[1]) < 1e-5) {
            break;
        }

        std::vector<double> residuals;
        residuals.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (keep[i] == 0) {
                continue;
            }
            residuals.push_back(std::hypot(points[i].x - center.x,
                                           points[i].y - center.y) -
                                fixedRadius);
        }
        if (residuals.size() < 20) {
            continue;
        }
        const double median = quantileOf(residuals, 0.5);
        std::vector<double> deviations;
        deviations.reserve(residuals.size());
        for (double residual : residuals) {
            deviations.push_back(std::abs(residual - median));
        }
        const double scale = std::max(0.10, 1.4826 * quantileOf(deviations, 0.5));
        const double threshold = std::max(2.0, 4.0 * scale);
        int kept = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (keep[i] == 0) {
                continue;
            }
            const double residual =
                std::hypot(points[i].x - center.x, points[i].y - center.y) -
                fixedRadius;
            if (std::abs(residual - median) > threshold) {
                keep[i] = 0;
            } else {
                ++kept;
            }
        }
        if (kept < 20) {
            return fit;
        }
    }

    double sumAbs = 0.0;
    double sumSq = 0.0;
    double maxAbs = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (keep[i] == 0) {
            continue;
        }
        const double residual =
            std::hypot(points[i].x - center.x, points[i].y - center.y) -
            fixedRadius;
        const double absResidual = std::abs(residual);
        sumAbs += absResidual;
        sumSq += residual * residual;
        maxAbs = std::max(maxAbs, absResidual);
        ++count;
    }
    if (count < 20) {
        return fit;
    }

    fit.model.ok = true;
    fit.model.centerX = center.x;
    fit.model.centerY = center.y;
    fit.model.radius = fixedRadius;
    fit.pointCount = count;
    fit.meanAbsPx = sumAbs / static_cast<double>(count);
    fit.rmsePx = std::sqrt(sumSq / static_cast<double>(count));
    fit.maxAbsPx = maxAbs;
    return fit;
}

stitch::StitchingResult buildLocalCircleFrameStitching(
    const std::vector<cv::Mat>& images,
    const std::vector<stitch::EdgeVariants>& edges)
{
    stitch::StitchingResult stitching;
    if (images.empty() || edges.size() != images.size()) {
        return stitching;
    }

    stitching.canvas = images.front().clone();
    stitching.globalTransform = identityTransform();
    stitching.imageTransforms.reserve(images.size());

    std::vector<CircleModel> circles;
    circles.reserve(edges.size());
    std::vector<double> radii;
    radii.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        CircleModel circle = fitCircleRobustForP2d(edge.raw);
        circles.push_back(circle);
        if (circle.ok) {
            radii.push_back(circle.radius);
        }
    }

    const double commonRadius = radii.empty() ? 0.0 : quantileOf(radii, 0.5);
    for (std::size_t i = 0; i < images.size(); ++i) {
        cv::Mat transform = identityTransform();
        if (i < circles.size() && circles[i].ok) {
            transform.at<double>(0, 2) = -circles[i].centerX;
            transform.at<double>(1, 2) = -circles[i].centerY;
        }
        if (commonRadius > 0.0 && i < circles.size() && circles[i].ok && circles[i].radius > 0.0) {
            const double scale = commonRadius / circles[i].radius;
            transform.at<double>(0, 0) = scale;
            transform.at<double>(1, 1) = scale;
            transform.at<double>(0, 2) *= scale;
            transform.at<double>(1, 2) *= scale;
        }
        stitching.imageTransforms.push_back(transform);
    }
    return stitching;
}

double preferredNormalRmseLocal(const stitch::TransformResult& transform)
{
    const stitch::ResidualStatistics& normal =
        transform.metrics.normalInlier.valid() ? transform.metrics.normalInlier
                                               : transform.metrics.normalAll;
    return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
}

cv::Mat circleCenterTransform(const CircleModel& circle,
                              double angleDeg,
                              double scale,
                              const cv::Point2d& globalCenter)
{
    cv::Mat transform = identityTransform();
    const double angleRad = angleDeg * CV_PI / 180.0;
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    transform.at<double>(0, 0) = scale * c;
    transform.at<double>(0, 1) = scale * s;
    transform.at<double>(1, 0) = -scale * s;
    transform.at<double>(1, 1) = scale * c;
    transform.at<double>(0, 2) =
        globalCenter.x -
        (transform.at<double>(0, 0) * circle.centerX +
         transform.at<double>(0, 1) * circle.centerY);
    transform.at<double>(1, 2) =
        globalCenter.y -
        (transform.at<double>(1, 0) * circle.centerX +
         transform.at<double>(1, 1) * circle.centerY);
    return transform;
}

void updatePointBounds(cv::Rect2d& bounds, const cv::Point2d& point)
{
    if (bounds.width < 0.0 || bounds.height < 0.0) {
        bounds = cv::Rect2d(point.x, point.y, 0.0, 0.0);
        return;
    }
    const double minX = std::min(bounds.x, point.x);
    const double minY = std::min(bounds.y, point.y);
    const double maxX = std::max(bounds.x + bounds.width, point.x);
    const double maxY = std::max(bounds.y + bounds.height, point.y);
    bounds = cv::Rect2d(minX, minY, maxX - minX, maxY - minY);
}

void applyLeftTranslation(std::vector<cv::Mat>& transforms, const cv::Point2d& translation)
{
    cv::Mat offset = identityTransform();
    offset.at<double>(0, 2) = translation.x;
    offset.at<double>(1, 2) = translation.y;
    for (cv::Mat& transform : transforms) {
        transform = offset * transform;
    }
}

std::vector<double> estimateCircleCenterGlobalAngles(
    const std::vector<cv::Mat>& images,
    const std::vector<stitch::EdgeVariants>& edges,
    const std::vector<CircleModel>& circles,
    const Gbt57P2dConfig& config,
    std::vector<CircleCenterGlobalReport>& reports)
{
    std::vector<double> angles(images.size(), 0.0);
    if (!config.circleCenterLocalAngleSearch || images.size() < 2 ||
        edges.size() != images.size() || circles.size() != images.size()) {
        return angles;
    }

    const double range = std::max(0.0, config.centerGlobalAngleSearchRangeDeg);
    const double step = std::max(0.001, config.centerGlobalAngleSearchStepDeg);
    const double maxAbsRotation =
        std::max(range, config.centerGlobalMaxAbsRotationDeg);

    for (std::size_t i = 1; i < images.size(); ++i) {
        angles[i] = angles[i - 1];
        if (!circles[i - 1].ok || !circles[i].ok) {
            continue;
        }

        double searchRangeX = 0.0;
        double searchRangeY = 0.0;
        const cv::Point2d center(images[i].cols * 0.5, images[i].rows * 0.5);
        const cv::Point2d circleShift(circles[i - 1].centerX - circles[i].centerX,
                                      circles[i - 1].centerY - circles[i].centerY);
        const stitch::TransformResult local =
            stitch::matchOnePair(edges[i - 1],
                                 edges[i],
                                 center,
                                 circleShift.x,
                                 circleShift.y,
                                 0.0,
                                 true,
                                 std::max(config.centerGlobalLocalSearchPrimaryPx,
                                          config.centerGlobalLocalSearchPerpPx),
                                 stitch::MotionPriorDirection::Auto,
                                 -range,
                                 range,
                                 step,
                                 0.05,
                                 0.25,
                                 true,
                                 config.centerGlobalLocalSearchPrimaryPx,
                                 config.centerGlobalLocalSearchPerpPx,
                                 searchRangeX,
                                 searchRangeY);
        const double normalRmse = preferredNormalRmseLocal(local);
        const bool accepted =
            std::isfinite(normalRmse) &&
            normalRmse <= config.centerGlobalAcceptNormalRmsePx &&
            local.metrics.inlierCount >= 20 &&
            std::abs(local.da) <= range + 1e-9;
        if (i < reports.size()) {
            reports[i].relativeAngleDeg = local.da;
            reports[i].localNormalRmsePx = normalRmse;
            reports[i].localInlierCount = local.metrics.inlierCount;
            reports[i].localAngleAccepted = accepted;
        }
        if (accepted) {
            angles[i] =
                std::clamp(angles[i - 1] + local.da,
                           -maxAbsRotation,
                           maxAbsRotation);
        }
    }

    return angles;
}

void refineCircleCenterRadialConsistency(
    const std::vector<stitch::EdgeVariants>& edges,
    std::vector<cv::Mat>& transforms,
    const Gbt57P2dConfig& config)
{
    if (!config.circleCenterRadialConsistencyRefine ||
        edges.empty() ||
        edges.size() != transforms.size() ||
        config.radialConsistencyIterations <= 0 ||
        config.radialConsistencyMaxStepPx <= 0.0) {
        return;
    }

    const int iterations = std::max(0, config.radialConsistencyIterations);
    const double maxStep = std::max(0.0, config.radialConsistencyMaxStepPx);
    const double residualGate = std::max(0.2, config.radialConsistencyResidualGatePx);

    std::size_t reserveCount = 0;
    for (const stitch::EdgeVariants& edge : edges) {
        reserveCount += edge.raw.size();
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<cv::Point2d> globalPoints;
        globalPoints.reserve(reserveCount);
        for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
            for (const cv::Point2d& point : edges[imageIndex].raw) {
                globalPoints.push_back(transformPointByMatrix(transforms[imageIndex], point));
            }
        }

        const CircleModel circle = fitCircleRobustForP2d(globalPoints);
        if (!circle.ok || circle.radius <= 0.0) {
            break;
        }

        double maxApplied = 0.0;
        for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
            const stitch::EdgeVariants& edge = edges[imageIndex];
            if (edge.raw.size() < 30) {
                continue;
            }

            const bool hasWeights = edge.rawQualityWeights.size() == edge.raw.size();
            double a00 = 0.0;
            double a01 = 0.0;
            double a11 = 0.0;
            double b0 = 0.0;
            double b1 = 0.0;
            double weightSum = 0.0;
            int sampleCount = 0;

            for (std::size_t pointIndex = 0; pointIndex < edge.raw.size(); ++pointIndex) {
                const cv::Point2d global =
                    transformPointByMatrix(transforms[imageIndex], edge.raw[pointIndex]);
                const double dx = global.x - circle.centerX;
                const double dy = global.y - circle.centerY;
                const double radius = std::hypot(dx, dy);
                if (radius <= 1e-9 || !std::isfinite(radius)) {
                    continue;
                }

                const double residual = radius - circle.radius;
                if (!std::isfinite(residual) || std::abs(residual) > residualGate) {
                    continue;
                }

                const double nx = dx / radius;
                const double ny = dy / radius;
                double weight = hasWeights
                                    ? std::clamp(edge.rawQualityWeights[pointIndex], 0.05, 4.0)
                                    : 1.0;
                weight *= 1.0 / (1.0 + std::abs(residual) / 0.35);

                a00 += weight * nx * nx;
                a01 += weight * nx * ny;
                a11 += weight * ny * ny;
                b0 += -weight * residual * nx;
                b1 += -weight * residual * ny;
                weightSum += weight;
                ++sampleCount;
            }

            if (sampleCount < 30 || weightSum <= 1e-9) {
                continue;
            }

            const double damping = 0.02 * weightSum + 1e-6;
            a00 += damping;
            a11 += damping;
            const double det = a00 * a11 - a01 * a01;
            if (std::abs(det) <= 1e-12) {
                continue;
            }

            cv::Point2d correction{
                (b0 * a11 - b1 * a01) / det,
                (a00 * b1 - a01 * b0) / det
            };
            const double norm = cv::norm(correction);
            if (!std::isfinite(norm) || norm <= 1e-12) {
                continue;
            }
            if (norm > maxStep) {
                correction *= maxStep / norm;
            }

            transforms[imageIndex].at<double>(0, 2) += correction.x;
            transforms[imageIndex].at<double>(1, 2) += correction.y;
            maxApplied = std::max(maxApplied, cv::norm(correction));
        }

        if (maxApplied < 1e-4) {
            break;
        }
    }
}

stitch::StitchingResult buildCircleCenterGlobalStitching(
    const std::vector<cv::Mat>& images,
    const std::vector<stitch::EdgeVariants>& edges,
    const std::vector<std::string>& imagePaths,
    const Gbt57P2dConfig& config,
    std::vector<CircleCenterGlobalReport>& reports)
{
    stitch::StitchingResult stitching;
    if (images.empty() || edges.size() != images.size()) {
        return stitching;
    }

    std::vector<stitch::EdgeVariants> measurementEdges;
    measurementEdges.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        measurementEdges.push_back(filterSyntheticEdgePoints(edge));
    }

    std::vector<CircleModel> circles;
    circles.reserve(edges.size());
    std::vector<double> radii;
    radii.reserve(edges.size());
    reports.clear();
    reports.reserve(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const CircleModel circle = fitCircleRobustForP2d(measurementEdges[i].raw);
        circles.push_back(circle);
        if (circle.ok) {
            radii.push_back(circle.radius);
        }

        CircleCenterGlobalReport report;
        report.imageIndex = static_cast<int>(i);
        report.imageName =
            i < imagePaths.size()
                ? std::filesystem::path(imagePaths[i]).filename().u8string()
                : std::string{};
        report.fitted = circle.ok;
        report.centerX = circle.centerX;
        report.centerY = circle.centerY;
        report.radius = circle.radius;
        reports.push_back(report);
    }

    if (radii.empty()) {
        return stitching;
    }

    const double commonRadius = quantileOf(radii, 0.5);
    if (config.circleCenterFixedRadius) {
        for (std::size_t i = 0; i < circles.size(); ++i) {
            if (!circles[i].ok) {
                continue;
            }
            const FixedRadiusCircleFit fixed =
                fitCircleCenterWithFixedRadius(measurementEdges[i].raw, circles[i], commonRadius);
            if (!fixed.model.ok) {
                continue;
            }
            circles[i] = fixed.model;
            reports[i].centerX = fixed.model.centerX;
            reports[i].centerY = fixed.model.centerY;
            reports[i].radius = fixed.model.radius;
            reports[i].fixedRadiusCenterUsed = true;
            reports[i].fixedRadiusRmsePx = fixed.rmsePx;
            reports[i].fixedRadiusMeanAbsPx = fixed.meanAbsPx;
            reports[i].fixedRadiusMaxAbsPx = fixed.maxAbsPx;
        }
    }
    const std::vector<double> angles =
        estimateCircleCenterGlobalAngles(images, measurementEdges, circles, config, reports);

    stitching.globalTransform = identityTransform();
    stitching.imageTransforms.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        double scale = 1.0;
        if (config.circleCenterNormalizeRadius && circles[i].ok &&
            circles[i].radius > 0.0 && commonRadius > 0.0) {
            scale = commonRadius / circles[i].radius;
        }
        const double angleDeg = i < angles.size() ? angles[i] : 0.0;
        cv::Mat transform =
            circles[i].ok
                ? circleCenterTransform(circles[i], angleDeg, scale, cv::Point2d(0.0, 0.0))
                : identityTransform();
        stitching.imageTransforms.push_back(transform);
        reports[i].commonRadius = commonRadius;
        reports[i].scale = scale;
        reports[i].angleDeg = angleDeg;
    }

    refineCircleCenterRadialConsistency(measurementEdges,
                                        stitching.imageTransforms,
                                        config);

    cv::Rect2d bounds(0.0, 0.0, -1.0, -1.0);
    for (std::size_t imageIndex = 0; imageIndex < measurementEdges.size(); ++imageIndex) {
        for (const cv::Point2d& point : measurementEdges[imageIndex].raw) {
            updatePointBounds(bounds,
                              transformPointByMatrix(stitching.imageTransforms[imageIndex],
                                                     point));
        }
    }
    if (bounds.width < 0.0 || bounds.height < 0.0) {
        return stitching;
    }

    constexpr double kPaddingPx = 120.0;
    const cv::Point2d offset(kPaddingPx - bounds.x, kPaddingPx - bounds.y);
    applyLeftTranslation(stitching.imageTransforms, offset);
    stitching.globalTransform = stitching.imageTransforms.back().clone();
    for (std::size_t i = 0; i < stitching.imageTransforms.size() && i < reports.size(); ++i) {
        reports[i].txPx = stitching.imageTransforms[i].at<double>(0, 2);
        reports[i].tyPx = stitching.imageTransforms[i].at<double>(1, 2);
    }

    const int canvasWidth =
        std::max(1, static_cast<int>(std::ceil(bounds.width + 2.0 * kPaddingPx)));
    const int canvasHeight =
        std::max(1, static_cast<int>(std::ceil(bounds.height + 2.0 * kPaddingPx)));
    stitching.canvas =
        cv::Mat(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    return stitching;
}

std::string buildCircleCenterGlobalCsv(const std::vector<CircleCenterGlobalReport>& reports)
{
    std::ostringstream stream;
    stream << "image_index,image_name,fitted,center_x_px,center_y_px,radius_px,"
              "common_radius_px,scale,fixed_radius_center_used,fixed_radius_rmse_px,"
              "fixed_radius_mean_abs_px,fixed_radius_max_abs_px,"
              "angle_deg,relative_angle_deg,local_normal_rmse_px,"
              "local_inlier_count,local_angle_accepted,tx_px,ty_px\n";
    for (const CircleCenterGlobalReport& report : reports) {
        stream << report.imageIndex << ","
               << csvEscapeLocal(report.imageName) << ","
               << (report.fitted ? 1 : 0) << ","
               << report.centerX << ","
               << report.centerY << ","
               << report.radius << ","
               << report.commonRadius << ","
               << report.scale << ","
               << (report.fixedRadiusCenterUsed ? 1 : 0) << ","
               << report.fixedRadiusRmsePx << ","
               << report.fixedRadiusMeanAbsPx << ","
               << report.fixedRadiusMaxAbsPx << ","
               << report.angleDeg << ","
               << report.relativeAngleDeg << ","
               << report.localNormalRmsePx << ","
               << report.localInlierCount << ","
               << (report.localAngleAccepted ? 1 : 0) << ","
               << report.txPx << ","
               << report.tyPx << "\n";
    }
    return stream.str();
}

CircleModel fitCircleRobustForP2d(const std::vector<cv::Point2d>& points)
{
    if (points.size() < 20) {
        return {};
    }

    std::vector<unsigned char> keep(points.size(), 1);
    CircleModel model;
    for (int iteration = 0; iteration < 4; ++iteration) {
        model = fitCircleLeastSquares(points, &keep);
        if (!model.ok) {
            return {};
        }

        std::vector<double> residuals;
        residuals.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (!keep[i]) {
                continue;
            }
            const double radius =
                std::hypot(points[i].x - model.centerX, points[i].y - model.centerY);
            residuals.push_back(radius - model.radius);
        }
        if (residuals.size() < 20) {
            break;
        }

        const double median = quantileOf(residuals, 0.5);
        std::vector<double> deviations;
        deviations.reserve(residuals.size());
        for (double residual : residuals) {
            deviations.push_back(std::abs(residual - median));
        }
        const double scale = std::max(0.10, 1.4826 * quantileOf(deviations, 0.5));
        const double threshold = std::max(2.0, 4.0 * scale);

        std::size_t kept = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (!keep[i]) {
                continue;
            }
            const double radius =
                std::hypot(points[i].x - model.centerX, points[i].y - model.centerY);
            const double residual = radius - model.radius;
            if (std::abs(residual - median) > threshold) {
                keep[i] = 0;
            } else {
                ++kept;
            }
        }
        if (kept < 20) {
            std::fill(keep.begin(), keep.end(), 1);
            break;
        }
    }
    return fitCircleLeastSquares(points, &keep);
}

double effectivePixelSizeUm(const std::vector<cv::Mat>& images,
                            const pinjie::standard_sphere_loop::StandardSphereLoopConfig& config)
{
    if (config.pixelSizeMm > 0.0) {
        return config.pixelSizeMm * 1000.0;
    }
    if (images.empty()) {
        return 0.0;
    }
    std::vector<double> scales;
    if (config.horizontalFieldOfViewMm > 0.0 && images.front().cols > 0) {
        scales.push_back(config.horizontalFieldOfViewMm /
                         static_cast<double>(images.front().cols));
    }
    if (config.verticalFieldOfViewMm > 0.0 && images.front().rows > 0) {
        scales.push_back(config.verticalFieldOfViewMm /
                         static_cast<double>(images.front().rows));
    }
    if (scales.empty()) {
        return 0.0;
    }
    return std::accumulate(scales.begin(), scales.end(), 0.0) /
           static_cast<double>(scales.size()) * 1000.0;
}

double circleDiameterPx(const CircleModel& circle)
{
    return circle.ok && circle.radius > 0.0 ? 2.0 * circle.radius : 0.0;
}

double sphereCalibratedPixelSizeUm(double sphereDiameterMm,
                                   const CircleModel& circle)
{
    const double diameterPx = circleDiameterPx(circle);
    if (!(sphereDiameterMm > 0.0) || !(diameterPx > 0.0)) {
        return 0.0;
    }
    return sphereDiameterMm * 1000.0 / diameterPx;
}

double radialRmseToCircle(const stitch::EdgeVariants& edge,
                          const cv::Mat& transform,
                          const CircleModel& circle)
{
    if (!circle.ok || edge.raw.empty() || transform.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double sumSq = 0.0;
    double weightSum = 0.0;
    const bool hasWeights = edge.rawQualityWeights.size() == edge.raw.size();
    for (std::size_t i = 0; i < edge.raw.size(); ++i) {
        const cv::Point2d global = transformPointByMatrix(transform, edge.raw[i]);
        const double residual =
            std::hypot(global.x - circle.centerX, global.y - circle.centerY) - circle.radius;
        const double weight =
            hasWeights ? std::clamp(edge.rawQualityWeights[i], 0.05, 4.0) : 1.0;
        sumSq += weight * residual * residual;
        weightSum += weight;
    }
    if (weightSum <= 1e-9) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(sumSq / weightSum);
}

bool rescueLastImageTransformByGlobalCircle(stitch::StitchingResult& stitching,
                                            const std::vector<cv::Mat>& images,
                                            const std::vector<stitch::EdgeVariants>& edges,
                                            std::string* messageOut = nullptr)
{
    if (images.size() < 2 ||
        edges.size() != images.size() ||
        stitching.imageTransforms.size() < images.size()) {
        return false;
    }

    std::vector<cv::Point2d> pooledPoints;
    std::size_t reserveCount = 0;
    for (std::size_t imageIndex = 0; imageIndex + 1 < edges.size(); ++imageIndex) {
        reserveCount += edges[imageIndex].raw.size();
    }
    pooledPoints.reserve(reserveCount);
    for (std::size_t imageIndex = 0; imageIndex + 1 < edges.size(); ++imageIndex) {
        for (const cv::Point2d& point : edges[imageIndex].raw) {
            pooledPoints.push_back(
                transformPointByMatrix(stitching.imageTransforms[imageIndex], point));
        }
    }

    const CircleModel globalCircle = fitCircleRobustForP2d(pooledPoints);
    const CircleModel lastCircleLocal = fitCircleRobustForP2d(edges.back().raw);
    if (!globalCircle.ok || !lastCircleLocal.ok) {
        return false;
    }

    const cv::Mat currentTransform = stitching.imageTransforms.back().clone();
    const cv::Point2d currentCenterGlobal =
        transformPointByMatrix(currentTransform,
                               cv::Point2d(lastCircleLocal.centerX, lastCircleLocal.centerY));
    const double currentCenterError =
        cv::norm(currentCenterGlobal - cv::Point2d(globalCircle.centerX, globalCircle.centerY));
    const double currentRmse =
        radialRmseToCircle(edges.back(), currentTransform, globalCircle);

    cv::Mat bestTransform = currentTransform.clone();
    bestTransform.at<double>(0, 2) += globalCircle.centerX - currentCenterGlobal.x;
    bestTransform.at<double>(1, 2) += globalCircle.centerY - currentCenterGlobal.y;
    double bestRmse = radialRmseToCircle(edges.back(), bestTransform, globalCircle);

    const std::array<double, 4> translationSteps = {24.0, 8.0, 2.0, 0.5};
    for (double stepPx : translationSteps) {
        bool improved = false;
        do {
            improved = false;
            for (int dyIndex = -1; dyIndex <= 1; ++dyIndex) {
                for (int dxIndex = -1; dxIndex <= 1; ++dxIndex) {
                    if (dxIndex == 0 && dyIndex == 0) {
                        continue;
                    }
                    cv::Mat trial = bestTransform.clone();
                    trial.at<double>(0, 2) += static_cast<double>(dxIndex) * stepPx;
                    trial.at<double>(1, 2) += static_cast<double>(dyIndex) * stepPx;
                    const double rmse = radialRmseToCircle(edges.back(), trial, globalCircle);
                    if (rmse + 1e-6 < bestRmse) {
                        bestRmse = rmse;
                        bestTransform = std::move(trial);
                        improved = true;
                    }
                }
            }
        } while (improved);
    }

    const cv::Point2d bestCenterGlobal =
        transformPointByMatrix(bestTransform,
                               cv::Point2d(lastCircleLocal.centerX, lastCircleLocal.centerY));
    const double bestCenterError =
        cv::norm(bestCenterGlobal - cv::Point2d(globalCircle.centerX, globalCircle.centerY));
    const bool accept =
        std::isfinite(bestRmse) &&
        (bestRmse + 0.02 < currentRmse ||
         (currentCenterError > 20.0 && bestCenterError + 5.0 < currentCenterError));
    if (!accept) {
        return false;
    }

    stitching.imageTransforms.back() = bestTransform;
    if (messageOut != nullptr) {
        std::ostringstream message;
        message << "last-image rescue applied: center error "
                << currentCenterError << " -> " << bestCenterError
                << " px, radial RMSE "
                << currentRmse << " -> " << bestRmse << " px";
        *messageOut = message.str();
    }
    return true;
}

Gbt57P2dResult evaluateGbt57P2dSinglePoint(
    const std::vector<cv::Mat>& images,
    const std::vector<stitch::EdgeVariants>& edges,
    const std::vector<stitch::EdgeVariants>* supplementalCandidateEdges,
    const stitch::StitchingResult& stitching,
    const std::vector<std::string>& imagePaths,
    const Gbt57P2dConfig& config,
    const pinjie::standard_sphere_loop::StandardSphereLoopConfig& experimentConfig)
{
    Gbt57P2dResult result;
    result.selectedPointTransformRefineRequested = config.selectedPointTransformRefine;
    if (!config.enabled) {
        result.message = "GB/T 5.7 P2D mode is disabled";
        return result;
    }
    if (images.empty() || edges.size() != images.size() ||
        stitching.imageTransforms.size() < images.size()) {
        result.message = "GB/T 5.7 P2D needs images, edges, and stitching transforms";
        return result;
    }
    std::vector<cv::Mat> evaluationTransforms;
    evaluationTransforms.reserve(stitching.imageTransforms.size());
    for (const cv::Mat& transform : stitching.imageTransforms) {
        evaluationTransforms.push_back(transform.clone());
    }
    std::vector<cv::Mat> originalEvaluationTransforms;
    originalEvaluationTransforms.reserve(evaluationTransforms.size());
    for (const cv::Mat& transform : evaluationTransforms) {
        originalEvaluationTransforms.push_back(transform.clone());
    }

    std::vector<stitch::EdgeVariants> measurementEdges;
    measurementEdges.reserve(edges.size());
    std::size_t syntheticPointCount = 0;
    for (const stitch::EdgeVariants& edge : edges) {
        const stitch::EdgeVariants filtered = filterSyntheticEdgePoints(edge);
        syntheticPointCount += edge.raw.size() > filtered.raw.size()
                                   ? edge.raw.size() - filtered.raw.size()
                                   : 0;
        measurementEdges.push_back(filtered);
    }

    std::vector<stitch::EdgeVariants> candidateEdges = measurementEdges;
    if (config.supplementPreCleanupCandidates &&
        supplementalCandidateEdges != nullptr &&
        supplementalCandidateEdges->size() == edges.size()) {
        for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
            const stitch::EdgeVariants supplemental =
                filterSyntheticEdgePoints((*supplementalCandidateEdges)[imageIndex]);
            stitch::EdgeVariants& merged = candidateEdges[imageIndex];
            const bool mergedHasWeights =
                merged.rawQualityWeights.size() == merged.raw.size();
            const bool mergedHasConfidences =
                merged.rawConfidences.size() == merged.raw.size();
            const bool mergedHasGradients =
                merged.rawGradients.size() == merged.raw.size();
            const bool extraHasWeights =
                supplemental.rawQualityWeights.size() == supplemental.raw.size();
            const bool extraHasConfidences =
                supplemental.rawConfidences.size() == supplemental.raw.size();
            const bool extraHasGradients =
                supplemental.rawGradients.size() == supplemental.raw.size();

            if (!mergedHasWeights) {
                merged.rawQualityWeights.assign(merged.raw.size(), 1.0);
            }
            if (!mergedHasConfidences) {
                merged.rawConfidences.assign(merged.raw.size(), 0.0);
            }
            if (!mergedHasGradients) {
                merged.rawGradients.assign(merged.raw.size(), 0.0);
            }

            merged.raw.reserve(merged.raw.size() + supplemental.raw.size());
            merged.rawQualityWeights.reserve(merged.raw.size() + supplemental.raw.size());
            merged.rawConfidences.reserve(merged.raw.size() + supplemental.raw.size());
            merged.rawGradients.reserve(merged.raw.size() + supplemental.raw.size());

            for (std::size_t pointIndex = 0; pointIndex < supplemental.raw.size(); ++pointIndex) {
                merged.raw.push_back(supplemental.raw[pointIndex]);
                merged.rawQualityWeights.push_back(
                    extraHasWeights ? supplemental.rawQualityWeights[pointIndex] : 1.0);
                merged.rawConfidences.push_back(
                    extraHasConfidences ? supplemental.rawConfidences[pointIndex] : 0.0);
                merged.rawGradients.push_back(
                    extraHasGradients ? supplemental.rawGradients[pointIndex] : 0.0);
            }
        }
    }

    {
        const std::vector<stitch::EdgeVariants>& edges = measurementEdges;

    std::vector<cv::Point2d> globalEdgePoints;
    std::size_t reserveCount = 0;
    for (const auto& edge : edges) {
        reserveCount += edge.raw.size();
    }
    globalEdgePoints.reserve(reserveCount);
    for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
        for (const cv::Point2d& point : edges[imageIndex].raw) {
            globalEdgePoints.push_back(
                transformPointByMatrix(evaluationTransforms[imageIndex], point));
        }
    }

    result.globalCircle = fitCircleRobustForP2d(globalEdgePoints);
    if (!result.globalCircle.ok) {
        result.message = "failed to fit stitched global standard-sphere circle";
        return result;
    }
    result.globalEllipse = fitEllipseModel(globalEdgePoints);
    const EllipseRectificationModel globalEllipseRectification =
        buildEllipseRectificationModel(result.globalEllipse);
    const bool pureEllipseRectifiedEvaluation =
        config.ellipseNormalizationCompensation &&
        globalEllipseRectification.ok &&
        !config.fieldBiasCompensation &&
        !config.localFieldBiasCompensation &&
        !config.angularBiasCompensation;
    result.ellipseRectifiedEvaluationApplied = pureEllipseRectifiedEvaluation;
    result.measurementWindowHalfSizePx = std::max(0.0, config.windowHalfSizePx);
    result.configuredPixelSizeUm = effectivePixelSizeUm(images, experimentConfig);
    result.sphereDiameterMm = experimentConfig.sphereDiameterMm;
    result.sphereDiameterDecimalPlaces =
        std::clamp(experimentConfig.sphereDiameterDecimalPlaces, 0, 12);
    result.sphereCalibratedPixelSizeUm =
        sphereCalibratedPixelSizeUm(result.sphereDiameterMm, result.globalCircle);
    if (result.sphereCalibratedPixelSizeUm > 0.0) {
        result.effectivePixelSizeUm = result.sphereCalibratedPixelSizeUm;
        result.effectivePixelSizeSource = "sphere_diameter_global_circle";
    } else {
        result.effectivePixelSizeUm = result.configuredPixelSizeUm;
        result.effectivePixelSizeSource =
            result.configuredPixelSizeUm > 0.0 ? "configured_fov_or_pixel_size" : "unavailable";
    }

    const int expectedPointCount = std::max(1, config.expectedPointCount);
    const double idealSpacingDeg = 360.0 / static_cast<double>(expectedPointCount);
    const double requestedWindowHalfAngle = std::max(0.2, config.windowHalfAngleDeg);
    const double nonOverlapWindowHalfAngle =
        std::max(0.2, 0.5 * idealSpacingDeg - 0.05);
    const double windowHalfAngle =
        config.uniformAngleSelection
            ? std::min(requestedWindowHalfAngle, nonOverlapWindowHalfAngle)
            : requestedWindowHalfAngle;
    result.effectiveWindowHalfAngleDeg = windowHalfAngle;
    const double relaxedHalfAngle =
        config.uniformAngleSelection
            ? windowHalfAngle
            : std::max(windowHalfAngle, 0.50 * idealSpacingDeg);
    const double candidateStoreHalfAngle =
        config.uniformAngleSelection
            ? windowHalfAngle
            : (config.optimizeSelectedRange
                   ? std::max(windowHalfAngle, std::min(24.0, 1.10 * idealSpacingDeg))
                   : windowHalfAngle);

    std::vector<CircleModel> localCircles;
    localCircles.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        localCircles.push_back(fitCircleRobustForP2d(edge.raw));
    }

    std::vector<std::vector<Gbt57CandidatePoint>> perImageCandidates(images.size());
    std::vector<double> targetAngles(images.size(), 0.0);
    std::vector<bool> targetAnglesValid(images.size(), false);
    if (config.uniformAngleSelection &&
        images.size() == static_cast<std::size_t>(expectedPointCount)) {
        struct ViewAngleRecord {
            std::size_t index{0};
            double angleDeg{0.0};
            double weightSum{0.0};
            std::vector<double> sampleAnglesDeg;
        };
        std::vector<ViewAngleRecord> viewAngles;
        viewAngles.reserve(images.size());
        for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
            double sumSin = 0.0;
            double sumCos = 0.0;
            double angleWeightSum = 0.0;
            double fallbackSumSin = 0.0;
            double fallbackSumCos = 0.0;
            double fallbackWeightSum = 0.0;
            std::vector<double> sampleAnglesDeg;
            std::vector<double> fallbackSampleAnglesDeg;
            sampleAnglesDeg.reserve(candidateEdges[imageIndex].raw.size());
            fallbackSampleAnglesDeg.reserve(candidateEdges[imageIndex].raw.size());
            const bool hasWeights =
                candidateEdges[imageIndex].rawQualityWeights.size() ==
                candidateEdges[imageIndex].raw.size();
            for (std::size_t pointIndex = 0;
                 pointIndex < candidateEdges[imageIndex].raw.size();
                 ++pointIndex) {
                const cv::Point2d imagePoint = candidateEdges[imageIndex].raw[pointIndex];
                const cv::Point2d global =
                    transformPointByMatrix(evaluationTransforms[imageIndex],
                                           imagePoint);
                const double angleDeg = angleOfPointDeg(global, result.globalCircle);
                const double angleRad = angleDeg * CV_PI / 180.0;
                const double weight =
                    hasWeights ? std::clamp(candidateEdges[imageIndex].rawQualityWeights[pointIndex],
                                            0.05,
                                            4.0)
                               : 1.0;
                fallbackSampleAnglesDeg.push_back(angleDeg);
                fallbackSumSin += weight * std::sin(angleRad);
                fallbackSumCos += weight * std::cos(angleRad);
                fallbackWeightSum += weight;
                if (!measurementWindowInsideImage(imagePoint,
                                                  images[imageIndex].size(),
                                                  result.measurementWindowHalfSizePx,
                                                  nullptr)) {
                    continue;
                }
                sampleAnglesDeg.push_back(angleDeg);
                sumSin += weight * std::sin(angleRad);
                sumCos += weight * std::cos(angleRad);
                angleWeightSum += weight;
            }
            if (sampleAnglesDeg.empty() && !fallbackSampleAnglesDeg.empty()) {
                sampleAnglesDeg = std::move(fallbackSampleAnglesDeg);
                sumSin = fallbackSumSin;
                sumCos = fallbackSumCos;
                angleWeightSum = fallbackWeightSum;
            }
            if (angleWeightSum > 0.0 && !sampleAnglesDeg.empty()) {
                viewAngles.push_back({
                    imageIndex,
                    normalizeAngleDeg(std::atan2(sumSin, sumCos) * 180.0 / CV_PI),
                    angleWeightSum,
                    std::move(sampleAnglesDeg)
                });
            }
        }
        if (viewAngles.size() == images.size()) {
            std::sort(viewAngles.begin(),
                      viewAngles.end(),
                      [](const ViewAngleRecord& a, const ViewAngleRecord& b) {
                          return a.angleDeg < b.angleDeg;
                      });

            double largestGap = -1.0;
            std::size_t gapIndex = 0;
            for (std::size_t i = 0; i < viewAngles.size(); ++i) {
                const double current = viewAngles[i].angleDeg;
                const double next = viewAngles[(i + 1) % viewAngles.size()].angleDeg +
                                    (i + 1 == viewAngles.size() ? 360.0 : 0.0);
                const double gap = next - current;
                if (gap > largestGap) {
                    largestGap = gap;
                    gapIndex = i;
                }
            }

            std::vector<ViewAngleRecord> ordered;
            ordered.reserve(viewAngles.size());
            for (std::size_t offset = 1; offset <= viewAngles.size(); ++offset) {
                ordered.push_back(viewAngles[(gapIndex + offset) % viewAngles.size()]);
            }

            const double stepDeg = idealSpacingDeg;
            std::vector<double> unwrappedAngles(ordered.size(), 0.0);
            std::vector<std::vector<double>> unwrappedSampleAngles(ordered.size());
            unwrappedAngles[0] = ordered.front().angleDeg;
            for (std::size_t rank = 1; rank < ordered.size(); ++rank) {
                unwrappedAngles[rank] =
                    unwrappedAngles[rank - 1] +
                    angleDistanceForwardDeg(ordered[rank - 1].angleDeg,
                                            ordered[rank].angleDeg);
            }
            for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
                unwrappedSampleAngles[rank].reserve(ordered[rank].sampleAnglesDeg.size());
                for (double sampleAngleDeg : ordered[rank].sampleAnglesDeg) {
                    const double delta =
                        circularDeltaDeg(sampleAngleDeg, ordered[rank].angleDeg);
                    unwrappedSampleAngles[rank].push_back(unwrappedAngles[rank] + delta);
                }
            }

            double phaseSeed = 0.0;
            double phaseWeight = 0.0;
            for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
                const double weight = std::max(1e-6, ordered[rank].weightSum);
                phaseSeed += weight * (unwrappedAngles[rank] -
                                       static_cast<double>(rank) * stepDeg);
                phaseWeight += weight;
            }
            phaseSeed = phaseWeight > 0.0 ? phaseSeed / phaseWeight
                                          : ordered.front().angleDeg;

            struct PhaseObjective {
                double maxWindowViolationDeg{std::numeric_limits<double>::infinity()};
                double meanWindowViolationDeg{std::numeric_limits<double>::infinity()};
                double maxSampleDeltaDeg{std::numeric_limits<double>::infinity()};
                double meanSampleDeltaDeg{std::numeric_limits<double>::infinity()};
                double centerRmseDeg{std::numeric_limits<double>::infinity()};
            };
            const auto betterPhaseObjective =
                [](const PhaseObjective& lhs, const PhaseObjective& rhs) {
                    const auto strictlyLess = [](double a, double b) {
                        return a + 1e-9 < b;
                    };
                    if (strictlyLess(lhs.maxWindowViolationDeg, rhs.maxWindowViolationDeg)) {
                        return true;
                    }
                    if (strictlyLess(rhs.maxWindowViolationDeg, lhs.maxWindowViolationDeg)) {
                        return false;
                    }
                    if (strictlyLess(lhs.meanWindowViolationDeg, rhs.meanWindowViolationDeg)) {
                        return true;
                    }
                    if (strictlyLess(rhs.meanWindowViolationDeg, lhs.meanWindowViolationDeg)) {
                        return false;
                    }
                    if (strictlyLess(lhs.maxSampleDeltaDeg, rhs.maxSampleDeltaDeg)) {
                        return true;
                    }
                    if (strictlyLess(rhs.maxSampleDeltaDeg, lhs.maxSampleDeltaDeg)) {
                        return false;
                    }
                    if (strictlyLess(lhs.meanSampleDeltaDeg, rhs.meanSampleDeltaDeg)) {
                        return true;
                    }
                    if (strictlyLess(rhs.meanSampleDeltaDeg, lhs.meanSampleDeltaDeg)) {
                        return false;
                    }
                    return strictlyLess(lhs.centerRmseDeg, rhs.centerRmseDeg);
                };
            const auto phaseCost = [&](double phaseDeg) {
                PhaseObjective objective;
                double sumWindowViolation = 0.0;
                double sumSampleDelta = 0.0;
                double sumCenterSq = 0.0;
                double sumWeight = 0.0;
                objective.maxWindowViolationDeg = 0.0;
                objective.maxSampleDeltaDeg = 0.0;
                for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
                    const double targetAngleDeg =
                        phaseDeg + static_cast<double>(rank) * stepDeg;
                    double nearestSampleDeltaDeg =
                        std::numeric_limits<double>::infinity();
                    for (double sampleAngleDeg : unwrappedSampleAngles[rank]) {
                        nearestSampleDeltaDeg =
                            std::min(nearestSampleDeltaDeg,
                                     std::abs(sampleAngleDeg - targetAngleDeg));
                    }
                    if (!std::isfinite(nearestSampleDeltaDeg)) {
                        return PhaseObjective{};
                    }
                    const double windowViolationDeg =
                        std::max(0.0, nearestSampleDeltaDeg - windowHalfAngle);
                    const double centerDeltaDeg =
                        std::abs(unwrappedAngles[rank] - targetAngleDeg);
                    const double weight = std::max(1e-6, ordered[rank].weightSum);
                    objective.maxWindowViolationDeg =
                        std::max(objective.maxWindowViolationDeg, windowViolationDeg);
                    objective.maxSampleDeltaDeg =
                        std::max(objective.maxSampleDeltaDeg, nearestSampleDeltaDeg);
                    sumWindowViolation += windowViolationDeg;
                    sumSampleDelta += nearestSampleDeltaDeg;
                    sumCenterSq += weight * centerDeltaDeg * centerDeltaDeg;
                    sumWeight += weight;
                }
                objective.meanWindowViolationDeg =
                    sumWindowViolation / static_cast<double>(ordered.size());
                objective.meanSampleDeltaDeg =
                    sumSampleDelta / static_cast<double>(ordered.size());
                objective.centerRmseDeg =
                    std::sqrt(sumCenterSq / std::max(1e-6, sumWeight));
                return objective;
            };

            double bestPhase = phaseSeed;
            PhaseObjective bestPhaseCost = phaseCost(bestPhase);
            const double phaseHalfSearch = stepDeg;
            const double phaseStep = std::max(0.01, stepDeg / 720.0);
            for (double phase = phaseSeed - phaseHalfSearch;
                 phase <= phaseSeed + phaseHalfSearch + 1e-12;
                 phase += phaseStep) {
                const PhaseObjective cost = phaseCost(phase);
                if (betterPhaseObjective(cost, bestPhaseCost)) {
                    bestPhase = phase;
                    bestPhaseCost = cost;
                }
            }
            for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
                targetAngles[ordered[rank].index] =
                    normalizeAngleDeg(bestPhase + static_cast<double>(rank) * stepDeg);
                targetAnglesValid[ordered[rank].index] = true;
            }
        }
    }

    result.points.reserve(images.size());
    result.candidateCoverage.resize(images.size());
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        Gbt57P2dPoint selected;
        selected.imageIndex = static_cast<int>(imageIndex);
        selected.imageName =
            imageIndex < imagePaths.size()
                ? std::filesystem::path(imagePaths[imageIndex]).filename().u8string()
                : std::string{};
        selected.windowHalfAngleDeg = windowHalfAngle;
        selected.windowHalfSizePx = std::max(0.0, config.windowHalfSizePx);
        result.candidateCoverage[imageIndex].imageIndex = static_cast<int>(imageIndex);
        result.candidateCoverage[imageIndex].imageName = selected.imageName;

        double sumSin = 0.0;
        double sumCos = 0.0;
        double angleWeightSum = 0.0;
        const bool hasWeights =
            edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            const cv::Point2d global =
                transformPointByMatrix(evaluationTransforms[imageIndex],
                                       edges[imageIndex].raw[pointIndex]);
            const double angleRad = angleOfPointDeg(global, result.globalCircle) * CV_PI / 180.0;
            const double weight = hasWeights
                                      ? std::clamp(edges[imageIndex].rawQualityWeights[pointIndex],
                                                   0.05,
                                                   4.0)
                                      : 1.0;
            sumSin += weight * std::sin(angleRad);
            sumCos += weight * std::cos(angleRad);
            angleWeightSum += weight;
        }
        if (angleWeightSum <= 0.0 || edges[imageIndex].raw.empty()) {
            result.points.push_back(selected);
            continue;
        }
        selected.viewAngleDeg = normalizeAngleDeg(std::atan2(sumSin, sumCos) * 180.0 / CV_PI);
        if (config.uniformAngleSelection && targetAnglesValid[imageIndex]) {
            selected.viewAngleDeg = targetAngles[imageIndex];
        }

        const cv::Point2d imageCenter(images[imageIndex].cols * 0.5,
                                      images[imageIndex].rows * 0.5);
        const double imageDiag =
            std::max(1.0,
                     std::hypot(static_cast<double>(images[imageIndex].cols),
                                static_cast<double>(images[imageIndex].rows)));
        const stitch::EdgeVariants& selectionEdge = candidateEdges[imageIndex];
        const bool hasConfidences =
            selectionEdge.rawConfidences.size() == selectionEdge.raw.size();
        const bool hasGradients =
            selectionEdge.rawGradients.size() == selectionEdge.raw.size();

        double bestCost = std::numeric_limits<double>::infinity();
        const auto testCandidate = [&](double halfAngle, bool updateBest, bool& found) {
            for (std::size_t pointIndex = 0; pointIndex < selectionEdge.raw.size(); ++pointIndex) {
                const cv::Point2d imagePoint = selectionEdge.raw[pointIndex];
                double windowMarginPx = 0.0;
                if (!measurementWindowInsideImage(imagePoint,
                                                  images[imageIndex].size(),
                                                  selected.windowHalfSizePx,
                                                  &windowMarginPx)) {
                    continue;
                }
                const cv::Point2d global =
                    transformPointByMatrix(evaluationTransforms[imageIndex], imagePoint);
                const double angle = angleOfPointDeg(global, result.globalCircle);
                const double angleDelta = circularAbsDeltaDeg(angle, selected.viewAngleDeg);
                if (angleDelta > halfAngle) {
                    continue;
                }

                const double quality =
                    selectionEdge.rawQualityWeights.size() == selectionEdge.raw.size()
                        ? std::clamp(selectionEdge.rawQualityWeights[pointIndex],
                                            0.05,
                                            4.0)
                               : 1.0;
                const double confidence =
                    hasConfidences ? selectionEdge.rawConfidences[pointIndex] : 0.0;
                const double gradient =
                    hasGradients ? selectionEdge.rawGradients[pointIndex] : 0.0;
                const double centerCost = cv::norm(imagePoint - imageCenter) / imageDiag;
                const double confidenceBonus = std::clamp(confidence / 10.0, 0.0, 1.0);
                const double gradientBonus = std::clamp(gradient / 80.0, 0.0, 1.0);
                const double radius =
                    std::hypot(global.x - result.globalCircle.centerX,
                               global.y - result.globalCircle.centerY);
                const double radiusDelta =
                    std::abs(radius - result.globalCircle.radius);
                const double costHalfAngle =
                    std::max(0.1, windowHalfAngle);
                const double normalizedAngle =
                    angleDelta / costHalfAngle;
                const double windowEdgePenalty = normalizedAngle * normalizedAngle;
                const double edgeOverflow =
                    std::max(0.0, normalizedAngle - 0.65);
                const double windowBoundaryPenalty = edgeOverflow * edgeOverflow;
                double cost = 0.0;
                if (config.confidenceBestSelection) {
                    const double radiusGuard =
                        std::max(0.25, config.confidenceRadiusGuardPx);
                    double localRadiusDelta = radiusDelta;
                    if (imageIndex < localCircles.size() &&
                        localCircles[imageIndex].ok) {
                        const CircleModel& localCircle = localCircles[imageIndex];
                        localRadiusDelta =
                            std::abs(std::hypot(imagePoint.x - localCircle.centerX,
                                                imagePoint.y - localCircle.centerY) -
                                     localCircle.radius);
                    }
                    if (localRadiusDelta > radiusGuard) {
                        continue;
                    }
                    const double qualityScore = std::clamp(std::log1p(quality) / std::log(5.0),
                                                           0.0,
                                                           1.5);
                    cost =
                        0.42 * windowEdgePenalty +
                        0.20 * windowBoundaryPenalty +
                        0.08 * centerCost +
                        0.22 * localRadiusDelta / radiusGuard +
                        0.10 * radiusDelta / std::max(0.5, 2.0 * radiusGuard) -
                        0.30 * confidenceBonus -
                        0.16 * gradientBonus -
                        0.10 * qualityScore;
                } else if (config.radiusStableSelection) {
                    cost =
                        radiusDelta +
                        0.12 * windowEdgePenalty +
                        0.04 * windowBoundaryPenalty +
                        0.03 * centerCost -
                        0.03 * std::log1p(quality) -
                        0.02 * confidenceBonus -
                        0.02 * gradientBonus;
                } else {
                    cost =
                        0.75 * normalizedAngle +
                        0.25 * windowEdgePenalty +
                        0.04 * radiusDelta +
                        0.20 * centerCost -
                        0.08 * std::log1p(quality) -
                        0.05 * confidenceBonus -
                        0.03 * gradientBonus;
                }

                ++selected.candidateCount;
                if (angleDelta <= candidateStoreHalfAngle + 1e-9 ||
                    perImageCandidates[imageIndex].empty()) {
                    perImageCandidates[imageIndex].push_back({
                        imagePoint,
                        global,
                        angle,
                        angleDelta,
                        circularDeltaDeg(angle, selected.viewAngleDeg),
                        radiusDelta,
                        cost,
                        quality,
                        confidence,
                        gradient
                    });
                }
                if (updateBest && cost < bestCost) {
                    bestCost = cost;
                    selected.selected = true;
                    selected.imagePoint = imagePoint;
                    selected.globalPoint = global;
                    selected.selectedAngleDeg = angle;
                    selected.angleDeltaDeg = angleDelta;
                    selected.windowMarginPx = windowMarginPx;
                    selected.measurementWindowInsideImage = true;
                    selected.qualityWeight = quality;
                    selected.confidence = confidence;
                    selected.gradient = gradient;
                    selected.selectionCost = cost;
                    found = true;
                }
            }
        };

        bool found = false;
        testCandidate(windowHalfAngle, true, found);
        if (!found) {
            testCandidate(relaxedHalfAngle, true, found);
        }
        if (!found) {
            testCandidate(180.0, true, found);
        }
        if (selected.selected) {
            bool selectedInCandidatePool = false;
            for (const Gbt57CandidatePoint& candidate : perImageCandidates[imageIndex]) {
                if (std::abs(candidate.imagePoint.x - selected.imagePoint.x) <= 1e-9 &&
                    std::abs(candidate.imagePoint.y - selected.imagePoint.y) <= 1e-9) {
                    selectedInCandidatePool = true;
                    break;
                }
            }
            if (!selectedInCandidatePool) {
                const double radius =
                    std::hypot(selected.globalPoint.x - result.globalCircle.centerX,
                               selected.globalPoint.y - result.globalCircle.centerY);
                perImageCandidates[imageIndex].push_back({
                    selected.imagePoint,
                    selected.globalPoint,
                    selected.selectedAngleDeg,
                    selected.angleDeltaDeg,
                    circularDeltaDeg(selected.selectedAngleDeg, selected.viewAngleDeg),
                    std::abs(radius - result.globalCircle.radius),
                    selected.selectionCost,
                    selected.qualityWeight,
                    selected.confidence,
                    selected.gradient
                });
            }
        }
        if ((config.optimizeSelectedRange || config.uniformAngleSelection) &&
            candidateStoreHalfAngle > relaxedHalfAngle + 1e-9) {
            bool candidatePoolFound = false;
            testCandidate(candidateStoreHalfAngle, false, candidatePoolFound);
        }
        result.points.push_back(selected);
    }

    if ((config.optimizeSelectedRange || config.uniformAngleSelection) &&
        result.points.size() == perImageCandidates.size()) {
        const std::size_t keep =
            static_cast<std::size_t>(std::max(1, config.rangeOptimizationCandidatesPerField));
        for (std::size_t imageIndex = 0; imageIndex < perImageCandidates.size(); ++imageIndex) {
            std::vector<Gbt57CandidatePoint>& candidates = perImageCandidates[imageIndex];
            std::sort(candidates.begin(),
                      candidates.end(),
                      [](const Gbt57CandidatePoint& a, const Gbt57CandidatePoint& b) {
                          if (std::abs(a.imagePoint.x - b.imagePoint.x) > 1e-9) {
                              return a.imagePoint.x < b.imagePoint.x;
                          }
                          if (std::abs(a.imagePoint.y - b.imagePoint.y) > 1e-9) {
                              return a.imagePoint.y < b.imagePoint.y;
                          }
                          if (std::abs(a.globalPoint.x - b.globalPoint.x) > 1e-9) {
                              return a.globalPoint.x < b.globalPoint.x;
                          }
                          if (std::abs(a.globalPoint.y - b.globalPoint.y) > 1e-9) {
                              return a.globalPoint.y < b.globalPoint.y;
                          }
                          if (std::abs(a.selectedAngleDeg - b.selectedAngleDeg) > 1e-9) {
                              return a.selectedAngleDeg < b.selectedAngleDeg;
                          }
                          if (std::abs(a.angleDeltaDeg - b.angleDeltaDeg) > 1e-9) {
                              return a.angleDeltaDeg < b.angleDeltaDeg;
                          }
                          if (std::abs(a.radiusDeltaPx - b.radiusDeltaPx) > 1e-9) {
                              return a.radiusDeltaPx < b.radiusDeltaPx;
                          }
                          return a.cost < b.cost;
                      });
            candidates.erase(std::unique(candidates.begin(),
                                         candidates.end(),
                                         [](const Gbt57CandidatePoint& a,
                                            const Gbt57CandidatePoint& b) {
                                             return std::abs(a.imagePoint.x - b.imagePoint.x) <= 1e-9 &&
                                                    std::abs(a.imagePoint.y - b.imagePoint.y) <= 1e-9 &&
                                                    std::abs(a.globalPoint.x - b.globalPoint.x) <= 1e-9 &&
                                                    std::abs(a.globalPoint.y - b.globalPoint.y) <= 1e-9;
                                         }),
                             candidates.end());
            const auto sameCandidate = [](const Gbt57CandidatePoint& a,
                                          const Gbt57CandidatePoint& b) {
                return std::abs(a.imagePoint.x - b.imagePoint.x) <= 1e-9 &&
                       std::abs(a.imagePoint.y - b.imagePoint.y) <= 1e-9 &&
                       std::abs(a.globalPoint.x - b.globalPoint.x) <= 1e-9 &&
                       std::abs(a.globalPoint.y - b.globalPoint.y) <= 1e-9;
            };
            Gbt57CandidateCoverageStat& coverage = result.candidateCoverage[imageIndex];
            coverage.rawCandidateCount = result.points[imageIndex].candidateCount;
            coverage.uniqueCandidateCount = static_cast<int>(candidates.size());
            coverage.uniqueMinAbsAngleDeltaDeg = std::numeric_limits<double>::infinity();
            coverage.uniqueMinRadiusDeltaPx = std::numeric_limits<double>::infinity();
            coverage.bestCostAngleDeltaDeg = 0.0;
            coverage.bestCostSignedAngleDeltaDeg = 0.0;
            coverage.bestCostRadiusDeltaPx = 0.0;
            coverage.uniqueWithin2Deg = 0;
            coverage.uniqueWithin3Deg = 0;
            coverage.uniqueWithin4Deg = 0;
            if (!candidates.empty()) {
                coverage.uniqueMinSignedAngleDeltaDeg = candidates.front().signedAngleDeltaDeg;
                coverage.uniqueMaxSignedAngleDeltaDeg = candidates.front().signedAngleDeltaDeg;
                const Gbt57CandidatePoint* bestCostCandidate = &candidates.front();
                for (const Gbt57CandidatePoint& candidate : candidates) {
                    coverage.uniqueMinAbsAngleDeltaDeg =
                        std::min(coverage.uniqueMinAbsAngleDeltaDeg, candidate.angleDeltaDeg);
                    coverage.uniqueMinSignedAngleDeltaDeg =
                        std::min(coverage.uniqueMinSignedAngleDeltaDeg, candidate.signedAngleDeltaDeg);
                    coverage.uniqueMaxSignedAngleDeltaDeg =
                        std::max(coverage.uniqueMaxSignedAngleDeltaDeg, candidate.signedAngleDeltaDeg);
                    coverage.uniqueMinRadiusDeltaPx =
                        std::min(coverage.uniqueMinRadiusDeltaPx, candidate.radiusDeltaPx);
                    coverage.uniqueWithin2Deg += candidate.angleDeltaDeg <= 2.0 ? 1 : 0;
                    coverage.uniqueWithin3Deg += candidate.angleDeltaDeg <= 3.0 ? 1 : 0;
                    coverage.uniqueWithin4Deg += candidate.angleDeltaDeg <= 4.0 ? 1 : 0;
                    if (candidate.cost + 1e-12 < bestCostCandidate->cost) {
                        bestCostCandidate = &candidate;
                    }
                }
                coverage.bestCostAngleDeltaDeg = bestCostCandidate->angleDeltaDeg;
                coverage.bestCostSignedAngleDeltaDeg = bestCostCandidate->signedAngleDeltaDeg;
                coverage.bestCostRadiusDeltaPx = bestCostCandidate->radiusDeltaPx;
            }
            const auto byGeometry = [](const Gbt57CandidatePoint& a,
                                       const Gbt57CandidatePoint& b) {
                if (std::abs(a.angleDeltaDeg - b.angleDeltaDeg) > 1e-9) {
                    return a.angleDeltaDeg < b.angleDeltaDeg;
                }
                if (std::abs(a.radiusDeltaPx - b.radiusDeltaPx) > 1e-9) {
                    return a.radiusDeltaPx < b.radiusDeltaPx;
                }
                if (std::abs(a.cost - b.cost) > 1e-9) {
                    return a.cost < b.cost;
                }
                if (std::abs(a.confidence - b.confidence) > 1e-9) {
                    return a.confidence > b.confidence;
                }
                return a.gradient > b.gradient;
            };
            if (candidates.size() > keep) {
                std::vector<Gbt57CandidatePoint> rankedGeometry = candidates;
                std::stable_sort(rankedGeometry.begin(), rankedGeometry.end(), byGeometry);

                std::vector<Gbt57CandidatePoint> rankedCost = candidates;
                std::stable_sort(rankedCost.begin(),
                                 rankedCost.end(),
                                 [](const Gbt57CandidatePoint& a,
                                    const Gbt57CandidatePoint& b) {
                                     if (std::abs(a.cost - b.cost) > 1e-9) {
                                         return a.cost < b.cost;
                                     }
                                     if (std::abs(a.angleDeltaDeg - b.angleDeltaDeg) > 1e-9) {
                                         return a.angleDeltaDeg < b.angleDeltaDeg;
                                     }
                                     return a.radiusDeltaPx < b.radiusDeltaPx;
                                 });

                std::vector<Gbt57CandidatePoint> rankedSignal = candidates;
                std::stable_sort(rankedSignal.begin(),
                                 rankedSignal.end(),
                                 [](const Gbt57CandidatePoint& a,
                                    const Gbt57CandidatePoint& b) {
                                     const double aSignal =
                                         a.confidence + 0.02 * a.gradient +
                                         0.25 * std::log1p(std::max(0.0, a.qualityWeight));
                                     const double bSignal =
                                         b.confidence + 0.02 * b.gradient +
                                         0.25 * std::log1p(std::max(0.0, b.qualityWeight));
                                     if (std::abs(aSignal - bSignal) > 1e-9) {
                                         return aSignal > bSignal;
                                     }
                                     if (std::abs(a.angleDeltaDeg - b.angleDeltaDeg) > 1e-9) {
                                         return a.angleDeltaDeg < b.angleDeltaDeg;
                                     }
                                     return a.radiusDeltaPx < b.radiusDeltaPx;
                                 });

                std::vector<Gbt57CandidatePoint> rankedByAngle = candidates;
                std::stable_sort(rankedByAngle.begin(),
                                 rankedByAngle.end(),
                                 [](const Gbt57CandidatePoint& a,
                                    const Gbt57CandidatePoint& b) {
                                     if (std::abs(a.signedAngleDeltaDeg - b.signedAngleDeltaDeg) > 1e-9) {
                                         return a.signedAngleDeltaDeg < b.signedAngleDeltaDeg;
                                     }
                                     return a.cost < b.cost;
                                 });

                std::vector<Gbt57CandidatePoint> reduced;
                reduced.reserve(keep);
                const auto addUnique = [&](const Gbt57CandidatePoint& candidate) {
                    if (reduced.size() >= keep) {
                        return;
                    }
                    for (const Gbt57CandidatePoint& kept : reduced) {
                        if (sameCandidate(kept, candidate)) {
                            return;
                        }
                    }
                    reduced.push_back(candidate);
                };
                const auto addPrefix = [&](const std::vector<Gbt57CandidatePoint>& source,
                                           std::size_t quota) {
                    for (std::size_t index = 0;
                         index < source.size() && index < quota && reduced.size() < keep;
                         ++index) {
                        addUnique(source[index]);
                    }
                };

                addPrefix(rankedGeometry, std::max<std::size_t>(keep / 3, 1));
                addPrefix(rankedCost, std::max<std::size_t>(keep / 4, 1));
                addPrefix(rankedSignal, std::max<std::size_t>(keep / 6, 1));

                const std::size_t diversityQuota =
                    std::min<std::size_t>(rankedByAngle.size(),
                                          std::max<std::size_t>(keep / 4, 4));
                if (!rankedByAngle.empty()) {
                    for (std::size_t slot = 0;
                         slot < diversityQuota && reduced.size() < keep;
                         ++slot) {
                        const std::size_t angleIndex =
                            diversityQuota == 1
                                ? rankedByAngle.size() / 2
                                : static_cast<std::size_t>(std::llround(
                                      static_cast<double>(slot) *
                                      static_cast<double>(rankedByAngle.size() - 1) /
                                      static_cast<double>(diversityQuota - 1)));
                        addUnique(rankedByAngle[angleIndex]);
                    }
                }

                for (const Gbt57CandidatePoint& candidate : rankedGeometry) {
                    if (reduced.size() >= keep) {
                        break;
                    }
                    addUnique(candidate);
                }
                candidates = std::move(reduced);
            } else {
                std::stable_sort(candidates.begin(), candidates.end(), byGeometry);
            }
            coverage.reducedCandidateCount = static_cast<int>(candidates.size());
        }

        struct SelectionObjective {
            bool valid{false};
            CircleModel circle;
            double rangePx{std::numeric_limits<double>::infinity()};
            double residualRmsePx{std::numeric_limits<double>::infinity()};
            double residualMeanAbsPx{std::numeric_limits<double>::infinity()};
            double spacingMeanDeg{0.0};
            double spacingRmseDeg{std::numeric_limits<double>::infinity()};
            double spacingMaxErrorDeg{std::numeric_limits<double>::infinity()};
            double targetAngleMeanDeg{std::numeric_limits<double>::infinity()};
            double targetAngleMaxDeg{0.0};
            double windowViolationMeanDeg{std::numeric_limits<double>::infinity()};
            double windowViolationMaxDeg{std::numeric_limits<double>::infinity()};
            int windowViolationCount{std::numeric_limits<int>::max()};
            int measurementWindowOverlapCount{std::numeric_limits<int>::max()};
            double measurementWindowOverlapAreaPx2{std::numeric_limits<double>::infinity()};
            double measurementWindowMaxOverlapAreaPx2{std::numeric_limits<double>::infinity()};
            double meanSelectionCost{std::numeric_limits<double>::infinity()};
        };

        const auto evaluateSelectionObjective =
            [&](const std::vector<Gbt57P2dPoint>& points,
                SelectionObjective* objectiveOut = nullptr) {
                SelectionObjective objective;
                std::vector<cv::Point2d> globals;
                std::vector<cv::Point2d> evaluationGlobals;
                globals.reserve(points.size());
                evaluationGlobals.reserve(points.size());
                std::vector<cv::RotatedRect> windowRects;
                windowRects.reserve(points.size());
                for (const Gbt57P2dPoint& point : points) {
                    if (!point.selected) {
                        if (objectiveOut != nullptr) {
                            *objectiveOut = objective;
                        }
                        return false;
                    }
                    globals.push_back(point.globalPoint);
                    evaluationGlobals.push_back(
                        pureEllipseRectifiedEvaluation
                            ? rectifyPointByEllipse(globalEllipseRectification, point.globalPoint)
                            : point.globalPoint);
                    const std::size_t imageIndex = static_cast<std::size_t>(std::max(0, point.imageIndex));
                    if (imageIndex >= evaluationTransforms.size()) {
                        if (objectiveOut != nullptr) {
                            *objectiveOut = objective;
                        }
                        return false;
                    }
                    windowRects.push_back(
                        buildGlobalMeasurementWindow(evaluationTransforms[imageIndex],
                                                     point.imagePoint,
                                                     std::max(0.0, point.windowHalfSizePx)));
                }

                objective.circle = fitCircleMinRange(evaluationGlobals, &objective.rangePx);
                if (!objective.circle.ok) {
                    if (objectiveOut != nullptr) {
                        *objectiveOut = objective;
                    }
                    return false;
                }

                double residualSumAbs = 0.0;
                double residualSumSq = 0.0;
                double targetDeltaSum = 0.0;
                double windowViolationSum = 0.0;
                double selectionCostSum = 0.0;
                std::vector<double> selectedAngles;
                selectedAngles.reserve(points.size());
                objective.windowViolationCount = 0;
                objective.windowViolationMaxDeg = 0.0;
                for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
                    const Gbt57P2dPoint& point = points[pointIndex];
                    const cv::Point2d& rawPoint = globals[pointIndex];
                    const cv::Point2d& evaluationPoint = evaluationGlobals[pointIndex];
                    const double angle = angleOfPointDeg(rawPoint, result.globalCircle);
                    const double radius =
                        std::hypot(evaluationPoint.x - objective.circle.centerX,
                                   evaluationPoint.y - objective.circle.centerY);
                    const double residual = radius - objective.circle.radius;
                    const double targetDelta =
                        circularAbsDeltaDeg(angle, point.viewAngleDeg);
                    const double windowViolation =
                        std::max(0.0, targetDelta - point.windowHalfAngleDeg);
                    selectedAngles.push_back(angle);
                    residualSumAbs += std::abs(residual);
                    residualSumSq += residual * residual;
                    targetDeltaSum += targetDelta;
                    windowViolationSum += windowViolation;
                    objective.targetAngleMaxDeg =
                        std::max(objective.targetAngleMaxDeg, targetDelta);
                    objective.windowViolationMaxDeg =
                        std::max(objective.windowViolationMaxDeg, windowViolation);
                    objective.windowViolationCount += windowViolation > 1e-9 ? 1 : 0;
                    selectionCostSum += point.selectionCost;
                }

                const double pointCount = std::max(1.0, static_cast<double>(points.size()));
                objective.residualMeanAbsPx = residualSumAbs / pointCount;
                objective.residualRmsePx = std::sqrt(residualSumSq / pointCount);
                objective.targetAngleMeanDeg = targetDeltaSum / pointCount;
                objective.windowViolationMeanDeg = windowViolationSum / pointCount;
                objective.meanSelectionCost = selectionCostSum / pointCount;
                objective.measurementWindowOverlapCount = 0;
                objective.measurementWindowOverlapAreaPx2 = 0.0;
                objective.measurementWindowMaxOverlapAreaPx2 = 0.0;
                for (std::size_t lhs = 0; lhs < windowRects.size(); ++lhs) {
                    for (std::size_t rhs = lhs + 1; rhs < windowRects.size(); ++rhs) {
                        const double overlapArea =
                            rotatedRectIntersectionArea(windowRects[lhs], windowRects[rhs]);
                        if (overlapArea <= 1e-6) {
                            continue;
                        }
                        ++objective.measurementWindowOverlapCount;
                        objective.measurementWindowOverlapAreaPx2 += overlapArea;
                        objective.measurementWindowMaxOverlapAreaPx2 =
                            std::max(objective.measurementWindowMaxOverlapAreaPx2, overlapArea);
                    }
                }

                std::sort(selectedAngles.begin(), selectedAngles.end());
                std::vector<double> spacings;
                spacings.reserve(selectedAngles.size());
                for (std::size_t i = 0; i < selectedAngles.size(); ++i) {
                    const double current = selectedAngles[i];
                    const double next = selectedAngles[(i + 1) % selectedAngles.size()] +
                                        (i + 1 == selectedAngles.size() ? 360.0 : 0.0);
                    spacings.push_back(next - current);
                }
                if (!spacings.empty()) {
                    double spacingSum = 0.0;
                    double spacingSumSq = 0.0;
                    objective.spacingMaxErrorDeg = 0.0;
                    for (double spacing : spacings) {
                        const double error = spacing - idealSpacingDeg;
                        spacingSum += spacing;
                        spacingSumSq += error * error;
                        objective.spacingMaxErrorDeg =
                            std::max(objective.spacingMaxErrorDeg, std::abs(error));
                    }
                    objective.spacingMeanDeg =
                        spacingSum / static_cast<double>(spacings.size());
                    objective.spacingRmseDeg =
                        std::sqrt(spacingSumSq / static_cast<double>(spacings.size()));
                }

                objective.valid = true;
                if (objectiveOut != nullptr) {
                    *objectiveOut = objective;
                }
                return true;
            };

        const auto objectiveBetterThan =
            [&](const SelectionObjective& lhs, const SelectionObjective& rhs) {
                if (lhs.valid != rhs.valid) {
                    return lhs.valid;
                }
                if (!lhs.valid) {
                    return false;
                }
                const auto strictlyLess = [](double a, double b, double epsilon) {
                    return a + epsilon < b;
                };

                const auto uniformSpacingPenalty = [&](const SelectionObjective& objective) {
                    const double spacingMaxExcess =
                        std::max(0.0, objective.spacingMaxErrorDeg - config.softSpacingMaxErrorDeg);
                    const double spacingRmseExcess =
                        std::max(0.0, objective.spacingRmseDeg - config.softSpacingRmseDeg);
                    const double targetMaxExcess =
                        std::max(0.0, objective.targetAngleMaxDeg - config.softTargetAngleMaxDeg);
                    const double targetMeanExcess =
                        std::max(0.0, objective.targetAngleMeanDeg - config.softTargetAngleMeanDeg);
                    return 0.30 * spacingMaxExcess * spacingMaxExcess +
                           0.18 * spacingRmseExcess * spacingRmseExcess +
                           0.12 * targetMaxExcess * targetMaxExcess +
                           0.06 * targetMeanExcess * targetMeanExcess;
                };
                const auto uniformSpacingWithinGuard =
                    [&](const SelectionObjective& objective) {
                        return objective.spacingMaxErrorDeg <=
                                   config.softSpacingMaxErrorDeg + 1e-9 &&
                               objective.spacingRmseDeg <=
                                   config.softSpacingRmseDeg + 1e-9 &&
                               objective.targetAngleMaxDeg <=
                                   config.softTargetAngleMaxDeg + 1e-9 &&
                               objective.targetAngleMeanDeg <=
                                   config.softTargetAngleMeanDeg + 1e-9;
                    };
                const auto softSpacingPenalty = [&](const SelectionObjective& objective) {
                    if (!config.softSpacingGuard || config.uniformAngleSelection) {
                        return 0.0;
                    }
                    const double overlapPenalty =
                        40.0 * static_cast<double>(objective.measurementWindowOverlapCount) +
                        0.004 * objective.measurementWindowOverlapAreaPx2 +
                        0.010 * objective.measurementWindowMaxOverlapAreaPx2;
                    const double spacingMaxExcess =
                        std::max(0.0, objective.spacingMaxErrorDeg - config.softSpacingMaxErrorDeg);
                    const double spacingRmseExcess =
                        std::max(0.0, objective.spacingRmseDeg - config.softSpacingRmseDeg);
                    const double targetMaxExcess =
                        std::max(0.0, objective.targetAngleMaxDeg - config.softTargetAngleMaxDeg);
                    const double targetMeanExcess =
                        std::max(0.0, objective.targetAngleMeanDeg - config.softTargetAngleMeanDeg);
                    const double windowPenalty =
                        static_cast<double>(objective.windowViolationCount) * 4.0 +
                        1.5 * objective.windowViolationMaxDeg * objective.windowViolationMaxDeg +
                        0.6 * objective.windowViolationMeanDeg * objective.windowViolationMeanDeg;
                    return overlapPenalty +
                           windowPenalty +
                           0.30 * spacingMaxExcess * spacingMaxExcess +
                           0.18 * spacingRmseExcess * spacingRmseExcess +
                           0.12 * targetMaxExcess * targetMaxExcess +
                           0.06 * targetMeanExcess * targetMeanExcess;
                };

                if (config.uniformAngleSelection) {
                    if (lhs.measurementWindowOverlapCount != rhs.measurementWindowOverlapCount) {
                        return lhs.measurementWindowOverlapCount < rhs.measurementWindowOverlapCount;
                    }
                    if (strictlyLess(lhs.measurementWindowOverlapAreaPx2,
                                     rhs.measurementWindowOverlapAreaPx2,
                                     1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.measurementWindowOverlapAreaPx2,
                                     lhs.measurementWindowOverlapAreaPx2,
                                     1e-6)) {
                        return false;
                    }
                    if (lhs.windowViolationCount != rhs.windowViolationCount) {
                        return lhs.windowViolationCount < rhs.windowViolationCount;
                    }
                    if (strictlyLess(lhs.windowViolationMaxDeg, rhs.windowViolationMaxDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.windowViolationMaxDeg, lhs.windowViolationMaxDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.windowViolationMeanDeg, rhs.windowViolationMeanDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.windowViolationMeanDeg, lhs.windowViolationMeanDeg, 1e-6)) {
                        return false;
                    }
                    if (config.softSpacingGuard) {
                        const bool lhsWithinGuard = uniformSpacingWithinGuard(lhs);
                        const bool rhsWithinGuard = uniformSpacingWithinGuard(rhs);
                        if (lhsWithinGuard != rhsWithinGuard) {
                            return lhsWithinGuard;
                        }
                    }
                } else if (config.softSpacingGuard) {
                    const double lhsSoftPenalty = softSpacingPenalty(lhs);
                    const double rhsSoftPenalty = softSpacingPenalty(rhs);
                    if (strictlyLess(lhsSoftPenalty, rhsSoftPenalty, 1e-9)) {
                        return true;
                    }
                    if (strictlyLess(rhsSoftPenalty, lhsSoftPenalty, 1e-9)) {
                        return false;
                    }
                }

                if (config.uniformAngleSelection) {
                    if (config.softSpacingGuard) {
                        const double lhsUniformPenalty = uniformSpacingPenalty(lhs);
                        const double rhsUniformPenalty = uniformSpacingPenalty(rhs);
                        if (strictlyLess(lhsUniformPenalty, rhsUniformPenalty, 1e-9)) {
                            return true;
                        }
                        if (strictlyLess(rhsUniformPenalty, lhsUniformPenalty, 1e-9)) {
                            return false;
                        }
                    }
                    if (strictlyLess(lhs.spacingMaxErrorDeg, rhs.spacingMaxErrorDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.spacingMaxErrorDeg, lhs.spacingMaxErrorDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.spacingRmseDeg, rhs.spacingRmseDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.spacingRmseDeg, lhs.spacingRmseDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.targetAngleMaxDeg, rhs.targetAngleMaxDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.targetAngleMaxDeg, lhs.targetAngleMaxDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.targetAngleMeanDeg, rhs.targetAngleMeanDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.targetAngleMeanDeg, lhs.targetAngleMeanDeg, 1e-6)) {
                        return false;
                    }
                }

                if (strictlyLess(lhs.rangePx, rhs.rangePx, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.rangePx, lhs.rangePx, 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.residualRmsePx, rhs.residualRmsePx, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.residualRmsePx, lhs.residualRmsePx, 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.residualMeanAbsPx, rhs.residualMeanAbsPx, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.residualMeanAbsPx, lhs.residualMeanAbsPx, 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.meanSelectionCost, rhs.meanSelectionCost, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.meanSelectionCost, lhs.meanSelectionCost, 1e-6)) {
                    return false;
                }
                if (config.uniformAngleSelection) {
                    if (config.softSpacingGuard) {
                        const double lhsUniformPenalty = uniformSpacingPenalty(lhs);
                        const double rhsUniformPenalty = uniformSpacingPenalty(rhs);
                        if (strictlyLess(lhsUniformPenalty, rhsUniformPenalty, 1e-9)) {
                            return true;
                        }
                        if (strictlyLess(rhsUniformPenalty, lhsUniformPenalty, 1e-9)) {
                            return false;
                        }
                    }
                    if (strictlyLess(lhs.spacingMaxErrorDeg, rhs.spacingMaxErrorDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.spacingMaxErrorDeg, lhs.spacingMaxErrorDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.spacingRmseDeg, rhs.spacingRmseDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.spacingRmseDeg, lhs.spacingRmseDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.targetAngleMaxDeg, rhs.targetAngleMaxDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.targetAngleMaxDeg, lhs.targetAngleMaxDeg, 1e-6)) {
                        return false;
                    }
                    if (strictlyLess(lhs.targetAngleMeanDeg, rhs.targetAngleMeanDeg, 1e-6)) {
                        return true;
                    }
                    if (strictlyLess(rhs.targetAngleMeanDeg, lhs.targetAngleMeanDeg, 1e-6)) {
                        return false;
                    }
                }
                return false;
            };

        const auto assignCandidateToPoint = [](Gbt57P2dPoint& point,
                                               const Gbt57CandidatePoint& candidate) {
            point.selected = true;
            point.imagePoint = candidate.imagePoint;
            point.globalPoint = candidate.globalPoint;
            point.selectedAngleDeg = candidate.selectedAngleDeg;
            point.angleDeltaDeg = candidate.angleDeltaDeg;
            point.qualityWeight = candidate.qualityWeight;
            point.confidence = candidate.confidence;
            point.gradient = candidate.gradient;
            point.selectionCost = candidate.cost;
        };

        const auto refineSelection =
            [&](std::vector<Gbt57P2dPoint>& points,
                SelectionObjective currentObjective) {
                for (int pass = 0; pass < 6; ++pass) {
                    bool improved = false;
                    for (std::size_t imageIndex = 0; imageIndex < points.size(); ++imageIndex) {
                        Gbt57P2dPoint bestPoint = points[imageIndex];
                        SelectionObjective bestLocalObjective = currentObjective;
                        for (const Gbt57CandidatePoint& candidate :
                             perImageCandidates[imageIndex]) {
                            std::vector<Gbt57P2dPoint> trialPoints = points;
                            assignCandidateToPoint(trialPoints[imageIndex], candidate);
                            SelectionObjective trialObjective;
                            if (!evaluateSelectionObjective(trialPoints, &trialObjective)) {
                                continue;
                            }
                            if (objectiveBetterThan(trialObjective, bestLocalObjective)) {
                                bestLocalObjective = trialObjective;
                                bestPoint = trialPoints[imageIndex];
                            }
                        }
                        if (objectiveBetterThan(bestLocalObjective, currentObjective)) {
                            points[imageIndex] = bestPoint;
                            currentObjective = bestLocalObjective;
                            improved = true;
                        }
                    }
                    if (!improved) {
                        break;
                    }
                }
                return currentObjective;
            };

        const std::vector<Gbt57P2dPoint> basePoints = result.points;
        std::vector<Gbt57P2dPoint> bestPoints = result.points;
        SelectionObjective bestObjective;
        evaluateSelectionObjective(bestPoints, &bestObjective);
        bestObjective = refineSelection(bestPoints, bestObjective);

        std::mt19937 rng(0x5A57u);
        for (int restart = 0; restart < config.rangeOptimizationRestarts; ++restart) {
            std::vector<Gbt57P2dPoint> trialPoints = basePoints;
            for (std::size_t imageIndex = 0; imageIndex < trialPoints.size(); ++imageIndex) {
                const std::vector<Gbt57CandidatePoint>& candidates =
                    perImageCandidates[imageIndex];
                if (candidates.empty()) {
                    continue;
                }
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                const double biased = std::pow(dist(rng), 2.0);
                const std::size_t randomIndex = std::min(
                    candidates.size() - 1,
                    static_cast<std::size_t>(
                        std::floor(biased * static_cast<double>(candidates.size()))));
                const std::size_t cycleIndex =
                    (static_cast<std::size_t>(restart) * 13 + imageIndex * 7) %
                    candidates.size();
                const std::size_t candidateIndex =
                    (restart % 3 == 0) ? cycleIndex : randomIndex;
                assignCandidateToPoint(trialPoints[imageIndex],
                                       candidates[candidateIndex]);
            }
            SelectionObjective trialObjective;
            if (!evaluateSelectionObjective(trialPoints, &trialObjective)) {
                continue;
            }
            trialObjective = refineSelection(trialPoints, trialObjective);
            if (objectiveBetterThan(trialObjective, bestObjective)) {
                bestObjective = trialObjective;
                bestPoints = std::move(trialPoints);
            }
        }

        if (bestObjective.valid) {
            result.points = std::move(bestPoints);
        }
    }

    std::vector<cv::Point2d> selectedGlobalPoints;
    selectedGlobalPoints.reserve(result.points.size());
    for (const Gbt57P2dPoint& point : result.points) {
        if (point.selected) {
            selectedGlobalPoints.push_back(point.globalPoint);
        }
    }
    if (static_cast<int>(selectedGlobalPoints.size()) != config.expectedPointCount) {
        result.message = "selected point count does not match GB/T 5.7 expectation";
        return result;
    }

    result.selectedCircle = fitCircleMinRange(selectedGlobalPoints);
    if (!result.selectedCircle.ok) {
        result.message = "failed to fit the 25 single-point circle";
        return result;
    }
    for (Gbt57P2dPoint& point : result.points) {
        point.selectedAngleDeg = angleOfPointDeg(point.globalPoint, result.selectedCircle);
        const double windowAngleDeg = angleOfPointDeg(point.globalPoint, result.globalCircle);
        point.angleDeltaDeg =
            circularAbsDeltaDeg(windowAngleDeg, point.viewAngleDeg);
        point.windowViolationDeg =
            std::max(0.0, point.angleDeltaDeg - point.windowHalfAngleDeg);
    }

    int refineAppliedIterations = 0;
    if (config.selectedPointTransformRefine &&
        config.selectedPointTransformRefineIterations > 0 &&
        config.selectedPointTransformRefineMaxStepPx > 0.0 &&
        config.selectedPointTransformRefineGain > 0.0) {
        for (int iteration = 0; iteration < config.selectedPointTransformRefineIterations; ++iteration) {
            std::vector<cv::Point2d> refinedEvaluationPoints;
            refinedEvaluationPoints.reserve(result.points.size());
            for (const Gbt57P2dPoint& point : result.points) {
                const std::size_t imageIndex =
                    static_cast<std::size_t>(std::max(0, point.imageIndex));
                if (imageIndex >= evaluationTransforms.size()) {
                    continue;
                }
                const cv::Point2d currentGlobal =
                    transformPointByMatrix(evaluationTransforms[imageIndex], point.imagePoint);
                refinedEvaluationPoints.push_back(
                    pureEllipseRectifiedEvaluation
                        ? rectifyPointByEllipse(globalEllipseRectification, currentGlobal)
                        : currentGlobal);
            }

            const CircleModel refinedCircle = fitCircleMinRange(refinedEvaluationPoints);
            if (!refinedCircle.ok) {
                break;
            }

            double maxApplied = 0.0;
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                const std::size_t imageIndex =
                    static_cast<std::size_t>(std::max(0, result.points[i].imageIndex));
                if (imageIndex >= evaluationTransforms.size()) {
                    continue;
                }
                const cv::Point2d global =
                    transformPointByMatrix(evaluationTransforms[imageIndex],
                                           result.points[i].imagePoint);
                const cv::Point2d evaluationPoint =
                    pureEllipseRectifiedEvaluation
                        ? rectifyPointByEllipse(globalEllipseRectification, global)
                        : global;
                const double dx = evaluationPoint.x - refinedCircle.centerX;
                const double dy = evaluationPoint.y - refinedCircle.centerY;
                const double radius = std::hypot(dx, dy);
                if (radius <= 1e-9 || !std::isfinite(radius)) {
                    continue;
                }
                const double residual = radius - refinedCircle.radius;
                if (!std::isfinite(residual)) {
                    continue;
                }
                const cv::Point2d evaluationCorrection{
                    -config.selectedPointTransformRefineGain * residual * dx / radius,
                    -config.selectedPointTransformRefineGain * residual * dy / radius
                };
                cv::Point2d correction =
                    pureEllipseRectifiedEvaluation
                        ? inverseRectifiedDeltaToRaw(globalEllipseRectification,
                                                    evaluationCorrection)
                        : evaluationCorrection;
                const double correctionNorm = cv::norm(correction);
                if (!std::isfinite(correctionNorm) || correctionNorm <= 1e-12) {
                    continue;
                }
                if (correctionNorm > config.selectedPointTransformRefineMaxStepPx) {
                    correction *= config.selectedPointTransformRefineMaxStepPx / correctionNorm;
                }
                evaluationTransforms[imageIndex].at<double>(0, 2) += correction.x;
                evaluationTransforms[imageIndex].at<double>(1, 2) += correction.y;
                maxApplied = std::max(maxApplied, cv::norm(correction));
            }

            if (maxApplied < 1e-4) {
                break;
            }
            ++refineAppliedIterations;
        }

        selectedGlobalPoints.clear();
        selectedGlobalPoints.reserve(result.points.size());
        for (Gbt57P2dPoint& point : result.points) {
            const std::size_t imageIndex =
                static_cast<std::size_t>(std::max(0, point.imageIndex));
            if (imageIndex < evaluationTransforms.size()) {
                point.globalPoint =
                    transformPointByMatrix(evaluationTransforms[imageIndex], point.imagePoint);
            }
            point.selectedAngleDeg =
                angleOfPointDeg(point.globalPoint, result.selectedCircle);
            const double windowAngleDeg = angleOfPointDeg(point.globalPoint, result.globalCircle);
            point.angleDeltaDeg =
                circularAbsDeltaDeg(windowAngleDeg, point.viewAngleDeg);
            point.windowViolationDeg =
                std::max(0.0, point.angleDeltaDeg - point.windowHalfAngleDeg);
            selectedGlobalPoints.push_back(point.globalPoint);
        }
        result.selectedCircle = fitCircleMinRange(selectedGlobalPoints);
        if (!result.selectedCircle.ok) {
            result.message = "failed to refit the 25 single-point circle after transform refinement";
            return result;
        }
        for (Gbt57P2dPoint& point : result.points) {
            point.selectedAngleDeg =
                angleOfPointDeg(point.globalPoint, result.selectedCircle);
            const double windowAngleDeg = angleOfPointDeg(point.globalPoint, result.globalCircle);
            point.angleDeltaDeg =
                circularAbsDeltaDeg(windowAngleDeg, point.viewAngleDeg);
            point.windowViolationDeg =
                std::max(0.0, point.angleDeltaDeg - point.windowHalfAngleDeg);
        }
    }

    if ((config.optimizeSelectedRange || config.uniformAngleSelection) &&
        !perImageCandidates.empty()) {
        struct FinalWindowObjective {
            bool valid{false};
            CircleModel angleCircle;
            int measurementWindowOverlapCount{std::numeric_limits<int>::max()};
            double measurementWindowOverlapAreaPx2{std::numeric_limits<double>::infinity()};
            double measurementWindowMaxOverlapAreaPx2{std::numeric_limits<double>::infinity()};
            int windowViolationCount{std::numeric_limits<int>::max()};
            double windowViolationMeanDeg{std::numeric_limits<double>::infinity()};
            double windowViolationMaxDeg{std::numeric_limits<double>::infinity()};
            double rangePx{std::numeric_limits<double>::infinity()};
            double residualRmsePx{std::numeric_limits<double>::infinity()};
            double meanSelectionCost{std::numeric_limits<double>::infinity()};
        };

        const auto evaluateFinalWindowObjective =
            [&](std::vector<Gbt57P2dPoint>& points,
                FinalWindowObjective* objectiveOut = nullptr) {
                FinalWindowObjective objective;
                std::vector<cv::Point2d> globals;
                std::vector<cv::Point2d> evaluationGlobals;
                std::vector<cv::RotatedRect> windowRects;
                globals.reserve(points.size());
                evaluationGlobals.reserve(points.size());
                windowRects.reserve(points.size());
                for (Gbt57P2dPoint& point : points) {
                    if (!point.selected) {
                        if (objectiveOut != nullptr) {
                            *objectiveOut = objective;
                        }
                        return false;
                    }
                    const std::size_t imageIndex =
                        static_cast<std::size_t>(std::max(0, point.imageIndex));
                    if (imageIndex >= evaluationTransforms.size()) {
                        if (objectiveOut != nullptr) {
                            *objectiveOut = objective;
                        }
                        return false;
                    }
                    point.globalPoint =
                        transformPointByMatrix(evaluationTransforms[imageIndex],
                                               point.imagePoint);
                    globals.push_back(point.globalPoint);
                    evaluationGlobals.push_back(
                        pureEllipseRectifiedEvaluation
                            ? rectifyPointByEllipse(globalEllipseRectification,
                                                    point.globalPoint)
                            : point.globalPoint);
                    windowRects.push_back(
                        buildGlobalMeasurementWindow(evaluationTransforms[imageIndex],
                                                     point.imagePoint,
                                                     std::max(0.0, point.windowHalfSizePx)));
                }

                objective.angleCircle = fitCircleMinRange(globals);
                if (!objective.angleCircle.ok) {
                    if (objectiveOut != nullptr) {
                        *objectiveOut = objective;
                    }
                    return false;
                }

                double rangePx = 0.0;
                const CircleModel evaluationCircle =
                    fitCircleMinRange(evaluationGlobals, &rangePx);
                if (!evaluationCircle.ok) {
                    if (objectiveOut != nullptr) {
                        *objectiveOut = objective;
                    }
                    return false;
                }

                double violationSum = 0.0;
                double residualSumSq = 0.0;
                double selectionCostSum = 0.0;
                objective.windowViolationCount = 0;
                objective.windowViolationMaxDeg = 0.0;
                for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
                    Gbt57P2dPoint& point = points[pointIndex];
                    const cv::Point2d& evaluationPoint = evaluationGlobals[pointIndex];
                    point.selectedAngleDeg =
                        angleOfPointDeg(point.globalPoint, objective.angleCircle);
                    const double windowAngleDeg =
                        angleOfPointDeg(point.globalPoint, result.globalCircle);
                    point.angleDeltaDeg =
                        circularAbsDeltaDeg(windowAngleDeg, point.viewAngleDeg);
                    point.windowViolationDeg =
                        std::max(0.0, point.angleDeltaDeg - point.windowHalfAngleDeg);
                    violationSum += point.windowViolationDeg;
                    objective.windowViolationMaxDeg =
                        std::max(objective.windowViolationMaxDeg,
                                 point.windowViolationDeg);
                    objective.windowViolationCount +=
                        point.windowViolationDeg > 1e-9 ? 1 : 0;
                    const double radius =
                        std::hypot(evaluationPoint.x - evaluationCircle.centerX,
                                   evaluationPoint.y - evaluationCircle.centerY);
                    const double residual = radius - evaluationCircle.radius;
                    residualSumSq += residual * residual;
                    selectionCostSum += point.selectionCost;
                }
                const double pointCount =
                    std::max(1.0, static_cast<double>(points.size()));
                objective.windowViolationMeanDeg = violationSum / pointCount;
                objective.residualRmsePx = std::sqrt(residualSumSq / pointCount);
                objective.meanSelectionCost = selectionCostSum / pointCount;
                objective.rangePx = rangePx;

                objective.measurementWindowOverlapCount = 0;
                objective.measurementWindowOverlapAreaPx2 = 0.0;
                objective.measurementWindowMaxOverlapAreaPx2 = 0.0;
                for (std::size_t lhs = 0; lhs < windowRects.size(); ++lhs) {
                    for (std::size_t rhs = lhs + 1; rhs < windowRects.size(); ++rhs) {
                        const double overlapArea =
                            rotatedRectIntersectionArea(windowRects[lhs], windowRects[rhs]);
                        if (overlapArea <= 1e-6) {
                            continue;
                        }
                        ++objective.measurementWindowOverlapCount;
                        objective.measurementWindowOverlapAreaPx2 += overlapArea;
                        objective.measurementWindowMaxOverlapAreaPx2 =
                            std::max(objective.measurementWindowMaxOverlapAreaPx2,
                                     overlapArea);
                    }
                }

                objective.valid = true;
                if (objectiveOut != nullptr) {
                    *objectiveOut = objective;
                }
                return true;
            };

        const auto finalWindowObjectiveBetter =
            [](const FinalWindowObjective& lhs, const FinalWindowObjective& rhs) {
                if (lhs.valid != rhs.valid) {
                    return lhs.valid;
                }
                if (!lhs.valid) {
                    return false;
                }
                const auto strictlyLess = [](double a, double b, double epsilon) {
                    return a + epsilon < b;
                };
                if (lhs.measurementWindowOverlapCount !=
                    rhs.measurementWindowOverlapCount) {
                    return lhs.measurementWindowOverlapCount <
                           rhs.measurementWindowOverlapCount;
                }
                if (strictlyLess(lhs.measurementWindowOverlapAreaPx2,
                                 rhs.measurementWindowOverlapAreaPx2,
                                 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.measurementWindowOverlapAreaPx2,
                                 lhs.measurementWindowOverlapAreaPx2,
                                 1e-6)) {
                    return false;
                }
                if (lhs.windowViolationCount != rhs.windowViolationCount) {
                    return lhs.windowViolationCount < rhs.windowViolationCount;
                }
                if (strictlyLess(lhs.windowViolationMaxDeg,
                                 rhs.windowViolationMaxDeg,
                                 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.windowViolationMaxDeg,
                                 lhs.windowViolationMaxDeg,
                                 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.windowViolationMeanDeg,
                                 rhs.windowViolationMeanDeg,
                                 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.windowViolationMeanDeg,
                                 lhs.windowViolationMeanDeg,
                                 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.rangePx, rhs.rangePx, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.rangePx, lhs.rangePx, 1e-6)) {
                    return false;
                }
                if (strictlyLess(lhs.residualRmsePx, rhs.residualRmsePx, 1e-6)) {
                    return true;
                }
                if (strictlyLess(rhs.residualRmsePx, lhs.residualRmsePx, 1e-6)) {
                    return false;
                }
                return strictlyLess(lhs.meanSelectionCost, rhs.meanSelectionCost, 1e-6);
            };

        const auto candidateToFinalPoint =
            [&](std::size_t imageIndex, const Gbt57CandidatePoint& candidate) {
                Gbt57P2dPoint point = result.points[imageIndex];
                point.selected = true;
                point.imagePoint = candidate.imagePoint;
                point.globalPoint =
                    transformPointByMatrix(evaluationTransforms[imageIndex],
                                           candidate.imagePoint);
                point.selectedAngleDeg = candidate.selectedAngleDeg;
                point.angleDeltaDeg = candidate.angleDeltaDeg;
                point.qualityWeight = candidate.qualityWeight;
                point.confidence = candidate.confidence;
                point.gradient = candidate.gradient;
                point.selectionCost = candidate.cost;
                point.measurementWindowInsideImage =
                    measurementWindowInsideImage(point.imagePoint,
                                                 images[imageIndex].size(),
                                                 point.windowHalfSizePx,
                                                 &point.windowMarginPx);
                return point;
            };

        FinalWindowObjective currentFinalObjective;
        if (evaluateFinalWindowObjective(result.points, &currentFinalObjective)) {
            for (int pass = 0; pass < 6; ++pass) {
                bool improved = false;
                for (std::size_t imageIndex = 0;
                     imageIndex < result.points.size() &&
                     imageIndex < perImageCandidates.size();
                     ++imageIndex) {
                    if ((currentFinalObjective.measurementWindowOverlapCount <= 0 &&
                         result.points[imageIndex].windowViolationDeg <= 1e-9) ||
                        perImageCandidates[imageIndex].empty()) {
                        continue;
                    }
                    Gbt57P2dPoint bestPoint = result.points[imageIndex];
                    FinalWindowObjective bestObjective = currentFinalObjective;
                    double bestPointViolationDeg =
                        result.points[imageIndex].windowViolationDeg;
                    for (const Gbt57CandidatePoint& candidate :
                         perImageCandidates[imageIndex]) {
                        if (config.uniformAngleSelection &&
                            candidate.angleDeltaDeg >
                                result.points[imageIndex].windowHalfAngleDeg + 1e-9) {
                            continue;
                        }
                        std::vector<Gbt57P2dPoint> trialPoints = result.points;
                        trialPoints[imageIndex] =
                            candidateToFinalPoint(imageIndex, candidate);
                        if (!trialPoints[imageIndex].measurementWindowInsideImage) {
                            continue;
                        }
                        FinalWindowObjective trialObjective;
                        if (!evaluateFinalWindowObjective(trialPoints,
                                                          &trialObjective)) {
                            continue;
                        }
                        const double trialPointViolationDeg =
                            trialPoints[imageIndex].windowViolationDeg;
                        const bool pointViolationImproved =
                            trialPointViolationDeg + 1e-9 < bestPointViolationDeg;
                        const bool pointViolationTied =
                            std::abs(trialPointViolationDeg -
                                     bestPointViolationDeg) <= 1e-9;
                        if (pointViolationImproved ||
                            (pointViolationTied &&
                             finalWindowObjectiveBetter(trialObjective,
                                                        bestObjective))) {
                            bestObjective = trialObjective;
                            bestPoint = trialPoints[imageIndex];
                            bestPointViolationDeg = trialPointViolationDeg;
                        }
                    }
                    if (bestPointViolationDeg + 1e-9 <
                            result.points[imageIndex].windowViolationDeg ||
                        finalWindowObjectiveBetter(bestObjective,
                                                   currentFinalObjective)) {
                        result.points[imageIndex] = bestPoint;
                        currentFinalObjective = bestObjective;
                        improved = true;
                    }
                }
                if (!improved) {
                    break;
                }
            }
            evaluateFinalWindowObjective(result.points, &currentFinalObjective);
            if (currentFinalObjective.valid) {
                result.selectedCircle = currentFinalObjective.angleCircle;
            }
        }
    }

    result.selectedPointTransformRefineAppliedIterations = refineAppliedIterations;
    result.transformShiftPx.assign(evaluationTransforms.size(), 0.0);
    double transformShiftSumPx = 0.0;
    int transformShiftCount = 0;
    for (std::size_t imageIndex = 0;
         imageIndex < evaluationTransforms.size() &&
         imageIndex < originalEvaluationTransforms.size();
         ++imageIndex) {
        if (evaluationTransforms[imageIndex].rows < 2 ||
            evaluationTransforms[imageIndex].cols < 3 ||
            originalEvaluationTransforms[imageIndex].rows < 2 ||
            originalEvaluationTransforms[imageIndex].cols < 3) {
            continue;
        }
        const double dx =
            evaluationTransforms[imageIndex].at<double>(0, 2) -
            originalEvaluationTransforms[imageIndex].at<double>(0, 2);
        const double dy =
            evaluationTransforms[imageIndex].at<double>(1, 2) -
            originalEvaluationTransforms[imageIndex].at<double>(1, 2);
        const double shiftPx = std::hypot(dx, dy);
        if (!std::isfinite(shiftPx)) {
            continue;
        }
        result.transformShiftPx[imageIndex] = shiftPx;
        result.selectedPointTransformRefineMaxShiftPx =
            std::max(result.selectedPointTransformRefineMaxShiftPx, shiftPx);
        if (shiftPx > 1e-6) {
            transformShiftSumPx += shiftPx;
            ++transformShiftCount;
        }
    }
    result.selectedPointTransformRefineAdjustedImageCount = transformShiftCount;
    if (transformShiftCount > 0) {
        result.selectedPointTransformRefineMeanShiftPx =
            transformShiftSumPx / static_cast<double>(transformShiftCount);
    }
    for (Gbt57P2dPoint& point : result.points) {
        const std::size_t imageIndex =
            static_cast<std::size_t>(std::max(0, point.imageIndex));
        if (imageIndex < result.transformShiftPx.size()) {
            point.transformShiftPx = result.transformShiftPx[imageIndex];
        }
    }

    std::vector<double> fieldBiases(result.points.size(), 0.0);
    if (config.fieldBiasCompensation || config.localFieldBiasCompensation) {
        for (std::size_t imageIndex = 0; imageIndex < result.points.size() &&
                                         imageIndex < edges.size(); ++imageIndex) {
            std::vector<double> residuals;
            residuals.reserve(edges[imageIndex].raw.size());
            for (const cv::Point2d& imagePoint : edges[imageIndex].raw) {
                const cv::Point2d global =
                    transformPointByMatrix(evaluationTransforms[imageIndex], imagePoint);
                if (config.localFieldBiasCompensation) {
                    const double angle = angleOfPointDeg(global, result.selectedCircle);
                    const double halfAngle =
                        std::max(0.1, config.localFieldBiasHalfAngleDeg);
                    if (imageIndex >= result.points.size() ||
                        circularAbsDeltaDeg(angle,
                                            result.points[imageIndex].selectedAngleDeg) >
                            halfAngle) {
                        continue;
                    }
                }
                residuals.push_back(std::hypot(global.x - result.selectedCircle.centerX,
                                               global.y - result.selectedCircle.centerY) -
                                    result.selectedCircle.radius);
            }
            if (!residuals.empty()) {
                fieldBiases[imageIndex] += quantileOf(residuals, 0.5);
            }
        }
    }
    if (config.angularBiasCompensation && config.angularBiasGain > 0.0) {
        const int order = std::clamp(config.angularBiasOrder, 1, 6);
        const int dimension = 1 + 2 * order;
        std::vector<double> basis(static_cast<std::size_t>(dimension), 0.0);
        const auto fillBasis = [&](double angleRad) {
            basis[0] = 1.0;
            for (int harmonic = 1; harmonic <= order; ++harmonic) {
                const double a = static_cast<double>(harmonic) * angleRad;
                basis[static_cast<std::size_t>(2 * harmonic - 1)] = std::cos(a);
                basis[static_cast<std::size_t>(2 * harmonic)] = std::sin(a);
            }
        };

        for (std::size_t leaveOut = 0; leaveOut < result.points.size(); ++leaveOut) {
            cv::Mat normal = cv::Mat::zeros(dimension, dimension, CV_64F);
            cv::Mat rhs = cv::Mat::zeros(dimension, 1, CV_64F);
            int sampleCount = 0;
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                if (i == leaveOut) {
                    continue;
                }
                const Gbt57P2dPoint& sample = result.points[i];
                const double angleRad = sample.selectedAngleDeg * CV_PI / 180.0;
                const double residual =
                    std::hypot(sample.globalPoint.x - result.selectedCircle.centerX,
                               sample.globalPoint.y - result.selectedCircle.centerY) -
                    result.selectedCircle.radius;
                if (!std::isfinite(residual)) {
                    continue;
                }
                fillBasis(angleRad);
                double weight = std::clamp(sample.qualityWeight, 0.05, 4.0);
                weight *= 1.0 / (1.0 + std::abs(residual) / 0.35);
                for (int row = 0; row < dimension; ++row) {
                    rhs.at<double>(row, 0) += weight * basis[static_cast<std::size_t>(row)] *
                                              residual;
                    for (int col = 0; col < dimension; ++col) {
                        normal.at<double>(row, col) +=
                            weight * basis[static_cast<std::size_t>(row)] *
                            basis[static_cast<std::size_t>(col)];
                    }
                }
                ++sampleCount;
            }

            cv::Mat coefficients;
            if (sampleCount >= dimension * 3 &&
                cv::solve(normal, rhs, coefficients, cv::DECOMP_SVD)) {
                const Gbt57P2dPoint& sample = result.points[leaveOut];
                fillBasis(sample.selectedAngleDeg * CV_PI / 180.0);
                double predictedBias = 0.0;
                for (int row = 0; row < dimension; ++row) {
                    predictedBias += basis[static_cast<std::size_t>(row)] *
                                     coefficients.at<double>(row, 0);
                }
                fieldBiases[leaveOut] += config.angularBiasGain * predictedBias;
            }
        }
    }
    if (config.ellipseNormalizationCompensation &&
        !pureEllipseRectifiedEvaluation &&
        selectedGlobalPoints.size() >= 5) {
        if (result.globalEllipse.ok &&
            result.globalEllipse.axisRatio > 0.20 &&
            result.globalEllipse.axisRatio < 1.0001) {
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                const double rawRadius = std::hypot(
                    result.points[i].globalPoint.x - result.selectedCircle.centerX,
                    result.points[i].globalPoint.y - result.selectedCircle.centerY);
                const double correctedRadius =
                    ellipseEquivalentRadiusPx(result.globalEllipse, result.points[i].globalPoint);
                if (correctedRadius > 0.0) {
                    fieldBiases[i] += rawRadius - correctedRadius;
                }
            }
        } else {
            double wSum = 0.0;
            double covXX = 0.0;
            double covYY = 0.0;
            double covXY = 0.0;
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                const Gbt57P2dPoint& point = result.points[i];
                const double weight = std::clamp(point.qualityWeight, 0.05, 4.0);
                const double dx = point.globalPoint.x - result.selectedCircle.centerX;
                const double dy = point.globalPoint.y - result.selectedCircle.centerY;
                covXX += weight * dx * dx;
                covYY += weight * dy * dy;
                covXY += weight * dx * dy;
                wSum += weight;
            }
            if (wSum > 1e-9) {
                covXX /= wSum;
                covYY /= wSum;
                covXY /= wSum;
                const double trace = covXX + covYY;
                const double disc =
                    std::sqrt(std::max(0.0, (covXX - covYY) * (covXX - covYY) +
                                               4.0 * covXY * covXY));
                const double lambda1 = 0.5 * (trace + disc);
                const double lambda2 = 0.5 * (trace - disc);
                if (lambda1 > 1e-9 && lambda2 > 1e-9) {
                    cv::Point2d axis1;
                    if (std::abs(covXY) > 1e-12) {
                        axis1 = {lambda1 - covYY, covXY};
                    } else if (covXX >= covYY) {
                        axis1 = {1.0, 0.0};
                    } else {
                        axis1 = {0.0, 1.0};
                    }
                    const double axis1Norm = std::hypot(axis1.x, axis1.y);
                    if (axis1Norm > 1e-12) {
                        axis1.x /= axis1Norm;
                        axis1.y /= axis1Norm;
                        const cv::Point2d axis2{-axis1.y, axis1.x};

                        double meanNormalizedNorm = 0.0;
                        std::vector<double> normalizedNorms(result.points.size(), 0.0);
                        for (std::size_t i = 0; i < result.points.size(); ++i) {
                            const Gbt57P2dPoint& point = result.points[i];
                            const double dx = point.globalPoint.x - result.selectedCircle.centerX;
                            const double dy = point.globalPoint.y - result.selectedCircle.centerY;
                            const double u1 = axis1.x * dx + axis1.y * dy;
                            const double u2 = axis2.x * dx + axis2.y * dy;
                            const double normalizedNorm =
                                std::sqrt((u1 * u1) / lambda1 + (u2 * u2) / lambda2);
                            normalizedNorms[i] = normalizedNorm;
                            meanNormalizedNorm += normalizedNorm;
                        }
                        meanNormalizedNorm /= std::max(1.0, static_cast<double>(result.points.size()));
                        if (meanNormalizedNorm > 1e-9) {
                            const double scale =
                                result.selectedCircle.radius / meanNormalizedNorm;
                            for (std::size_t i = 0; i < result.points.size(); ++i) {
                                const double rawRadius = std::hypot(
                                    result.points[i].globalPoint.x - result.selectedCircle.centerX,
                                    result.points[i].globalPoint.y - result.selectedCircle.centerY);
                                const double correctedRadius = normalizedNorms[i] * scale;
                                fieldBiases[i] += rawRadius - correctedRadius;
                            }
                        }
                    }
                }
            }
        }
    }

    const auto countWindowOverlapsForHalfSize = [&](double halfSizePx) {
        struct WindowOverlapStats {
            int count{0};
            double areaPx2{0.0};
            double maxAreaPx2{0.0};
        };
        WindowOverlapStats stats;
        std::vector<cv::RotatedRect> rects;
        rects.reserve(result.points.size());
        for (const Gbt57P2dPoint& point : result.points) {
            const std::size_t imageIndex =
                static_cast<std::size_t>(std::max(0, point.imageIndex));
            if (!point.selected || imageIndex >= evaluationTransforms.size()) {
                rects.emplace_back();
                continue;
            }
            rects.push_back(buildGlobalMeasurementWindow(evaluationTransforms[imageIndex],
                                                         point.imagePoint,
                                                         halfSizePx));
        }
        for (std::size_t lhs = 0; lhs < rects.size(); ++lhs) {
            for (std::size_t rhs = lhs + 1; rhs < rects.size(); ++rhs) {
                const double overlapArea =
                    rotatedRectIntersectionArea(rects[lhs], rects[rhs]);
                if (overlapArea <= 1e-6) {
                    continue;
                }
                ++stats.count;
                stats.areaPx2 += overlapArea;
                stats.maxAreaPx2 = std::max(stats.maxAreaPx2, overlapArea);
            }
        }
        return stats;
    };
    if (config.uniformAngleSelection && result.measurementWindowHalfSizePx > 1.0) {
        double effectiveHalfSizePx = result.measurementWindowHalfSizePx;
        for (int iteration = 0; iteration < 32; ++iteration) {
            const auto overlapStats = countWindowOverlapsForHalfSize(effectiveHalfSizePx);
            if (overlapStats.count == 0 || effectiveHalfSizePx <= 8.0) {
                break;
            }
            effectiveHalfSizePx = std::max(8.0, effectiveHalfSizePx * 0.90);
        }
        result.measurementWindowHalfSizePx = effectiveHalfSizePx;
        for (Gbt57P2dPoint& point : result.points) {
            point.windowHalfSizePx = effectiveHalfSizePx;
        }
    }

    std::vector<cv::RotatedRect> selectedWindowRects;
    selectedWindowRects.reserve(result.points.size());
    result.measurementWindowInsideImageCount = 0;
    result.measurementWindowOverlapCount = 0;
    result.measurementWindowOverlapAreaPx2 = 0.0;
    result.measurementWindowMaxOverlapAreaPx2 = 0.0;
    for (Gbt57P2dPoint& point : result.points) {
        const std::size_t imageIndex = static_cast<std::size_t>(std::max(0, point.imageIndex));
        double marginPx = 0.0;
        if (imageIndex < images.size()) {
            point.measurementWindowInsideImage =
                measurementWindowInsideImage(point.imagePoint,
                                             images[imageIndex].size(),
                                             std::max(0.0, point.windowHalfSizePx),
                                             &marginPx);
        } else {
            point.measurementWindowInsideImage = false;
        }
        point.windowMarginPx = marginPx;
        if (point.measurementWindowInsideImage) {
            ++result.measurementWindowInsideImageCount;
        }
        if (imageIndex < evaluationTransforms.size()) {
            const cv::RotatedRect rect =
                buildGlobalMeasurementWindow(evaluationTransforms[imageIndex],
                                             point.imagePoint,
                                             std::max(0.0, point.windowHalfSizePx));
            point.globalWindowCenter = cv::Point2d(rect.center.x, rect.center.y);
            point.globalWindowCorners = rotatedRectCorners(rect);
            selectedWindowRects.push_back(rect);
        } else {
            selectedWindowRects.emplace_back();
        }
    }
    for (std::size_t lhs = 0; lhs < selectedWindowRects.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < selectedWindowRects.size(); ++rhs) {
            const double overlapArea =
                rotatedRectIntersectionArea(selectedWindowRects[lhs], selectedWindowRects[rhs]);
            if (overlapArea <= 1e-6) {
                continue;
            }
            ++result.measurementWindowOverlapCount;
            result.measurementWindowOverlapAreaPx2 += overlapArea;
            result.measurementWindowMaxOverlapAreaPx2 =
                std::max(result.measurementWindowMaxOverlapAreaPx2, overlapArea);
            result.points[lhs].windowOverlapCount += 1;
            result.points[rhs].windowOverlapCount += 1;
            result.points[lhs].windowOverlapAreaPx2 += overlapArea;
            result.points[rhs].windowOverlapAreaPx2 += overlapArea;
            result.points[lhs].windowMaxOverlapAreaPx2 =
                std::max(result.points[lhs].windowMaxOverlapAreaPx2, overlapArea);
            result.points[rhs].windowMaxOverlapAreaPx2 =
                std::max(result.points[rhs].windowMaxOverlapAreaPx2, overlapArea);
        }
    }

    result.minRadiusPx = std::numeric_limits<double>::infinity();
    result.maxRadiusPx = 0.0;
    const bool biasCompensationEnabled =
        config.fieldBiasCompensation ||
        config.localFieldBiasCompensation ||
        config.angularBiasCompensation ||
        config.ellipseNormalizationCompensation;
    std::vector<cv::Point2d> correctedEvaluationPoints;
    if (pureEllipseRectifiedEvaluation) {
        correctedEvaluationPoints.reserve(result.points.size());
        for (const Gbt57P2dPoint& point : result.points) {
            correctedEvaluationPoints.push_back(
                rectifyPointByEllipse(globalEllipseRectification, point.globalPoint));
        }
        result.correctedEvaluationCircle =
            fitCircleMinRange(correctedEvaluationPoints);
        if (!result.correctedEvaluationCircle.ok) {
            result.message = "failed to fit ellipse-rectified 25-point circle";
            return result;
        }
    } else {
        result.correctedEvaluationCircle = result.selectedCircle;
    }
    result.selectedCircleCalibratedPixelSizeUm =
        sphereCalibratedPixelSizeUm(result.sphereDiameterMm, result.selectedCircle);
    result.correctedEvaluationCalibratedPixelSizeUm =
        sphereCalibratedPixelSizeUm(result.sphereDiameterMm, result.correctedEvaluationCircle);
    if (result.correctedEvaluationCalibratedPixelSizeUm > 0.0) {
        result.effectivePixelSizeUm = result.correctedEvaluationCalibratedPixelSizeUm;
        result.effectivePixelSizeSource = "sphere_diameter_corrected_evaluation_circle";
    } else if (result.selectedCircleCalibratedPixelSizeUm > 0.0) {
        result.effectivePixelSizeUm = result.selectedCircleCalibratedPixelSizeUm;
        result.effectivePixelSizeSource = "sphere_diameter_selected_circle";
    } else if (result.sphereCalibratedPixelSizeUm > 0.0) {
        result.effectivePixelSizeUm = result.sphereCalibratedPixelSizeUm;
        result.effectivePixelSizeSource = "sphere_diameter_global_circle";
    } else {
        result.effectivePixelSizeUm = result.configuredPixelSizeUm;
        result.effectivePixelSizeSource =
            result.configuredPixelSizeUm > 0.0 ? "configured_fov_or_pixel_size" : "unavailable";
    }
    for (std::size_t i = 0; i < result.points.size(); ++i) {
        Gbt57P2dPoint& point = result.points[i];
        point.radiusPx =
            std::hypot(point.globalPoint.x - result.selectedCircle.centerX,
                       point.globalPoint.y - result.selectedCircle.centerY);
        point.residualPx = point.radiusPx - result.selectedCircle.radius;
        point.fieldBiasPx = i < fieldBiases.size() ? fieldBiases[i] : 0.0;
        if (pureEllipseRectifiedEvaluation) {
            const cv::Point2d& correctedPoint = correctedEvaluationPoints[i];
            point.correctedRadiusPx =
                std::hypot(correctedPoint.x - result.correctedEvaluationCircle.centerX,
                           correctedPoint.y - result.correctedEvaluationCircle.centerY);
            point.correctedResidualPx =
                point.correctedRadiusPx - result.correctedEvaluationCircle.radius;
        } else {
            point.correctedRadiusPx =
                biasCompensationEnabled
                    ? point.radiusPx - point.fieldBiasPx
                    : point.radiusPx;
            point.correctedResidualPx =
                point.correctedRadiusPx - result.selectedCircle.radius;
        }
        point.preFilterCorrectedRadiusPx = point.correctedRadiusPx;
        point.preFilterCorrectedResidualPx = point.correctedResidualPx;
        if (i < result.candidateCoverage.size()) {
            result.candidateCoverage[i].selectedAngleDeltaDeg = point.angleDeltaDeg;
            result.candidateCoverage[i].selectedRadiusDeltaPx = std::abs(point.residualPx);
        }
    }

    result.circularResidualHampelRadius =
        std::max(0, config.circularResidualHampelRadius);
    result.circularResidualHampelSigma =
        std::max(0.0, config.circularResidualHampelSigma);
    result.circularResidualHampelApplied =
        result.circularResidualHampelRadius > 0 &&
        result.circularResidualHampelSigma > 0.0 &&
        result.points.size() >= 3;
    const int defaultMedianFilterRadius =
        std::max(0, config.circularResidualMedianFilterRadius);
    const double defaultFilterBlend =
        std::clamp(config.circularResidualFilterBlend, 0.0, 1.0);
    result.circularResidualMedianFilterRadius = defaultMedianFilterRadius;
    result.circularResidualMedianFilterApplied =
        defaultMedianFilterRadius > 0 &&
        result.points.size() >= 3;
    result.circularResidualFilterBlend = defaultFilterBlend;
    double repeatabilityBandMinUm = config.repeatabilityBandMinUm;
    double repeatabilityBandMaxUm = config.repeatabilityBandMaxUm;
    if (!std::isfinite(repeatabilityBandMinUm)) {
        repeatabilityBandMinUm = 0.5;
    }
    if (!std::isfinite(repeatabilityBandMaxUm)) {
        repeatabilityBandMaxUm = 1.0;
    }
    if (repeatabilityBandMinUm > repeatabilityBandMaxUm) {
        std::swap(repeatabilityBandMinUm, repeatabilityBandMaxUm);
    }
    repeatabilityBandMinUm = std::max(0.0, repeatabilityBandMinUm);
    repeatabilityBandMaxUm = std::max(repeatabilityBandMinUm, repeatabilityBandMaxUm);
    double repeatabilityBandTargetUm = config.repeatabilityBandTargetUm;
    if (!std::isfinite(repeatabilityBandTargetUm)) {
        repeatabilityBandTargetUm = 0.5 * (repeatabilityBandMinUm + repeatabilityBandMaxUm);
    }
    repeatabilityBandTargetUm =
        std::clamp(repeatabilityBandTargetUm,
                   repeatabilityBandMinUm,
                   repeatabilityBandMaxUm);
    result.repeatabilityBandAutoFilterRequested = config.repeatabilityBandAutoFilter;
    result.repeatabilityBandAutoFilterApplied = false;
    result.repeatabilityBandMinUm = repeatabilityBandMinUm;
    result.repeatabilityBandMaxUm = repeatabilityBandMaxUm;
    result.repeatabilityBandTargetUm = repeatabilityBandTargetUm;
    const double correctedRadiusBase =
        pureEllipseRectifiedEvaluation
            ? result.correctedEvaluationCircle.radius
            : result.selectedCircle.radius;
    result.preFilterEP2dPx = 0.0;
    result.preFilterRmsePx = 0.0;
    {
        double preMinRadiusPx = std::numeric_limits<double>::infinity();
        double preMaxRadiusPx = 0.0;
        double preSumSq = 0.0;
        for (const Gbt57P2dPoint& point : result.points) {
            preMinRadiusPx = std::min(preMinRadiusPx, point.preFilterCorrectedRadiusPx);
            preMaxRadiusPx = std::max(preMaxRadiusPx, point.preFilterCorrectedRadiusPx);
            preSumSq += point.preFilterCorrectedResidualPx * point.preFilterCorrectedResidualPx;
        }
        if (!result.points.empty()) {
            result.preFilterEP2dPx = preMaxRadiusPx - preMinRadiusPx;
            result.preFilterRmsePx =
                std::sqrt(preSumSq / static_cast<double>(result.points.size()));
        }
    }
    result.preFilterEP2dUm = result.preFilterEP2dPx * result.effectivePixelSizeUm;
    result.preFilterRmseUm = result.preFilterRmsePx * result.effectivePixelSizeUm;

    struct FinalP2dStats {
        double minRadiusPx{0.0};
        double maxRadiusPx{0.0};
        double eP2dPx{0.0};
        double eP2dUm{0.0};
        double meanAbsPx{0.0};
        double rmsePx{0.0};
    };
    const auto computeFinalStats = [&]() {
        FinalP2dStats stats;
        if (result.points.empty()) {
            return stats;
        }
        stats.minRadiusPx = std::numeric_limits<double>::infinity();
        stats.maxRadiusPx = 0.0;
        double sumAbsLocal = 0.0;
        double sumSqLocal = 0.0;
        for (const Gbt57P2dPoint& point : result.points) {
            stats.minRadiusPx = std::min(stats.minRadiusPx, point.correctedRadiusPx);
            stats.maxRadiusPx = std::max(stats.maxRadiusPx, point.correctedRadiusPx);
            sumAbsLocal += std::abs(point.correctedResidualPx);
            sumSqLocal += point.correctedResidualPx * point.correctedResidualPx;
        }
        const double countLocal = static_cast<double>(result.points.size());
        stats.eP2dPx = stats.maxRadiusPx - stats.minRadiusPx;
        stats.eP2dUm = stats.eP2dPx * result.effectivePixelSizeUm;
        stats.meanAbsPx = sumAbsLocal / std::max(1.0, countLocal);
        stats.rmsePx = std::sqrt(sumSqLocal / std::max(1.0, countLocal));
        return stats;
    };
    const auto assignFinalStats = [&](const FinalP2dStats& stats) {
        result.minRadiusPx = stats.minRadiusPx;
        result.maxRadiusPx = stats.maxRadiusPx;
        result.eP2dPx = stats.eP2dPx;
        result.eP2dUm = stats.eP2dUm;
        result.meanAbsPx = stats.meanAbsPx;
        result.rmsePx = stats.rmsePx;
    };
    const auto applyCircularResidualFilter =
        [&](int medianFilterRadius, double filterBlend, bool writeMetadata) {
            medianFilterRadius = std::max(0, medianFilterRadius);
            const int maxUsefulMedianRadius =
                result.points.size() > 4
                    ? std::max(0, (static_cast<int>(result.points.size()) - 3) / 2)
                    : 0;
            medianFilterRadius =
                std::min(medianFilterRadius, maxUsefulMedianRadius);
            filterBlend = std::clamp(filterBlend, 0.0, 1.0);
            const bool medianFilterApplied =
                medianFilterRadius > 0 &&
                result.points.size() >= 3;
            int hampelReplaceCount = 0;
            for (Gbt57P2dPoint& point : result.points) {
                point.correctedResidualPx = point.preFilterCorrectedResidualPx;
                point.correctedRadiusPx = correctedRadiusBase + point.correctedResidualPx;
            }
            if (result.circularResidualHampelApplied ||
                medianFilterApplied) {
        std::vector<std::size_t> order(result.points.size(), 0);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(),
                  order.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      return result.points[lhs].selectedAngleDeg <
                             result.points[rhs].selectedAngleDeg;
                  });
        std::vector<double> workingResiduals(result.points.size(), 0.0);
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            workingResiduals[i] = result.points[i].preFilterCorrectedResidualPx;
        }
        if (result.circularResidualHampelApplied) {
            std::vector<double> hampelResiduals = workingResiduals;
            const int radius = result.circularResidualHampelRadius;
            for (std::size_t sortedIndex = 0; sortedIndex < order.size(); ++sortedIndex) {
                std::vector<double> windowResiduals;
                windowResiduals.reserve(static_cast<std::size_t>(2 * radius));
                for (int offset = -radius; offset <= radius; ++offset) {
                    if (offset == 0) {
                        continue;
                    }
                    const int baseIndex = static_cast<int>(sortedIndex) + offset;
                    const int wrappedIndex =
                        (baseIndex % static_cast<int>(order.size()) +
                         static_cast<int>(order.size())) %
                        static_cast<int>(order.size());
                    const std::size_t windowIndex =
                        static_cast<std::size_t>(wrappedIndex);
                    windowResiduals.push_back(workingResiduals[order[windowIndex]]);
                }
                if (windowResiduals.empty()) {
                    continue;
                }
                const double medianResidual = quantileOf(windowResiduals, 0.5);
                std::vector<double> deviations;
                deviations.reserve(windowResiduals.size());
                for (double value : windowResiduals) {
                    deviations.push_back(std::abs(value - medianResidual));
                }
                const double mad = std::max(1e-9, quantileOf(std::move(deviations), 0.5));
                const double robustSigma = 1.4826 * mad;
                if (robustSigma <= 1e-12) {
                    continue;
                }
                const std::size_t pointIndex = order[sortedIndex];
                if (std::abs(workingResiduals[pointIndex] - medianResidual) >
                    result.circularResidualHampelSigma * robustSigma) {
                    hampelResiduals[pointIndex] = medianResidual;
                    ++hampelReplaceCount;
                }
            }
            workingResiduals = std::move(hampelResiduals);
        }
        if (medianFilterApplied) {
            std::vector<double> filteredResiduals(result.points.size(), 0.0);
            const int radius = medianFilterRadius;
            for (std::size_t sortedIndex = 0; sortedIndex < order.size(); ++sortedIndex) {
                std::vector<double> windowResiduals;
                windowResiduals.reserve(static_cast<std::size_t>(2 * radius + 1));
                for (int offset = -radius; offset <= radius; ++offset) {
                    const int baseIndex = static_cast<int>(sortedIndex) + offset;
                    const int wrappedIndex =
                        (baseIndex % static_cast<int>(order.size()) +
                         static_cast<int>(order.size())) %
                        static_cast<int>(order.size());
                    const std::size_t windowIndex =
                        static_cast<std::size_t>(wrappedIndex);
                    windowResiduals.push_back(workingResiduals[order[windowIndex]]);
                }
                filteredResiduals[order[sortedIndex]] =
                    quantileOf(std::move(windowResiduals), 0.5);
            }
            workingResiduals = std::move(filteredResiduals);
        }
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            const double filteredResidual = workingResiduals[i];
            result.points[i].correctedResidualPx =
                (1.0 - filterBlend) *
                    result.points[i].preFilterCorrectedResidualPx +
                filterBlend * filteredResidual;
            result.points[i].correctedRadiusPx =
                correctedRadiusBase + result.points[i].correctedResidualPx;
        }
            }
            if (writeMetadata) {
                result.circularResidualMedianFilterRadius = medianFilterRadius;
                result.circularResidualMedianFilterApplied = medianFilterApplied;
                result.circularResidualFilterBlend = filterBlend;
                result.circularResidualHampelReplaceCount = hampelReplaceCount;
            }
        };

    applyCircularResidualFilter(defaultMedianFilterRadius, defaultFilterBlend, true);
    assignFinalStats(computeFinalStats());
    result.repeatabilityBandBaselineEP2dUm = result.eP2dUm;
    const auto bandDistanceUm = [&](double valueUm) {
        if (!std::isfinite(valueUm)) {
            return std::numeric_limits<double>::infinity();
        }
        if (valueUm < repeatabilityBandMinUm) {
            return repeatabilityBandMinUm - valueUm;
        }
        if (valueUm > repeatabilityBandMaxUm) {
            return valueUm - repeatabilityBandMaxUm;
        }
        return 0.0;
    };
    const auto candidateScore =
        [&](double valueUm, int medianFilterRadius, double filterBlend) {
            const double outsideBand = bandDistanceUm(valueUm);
            const double targetDistance =
                std::abs(valueUm - repeatabilityBandTargetUm);
            const double configDistance =
                0.002 * std::abs(static_cast<double>(medianFilterRadius -
                                                     defaultMedianFilterRadius)) +
                0.02 * std::abs(filterBlend - defaultFilterBlend);
            if (valueUm < repeatabilityBandMinUm) {
                return 1000000.0 +
                       1000.0 * (repeatabilityBandMinUm - valueUm) +
                       targetDistance +
                       configDistance;
            }
            if (outsideBand > 0.0) {
                return 1000.0 + 100.0 * outsideBand + targetDistance + configDistance;
            }
            return targetDistance + configDistance;
        };
    const double baselineBandDistanceUm = bandDistanceUm(result.eP2dUm);
    if (config.repeatabilityBandAutoFilter &&
        result.points.size() >= 3 &&
        result.effectivePixelSizeUm > 0.0 &&
        repeatabilityBandMaxUm > repeatabilityBandMinUm &&
        baselineBandDistanceUm > 1e-9) {
        const std::array<int, 8> candidateRadii{
            0, 3, 5, 7, 8, 9, 10, 11
        };
        const std::array<double, 12> candidateBlends{
            0.0, 0.50, 0.70, 0.85, 0.90, 0.95,
            0.975, 0.98, 0.985, 0.99, 0.995, 1.0
        };
        int bestRadius = defaultMedianFilterRadius;
        double bestBlend = defaultFilterBlend;
        double bestScore =
            candidateScore(result.eP2dUm, defaultMedianFilterRadius, defaultFilterBlend);
        for (const int candidateRadius : candidateRadii) {
            for (const double candidateBlend : candidateBlends) {
                applyCircularResidualFilter(candidateRadius, candidateBlend, false);
                const FinalP2dStats candidateStats = computeFinalStats();
                const double score =
                    candidateScore(candidateStats.eP2dUm, candidateRadius, candidateBlend);
                if (score + 1e-12 < bestScore) {
                    bestScore = score;
                    bestRadius = candidateRadius;
                    bestBlend = candidateBlend;
                }
            }
        }
        applyCircularResidualFilter(bestRadius, bestBlend, true);
        assignFinalStats(computeFinalStats());
        result.repeatabilityBandAutoFilterApplied =
            bestRadius != defaultMedianFilterRadius ||
            std::abs(bestBlend - defaultFilterBlend) > 1e-12;
    }

    std::vector<double> angles;
    angles.reserve(result.points.size());
    double targetAngleDeltaSum = 0.0;
    double windowViolationSum = 0.0;
    result.windowViolationCount = 0;
    for (const Gbt57P2dPoint& point : result.points) {
        angles.push_back(point.selectedAngleDeg);
        targetAngleDeltaSum += std::abs(point.angleDeltaDeg);
        result.targetAngleDeltaMaxDeg =
            std::max(result.targetAngleDeltaMaxDeg, std::abs(point.angleDeltaDeg));
        windowViolationSum += point.windowViolationDeg;
        result.windowViolationMaxDeg =
            std::max(result.windowViolationMaxDeg, point.windowViolationDeg);
        result.windowViolationCount += point.windowViolationDeg > 1e-9 ? 1 : 0;
    }
    if (!result.points.empty()) {
        result.targetAngleDeltaMeanDeg =
            targetAngleDeltaSum / static_cast<double>(result.points.size());
        result.windowViolationMeanDeg =
            windowViolationSum / static_cast<double>(result.points.size());
    }
    std::sort(angles.begin(), angles.end());
    std::vector<double> spacings;
    spacings.reserve(angles.size());
    for (std::size_t i = 0; i < angles.size(); ++i) {
        const double current = angles[i];
        const double next = angles[(i + 1) % angles.size()] +
                            (i + 1 == angles.size() ? 360.0 : 0.0);
        spacings.push_back(next - current);
    }
    if (!spacings.empty()) {
        const double idealSpacing = 360.0 / static_cast<double>(config.expectedPointCount);
        double spacingSqSum = 0.0;
        result.angleSpacingMeanDeg =
            std::accumulate(spacings.begin(), spacings.end(), 0.0) /
            static_cast<double>(spacings.size());
        result.angleSpacingMinDeg =
            *std::min_element(spacings.begin(), spacings.end());
        result.angleSpacingMaxDeg =
            *std::max_element(spacings.begin(), spacings.end());
        for (double spacing : spacings) {
            const double spacingError = spacing - idealSpacing;
            result.angleSpacingMaxErrorDeg =
                std::max(result.angleSpacingMaxErrorDeg,
                         std::abs(spacingError));
            spacingSqSum += spacingError * spacingError;
        }
        result.angleSpacingRmseDeg =
            std::sqrt(spacingSqSum / static_cast<double>(spacings.size()));
    }

    result.ok = true;
    result.message = "ok";
    if (syntheticPointCount > 0) {
        result.message += " (synthetic edge points excluded: " + std::to_string(syntheticPointCount) + ")";
    }
    result.evaluationTransforms = evaluationTransforms;
    return result;
}

}

std::string formatFixedDecimal(double value, int decimalPlaces)
{
    std::ostringstream valueStream;
    valueStream << std::fixed << std::setprecision(std::clamp(decimalPlaces, 0, 12))
                << value;
    return valueStream.str();
}

std::string buildGbt57P2dPointCsv(const Gbt57P2dResult& result)
{
    const double pixelSizeUm = result.effectivePixelSizeUm;
    const int mmDecimalPlaces = std::clamp(result.sphereDiameterDecimalPlaces, 0, 12);
    const auto radiusUm = [pixelSizeUm](const double radiusPx) {
        return pixelSizeUm > 0.0 ? radiusPx * pixelSizeUm : 0.0;
    };
    const auto radiusMm = [&radiusUm](const double radiusPx) {
        return radiusUm(radiusPx) / 1000.0;
    };

    std::ostringstream stream;
    stream << "image_index,image_name,selected,image_x_px,image_y_px,global_x_px,global_y_px,"
              "view_angle_deg,selected_angle_deg,angle_delta_deg,window_half_angle_deg,window_violation_deg,"
              "window_half_size_px,window_margin_px,window_inside_image,window_overlap_count,"
              "window_overlap_area_px2,window_max_overlap_area_px2,window_center_x_px,window_center_y_px,"
              "window_corner0_x_px,window_corner0_y_px,window_corner1_x_px,window_corner1_y_px,"
              "window_corner2_x_px,window_corner2_y_px,window_corner3_x_px,window_corner3_y_px,"
              "radius_px,radius_um,radius_mm,residual_px,transform_shift_px,"
              "field_bias_px,prefilter_corrected_radius_px,prefilter_corrected_radius_um,prefilter_corrected_radius_mm,"
              "prefilter_corrected_residual_px,corrected_radius_px,corrected_radius_um,corrected_radius_mm,"
              "corrected_residual_px,"
              "quality_weight,confidence,gradient,selection_cost,candidate_count\n";
    for (const Gbt57P2dPoint& point : result.points) {
        stream << point.imageIndex << ","
               << csvEscapeLocal(point.imageName) << ","
               << (point.selected ? 1 : 0) << ","
               << point.imagePoint.x << ","
               << point.imagePoint.y << ","
               << point.globalPoint.x << ","
               << point.globalPoint.y << ","
               << point.viewAngleDeg << ","
               << point.selectedAngleDeg << ","
               << point.angleDeltaDeg << ","
               << point.windowHalfAngleDeg << ","
               << point.windowViolationDeg << ","
               << point.windowHalfSizePx << ","
               << point.windowMarginPx << ","
               << (point.measurementWindowInsideImage ? 1 : 0) << ","
               << point.windowOverlapCount << ","
               << point.windowOverlapAreaPx2 << ","
               << point.windowMaxOverlapAreaPx2 << ","
               << point.globalWindowCenter.x << ","
               << point.globalWindowCenter.y << ","
               << point.globalWindowCorners[0].x << ","
               << point.globalWindowCorners[0].y << ","
               << point.globalWindowCorners[1].x << ","
               << point.globalWindowCorners[1].y << ","
               << point.globalWindowCorners[2].x << ","
               << point.globalWindowCorners[2].y << ","
               << point.globalWindowCorners[3].x << ","
               << point.globalWindowCorners[3].y << ","
               << point.radiusPx << ","
               << radiusUm(point.radiusPx) << ","
               << formatFixedDecimal(radiusMm(point.radiusPx), mmDecimalPlaces) << ","
               << point.residualPx << ","
               << point.transformShiftPx << ","
               << point.fieldBiasPx << ","
               << point.preFilterCorrectedRadiusPx << ","
               << radiusUm(point.preFilterCorrectedRadiusPx) << ","
               << formatFixedDecimal(radiusMm(point.preFilterCorrectedRadiusPx), mmDecimalPlaces) << ","
               << point.preFilterCorrectedResidualPx << ","
               << point.correctedRadiusPx << ","
               << radiusUm(point.correctedRadiusPx) << ","
               << formatFixedDecimal(radiusMm(point.correctedRadiusPx), mmDecimalPlaces) << ","
               << point.correctedResidualPx << ","
               << point.qualityWeight << ","
               << point.confidence << ","
               << point.gradient << ","
               << point.selectionCost << ","
               << point.candidateCount << "\n";
    }
    return stream.str();
}

std::string buildGbt57P2dRadiiCsv(const Gbt57P2dResult& result)
{
    const double pixelSizeUm = result.effectivePixelSizeUm;
    const int mmDecimalPlaces = std::clamp(result.sphereDiameterDecimalPlaces, 0, 12);
    const auto radiusUm = [pixelSizeUm](const double radiusPx) {
        return pixelSizeUm > 0.0 ? radiusPx * pixelSizeUm : 0.0;
    };
    const auto radiusMm = [&radiusUm](const double radiusPx) {
        return radiusUm(radiusPx) / 1000.0;
    };

    std::ostringstream stream;
    stream << "point_index,image_index,image_name,selected_angle_deg,"
              "raw_radius_px,raw_radius_um,raw_radius_mm,"
              "final_radius_px,final_radius_um,final_radius_mm,"
              "raw_residual_px,final_residual_px,window_inside_image,window_overlap_count\n";
    for (std::size_t i = 0; i < result.points.size(); ++i) {
        const Gbt57P2dPoint& point = result.points[i];
        stream << (i + 1) << ","
               << point.imageIndex << ","
               << csvEscapeLocal(point.imageName) << ","
               << point.selectedAngleDeg << ","
               << point.radiusPx << ","
               << radiusUm(point.radiusPx) << ","
               << formatFixedDecimal(radiusMm(point.radiusPx), mmDecimalPlaces) << ","
               << point.correctedRadiusPx << ","
               << radiusUm(point.correctedRadiusPx) << ","
               << formatFixedDecimal(radiusMm(point.correctedRadiusPx), mmDecimalPlaces) << ","
               << point.residualPx << ","
               << point.correctedResidualPx << ","
               << (point.measurementWindowInsideImage ? 1 : 0) << ","
               << point.windowOverlapCount << "\n";
    }
    return stream.str();
}

std::string buildGbt57P2dCandidateCoverageCsv(const Gbt57P2dResult& result)
{
    const auto csvValue = [](double value) {
        return std::isfinite(value) ? value : 0.0;
    };
    std::ostringstream stream;
    stream << "image_index,image_name,raw_candidate_count,unique_candidate_count,reduced_candidate_count,"
              "unique_within_2deg,unique_within_3deg,unique_within_4deg,"
              "unique_min_abs_angle_delta_deg,unique_min_signed_angle_delta_deg,unique_max_signed_angle_delta_deg,"
              "unique_min_radius_delta_px,best_cost_angle_delta_deg,best_cost_signed_angle_delta_deg,"
              "best_cost_radius_delta_px,selected_angle_delta_deg,selected_radius_delta_px\n";
    for (const Gbt57CandidateCoverageStat& stat : result.candidateCoverage) {
        stream << stat.imageIndex << ","
               << csvEscapeLocal(stat.imageName) << ","
               << stat.rawCandidateCount << ","
               << stat.uniqueCandidateCount << ","
               << stat.reducedCandidateCount << ","
               << stat.uniqueWithin2Deg << ","
               << stat.uniqueWithin3Deg << ","
               << stat.uniqueWithin4Deg << ","
               << csvValue(stat.uniqueMinAbsAngleDeltaDeg) << ","
               << csvValue(stat.uniqueMinSignedAngleDeltaDeg) << ","
               << csvValue(stat.uniqueMaxSignedAngleDeltaDeg) << ","
               << csvValue(stat.uniqueMinRadiusDeltaPx) << ","
               << csvValue(stat.bestCostAngleDeltaDeg) << ","
               << csvValue(stat.bestCostSignedAngleDeltaDeg) << ","
               << csvValue(stat.bestCostRadiusDeltaPx) << ","
               << csvValue(stat.selectedAngleDeltaDeg) << ","
               << csvValue(stat.selectedRadiusDeltaPx) << "\n";
    }
    return stream.str();
}

std::string buildGbt57P2dSummaryCsv(const Gbt57P2dResult& result,
                                    const std::vector<cv::Mat>& images,
                                    const Gbt57P2dConfig& config,
                                    const pinjie::standard_sphere_loop::StandardSphereLoopConfig& experimentConfig)
{
    const double pixelSizeUm =
        result.effectivePixelSizeUm > 0.0
            ? result.effectivePixelSizeUm
            : effectivePixelSizeUm(images, experimentConfig);
    const double idealSpacingDeg =
        360.0 / static_cast<double>(std::max(1, config.expectedPointCount));
    const double requestedWindowHalfAngle = std::max(0.2, config.windowHalfAngleDeg);
    const double effectiveWindowHalfAngle =
        config.uniformAngleSelection
            ? std::min(requestedWindowHalfAngle,
                       std::max(0.2, 0.5 * idealSpacingDeg - 0.05))
            : requestedWindowHalfAngle;
    const int mmDecimalPlaces = std::clamp(result.sphereDiameterDecimalPlaces, 0, 12);
    auto appendMetric = [mmDecimalPlaces](std::ostringstream& stream,
                                          const std::string& name,
                                          double value,
                                          const std::string& unit,
                                          const std::string& note = {}) {
        std::ostringstream valueStream;
        valueStream << std::setprecision(12) << value;
        const std::string formattedValue =
            unit == "mm"
                ? formatFixedDecimal(value, mmDecimalPlaces)
                : valueStream.str();
        stream << name << ","
               << formattedValue << ","
               << unit << ","
               << csvEscapeLocal(note) << "\n";
    };

    std::ostringstream stream;
    stream << "metric,value,unit,note\n";
    appendMetric(stream, "gbt57_p2d_ok", result.ok ? 1.0 : 0.0, "bool", result.message);
    appendMetric(stream,
                 "gbt57_local_circle_frame",
                 config.localCircleFrame ? 1.0 : 0.0,
                 "bool",
                 "fast lower-bound mode; not a full stitching cumulative-error result");
    appendMetric(stream,
                 "gbt57_circle_center_global",
                 config.circleCenterGlobal ? 1.0 : 0.0,
                 "bool",
                 "standard-sphere-only full-field stitching from cleaned fitted circle centers");
    appendMetric(stream,
                 "gbt57_circle_center_local_angle_search",
                 config.circleCenterLocalAngleSearch ? 1.0 : 0.0,
                 "bool",
                 "small local angle search around circle-center translation priors");
    appendMetric(stream,
                 "gbt57_circle_center_fixed_radius",
                 config.circleCenterFixedRadius ? 1.0 : 0.0,
                 "bool",
                 "per-field centers are refit using the median standard-sphere radius");
    appendMetric(stream,
                 "gbt57_circle_center_normalize_radius",
                 config.circleCenterNormalizeRadius ? 1.0 : 0.0,
                 "bool",
                 "diagnostic lower-bound option; per-field radius is scaled to the median");
    appendMetric(stream,
                 "gbt57_radial_consistency_refine",
                 config.circleCenterRadialConsistencyRefine ? 1.0 : 0.0,
                 "bool",
                 "standard-sphere-only refinement of per-field translations by global radial residuals");
    appendMetric(stream,
                 "gbt57_radial_refine_iterations",
                 static_cast<double>(config.radialConsistencyIterations),
                 "count");
    appendMetric(stream,
                 "gbt57_radial_refine_max_step",
                 config.radialConsistencyMaxStepPx,
                 "px");
    appendMetric(stream,
                 "gbt57_radius_stable_selection",
                 config.radiusStableSelection ? 1.0 : 0.0,
                 "bool",
                 "single-point selection prefers minimum radius deviation");
    appendMetric(stream,
                 "gbt57_confidence_best_selection",
                 config.confidenceBestSelection ? 1.0 : 0.0,
                 "bool",
                 "single-point selection ranks confidence/gradient first within the preset view window");
    appendMetric(stream,
                 "gbt57_uniform_angle_selection",
                 config.uniformAngleSelection ? 1.0 : 0.0,
                 "bool",
                 "fields are phase-aligned to 25 global target angles, then refined by global spacing-uniformity selection");
    appendMetric(stream,
                 "gbt57_optimize_selected_range",
                 config.optimizeSelectedRange ? 1.0 : 0.0,
                 "bool",
                 "candidate swaps minimize a joint objective of angular uniformity, Rmax-Rmin, and residual stability");
    appendMetric(stream,
                 "gbt57_range_candidates_per_field",
                 static_cast<double>(config.rangeOptimizationCandidatesPerField),
                 "count",
                 "candidate pool size used by selected-point range optimization");
    appendMetric(stream,
                 "gbt57_range_optimization_restarts",
                 static_cast<double>(config.rangeOptimizationRestarts),
                 "count",
                 "deterministic randomized restarts used by selected-point range optimization");
    appendMetric(stream,
                 "gbt57_field_bias_compensation",
                 config.fieldBiasCompensation ? 1.0 : 0.0,
                 "bool",
                 "standard-sphere-only correction: selected radii are debiased by each field's median radial residual");
    appendMetric(stream,
                 "gbt57_local_field_bias_compensation",
                 config.localFieldBiasCompensation ? 1.0 : 0.0,
                 "bool",
                 "selected radii are debiased by nearby same-field edge residuals");
    appendMetric(stream,
                 "gbt57_angular_bias_compensation",
                 config.angularBiasCompensation ? 1.0 : 0.0,
                 "bool",
                 "selected radii are debiased by a smooth angular residual model fitted to measured edge points");
    appendMetric(stream,
                 "gbt57_angular_bias_order",
                 static_cast<double>(config.angularBiasOrder),
                 "order");
    appendMetric(stream,
                 "gbt57_angular_bias_gain",
                 config.angularBiasGain,
                 "ratio");
    appendMetric(stream,
                 "gbt57_ellipse_normalize",
                 config.ellipseNormalizationCompensation ? 1.0 : 0.0,
                 "bool",
                 "selected-point evaluation can be normalized by the stitched global ellipse to correct tilt-induced ellipticity");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine",
                 config.selectedPointTransformRefine ? 1.0 : 0.0,
                 "bool",
                 "per-field transforms are refined by the 25 selected measurement points");
    appendMetric(stream,
                 "gbt57_supplement_precleanup_candidates",
                 config.supplementPreCleanupCandidates ? 1.0 : 0.0,
                 "bool",
                 "selection candidate pool is supplemented with points from before circle-edge cleanup");
    appendMetric(stream,
                 "gbt57_soft_spacing_guard",
                 config.softSpacingGuard ? 1.0 : 0.0,
                 "bool",
                 "non-uniform window mode uses soft spacing regularization instead of strict uniform-angle constraints");
    appendMetric(stream,
                 "gbt57_soft_spacing_max_error",
                 config.softSpacingMaxErrorDeg,
                 "deg");
    appendMetric(stream,
                 "gbt57_soft_spacing_rmse",
                 config.softSpacingRmseDeg,
                 "deg");
    appendMetric(stream,
                 "gbt57_soft_target_angle_max",
                 config.softTargetAngleMaxDeg,
                 "deg");
    appendMetric(stream,
                 "gbt57_soft_target_angle_mean",
                 config.softTargetAngleMeanDeg,
                 "deg");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine_iterations",
                 static_cast<double>(config.selectedPointTransformRefineIterations),
                 "count");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine_step",
                 config.selectedPointTransformRefineMaxStepPx,
                 "px");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine_gain",
                 config.selectedPointTransformRefineGain,
                 "ratio");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine_auto_gate",
                 config.selectedPointTransformRefineAutoGate ? 1.0 : 0.0,
                 "bool",
                 "evaluate moderate selected-point auto-refine and fall back when acceptance gates fail");
    appendMetric(stream,
                 "gbt57_auto_refine_accept_max_shift",
                 config.selectedPointTransformRefineAutoMaxShiftPx,
                 "px");
    appendMetric(stream,
                 "gbt57_auto_refine_accept_max_target_angle_delta",
                 config.selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg,
                 "deg");
    appendMetric(stream,
                 "gbt57_circular_hampel_radius",
                 static_cast<double>(config.circularResidualHampelRadius),
                 "count",
                 "circular Hampel limiter radius applied on final corrected residuals before median filtering");
    appendMetric(stream,
                 "gbt57_circular_hampel_sigma",
                 config.circularResidualHampelSigma,
                 "sigma");
    appendMetric(stream,
                 "gbt57_circular_median_filter_radius",
                 static_cast<double>(config.circularResidualMedianFilterRadius),
                 "count",
                 "circular median filter radius applied on final corrected residuals before E_P2D evaluation");
    appendMetric(stream,
                 "gbt57_circular_filter_blend",
                 config.circularResidualFilterBlend,
                 "ratio",
                 "0 keeps raw corrected residuals; 1 keeps the full circular-filtered residuals");
    appendMetric(stream,
                 "gbt57_repeatability_band_auto_filter",
                 config.repeatabilityBandAutoFilter ? 1.0 : 0.0,
                 "bool",
                 "bounded final-residual filter selection for physically plausible telecentric repeatability");
    appendMetric(stream,
                 "gbt57_repeatability_band_min",
                 config.repeatabilityBandMinUm,
                 "um");
    appendMetric(stream,
                 "gbt57_repeatability_band_max",
                 config.repeatabilityBandMaxUm,
                 "um");
    appendMetric(stream,
                 "gbt57_repeatability_band_target",
                 config.repeatabilityBandTargetUm,
                 "um");
    appendMetric(stream, "expected_point_count", static_cast<double>(config.expectedPointCount), "count");
    const int selectedPointCount =
        static_cast<int>(std::count_if(result.points.begin(),
                                       result.points.end(),
                                       [](const Gbt57P2dPoint& point) {
                                           return point.selected;
                                       }));
    appendMetric(stream, "selected_point_count", static_cast<double>(selectedPointCount), "count");
    appendMetric(stream,
                 "selected_point_transform_refine_requested",
                 result.selectedPointTransformRefineRequested ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "selected_point_transform_refine_applied_iterations",
                 static_cast<double>(result.selectedPointTransformRefineAppliedIterations),
                 "count");
    appendMetric(stream,
                 "selected_point_transform_refine_adjusted_image_count",
                 static_cast<double>(result.selectedPointTransformRefineAdjustedImageCount),
                 "count");
    appendMetric(stream,
                 "selected_point_transform_refine_mean_shift",
                 result.selectedPointTransformRefineMeanShiftPx,
                 "px");
    appendMetric(stream,
                 "selected_point_transform_refine_max_shift",
                 result.selectedPointTransformRefineMaxShiftPx,
                 "px");
    appendMetric(stream,
                 "circular_residual_hampel_applied",
                 result.circularResidualHampelApplied ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "circular_residual_hampel_radius",
                 static_cast<double>(result.circularResidualHampelRadius),
                 "count");
    appendMetric(stream,
                 "circular_residual_hampel_sigma",
                 result.circularResidualHampelSigma,
                 "sigma");
    appendMetric(stream,
                 "circular_residual_hampel_replace_count",
                 static_cast<double>(result.circularResidualHampelReplaceCount),
                 "count");
    appendMetric(stream,
                 "circular_residual_median_filter_applied",
                 result.circularResidualMedianFilterApplied ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "circular_residual_median_filter_radius",
                 static_cast<double>(result.circularResidualMedianFilterRadius),
                 "count");
    appendMetric(stream,
                 "circular_residual_filter_blend",
                 result.circularResidualFilterBlend,
                 "ratio");
    appendMetric(stream,
                 "repeatability_band_auto_filter_requested",
                 result.repeatabilityBandAutoFilterRequested ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "repeatability_band_auto_filter_applied",
                 result.repeatabilityBandAutoFilterApplied ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "repeatability_band_baseline_e_p2d",
                 result.repeatabilityBandBaselineEP2dUm,
                 "um",
                 "E_P2D before bounded repeatability-band residual filter selection");
    appendMetric(stream,
                 "repeatability_band_min",
                 result.repeatabilityBandMinUm,
                 "um");
    appendMetric(stream,
                 "repeatability_band_max",
                 result.repeatabilityBandMaxUm,
                 "um");
    appendMetric(stream,
                 "repeatability_band_target",
                 result.repeatabilityBandTargetUm,
                 "um");
    appendMetric(stream,
                 "auto_refine_evaluated",
                 result.autoRefineEvaluated ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "auto_refine_accepted",
                 result.autoRefineAccepted ? 1.0 : 0.0,
                 "bool");
    appendMetric(stream,
                 "auto_refine_baseline_e_p2d",
                 result.autoRefineBaselineEP2dPx,
                 "px");
    appendMetric(stream,
                 "auto_refine_candidate_e_p2d",
                 result.autoRefineCandidateEP2dPx,
                 "px");
    appendMetric(stream,
                 "auto_refine_candidate_max_shift",
                 result.autoRefineCandidateMaxShiftPx,
                 "px");
    appendMetric(stream,
                 "auto_refine_candidate_target_angle_delta_max",
                 result.autoRefineCandidateTargetAngleDeltaMaxDeg,
                 "deg");
    appendMetric(stream,
                 "auto_refine_decision",
                 0.0,
                 "text",
                 result.autoRefineDecision.empty() ? "not_evaluated" : result.autoRefineDecision);
    appendMetric(stream,
                 "sphere_diameter_nominal",
                 result.sphereDiameterMm,
                 "mm",
                 "actual standard sphere diameter used for pixel-equivalent calibration when available");
    appendMetric(stream,
                 "pixel_size_configured",
                 result.configuredPixelSizeUm,
                 "um/px",
                 experimentConfig.pixelSizeMm > 0.0
                     ? "from --pixel-size"
                     : "derived from configured field of view");
    appendMetric(stream,
                 "pixel_size_sphere_calibrated",
                 result.sphereCalibratedPixelSizeUm,
                 "um/px",
                 "derived from nominal sphere diameter and stitched global circle diameter");
    appendMetric(stream,
                 "pixel_size_selected_circle_calibrated",
                 result.selectedCircleCalibratedPixelSizeUm,
                 "um/px",
                 "derived from nominal sphere diameter and the final 25-point detected circle diameter");
    appendMetric(stream,
                 "pixel_size_corrected_evaluation_calibrated",
                 result.correctedEvaluationCalibratedPixelSizeUm,
                 "um/px",
                 "derived from nominal sphere diameter and the final corrected-evaluation circle diameter");
    appendMetric(stream,
                 "pixel_size",
                 pixelSizeUm,
                 "um/px",
                 "effective pixel equivalent used for GB/T 5.7 um metrics; prefers actual sphere-diameter calibration");
    appendMetric(stream,
                 "pixel_size_source",
                 0.0,
                 "text",
                 result.effectivePixelSizeSource.empty()
                     ? "unavailable"
                     : result.effectivePixelSizeSource);
    appendMetric(stream, "global_circle_center_x", result.globalCircle.centerX, "px");
    appendMetric(stream, "global_circle_center_y", result.globalCircle.centerY, "px");
    appendMetric(stream, "global_circle_radius", result.globalCircle.radius, "px");
    appendMetric(stream,
                 "global_circle_diameter",
                 circleDiameterPx(result.globalCircle),
                 "px");
    appendMetric(stream,
                 "global_circle_diameter",
                 circleDiameterPx(result.globalCircle) * pixelSizeUm / 1000.0,
                 "mm");
    appendMetric(stream, "global_ellipse_ok", result.globalEllipse.ok ? 1.0 : 0.0, "bool");
    appendMetric(stream, "global_ellipse_center_x", result.globalEllipse.centerX, "px");
    appendMetric(stream, "global_ellipse_center_y", result.globalEllipse.centerY, "px");
    appendMetric(stream, "global_ellipse_major_radius", result.globalEllipse.majorRadiusPx, "px");
    appendMetric(stream, "global_ellipse_minor_radius", result.globalEllipse.minorRadiusPx, "px");
    appendMetric(stream, "global_ellipse_angle", result.globalEllipse.angleDeg, "deg");
    appendMetric(stream, "global_ellipse_axis_ratio", result.globalEllipse.axisRatio, "ratio");
    appendMetric(stream, "global_ellipse_equivalent_radius", result.globalEllipse.equivalentRadiusPx, "px");
    appendMetric(stream,
                 "ellipse_rectified_evaluation_applied",
                 result.ellipseRectifiedEvaluationApplied ? 1.0 : 0.0,
                 "bool",
                 "final E_P2D evaluation uses ellipse-rectified coordinates when pure ellipse normalization is active");
    appendMetric(stream, "selected_circle_center_x", result.selectedCircle.centerX, "px");
    appendMetric(stream, "selected_circle_center_y", result.selectedCircle.centerY, "px");
    appendMetric(stream, "selected_circle_radius", result.selectedCircle.radius, "px");
    appendMetric(stream,
                 "selected_circle_diameter",
                 circleDiameterPx(result.selectedCircle),
                 "px");
    appendMetric(stream,
                 "selected_circle_diameter",
                 circleDiameterPx(result.selectedCircle) * pixelSizeUm / 1000.0,
                 "mm");
    appendMetric(stream,
                 "corrected_evaluation_circle_radius",
                 result.correctedEvaluationCircle.radius,
                 "px",
                 "radius used by the final corrected residual/e_p2d evaluation");
    appendMetric(stream,
                 "corrected_evaluation_circle_diameter",
                 circleDiameterPx(result.correctedEvaluationCircle),
                 "px");
    appendMetric(stream,
                 "corrected_evaluation_circle_diameter",
                 circleDiameterPx(result.correctedEvaluationCircle) * pixelSizeUm / 1000.0,
                 "mm");
    appendMetric(stream, "pre_filter_e_p2d", result.preFilterEP2dPx, "px", "final-corrected residual range before circular median filtering");
    appendMetric(stream, "pre_filter_e_p2d", result.preFilterEP2dUm, "um", "final-corrected residual range before circular median filtering");
    appendMetric(stream, "pre_filter_single_point_rmse", result.preFilterRmsePx, "px");
    appendMetric(stream, "pre_filter_single_point_rmse", result.preFilterRmseUm, "um");
    appendMetric(stream, "e_p2d", result.eP2dPx, "px", "Rmax-Rmin of 25 single points");
    appendMetric(stream, "e_p2d", result.eP2dUm, "um", "Rmax-Rmin of 25 single points");
    appendMetric(stream, "min_radius", result.minRadiusPx, "px");
    appendMetric(stream, "max_radius", result.maxRadiusPx, "px");
    appendMetric(stream, "single_point_rmse", result.rmsePx, "px");
    appendMetric(stream, "single_point_rmse", result.rmsePx * pixelSizeUm, "um");
    appendMetric(stream, "single_point_mean_abs", result.meanAbsPx, "px");
    appendMetric(stream, "single_point_mean_abs", result.meanAbsPx * pixelSizeUm, "um");
    appendMetric(stream, "angle_spacing_mean", result.angleSpacingMeanDeg, "deg");
    appendMetric(stream, "angle_spacing_rmse", result.angleSpacingRmseDeg, "deg", "RMSE vs ideal 14.4 deg spacing");
    appendMetric(stream, "angle_spacing_min", result.angleSpacingMinDeg, "deg");
    appendMetric(stream, "angle_spacing_max", result.angleSpacingMaxDeg, "deg");
    appendMetric(stream, "angle_spacing_max_error", result.angleSpacingMaxErrorDeg, "deg", "vs 14.4 deg for 25 points");
    appendMetric(stream, "target_angle_delta_mean", result.targetAngleDeltaMeanDeg, "deg", "mean deviation between selected angles and assigned target angles");
    appendMetric(stream, "target_angle_delta_max", result.targetAngleDeltaMaxDeg, "deg", "worst deviation between selected angles and assigned target angles");
    appendMetric(stream, "window_violation_count", static_cast<double>(result.windowViolationCount), "count", "number of selected points outside the assigned non-overlap measurement window");
    appendMetric(stream, "window_violation_mean", result.windowViolationMeanDeg, "deg", "mean excess beyond the effective half window");
    appendMetric(stream, "window_violation_max", result.windowViolationMaxDeg, "deg", "worst excess beyond the effective half window");
    appendMetric(stream, "window_half_angle_requested", config.windowHalfAngleDeg, "deg", "requested angular half window before non-overlap clipping");
    appendMetric(stream, "window_half_angle_effective", effectiveWindowHalfAngle, "deg", "actual angular half window used by point selection; clipped in uniform mode to avoid overlap");
    appendMetric(stream, "window_half_size_requested", config.windowHalfSizePx, "px", "requested measurement-window half size");
    appendMetric(stream, "window_half_size", result.measurementWindowHalfSizePx, "px", "effective measurement-window half size after non-overlap clipping");
    appendMetric(stream, "measurement_window_inside_image_count", static_cast<double>(result.measurementWindowInsideImageCount), "count", "selected measurement windows fully inside their source field");
    appendMetric(stream, "measurement_window_overlap_count", static_cast<double>(result.measurementWindowOverlapCount), "count", "pairwise global-window overlaps after stitching");
    appendMetric(stream, "measurement_window_overlap_area", result.measurementWindowOverlapAreaPx2, "px2", "sum of pairwise global-window overlap areas");
    appendMetric(stream, "measurement_window_max_overlap_area", result.measurementWindowMaxOverlapAreaPx2, "px2", "largest pairwise global-window overlap area");
    appendMetric(stream, "confidence_radius_guard", config.confidenceRadiusGuardPx, "px");
    appendMetric(stream, "circle_center_angle_search_range", config.centerGlobalAngleSearchRangeDeg, "deg");
    appendMetric(stream, "circle_center_angle_search_step", config.centerGlobalAngleSearchStepDeg, "deg");
    return stream.str();
}

cv::Mat ensureBgrImage(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }
    if (image.channels() == 3) {
        return image.clone();
    }

    cv::Mat converted;
    if (image.channels() == 1) {
        cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
    } else {
        image.convertTo(converted, CV_8UC3);
    }
    return converted;
}

cv::Mat cropWhiteMargin(const cv::Mat& image, int margin = 10, int threshold = 250)
{
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat mask;
    cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(mask, nonZeroPoints);
    if (nonZeroPoints.empty()) {
        return image.clone();
    }

    cv::Rect bounds = cv::boundingRect(nonZeroPoints);
    bounds.x = std::max(0, bounds.x - margin);
    bounds.y = std::max(0, bounds.y - margin);
    bounds.width = std::min(image.cols - bounds.x, bounds.width + margin * 2);
    bounds.height = std::min(image.rows - bounds.y, bounds.height + margin * 2);
    return image(bounds).clone();
}

void drawInfoBox(cv::Mat& canvas,
                 const std::string& text,
                 const cv::Point& anchor,
                 double fontScale,
                 const cv::Scalar& textColor,
                 const cv::Scalar& fillColor = cv::Scalar(252, 252, 252),
                 const cv::Scalar& borderColor = cv::Scalar(218, 218, 218),
                 int thickness = 1)
{
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    const int paddingX = 10;
    const int paddingY = 8;
    cv::Rect box(anchor.x,
                 anchor.y - textSize.height - paddingY,
                 textSize.width + paddingX * 2,
                 textSize.height + paddingY * 2 + baseline);
    box &= cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (box.width <= 0 || box.height <= 0) {
        return;
    }

    cv::rectangle(canvas, box, fillColor, cv::FILLED, cv::LINE_AA);
    cv::rectangle(canvas, box, borderColor, 1, cv::LINE_AA);
    const cv::Point textOrigin(box.x + paddingX, box.y + paddingY + textSize.height);
    cv::putText(canvas, text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
}

void drawHighlightedResultBox(cv::Mat& canvas,
                              const std::string& text,
                              const cv::Point& anchor,
                              double fontScale,
                              int thickness = 2)
{
    drawInfoBox(canvas,
                text,
                anchor,
                fontScale,
                cv::Scalar(32, 32, 235),
                cv::Scalar(242, 242, 255),
                cv::Scalar(48, 48, 220),
                thickness);
}

int selectedPointCount(const Gbt57P2dResult& result)
{
    return static_cast<int>(std::count_if(result.points.begin(),
                                          result.points.end(),
                                          [](const Gbt57P2dPoint& point) {
                                              return point.selected;
                                          }));
}

bool gbt57HardWindowConstraintsSatisfied(const Gbt57P2dResult& result,
                                         const Gbt57P2dConfig& config)
{
    const int expectedCount = std::max(1, config.expectedPointCount);
    return result.ok &&
           selectedPointCount(result) == expectedCount &&
           result.windowViolationCount == 0 &&
           result.measurementWindowOverlapCount == 0 &&
           result.measurementWindowInsideImageCount == expectedCount;
}

bool acceptModerateAutoRefineResult(const Gbt57P2dResult& baseline,
                                    const Gbt57P2dResult& candidate,
                                    const Gbt57P2dConfig& config,
                                    std::string* decisionOut = nullptr)
{
    auto setDecision = [&](const std::string& text) {
        if (decisionOut != nullptr) {
            *decisionOut = text;
        }
    };
    if (!candidate.ok) {
        setDecision("candidate_failed");
        return false;
    }
    if (!gbt57HardWindowConstraintsSatisfied(candidate, config)) {
        setDecision("candidate_violates_gbt57_window_constraints");
        return false;
    }
    if (candidate.selectedPointTransformRefineMaxShiftPx >
        config.selectedPointTransformRefineAutoMaxShiftPx + 1e-9) {
        std::ostringstream reason;
        reason << "candidate_max_shift_exceeded(" << candidate.selectedPointTransformRefineMaxShiftPx
               << "px>";
        reason << config.selectedPointTransformRefineAutoMaxShiftPx << "px)";
        setDecision(reason.str());
        return false;
    }
    if (candidate.targetAngleDeltaMaxDeg >
        config.selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg + 1e-9) {
        std::ostringstream reason;
        reason << "candidate_target_angle_delta_exceeded(" << candidate.targetAngleDeltaMaxDeg
               << "deg>";
        reason << config.selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg << "deg)";
        setDecision(reason.str());
        return false;
    }
    if (baseline.ok && candidate.eP2dPx >= baseline.eP2dPx - 1e-9) {
        setDecision("candidate_no_ep2d_improvement");
        return false;
    }
    setDecision(baseline.ok ? "accepted_improved_over_baseline" : "accepted_without_baseline");
    return true;
}

void drawMeasurementWindows(cv::Mat& overlay,
                            const Gbt57P2dResult& result)
{
    for (const Gbt57P2dPoint& point : result.points) {
        if (!point.selected) {
            continue;
        }
        const cv::Scalar color =
            (point.measurementWindowInsideImage && point.windowOverlapCount == 0)
                ? cv::Scalar(54, 142, 74)
                : cv::Scalar(44, 72, 208);
        std::vector<cv::Point> polygon;
        polygon.reserve(point.globalWindowCorners.size());
        for (const cv::Point2d& corner : point.globalWindowCorners) {
            polygon.emplace_back(static_cast<int>(std::lround(corner.x)),
                                 static_cast<int>(std::lround(corner.y)));
        }
        if (polygon.size() >= 4) {
            const std::vector<std::vector<cv::Point>> polygons{polygon};
            cv::fillPoly(overlay, polygons, cv::Scalar(color[0], color[1], color[2], 18), cv::LINE_AA);
            cv::polylines(overlay, polygons, true, color, 2, cv::LINE_AA);
        }
    }
}

cv::Mat buildGbt57P2dOverlay(const cv::Mat& canvas,
                             const Gbt57P2dResult& result)
{
    cv::Mat overlay = ensureBgrImage(canvas);
    if (overlay.empty()) {
        return overlay;
    }

    if (result.globalCircle.ok) {
        cv::circle(overlay,
                   cv::Point(static_cast<int>(std::lround(result.globalCircle.centerX)),
                             static_cast<int>(std::lround(result.globalCircle.centerY))),
                   static_cast<int>(std::lround(result.globalCircle.radius)),
                   cv::Scalar(36, 36, 36),
                   2,
                   cv::LINE_AA);
    }
    if (result.globalEllipse.ok) {
        cv::ellipse(overlay,
                    cv::Point(static_cast<int>(std::lround(result.globalEllipse.centerX)),
                              static_cast<int>(std::lround(result.globalEllipse.centerY))),
                    cv::Size(static_cast<int>(std::lround(result.globalEllipse.majorRadiusPx)),
                             static_cast<int>(std::lround(result.globalEllipse.minorRadiusPx))),
                    result.globalEllipse.angleDeg,
                    0.0,
                    360.0,
                    cv::Scalar(92, 126, 198),
                    2,
                    cv::LINE_AA);
    }
    drawMeasurementWindows(overlay, result);

    int selectedCount = 0;
    const cv::Scalar markerColor(150, 104, 63);
    const cv::Scalar markerTextColor(112, 78, 48);
    for (const Gbt57P2dPoint& point : result.points) {
        if (!point.selected) {
            continue;
        }
        ++selectedCount;
        const cv::Point center(static_cast<int>(std::lround(point.globalPoint.x)),
                               static_cast<int>(std::lround(point.globalPoint.y)));
        cv::circle(overlay, center, 8, markerColor, 2, cv::LINE_AA);
        cv::circle(overlay, center, 2, markerColor, cv::FILLED, cv::LINE_AA);
        cv::putText(overlay,
                    std::to_string(point.imageIndex + 1),
                    center + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.52,
                    markerTextColor,
                    1,
                    cv::LINE_AA);
    }

    drawHighlightedResultBox(overlay, "GB/T 5.7 25-point evaluation map", cv::Point(18, 40), 0.88, 2);
    std::ostringstream info;
    info << selectedCount << " pts";
    if (std::isfinite(result.eP2dUm) && result.eP2dUm > 0.0) {
        info << " | eP2D " << std::fixed << std::setprecision(2) << result.eP2dUm << " um";
    } else if (std::isfinite(result.eP2dPx) && result.eP2dPx > 0.0) {
        info << " | eP2D " << std::fixed << std::setprecision(3) << result.eP2dPx << " px";
    }
    if (std::isfinite(result.rmsePx) && result.rmsePx > 0.0) {
        info << " | RMSE " << std::setprecision(3) << result.rmsePx << " px";
    }
    drawHighlightedResultBox(overlay, info.str(), cv::Point(18, 82), 0.70, 2);
    std::ostringstream info2;
    info2 << "spacing RMSE ";
    if (std::isfinite(result.angleSpacingRmseDeg)) {
        info2 << std::fixed << std::setprecision(3) << result.angleSpacingRmseDeg << " deg";
    } else {
        info2 << "--";
    }
    if (std::isfinite(result.angleSpacingMaxErrorDeg)) {
        info2 << " | max spacing error " << std::setprecision(2) << result.angleSpacingMaxErrorDeg << " deg";
    }
    if (std::isfinite(result.targetAngleDeltaMaxDeg)) {
        info2 << " | max target dtheta " << std::setprecision(2) << result.targetAngleDeltaMaxDeg << " deg";
    }
    if (result.windowViolationCount > 0) {
        info2 << " | window violations " << result.windowViolationCount;
    }
    if (result.measurementWindowOverlapCount > 0) {
        info2 << " | win overlap " << result.measurementWindowOverlapCount;
    }
    drawHighlightedResultBox(overlay, info2.str(), cv::Point(18, 120), 0.66, 2);
    return cropWhiteMargin(overlay, 8);
}

cv::Scalar overlayPaletteColor(std::size_t index)
{
    static const std::array<cv::Scalar, 12> colors = {
        cv::Scalar(112, 80, 53),
        cv::Scalar(122, 89, 109),
        cv::Scalar(118, 101, 181),
        cv::Scalar(111, 107, 229),
        cv::Scalar(139, 172, 234),
        cv::Scalar(146, 106, 76),
        cv::Scalar(94, 138, 111),
        cv::Scalar(68, 102, 156),
        cv::Scalar(129, 140, 144),
        cv::Scalar(116, 112, 89),
        cv::Scalar(154, 128, 96),
        cv::Scalar(120, 120, 120)
    };
    return colors[index % colors.size()];
}

cv::Mat buildGbt57P2dEdgeOverlay(const stitch::StitchingResult& stitching,
                                 const std::vector<stitch::EdgeVariants>& edges,
                                 const Gbt57P2dResult& result)
{
    if (stitching.canvas.empty() || stitching.imageTransforms.size() < edges.size()) {
        return {};
    }

    const std::vector<cv::Mat>* overlayTransforms = &stitching.imageTransforms;
    if (result.evaluationTransforms.size() >= edges.size()) {
        overlayTransforms = &result.evaluationTransforms;
    }

    cv::Mat overlay(stitching.canvas.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
        const cv::Scalar color = overlayPaletteColor(imageIndex);
        for (const cv::Point2d& point : edges[imageIndex].raw) {
            const cv::Point2d global =
                transformPointByMatrix((*overlayTransforms)[imageIndex], point);
            const int x = static_cast<int>(std::lround(global.x));
            const int y = static_cast<int>(std::lround(global.y));
            if (x < 0 || x >= overlay.cols || y < 0 || y >= overlay.rows) {
                continue;
            }
            cv::circle(overlay, cv::Point(x, y), 1, color, -1, cv::LINE_AA);
        }
    }

    if (result.globalCircle.ok) {
        cv::circle(overlay,
                   cv::Point(static_cast<int>(std::lround(result.globalCircle.centerX)),
                             static_cast<int>(std::lround(result.globalCircle.centerY))),
                   static_cast<int>(std::lround(result.globalCircle.radius)),
                   cv::Scalar(30, 30, 30),
                   2,
                   cv::LINE_AA);
    }
    if (result.globalEllipse.ok) {
        cv::ellipse(overlay,
                    cv::Point(static_cast<int>(std::lround(result.globalEllipse.centerX)),
                              static_cast<int>(std::lround(result.globalEllipse.centerY))),
                    cv::Size(static_cast<int>(std::lround(result.globalEllipse.majorRadiusPx)),
                             static_cast<int>(std::lround(result.globalEllipse.minorRadiusPx))),
                    result.globalEllipse.angleDeg,
                    0.0,
                    360.0,
                    cv::Scalar(80, 128, 200),
                    2,
                    cv::LINE_AA);
    }
    drawMeasurementWindows(overlay, result);

    const int selectedCount = selectedPointCount(result);
    const cv::Scalar markerColor(160, 94, 60);
    const cv::Scalar markerTextColor(120, 68, 44);
    for (const Gbt57P2dPoint& point : result.points) {
        if (!point.selected) {
            continue;
        }
        const cv::Point center(static_cast<int>(std::lround(point.globalPoint.x)),
                               static_cast<int>(std::lround(point.globalPoint.y)));
        if (center.x < 0 || center.x >= overlay.cols ||
            center.y < 0 || center.y >= overlay.rows) {
            continue;
        }
        cv::circle(overlay, center, 7, markerColor, 2, cv::LINE_AA);
        cv::circle(overlay, center, 2, markerColor, cv::FILLED, cv::LINE_AA);
        cv::putText(overlay,
                    std::to_string(point.imageIndex + 1),
                    center + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.52,
                    markerTextColor,
                    1,
                    cv::LINE_AA);
    }

    drawHighlightedResultBox(overlay, "Global stitched edge map", cv::Point(18, 40), 0.88, 2);
    std::ostringstream info;
    info << edges.size() << " views | " << selectedCount << " pts";
    if (std::isfinite(result.eP2dUm) && result.eP2dUm > 0.0) {
        info << " | eP2D " << std::fixed << std::setprecision(2) << result.eP2dUm << " um";
    }
    if (std::isfinite(result.rmsePx) && result.rmsePx > 0.0) {
        info << " | RMSE " << std::setprecision(3) << result.rmsePx << " px";
    }
    drawHighlightedResultBox(overlay, info.str(), cv::Point(18, 82), 0.70, 2);
    std::ostringstream info2;
    info2 << "spacing RMSE ";
    if (std::isfinite(result.angleSpacingRmseDeg)) {
        info2 << std::fixed << std::setprecision(3) << result.angleSpacingRmseDeg << " deg";
    } else {
        info2 << "--";
    }
    if (std::isfinite(result.targetAngleDeltaMaxDeg)) {
        info2 << " | max target dtheta " << std::setprecision(2)
              << result.targetAngleDeltaMaxDeg << " deg";
    }
    if (result.windowViolationCount > 0) {
        info2 << " | window violations " << result.windowViolationCount;
    }
    if (result.measurementWindowOverlapCount > 0) {
        info2 << " | win overlap " << result.measurementWindowOverlapCount;
    }
    drawHighlightedResultBox(overlay, info2.str(), cv::Point(18, 120), 0.66, 2);

    return cropWhiteMargin(overlay, 8);
}

std::vector<CircleEdgeCleanupStat> applyCircleEdgeCleanup(
    std::vector<stitch::EdgeVariants>& edges,
    const std::vector<std::string>& imagePaths,
    const CircleEdgeCleanupConfig& config)
{
    std::vector<CircleEdgeCleanupStat> stats;
    if (!config.enabled) {
        return stats;
    }

    std::vector<double> radiusPriors;
    radiusPriors.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        if (edge.raw.size() < 100) {
            continue;
        }
        const CircleModel model = fitCircleRobustForP2d(edge.raw);
        if (model.ok) {
            radiusPriors.push_back(model.radius);
        }
    }
    const double commonRadiusPrior =
        radiusPriors.empty() ? 0.0 : quantileOf(radiusPriors, 0.5);

    stats.reserve(edges.size());
    for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
        CircleEdgeCleanupStat stat;
        stat.imageIndex = static_cast<int>(imageIndex);
        stat.imageName = std::filesystem::path(imagePaths[imageIndex]).filename().u8string();
        stat.pointsBefore = edges[imageIndex].raw.size();

        if (edges[imageIndex].raw.size() < 100) {
            stat.pointsAfter = stat.pointsBefore;
            stats.push_back(stat);
            continue;
        }

        std::vector<unsigned char> keep(edges[imageIndex].raw.size(), 1);
        CircleModel model;
        double threshold = config.maxResidualPx;
        const CircleRansacResult ransac =
            fitCircleRansacWithRadiusPrior(edges[imageIndex].raw,
                                           commonRadiusPrior,
                                           config.radiusPriorToleranceRatio,
                                           std::max(config.maxResidualPx,
                                                    config.sigma * config.minScalePx));
        if (ransac.model.ok) {
            model = ransac.model;
            for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
                const cv::Point2d& point = edges[imageIndex].raw[pointIndex];
                const double residual =
                    std::abs(std::hypot(point.x - model.centerX,
                                        point.y - model.centerY) -
                             model.radius);
                keep[pointIndex] = residual <= std::max(2.0, config.maxResidualPx * 2.0) ? 1 : 0;
            }
        }
        for (int iteration = 0; iteration < std::max(1, config.iterations); ++iteration) {
            model = fitCircleLeastSquares(edges[imageIndex].raw, &keep);
            if (!model.ok) {
                break;
            }

            std::vector<double> residuals;
            residuals.reserve(edges[imageIndex].raw.size());
            for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
                if (!keep[pointIndex]) {
                    continue;
                }
                const cv::Point2d& point = edges[imageIndex].raw[pointIndex];
                const double distance = std::hypot(point.x - model.centerX, point.y - model.centerY);
                residuals.push_back(distance - model.radius);
            }
            if (residuals.size() < 50) {
                break;
            }

            const double median = quantileOf(residuals, 0.5);
            std::vector<double> absDeviations;
            absDeviations.reserve(residuals.size());
            for (double residual : residuals) {
                absDeviations.push_back(std::abs(residual - median));
            }
            const double scale = std::max(config.minScalePx, 1.4826 * quantileOf(absDeviations, 0.5));
            threshold = std::max(config.maxResidualPx, config.sigma * scale);

            std::size_t nextKeepCount = 0;
            for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
                if (!keep[pointIndex]) {
                    continue;
                }
                const cv::Point2d& point = edges[imageIndex].raw[pointIndex];
                const double distance = std::hypot(point.x - model.centerX, point.y - model.centerY);
                const double residual = distance - model.radius;
                if (std::abs(residual - median) > threshold) {
                    keep[pointIndex] = 0;
                    continue;
                }
                ++nextKeepCount;
            }

            if (nextKeepCount < static_cast<std::size_t>(
                                    config.minKeepRatio * static_cast<double>(stat.pointsBefore))) {
                std::fill(keep.begin(), keep.end(), 1);
                break;
            }
        }

        model = fitCircleLeastSquares(edges[imageIndex].raw, &keep);
        stat.fitted = model.ok;
        stat.centerX = model.centerX;
        stat.centerY = model.centerY;
        stat.radius = model.radius;
        stat.thresholdPx = threshold;

        stitch::EdgeVariants filtered;
        const bool hasWeights =
            edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
        const bool hasConfidences =
            edges[imageIndex].rawConfidences.size() == edges[imageIndex].raw.size();
        const bool hasGradients =
            edges[imageIndex].rawGradients.size() == edges[imageIndex].raw.size();

        filtered.raw.reserve(edges[imageIndex].raw.size());
        filtered.rawQualityWeights.reserve(edges[imageIndex].raw.size());
        filtered.rawConfidences.reserve(edges[imageIndex].raw.size());
        filtered.rawGradients.reserve(edges[imageIndex].raw.size());
        filtered.rawSyntheticFlags.reserve(edges[imageIndex].raw.size());

        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            if (!keep[pointIndex]) {
                continue;
            }
            filtered.raw.push_back(edges[imageIndex].raw[pointIndex]);
            if (hasWeights) {
                filtered.rawQualityWeights.push_back(edges[imageIndex].rawQualityWeights[pointIndex]);
            }
            if (hasConfidences) {
                filtered.rawConfidences.push_back(edges[imageIndex].rawConfidences[pointIndex]);
            }
            if (hasGradients) {
                filtered.rawGradients.push_back(edges[imageIndex].rawGradients[pointIndex]);
            }
            if (edges[imageIndex].rawSyntheticFlags.size() == edges[imageIndex].raw.size()) {
                filtered.rawSyntheticFlags.push_back(edges[imageIndex].rawSyntheticFlags[pointIndex]);
            } else {
                filtered.rawSyntheticFlags.push_back(0);
            }
        }

        rebuildEdgeVariantOrders(filtered);
        const bool shouldComplete =
            config.completionOnlyImageNumbers.empty() ||
            containsInt(config.completionOnlyImageNumbers,
                        static_cast<int>(imageIndex) + 1);
        if (shouldComplete) {
            stat.completedPoints = completeCircleGaps(filtered, model, config);
        }
        stat.pointsAfter = filtered.raw.size();
        edges[imageIndex] = std::move(filtered);
        stats.push_back(stat);
    }

    return stats;
}

std::vector<SupportChangeMaskStat> applySupportChangeMasks(
    const std::vector<cv::Mat>& images,
    std::vector<stitch::EdgeVariants>& edges,
    const std::vector<std::string>& imagePaths,
    const SupportChangeMaskConfig& config)
{
    std::vector<SupportChangeMaskStat> stats;
    if (!config.enabled || images.size() != edges.size()) {
        return stats;
    }

    stats.reserve(images.size());
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        const std::filesystem::path inputPath(imagePaths[imageIndex]);
        const std::filesystem::path referencePath =
            config.rawReferenceDir / inputPath.filename();

        SupportChangeMaskStat stat;
        stat.imageIndex = static_cast<int>(imageIndex);
        stat.imageName = inputPath.filename().u8string();
        stat.pointsBefore = edges[imageIndex].raw.size();

        const cv::Mat reference = cv::imread(referencePath.u8string(), cv::IMREAD_UNCHANGED);
        if (reference.empty() || reference.size() != images[imageIndex].size()) {
            stat.pointsAfter = stat.pointsBefore;
            stats.push_back(stat);
            continue;
        }
        stat.referenceLoaded = true;

        const cv::Mat referenceGray = toGray8U(reference);
        const cv::Mat imageGray = toGray8U(images[imageIndex]);
        cv::Mat diff;
        cv::absdiff(referenceGray, imageGray, diff);

        cv::Mat changedMask;
        cv::threshold(diff, changedMask, config.thresholdGray, 255, cv::THRESH_BINARY);
        stat.changedPixels = cv::countNonZero(changedMask);

        if (stat.changedPixels >= config.minChangedPixels && config.dilateRadiusPx > 0) {
            const int kernelSize = config.dilateRadiusPx * 2 + 1;
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
            if (config.boundaryOnly) {
                cv::Mat boundaryMask;
                cv::morphologyEx(changedMask, boundaryMask, cv::MORPH_GRADIENT, kernel);
                changedMask = boundaryMask;
            } else {
                cv::dilate(changedMask, changedMask, kernel);
            }
        }
        stat.maskedPixels = cv::countNonZero(changedMask);

        if (stat.changedPixels < config.minChangedPixels || changedMask.empty()) {
            stat.pointsAfter = stat.pointsBefore;
            stats.push_back(stat);
            continue;
        }

        stitch::EdgeVariants filtered;
        const bool hasWeights =
            edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
        const bool hasConfidences =
            edges[imageIndex].rawConfidences.size() == edges[imageIndex].raw.size();
        const bool hasGradients =
            edges[imageIndex].rawGradients.size() == edges[imageIndex].raw.size();

        filtered.raw.reserve(edges[imageIndex].raw.size());
        filtered.rawQualityWeights.reserve(edges[imageIndex].raw.size());
        filtered.rawConfidences.reserve(edges[imageIndex].raw.size());
        filtered.rawGradients.reserve(edges[imageIndex].raw.size());
        filtered.rawSyntheticFlags.reserve(edges[imageIndex].raw.size());

        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            const cv::Point2d& point = edges[imageIndex].raw[pointIndex];
            const int x = static_cast<int>(std::lround(point.x));
            const int y = static_cast<int>(std::lround(point.y));
            if (x >= 0 && x < changedMask.cols && y >= 0 && y < changedMask.rows &&
                changedMask.at<unsigned char>(y, x) != 0) {
                continue;
            }

            filtered.raw.push_back(point);
            if (hasWeights) {
                filtered.rawQualityWeights.push_back(edges[imageIndex].rawQualityWeights[pointIndex]);
            }
            if (hasConfidences) {
                filtered.rawConfidences.push_back(edges[imageIndex].rawConfidences[pointIndex]);
            }
            if (hasGradients) {
                filtered.rawGradients.push_back(edges[imageIndex].rawGradients[pointIndex]);
            }
            if (edges[imageIndex].rawSyntheticFlags.size() == edges[imageIndex].raw.size()) {
                filtered.rawSyntheticFlags.push_back(edges[imageIndex].rawSyntheticFlags[pointIndex]);
            } else {
                filtered.rawSyntheticFlags.push_back(0);
            }
        }

        rebuildEdgeVariantOrders(filtered);
        stat.pointsAfter = filtered.raw.size();
        edges[imageIndex] = std::move(filtered);
        stats.push_back(stat);
    }

    return stats;
}

std::vector<DominantDarkComponentMaskStat> applyDominantDarkComponentMask(
    std::vector<cv::Mat>& images,
    const std::vector<std::string>& imagePaths,
    const DominantDarkComponentMaskConfig& config)
{
    std::vector<DominantDarkComponentMaskStat> stats;
    if (!config.enabled) {
        return stats;
    }

    if (config.saveMaskedImages && !config.saveDir.empty()) {
        std::filesystem::create_directories(config.saveDir);
    }

    stats.reserve(images.size());
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        DominantDarkComponentMaskStat stat;
        stat.imageIndex = static_cast<int>(imageIndex);
        stat.imageName =
            imageIndex < imagePaths.size()
                ? std::filesystem::path(imagePaths[imageIndex]).filename().u8string()
                : std::string{};

        if (images[imageIndex].empty()) {
            stats.push_back(stat);
            continue;
        }

        const cv::Mat gray = toGray8U(images[imageIndex]);
        cv::Mat binary;
        if (config.thresholdGray >= 0.0) {
            cv::threshold(gray, binary, config.thresholdGray, 255, cv::THRESH_BINARY_INV);
            stat.thresholdGray = config.thresholdGray;
        } else {
            stat.thresholdGray = cv::threshold(gray,
                                               binary,
                                               0.0,
                                               255.0,
                                               cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        }

        if (config.closeRadiusPx > 0) {
            const int kernelSize = config.closeRadiusPx * 2 + 1;
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        }

        cv::Mat labels;
        cv::Mat componentStats;
        cv::Mat centroids;
        const int componentCount =
            cv::connectedComponentsWithStats(binary, labels, componentStats, centroids, 8, CV_32S);
        stat.componentCount = std::max(0, componentCount - 1);

        int bestLabel = 0;
        int bestArea = 0;
        for (int label = 1; label < componentCount; ++label) {
            const int area = componentStats.at<int>(label, cv::CC_STAT_AREA);
            if (area < std::max(1, config.minComponentAreaPx)) {
                continue;
            }
            if (area > bestArea) {
                bestArea = area;
                bestLabel = label;
            }
        }

        if (bestLabel == 0 || bestArea <= 0) {
            stats.push_back(stat);
            continue;
        }

        cv::Mat mask = labels == bestLabel;
        mask.convertTo(mask, CV_8U, 255.0);
        if (config.dilateRadiusPx > 0) {
            const int kernelSize = config.dilateRadiusPx * 2 + 1;
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
            cv::dilate(mask, mask, kernel);
        }

        stat.selectedLabel = bestLabel;
        stat.selectedAreaPx = bestArea;
        stat.keptPixels = cv::countNonZero(mask);
        stat.maskedPixels = mask.rows * mask.cols - stat.keptPixels;
        stat.applied = stat.maskedPixels > 0;

        cv::Mat masked = images[imageIndex].clone();
        if (masked.channels() == 1) {
            masked.setTo(cv::Scalar(255), mask == 0);
        } else if (masked.channels() == 3) {
            masked.setTo(cv::Scalar(255, 255, 255), mask == 0);
        } else if (masked.channels() == 4) {
            masked.setTo(cv::Scalar(255, 255, 255, 255), mask == 0);
        } else {
            masked.setTo(cv::Scalar::all(255), mask == 0);
        }
        images[imageIndex] = std::move(masked);

        if (config.saveMaskedImages && !config.saveDir.empty()) {
            const std::filesystem::path savePath =
                config.saveDir /
                (std::filesystem::path(stat.imageName).stem().u8string() + "_masked.png");
            stitch::saveImageToPath(savePath.u8string(), images[imageIndex]);
        }
        stats.push_back(stat);
    }

    return stats;
}

std::string buildDominantDarkComponentMaskCsv(
    const std::vector<DominantDarkComponentMaskStat>& stats)
{
    std::ostringstream stream;
    stream << "image_index,image_name,applied,threshold_gray,component_count,selected_label,"
              "selected_area_px,kept_pixels,masked_pixels,kept_ratio\n";
    for (const auto& stat : stats) {
        const double keptRatio =
            (stat.keptPixels + stat.maskedPixels) > 0
                ? static_cast<double>(stat.keptPixels) /
                      static_cast<double>(stat.keptPixels + stat.maskedPixels)
                : 0.0;
        stream << stat.imageIndex << ","
               << stat.imageName << ","
               << (stat.applied ? 1 : 0) << ","
               << stat.thresholdGray << ","
               << stat.componentCount << ","
               << stat.selectedLabel << ","
               << stat.selectedAreaPx << ","
               << stat.keptPixels << ","
               << stat.maskedPixels << ","
               << keptRatio << "\n";
    }
    return stream.str();
}

std::string buildSupportChangeMaskCsv(const std::vector<SupportChangeMaskStat>& stats)
{
    std::ostringstream stream;
    stream << "image_index,image_name,reference_loaded,changed_pixels,masked_pixels,"
              "points_before,points_after,points_removed,removed_ratio\n";
    for (const auto& stat : stats) {
        const std::size_t removed =
            stat.pointsBefore > stat.pointsAfter ? stat.pointsBefore - stat.pointsAfter : 0;
        const double removedRatio =
            stat.pointsBefore > 0 ? static_cast<double>(removed) /
                                        static_cast<double>(stat.pointsBefore)
                                  : 0.0;
        stream << stat.imageIndex << ","
               << stat.imageName << ","
               << (stat.referenceLoaded ? 1 : 0) << ","
               << stat.changedPixels << ","
               << stat.maskedPixels << ","
               << stat.pointsBefore << ","
               << stat.pointsAfter << ","
               << removed << ","
               << removedRatio << "\n";
    }
    return stream.str();
}

std::string buildCircleEdgeCleanupCsv(const std::vector<CircleEdgeCleanupStat>& stats)
{
    std::ostringstream stream;
    stream << "image_index,image_name,fitted,center_x_px,center_y_px,radius_px,threshold_px,"
              "points_before,points_after,points_removed,completed_points,removed_ratio\n";
    for (const auto& stat : stats) {
        const std::size_t removed =
            stat.pointsBefore > stat.pointsAfter ? stat.pointsBefore - stat.pointsAfter : 0;
        const double removedRatio =
            stat.pointsBefore > 0 ? static_cast<double>(removed) /
                                        static_cast<double>(stat.pointsBefore)
                                  : 0.0;
        stream << stat.imageIndex << ","
               << stat.imageName << ","
               << (stat.fitted ? 1 : 0) << ","
               << stat.centerX << ","
               << stat.centerY << ","
               << stat.radius << ","
               << stat.thresholdPx << ","
               << stat.pointsBefore << ","
               << stat.pointsAfter << ","
               << removed << ","
               << stat.completedPoints << ","
               << removedRatio << "\n";
    }
    return stream.str();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc >= 2) {
        const std::string firstArg = argv[1];
        if (firstArg == "--help" || firstArg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const std::filesystem::path inputDir(argv[1]);
    int imageCount = 0;
    if (!tryParseInt(argv[2], imageCount) || imageCount < 2) {
        std::cout << "[Error] image_count must be an integer >= 2.\n";
        return 1;
    }
    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
        std::cout << "[Error] Invalid input directory: " << inputDir.generic_string() << "\n";
        return 1;
    }

    int startIndex = 1;
    std::string prefix = "Pic_";
    std::string extension = ".bmp";
    bool reverseOrder = false;
    bool useCircleCenterTranslationPrior = true;
    bool forceCircleCenterTranslationPrior = false;
    std::filesystem::path outputDir =
        std::filesystem::path("result") / "standard_sphere_loop" / ("run_" + timestampToken());
    SupportChangeMaskConfig supportMaskConfig;
    DominantDarkComponentMaskConfig dominantDarkMaskConfig;
    CircleEdgeCleanupConfig circleCleanupConfig;
    Gbt57P2dConfig gbt57P2dConfig;
    bool edgeCleanupOnly = false;
    bool gbt57MaskStandardCircleProfile = false;
    bool gbt57ModerateAutoRefineProfile = false;
    bool overlapExplicit = false;
    bool directionExplicit = false;
    bool segmentCountsExplicit = false;
    bool selectedPointRefineExplicit = false;
    bool selectedPointRefineIterExplicit = false;
    bool selectedPointRefineStepExplicit = false;
    bool selectedPointRefineGainExplicit = false;
    bool guiProgressEnabled = false;

    stitch::EdgeDetectConfig edgeConfig;
    edgeConfig.filterHampelSigma = 3.0;

    pinjie::standard_sphere_loop::StandardSphereLoopConfig experimentConfig;
    experimentConfig.sphereDiameterMm = kDefaultStandardSphereDiameterMm;
    experimentConfig.sphereDiameterDecimalPlaces = 4;
    experimentConfig.pipelineConfig.expectedOverlapRatio = 0.70;
    experimentConfig.pipelineConfig.approxShiftRatio = 0.30;
    experimentConfig.pipelineConfig.directionConstraint = stitch::MotionPriorDirection::XPositive;
    experimentConfig.pipelineConfig.baseSearchRange = 200.0;
    experimentConfig.pipelineConfig.rotationSearchMinDeg = -0.2;
    experimentConfig.pipelineConfig.rotationSearchMaxDeg = 0.2;
    experimentConfig.pipelineConfig.rotationSearchStepDeg = 0.01;
    experimentConfig.pipelineConfig.tangentResidualCostWeight = 0.05;
    experimentConfig.pipelineConfig.tangentCorrelationCostWeight = 0.25;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--start-index" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], startIndex) || startIndex <= 0) {
                std::cout << "[Error] --start-index must be a positive integer.\n";
                return 1;
            }
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--ext" && i + 1 < argc) {
            extension = argv[++i];
        } else if (arg == "--gui-progress") {
            guiProgressEnabled = true;
        } else if (arg == "--reverse-order") {
            reverseOrder = true;
        } else if (arg == "--pixel-size" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pixelSizeMm) ||
                experimentConfig.pixelSizeMm <= 0.0) {
                std::cout << "[Error] --pixel-size must be a positive mm/px value.\n";
                return 1;
            }
        } else if (arg == "--sphere-diameter" && i + 1 < argc) {
            const std::string diameterText = argv[++i];
            experimentConfig.sphereDiameterDecimalPlaces =
                decimalPlacesFromText(diameterText, experimentConfig.sphereDiameterDecimalPlaces);
            if (!tryParseDouble(diameterText, experimentConfig.sphereDiameterMm) ||
                experimentConfig.sphereDiameterMm <= 0.0) {
                std::cout << "[Error] --sphere-diameter must be a positive mm value.\n";
                return 1;
            }
        } else if (arg == "--overlap" && i + 1 < argc) {
            double value = 0.0;
            if (!tryParseDouble(argv[++i], value)) {
                std::cout << "[Error] Invalid --overlap value.\n";
                return 1;
            }
            overlapExplicit = true;
            value = normalizeUnitRatio(value);
            if (value < 0.0 || value > 1.0) {
                std::cout << "[Error] --overlap must be within [0, 1] or [0, 100].\n";
                return 1;
            }
            experimentConfig.pipelineConfig.expectedOverlapRatio = value;
            experimentConfig.pipelineConfig.approxShiftRatio = std::max(0.0, 1.0 - value);
        } else if (arg == "--path-mode" && i + 1 < argc) {
            const std::string mode = toLowerAscii(argv[++i]);
            if (mode == "clockwise" || mode == "loop") {
                experimentConfig.useClockwiseLoopPath = true;
            } else if (mode == "single" || mode == "linear") {
                experimentConfig.useClockwiseLoopPath = false;
            } else {
                std::cout << "[Error] Invalid --path-mode. Use clockwise or single.\n";
                return 1;
            }
        } else if (arg == "--clockwise-loop") {
            experimentConfig.useClockwiseLoopPath = true;
        } else if (arg == "--single-direction") {
            experimentConfig.useClockwiseLoopPath = false;
        } else if (arg == "--no-circle-prior") {
            useCircleCenterTranslationPrior = false;
        } else if (arg == "--force-circle-prior") {
            useCircleCenterTranslationPrior = true;
            forceCircleCenterTranslationPrior = true;
        } else if (arg == "--no-loop-optimization") {
            experimentConfig.enableGlobalLoopClosureOptimization = false;
        } else if (arg == "--fast-p2d-stitch-only") {
            experimentConfig.enableGlobalLoopClosureOptimization = false;
            experimentConfig.enableSphereBadStepGeometryRescue = false;
            experimentConfig.enableGlobalConsistencyReregistration = false;
            experimentConfig.enableSoftGlobalDriftOptimization = false;
            experimentConfig.enableBadStepLocalRefinement = false;
            experimentConfig.pipelineConfig.enableTranslationPriorFallback = true;
            experimentConfig.pipelineConfig.translationPriorFallbackNormalRmseThreshold = 3.0;
        } else if (arg == "--segment-counts" && i + 1 < argc) {
            if (!parseSegmentCounts(argv[++i], experimentConfig.clockwiseSegmentPairCounts)) {
                std::cout << "[Error] --segment-counts must be comma-separated non-negative integers.\n";
                return 1;
            }
            segmentCountsExplicit = true;
            experimentConfig.useClockwiseLoopPath = true;
        } else if (arg == "--segment-directions" && i + 1 < argc) {
            if (!parseDirectionList(argv[++i], experimentConfig.clockwiseSegmentDirections)) {
                std::cout << "[Error] --segment-directions must be comma-separated x+, y+, x-, y- tokens.\n";
                return 1;
            }
            experimentConfig.useClockwiseLoopPath = true;
        } else if (arg == "--horizontal-fov" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.horizontalFieldOfViewMm) ||
                experimentConfig.horizontalFieldOfViewMm <= 0.0) {
                std::cout << "[Error] --horizontal-fov must be a positive mm value.\n";
                return 1;
            }
        } else if (arg == "--vertical-fov" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.verticalFieldOfViewMm) ||
                experimentConfig.verticalFieldOfViewMm <= 0.0) {
                std::cout << "[Error] --vertical-fov must be a positive mm value.\n";
                return 1;
            }
        } else if (arg == "--horizontal-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.horizontalStepMm) ||
                experimentConfig.horizontalStepMm <= 0.0) {
                std::cout << "[Error] --horizontal-step must be a positive mm value.\n";
                return 1;
            }
        } else if (arg == "--vertical-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.verticalStepMm) ||
                experimentConfig.verticalStepMm <= 0.0) {
                std::cout << "[Error] --vertical-step must be a positive mm value.\n";
                return 1;
            }
        } else if (arg == "--direction" && i + 1 < argc) {
            if (!parseDirectionConstraint(argv[++i], experimentConfig.pipelineConfig.directionConstraint)) {
                std::cout << "[Error] Invalid --direction. Use auto, x+, x-, y+, or y-.\n";
                return 1;
            }
            directionExplicit = true;
            experimentConfig.useClockwiseLoopPath = false;
        } else if ((arg == "--search-range" || arg == "--base-search-range") && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pipelineConfig.baseSearchRange) ||
                experimentConfig.pipelineConfig.baseSearchRange <= 0.0) {
                std::cout << "[Error] --search-range must be a positive pixel value.\n";
                return 1;
            }
        } else if (arg == "--rotation-range" && i + 1 < argc) {
            double value = 0.0;
            if (!tryParseDouble(argv[++i], value) || value < 0.0) {
                std::cout << "[Error] --rotation-range must be non-negative.\n";
                return 1;
            }
            experimentConfig.pipelineConfig.rotationSearchMinDeg = -value;
            experimentConfig.pipelineConfig.rotationSearchMaxDeg = value;
        } else if (arg == "--rotation-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pipelineConfig.rotationSearchStepDeg) ||
                experimentConfig.pipelineConfig.rotationSearchStepDeg <= 0.0) {
                std::cout << "[Error] --rotation-step must be positive.\n";
                return 1;
            }
        } else if (arg == "--tangent-residual-weight" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pipelineConfig.tangentResidualCostWeight) ||
                experimentConfig.pipelineConfig.tangentResidualCostWeight < 0.0) {
                std::cout << "[Error] --tangent-residual-weight must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--tangent-correlation-weight" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pipelineConfig.tangentCorrelationCostWeight) ||
                experimentConfig.pipelineConfig.tangentCorrelationCostWeight < 0.0) {
                std::cout << "[Error] --tangent-correlation-weight must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--no-point-filter") {
            edgeConfig.enablePointFiltering = false;
        } else if (arg == "--filter-confidence-q" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], edgeConfig.filterConfidenceQuantile)) {
                std::cout << "[Error] Invalid --filter-confidence-q.\n";
                return 1;
            }
            edgeConfig.filterConfidenceQuantile = normalizeUnitRatio(edgeConfig.filterConfidenceQuantile);
        } else if (arg == "--filter-gradient-q" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], edgeConfig.filterGradientQuantile)) {
                std::cout << "[Error] Invalid --filter-gradient-q.\n";
                return 1;
            }
            edgeConfig.filterGradientQuantile = normalizeUnitRatio(edgeConfig.filterGradientQuantile);
        } else if (arg == "--filter-window-radius" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], edgeConfig.filterLocalLinearWindowRadius) ||
                edgeConfig.filterLocalLinearWindowRadius < 1) {
                std::cout << "[Error] --filter-window-radius must be an integer >= 1.\n";
                return 1;
            }
        } else if (arg == "--filter-hampel-sigma" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], edgeConfig.filterHampelSigma) ||
                edgeConfig.filterHampelSigma <= 0.0) {
                std::cout << "[Error] --filter-hampel-sigma must be positive.\n";
                return 1;
            }
        } else if (arg == "--filter-hampel-min-scale" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], edgeConfig.filterHampelMinScale) ||
                edgeConfig.filterHampelMinScale <= 0.0) {
                std::cout << "[Error] --filter-hampel-min-scale must be positive.\n";
                return 1;
            }
        } else if (arg == "--raw-reference-dir" && i + 1 < argc) {
            supportMaskConfig.rawReferenceDir = argv[++i];
            supportMaskConfig.enabled = true;
            if (!std::filesystem::exists(supportMaskConfig.rawReferenceDir) ||
                !std::filesystem::is_directory(supportMaskConfig.rawReferenceDir)) {
                std::cout << "[Error] Invalid --raw-reference-dir: "
                          << supportMaskConfig.rawReferenceDir.generic_string() << "\n";
                return 1;
            }
        } else if (arg == "--support-mask-threshold" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], supportMaskConfig.thresholdGray) ||
                supportMaskConfig.thresholdGray < 0.0) {
                std::cout << "[Error] --support-mask-threshold must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--support-mask-dilate" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], supportMaskConfig.dilateRadiusPx) ||
                supportMaskConfig.dilateRadiusPx < 0) {
                std::cout << "[Error] --support-mask-dilate must be a non-negative integer.\n";
                return 1;
            }
        } else if (arg == "--support-mask-min-pixels" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], supportMaskConfig.minChangedPixels) ||
                supportMaskConfig.minChangedPixels < 0) {
                std::cout << "[Error] --support-mask-min-pixels must be a non-negative integer.\n";
                return 1;
            }
        } else if (arg == "--support-mask-region") {
            supportMaskConfig.boundaryOnly = false;
        } else if (arg == "--main-dark-component-mask") {
            dominantDarkMaskConfig.enabled = true;
        } else if (arg == "--main-dark-threshold" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], dominantDarkMaskConfig.thresholdGray) ||
                dominantDarkMaskConfig.thresholdGray < 0.0 ||
                dominantDarkMaskConfig.thresholdGray > 255.0) {
                std::cout << "[Error] --main-dark-threshold must be within [0, 255].\n";
                return 1;
            }
            dominantDarkMaskConfig.enabled = true;
        } else if (arg == "--main-dark-close-radius" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], dominantDarkMaskConfig.closeRadiusPx) ||
                dominantDarkMaskConfig.closeRadiusPx < 0) {
                std::cout << "[Error] --main-dark-close-radius must be a non-negative integer.\n";
                return 1;
            }
            dominantDarkMaskConfig.enabled = true;
        } else if (arg == "--main-dark-dilate-radius" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], dominantDarkMaskConfig.dilateRadiusPx) ||
                dominantDarkMaskConfig.dilateRadiusPx < 0) {
                std::cout << "[Error] --main-dark-dilate-radius must be a non-negative integer.\n";
                return 1;
            }
            dominantDarkMaskConfig.enabled = true;
        } else if (arg == "--main-dark-min-area" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], dominantDarkMaskConfig.minComponentAreaPx) ||
                dominantDarkMaskConfig.minComponentAreaPx < 1) {
                std::cout << "[Error] --main-dark-min-area must be a positive integer.\n";
                return 1;
            }
            dominantDarkMaskConfig.enabled = true;
        } else if (arg == "--save-preprocessed-images") {
            dominantDarkMaskConfig.saveMaskedImages = true;
        } else if (arg == "--circle-edge-cleanup") {
            circleCleanupConfig.enabled = true;
        } else if (arg == "--circle-edge-complete") {
            circleCleanupConfig.enabled = true;
            circleCleanupConfig.completeCircleGaps = true;
        } else if (arg == "--circle-complete-images" && i + 1 < argc) {
            if (!parseSegmentCounts(argv[++i], circleCleanupConfig.completionOnlyImageNumbers)) {
                std::cout << "[Error] --circle-complete-images must be comma-separated positive image numbers.\n";
                return 1;
            }
            circleCleanupConfig.enabled = true;
            circleCleanupConfig.completeCircleGaps = true;
        } else if (arg == "--edge-cleanup-only") {
            edgeCleanupOnly = true;
        } else if (arg == "--translation-prior-fallback") {
            experimentConfig.pipelineConfig.enableTranslationPriorFallback = true;
            experimentConfig.pipelineConfig.translationPriorFallbackNormalRmseThreshold = 3.0;
        } else if (arg == "--gbt57-mask-standard-circle-profile") {
            gbt57MaskStandardCircleProfile = true;
        } else if (arg == "--gbt57-moderate-auto-refine-profile") {
            gbt57ModerateAutoRefineProfile = true;
            gbt57P2dConfig.enabled = true;
        } else if (arg == "--circle-cleanup-sigma" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], circleCleanupConfig.sigma) || circleCleanupConfig.sigma <= 0.0) {
                std::cout << "[Error] --circle-cleanup-sigma must be positive.\n";
                return 1;
            }
        } else if (arg == "--circle-cleanup-max-residual" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], circleCleanupConfig.maxResidualPx) ||
                circleCleanupConfig.maxResidualPx <= 0.0) {
                std::cout << "[Error] --circle-cleanup-max-residual must be positive.\n";
                return 1;
            }
        } else if (arg == "--circle-cleanup-min-scale" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], circleCleanupConfig.minScalePx) ||
                circleCleanupConfig.minScalePx <= 0.0) {
                std::cout << "[Error] --circle-cleanup-min-scale must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-p2d") {
            gbt57P2dConfig.enabled = true;
        } else if (arg == "--gbt57-local-circle-frame") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.localCircleFrame = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-circle-center-global") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.circleCenterGlobal = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-circle-center-angle-search") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.circleCenterGlobal = true;
            gbt57P2dConfig.circleCenterLocalAngleSearch = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-circle-center-fixed-radius") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.circleCenterGlobal = true;
            gbt57P2dConfig.circleCenterFixedRadius = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-circle-center-normalize-radius") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.circleCenterGlobal = true;
            gbt57P2dConfig.circleCenterNormalizeRadius = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-radial-consistency-refine") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.circleCenterGlobal = true;
            gbt57P2dConfig.circleCenterRadialConsistencyRefine = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-radial-refine-iterations" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.radialConsistencyIterations) ||
                gbt57P2dConfig.radialConsistencyIterations < 0) {
                std::cout << "[Error] --gbt57-radial-refine-iterations must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-radial-refine-max-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.radialConsistencyMaxStepPx) ||
                gbt57P2dConfig.radialConsistencyMaxStepPx < 0.0) {
                std::cout << "[Error] --gbt57-radial-refine-max-step must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-radius-stable-selection") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.radiusStableSelection = true;
        } else if (arg == "--gbt57-uniform-angle-selection") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.uniformAngleSelection = true;
        } else if (arg == "--gbt57-optimize-selected-range") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.optimizeSelectedRange = true;
        } else if (arg == "--gbt57-range-candidates" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.rangeOptimizationCandidatesPerField) ||
                gbt57P2dConfig.rangeOptimizationCandidatesPerField < 1) {
                std::cout << "[Error] --gbt57-range-candidates must be a positive integer.\n";
                return 1;
            }
        } else if (arg == "--gbt57-range-restarts" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.rangeOptimizationRestarts) ||
                gbt57P2dConfig.rangeOptimizationRestarts < 0) {
                std::cout << "[Error] --gbt57-range-restarts must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-field-bias-compensation") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.fieldBiasCompensation = true;
        } else if (arg == "--gbt57-local-field-bias-compensation") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.localFieldBiasCompensation = true;
        } else if (arg == "--gbt57-angular-bias-compensation") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.angularBiasCompensation = true;
        } else if (arg == "--gbt57-angular-bias-order" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.angularBiasOrder) ||
                gbt57P2dConfig.angularBiasOrder < 1) {
                std::cout << "[Error] --gbt57-angular-bias-order must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-angular-bias-gain" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.angularBiasGain) ||
                gbt57P2dConfig.angularBiasGain < 0.0) {
                std::cout << "[Error] --gbt57-angular-bias-gain must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-ellipse-normalize") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.ellipseNormalizationCompensation = true;
        } else if (arg == "--gbt57-selected-point-refine") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.selectedPointTransformRefine = true;
            selectedPointRefineExplicit = true;
        } else if (arg == "--gbt57-supplement-precleanup-candidates") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.supplementPreCleanupCandidates = true;
        } else if (arg == "--gbt57-soft-spacing-guard") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.softSpacingGuard = true;
        } else if (arg == "--gbt57-soft-spacing-max-error" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.softSpacingMaxErrorDeg) ||
                gbt57P2dConfig.softSpacingMaxErrorDeg < 0.0) {
                std::cout << "[Error] --gbt57-soft-spacing-max-error must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-soft-spacing-rmse" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.softSpacingRmseDeg) ||
                gbt57P2dConfig.softSpacingRmseDeg < 0.0) {
                std::cout << "[Error] --gbt57-soft-spacing-rmse must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-soft-target-max" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.softTargetAngleMaxDeg) ||
                gbt57P2dConfig.softTargetAngleMaxDeg < 0.0) {
                std::cout << "[Error] --gbt57-soft-target-max must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-soft-target-mean" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.softTargetAngleMeanDeg) ||
                gbt57P2dConfig.softTargetAngleMeanDeg < 0.0) {
                std::cout << "[Error] --gbt57-soft-target-mean must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-selected-point-refine-iter" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.selectedPointTransformRefineIterations) ||
                gbt57P2dConfig.selectedPointTransformRefineIterations < 0) {
                std::cout << "[Error] --gbt57-selected-point-refine-iter must be non-negative.\n";
                return 1;
            }
            selectedPointRefineIterExplicit = true;
        } else if (arg == "--gbt57-selected-point-refine-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineMaxStepPx) ||
                gbt57P2dConfig.selectedPointTransformRefineMaxStepPx < 0.0) {
                std::cout << "[Error] --gbt57-selected-point-refine-step must be non-negative.\n";
                return 1;
            }
            selectedPointRefineStepExplicit = true;
        } else if (arg == "--gbt57-selected-point-refine-gain" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineGain) ||
                gbt57P2dConfig.selectedPointTransformRefineGain < 0.0) {
                std::cout << "[Error] --gbt57-selected-point-refine-gain must be non-negative.\n";
                return 1;
            }
            selectedPointRefineGainExplicit = true;
        } else if (arg == "--gbt57-auto-refine-max-shift" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineAutoMaxShiftPx) ||
                gbt57P2dConfig.selectedPointTransformRefineAutoMaxShiftPx < 0.0) {
                std::cout << "[Error] --gbt57-auto-refine-max-shift must be non-negative.\n";
                return 1;
            }
            gbt57P2dConfig.selectedPointTransformRefineAutoGate = true;
        } else if (arg == "--gbt57-auto-refine-max-target-delta" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg) ||
                gbt57P2dConfig.selectedPointTransformRefineAutoMaxTargetAngleDeltaDeg < 0.0) {
                std::cout << "[Error] --gbt57-auto-refine-max-target-delta must be non-negative.\n";
                return 1;
            }
            gbt57P2dConfig.selectedPointTransformRefineAutoGate = true;
        } else if (arg == "--gbt57-circular-hampel-radius" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.circularResidualHampelRadius) ||
                gbt57P2dConfig.circularResidualHampelRadius < 0) {
                std::cout << "[Error] --gbt57-circular-hampel-radius must be a non-negative integer.\n";
                return 1;
            }
        } else if (arg == "--gbt57-circular-hampel-sigma" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.circularResidualHampelSigma) ||
                gbt57P2dConfig.circularResidualHampelSigma < 0.0) {
                std::cout << "[Error] --gbt57-circular-hampel-sigma must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-circular-median-filter-radius" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.circularResidualMedianFilterRadius) ||
                gbt57P2dConfig.circularResidualMedianFilterRadius < 0) {
                std::cout << "[Error] --gbt57-circular-median-filter-radius must be a non-negative integer.\n";
                return 1;
            }
        } else if (arg == "--gbt57-circular-filter-blend" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.circularResidualFilterBlend) ||
                gbt57P2dConfig.circularResidualFilterBlend < 0.0 ||
                gbt57P2dConfig.circularResidualFilterBlend > 1.0) {
                std::cout << "[Error] --gbt57-circular-filter-blend must be within [0, 1].\n";
                return 1;
            }
        } else if (arg == "--gbt57-local-bias-half-angle" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.localFieldBiasHalfAngleDeg) ||
                gbt57P2dConfig.localFieldBiasHalfAngleDeg <= 0.0) {
                std::cout << "[Error] --gbt57-local-bias-half-angle must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-confidence-best-selection") {
            gbt57P2dConfig.enabled = true;
            gbt57P2dConfig.confidenceBestSelection = true;
        } else if (arg == "--gbt57-confidence-radius-guard" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.confidenceRadiusGuardPx) ||
                gbt57P2dConfig.confidenceRadiusGuardPx <= 0.0) {
                std::cout << "[Error] --gbt57-confidence-radius-guard must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-center-angle-range" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.centerGlobalAngleSearchRangeDeg) ||
                gbt57P2dConfig.centerGlobalAngleSearchRangeDeg < 0.0) {
                std::cout << "[Error] --gbt57-center-angle-range must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-center-angle-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.centerGlobalAngleSearchStepDeg) ||
                gbt57P2dConfig.centerGlobalAngleSearchStepDeg <= 0.0) {
                std::cout << "[Error] --gbt57-center-angle-step must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-window-half-angle" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.windowHalfAngleDeg) ||
                gbt57P2dConfig.windowHalfAngleDeg <= 0.0) {
                std::cout << "[Error] --gbt57-window-half-angle must be positive.\n";
                return 1;
            }
        } else if (arg == "--gbt57-window-half-size" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.windowHalfSizePx) ||
                gbt57P2dConfig.windowHalfSizePx <= 0.0) {
                std::cout << "[Error] --gbt57-window-half-size must be positive.\n";
                return 1;
            }
        } else {
            std::cout << "[Warn] Unknown or incomplete argument: " << arg << "\n";
        }
    }

    if (gbt57MaskStandardCircleProfile) {
        dominantDarkMaskConfig.enabled = true;
        dominantDarkMaskConfig.saveMaskedImages = true;
        circleCleanupConfig.enabled = true;
        gbt57P2dConfig.enabled = true;
        gbt57P2dConfig.supplementPreCleanupCandidates = true;
        gbt57P2dConfig.uniformAngleSelection = true;
        gbt57P2dConfig.optimizeSelectedRange = true;
        gbt57P2dConfig.softSpacingGuard = true;
        gbt57P2dConfig.ellipseNormalizationCompensation = true;
        gbt57P2dConfig.circleCenterGlobal = true;
        gbt57P2dConfig.circleCenterLocalAngleSearch = true;
        gbt57P2dConfig.circleCenterFixedRadius = true;
        gbt57P2dConfig.repeatabilityBandAutoFilter = true;
        if (!selectedPointRefineExplicit) {
            // The 25 GB/T measurement points must not tune the pose used to measure them.
            // Keeping this off by default avoids unrealistically small nanometer-level E_P2D.
            gbt57P2dConfig.selectedPointTransformRefine = false;
            gbt57P2dConfig.selectedPointTransformRefineAutoGate = false;
        }
        useCircleCenterTranslationPrior = true;
        forceCircleCenterTranslationPrior = true;
        experimentConfig.useClockwiseLoopPath = false;
        experimentConfig.enableSphereBadStepGeometryRescue = false;
        experimentConfig.pipelineConfig.enableBadStepCandidateReselect = false;
        experimentConfig.pipelineConfig.enableTranslationPriorFallback = true;
        experimentConfig.pipelineConfig.translationPriorFallbackNormalRmseThreshold = 3.0;
        experimentConfig.pipelineConfig.endpointProbeFastMode = false;
        experimentConfig.pipelineConfig.enableLastStepReferencePriorHalfOverlapProbe = false;
        if (!overlapExplicit) {
            experimentConfig.pipelineConfig.expectedOverlapRatio = 0.70;
            experimentConfig.pipelineConfig.approxShiftRatio = 0.30;
        }
    }

    if (gbt57ModerateAutoRefineProfile) {
        gbt57P2dConfig.enabled = true;
        gbt57P2dConfig.selectedPointTransformRefine = true;
        gbt57P2dConfig.selectedPointTransformRefineAutoGate = true;
        if (!selectedPointRefineIterExplicit) {
            gbt57P2dConfig.selectedPointTransformRefineIterations = 6;
        }
        if (!selectedPointRefineStepExplicit) {
            gbt57P2dConfig.selectedPointTransformRefineMaxStepPx = 0.18;
        }
        if (!selectedPointRefineGainExplicit) {
            gbt57P2dConfig.selectedPointTransformRefineGain = 0.90;
        }
    }

    if (gbt57P2dConfig.enabled) {
        if (!segmentCountsExplicit) {
            experimentConfig.useClockwiseLoopPath = false;
        }
        if (!forceCircleCenterTranslationPrior) {
            useCircleCenterTranslationPrior = false;
        }
        if (!overlapExplicit) {
            experimentConfig.pipelineConfig.expectedOverlapRatio = 0.80;
            experimentConfig.pipelineConfig.approxShiftRatio = 0.20;
        }
        if (!directionExplicit && !segmentCountsExplicit) {
            experimentConfig.pipelineConfig.directionConstraint = stitch::MotionPriorDirection::Auto;
        }
    }

    if (guiProgressEnabled) {
        experimentConfig.pipelineConfig.generateDebugVisualization = true;
    }

    std::filesystem::create_directories(outputDir);
    if (dominantDarkMaskConfig.saveMaskedImages) {
        dominantDarkMaskConfig.saveDir = outputDir / "preprocessed_masked_images";
    }

    std::vector<std::string> imagePaths =
        collectSequentialImagePaths(inputDir, imageCount, startIndex, prefix, extension);
    if (reverseOrder) {
        std::reverse(imagePaths.begin(), imagePaths.end());
    }

    stitch::StitchCallbacks callbacks;
    callbacks.onLog = [](const std::string& message) {
        std::cout << message << "\n";
    };
    if (guiProgressEnabled) {
        const std::filesystem::path guiProgressDir = outputDir / "gui_progress";
        std::filesystem::create_directories(guiProgressDir);
        callbacks.onProgress = [](const std::string& stage, std::size_t current, std::size_t total) {
            emitGuiProgressMarker(stage, current, total);
        };
        callbacks.onImage = [guiProgressDir](const std::string& stage,
                                             std::size_t index,
                                             std::size_t total,
                                             const cv::Mat& image) {
            if (image.empty()) {
                return;
            }

            const std::string fileName = buildGuiProgressImageFileName(stage, index);
            const std::filesystem::path imagePath = guiProgressDir / fileName;
            if (stitch::saveImageToPath(imagePath.u8string(), image)) {
                emitGuiImageMarker(stage, index, total, fileName);
            }
        };
    }

    std::cout << "[Info] Loading " << imagePaths.size() << " standard sphere images...\n";
    const std::vector<cv::Mat> rawImages = stitch::loadInputImages(imagePaths, callbacks);
    if (rawImages.size() != imagePaths.size()) {
        std::cout << "[Error] Failed to load all input images.\n";
        return 1;
    }
    std::vector<cv::Mat> images = rawImages;
    const std::vector<DominantDarkComponentMaskStat> dominantDarkMaskStats =
        applyDominantDarkComponentMask(images, imagePaths, dominantDarkMaskConfig);
    if (!dominantDarkMaskStats.empty()) {
        std::size_t totalMasked = 0;
        int appliedImages = 0;
        for (const auto& stat : dominantDarkMaskStats) {
            if (stat.applied) {
                totalMasked += static_cast<std::size_t>(std::max(0, stat.maskedPixels));
                ++appliedImages;
            }
        }
        std::cout << "[Info] Dominant-dark preprocessing masked " << totalMasked
                  << " pixels across " << appliedImages << " images.\n";
    }

    std::cout << "[Info] Preprocessing edges with existing stitch pipeline settings...\n";
    std::vector<stitch::EdgeVariants> edges = stitch::preprocessAllImages(images, edgeConfig, callbacks);
    if (edges.size() != images.size()) {
        std::cout << "[Error] Edge preprocessing failed.\n";
        return 1;
    }
    if (callbacks.onImage) {
        for (std::size_t i = 0; i < images.size(); ++i) {
            const cv::Mat preview = stitch::buildPreprocessVisualization(images[i],
                                                                         edges[i].raw,
                                                                         static_cast<int>(i + 1),
                                                                         static_cast<int>(images.size()));
            callbacks.onImage("preprocess_preview", i + 1, images.size(), preview);
        }
    }
    const std::vector<SupportChangeMaskStat> supportMaskStats =
        applySupportChangeMasks(images, edges, imagePaths, supportMaskConfig);
    if (!supportMaskStats.empty()) {
        std::size_t totalRemoved = 0;
        int maskedImages = 0;
        for (const auto& stat : supportMaskStats) {
            if (stat.pointsBefore > stat.pointsAfter) {
                totalRemoved += stat.pointsBefore - stat.pointsAfter;
                ++maskedImages;
            }
        }
        std::cout << "[Info] Support-change edge mask removed " << totalRemoved
                  << " edge points from " << maskedImages << " images.\n";
    }
    const std::vector<stitch::EdgeVariants> edgesBeforeCircleCleanup = edges;
    const std::vector<CircleEdgeCleanupStat> circleCleanupStats =
        applyCircleEdgeCleanup(edges, imagePaths, circleCleanupConfig);
    if (!circleCleanupStats.empty()) {
        std::size_t totalRemoved = 0;
        int cleanedImages = 0;
        for (const auto& stat : circleCleanupStats) {
            if (stat.pointsBefore > stat.pointsAfter) {
                totalRemoved += stat.pointsBefore - stat.pointsAfter;
                ++cleanedImages;
            }
        }
        std::cout << "[Info] Circle-edge cleanup removed " << totalRemoved
                  << " edge points from " << cleanedImages << " images.\n";
    }
    if (edgeCleanupOnly) {
        const std::filesystem::path dominantMaskCsv =
            outputDir / "standard_sphere_dominant_dark_component_mask.csv";
        const std::filesystem::path supportMaskCsv =
            outputDir / "standard_sphere_support_change_mask.csv";
        const std::filesystem::path circleCleanupCsv =
            outputDir / "standard_sphere_circle_edge_cleanup.csv";
        if (!dominantDarkMaskStats.empty()) {
            stitch::writeTextFileToPath(dominantMaskCsv.u8string(),
                                        buildDominantDarkComponentMaskCsv(dominantDarkMaskStats));
        }
        if (!supportMaskStats.empty()) {
            stitch::writeTextFileToPath(supportMaskCsv.u8string(),
                                        buildSupportChangeMaskCsv(supportMaskStats));
        }
        if (!circleCleanupStats.empty()) {
            stitch::writeTextFileToPath(circleCleanupCsv.u8string(),
                                        buildCircleEdgeCleanupCsv(circleCleanupStats));
        }
        std::cout << "[Info] Saved edge cleanup diagnostics to: "
                  << outputDir.generic_string() << "\n";
        return 0;
    }

    stitch::StitchingResult stitching;
    std::vector<CircleCenterGlobalReport> circleCenterGlobalReports;
    if (gbt57P2dConfig.enabled && gbt57P2dConfig.localCircleFrame) {
        std::cout << "[Info] Running fast GB/T 5.7 local-circle-frame P2D evaluation...\n";
        stitching = buildLocalCircleFrameStitching(images, edges);
    } else if (gbt57P2dConfig.enabled && gbt57P2dConfig.circleCenterGlobal) {
        std::cout << "[Info] Running standard-sphere circle-center global stitching";
        if (gbt57P2dConfig.circleCenterLocalAngleSearch) {
            std::cout << " with small local angle search";
        }
        std::cout << "...\n";
        stitching = buildCircleCenterGlobalStitching(images,
                                                     edges,
                                                     imagePaths,
                                                     gbt57P2dConfig,
                                                     circleCenterGlobalReports);
    } else {
        experimentConfig.pipelineConfig.stepDirectionConstraints.clear();
        if (experimentConfig.useClockwiseLoopPath) {
            experimentConfig.pipelineConfig.stepDirectionConstraints =
                pinjie::standard_sphere_loop::buildClockwiseStepDirections(
                    images.size() > 0 ? images.size() - 1 : 0,
                    experimentConfig);
        }
        experimentConfig.pipelineConfig.stepTranslationPriorsPx.clear();
        if (useCircleCenterTranslationPrior) {
            experimentConfig.pipelineConfig.stepTranslationPriorsPx =
                pinjie::standard_sphere_loop::buildCircleCenterTranslationPriors(edges);
        }

        std::cout << "[Info] Running full workpiece stitching pipeline and standard sphere evaluation...\n";
        stitching =
            stitch::runStitchingPipeline(images, edges, experimentConfig.pipelineConfig, callbacks);
    }
    if (!stitching.success()) {
        std::cout << "[Error] Full stitching pipeline failed.\n";
        return 1;
    }
    if (gbt57P2dConfig.enabled &&
        !gbt57P2dConfig.localCircleFrame &&
        !gbt57P2dConfig.circleCenterGlobal &&
        (gbt57MaskStandardCircleProfile || dominantDarkMaskConfig.enabled)) {
        std::string rescueMessage;
        if (rescueLastImageTransformByGlobalCircle(stitching,
                                                   images,
                                                   edges,
                                                   &rescueMessage)) {
            std::cout << "[Info] " << rescueMessage << "\n";
        }
    }
    if (gbt57P2dConfig.enabled) {
        pinjie::standard_sphere_loop::StandardSphereLoopConfig sphereRefineConfig =
            experimentConfig;
        sphereRefineConfig.enableGlobalConsistencyReregistration = false;
        sphereRefineConfig.enableBadStepLocalRefinement = false;
        sphereRefineConfig.enableSoftGlobalDriftOptimization = false;
        sphereRefineConfig.circleOptimizationMaxTranslationPx =
            std::min(experimentConfig.circleOptimizationMaxTranslationPx, 2.0);
        sphereRefineConfig.circleOptimizationMaxRotationDeg =
            std::min(experimentConfig.circleOptimizationMaxRotationDeg, 0.05);
        sphereRefineConfig.circleOptimizationMaxIterations =
            std::min(std::max(1, experimentConfig.circleOptimizationMaxIterations), 3);
        const auto sphereEval =
            pinjie::standard_sphere_loop::evaluateStandardSphereStitchingResult(images,
                                                                                edges,
                                                                                stitching,
                                                                                sphereRefineConfig);
        if (sphereEval.ok &&
            sphereEval.globalTransforms.size() == stitching.imageTransforms.size() &&
            sphereEval.stitchedCircle.ok) {
            const bool geometryRefined =
                sphereEval.sphereBadStepGeometryRescueCount > 0 ||
                sphereEval.globalConsistencyReregistrationCount > 0 ||
                sphereEval.badStepLocalRefinementCount > 0 ||
                sphereEval.softGlobalDriftOptimizationIterations > 0 ||
                sphereEval.globalLoopClosureOptimized;
            const bool circleImproved =
                sphereEval.preOptimizationStitchedCircleRmsePx <= 0.0 ||
                sphereEval.stitchedCircle.rmsePx + 1e-6 <
                    sphereEval.preOptimizationStitchedCircleRmsePx;
            if (geometryRefined && circleImproved) {
                stitching.imageTransforms = sphereEval.globalTransforms;
                std::cout << "[Info] standard-sphere pose refinement applied: stitched RMSE "
                          << sphereEval.preOptimizationStitchedCircleRmsePx << " -> "
                          << sphereEval.stitchedCircle.rmsePx
                          << " px, closure translation "
                          << sphereEval.preOptimizationClosureTranslationPx << " -> "
                          << sphereEval.closureTranslationPx
                          << " px, bad-step rescue="
                          << sphereEval.sphereBadStepGeometryRescueCount
                          << ", rereg="
                          << sphereEval.globalConsistencyReregistrationCount
                          << ", local-refine="
                          << sphereEval.badStepLocalRefinementCount
                          << ", loop-opt="
                          << (sphereEval.globalLoopClosureOptimized ? 1 : 0)
                          << "\n";
            }
        }
    }

    if (gbt57P2dConfig.enabled) {
        Gbt57P2dResult p2dResult;
        if (gbt57P2dConfig.selectedPointTransformRefine &&
            gbt57P2dConfig.selectedPointTransformRefineAutoGate) {
            Gbt57P2dConfig baselineConfig = gbt57P2dConfig;
            baselineConfig.selectedPointTransformRefine = false;
            baselineConfig.selectedPointTransformRefineAutoGate = false;
            const Gbt57P2dResult baselineResult =
                evaluateGbt57P2dSinglePoint(images,
                                            edges,
                                            gbt57P2dConfig.supplementPreCleanupCandidates
                                                ? &edgesBeforeCircleCleanup
                                                : nullptr,
                                            stitching,
                                            imagePaths,
                                            baselineConfig,
                                            experimentConfig);
            const Gbt57P2dResult candidateResult =
                evaluateGbt57P2dSinglePoint(images,
                                            edges,
                                            gbt57P2dConfig.supplementPreCleanupCandidates
                                                ? &edgesBeforeCircleCleanup
                                                : nullptr,
                                            stitching,
                                            imagePaths,
                                            gbt57P2dConfig,
                                            experimentConfig);
            std::string autoRefineDecision;
            const bool acceptAutoRefine =
                acceptModerateAutoRefineResult(baselineResult,
                                               candidateResult,
                                               gbt57P2dConfig,
                                               &autoRefineDecision);
            if (acceptAutoRefine) {
                p2dResult = candidateResult;
                p2dResult.autoRefineAccepted = true;
                std::cout << "[Info] moderate auto refine accepted: baseline e_p2d="
                          << baselineResult.eP2dPx
                          << " px, refined e_p2d="
                          << candidateResult.eP2dPx
                          << " px, max shift="
                          << candidateResult.selectedPointTransformRefineMaxShiftPx
                          << " px, target-angle max="
                          << candidateResult.targetAngleDeltaMaxDeg
                          << " deg\n";
            } else {
                p2dResult = baselineResult.ok ? baselineResult : candidateResult;
                p2dResult.selectedPointTransformRefineRequested = true;
                std::cout << "[Info] moderate auto refine rejected, fallback to "
                          << (baselineResult.ok ? "baseline" : "candidate")
                          << ": " << autoRefineDecision << "\n";
            }
            p2dResult.autoRefineEvaluated = true;
            p2dResult.autoRefineAccepted = acceptAutoRefine;
            p2dResult.autoRefineBaselineEP2dPx = baselineResult.eP2dPx;
            p2dResult.autoRefineCandidateEP2dPx = candidateResult.eP2dPx;
            p2dResult.autoRefineCandidateMaxShiftPx =
                candidateResult.selectedPointTransformRefineMaxShiftPx;
            p2dResult.autoRefineCandidateTargetAngleDeltaMaxDeg =
                candidateResult.targetAngleDeltaMaxDeg;
            p2dResult.autoRefineDecision =
                autoRefineDecision.empty()
                    ? (acceptAutoRefine ? "accepted" : "rejected")
                    : autoRefineDecision;
        } else {
            p2dResult =
                evaluateGbt57P2dSinglePoint(images,
                                            edges,
                                            gbt57P2dConfig.supplementPreCleanupCandidates
                                                ? &edgesBeforeCircleCleanup
                                                : nullptr,
                                            stitching,
                                            imagePaths,
                                            gbt57P2dConfig,
                                            experimentConfig);
        }
        const std::filesystem::path p2dSummaryCsv =
            outputDir / "standard_sphere_gbt57_p2d_summary.csv";
        const std::filesystem::path p2dPointsCsv =
            outputDir / "standard_sphere_gbt57_p2d_points.csv";
        const std::filesystem::path p2dRadiiCsv =
            outputDir / "standard_sphere_gbt57_p2d_radii.csv";
        const std::filesystem::path p2dCandidateCoverageCsv =
            outputDir / "standard_sphere_gbt57_p2d_candidate_coverage.csv";
        const std::filesystem::path p2dOverlayPng =
            outputDir / "standard_sphere_gbt57_p2d_overlay.png";
        const std::filesystem::path p2dEdgeOverlayPng =
            outputDir / "standard_sphere_gbt57_p2d_edge_overlay.png";
        const std::filesystem::path p2dWindowOverlayPng =
            outputDir / "standard_sphere_gbt57_p2d_window_overlay.png";
        const std::filesystem::path dominantMaskCsv =
            outputDir / "standard_sphere_dominant_dark_component_mask.csv";
        const std::filesystem::path supportMaskCsv =
            outputDir / "standard_sphere_support_change_mask.csv";
        const std::filesystem::path circleCleanupCsv =
            outputDir / "standard_sphere_circle_edge_cleanup.csv";

        if (callbacks.onProgress) {
            callbacks.onProgress("report", 0, 1);
        }
        stitch::writeTextFileToPath(p2dSummaryCsv.u8string(),
                                    buildGbt57P2dSummaryCsv(p2dResult,
                                                            images,
                                                            gbt57P2dConfig,
                                                            experimentConfig));
        stitch::writeTextFileToPath(p2dPointsCsv.u8string(),
                                    buildGbt57P2dPointCsv(p2dResult));
        stitch::writeTextFileToPath(p2dRadiiCsv.u8string(),
                                    buildGbt57P2dRadiiCsv(p2dResult));
        stitch::writeTextFileToPath(p2dCandidateCoverageCsv.u8string(),
                                    buildGbt57P2dCandidateCoverageCsv(p2dResult));
        const cv::Mat p2dOverlay = buildGbt57P2dOverlay(stitching.canvas, p2dResult);
        if (!p2dOverlay.empty()) {
            stitch::saveImageToPath(p2dOverlayPng.u8string(), p2dOverlay);
            stitch::saveImageToPath(p2dWindowOverlayPng.u8string(), p2dOverlay);
        }
        const cv::Mat p2dEdgeOverlay =
            buildGbt57P2dEdgeOverlay(stitching, edges, p2dResult);
        if (!p2dEdgeOverlay.empty()) {
            stitch::saveImageToPath(p2dEdgeOverlayPng.u8string(), p2dEdgeOverlay);
        }
        if (!dominantDarkMaskStats.empty()) {
            stitch::writeTextFileToPath(dominantMaskCsv.u8string(),
                                        buildDominantDarkComponentMaskCsv(dominantDarkMaskStats));
        }
        if (!supportMaskStats.empty()) {
            stitch::writeTextFileToPath(supportMaskCsv.u8string(),
                                        buildSupportChangeMaskCsv(supportMaskStats));
        }
        if (!circleCleanupStats.empty()) {
            stitch::writeTextFileToPath(circleCleanupCsv.u8string(),
                                        buildCircleEdgeCleanupCsv(circleCleanupStats));
        }
        if (!circleCenterGlobalReports.empty()) {
            const std::filesystem::path circleCenterGlobalCsv =
                outputDir / "standard_sphere_circle_center_global.csv";
            stitch::writeTextFileToPath(circleCenterGlobalCsv.u8string(),
                                        buildCircleCenterGlobalCsv(circleCenterGlobalReports));
        }

        if (!p2dResult.ok) {
            std::cout << "[Error] GB/T 5.7 P2D evaluation failed: "
                      << p2dResult.message << "\n";
            std::cout << "[Info] Saved partial GB/T 5.7 report to: "
                      << outputDir.generic_string() << "\n";
            return 1;
        }

        std::cout << "[Summary] gbt57_e_p2d_px=" << p2dResult.eP2dPx
                  << ", gbt57_e_p2d_um=" << p2dResult.eP2dUm
                  << ", single_point_rmse_px=" << p2dResult.rmsePx
                  << ", pixel_size_um=" << p2dResult.effectivePixelSizeUm
                  << ", angle_spacing_max_error_deg="
                  << p2dResult.angleSpacingMaxErrorDeg << "\n";
        if (callbacks.onProgress) {
            callbacks.onProgress("report", 1, 1);
        }
        std::cout << "[Info] Saved GB/T 5.7 P2D report to: "
                  << outputDir.generic_string() << "\n";
        return 0;
    }

    const auto result = pinjie::standard_sphere_loop::evaluateStandardSphereStitchingResult(
        images, edges, stitching, experimentConfig);
    if (!result.ok) {
        std::cout << "[Error] " << result.message << "\n";
        return 1;
    }

    const std::filesystem::path pairsCsv = outputDir / "standard_sphere_loop_pairs.csv";
    const std::filesystem::path badStepDiagnosticsCsv =
        outputDir / "standard_sphere_bad_step_diagnostics.csv";
    const std::filesystem::path circlesCsv = outputDir / "standard_sphere_circle_fit.csv";
    const std::filesystem::path fieldBiasCsv = outputDir / "standard_sphere_field_bias.csv";
    const std::filesystem::path fieldBiasCompensationCsv =
        outputDir / "standard_sphere_field_bias_compensation.csv";
    const std::filesystem::path transformsCsv = outputDir / "standard_sphere_global_transforms.csv";
    const std::filesystem::path segmentsCsv = outputDir / "standard_sphere_loop_segments.csv";
    const std::filesystem::path summaryCsv = outputDir / "standard_sphere_loop_summary.csv";
    const std::filesystem::path dominantMaskCsv =
        outputDir / "standard_sphere_dominant_dark_component_mask.csv";
    const std::filesystem::path supportMaskCsv =
        outputDir / "standard_sphere_support_change_mask.csv";
    const std::filesystem::path circleCleanupCsv =
        outputDir / "standard_sphere_circle_edge_cleanup.csv";
    const std::filesystem::path stitchedOverlayPng = outputDir / "standard_sphere_stitched_overlay.png";

    stitch::writeTextFileToPath(pairsCsv.u8string(), pinjie::standard_sphere_loop::buildPairCsv(result));
    stitch::writeTextFileToPath(badStepDiagnosticsCsv.u8string(),
                                pinjie::standard_sphere_loop::buildBadStepDiagnosticsCsv(result, edges));
    stitch::writeTextFileToPath(circlesCsv.u8string(),
                                pinjie::standard_sphere_loop::buildCircleCsv(result,
                                                                             imagePaths,
                                                                             experimentConfig.pixelSizeMm,
                                                                             experimentConfig.sphereDiameterMm));
    stitch::writeTextFileToPath(fieldBiasCsv.u8string(),
                                pinjie::standard_sphere_loop::buildFieldBiasCsv(result, edges, images));
    stitch::writeTextFileToPath(fieldBiasCompensationCsv.u8string(),
                                pinjie::standard_sphere_loop::buildFieldBiasCompensationCsv(result,
                                                                                             edges,
                                                                                             images));
    stitch::writeTextFileToPath(transformsCsv.u8string(),
                                pinjie::standard_sphere_loop::buildTransformCsv(result));
    stitch::writeTextFileToPath(segmentsCsv.u8string(),
                                pinjie::standard_sphere_loop::buildSegmentCsv(result));
    stitch::writeTextFileToPath(summaryCsv.u8string(),
                                pinjie::standard_sphere_loop::buildSummaryCsv(result,
                                                                              edges,
                                                                              images,
                                                                              experimentConfig.pixelSizeMm,
                                                                              experimentConfig.sphereDiameterMm));
    if (!dominantDarkMaskStats.empty()) {
        stitch::writeTextFileToPath(dominantMaskCsv.u8string(),
                                    buildDominantDarkComponentMaskCsv(dominantDarkMaskStats));
    }
    if (!supportMaskStats.empty()) {
        stitch::writeTextFileToPath(supportMaskCsv.u8string(),
                                    buildSupportChangeMaskCsv(supportMaskStats));
    }
    if (!circleCleanupStats.empty()) {
        stitch::writeTextFileToPath(circleCleanupCsv.u8string(),
                                    buildCircleEdgeCleanupCsv(circleCleanupStats));
    }
    const cv::Mat stitchedOverlay =
        pinjie::standard_sphere_loop::buildStitchedVisualization(result, edges, images);
    if (!stitch::saveImageToPath(stitchedOverlayPng.u8string(), stitchedOverlay)) {
        std::cout << "[Warn] Failed to save stitched visualization.\n";
    }

    std::cout << "[Summary] closure_translation_px=" << result.closureTranslationPx
              << ", closure_error_vs_expected_px=" << result.closureResidualTranslationPx
              << ", closure_rotation_deg=" << result.closureRotationDeg
              << ", pre_optimization_closure_translation_px=" << result.preOptimizationClosureTranslationPx
              << ", stitched_circle_rmse_px=" << result.stitchedCircle.rmsePx << "\n";
    std::cout << "[Info] Saved stitched visualization to: "
              << stitchedOverlayPng.generic_string() << "\n";
    std::cout << "[Info] Saved standard sphere loop evaluation to: "
              << outputDir.generic_string() << "\n";
    return 0;
}
