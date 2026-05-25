#include "StandardSphereLoopExperiment.h"

#include "stitch/Alignment.h"
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
    int expectedPointCount{25};
    double windowHalfAngleDeg{4.0};
    double windowHalfSizePx{120.0};
    double confidenceRadiusGuardPx{8.0};
    int rangeOptimizationCandidatesPerField{64};
    int rangeOptimizationRestarts{32};
    double localFieldBiasHalfAngleDeg{2.0};
    int angularBiasOrder{3};
    double angularBiasGain{1.0};
    int selectedPointTransformRefineIterations{4};
    double selectedPointTransformRefineMaxStepPx{0.15};
    double selectedPointTransformRefineGain{0.85};
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
    double viewAngleDeg{0.0};
    double selectedAngleDeg{0.0};
    double angleDeltaDeg{0.0};
    double radiusPx{0.0};
    double residualPx{0.0};
    double qualityWeight{0.0};
    double confidence{0.0};
    double gradient{0.0};
    double fieldBiasPx{0.0};
    double correctedRadiusPx{0.0};
    double correctedResidualPx{0.0};
    int candidateCount{0};
};

struct Gbt57CandidatePoint {
    cv::Point2d imagePoint{};
    cv::Point2d globalPoint{};
    double selectedAngleDeg{0.0};
    double angleDeltaDeg{0.0};
    double radiusDeltaPx{0.0};
    double cost{0.0};
    double qualityWeight{0.0};
    double confidence{0.0};
    double gradient{0.0};
};

struct Gbt57P2dResult {
    bool ok{false};
    std::string message;
    CircleModel globalCircle;
    CircleModel selectedCircle;
    std::vector<Gbt57P2dPoint> points;
    double eP2dPx{0.0};
    double eP2dUm{0.0};
    double minRadiusPx{0.0};
    double maxRadiusPx{0.0};
    double rmsePx{0.0};
    double meanAbsPx{0.0};
    double angleSpacingMeanDeg{0.0};
    double angleSpacingMinDeg{0.0};
    double angleSpacingMaxDeg{0.0};
    double angleSpacingMaxErrorDeg{0.0};
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
        << "  --pixel-size <mm_per_px>              Convert pixel metrics to millimeters.\n"
        << "  --sphere-diameter <mm>                Standard sphere diameter for diameter error.\n"
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
        << "  --gbt57-confidence-radius-guard <px>  Radius guard for confidence selection, default 8.\n"
        << "  --gbt57-radius-stable-selection       Prefer points closest to the fitted global radius.\n"
        << "  --gbt57-uniform-angle-selection       Assign 25 preset angles to fields before point selection.\n"
        << "  --gbt57-optimize-selected-range       Locally minimize 25-point Rmax-Rmin within preset windows.\n"
        << "  --gbt57-range-candidates <n>          Candidates kept per field for range optimization, default 64.\n"
        << "  --gbt57-range-restarts <n>            Deterministic restarts for range optimization, default 32.\n"
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
        << "  --gbt57-local-bias-half-angle <deg>   Local bias angular half window, default 2.\n"
        << "  --gbt57-window-half-angle <deg>       Single-point selection angular half window, default 4.\n"
        << "  --gbt57-window-half-size <px>         Reported measurement-window half size, default 120.\n";
}

std::vector<std::string> collectSequentialImagePaths(const std::filesystem::path& inputDir,
                                                     int imageCount,
                                                     int startIndex,
                                                     const std::string& prefix,
                                                     const std::string& extension)
{
    std::vector<std::string> paths;
    paths.reserve(static_cast<std::size_t>(std::max(imageCount, 0)));
    for (int i = 0; i < imageCount; ++i) {
        const std::filesystem::path imagePath =
            inputDir / (prefix + std::to_string(startIndex + i) + extension);
        paths.push_back(imagePath.u8string());
    }
    return paths;
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

Gbt57P2dResult evaluateGbt57P2dSinglePoint(
    const std::vector<cv::Mat>& images,
    const std::vector<stitch::EdgeVariants>& edges,
    const stitch::StitchingResult& stitching,
    const std::vector<std::string>& imagePaths,
    const Gbt57P2dConfig& config,
    const pinjie::standard_sphere_loop::StandardSphereLoopConfig& experimentConfig)
{
    Gbt57P2dResult result;
    if (!config.enabled) {
        result.message = "GB/T 5.7 P2D mode is disabled";
        return result;
    }
    if (images.empty() || edges.size() != images.size() ||
        stitching.imageTransforms.size() < images.size()) {
        result.message = "GB/T 5.7 P2D needs images, edges, and stitching transforms";
        return result;
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
                transformPointByMatrix(stitching.imageTransforms[imageIndex], point));
        }
    }

    result.globalCircle = fitCircleRobustForP2d(globalEdgePoints);
    if (!result.globalCircle.ok) {
        result.message = "failed to fit stitched global standard-sphere circle";
        return result;
    }

    const double windowHalfAngle =
        std::max(0.2, config.windowHalfAngleDeg);
    const double relaxedHalfAngle =
        std::max(windowHalfAngle, 0.50 * 360.0 /
                                      std::max(1, config.expectedPointCount));
    const double rangeOptimizationStoreHalfAngle =
        config.optimizeSelectedRange ? std::max(windowHalfAngle, 30.0) : windowHalfAngle;

    std::vector<CircleModel> localCircles;
    localCircles.reserve(edges.size());
    for (const stitch::EdgeVariants& edge : edges) {
        localCircles.push_back(fitCircleRobustForP2d(edge.raw));
    }

    std::vector<std::vector<Gbt57CandidatePoint>> perImageCandidates(images.size());
    std::vector<double> targetAngles(images.size(), 0.0);
    if (config.uniformAngleSelection && images.size() == static_cast<std::size_t>(
            std::max(1, config.expectedPointCount))) {
        struct ViewAngleRecord {
            std::size_t index{0};
            double angleDeg{0.0};
        };
        std::vector<ViewAngleRecord> viewAngles;
        viewAngles.reserve(images.size());
        for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
            double sumSin = 0.0;
            double sumCos = 0.0;
            double angleWeightSum = 0.0;
            const bool hasWeights =
                edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
            for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
                const cv::Point2d global =
                    transformPointByMatrix(stitching.imageTransforms[imageIndex],
                                           edges[imageIndex].raw[pointIndex]);
                const double angleRad =
                    angleOfPointDeg(global, result.globalCircle) * CV_PI / 180.0;
                const double weight =
                    hasWeights ? std::clamp(edges[imageIndex].rawQualityWeights[pointIndex],
                                            0.05,
                                            4.0)
                               : 1.0;
                sumSin += weight * std::sin(angleRad);
                sumCos += weight * std::cos(angleRad);
                angleWeightSum += weight;
            }
            if (angleWeightSum > 0.0) {
                viewAngles.push_back({
                    imageIndex,
                    normalizeAngleDeg(std::atan2(sumSin, sumCos) * 180.0 / CV_PI)
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

            const double stepDeg = 360.0 / static_cast<double>(config.expectedPointCount);
            const double startAngle =
                normalizeAngleDeg(ordered.front().angleDeg -
                                  0.5 * circularDeltaDeg(ordered.front().angleDeg,
                                                         ordered.back().angleDeg));
            for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
                targetAngles[ordered[rank].index] =
                    normalizeAngleDeg(startAngle + static_cast<double>(rank) * stepDeg);
            }
        }
    }

    result.points.reserve(images.size());
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        Gbt57P2dPoint selected;
        selected.imageIndex = static_cast<int>(imageIndex);
        selected.imageName =
            imageIndex < imagePaths.size()
                ? std::filesystem::path(imagePaths[imageIndex]).filename().u8string()
                : std::string{};

        double sumSin = 0.0;
        double sumCos = 0.0;
        double angleWeightSum = 0.0;
        const bool hasWeights =
            edges[imageIndex].rawQualityWeights.size() == edges[imageIndex].raw.size();
        for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
            const cv::Point2d global =
                transformPointByMatrix(stitching.imageTransforms[imageIndex],
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
        if (config.uniformAngleSelection) {
            selected.viewAngleDeg = targetAngles[imageIndex];
        }

        const cv::Point2d imageCenter(images[imageIndex].cols * 0.5,
                                      images[imageIndex].rows * 0.5);
        const double imageDiag =
            std::max(1.0,
                     std::hypot(static_cast<double>(images[imageIndex].cols),
                                static_cast<double>(images[imageIndex].rows)));
        const bool hasConfidences =
            edges[imageIndex].rawConfidences.size() == edges[imageIndex].raw.size();
        const bool hasGradients =
            edges[imageIndex].rawGradients.size() == edges[imageIndex].raw.size();

        double bestCost = std::numeric_limits<double>::infinity();
        const auto testCandidate = [&](double halfAngle, bool& found) {
            for (std::size_t pointIndex = 0; pointIndex < edges[imageIndex].raw.size(); ++pointIndex) {
                const cv::Point2d imagePoint = edges[imageIndex].raw[pointIndex];
                const cv::Point2d global =
                    transformPointByMatrix(stitching.imageTransforms[imageIndex], imagePoint);
                const double angle = angleOfPointDeg(global, result.globalCircle);
                const double angleDelta = circularAbsDeltaDeg(angle, selected.viewAngleDeg);
                if (angleDelta > halfAngle) {
                    continue;
                }

                const double quality =
                    hasWeights ? std::clamp(edges[imageIndex].rawQualityWeights[pointIndex],
                                            0.05,
                                            4.0)
                               : 1.0;
                const double confidence =
                    hasConfidences ? edges[imageIndex].rawConfidences[pointIndex] : 0.0;
                const double gradient =
                    hasGradients ? edges[imageIndex].rawGradients[pointIndex] : 0.0;
                const double centerCost = cv::norm(imagePoint - imageCenter) / imageDiag;
                const double confidenceBonus = std::clamp(confidence / 10.0, 0.0, 1.0);
                const double gradientBonus = std::clamp(gradient / 80.0, 0.0, 1.0);
                const double radius =
                    std::hypot(global.x - result.globalCircle.centerX,
                               global.y - result.globalCircle.centerY);
                const double radiusDelta =
                    std::abs(radius - result.globalCircle.radius);
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
                        0.28 * angleDelta / std::max(0.1, halfAngle) +
                        0.08 * centerCost +
                        0.12 * localRadiusDelta / radiusGuard -
                        1.00 * confidenceBonus -
                        0.65 * gradientBonus -
                        0.35 * qualityScore;
                } else if (config.radiusStableSelection) {
                    cost =
                        radiusDelta +
                        0.02 * angleDelta / std::max(0.1, halfAngle) +
                        0.02 * centerCost -
                        0.03 * std::log1p(quality) -
                        0.02 * confidenceBonus -
                        0.02 * gradientBonus;
                } else {
                    cost =
                        angleDelta / std::max(0.1, halfAngle) +
                        0.20 * centerCost -
                        0.08 * std::log1p(quality) -
                        0.05 * confidenceBonus -
                        0.03 * gradientBonus;
                }

                ++selected.candidateCount;
                if (halfAngle <= rangeOptimizationStoreHalfAngle + 1e-9 ||
                    perImageCandidates[imageIndex].empty()) {
                    perImageCandidates[imageIndex].push_back({
                        imagePoint,
                        global,
                        angle,
                        angleDelta,
                        radiusDelta,
                        cost,
                        quality,
                        confidence,
                        gradient
                    });
                }
                if (cost < bestCost) {
                    bestCost = cost;
                    selected.selected = true;
                    selected.imagePoint = imagePoint;
                    selected.globalPoint = global;
                    selected.selectedAngleDeg = angle;
                    selected.angleDeltaDeg = angleDelta;
                    selected.qualityWeight = quality;
                    selected.confidence = confidence;
                    selected.gradient = gradient;
                    found = true;
                }
            }
        };

        bool found = false;
        testCandidate(windowHalfAngle, found);
        if (!found) {
            testCandidate(relaxedHalfAngle, found);
        }
        if (!found) {
            testCandidate(180.0, found);
        }
        result.points.push_back(selected);
    }

    if (config.optimizeSelectedRange &&
        result.points.size() == perImageCandidates.size()) {
        for (std::vector<Gbt57CandidatePoint>& candidates : perImageCandidates) {
            std::sort(candidates.begin(),
                      candidates.end(),
                      [](const Gbt57CandidatePoint& a, const Gbt57CandidatePoint& b) {
                          if (std::abs(a.radiusDeltaPx - b.radiusDeltaPx) > 1e-9) {
                              return a.radiusDeltaPx < b.radiusDeltaPx;
                          }
                          return a.cost < b.cost;
                      });
            const std::size_t keep =
                static_cast<std::size_t>(std::max(1, config.rangeOptimizationCandidatesPerField));
            if (candidates.size() > keep) {
                candidates.resize(keep);
            }
        }

        const auto rangeCostForPoints = [](const std::vector<Gbt57P2dPoint>& points,
                                           CircleModel* selectedCircleOut = nullptr) {
            std::vector<cv::Point2d> globals;
            globals.reserve(points.size());
            for (const Gbt57P2dPoint& point : points) {
                if (!point.selected) {
                    return std::numeric_limits<double>::infinity();
                }
                globals.push_back(point.globalPoint);
            }
            double rangePx = 0.0;
            const CircleModel circle = fitCircleMinRange(globals, &rangePx);
            if (!circle.ok) {
                return std::numeric_limits<double>::infinity();
            }
            if (selectedCircleOut != nullptr) {
                *selectedCircleOut = circle;
            }
            return rangePx;
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
        };

        const auto refineSelection =
            [&](std::vector<Gbt57P2dPoint>& points, double currentRange) {
                for (int pass = 0; pass < 6; ++pass) {
                    bool improved = false;
                    for (std::size_t imageIndex = 0; imageIndex < points.size(); ++imageIndex) {
                        Gbt57P2dPoint bestPoint = points[imageIndex];
                        double bestLocalRange = currentRange;
                        for (const Gbt57CandidatePoint& candidate :
                             perImageCandidates[imageIndex]) {
                            std::vector<Gbt57P2dPoint> trialPoints = points;
                            assignCandidateToPoint(trialPoints[imageIndex], candidate);
                            const double trialRange = rangeCostForPoints(trialPoints);
                            const double costGuard =
                                candidate.cost -
                                (perImageCandidates[imageIndex].empty()
                                     ? candidate.cost
                                     : perImageCandidates[imageIndex].front().cost);
                            if (trialRange + 0.002 * std::max(0.0, costGuard) + 1e-9 <
                                bestLocalRange) {
                                bestLocalRange = trialRange;
                                bestPoint = trialPoints[imageIndex];
                            }
                        }
                        if (bestLocalRange + 1e-9 < currentRange) {
                            points[imageIndex] = bestPoint;
                            currentRange = bestLocalRange;
                            improved = true;
                        }
                    }
                    if (!improved) {
                        break;
                    }
                }
                return currentRange;
            };

        const std::vector<Gbt57P2dPoint> basePoints = result.points;
        std::vector<Gbt57P2dPoint> bestPoints = result.points;
        double bestRange = rangeCostForPoints(bestPoints);
        bestRange = refineSelection(bestPoints, bestRange);

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
            double trialRange = rangeCostForPoints(trialPoints);
            if (!std::isfinite(trialRange)) {
                continue;
            }
            trialRange = refineSelection(trialPoints, trialRange);
            if (trialRange + 1e-9 < bestRange) {
                bestRange = trialRange;
                bestPoints = std::move(trialPoints);
            }
        }

        result.points = std::move(bestPoints);
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

    if (config.selectedPointTransformRefine &&
        config.selectedPointTransformRefineIterations > 0 &&
        config.selectedPointTransformRefineMaxStepPx > 0.0 &&
        config.selectedPointTransformRefineGain > 0.0) {
        for (int iteration = 0; iteration < config.selectedPointTransformRefineIterations; ++iteration) {
            std::vector<cv::Point2d> refinedGlobalPoints;
            refinedGlobalPoints.reserve(result.points.size());
            for (const Gbt57P2dPoint& point : result.points) {
                refinedGlobalPoints.push_back(point.globalPoint);
            }

            const CircleModel refinedCircle = fitCircleMinRange(refinedGlobalPoints);
            if (!refinedCircle.ok) {
                break;
            }

            double maxApplied = 0.0;
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                cv::Point2d& global = result.points[i].globalPoint;
                const double dx = global.x - refinedCircle.centerX;
                const double dy = global.y - refinedCircle.centerY;
                const double radius = std::hypot(dx, dy);
                if (radius <= 1e-9 || !std::isfinite(radius)) {
                    continue;
                }
                const double residual = radius - refinedCircle.radius;
                if (!std::isfinite(residual)) {
                    continue;
                }
                cv::Point2d correction{
                    -config.selectedPointTransformRefineGain * residual * dx / radius,
                    -config.selectedPointTransformRefineGain * residual * dy / radius
                };
                const double correctionNorm = cv::norm(correction);
                if (!std::isfinite(correctionNorm) || correctionNorm <= 1e-12) {
                    continue;
                }
                if (correctionNorm > config.selectedPointTransformRefineMaxStepPx) {
                    correction *= config.selectedPointTransformRefineMaxStepPx / correctionNorm;
                }
                global += correction;
                maxApplied = std::max(maxApplied, cv::norm(correction));
            }

            if (maxApplied < 1e-4) {
                break;
            }
        }

        selectedGlobalPoints.clear();
        selectedGlobalPoints.reserve(result.points.size());
        for (Gbt57P2dPoint& point : result.points) {
            point.selectedAngleDeg =
                angleOfPointDeg(point.globalPoint, result.selectedCircle);
            point.angleDeltaDeg =
                circularAbsDeltaDeg(point.selectedAngleDeg, point.viewAngleDeg);
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
            point.angleDeltaDeg =
                circularAbsDeltaDeg(point.selectedAngleDeg, point.viewAngleDeg);
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
                    transformPointByMatrix(stitching.imageTransforms[imageIndex], imagePoint);
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
    if (config.ellipseNormalizationCompensation && selectedGlobalPoints.size() >= 5) {
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

    double sumAbs = 0.0;
    double sumSq = 0.0;
    result.minRadiusPx = std::numeric_limits<double>::infinity();
    result.maxRadiusPx = 0.0;
    const bool biasCompensationEnabled =
        config.fieldBiasCompensation ||
        config.localFieldBiasCompensation ||
        config.angularBiasCompensation ||
        config.ellipseNormalizationCompensation;
    for (std::size_t i = 0; i < result.points.size(); ++i) {
        Gbt57P2dPoint& point = result.points[i];
        point.radiusPx =
            std::hypot(point.globalPoint.x - result.selectedCircle.centerX,
                       point.globalPoint.y - result.selectedCircle.centerY);
        point.residualPx = point.radiusPx - result.selectedCircle.radius;
        point.fieldBiasPx = i < fieldBiases.size() ? fieldBiases[i] : 0.0;
        point.correctedRadiusPx =
            biasCompensationEnabled
                ? point.radiusPx - point.fieldBiasPx
                : point.radiusPx;
        point.correctedResidualPx =
            point.correctedRadiusPx - result.selectedCircle.radius;
        result.minRadiusPx = std::min(result.minRadiusPx, point.correctedRadiusPx);
        result.maxRadiusPx = std::max(result.maxRadiusPx, point.correctedRadiusPx);
        sumAbs += std::abs(point.correctedResidualPx);
        sumSq += point.correctedResidualPx * point.correctedResidualPx;
    }

    const double count = static_cast<double>(result.points.size());
    result.eP2dPx = result.maxRadiusPx - result.minRadiusPx;
    result.meanAbsPx = sumAbs / std::max(1.0, count);
    result.rmsePx = std::sqrt(sumSq / std::max(1.0, count));
    result.eP2dUm = result.eP2dPx * effectivePixelSizeUm(images, experimentConfig);

    std::vector<double> angles;
    angles.reserve(result.points.size());
    for (const Gbt57P2dPoint& point : result.points) {
        angles.push_back(point.selectedAngleDeg);
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
        result.angleSpacingMeanDeg =
            std::accumulate(spacings.begin(), spacings.end(), 0.0) /
            static_cast<double>(spacings.size());
        result.angleSpacingMinDeg =
            *std::min_element(spacings.begin(), spacings.end());
        result.angleSpacingMaxDeg =
            *std::max_element(spacings.begin(), spacings.end());
        for (double spacing : spacings) {
            result.angleSpacingMaxErrorDeg =
                std::max(result.angleSpacingMaxErrorDeg,
                         std::abs(spacing - idealSpacing));
        }
    }

    result.ok = true;
    result.message = "ok";
    if (syntheticPointCount > 0) {
        result.message += " (synthetic edge points excluded: " + std::to_string(syntheticPointCount) + ")";
    }
    return result;
}

}

std::string buildGbt57P2dPointCsv(const Gbt57P2dResult& result)
{
    std::ostringstream stream;
    stream << "image_index,image_name,selected,image_x_px,image_y_px,global_x_px,global_y_px,"
              "view_angle_deg,selected_angle_deg,angle_delta_deg,radius_px,residual_px,"
              "field_bias_px,corrected_radius_px,corrected_residual_px,"
              "quality_weight,confidence,gradient,candidate_count\n";
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
               << point.radiusPx << ","
               << point.residualPx << ","
               << point.fieldBiasPx << ","
               << point.correctedRadiusPx << ","
               << point.correctedResidualPx << ","
               << point.qualityWeight << ","
               << point.confidence << ","
               << point.gradient << ","
               << point.candidateCount << "\n";
    }
    return stream.str();
}

std::string buildGbt57P2dSummaryCsv(const Gbt57P2dResult& result,
                                    const std::vector<cv::Mat>& images,
                                    const Gbt57P2dConfig& config,
                                    const pinjie::standard_sphere_loop::StandardSphereLoopConfig& experimentConfig)
{
    const double pixelSizeUm = effectivePixelSizeUm(images, experimentConfig);
    auto appendMetric = [](std::ostringstream& stream,
                           const std::string& name,
                           double value,
                           const std::string& unit,
                           const std::string& note = {}) {
        stream << name << ","
               << std::setprecision(12) << value << ","
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
                 "fields are assigned to 25 preset angles sorted by their observed view angles");
    appendMetric(stream,
                 "gbt57_optimize_selected_range",
                 config.optimizeSelectedRange ? 1.0 : 0.0,
                 "bool",
                 "after fixed preset windows, selected points are locally swapped to reduce Rmax-Rmin");
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
                 "selected radii are normalized by a covariance ellipse model fitted to the 25 selected points");
    appendMetric(stream,
                 "gbt57_selected_point_transform_refine",
                 config.selectedPointTransformRefine ? 1.0 : 0.0,
                 "bool",
                 "per-field transforms are refined by the 25 selected measurement points");
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
    appendMetric(stream, "expected_point_count", static_cast<double>(config.expectedPointCount), "count");
    const int selectedPointCount =
        static_cast<int>(std::count_if(result.points.begin(),
                                       result.points.end(),
                                       [](const Gbt57P2dPoint& point) {
                                           return point.selected;
                                       }));
    appendMetric(stream, "selected_point_count", static_cast<double>(selectedPointCount), "count");
    appendMetric(stream, "pixel_size", pixelSizeUm, "um/px", "derived from configured field of view unless --pixel-size is supplied");
    appendMetric(stream, "global_circle_center_x", result.globalCircle.centerX, "px");
    appendMetric(stream, "global_circle_center_y", result.globalCircle.centerY, "px");
    appendMetric(stream, "global_circle_radius", result.globalCircle.radius, "px");
    appendMetric(stream, "selected_circle_center_x", result.selectedCircle.centerX, "px");
    appendMetric(stream, "selected_circle_center_y", result.selectedCircle.centerY, "px");
    appendMetric(stream, "selected_circle_radius", result.selectedCircle.radius, "px");
    appendMetric(stream, "e_p2d", result.eP2dPx, "px", "Rmax-Rmin of 25 single points");
    appendMetric(stream, "e_p2d", result.eP2dUm, "um", "Rmax-Rmin of 25 single points");
    appendMetric(stream, "min_radius", result.minRadiusPx, "px");
    appendMetric(stream, "max_radius", result.maxRadiusPx, "px");
    appendMetric(stream, "single_point_rmse", result.rmsePx, "px");
    appendMetric(stream, "single_point_rmse", result.rmsePx * pixelSizeUm, "um");
    appendMetric(stream, "single_point_mean_abs", result.meanAbsPx, "px");
    appendMetric(stream, "single_point_mean_abs", result.meanAbsPx * pixelSizeUm, "um");
    appendMetric(stream, "angle_spacing_mean", result.angleSpacingMeanDeg, "deg");
    appendMetric(stream, "angle_spacing_min", result.angleSpacingMinDeg, "deg");
    appendMetric(stream, "angle_spacing_max", result.angleSpacingMaxDeg, "deg");
    appendMetric(stream, "angle_spacing_max_error", result.angleSpacingMaxErrorDeg, "deg", "vs 14.4 deg for 25 points");
    appendMetric(stream, "window_half_angle", config.windowHalfAngleDeg, "deg");
    appendMetric(stream, "window_half_size", config.windowHalfSizePx, "px", "reported window size for traceability");
    appendMetric(stream, "confidence_radius_guard", config.confidenceRadiusGuardPx, "px");
    appendMetric(stream, "circle_center_angle_search_range", config.centerGlobalAngleSearchRangeDeg, "deg");
    appendMetric(stream, "circle_center_angle_search_step", config.centerGlobalAngleSearchStepDeg, "deg");
    return stream.str();
}

cv::Mat buildGbt57P2dOverlay(const cv::Mat& canvas,
                             const Gbt57P2dResult& result)
{
    cv::Mat overlay;
    if (canvas.empty()) {
        return overlay;
    }
    if (canvas.channels() == 1) {
        cv::cvtColor(canvas, overlay, cv::COLOR_GRAY2BGR);
    } else {
        overlay = canvas.clone();
    }

    if (result.globalCircle.ok) {
        cv::circle(overlay,
                   cv::Point(static_cast<int>(std::lround(result.globalCircle.centerX)),
                             static_cast<int>(std::lround(result.globalCircle.centerY))),
                   static_cast<int>(std::lround(result.globalCircle.radius)),
                   cv::Scalar(70, 180, 255),
                   2,
                   cv::LINE_AA);
    }

    for (const Gbt57P2dPoint& point : result.points) {
        if (!point.selected) {
            continue;
        }
        const cv::Point center(static_cast<int>(std::lround(point.globalPoint.x)),
                               static_cast<int>(std::lround(point.globalPoint.y)));
        cv::circle(overlay, center, 9, cv::Scalar(40, 220, 80), 2, cv::LINE_AA);
        cv::putText(overlay,
                    std::to_string(point.imageIndex + 1),
                    center + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(30, 30, 255),
                    1,
                    cv::LINE_AA);
    }
    return overlay;
}

cv::Scalar overlayPaletteColor(std::size_t index)
{
    static const std::array<cv::Scalar, 12> colors = {
        cv::Scalar(230, 57, 70),
        cv::Scalar(29, 53, 87),
        cv::Scalar(42, 157, 143),
        cv::Scalar(244, 162, 97),
        cv::Scalar(131, 56, 236),
        cv::Scalar(255, 0, 110),
        cv::Scalar(58, 134, 255),
        cv::Scalar(0, 150, 136),
        cv::Scalar(255, 183, 3),
        cv::Scalar(90, 24, 154),
        cv::Scalar(76, 175, 80),
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

    cv::Mat overlay(stitching.canvas.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (std::size_t imageIndex = 0; imageIndex < edges.size(); ++imageIndex) {
        const cv::Scalar color = overlayPaletteColor(imageIndex);
        for (const cv::Point2d& point : edges[imageIndex].raw) {
            const cv::Point2d global =
                transformPointByMatrix(stitching.imageTransforms[imageIndex], point);
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
        cv::circle(overlay, center, 8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(overlay,
                    std::to_string(point.imageIndex + 1),
                    center + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 0, 180),
                    1,
                    cv::LINE_AA);
    }

    return overlay;
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
    CircleEdgeCleanupConfig circleCleanupConfig;
    Gbt57P2dConfig gbt57P2dConfig;
    bool edgeCleanupOnly = false;
    bool overlapExplicit = false;
    bool directionExplicit = false;
    bool segmentCountsExplicit = false;

    stitch::EdgeDetectConfig edgeConfig;
    edgeConfig.filterHampelSigma = 3.0;

    pinjie::standard_sphere_loop::StandardSphereLoopConfig experimentConfig;
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
        } else if (arg == "--reverse-order") {
            reverseOrder = true;
        } else if (arg == "--pixel-size" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.pixelSizeMm) ||
                experimentConfig.pixelSizeMm <= 0.0) {
                std::cout << "[Error] --pixel-size must be a positive mm/px value.\n";
                return 1;
            }
        } else if (arg == "--sphere-diameter" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], experimentConfig.sphereDiameterMm) ||
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
        } else if (arg == "--gbt57-selected-point-refine-iter" && i + 1 < argc) {
            if (!tryParseInt(argv[++i], gbt57P2dConfig.selectedPointTransformRefineIterations) ||
                gbt57P2dConfig.selectedPointTransformRefineIterations < 0) {
                std::cout << "[Error] --gbt57-selected-point-refine-iter must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-selected-point-refine-step" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineMaxStepPx) ||
                gbt57P2dConfig.selectedPointTransformRefineMaxStepPx < 0.0) {
                std::cout << "[Error] --gbt57-selected-point-refine-step must be non-negative.\n";
                return 1;
            }
        } else if (arg == "--gbt57-selected-point-refine-gain" && i + 1 < argc) {
            if (!tryParseDouble(argv[++i], gbt57P2dConfig.selectedPointTransformRefineGain) ||
                gbt57P2dConfig.selectedPointTransformRefineGain < 0.0) {
                std::cout << "[Error] --gbt57-selected-point-refine-gain must be non-negative.\n";
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

    std::filesystem::create_directories(outputDir);

    std::vector<std::string> imagePaths =
        collectSequentialImagePaths(inputDir, imageCount, startIndex, prefix, extension);
    if (reverseOrder) {
        std::reverse(imagePaths.begin(), imagePaths.end());
    }

    stitch::StitchCallbacks callbacks;
    callbacks.onLog = [](const std::string& message) {
        std::cout << message << "\n";
    };

    std::cout << "[Info] Loading " << imagePaths.size() << " standard sphere images...\n";
    const std::vector<cv::Mat> images = stitch::loadInputImages(imagePaths, callbacks);
    if (images.size() != imagePaths.size()) {
        std::cout << "[Error] Failed to load all input images.\n";
        return 1;
    }

    std::cout << "[Info] Preprocessing edges with existing stitch pipeline settings...\n";
    std::vector<stitch::EdgeVariants> edges = stitch::preprocessAllImages(images, edgeConfig, callbacks);
    if (edges.size() != images.size()) {
        std::cout << "[Error] Edge preprocessing failed.\n";
        return 1;
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
        const std::filesystem::path supportMaskCsv =
            outputDir / "standard_sphere_support_change_mask.csv";
        const std::filesystem::path circleCleanupCsv =
            outputDir / "standard_sphere_circle_edge_cleanup.csv";
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

    if (gbt57P2dConfig.enabled) {
        const Gbt57P2dResult p2dResult =
            evaluateGbt57P2dSinglePoint(images,
                                        edges,
                                        stitching,
                                        imagePaths,
                                        gbt57P2dConfig,
                                        experimentConfig);
        const std::filesystem::path p2dSummaryCsv =
            outputDir / "standard_sphere_gbt57_p2d_summary.csv";
        const std::filesystem::path p2dPointsCsv =
            outputDir / "standard_sphere_gbt57_p2d_points.csv";
        const std::filesystem::path p2dOverlayPng =
            outputDir / "standard_sphere_gbt57_p2d_overlay.png";
        const std::filesystem::path p2dEdgeOverlayPng =
            outputDir / "standard_sphere_gbt57_p2d_edge_overlay.png";
        const std::filesystem::path supportMaskCsv =
            outputDir / "standard_sphere_support_change_mask.csv";
        const std::filesystem::path circleCleanupCsv =
            outputDir / "standard_sphere_circle_edge_cleanup.csv";

        stitch::writeTextFileToPath(p2dSummaryCsv.u8string(),
                                    buildGbt57P2dSummaryCsv(p2dResult,
                                                            images,
                                                            gbt57P2dConfig,
                                                            experimentConfig));
        stitch::writeTextFileToPath(p2dPointsCsv.u8string(),
                                    buildGbt57P2dPointCsv(p2dResult));
        const cv::Mat p2dOverlay = buildGbt57P2dOverlay(stitching.canvas, p2dResult);
        if (!p2dOverlay.empty()) {
            stitch::saveImageToPath(p2dOverlayPng.u8string(), p2dOverlay);
        }
        const cv::Mat p2dEdgeOverlay =
            buildGbt57P2dEdgeOverlay(stitching, edges, p2dResult);
        if (!p2dEdgeOverlay.empty()) {
            stitch::saveImageToPath(p2dEdgeOverlayPng.u8string(), p2dEdgeOverlay);
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

        const double pixelSizeUm = effectivePixelSizeUm(images, experimentConfig);
        std::cout << "[Summary] gbt57_e_p2d_px=" << p2dResult.eP2dPx
                  << ", gbt57_e_p2d_um=" << p2dResult.eP2dPx * pixelSizeUm
                  << ", single_point_rmse_px=" << p2dResult.rmsePx
                  << ", angle_spacing_max_error_deg="
                  << p2dResult.angleSpacingMaxErrorDeg << "\n";
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
