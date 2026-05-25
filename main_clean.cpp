#include "stitch/StitchService.h"
#include "common/ResultPathUtils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace
{

    const std::string kDefaultInputDir = u8"D:/VSCode_Project/pinjie/\u706b\u7130\u7b52/\u6bcd\u7ebf\u62fc\u63a5/";
    constexpr int kDefaultImageCount = 2;

    std::string toLowerAscii(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    bool tryParseDouble(const std::string &text, double &value)
    {
        char *end = nullptr;
        value = std::strtod(text.c_str(), &end);
        return end != text.c_str() && end != nullptr && *end == '\0';
    }

    bool tryParseInt(const std::string &text, int &value)
    {
        char *end = nullptr;
        const long parsed = std::strtol(text.c_str(), &end, 10);
        if (end == text.c_str() || end == nullptr || *end != '\0')
        {
            return false;
        }

        value = static_cast<int>(parsed);
        return true;
    }

    double normalizeUnitRatio(double value)
    {
        if (value > 1.0)
        {
            value /= 100.0;
        }
        return value;
    }

    bool parseDirectionConstraint(const std::string &text, stitch::MotionPriorDirection &direction)
    {
        const std::string normalized = toLowerAscii(text);
        if (normalized == "auto")
        {
            direction = stitch::MotionPriorDirection::Auto;
            return true;
        }
        if (normalized == "x+" || normalized == "xp" || normalized == "xpositive")
        {
            direction = stitch::MotionPriorDirection::XPositive;
            return true;
        }
        if (normalized == "x-" || normalized == "xn" || normalized == "xnegative")
        {
            direction = stitch::MotionPriorDirection::XNegative;
            return true;
        }
        if (normalized == "y+" || normalized == "yp" || normalized == "ypositive")
        {
            direction = stitch::MotionPriorDirection::YPositive;
            return true;
        }
        if (normalized == "y-" || normalized == "yn" || normalized == "ynegative")
        {
            direction = stitch::MotionPriorDirection::YNegative;
            return true;
        }

        return false;
    }

    // 打印当前 CLI 支持的实验参数，便于批处理脚本直接调用。
    void printUsage(const char *argv0)
    {
        std::cout << "Usage:\n"
                  << "  " << argv0 << " <input_dir> <image_count> [--out <out_png>]\n"
                  << "               [--csv <out_csv>] [--debug-dir <process_dir>] [--no-process-vis]\n"
                  << "               [--start-index <n>]\n"
                  << "               [--overlap <ratio_or_percent>] [--direction {auto|x+|x-|y+|y-}]\n"
                  << "               [--search-range <pixels>] [--base-search-range <pixels>]\n"
                  << "               [--rotation-range <deg>] [--rotation-step <deg>]\n"
                  << "               [--rotation-min <deg>] [--rotation-max <deg>]\n"
                  << "               [--tangent-residual-weight <w>] [--tangent-correlation-weight <w>]\n"
                  << "               [--no-point-filter] [--filter-confidence-q <ratio_or_percent>]\n"
                  << "               [--filter-gradient-q <ratio_or_percent>] [--filter-window-radius <points>]\n"
                  << "               [--filter-hampel-sigma <sigma>] [--filter-hampel-min-scale <px>]\n"
                  << "               defaults: --overlap 0.875 --direction x+ --search-range 3000\n"
                  << "                         --rotation-range 1.0 --rotation-step 0.05\n"
                  << "                         tangent weights: residual=0.05, correlation=0.25\n"
                  << "                         point-filter on, q=0.15/0.15, radius=5, sigma=3.0\n\n"
                  << "Example:\n"
                  << "  " << argv0 << " D:/VSCode_Project/pinjie/<input_dir> 2\n";
    }

    std::vector<std::string> collectSequentialImagePaths(const std::filesystem::path &inputDir,
                                                         int imageCount,
                                                         int startIndex)
    {
        std::vector<std::string> paths;
        paths.reserve(static_cast<std::size_t>(std::max(imageCount, 0)));

        for (int i = 0; i < imageCount; ++i)
        {
            const std::filesystem::path imagePath = inputDir / ("Pic_" + std::to_string(startIndex + i) + ".bmp");
            paths.push_back(imagePath.u8string());
        }

        return paths;
    }

} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2)
    {
        const std::string firstArg = argv[1];
        if (firstArg == "--help" || firstArg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::string inputDir = (argc >= 2) ? argv[1] : kDefaultInputDir;
    int imageCount = (argc >= 3) ? std::atoi(argv[2]) : kDefaultImageCount;

    if (imageCount <= 0)
    {
        std::cout << "[Error] image_count must be > 0.\n";
        return -1;
    }

    std::filesystem::path inputPath(inputDir);
    if (!std::filesystem::exists(inputPath) || !std::filesystem::is_directory(inputPath))
    {
        std::cout << "[Error] Invalid input directory: " << inputPath.generic_string() << std::endl;
        return -1;
    }

    // 设置默认结果路径在当前工作目录下的result文件夹中，按类型区分
    bool saveDebugVisualization = true;
    const pinjie::StitchResultPathSet defaultResultPaths =
        pinjie::buildDefaultStitchResultPaths("stitch_app", "workpiece", saveDebugVisualization);
    if (!pinjie::ensureStitchResultDirectories(defaultResultPaths))
    {
        std::cout << "[Error] Failed to create default result directory: "
                  << defaultResultPaths.runDir.generic_string() << std::endl;
        return -1;
    }

    std::filesystem::path panoramaPath = defaultResultPaths.panoramaPath;
    std::filesystem::path csvPath = defaultResultPaths.csvPath;
    std::filesystem::path debugDir = defaultResultPaths.debugDir;
    int startIndex = 1;
    double overlapRatio = 0.875;
    double baseSearchRange = 3000.0;
    double rotationMinDeg = -1.0;
    double rotationMaxDeg = 1.0;
    double rotationStepDeg = 0.05;
    double tangentResidualWeight = 0.05;
    double tangentCorrelationWeight = 0.25;
    bool enablePointFiltering = true;
    double filterConfidenceQuantile = 0.15;
    double filterGradientQuantile = 0.15;
    int filterWindowRadius = 5;
    double filterHampelSigma = 3.0;
    double filterHampelMinScale = 0.05;
    stitch::MotionPriorDirection directionConstraint = stitch::MotionPriorDirection::XPositive;
    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc)
        {
            panoramaPath = argv[++i];
        }
        else if (arg == "--csv" && i + 1 < argc)
        {
            csvPath = argv[++i];
        }
        else if (arg == "--debug-dir" && i + 1 < argc)
        {
            debugDir = argv[++i];
        }
        else if (arg == "--no-process-vis")
        {
            saveDebugVisualization = false;
        }
        else if (arg == "--overlap" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue))
            {
                std::cout << "[Error] Invalid overlap value.\n";
                return -1;
            }

            overlapRatio = normalizeUnitRatio(parsedValue);
            if (overlapRatio < 0.0 || overlapRatio > 1.0)
            {
                std::cout << "[Error] overlap must be within [0, 1] or [0, 100].\n";
                return -1;
            }
        }
        else if (arg == "--start-index" && i + 1 < argc)
        {
            int parsedValue = 0;
            if (!tryParseInt(argv[++i], parsedValue) || parsedValue <= 0)
            {
                std::cout << "[Error] start-index must be a positive integer.\n";
                return -1;
            }

            startIndex = parsedValue;
        }
        else if (arg == "--direction" && i + 1 < argc)
        {
            if (!parseDirectionConstraint(argv[++i], directionConstraint))
            {
                std::cout << "[Error] Invalid direction. Use auto, x+, x-, y+, or y-.\n";
                return -1;
            }
        }
        else if ((arg == "--search-range" || arg == "--base-search-range") && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue <= 0.0)
            {
                std::cout << "[Error] search-range must be a positive number of pixels.\n";
                return -1;
            }

            baseSearchRange = parsedValue;
        }
        else if (arg == "--rotation-range" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue < 0.0)
            {
                std::cout << "[Error] rotation-range must be a non-negative degree value.\n";
                return -1;
            }

            rotationMinDeg = -parsedValue;
            rotationMaxDeg = parsedValue;
        }
        else if (arg == "--rotation-step" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue <= 0.0)
            {
                std::cout << "[Error] rotation-step must be a positive degree value.\n";
                return -1;
            }

            rotationStepDeg = parsedValue;
        }
        else if (arg == "--tangent-residual-weight" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue < 0.0)
            {
                std::cout << "[Error] tangent-residual-weight must be a non-negative value.\n";
                return -1;
            }

            tangentResidualWeight = parsedValue;
        }
        else if (arg == "--tangent-correlation-weight" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue < 0.0)
            {
                std::cout << "[Error] tangent-correlation-weight must be a non-negative value.\n";
                return -1;
            }

            tangentCorrelationWeight = parsedValue;
        }
        else if (arg == "--rotation-min" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue))
            {
                std::cout << "[Error] rotation-min must be a degree value.\n";
                return -1;
            }

            rotationMinDeg = parsedValue;
        }
        else if (arg == "--rotation-max" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue))
            {
                std::cout << "[Error] rotation-max must be a degree value.\n";
                return -1;
            }

            rotationMaxDeg = parsedValue;
        }
        else if (arg == "--no-point-filter")
        {
            enablePointFiltering = false;
        }
        else if (arg == "--filter-confidence-q" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue))
            {
                std::cout << "[Error] filter-confidence-q must be within [0, 1] or [0, 100].\n";
                return -1;
            }

            filterConfidenceQuantile = normalizeUnitRatio(parsedValue);
            if (filterConfidenceQuantile < 0.0 || filterConfidenceQuantile > 1.0)
            {
                std::cout << "[Error] filter-confidence-q must be within [0, 1] or [0, 100].\n";
                return -1;
            }
        }
        else if (arg == "--filter-gradient-q" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue))
            {
                std::cout << "[Error] filter-gradient-q must be within [0, 1] or [0, 100].\n";
                return -1;
            }

            filterGradientQuantile = normalizeUnitRatio(parsedValue);
            if (filterGradientQuantile < 0.0 || filterGradientQuantile > 1.0)
            {
                std::cout << "[Error] filter-gradient-q must be within [0, 1] or [0, 100].\n";
                return -1;
            }
        }
        else if (arg == "--filter-window-radius" && i + 1 < argc)
        {
            int parsedValue = 0;
            if (!tryParseInt(argv[++i], parsedValue) || parsedValue < 1)
            {
                std::cout << "[Error] filter-window-radius must be an integer >= 1.\n";
                return -1;
            }

            filterWindowRadius = parsedValue;
        }
        else if (arg == "--filter-hampel-sigma" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue <= 0.0)
            {
                std::cout << "[Error] filter-hampel-sigma must be a positive number.\n";
                return -1;
            }

            filterHampelSigma = parsedValue;
        }
        else if (arg == "--filter-hampel-min-scale" && i + 1 < argc)
        {
            double parsedValue = 0.0;
            if (!tryParseDouble(argv[++i], parsedValue) || parsedValue <= 0.0)
            {
                std::cout << "[Error] filter-hampel-min-scale must be a positive pixel value.\n";
                return -1;
            }

            filterHampelMinScale = parsedValue;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cout << "[Warn] Unknown or incomplete argument: " << arg << std::endl;
        }
    }

    panoramaPath = pinjie::resolveResultPathUnderBase(panoramaPath, defaultResultPaths.runDir);
    const std::filesystem::path outputDir =
        panoramaPath.parent_path().empty() ? defaultResultPaths.runDir : panoramaPath.parent_path();
    csvPath = pinjie::resolveResultPathUnderBase(csvPath, outputDir);
    if (saveDebugVisualization)
    {
        debugDir = pinjie::resolveResultPathUnderBase(debugDir, outputDir);
    }

    std::error_code dirError;
    std::filesystem::create_directories(outputDir, dirError);
    if (dirError)
    {
        std::cout << "[Error] Failed to create panorama output directory: " << outputDir.generic_string()
                  << std::endl;
        return -1;
    }

    std::filesystem::create_directories(csvPath.parent_path(), dirError);
    if (dirError)
    {
        std::cout << "[Error] Failed to create CSV output directory: " << csvPath.parent_path().generic_string()
                  << std::endl;
        return -1;
    }

    if (saveDebugVisualization)
    {
        std::filesystem::create_directories(debugDir, dirError);
        if (dirError)
        {
            std::cout << "[Error] Failed to create debug output directory: " << debugDir.generic_string()
                      << std::endl;
            return -1;
        }
    }

    stitch::StitchRunRequest request;
    request.imagePaths = collectSequentialImagePaths(inputPath, imageCount, startIndex);
    request.edgeConfig.cannyLow = 50.0;
    request.edgeConfig.cannyHigh = 150.0;
    request.edgeConfig.subpixWindow = 7;
    request.edgeConfig.subpixSigma = 1.0;
    // 先把边缘提取与点级滤波参数写入请求。
    request.edgeConfig.enablePointFiltering = enablePointFiltering;
    request.edgeConfig.filterConfidenceQuantile = filterConfidenceQuantile;
    request.edgeConfig.filterGradientQuantile = filterGradientQuantile;
    request.edgeConfig.filterLocalLinearWindowRadius = filterWindowRadius;
    request.edgeConfig.filterHampelSigma = filterHampelSigma;
    request.edgeConfig.filterHampelMinScale = filterHampelMinScale;
    // 再写入拼接搜索先验与旋转窗口参数。
    request.pipelineConfig.baseSearchRange = baseSearchRange;
    request.pipelineConfig.approxShiftRatio = std::max(0.0, 1.0 - overlapRatio);
    request.pipelineConfig.expectedOverlapRatio = overlapRatio;
    request.pipelineConfig.directionConstraint = directionConstraint;
    request.pipelineConfig.rotationSearchMinDeg = rotationMinDeg;
    request.pipelineConfig.rotationSearchMaxDeg = rotationMaxDeg;
    request.pipelineConfig.rotationSearchStepDeg = rotationStepDeg;
    request.pipelineConfig.tangentResidualCostWeight = tangentResidualWeight;
    request.pipelineConfig.tangentCorrelationCostWeight = tangentCorrelationWeight;
    request.pipelineConfig.generateDebugVisualization = saveDebugVisualization;
    request.resultOutputDir = pinjie::genericUtf8String(outputDir);
    request.panoramaOutputPath = panoramaPath.u8string();
    request.csvOutputPath = csvPath.u8string();
    request.designErrorProfileCsvOutputPath =
        pinjie::genericUtf8String(outputDir / "design_error_profile.csv");
    request.designErrorSummaryCsvOutputPath =
        pinjie::genericUtf8String(outputDir / "design_error_summary.csv");
    request.qualityReviewCsvOutputPath =
        pinjie::genericUtf8String(outputDir / "quality_review.csv");
    request.contourOverlayOutputPath = pinjie::genericUtf8String(outputDir / "origin_contour_overlay.png");
    request.tangentCorrelationAllOutputPath = pinjie::genericUtf8String(outputDir / "tangent_correlation_all.png");
    request.tangentCorrelationInlierOutputPath =
        pinjie::genericUtf8String(outputDir / "tangent_correlation_inlier.png");
    if (saveDebugVisualization)
    {
        request.debugImageOutputDir = pinjie::genericUtf8String(debugDir);
    }

    stitch::StitchCallbacks callbacks;
    callbacks.onLog = [](const std::string &message)
    {
        std::cout << message << std::endl;
    };

    stitch::StitchRunResult result = stitch::runStitching(request, stitch::StitchRunMode::Full, callbacks);
    if (!result.ok)
    {
        if (!result.message.empty())
        {
            std::cout << "[Error] " << result.message << std::endl;
        }
        return -1;
    }

    return 0;
}
