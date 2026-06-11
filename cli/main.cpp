#include "app/StitchWorkflowService.h"
#include "common/ResultPathUtils.h"
#include "registration/RegistrationTypes.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
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

    enum class CliPreset
    {
        Gui,
        PinjieCliLegacy,
        StitchAppLegacy,
    };

    enum class ImageCollectionMode
    {
        SequentialPicBmp,
        ScanAllNatural,
    };

    enum class CliRunMode
    {
        Full,
        Registration,
    };

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

    bool parseDirectionConstraint(const std::string &text, pinjie::MotionPriorDirection &direction)
    {
        const std::string normalized = toLowerAscii(text);
        if (normalized == "auto")
        {
            direction = pinjie::MotionPriorDirection::Auto;
            return true;
        }
        if (normalized == "x+" || normalized == "xp" || normalized == "xpositive")
        {
            direction = pinjie::MotionPriorDirection::XPositive;
            return true;
        }
        if (normalized == "x-" || normalized == "xn" || normalized == "xnegative")
        {
            direction = pinjie::MotionPriorDirection::XNegative;
            return true;
        }
        if (normalized == "y+" || normalized == "yp" || normalized == "ypositive")
        {
            direction = pinjie::MotionPriorDirection::YPositive;
            return true;
        }
        if (normalized == "y-" || normalized == "yn" || normalized == "ynegative")
        {
            direction = pinjie::MotionPriorDirection::YNegative;
            return true;
        }

        return false;
    }

    bool parseCliPreset(const std::string &text, CliPreset &preset)
    {
        const std::string normalized = toLowerAscii(text);
        if (normalized == "gui" || normalized == "gui-like" || normalized == "gui_compatible")
        {
            preset = CliPreset::Gui;
            return true;
        }
        if (normalized == "pinjie-cli" || normalized == "pinjie_cli" || normalized == "cli-legacy")
        {
            preset = CliPreset::PinjieCliLegacy;
            return true;
        }
        if (normalized == "stitch-app" || normalized == "stitch_app" || normalized == "legacy-cli")
        {
            preset = CliPreset::StitchAppLegacy;
            return true;
        }

        return false;
    }

    bool parseCliRunMode(const std::string &text, CliRunMode &mode)
    {
        const std::string normalized = toLowerAscii(text);
        if (normalized == "full")
        {
            mode = CliRunMode::Full;
            return true;
        }
        if (normalized == "registration" || normalized == "register" || normalized == "reg")
        {
            mode = CliRunMode::Registration;
            return true;
        }

        return false;
    }

    const char *cliPresetName(CliPreset preset)
    {
        switch (preset)
        {
        case CliPreset::Gui:
            return "gui";
        case CliPreset::PinjieCliLegacy:
            return "pinjie-cli";
        case CliPreset::StitchAppLegacy:
            return "stitch-app";
        }

        return "gui";
    }

    const char *cliRunModeName(CliRunMode mode)
    {
        switch (mode)
        {
        case CliRunMode::Registration:
            return "registration";
        case CliRunMode::Full:
        default:
            return "full";
        }
    }

    void applyCliPreset(CliPreset preset,
                        double &overlapRatio,
                        double &baseSearchRange,
                        double &rotationMinDeg,
                        double &rotationMaxDeg,
                        double &rotationStepDeg,
                        double &tangentResidualWeight,
                        double &tangentCorrelationWeight,
                        bool &enablePointFiltering,
                        double &filterConfidenceQuantile,
                        double &filterGradientQuantile,
                        int &filterWindowRadius,
                        double &filterHampelSigma,
                        double &filterHampelMinScale,
                        pinjie::MotionPriorDirection &directionConstraint,
                        ImageCollectionMode &imageCollectionMode)
    {
        overlapRatio = 0.875;
        tangentResidualWeight = 0.05;
        tangentCorrelationWeight = 0.25;
        enablePointFiltering = true;
        filterConfidenceQuantile = 0.15;
        filterGradientQuantile = 0.15;
        filterWindowRadius = 5;
        filterHampelSigma = 3.0;
        filterHampelMinScale = 0.05;
        directionConstraint = pinjie::MotionPriorDirection::XPositive;

        switch (preset)
        {
        case CliPreset::Gui:
            baseSearchRange = 200.0;
            rotationMinDeg = -0.5;
            rotationMaxDeg = 0.5;
            rotationStepDeg = 0.01;
            imageCollectionMode = ImageCollectionMode::ScanAllNatural;
            break;
        case CliPreset::PinjieCliLegacy:
            baseSearchRange = 200.0;
            rotationMinDeg = -1.0;
            rotationMaxDeg = 1.0;
            rotationStepDeg = 0.05;
            imageCollectionMode = ImageCollectionMode::SequentialPicBmp;
            break;
        case CliPreset::StitchAppLegacy:
            baseSearchRange = 3000.0;
            rotationMinDeg = -1.0;
            rotationMaxDeg = 1.0;
            rotationStepDeg = 0.05;
            imageCollectionMode = ImageCollectionMode::SequentialPicBmp;
            break;
        }
    }

    bool parseImageCollectionMode(const std::string &text, ImageCollectionMode &mode)
    {
        const std::string normalized = toLowerAscii(text);
        if (normalized == "sequential" || normalized == "pic" || normalized == "pic-seq")
        {
            mode = ImageCollectionMode::SequentialPicBmp;
            return true;
        }
        if (normalized == "scan" || normalized == "scan-all" || normalized == "natural")
        {
            mode = ImageCollectionMode::ScanAllNatural;
            return true;
        }

        return false;
    }

    std::string imageCollectionModeName(ImageCollectionMode mode)
    {
        switch (mode)
        {
        case ImageCollectionMode::SequentialPicBmp:
            return "sequential-pic-bmp";
        case ImageCollectionMode::ScanAllNatural:
            return "scan-all-natural";
        }

        return "scan-all-natural";
    }

    bool isSupportedImageExtension(const std::filesystem::path &path)
    {
        const std::string ext = toLowerAscii(path.extension().u8string());
        return ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
               ext == ".tif" || ext == ".tiff";
    }

    bool naturalFilenameLess(const std::string &lhs, const std::string &rhs)
    {
        std::size_t i = 0;
        std::size_t j = 0;
        while (i < lhs.size() && j < rhs.size())
        {
            const unsigned char leftCh = static_cast<unsigned char>(lhs[i]);
            const unsigned char rightCh = static_cast<unsigned char>(rhs[j]);
            const bool leftDigit = std::isdigit(leftCh) != 0;
            const bool rightDigit = std::isdigit(rightCh) != 0;

            if (leftDigit && rightDigit)
            {
                std::size_t leftEnd = i;
                while (leftEnd < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[leftEnd])) != 0)
                {
                    ++leftEnd;
                }

                std::size_t rightEnd = j;
                while (rightEnd < rhs.size() && std::isdigit(static_cast<unsigned char>(rhs[rightEnd])) != 0)
                {
                    ++rightEnd;
                }

                const std::string leftDigits = lhs.substr(i, leftEnd - i);
                const std::string rightDigits = rhs.substr(j, rightEnd - j);
                std::size_t leftTrimmed = 0;
                while (leftTrimmed + 1 < leftDigits.size() && leftDigits[leftTrimmed] == '0')
                {
                    ++leftTrimmed;
                }
                std::size_t rightTrimmed = 0;
                while (rightTrimmed + 1 < rightDigits.size() && rightDigits[rightTrimmed] == '0')
                {
                    ++rightTrimmed;
                }

                const std::string leftNormalized = leftDigits.substr(leftTrimmed);
                const std::string rightNormalized = rightDigits.substr(rightTrimmed);
                if (leftNormalized.size() != rightNormalized.size())
                {
                    return leftNormalized.size() < rightNormalized.size();
                }
                if (leftNormalized != rightNormalized)
                {
                    return leftNormalized < rightNormalized;
                }
                if (leftDigits.size() != rightDigits.size())
                {
                    return leftDigits.size() < rightDigits.size();
                }

                i = leftEnd;
                j = rightEnd;
                continue;
            }

            const char leftLower = static_cast<char>(std::tolower(leftCh));
            const char rightLower = static_cast<char>(std::tolower(rightCh));
            if (leftLower != rightLower)
            {
                return leftLower < rightLower;
            }

            ++i;
            ++j;
        }

        return lhs.size() < rhs.size();
    }

    void printUsage(const char *argv0)
    {
        std::cout << "Usage:\n"
                  << "  " << argv0 << " <input_dir> <image_count> [--out <out_png>]\n"
                  << "               [--csv <out_csv>] [--debug-dir <process_dir>] [--no-process-vis]\n"
                  << "               [--preset {gui|pinjie-cli|stitch-app}] [--input-mode {scan|sequential}]\n"
                  << "               [--run-mode {full|registration}]\n"
                  << "               [--endpoint-probe-fast]\n"
                  << "               [--scan-all-images]\n"
                  << "               [--start-index <n>]\n"
                  << "               [--overlap <ratio_or_percent>] [--direction {auto|x+|x-|y+|y-}]\n"
                  << "               [--search-range <pixels>] [--base-search-range <pixels>]\n"
                  << "               [--rotation-range <deg>] [--rotation-step <deg>]\n"
                  << "               [--rotation-min <deg>] [--rotation-max <deg>]\n"
                  << "               [--tangent-residual-weight <w>] [--tangent-correlation-weight <w>]\n"
                  << "               [--no-point-filter] [--filter-confidence-q <ratio_or_percent>]\n"
                  << "               [--filter-gradient-q <ratio_or_percent>] [--filter-window-radius <points>]\n"
                  << "               [--filter-hampel-sigma <sigma>] [--filter-hampel-min-scale <px>]\n"
                  << "               defaults: --preset gui --input-mode scan --overlap 0.875\n"
                  << "                         --direction x+ --search-range 200\n"
                  << "                         --rotation-range 0.5 --rotation-step 0.01\n"
                  << "                         tangent weights: residual=0.05, correlation=0.25\n"
                  << "                         point-filter on, q=0.15/0.15, radius=5, sigma=3.0\n"
                  << "               presets:\n"
                  << "                         gui         => search-range 200, rotation +/-0.5 deg, step 0.01,\n"
                  << "                                        scan all supported images using natural sort\n"
                  << "                         pinjie-cli  => historical pinjie_cli defaults (200, +/-1.0, step 0.05)\n"
                  << "                         stitch-app  => legacy stitch_app defaults (3000, +/-1.0, step 0.05)\n\n"
                  << "Example:\n"
                  << "  " << argv0 << " D:/VSCode_Project/pinjie/<input_dir> 2\n";
    }

    bool collectImagePaths(const std::filesystem::path &inputDir,
                           int imageCount,
                           int startIndex,
                           ImageCollectionMode imageCollectionMode,
                           std::vector<std::string> &paths,
                           std::string &errorMessage)
    {
        paths.clear();
        paths.reserve(static_cast<std::size_t>(std::max(imageCount, 0)));

        if (imageCollectionMode == ImageCollectionMode::SequentialPicBmp)
        {
            for (int i = 0; i < imageCount; ++i)
            {
                const std::filesystem::path imagePath = inputDir / ("Pic_" + std::to_string(startIndex + i) + ".bmp");
                if (!std::filesystem::exists(imagePath))
                {
                    errorMessage = "Missing sequential image: " + imagePath.generic_string();
                    paths.clear();
                    return false;
                }
                paths.push_back(imagePath.u8string());
            }
            return true;
        }

        std::vector<std::filesystem::path> scannedImages;
        std::error_code iterError;
        for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(inputDir, iterError))
        {
            if (iterError)
            {
                errorMessage = "Failed to scan input directory: " + inputDir.generic_string();
                paths.clear();
                return false;
            }
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (!isSupportedImageExtension(entry.path()))
            {
                continue;
            }
            scannedImages.push_back(entry.path());
        }

        std::sort(scannedImages.begin(), scannedImages.end(),
                  [](const std::filesystem::path &lhs, const std::filesystem::path &rhs)
                  {
                      return naturalFilenameLess(lhs.filename().u8string(), rhs.filename().u8string());
                  });

        const int zeroBasedStart = std::max(0, startIndex - 1);
        if (zeroBasedStart >= static_cast<int>(scannedImages.size()))
        {
            errorMessage = "start-index exceeds scanned image count.";
            paths.clear();
            return false;
        }

        if (zeroBasedStart + imageCount > static_cast<int>(scannedImages.size()))
        {
            errorMessage = "Requested image_count exceeds scanned image count.";
            paths.clear();
            return false;
        }

        for (int i = 0; i < imageCount; ++i)
        {
            paths.push_back(scannedImages[static_cast<std::size_t>(zeroBasedStart + i)].u8string());
        }

        return true;
    }

    void printResolvedConfiguration(CliPreset preset,
                                    CliRunMode runMode,
                                    ImageCollectionMode imageCollectionMode,
                                    int startIndex,
                                    int imageCount,
                                    double overlapRatio,
                                    double baseSearchRange,
                                    double rotationMinDeg,
                                    double rotationMaxDeg,
                                    double rotationStepDeg)
    {
        std::cout << "[Config] preset=" << cliPresetName(preset)
                  << ", run-mode=" << cliRunModeName(runMode)
                  << ", input-mode=" << imageCollectionModeName(imageCollectionMode)
                  << ", start-index=" << startIndex
                  << ", image-count=" << imageCount << '\n';
        std::cout << "[Config] overlap=" << overlapRatio
                  << ", search-range=" << baseSearchRange
                  << ", rotation=[" << rotationMinDeg << ", " << rotationMaxDeg
                  << "], step=" << rotationStepDeg << std::endl;
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
        pinjie::buildDefaultStitchResultPaths("cli", "workpiece", saveDebugVisualization);
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
    double baseSearchRange = 200.0;
    double rotationMinDeg = -0.5;
    double rotationMaxDeg = 0.5;
    double rotationStepDeg = 0.01;
    double tangentResidualWeight = 0.05;
    double tangentCorrelationWeight = 0.25;
    bool enablePointFiltering = true;
    double filterConfidenceQuantile = 0.15;
    double filterGradientQuantile = 0.15;
    int filterWindowRadius = 5;
    double filterHampelSigma = 3.0;
    double filterHampelMinScale = 0.05;
    pinjie::MotionPriorDirection directionConstraint = pinjie::MotionPriorDirection::XPositive;
    CliPreset preset = CliPreset::Gui;
    CliRunMode runMode = CliRunMode::Full;
    bool endpointProbeFastMode = false;
    ImageCollectionMode imageCollectionMode = ImageCollectionMode::ScanAllNatural;
    applyCliPreset(preset,
                   overlapRatio,
                   baseSearchRange,
                   rotationMinDeg,
                   rotationMaxDeg,
                   rotationStepDeg,
                   tangentResidualWeight,
                   tangentCorrelationWeight,
                   enablePointFiltering,
                   filterConfidenceQuantile,
                   filterGradientQuantile,
                   filterWindowRadius,
                   filterHampelSigma,
                   filterHampelMinScale,
                   directionConstraint,
                   imageCollectionMode);
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
        else if (arg == "--preset" && i + 1 < argc)
        {
            CliPreset parsedPreset = CliPreset::Gui;
            if (!parseCliPreset(argv[++i], parsedPreset))
            {
                std::cout << "[Error] Invalid preset. Use gui, pinjie-cli, or stitch-app.\n";
                return -1;
            }

            preset = parsedPreset;
            applyCliPreset(preset,
                           overlapRatio,
                           baseSearchRange,
                           rotationMinDeg,
                           rotationMaxDeg,
                           rotationStepDeg,
                           tangentResidualWeight,
                           tangentCorrelationWeight,
                           enablePointFiltering,
                           filterConfidenceQuantile,
                           filterGradientQuantile,
                           filterWindowRadius,
                           filterHampelSigma,
                           filterHampelMinScale,
                           directionConstraint,
                           imageCollectionMode);
        }
        else if (arg == "--run-mode" && i + 1 < argc)
        {
            if (!parseCliRunMode(argv[++i], runMode))
            {
                std::cout << "[Error] Invalid run-mode. Use full or registration.\n";
                return -1;
            }
        }
        else if (arg == "--endpoint-probe-fast")
        {
            endpointProbeFastMode = true;
        }
        else if (arg == "--input-mode" && i + 1 < argc)
        {
            if (!parseImageCollectionMode(argv[++i], imageCollectionMode))
            {
                std::cout << "[Error] Invalid input-mode. Use scan or sequential.\n";
                return -1;
            }
        }
        else if (arg == "--scan-all-images")
        {
            imageCollectionMode = ImageCollectionMode::ScanAllNatural;
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

    printResolvedConfiguration(preset,
                               runMode,
                               imageCollectionMode,
                               startIndex,
                               imageCount,
                               overlapRatio,
                               baseSearchRange,
                               rotationMinDeg,
                               rotationMaxDeg,
                               rotationStepDeg);

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

    pinjie::StitchRunRequest request;
    std::string imageCollectionError;
    if (!collectImagePaths(inputPath,
                           imageCount,
                           startIndex,
                           imageCollectionMode,
                           request.imagePaths,
                           imageCollectionError))
    {
        std::cout << "[Error] " << imageCollectionError << std::endl;
        return -1;
    }
    request.edgeConfig.cannyLow = 50.0;
    request.edgeConfig.cannyHigh = 150.0;
    request.edgeConfig.subpixWindow = 7;
    request.edgeConfig.subpixSigma = 1.0;
    request.edgeConfig.enablePointFiltering = enablePointFiltering;
    request.edgeConfig.filterConfidenceQuantile = filterConfidenceQuantile;
    request.edgeConfig.filterGradientQuantile = filterGradientQuantile;
    request.edgeConfig.filterLocalLinearWindowRadius = filterWindowRadius;
    request.edgeConfig.filterHampelSigma = filterHampelSigma;
    request.edgeConfig.filterHampelMinScale = filterHampelMinScale;
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
    request.pipelineConfig.enableDesignComparison = runMode != CliRunMode::Registration;
    request.pipelineConfig.endpointProbeFastMode = endpointProbeFastMode;
    request.resultOutputDir = pinjie::genericUtf8String(outputDir);
    request.panoramaOutputPath = panoramaPath.u8string();
    request.csvOutputPath = csvPath.u8string();
    request.alignmentCandidateDiagnosticsCsvOutputPath =
        pinjie::genericUtf8String(outputDir / "alignment_candidate_diagnostics.csv");
    if (runMode == CliRunMode::Full)
    {
        request.designErrorProfileCsvOutputPath =
            pinjie::genericUtf8String(outputDir / "design_error_profile.csv");
        request.designErrorSummaryCsvOutputPath =
            pinjie::genericUtf8String(outputDir / "design_error_summary.csv");
        request.designComparisonOverlayOutputPath =
            pinjie::genericUtf8String(outputDir / "design_comparison_overlay.png");
        request.qualityReviewCsvOutputPath =
            pinjie::genericUtf8String(outputDir / "quality_review.csv");
        request.contourOverlayOutputPath = pinjie::genericUtf8String(outputDir / "origin_contour_overlay.png");
        request.tangentCorrelationAllOutputPath =
            pinjie::genericUtf8String(outputDir / "tangent_correlation_all.png");
        request.tangentCorrelationInlierOutputPath =
            pinjie::genericUtf8String(outputDir / "tangent_correlation_inlier.png");
    }
    if (saveDebugVisualization)
    {
        request.debugImageOutputDir = pinjie::genericUtf8String(debugDir);
    }

    pinjie::TaskCallbacks callbacks;
    callbacks.onLog = [](const std::string &message)
    {
        std::cout << message << std::endl;
    };

    const pinjie::StitchWorkflowService workflow;
    const pinjie::StitchRunMode workflowRunMode =
        runMode == CliRunMode::Registration ? pinjie::StitchRunMode::Registration
                                            : pinjie::StitchRunMode::Full;
    pinjie::StitchRunResult result = workflow.run(request, workflowRunMode, callbacks);
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
