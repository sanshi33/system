#include "cad_design/DesignProfileAlignment.h"
#include "stitch/StitchTypes.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::string current;
    bool inQuotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
            continue;
        }
        if (ch == ',' && !inQuotes) {
            cells.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    cells.push_back(current);
    return cells;
}

bool tryParseDouble(const std::string& text, double& value)
{
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && end != nullptr && *end == '\0';
}

std::string csvCell(const std::vector<std::string>& cells,
                    const std::unordered_map<std::string, std::size_t>& headerIndex,
                    const std::string& key)
{
    const auto it = headerIndex.find(key);
    if (it == headerIndex.end() || it->second >= cells.size()) {
        return {};
    }
    return cells[it->second];
}

std::vector<cv::Point2d> loadContourSamplesFromCsv(const std::filesystem::path& contourCsvPath,
                                                   double& inferredStepPx,
                                                   std::string& errorMessage)
{
    std::ifstream stream(contourCsvPath, std::ios::binary);
    if (!stream) {
        errorMessage = "failed to open contour_points.csv: " + contourCsvPath.generic_string();
        return {};
    }

    std::string headerLine;
    if (!std::getline(stream, headerLine)) {
        errorMessage = "contour_points.csv is empty: " + contourCsvPath.generic_string();
        return {};
    }

    const std::vector<std::string> headers = splitCsvLine(headerLine);
    std::unordered_map<std::string, std::size_t> headerIndex;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        headerIndex.emplace(headers[i], i);
    }

    if (headerIndex.count("CanvasX(px)") == 0 || headerIndex.count("CanvasY(px)") == 0) {
        errorMessage = "contour_points.csv is missing CanvasX(px) or CanvasY(px) columns.";
        return {};
    }

    std::vector<cv::Point2d> contourSamples;
    contourSamples.reserve(200000);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = splitCsvLine(line);
        double xPx = 0.0;
        double yPx = 0.0;
        if (!tryParseDouble(csvCell(cells, headerIndex, "CanvasX(px)"), xPx) ||
            !tryParseDouble(csvCell(cells, headerIndex, "CanvasY(px)"), yPx)) {
            continue;
        }
        contourSamples.emplace_back(xPx, yPx);
    }

    if (contourSamples.size() < 2) {
        errorMessage = "contour_points.csv does not contain enough valid contour points.";
        return {};
    }

    std::sort(contourSamples.begin(), contourSamples.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    });

    std::vector<double> positiveStepsPx;
    positiveStepsPx.reserve(contourSamples.size());
    double lastX = contourSamples.front().x;
    for (std::size_t i = 1; i < contourSamples.size(); ++i) {
        const double dx = contourSamples[i].x - lastX;
        if (dx > 1e-6) {
            positiveStepsPx.push_back(dx);
            lastX = contourSamples[i].x;
        }
    }

    if (positiveStepsPx.empty()) {
        errorMessage = "failed to infer contour x-step from contour_points.csv.";
        return {};
    }

    std::sort(positiveStepsPx.begin(), positiveStepsPx.end());
    inferredStepPx = positiveStepsPx[positiveStepsPx.size() / 2];
    return contourSamples;
}

bool writeTextFile(const std::filesystem::path& path, const std::string& content)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    stream << content;
    return stream.good();
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: design_reval_from_contour <result_dir>\n";
        return 1;
    }

    const std::filesystem::path resultDir = std::filesystem::u8path(argv[1]);
    const std::filesystem::path contourCsvPath = resultDir / "contour_points.csv";
    if (!std::filesystem::exists(contourCsvPath)) {
        std::cerr << "[Error] missing contour_points.csv under " << resultDir.generic_string() << "\n";
        return 1;
    }

    double inferredStepPx = 0.0;
    std::string loadError;
    const std::vector<cv::Point2d> contourSamples =
        loadContourSamplesFromCsv(contourCsvPath, inferredStepPx, loadError);
    if (contourSamples.empty()) {
        std::cerr << "[Error] " << loadError << "\n";
        return 1;
    }

    stitch::StitchPipelineConfig config;
    config.enableDesignComparison = true;
    config.designPixelSizeMm = 0.010057;
    config.designUseLeftEndpointAnchor = true;
    config.designAnchorRadialToLeftEndpoint = true;
    config.designUseUpperEnvelope = true;
    config.designEvaluateProfileForm = true;
    config.designFilterEndFaceEdges = true;
    config.designIgnoreStepTransition = true;
    config.designEnableBestFitTranslation = true;
    config.designEnableBestFitRotation = true;

    const pinjie::cad_design::DesignAlignmentResult designResult =
        pinjie::cad_design::compareMeasuredContourSamplesToDesign(contourSamples, config);
    if (!designResult.ok) {
        std::cerr << "[Error] design re-evaluation failed: " << designResult.message << "\n";
        return 1;
    }

    if (!writeTextFile(resultDir / "design_error_profile.csv", designResult.profileCsvText) ||
        !writeTextFile(resultDir / "design_error_summary.csv", designResult.summaryCsvText) ||
        !writeTextFile(resultDir / "design_3d_error_points.csv", designResult.error3dCsvText) ||
        !writeTextFile(resultDir / "design_compensation.csv", designResult.compensationCsvText) ||
        !writeTextFile(resultDir / "design_feature_compensation.csv", designResult.featureCompensationCsvText)) {
        std::cerr << "[Error] failed to write re-evaluated result files.\n";
        return 1;
    }

    std::cout << "[OK] re-evaluated design comparison under " << resultDir.generic_string() << "\n";
    std::cout << "[Info] inferred contour x-step px=" << std::fixed << std::setprecision(6) << inferredStepPx
              << ", design_pixel_size_mm=" << config.designPixelSizeMm << "\n";
    std::cout << "[Info] reverse_axial=" << (designResult.summary.designReverseAxial ? 1 : 0)
              << ", used_count=" << designResult.summary.usedCount
              << ", abs_filtered_rmse_um=" << designResult.summary.absoluteFilteredStats.rmseUm << "\n";
    return 0;
}
