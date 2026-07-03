#include "cad_model/CadModelLoader.h"

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void printUsage()
{
    std::cout
        << "Usage: cad_model_probe <cad_file> [--axial X|Y|Z] [--radial X|Y|Z]\n"
        << "                       [--step mm] [--section-coordinate mm]\n"
        << "                       [--reverse-axial] [--no-upper-envelope]\n"
        << "                       [--out-profile path]\n";
}

bool writeProfileCsv(const std::filesystem::path& path,
                     const pinjie::cad_model::CadModelDocument& document)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    stream << "index,s_mm,r_mm,has_cad_point,cad_x_mm,cad_y_mm,cad_z_mm\n";
    std::size_t index = 0;
    const auto& previewSamples = document.hasSectionSamples ? document.sectionSamples : document.profileSamples;
    for (const auto& sample : previewSamples) {
        stream << (++index) << ","
               << sample.sMm << ","
               << sample.rMm << ","
               << (sample.hasCadPoint ? 1 : 0) << ","
               << sample.cadXMm << ","
               << sample.cadYMm << ","
               << sample.cadZMm << "\n";
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage();
        return 2;
    }
    const std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        printUsage();
        return 0;
    }

    pinjie::cad_model::DesignModelRequest request;
    request.cadFilePath = argv[1];
    std::filesystem::path outputProfilePath;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto requireValue = [&](const char* option) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << option << "\n";
                std::exit(2);
            }
            return argv[++index];
        };

        if (arg == "--axial") {
            request.axialAxis = requireValue("--axial");
        } else if (arg == "--radial") {
            request.radialAxis = requireValue("--radial");
        } else if (arg == "--step") {
            request.profileSamplingStepMm = std::stod(requireValue("--step"));
        } else if (arg == "--section-coordinate") {
            request.sectionCoordinateMm = std::stod(requireValue("--section-coordinate"));
        } else if (arg == "--reverse-axial") {
            request.reverseAxialDirection = true;
        } else if (arg == "--no-upper-envelope") {
            request.extractUpperEnvelope = false;
        } else if (arg == "--out-profile") {
            outputProfilePath = std::filesystem::u8path(requireValue("--out-profile"));
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 2;
        }
    }

    const auto result = pinjie::cad_model::loadCadModelDocument(
        std::filesystem::u8path(request.cadFilePath),
        request);

    std::cout << pinjie::cad_model::buildCadModelSummaryText(result.document);
    if (!result.ok) {
        std::cerr << result.message << "\n";
        return 1;
    }

    if (!outputProfilePath.empty()) {
        if (!writeProfileCsv(outputProfilePath, result.document)) {
            std::cerr << "Failed to write profile CSV: " << outputProfilePath.u8string() << "\n";
            return 1;
        }
        std::cout << "CAD preview CSV: " << outputProfilePath.u8string() << "\n";
    }

    return result.document.hasProfileSamples ? 0 : 1;
}
