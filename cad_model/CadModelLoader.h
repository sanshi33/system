#pragma once

#include "cad_design/DesignProfileTypes.h"

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace pinjie::cad_model {

enum class CadFileFormat {
    Unknown,
    Step,
    Iges,
    Stl,
};

struct DesignModelRequest {
    std::string cadFilePath;
    std::string modelName;
    std::string axialAxis{"X"};
    std::string radialAxis{"Y"};
    bool reverseAxialDirection{false};
    bool useLeftEndpointAsOrigin{true};
    bool extractUpperEnvelope{false};
    double profileSamplingStepMm{0.005};
    double sectionCoordinateMm{std::numeric_limits<double>::quiet_NaN()};
    double targetSlotWidthMm{0.0};
    double targetSlotDepthMm{0.0};
    double temporaryPixelSizeMm{0.0};
    bool localSlotImageMode{false};
    double localSlotBottomWidthMm{0.05};
};

struct CadMeshVertex {
    double xMm{0.0};
    double yMm{0.0};
    double zMm{0.0};
};

struct CadMeshTriangle {
    std::size_t a{0};
    std::size_t b{0};
    std::size_t c{0};
};

struct CadModelDocument {
    bool valid{false};
    bool occtEnabled{false};
    CadFileFormat format{CadFileFormat::Unknown};
    std::string sourcePath;
    std::string fileName;
    std::string modelLabel;
    std::string axialAxis{"X"};
    std::string radialAxis{"Y"};
    int rootShapeCount{0};
    int solidCount{0};
    int shellCount{0};
    int faceCount{0};
    int edgeCount{0};
    bool hasBounds{false};
    double minXmm{0.0};
    double minYmm{0.0};
    double minZmm{0.0};
    double maxXmm{0.0};
    double maxYmm{0.0};
    double maxZmm{0.0};
    bool hasMesh{false};
    std::vector<CadMeshVertex> meshVertices;
    std::vector<CadMeshTriangle> meshTriangles;
    bool hasProfileSamples{false};
    bool hasSectionSamples{false};
    double profileSamplingStepMm{0.05};
    double profileMinSMm{0.0};
    double profileMaxSMm{0.0};
    double profileMinRMm{0.0};
    double profileMaxRMm{0.0};
    std::vector<pinjie::cad_design::DesignProfileSample> sectionSamples;
    std::vector<pinjie::cad_design::DesignProfileSample> profileSamples;
    pinjie::cad_design::DesignProfileMetadata profileMetadata;
    std::string profileExtractionMessage;
    std::string importMessage;
};

struct CadModelLoadResult {
    bool ok{false};
    std::string message;
    CadModelDocument document;
};

bool occtCadImportAvailable();
std::string occtCadImportBackendSummary();
CadFileFormat detectCadFileFormat(const std::filesystem::path& path);
std::string cadFileFormatLabel(CadFileFormat format);
CadModelLoadResult loadCadModelDocument(const std::filesystem::path& path);
CadModelLoadResult loadCadModelDocument(const std::filesystem::path& path, const DesignModelRequest& request);
std::string buildCadModelSummaryText(const CadModelDocument& document);

} // namespace pinjie::cad_model
