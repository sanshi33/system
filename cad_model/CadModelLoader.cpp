#include "cad_model/CadModelLoader.h"

#include "common/ResultPathUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#if PINJIE_HAS_OCCT
#include <BRepAdaptor_Curve.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IGESControl_Reader.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
#include <STEPControl_Reader.hxx>
#include <TopLoc_Location.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#endif

namespace pinjie::cad_model {

namespace {

std::string toLowerCopy(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(),
                   lower.end(),
                   lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lower;
}

std::string formatDouble(const double value, const int precision = 3)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

#if PINJIE_HAS_OCCT
struct RawProfilePoint {
    double axialMm{0.0};
    double radialMm{0.0};
    double xMm{0.0};
    double yMm{0.0};
    double zMm{0.0};
};

struct ProfileBin {
    std::size_t count{0};
    double sSumMm{0.0};
    double xSumMm{0.0};
    double ySumMm{0.0};
    double zSumMm{0.0};
    double bestSMm{0.0};
    double bestRMm{-std::numeric_limits<double>::infinity()};
    double bestXMm{0.0};
    double bestYMm{0.0};
    double bestZMm{0.0};
    std::vector<double> radiiMm;
};

struct CartesianPoint {
    double xMm{0.0};
    double yMm{0.0};
    double zMm{0.0};
};

void updateProfileStats(CadModelDocument& document);

int countSubShapes(const TopoDS_Shape& shape, const TopAbs_ShapeEnum type)
{
    int count = 0;
    for (TopExp_Explorer explorer(shape, type); explorer.More(); explorer.Next()) {
        ++count;
    }
    return count;
}

std::string ifSelectStatusMessage(const IFSelect_ReturnStatus status)
{
    switch (status) {
    case IFSelect_RetDone:
        return "import completed";
    case IFSelect_RetVoid:
        return "reader returned empty content";
    case IFSelect_RetError:
        return "reader reported a generic error";
    case IFSelect_RetFail:
        return "reader failed to transfer the source file";
    case IFSelect_RetStop:
        return "reader stopped before transfer finished";
    default:
        return "reader returned an unknown status";
    }
}

bool containsNonAsciiByte(const std::string& value)
{
    return std::any_of(value.begin(), value.end(), [](const unsigned char ch) {
        return ch >= 128;
    });
}

struct CadReaderPathAlias {
    std::filesystem::path readPath;
    std::filesystem::path temporaryPath;
    std::string note;

    CadReaderPathAlias() = default;
    CadReaderPathAlias(const CadReaderPathAlias&) = delete;
    CadReaderPathAlias& operator=(const CadReaderPathAlias&) = delete;

    CadReaderPathAlias(CadReaderPathAlias&& other) noexcept
        : readPath(std::move(other.readPath)),
          temporaryPath(std::move(other.temporaryPath)),
          note(std::move(other.note))
    {
        other.temporaryPath.clear();
    }

    CadReaderPathAlias& operator=(CadReaderPathAlias&& other) noexcept
    {
        if (this != &other) {
            cleanup();
            readPath = std::move(other.readPath);
            temporaryPath = std::move(other.temporaryPath);
            note = std::move(other.note);
            other.temporaryPath.clear();
        }
        return *this;
    }

    ~CadReaderPathAlias()
    {
        cleanup();
    }

private:
    void cleanup()
    {
        if (temporaryPath.empty()) {
            return;
        }
        std::error_code error;
        std::filesystem::remove(temporaryPath, error);
        temporaryPath.clear();
    }
};

CadReaderPathAlias prepareCadReaderPathAlias(const std::filesystem::path& sourcePath,
                                             const CadFileFormat format)
{
    CadReaderPathAlias alias;
    alias.readPath = sourcePath;

    const std::string sourceUtf8 = pinjie::genericUtf8String(sourcePath);
    if (!containsNonAsciiByte(sourceUtf8)) {
        return alias;
    }

    std::error_code error;
    const std::filesystem::path cacheDir = pinjie::projectRootPath() / "result" / "cad_import_cache";
    std::filesystem::create_directories(cacheDir, error);
    if (error) {
        return alias;
    }

    static std::uint64_t sequence = 0;
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string extension = format == CadFileFormat::Iges ? ".iges" : ".step";
    const std::filesystem::path temporaryPath =
        cacheDir / ("pinjie_cad_import_" + std::to_string(now) + "_" +
                    std::to_string(++sequence) + extension);

    std::filesystem::copy_file(sourcePath,
                               temporaryPath,
                               std::filesystem::copy_options::overwrite_existing,
                               error);
    if (error) {
        return alias;
    }

    alias.readPath = temporaryPath;
    alias.temporaryPath = temporaryPath;
    alias.note = "CAD source path contains non-ASCII characters; OpenCascade read used a temporary ASCII copy";
    return alias;
}

int axisIndexFromLabel(const std::string& label)
{
    if (label == "X" || label == "x") {
        return 0;
    }
    if (label == "Y" || label == "y") {
        return 1;
    }
    if (label == "Z" || label == "z") {
        return 2;
    }
    return -1;
}

std::string axisLabelFromIndex(const int axisIndex)
{
    switch (axisIndex) {
    case 0:
        return "X";
    case 1:
        return "Y";
    case 2:
        return "Z";
    default:
        return {};
    }
}

int remainingAxisIndex(const int firstAxis, const int secondAxis)
{
    if (firstAxis < 0 || firstAxis > 2 || secondAxis < 0 || secondAxis > 2 || firstAxis == secondAxis) {
        return -1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != firstAxis && axis != secondAxis) {
            return axis;
        }
    }
    return -1;
}

double pointAxisValue(const gp_Pnt& point, const int axisIndex)
{
    switch (axisIndex) {
    case 0:
        return point.X();
    case 1:
        return point.Y();
    case 2:
        return point.Z();
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double coordinateAxisValue(const double x, const double y, const double z, const int axisIndex)
{
    switch (axisIndex) {
    case 0:
        return x;
    case 1:
        return y;
    case 2:
        return z;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

void setCoordinateAxisValue(double& x, double& y, double& z, const int axisIndex, const double value)
{
    switch (axisIndex) {
    case 0:
        x = value;
        break;
    case 1:
        y = value;
        break;
    case 2:
        z = value;
        break;
    default:
        break;
    }
}

gp_Dir directionFromAxisIndex(const int axisIndex)
{
    switch (axisIndex) {
    case 0:
        return gp_Dir(1.0, 0.0, 0.0);
    case 1:
        return gp_Dir(0.0, 1.0, 0.0);
    case 2:
    default:
        return gp_Dir(0.0, 0.0, 1.0);
    }
}

double medianValue(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        const auto lowerMid = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        median = 0.5 * (median + *lowerMid);
    }
    return median;
}

bool parseThreeDoubles(const std::string& text, std::array<double, 3>& values)
{
    std::vector<double> parsed;
    parsed.reserve(3);
    const char* cursor = text.c_str();
    while (*cursor != '\0' && parsed.size() < 3) {
        char* end = nullptr;
        const double value = std::strtod(cursor, &end);
        if (end != cursor) {
            parsed.push_back(value);
            cursor = end;
            continue;
        }
        ++cursor;
    }

    if (parsed.size() < 3 ||
        !std::isfinite(parsed[0]) ||
        !std::isfinite(parsed[1]) ||
        !std::isfinite(parsed[2])) {
        return false;
    }

    values = {parsed[0], parsed[1], parsed[2]};
    return true;
}

int parseEntityIdBeforeMarker(const std::string& content, const std::size_t markerPosition)
{
    const std::size_t hashPosition = content.rfind('#', markerPosition);
    if (hashPosition == std::string::npos) {
        return -1;
    }

    const std::size_t previousStatement = content.rfind(';', markerPosition);
    if (previousStatement != std::string::npos && hashPosition < previousStatement) {
        return -1;
    }

    std::size_t cursor = hashPosition + 1;
    int id = 0;
    bool hasDigit = false;
    while (cursor < content.size() &&
           std::isdigit(static_cast<unsigned char>(content[cursor]))) {
        hasDigit = true;
        id = id * 10 + (content[cursor] - '0');
        ++cursor;
    }
    return hasDigit ? id : -1;
}

std::map<int, CartesianPoint> parseStepCartesianPointMap(const std::string& content)
{
    std::map<int, CartesianPoint> pointById;
    constexpr const char* kMarker = "CARTESIAN_POINT";
    std::size_t position = 0;
    while ((position = content.find(kMarker, position)) != std::string::npos) {
        const int entityId = parseEntityIdBeforeMarker(content, position);
        std::size_t statementEnd = content.find(';', position);
        if (statementEnd == std::string::npos) {
            statementEnd = std::min(content.size(), position + std::size_t{512});
        }

        const std::size_t tupleOpen = content.rfind('(', statementEnd);
        if (entityId >= 0 && tupleOpen != std::string::npos && tupleOpen > position) {
            std::size_t tupleClose = content.find(')', tupleOpen + 1);
            if (tupleClose == std::string::npos || tupleClose > statementEnd) {
                tupleClose = statementEnd;
            }

            std::array<double, 3> xyz{};
            if (tupleClose > tupleOpen &&
                parseThreeDoubles(content.substr(tupleOpen + 1, tupleClose - tupleOpen - 1), xyz)) {
                pointById[entityId] = {xyz[0], xyz[1], xyz[2]};
            }
        }

        position = statementEnd + 1;
    }
    return pointById;
}

std::vector<int> parseStepCurveSetPointReferences(const std::string& content)
{
    std::vector<int> references;
    std::set<int> seen;
    constexpr const char* kMarker = "GEOMETRIC_CURVE_SET";
    std::size_t position = 0;
    while ((position = content.find(kMarker, position)) != std::string::npos) {
        std::size_t statementEnd = content.find(';', position);
        if (statementEnd == std::string::npos) {
            statementEnd = std::min(content.size(), position + std::size_t{4096});
        }

        std::size_t cursor = position;
        while (cursor < statementEnd) {
            if (content[cursor] != '#') {
                ++cursor;
                continue;
            }

            ++cursor;
            int id = 0;
            bool hasDigit = false;
            while (cursor < statementEnd &&
                   std::isdigit(static_cast<unsigned char>(content[cursor]))) {
                hasDigit = true;
                id = id * 10 + (content[cursor] - '0');
                ++cursor;
            }
            if (hasDigit && seen.insert(id).second) {
                references.push_back(id);
            }
        }
        position = statementEnd + 1;
    }
    return references;
}

bool parseStepPointCloud(const std::filesystem::path& path,
                         std::vector<CartesianPoint>& output,
                         std::string& message)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        message = "STEP point-cloud fallback failed: source file cannot be opened.";
        return false;
    }

    const std::string content((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
    const std::map<int, CartesianPoint> pointById = parseStepCartesianPointMap(content);
    if (pointById.empty()) {
        message = "STEP point-cloud fallback failed: no CARTESIAN_POINT entities were found.";
        return false;
    }

    const std::vector<int> curveSetReferences = parseStepCurveSetPointReferences(content);
    output.clear();
    output.reserve(curveSetReferences.empty() ? pointById.size() : curveSetReferences.size());
    if (!curveSetReferences.empty()) {
        for (const int id : curveSetReferences) {
            const auto found = pointById.find(id);
            if (found != pointById.end()) {
                output.push_back(found->second);
            }
        }
    }

    if (output.empty()) {
        for (const auto& [id, point] : pointById) {
            (void)id;
            output.push_back(point);
        }
    }

    if (output.size() < 2) {
        message = "STEP point-cloud fallback failed: fewer than two valid CAD points were parsed.";
        return false;
    }

    std::ostringstream streamMessage;
    streamMessage << "parsed " << output.size() << " CARTESIAN_POINT samples";
    if (!curveSetReferences.empty()) {
        streamMessage << " from GEOMETRIC_CURVE_SET references";
    }
    message = streamMessage.str();
    return true;
}

int largestSpanAxis(const std::array<double, 3>& spans, const int excludedAxis = -1)
{
    int bestAxis = -1;
    double bestSpan = -std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        if (axis == excludedAxis) {
            continue;
        }
        if (spans[axis] > bestSpan) {
            bestSpan = spans[axis];
            bestAxis = axis;
        }
    }
    return bestAxis;
}

void updatePointCloudBounds(const std::vector<CartesianPoint>& points, CadModelDocument& document)
{
    document.minXmm = std::numeric_limits<double>::infinity();
    document.minYmm = std::numeric_limits<double>::infinity();
    document.minZmm = std::numeric_limits<double>::infinity();
    document.maxXmm = -std::numeric_limits<double>::infinity();
    document.maxYmm = -std::numeric_limits<double>::infinity();
    document.maxZmm = -std::numeric_limits<double>::infinity();

    for (const CartesianPoint& point : points) {
        document.minXmm = std::min(document.minXmm, point.xMm);
        document.minYmm = std::min(document.minYmm, point.yMm);
        document.minZmm = std::min(document.minZmm, point.zMm);
        document.maxXmm = std::max(document.maxXmm, point.xMm);
        document.maxYmm = std::max(document.maxYmm, point.yMm);
        document.maxZmm = std::max(document.maxZmm, point.zMm);
    }

    document.hasBounds =
        std::isfinite(document.minXmm) &&
        std::isfinite(document.minYmm) &&
        std::isfinite(document.minZmm) &&
        std::isfinite(document.maxXmm) &&
        std::isfinite(document.maxYmm) &&
        std::isfinite(document.maxZmm);
}

void populateProfileSamplesFromPointCloud(const std::vector<CartesianPoint>& points,
                                          const DesignModelRequest& request,
                                          CadModelDocument& document)
{
    if (points.size() < 2 || !document.hasBounds) {
        document.profileExtractionMessage =
            "STEP point-cloud profile extraction failed: insufficient parsed points.";
        return;
    }

    const std::array<double, 3> spans{
        document.maxXmm - document.minXmm,
        document.maxYmm - document.minYmm,
        document.maxZmm - document.minZmm,
    };

    int axialAxis = axisIndexFromLabel(request.axialAxis);
    int radialAxis = axisIndexFromLabel(request.radialAxis);
    bool adjustedAxes = false;
    if (axialAxis < 0 || axialAxis > 2 || spans[axialAxis] <= 1e-9) {
        axialAxis = largestSpanAxis(spans);
        adjustedAxes = true;
    }
    if (radialAxis < 0 || radialAxis > 2 || radialAxis == axialAxis) {
        radialAxis = largestSpanAxis(spans, axialAxis);
        adjustedAxes = true;
    }
    const int normalAxis = remainingAxisIndex(axialAxis, radialAxis);
    if (axialAxis < 0 || radialAxis < 0 || normalAxis < 0) {
        document.profileExtractionMessage =
            "STEP point-cloud profile extraction failed: unable to choose valid coordinate axes.";
        return;
    }

    document.axialAxis = axisLabelFromIndex(axialAxis);
    document.radialAxis = axisLabelFromIndex(radialAxis);
    document.profileSamplingStepMm = request.profileSamplingStepMm;

    std::vector<RawProfilePoint> rawPoints;
    rawPoints.reserve(points.size());
    std::vector<double> normalValues;
    normalValues.reserve(points.size());
    double minAxial = std::numeric_limits<double>::infinity();
    double maxAxial = -std::numeric_limits<double>::infinity();
    for (const CartesianPoint& point : points) {
        const double axial = coordinateAxisValue(point.xMm, point.yMm, point.zMm, axialAxis);
        const double radial = coordinateAxisValue(point.xMm, point.yMm, point.zMm, radialAxis);
        const double normal = coordinateAxisValue(point.xMm, point.yMm, point.zMm, normalAxis);
        if (!std::isfinite(axial) || !std::isfinite(radial) || !std::isfinite(normal)) {
            continue;
        }
        rawPoints.push_back({axial, radial, point.xMm, point.yMm, point.zMm});
        normalValues.push_back(normal);
        minAxial = std::min(minAxial, axial);
        maxAxial = std::max(maxAxial, axial);
    }

    if (rawPoints.size() < 2 ||
        !std::isfinite(minAxial) ||
        !std::isfinite(maxAxial) ||
        maxAxial <= minAxial) {
        document.profileExtractionMessage =
            "STEP point-cloud profile extraction failed: axial range is invalid.";
        return;
    }

    const double binWidthMm = std::max(0.001, request.profileSamplingStepMm);
    const double cadAxialDirectionSign = request.reverseAxialDirection ? -1.0 : 1.0;
    const double cadAxialOriginMm = request.reverseAxialDirection ? maxAxial : minAxial;
    std::map<long long, ProfileBin> bins;
    document.sectionSamples.clear();
    document.sectionSamples.reserve(rawPoints.size());
    for (const RawProfilePoint& raw : rawPoints) {
        const double sMm =
            request.reverseAxialDirection ? (maxAxial - raw.axialMm) : (raw.axialMm - minAxial);
        if (!std::isfinite(sMm) || !std::isfinite(raw.radialMm) || sMm < -1e-9) {
            continue;
        }

        pinjie::cad_design::DesignProfileSample sectionSample;
        sectionSample.sMm = sMm;
        sectionSample.rMm = raw.radialMm;
        sectionSample.cadXMm = raw.xMm;
        sectionSample.cadYMm = raw.yMm;
        sectionSample.cadZMm = raw.zMm;
        sectionSample.hasCadPoint = true;
        document.sectionSamples.push_back(sectionSample);

        const long long binIndex = static_cast<long long>(std::floor(sMm / binWidthMm + 1e-9));
        ProfileBin& bin = bins[binIndex];
        ++bin.count;
        bin.sSumMm += sMm;
        bin.xSumMm += raw.xMm;
        bin.ySumMm += raw.yMm;
        bin.zSumMm += raw.zMm;
        if (raw.radialMm > bin.bestRMm) {
            bin.bestRMm = raw.radialMm;
            bin.bestSMm = sMm;
            bin.bestXMm = raw.xMm;
            bin.bestYMm = raw.yMm;
            bin.bestZMm = raw.zMm;
        }
        if (!request.extractUpperEnvelope) {
            bin.radiiMm.push_back(raw.radialMm);
        }
    }

    document.profileSamples.clear();
    document.profileSamples.reserve(bins.size());
    for (auto& [binIndex, bin] : bins) {
        (void)binIndex;
        if (bin.count == 0) {
            continue;
        }

        pinjie::cad_design::DesignProfileSample sample;
        if (request.extractUpperEnvelope) {
            sample.sMm = bin.bestSMm;
            sample.rMm = bin.bestRMm;
            sample.cadXMm = bin.bestXMm;
            sample.cadYMm = bin.bestYMm;
            sample.cadZMm = bin.bestZMm;
        } else {
            sample.sMm = bin.sSumMm / static_cast<double>(bin.count);
            sample.rMm = medianValue(std::move(bin.radiiMm));
            sample.cadXMm = bin.xSumMm / static_cast<double>(bin.count);
            sample.cadYMm = bin.ySumMm / static_cast<double>(bin.count);
            sample.cadZMm = bin.zSumMm / static_cast<double>(bin.count);
        }
        sample.hasCadPoint = std::isfinite(sample.cadXMm) &&
                             std::isfinite(sample.cadYMm) &&
                             std::isfinite(sample.cadZMm);
        if (sample.hasCadPoint && std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            document.profileSamples.push_back(sample);
        }
    }

    std::stable_sort(document.profileSamples.begin(), document.profileSamples.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return lhs.rMm < rhs.rMm;
    });

    document.hasSectionSamples = document.sectionSamples.size() >= 2;
    updateProfileStats(document);
    document.profileMetadata.cadAxialOriginMm = cadAxialOriginMm;
    document.profileMetadata.cadAxialDirectionSign = cadAxialDirectionSign;
    document.profileMetadata.extractionMethod = "stp_cartesian_point_fallback";
    document.profileMetadata.sectionNormalAxis = axisLabelFromIndex(normalAxis);
    document.profileMetadata.sectionCoordinateMm = medianValue(std::move(normalValues));

    std::ostringstream message;
    message << "STEP point-cloud profile extracted: raw_points=" << rawPoints.size()
            << ", profile_samples=" << document.profileSamples.size()
            << ", section_samples=" << document.sectionSamples.size()
            << ", axes=" << document.axialAxis << "/" << document.radialAxis;
    if (adjustedAxes) {
        message << " (auto-adjusted from requested axes)";
    }
    document.profileExtractionMessage = message.str();
}

int sampleDivisionCount(const Bnd_Box& box, const double samplingStepMm)
{
    if (box.IsVoid()) {
        return 16;
    }

    double minX = 0.0;
    double minY = 0.0;
    double minZ = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
    double maxZ = 0.0;
    box.Get(minX, minY, minZ, maxX, maxY, maxZ);
    const double dx = maxX - minX;
    const double dy = maxY - minY;
    const double dz = maxZ - minZ;
    const double spanMm = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double safeStepMm = std::max(0.005, samplingStepMm);
    return std::clamp(static_cast<int>(std::ceil(spanMm / safeStepMm)), 8, 512);
}

void appendProfilePoint(std::vector<RawProfilePoint>& output,
                        const gp_Pnt& point,
                        const int axialAxis,
                        const int radialAxis)
{
    const double axial = pointAxisValue(point, axialAxis);
    const double radial = pointAxisValue(point, radialAxis);
    if (!std::isfinite(axial) || !std::isfinite(radial)) {
        return;
    }

    output.push_back({axial, radial, point.X(), point.Y(), point.Z()});
}

void sampleTriangulatedFaces(const TopoDS_Shape& shape,
                             const DesignModelRequest& request,
                             const int axialAxis,
                             const int radialAxis,
                             std::vector<RawProfilePoint>& output)
{
    const double deflectionMm = std::clamp(request.profileSamplingStepMm * 2.0, 0.01, 1.0);
    BRepMesh_IncrementalMesh mesh(shape, deflectionMm);
    mesh.Perform();

    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);
        if (triangulation.IsNull() || triangulation->NbNodes() <= 0) {
            continue;
        }

        const gp_Trsf transform = location.Transformation();
        const int nodeCount = triangulation->NbNodes();
        const int stride = std::max(1, nodeCount / 6000);
        for (int index = 1; index <= nodeCount; index += stride) {
            gp_Pnt point = triangulation->Node(index);
            point.Transform(transform);
            appendProfilePoint(output, point, axialAxis, radialAxis);
        }
    }
}

void extractCadMeshFromShape(const TopoDS_Shape& shape, CadModelDocument& document)
{
    document.meshVertices.clear();
    document.meshTriangles.clear();
    document.hasMesh = false;

    double meshDeflectionMm = 0.2;
    if (document.hasBounds) {
        const double dx = document.maxXmm - document.minXmm;
        const double dy = document.maxYmm - document.minYmm;
        const double dz = document.maxZmm - document.minZmm;
        const double maxSpan = std::max({dx, dy, dz, 1.0});
        meshDeflectionMm = std::clamp(maxSpan / 260.0, 0.03, 1.2);
    }

    BRepMesh_IncrementalMesh mesh(shape, meshDeflectionMm);
    mesh.Perform();

    constexpr std::size_t kMaxGuiTriangles = 180000;
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);
        if (triangulation.IsNull() ||
            triangulation->NbNodes() <= 0 ||
            triangulation->NbTriangles() <= 0) {
            continue;
        }

        const gp_Trsf transform = location.Transformation();
        const int nodeCount = triangulation->NbNodes();
        std::vector<std::size_t> localToGlobal(static_cast<std::size_t>(nodeCount) + 1, 0);
        for (int nodeIndex = 1; nodeIndex <= nodeCount; ++nodeIndex) {
            gp_Pnt point = triangulation->Node(nodeIndex);
            point.Transform(transform);
            localToGlobal[static_cast<std::size_t>(nodeIndex)] = document.meshVertices.size();
            document.meshVertices.push_back({point.X(), point.Y(), point.Z()});
        }

        const int triangleCount = triangulation->NbTriangles();
        for (int triangleIndex = 1; triangleIndex <= triangleCount; ++triangleIndex) {
            int n1 = 0;
            int n2 = 0;
            int n3 = 0;
            const Poly_Triangle triangle = triangulation->Triangle(triangleIndex);
            triangle.Get(n1, n2, n3);
            if (n1 < 1 || n1 > nodeCount ||
                n2 < 1 || n2 > nodeCount ||
                n3 < 1 || n3 > nodeCount) {
                continue;
            }
            document.meshTriangles.push_back({
                localToGlobal[static_cast<std::size_t>(n1)],
                localToGlobal[static_cast<std::size_t>(n2)],
                localToGlobal[static_cast<std::size_t>(n3)],
            });
            if (document.meshTriangles.size() >= kMaxGuiTriangles) {
                break;
            }
        }
        if (document.meshTriangles.size() >= kMaxGuiTriangles) {
            break;
        }
    }

    document.hasMesh = document.meshVertices.size() >= 3 && !document.meshTriangles.empty();
}

void sampleEdges(const TopoDS_Shape& shape,
                 const DesignModelRequest& request,
                 const int axialAxis,
                 const int radialAxis,
                 std::vector<RawProfilePoint>& output)
{
    for (TopExp_Explorer explorer(shape, TopAbs_EDGE); explorer.More(); explorer.Next()) {
        const TopoDS_Edge edge = TopoDS::Edge(explorer.Current());
        BRepAdaptor_Curve curve(edge);
        const double first = curve.FirstParameter();
        const double last = curve.LastParameter();
        if (!std::isfinite(first) || !std::isfinite(last) || last <= first) {
            continue;
        }

        Bnd_Box edgeBox;
        BRepBndLib::Add(edge, edgeBox);
        const int divisions = sampleDivisionCount(edgeBox, request.profileSamplingStepMm);
        for (int index = 0; index <= divisions; ++index) {
            const double t = first + (last - first) * static_cast<double>(index) / static_cast<double>(divisions);
            appendProfilePoint(output, curve.Value(t), axialAxis, radialAxis);
        }
    }
}

bool trySampleSectionEdges(const TopoDS_Shape& shape,
                           const DesignModelRequest& request,
                           const int axialAxis,
                           const int radialAxis,
                           const int normalAxis,
                           double& sectionCoordinateMm,
                           std::vector<RawProfilePoint>& output,
                           std::string& message)
{
    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (box.IsVoid()) {
        message = "section extraction skipped: CAD bounding box is unavailable.";
        return false;
    }

    double minX = 0.0;
    double minY = 0.0;
    double minZ = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
    double maxZ = 0.0;
    box.Get(minX, minY, minZ, maxX, maxY, maxZ);

    double planeX = 0.5 * (minX + maxX);
    double planeY = 0.5 * (minY + maxY);
    double planeZ = 0.5 * (minZ + maxZ);
    const double autoSectionCoordinate =
        coordinateAxisValue(planeX, planeY, planeZ, normalAxis);
    sectionCoordinateMm = std::isfinite(request.sectionCoordinateMm)
                              ? request.sectionCoordinateMm
                              : autoSectionCoordinate;
    if (!std::isfinite(sectionCoordinateMm)) {
        message = "section extraction skipped: section coordinate is invalid.";
        return false;
    }
    setCoordinateAxisValue(planeX, planeY, planeZ, normalAxis, sectionCoordinateMm);

    BRepAlgoAPI_Section section(shape,
                                gp_Pln(gp_Pnt(planeX, planeY, planeZ), directionFromAxisIndex(normalAxis)),
                                false);
    section.Approximation(true);
    section.Build();
    if (!section.IsDone()) {
        message = "section extraction failed: OpenCascade section operation was not completed.";
        return false;
    }

    const TopoDS_Shape sectionShape = section.Shape();
    if (sectionShape.IsNull() || countSubShapes(sectionShape, TopAbs_EDGE) <= 0) {
        message = "section extraction failed: no section edges were produced.";
        return false;
    }

    const std::size_t beforeCount = output.size();
    sampleEdges(sectionShape, request, axialAxis, radialAxis, output);
    const std::size_t producedCount = output.size() - beforeCount;
    if (producedCount < 2) {
        output.resize(beforeCount);
        message = "section extraction failed: section edges produced fewer than two finite samples.";
        return false;
    }

    std::ostringstream stream;
    stream << "section extraction completed: normal_axis=" << axisLabelFromIndex(normalAxis)
           << ", coordinate=" << formatDouble(sectionCoordinateMm)
           << " mm, raw_samples=" << producedCount;
    message = stream.str();
    return true;
}

void updateProfileStats(CadModelDocument& document)
{
    if (document.profileSamples.empty()) {
        return;
    }

    document.profileMinSMm = std::numeric_limits<double>::infinity();
    document.profileMaxSMm = -std::numeric_limits<double>::infinity();
    document.profileMinRMm = std::numeric_limits<double>::infinity();
    document.profileMaxRMm = -std::numeric_limits<double>::infinity();
    for (const auto& sample : document.profileSamples) {
        document.profileMinSMm = std::min(document.profileMinSMm, sample.sMm);
        document.profileMaxSMm = std::max(document.profileMaxSMm, sample.sMm);
        document.profileMinRMm = std::min(document.profileMinRMm, sample.rMm);
        document.profileMaxRMm = std::max(document.profileMaxRMm, sample.rMm);
    }

    document.hasProfileSamples = document.profileSamples.size() >= 2;
    document.profileMetadata.sourceType = "external_cad";
    document.profileMetadata.sourceName = document.modelLabel.empty() ? document.fileName : document.modelLabel;
    document.profileMetadata.sourcePath = document.sourcePath;
    document.profileMetadata.axialAxis = document.axialAxis;
    document.profileMetadata.radialAxis = document.radialAxis;
    document.profileMetadata.sampleCount = document.profileSamples.size();
    document.profileMetadata.minSMm = document.profileMinSMm;
    document.profileMetadata.maxSMm = document.profileMaxSMm;
    document.profileMetadata.minRMm = document.profileMinRMm;
    document.profileMetadata.maxRMm = document.profileMaxRMm;
    document.profileMetadata.hasCadBounds = document.hasBounds;
    if (document.hasBounds) {
        document.profileMetadata.minCadXMm = document.minXmm;
        document.profileMetadata.minCadYMm = document.minYmm;
        document.profileMetadata.minCadZMm = document.minZmm;
        document.profileMetadata.maxCadXMm = document.maxXmm;
        document.profileMetadata.maxCadYMm = document.maxYmm;
        document.profileMetadata.maxCadZMm = document.maxZmm;
    }
}

void extractProfileSamplesFromShape(const TopoDS_Shape& shape,
                                    const DesignModelRequest& request,
                                    CadModelDocument& document)
{
    document.axialAxis = request.axialAxis;
    document.radialAxis = request.radialAxis;
    document.profileSamplingStepMm = request.profileSamplingStepMm;

    const int axialAxis = axisIndexFromLabel(request.axialAxis);
    const int radialAxis = axisIndexFromLabel(request.radialAxis);
    const int normalAxis = remainingAxisIndex(axialAxis, radialAxis);
    if (axialAxis < 0 || radialAxis < 0 || normalAxis < 0 || axialAxis == radialAxis) {
        document.profileExtractionMessage = "CAD profile extraction skipped: axial and radial axes must be valid and different.";
        return;
    }

    std::vector<RawProfilePoint> rawPoints;
    rawPoints.reserve(32768);
    double sectionCoordinateMm = std::numeric_limits<double>::quiet_NaN();
    std::string sectionMessage;
    bool usedSectionExtraction =
        trySampleSectionEdges(shape,
                              request,
                              axialAxis,
                              radialAxis,
                              normalAxis,
                              sectionCoordinateMm,
                              rawPoints,
                              sectionMessage);
    std::string extractionMethod = "occt_section";
    const bool useMicrochannelEdgeProjection =
        request.targetSlotWidthMm > 0.0 &&
        request.targetSlotWidthMm <= 0.10 &&
        !request.extractUpperEnvelope;
    if (useMicrochannelEdgeProjection) {
        std::vector<RawProfilePoint> projectedEdgePoints;
        projectedEdgePoints.reserve(rawPoints.size());
        DesignModelRequest projectionRequest = request;
        projectionRequest.profileSamplingStepMm = std::max(request.profileSamplingStepMm, 0.02);
        sampleEdges(shape, projectionRequest, axialAxis, radialAxis, projectedEdgePoints);
        if (projectedEdgePoints.size() >= 2) {
            rawPoints = std::move(projectedEdgePoints);
            usedSectionExtraction = true;
            extractionMethod = "occt_edge_projection_xoy";
            if (!std::isfinite(sectionCoordinateMm)) {
                if (std::isfinite(request.sectionCoordinateMm)) {
                    sectionCoordinateMm = request.sectionCoordinateMm;
                } else if (document.hasBounds) {
                    switch (normalAxis) {
                    case 0:
                        sectionCoordinateMm = 0.5 * (document.minXmm + document.maxXmm);
                        break;
                    case 1:
                        sectionCoordinateMm = 0.5 * (document.minYmm + document.maxYmm);
                        break;
                    case 2:
                    default:
                        sectionCoordinateMm = 0.5 * (document.minZmm + document.maxZmm);
                        break;
                    }
                } else {
                    sectionCoordinateMm = 0.0;
                }
            }
            std::ostringstream projectionMessage;
            projectionMessage << "edge projection completed for microchannel preview: normal_axis="
                              << axisLabelFromIndex(normalAxis)
                              << ", coordinate=" << formatDouble(sectionCoordinateMm)
                              << " mm, raw_samples=" << rawPoints.size();
            sectionMessage = projectionMessage.str();
        }
    }
    if (!usedSectionExtraction) {
        rawPoints.clear();
        sampleTriangulatedFaces(shape, request, axialAxis, radialAxis, rawPoints);
        sampleEdges(shape, request, axialAxis, radialAxis, rawPoints);
        extractionMethod = "fallback_projected_envelope";
    }
    if (rawPoints.size() < 2) {
        document.profileExtractionMessage = "CAD profile extraction failed: no finite topology samples were produced.";
        if (!sectionMessage.empty()) {
            document.profileExtractionMessage += " Section note: " + sectionMessage;
        }
        return;
    }

    double minAxial = std::numeric_limits<double>::infinity();
    double maxAxial = -std::numeric_limits<double>::infinity();
    for (const RawProfilePoint& point : rawPoints) {
        minAxial = std::min(minAxial, point.axialMm);
        maxAxial = std::max(maxAxial, point.axialMm);
    }
    if (!std::isfinite(minAxial) || !std::isfinite(maxAxial) || maxAxial <= minAxial) {
        document.profileExtractionMessage = "CAD profile extraction failed: axial range is invalid.";
        return;
    }

    const double binWidthMm = std::max(0.001, request.profileSamplingStepMm);
    std::map<long long, ProfileBin> bins;
    const double cadAxialDirectionSign = request.reverseAxialDirection ? -1.0 : 1.0;
    double cadAxialOriginMm = request.reverseAxialDirection ? maxAxial : minAxial;
    document.sectionSamples.clear();
    document.sectionSamples.reserve(rawPoints.size());
    for (const RawProfilePoint& raw : rawPoints) {
        const double sMm = request.reverseAxialDirection ? (maxAxial - raw.axialMm) : (raw.axialMm - minAxial);
        if (!std::isfinite(sMm) || !std::isfinite(raw.radialMm) || sMm < -1e-9) {
            continue;
        }

        if (usedSectionExtraction) {
            pinjie::cad_design::DesignProfileSample sectionSample;
            sectionSample.sMm = sMm;
            sectionSample.rMm = raw.radialMm;
            sectionSample.cadXMm = raw.xMm;
            sectionSample.cadYMm = raw.yMm;
            sectionSample.cadZMm = raw.zMm;
            sectionSample.hasCadPoint = std::isfinite(sectionSample.cadXMm) &&
                                        std::isfinite(sectionSample.cadYMm) &&
                                        std::isfinite(sectionSample.cadZMm);
            if (sectionSample.hasCadPoint) {
                document.sectionSamples.push_back(sectionSample);
            }
        }

        const long long binIndex = static_cast<long long>(std::floor(sMm / binWidthMm + 1e-9));
        ProfileBin& bin = bins[binIndex];
        ++bin.count;
        bin.sSumMm += sMm;
        bin.xSumMm += raw.xMm;
        bin.ySumMm += raw.yMm;
        bin.zSumMm += raw.zMm;
        if (raw.radialMm > bin.bestRMm) {
            bin.bestRMm = raw.radialMm;
            bin.bestSMm = sMm;
            bin.bestXMm = raw.xMm;
            bin.bestYMm = raw.yMm;
            bin.bestZMm = raw.zMm;
        }
        if (!request.extractUpperEnvelope) {
            bin.radiiMm.push_back(raw.radialMm);
        }
    }

    document.profileSamples.clear();
    document.profileSamples.reserve(bins.size());
    for (auto& [binIndex, bin] : bins) {
        (void)binIndex;
        if (bin.count == 0) {
            continue;
        }

        pinjie::cad_design::DesignProfileSample sample;
        if (request.extractUpperEnvelope) {
            sample.sMm = bin.bestSMm;
            sample.rMm = bin.bestRMm;
            sample.cadXMm = bin.bestXMm;
            sample.cadYMm = bin.bestYMm;
            sample.cadZMm = bin.bestZMm;
        } else {
            sample.sMm = bin.sSumMm / static_cast<double>(bin.count);
            sample.rMm = medianValue(std::move(bin.radiiMm));
            sample.cadXMm = bin.xSumMm / static_cast<double>(bin.count);
            sample.cadYMm = bin.ySumMm / static_cast<double>(bin.count);
            sample.cadZMm = bin.zSumMm / static_cast<double>(bin.count);
        }

        if (std::isfinite(sample.sMm) && std::isfinite(sample.rMm)) {
            sample.hasCadPoint = std::isfinite(sample.cadXMm) &&
                                 std::isfinite(sample.cadYMm) &&
                                 std::isfinite(sample.cadZMm);
            document.profileSamples.push_back(sample);
        }
    }

    std::stable_sort(document.profileSamples.begin(), document.profileSamples.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.sMm != rhs.sMm) {
            return lhs.sMm < rhs.sMm;
        }
        return lhs.rMm < rhs.rMm;
    });

    std::vector<pinjie::cad_design::DesignProfileSample> uniqueSamples;
    uniqueSamples.reserve(document.profileSamples.size());
    for (const auto& sample : document.profileSamples) {
        if (!uniqueSamples.empty() && std::abs(sample.sMm - uniqueSamples.back().sMm) < 1e-6) {
            if (request.extractUpperEnvelope && sample.rMm > uniqueSamples.back().rMm) {
                uniqueSamples.back().rMm = sample.rMm;
                uniqueSamples.back().hasCadPoint = sample.hasCadPoint;
                uniqueSamples.back().cadXMm = sample.cadXMm;
                uniqueSamples.back().cadYMm = sample.cadYMm;
                uniqueSamples.back().cadZMm = sample.cadZMm;
            }
            continue;
        }
        uniqueSamples.push_back(sample);
    }
    document.profileSamples = std::move(uniqueSamples);

    if (document.profileSamples.size() < 2) {
        document.profileExtractionMessage = "CAD profile extraction failed: fewer than two envelope samples remained.";
        return;
    }

    const double sOffsetMm = document.profileSamples.front().sMm;
    if (std::isfinite(sOffsetMm) && std::abs(sOffsetMm) > 1e-12) {
        for (auto& sample : document.profileSamples) {
            sample.sMm -= sOffsetMm;
        }
        cadAxialOriginMm += cadAxialDirectionSign * sOffsetMm;
    }
    if (std::isfinite(sOffsetMm) && std::abs(sOffsetMm) > 1e-12) {
        for (auto& sample : document.sectionSamples) {
            sample.sMm -= sOffsetMm;
        }
    }

    document.hasSectionSamples = document.sectionSamples.size() >= 2;
    updateProfileStats(document);
    document.profileMetadata.cadAxialOriginMm = cadAxialOriginMm;
    document.profileMetadata.cadAxialDirectionSign = cadAxialDirectionSign;
    document.profileMetadata.extractionMethod = extractionMethod;
    document.profileMetadata.sectionNormalAxis = axisLabelFromIndex(normalAxis);
    document.profileMetadata.sectionCoordinateMm =
        std::isfinite(sectionCoordinateMm)
            ? sectionCoordinateMm
            : std::numeric_limits<double>::quiet_NaN();
    std::ostringstream message;
    message << "CAD profile extracted: samples=" << document.profileSamples.size()
            << ", method=" << extractionMethod
            << ", s=[" << formatDouble(document.profileMinSMm) << ", "
            << formatDouble(document.profileMaxSMm) << "] mm"
            << ", r=[" << formatDouble(document.profileMinRMm) << ", "
            << formatDouble(document.profileMaxRMm) << "] mm";
    if (!sectionMessage.empty()) {
        message << ", section_note=" << sectionMessage;
    }
    document.profileExtractionMessage = message.str();
}

CadModelLoadResult loadStepPointCloudFallback(const std::filesystem::path& path,
                                              const DesignModelRequest& request,
                                              const std::string& occtFailureNote)
{
    CadModelLoadResult result;
    result.document.occtEnabled = true;
    result.document.format = CadFileFormat::Step;
    result.document.sourcePath = pinjie::genericUtf8String(path);
    result.document.fileName = pinjie::genericUtf8String(path.filename());
    result.document.modelLabel =
        request.modelName.empty() ? pinjie::genericUtf8String(path.stem()) : request.modelName;
    result.document.axialAxis = request.axialAxis;
    result.document.radialAxis = request.radialAxis;
    result.document.profileSamplingStepMm = request.profileSamplingStepMm;

    std::vector<CartesianPoint> points;
    std::string parseMessage;
    if (!parseStepPointCloud(path, points, parseMessage)) {
        result.message = "CAD import failed: " + occtFailureNote + "; " + parseMessage;
        result.document.importMessage = result.message;
        return result;
    }

    result.document.valid = true;
    result.document.rootShapeCount = 1;
    result.document.solidCount = 0;
    result.document.shellCount = 0;
    result.document.faceCount = 0;
    result.document.edgeCount = 0;
    updatePointCloudBounds(points, result.document);

    result.document.meshVertices.clear();
    result.document.meshVertices.reserve(points.size());
    for (const CartesianPoint& point : points) {
        result.document.meshVertices.push_back({point.xMm, point.yMm, point.zMm});
    }
    result.document.meshTriangles.clear();
    result.document.hasMesh = !result.document.meshVertices.empty();

    populateProfileSamplesFromPointCloud(points, request, result.document);

    std::ostringstream message;
    message << "CAD import completed via STEP point-cloud fallback: "
            << parseMessage
            << ", original_reader_note=" << occtFailureNote;
    if (result.document.hasBounds) {
        message << ", bounds=[" << formatDouble(result.document.minXmm) << ", "
                << formatDouble(result.document.minYmm) << ", "
                << formatDouble(result.document.minZmm) << "] -> ["
                << formatDouble(result.document.maxXmm) << ", "
                << formatDouble(result.document.maxYmm) << ", "
                << formatDouble(result.document.maxZmm) << "]";
    }
    if (result.document.hasProfileSamples) {
        message << ", profile_samples=" << result.document.profileSamples.size();
    }
    if (result.document.hasSectionSamples) {
        message << ", point_cloud_samples=" << result.document.sectionSamples.size();
    }
    if (!result.document.profileExtractionMessage.empty()) {
        message << ", " << result.document.profileExtractionMessage;
    }

    result.ok = true;
    result.message = message.str();
    result.document.importMessage = result.message;
    return result;
}

CadModelLoadResult loadBinaryStlMeshDocument(const std::filesystem::path& path,
                                             const DesignModelRequest& request)
{
    CadModelLoadResult result;
    result.document.occtEnabled = true;
    result.document.format = CadFileFormat::Stl;
    result.document.sourcePath = pinjie::genericUtf8String(path);
    result.document.fileName = pinjie::genericUtf8String(path.filename());
    result.document.modelLabel =
        request.modelName.empty() ? pinjie::genericUtf8String(path.stem()) : request.modelName;
    result.document.axialAxis = request.axialAxis;
    result.document.radialAxis = request.radialAxis;
    result.document.profileSamplingStepMm = request.profileSamplingStepMm;

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        result.message = "CAD import failed: STL file cannot be opened.";
        result.document.importMessage = result.message;
        return result;
    }

    const std::streamoff fileSize = stream.tellg();
    if (fileSize < 84) {
        result.message = "CAD import failed: STL file is too small to be a valid binary STL.";
        result.document.importMessage = result.message;
        return result;
    }
    stream.seekg(0, std::ios::beg);

    char header[80]{};
    stream.read(header, sizeof(header));
    std::uint32_t triangleCount = 0;
    stream.read(reinterpret_cast<char*>(&triangleCount), sizeof(triangleCount));
    if (!stream) {
        result.message = "CAD import failed: STL header cannot be parsed.";
        result.document.importMessage = result.message;
        return result;
    }

    const std::uint64_t expectedSize =
        static_cast<std::uint64_t>(84) + static_cast<std::uint64_t>(triangleCount) * static_cast<std::uint64_t>(50);
    if (static_cast<std::uint64_t>(fileSize) < expectedSize) {
        result.message =
            "CAD import failed: STL file size does not match the binary STL triangle count.";
        result.document.importMessage = result.message;
        return result;
    }

    constexpr std::size_t kMaxDisplayTriangles = 60000;
    constexpr std::size_t kMaxProfilePoints = 250000;
    const std::size_t displayStride =
        std::max<std::size_t>(1, (static_cast<std::size_t>(triangleCount) + kMaxDisplayTriangles - 1) /
                                     kMaxDisplayTriangles);
    const std::size_t profileStride =
        std::max<std::size_t>(1, (static_cast<std::size_t>(triangleCount) * 3 + kMaxProfilePoints - 1) /
                                     kMaxProfilePoints);

    std::vector<CartesianPoint> sampledProfilePoints;
    sampledProfilePoints.reserve(
        std::min<std::size_t>(static_cast<std::size_t>(triangleCount) * 3, kMaxProfilePoints));
    result.document.meshVertices.reserve(
        std::min<std::size_t>(static_cast<std::size_t>(triangleCount) / displayStride * 3,
                              kMaxDisplayTriangles * 3));
    result.document.meshTriangles.reserve(
        std::min<std::size_t>(static_cast<std::size_t>(triangleCount), kMaxDisplayTriangles));

    result.document.minXmm = std::numeric_limits<double>::infinity();
    result.document.minYmm = std::numeric_limits<double>::infinity();
    result.document.minZmm = std::numeric_limits<double>::infinity();
    result.document.maxXmm = -std::numeric_limits<double>::infinity();
    result.document.maxYmm = -std::numeric_limits<double>::infinity();
    result.document.maxZmm = -std::numeric_limits<double>::infinity();

    std::size_t sampledTriangleCount = 0;
    std::size_t sampledPointCounter = 0;
    const auto updateBounds = [&result](const CartesianPoint& point) {
        result.document.minXmm = std::min(result.document.minXmm, point.xMm);
        result.document.minYmm = std::min(result.document.minYmm, point.yMm);
        result.document.minZmm = std::min(result.document.minZmm, point.zMm);
        result.document.maxXmm = std::max(result.document.maxXmm, point.xMm);
        result.document.maxYmm = std::max(result.document.maxYmm, point.yMm);
        result.document.maxZmm = std::max(result.document.maxZmm, point.zMm);
    };

    for (std::uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        float normal[3]{};
        float coordinates[9]{};
        std::uint16_t attributeByteCount = 0;
        stream.read(reinterpret_cast<char*>(normal), sizeof(normal));
        stream.read(reinterpret_cast<char*>(coordinates), sizeof(coordinates));
        stream.read(reinterpret_cast<char*>(&attributeByteCount), sizeof(attributeByteCount));
        if (!stream) {
            result.message = "CAD import failed: STL triangle payload is truncated.";
            result.document.importMessage = result.message;
            return result;
        }

        std::array<CartesianPoint, 3> vertices{};
        bool validTriangle = true;
        for (int vertexIndex = 0; vertexIndex < 3; ++vertexIndex) {
            const CartesianPoint point{
                static_cast<double>(coordinates[vertexIndex * 3 + 0]),
                static_cast<double>(coordinates[vertexIndex * 3 + 1]),
                static_cast<double>(coordinates[vertexIndex * 3 + 2]),
            };
            if (!std::isfinite(point.xMm) || !std::isfinite(point.yMm) || !std::isfinite(point.zMm)) {
                validTriangle = false;
                break;
            }
            vertices[vertexIndex] = point;
            updateBounds(point);
            if (sampledPointCounter % profileStride == 0) {
                sampledProfilePoints.push_back(point);
            }
            ++sampledPointCounter;
        }

        if (!validTriangle) {
            continue;
        }

        if (triangleIndex % displayStride == 0) {
            const std::size_t baseIndex = result.document.meshVertices.size();
            for (const CartesianPoint& point : vertices) {
                result.document.meshVertices.push_back({point.xMm, point.yMm, point.zMm});
            }
            result.document.meshTriangles.push_back({baseIndex, baseIndex + 1, baseIndex + 2});
            ++sampledTriangleCount;
        }
    }

    result.document.hasBounds =
        std::isfinite(result.document.minXmm) &&
        std::isfinite(result.document.minYmm) &&
        std::isfinite(result.document.minZmm) &&
        std::isfinite(result.document.maxXmm) &&
        std::isfinite(result.document.maxYmm) &&
        std::isfinite(result.document.maxZmm);
    result.document.hasMesh =
        !result.document.meshVertices.empty() && !result.document.meshTriangles.empty();

    if (sampledProfilePoints.size() < 2 || !result.document.hasBounds) {
        result.message = "CAD import failed: STL sampling did not produce enough finite points.";
        result.document.importMessage = result.message;
        return result;
    }

    result.document.valid = true;
    result.document.rootShapeCount = 1;
    result.document.solidCount = 0;
    result.document.shellCount = 1;
    result.document.faceCount = static_cast<int>(triangleCount);
    result.document.edgeCount = 0;

    populateProfileSamplesFromPointCloud(sampledProfilePoints, request, result.document);
    if (!result.document.profileExtractionMessage.empty()) {
        const auto replaceToken = [](std::string& text, const std::string& from, const std::string& to) {
            const std::size_t position = text.find(from);
            if (position != std::string::npos) {
                text.replace(position, from.size(), to);
            }
        };
        replaceToken(result.document.profileExtractionMessage, "STEP point-cloud", "STL point-cloud");
        replaceToken(result.document.profileExtractionMessage,
                     "stp_cartesian_point_fallback",
                     "stl_mesh_vertex_sampling");
        result.document.profileMetadata.extractionMethod = "stl_mesh_vertex_sampling";
    }
    if (result.document.profileMetadata.sourceType == "external_cad") {
        result.document.profileMetadata.sourceName =
            result.document.modelLabel.empty() ? result.document.fileName : result.document.modelLabel;
    }

    std::ostringstream message;
    message << "CAD import completed via binary STL sampling: triangles=" << triangleCount
            << ", display_triangles=" << sampledTriangleCount
            << ", profile_points=" << sampledProfilePoints.size();
    if (result.document.hasProfileSamples) {
        message << ", profile_samples=" << result.document.profileSamples.size();
    }
    if (result.document.hasSectionSamples) {
        message << ", section_samples=" << result.document.sectionSamples.size();
    }
    if (result.document.hasBounds) {
        message << ", bounds=[" << formatDouble(result.document.minXmm) << ", "
                << formatDouble(result.document.minYmm) << ", "
                << formatDouble(result.document.minZmm) << "] -> ["
                << formatDouble(result.document.maxXmm) << ", "
                << formatDouble(result.document.maxYmm) << ", "
                << formatDouble(result.document.maxZmm) << "]";
    }
    if (!result.document.profileExtractionMessage.empty()) {
        message << ", " << result.document.profileExtractionMessage;
    }

    result.ok = true;
    result.message = message.str();
    result.document.importMessage = result.message;
    return result;
}

template <typename Reader>
CadModelLoadResult loadShapeWithReader(Reader& reader,
                                        const std::filesystem::path& path,
                                        const CadFileFormat format,
                                        const DesignModelRequest& request)
{
    CadModelLoadResult result;
    result.document.occtEnabled = true;
    result.document.format = format;
    result.document.sourcePath = pinjie::genericUtf8String(path);
    result.document.fileName = pinjie::genericUtf8String(path.filename());
    result.document.modelLabel =
        request.modelName.empty() ? pinjie::genericUtf8String(path.stem()) : request.modelName;
    result.document.axialAxis = request.axialAxis;
    result.document.radialAxis = request.radialAxis;

    const CadReaderPathAlias readPathAlias = prepareCadReaderPathAlias(path, format);
    const std::string readPath = pinjie::genericUtf8String(readPathAlias.readPath);
    const IFSelect_ReturnStatus status = reader.ReadFile(readPath.c_str());
    if (status != IFSelect_RetDone) {
        const std::string statusMessage = ifSelectStatusMessage(status);
        if (format == CadFileFormat::Step) {
            CadModelLoadResult fallback =
                loadStepPointCloudFallback(path, request, "OpenCascade read failed: " + statusMessage);
            if (fallback.ok) {
                return fallback;
            }
        }
        result.message = "CAD import failed: " + statusMessage;
        result.document.importMessage = result.message;
        return result;
    }

    const int transferredRootCount = static_cast<int>(reader.TransferRoots());
    const TopoDS_Shape shape = reader.OneShape();
    if (transferredRootCount <= 0 || shape.IsNull()) {
        if (format == CadFileFormat::Step) {
            CadModelLoadResult fallback =
                loadStepPointCloudFallback(path,
                                           request,
                                           "OpenCascade transfer failed: no transferable root shape was produced");
            if (fallback.ok) {
                return fallback;
            }
        }
        result.message = "CAD import failed: no transferable root shape was produced.";
        result.document.importMessage = result.message;
        return result;
    }

    result.document.valid = true;
    result.document.rootShapeCount = transferredRootCount;
    result.document.solidCount = countSubShapes(shape, TopAbs_SOLID);
    result.document.shellCount = countSubShapes(shape, TopAbs_SHELL);
    result.document.faceCount = countSubShapes(shape, TopAbs_FACE);
    result.document.edgeCount = countSubShapes(shape, TopAbs_EDGE);

    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (!box.IsVoid()) {
        box.Get(result.document.minXmm,
                result.document.minYmm,
                result.document.minZmm,
                result.document.maxXmm,
                result.document.maxYmm,
                result.document.maxZmm);
        result.document.hasBounds = true;
    }

    extractCadMeshFromShape(shape, result.document);
    extractProfileSamplesFromShape(shape, request, result.document);

    std::ostringstream message;
    message << "CAD import completed via OpenCascade: "
            << cadFileFormatLabel(format)
            << ", solids=" << result.document.solidCount
            << ", faces=" << result.document.faceCount
            << ", edges=" << result.document.edgeCount;
    if (result.document.hasProfileSamples) {
        message << ", profile_samples=" << result.document.profileSamples.size();
    }
    if (result.document.hasSectionSamples) {
        message << ", section_samples=" << result.document.sectionSamples.size();
    }
    if (result.document.hasMesh) {
        message << ", mesh_vertices=" << result.document.meshVertices.size()
                << ", mesh_triangles=" << result.document.meshTriangles.size();
    }
    if (result.document.hasBounds) {
        message << ", bounds=[" << formatDouble(result.document.minXmm) << ", "
                << formatDouble(result.document.minYmm) << ", "
                << formatDouble(result.document.minZmm) << "] -> ["
                << formatDouble(result.document.maxXmm) << ", "
                << formatDouble(result.document.maxYmm) << ", "
                << formatDouble(result.document.maxZmm) << "]";
    }
    if (!result.document.profileExtractionMessage.empty()) {
        message << ", " << result.document.profileExtractionMessage;
    }
    if (!readPathAlias.note.empty()) {
        message << ", " << readPathAlias.note;
    }

    result.ok = true;
    result.message = message.str();
    result.document.importMessage = result.message;
    return result;
}
#endif

} // namespace

bool occtCadImportAvailable()
{
#if PINJIE_HAS_OCCT
    return true;
#else
    return false;
#endif
}

std::string occtCadImportBackendSummary()
{
#if PINJIE_HAS_OCCT
    return "OpenCascade/OCCT enabled";
#else
    return "OpenCascade/OCCT unavailable in current build";
#endif
}

CadFileFormat detectCadFileFormat(const std::filesystem::path& path)
{
    const std::string extension = toLowerCopy(path.extension().string());
    if (extension == ".step" || extension == ".stp") {
        return CadFileFormat::Step;
    }
    if (extension == ".iges" || extension == ".igs") {
        return CadFileFormat::Iges;
    }
    if (extension == ".stl") {
        return CadFileFormat::Stl;
    }
    return CadFileFormat::Unknown;
}

std::string cadFileFormatLabel(const CadFileFormat format)
{
    switch (format) {
    case CadFileFormat::Step:
        return "STEP";
    case CadFileFormat::Iges:
        return "IGES";
    case CadFileFormat::Stl:
        return "STL";
    case CadFileFormat::Unknown:
    default:
        return "Unknown";
    }
}

CadModelLoadResult loadCadModelDocument(const std::filesystem::path& path)
{
    DesignModelRequest request;
    request.cadFilePath = pinjie::genericUtf8String(path);
    request.modelName = pinjie::genericUtf8String(path.stem());
    return loadCadModelDocument(path, request);
}

CadModelLoadResult loadCadModelDocument(const std::filesystem::path& path, const DesignModelRequest& request)
{
    CadModelLoadResult result;
    result.document.sourcePath = pinjie::genericUtf8String(path);
    result.document.fileName = pinjie::genericUtf8String(path.filename());
    result.document.modelLabel =
        request.modelName.empty() ? pinjie::genericUtf8String(path.stem()) : request.modelName;
    result.document.format = detectCadFileFormat(path);
    result.document.occtEnabled = occtCadImportAvailable();
    result.document.axialAxis = request.axialAxis;
    result.document.radialAxis = request.radialAxis;
    result.document.profileSamplingStepMm = request.profileSamplingStepMm;

    if (path.empty()) {
        result.message = "CAD import failed: source path is empty.";
        result.document.importMessage = result.message;
        return result;
    }
    if (!std::filesystem::exists(path)) {
        result.message = "CAD import failed: source file does not exist.";
        result.document.importMessage = result.message;
        return result;
    }
    if (result.document.format == CadFileFormat::Unknown) {
        result.message = "CAD import failed: supported formats are STEP/STP, IGES and STL.";
        result.document.importMessage = result.message;
        return result;
    }

#if PINJIE_HAS_OCCT
    if (result.document.format == CadFileFormat::Stl) {
        return loadBinaryStlMeshDocument(path, request);
    }
    if (result.document.format == CadFileFormat::Step) {
        STEPControl_Reader reader;
        return loadShapeWithReader(reader, path, CadFileFormat::Step, request);
    }
    if (result.document.format == CadFileFormat::Iges) {
        IGESControl_Reader reader;
        return loadShapeWithReader(reader, path, CadFileFormat::Iges, request);
    }
#endif

    result.message = "CAD import failed: OpenCascade/OCCT support is not available in this build.";
    result.document.importMessage = result.message;
    return result;
}

std::string buildCadModelSummaryText(const CadModelDocument& document)
{
    std::ostringstream stream;
    stream << "CAD backend: " << occtCadImportBackendSummary() << "\n";
    stream << "Source: " << (document.sourcePath.empty() ? "N/A" : document.sourcePath) << "\n";
    stream << "Format: " << cadFileFormatLabel(document.format) << "\n";
    stream << "Model label: " << (document.modelLabel.empty() ? "N/A" : document.modelLabel) << "\n";
    stream << "Coordinate axes: axial=" << document.axialAxis << ", radial=" << document.radialAxis << "\n";
    stream << "Transferred roots: " << document.rootShapeCount << "\n";
    stream << "Solids/Shells/Faces/Edges: "
           << document.solidCount << " / "
           << document.shellCount << " / "
           << document.faceCount << " / "
           << document.edgeCount << "\n";
    if (document.hasBounds) {
        stream << "Bounding box (model units, assumed mm): ["
               << formatDouble(document.minXmm) << ", "
               << formatDouble(document.minYmm) << ", "
               << formatDouble(document.minZmm) << "] -> ["
               << formatDouble(document.maxXmm) << ", "
               << formatDouble(document.maxYmm) << ", "
               << formatDouble(document.maxZmm) << "]\n";
    } else {
        stream << "Bounding box: unavailable\n";
    }
    if (document.hasMesh) {
        stream << "CAD display points/mesh: "
               << document.meshVertices.size() << " vertices, "
               << document.meshTriangles.size() << " triangles\n";
    } else {
        stream << "CAD display points/mesh: unavailable\n";
    }
    if (document.hasProfileSamples) {
        if (document.hasSectionSamples) {
            stream << "True CAD section preview samples: " << document.sectionSamples.size()
                   << " raw section points\n";
        } else {
            stream << "True CAD section preview samples: unavailable\n";
        }
        stream << "Design comparison profile samples: " << document.profileSamples.size()
               << " points, step=" << formatDouble(document.profileSamplingStepMm, 4)
               << " mm, method=" << document.profileMetadata.extractionMethod
               << ", s=[" << formatDouble(document.profileMinSMm) << ", "
               << formatDouble(document.profileMaxSMm) << "] mm, r=["
               << formatDouble(document.profileMinRMm) << ", "
               << formatDouble(document.profileMaxRMm) << "] mm\n";
        if (!document.profileMetadata.sectionNormalAxis.empty()) {
            stream << "CAD section plane: normal=" << document.profileMetadata.sectionNormalAxis
                   << ", coordinate=" << formatDouble(document.profileMetadata.sectionCoordinateMm)
                   << " mm\n";
        }
        stream << "CAD coordinate mapping: axial=" << document.profileMetadata.axialAxis
               << ", radial=" << document.profileMetadata.radialAxis
               << ", axial_origin=" << formatDouble(document.profileMetadata.cadAxialOriginMm)
               << " mm, axial_sign=" << formatDouble(document.profileMetadata.cadAxialDirectionSign, 0)
               << "\n";
    } else {
        stream << "Design comparison profile samples: unavailable\n";
    }
    if (!document.profileExtractionMessage.empty()) {
        stream << "Profile note: " << document.profileExtractionMessage << "\n";
    }
    if (!document.importMessage.empty()) {
        stream << "Import note: " << document.importMessage << "\n";
    }
    return stream.str();
}

} // namespace pinjie::cad_model
