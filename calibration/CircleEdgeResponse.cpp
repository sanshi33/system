#include "CircleEdgeResponse.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace telecentric {

namespace {

bool WriteTextFile(const fs::path& path, const std::string& text, std::string* err) {
    std::error_code ec;
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path(), ec);
        if (ec) {
            if (err) *err = "Failed to create directory: " + path.parent_path().u8string();
            return false;
        }
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "Failed to open file for write: " + path.u8string();
        return false;
    }
    out << text;
    return true;
}

fs::path ResolveOutputPath(const std::string& configured, const std::string& init_path) {
    fs::path out = configured.empty() ? fs::path() : fs::u8path(configured);
    if (out.empty()) return {};
    if (out.is_absolute()) return out;

    const fs::path init = fs::u8path(init_path);
    if (!init.empty() && init.has_parent_path()) {
        return init.parent_path() / out;
    }
    return out;
}

} // namespace

bool RunCircleEdgeResponseCalibration(
    const std::vector<std::string>& image_files,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size,
    const CalibOptions& opt,
    EdgeResponseResult& out,
    std::string* err) {
    out = EdgeResponseResult{};
    out.stats_erf_bg.total_count = static_cast<int>(image_files.size());
    out.stats_erf_integral.total_count = static_cast<int>(image_files.size());

    const std::string message =
        "CircleEdgeResponse module is currently unavailable in this workspace. "
        "The calibration pipeline was rebuilt with a compile-safe placeholder so the "
        "default geometric calibration can still run. Reintroduce the original "
        "CircleEdgeResponse implementation before enabling edge-response calibration.";

    std::ostringstream txt;
    txt << "# Circle edge-response calibration placeholder\n"
        << "status=unavailable\n"
        << "reason=" << message << "\n"
        << "image_count=" << image_files.size() << "\n"
        << "point_set_count=" << image_points.size() << "\n"
        << "image_width=" << image_size.width << "\n"
        << "image_height=" << image_size.height << "\n";

    std::ostringstream csv;
    csv << "model,status,reason,total_count,valid_count,sigma0\n";
    csv << "ERF_BG,unavailable,\"" << message << "\"," << image_files.size() << ",0,0\n";
    csv << "ERF_Integral_BG,unavailable,\"" << message << "\"," << image_files.size() << ",0,0\n";

    const fs::path txt_path = ResolveOutputPath(opt.edge_output_txt, opt.init_path);
    const fs::path csv_path = ResolveOutputPath(opt.edge_summary_csv, opt.init_path);

    std::string io_err;
    if (!txt_path.empty() && !WriteTextFile(txt_path, txt.str(), &io_err)) {
        if (err) *err = io_err;
        return false;
    }
    if (!csv_path.empty() && !WriteTextFile(csv_path, csv.str(), &io_err)) {
        if (err) *err = io_err;
        return false;
    }

    if (err) *err = message;
    return false;
}

} // namespace telecentric
