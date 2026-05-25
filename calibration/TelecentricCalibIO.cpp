/*
==============================================================================
文件：TelecentricCalibIO.cpp
------------------------------------------------------------------------------
实现 init 文件解析与写入。

读取策略（兼容旧格式）：
  - 逐行扫描，遇到 intr / dist 就解析
  - 其它行（例如 image / rvec / tvec）会被忽略
这样做的好处：
  - 你仍然可以继续沿用旧工程生成的 init 文件
  - 文件里即使多了其它信息也不会影响解析

==============================================================================
*/

#include "TelecentricCalibIO.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace telecentric {
namespace fs = std::filesystem;

static std::string Trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool LoadFromInitTxt(const std::string& path, CalibParams& out, std::string* err) {
    std::ifstream in(fs::u8path(path));
    if (!in) {
        if (err) *err = "Failed to open: " + path;
        return false;
    }
    bool has_intr=false, has_dist=false;
    std::string line;
    while (std::getline(in, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "intr") {
            double v[5];
            if (iss >> v[0] >> v[1] >> v[2] >> v[3] >> v[4]) {
                for (int i=0;i<5;++i) out.intr[i]=v[i];
                has_intr=true;
            }
        } else if (tag == "dist") {
            double v[5];
            if (iss >> v[0] >> v[1] >> v[2] >> v[3] >> v[4]) {
                for (int i=0;i<5;++i) out.dist[i]=v[i];
                has_dist=true;
            }
        } else if (tag == "image_residual_quad") {
            double v[6];
            if (iss >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]) {
                for (int i=0;i<6;++i) out.image_residual_quad[i]=v[i];
            }
        }
    }
    if (!has_intr || !has_dist) {
        if (err) *err = "Init file missing intr/dist line: " + path;
        return false;
    }
    return true;
}

bool SaveToInitTxt(const std::string& path, const CalibParams& p, std::string* err) {
    std::ofstream outF(fs::u8path(path));
    if (!outF) {
        if (err) *err = "Failed to write: " + path;
        return false;
    }
    outF << "# telecentric_calib_init v2\n";
    outF << std::setprecision(17);
    outF << "intr " << p.intr[0] << " " << p.intr[1] << " " << p.intr[2] << " "
         << p.intr[3] << " " << p.intr[4] << "\n";
    outF << "dist " << p.dist[0] << " " << p.dist[1] << " " << p.dist[2] << " "
         << p.dist[3] << " " << p.dist[4] << "\n";
    outF << "image_residual_quad "
         << p.image_residual_quad[0] << " " << p.image_residual_quad[1] << " " << p.image_residual_quad[2] << " "
         << p.image_residual_quad[3] << " " << p.image_residual_quad[4] << " " << p.image_residual_quad[5] << "\n";
    return true;
}

} // namespace telecentric
