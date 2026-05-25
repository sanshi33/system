#include "Evaluation.h"
#include "CircleFit.h"
#include "CalibrationShared.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>

#include "../core/SubpixelEdgeDetector.h"

namespace stitch {
using namespace cv;
using namespace std;
using std::cout;
using std::endl;

struct CameraCalib {
    cv::Matx33d K;
    cv::Vec4d dist; // k1, k2, p1, p2
};

static CameraCalib getDefaultCameraCalib()
{
    CameraCalib c;
    c.K = cv::Matx33d(99.41695018645437, -0.01324714602184987, 0.0,
                      0.0,                 99.41695018645437, 0.0,
                      0.0,                  0.0,               1.0);
    c.dist = cv::Vec4d(3.308394140678505e-07, -9.94996961948438e-11, 0.0, 0.0);
    return c;
}

RealEvalResult evaluateCircleFit(const std::vector<cv::Point2d>& pts,
                                 const std::string& imageName,
                                 const std::string& algoName)
{
    RealEvalResult out;
    out.imageName = imageName;
    out.algoName = algoName;
    out.points = static_cast<int>(pts.size());

    if (pts.size() < 20) return out;

    cv::Point2d c;
    double r = 0.0;
    std::vector<int> inliers;
    if (!ransacCircleFit(pts, c, r, inliers, /*iterations=*/300, /*th=*/2.0, /*min_inliers=*/20)) {
        if (!kasaCircleFit(pts, c, r)) return out;
    }

    double se = 0.0;
    double maxe = 0.0;
    size_t eval_count = (!inliers.empty()) ? inliers.size() : pts.size();
    if (eval_count == 0) return out;

    auto eval_error = [&](const cv::Point2d& p) {
        double e = std::abs(cv::norm(p - c) - r);
        se += e * e;
        maxe = std::max(maxe, e);
    };

    if (!inliers.empty()) {
        for (int idx : inliers) eval_error(pts[idx]);
    } else {
        for (const auto& p : pts) eval_error(p);
    }

    out.rmse = std::sqrt(se / static_cast<double>(eval_count));
    out.maxErr = maxe;
    out.radius = r;
    out.center = c;
    return out;
}

RealEvalResult runDetectionOnImage(const cv::Mat& img,
                                   const std::string& imageName,
                                   const EdgeDetectConfig& cfg)
{
    SubpixelEdgeDetector det;
    det.setCannyThresholds(cfg.cannyLow, cfg.cannyHigh);
    det.setImage(img);
    det.refineEdgesSubpixel(cfg.subpixWindow, cfg.subpixSigma);

    const auto& pts = det.getSubpixelEdges();
    return evaluateCircleFit(pts, imageName, "ERF");
}

void runRealBallEvaluation(const std::string& realBallDir,
                           const EdgeDetectConfig& cfg,
                           double targetDiameter,
                           double pixelScale,
                           bool preferPixelScale)
{
    namespace fs = std::filesystem;
    const CameraCalib cam = getDefaultCameraCalib();
    constexpr double kDefaultTargetDiameter = 5.9989; // 真实球直径，用于选择最佳外参并作为物理直径期望值
    const double effectiveTarget = (targetDiameter > 0.0) ? targetDiameter : kDefaultTargetDiameter;

    if (!fs::exists(realBallDir) || !fs::is_directory(realBallDir)) {
        std::cout << "[Eval] Invalid directory: " << realBallDir << std::endl;
        return;
    }

    std::vector<std::string> imagePaths;
    for (auto& entry : fs::directory_iterator(realBallDir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tif" || ext == ".tiff") {
            imagePaths.push_back(entry.path().string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        std::cout << "[Eval] No images found in: " << realBallDir << std::endl;
        return;
    }

    const std::string csvPath = (fs::path(realBallDir) / "real_ball_eval.csv").generic_string();
    std::ofstream csv(csvPath);
    if (!csv.is_open()) {
        std::cout << "[Eval] Failed to open CSV: " << csvPath << std::endl;
        return;
    }
    csv << "Image,Algo,Points,RMSE_px,MaxErr_px,Radius_px,CenterX_px,CenterY_px,"
           "RMSE_ud_px,Radius_ud_px,Diameter_ud_px,RMSE_world,Radius_world,Diameter_world,ExtrinsicIdx,"
           "Diameter_phys,RMSE_phys,PhysSource\n";

    struct Summary {
        int n{0};
        int n_world{0};
        int n_phys{0};
        double sum_radius_ud_px{0};
        double sum_rmse_ud_px{0};
        double sum_radius_world{0};
        double sum_rmse_world{0};
        double sum_diam_phys{0};
        double sum_rmse_phys{0};
    };
    std::unordered_map<std::string, Summary> summaries;
    bool any_phys = false;

    for (const auto& p : imagePaths) {
        const auto imgName = fs::path(p).filename().string();
        cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "[Eval] Failed to load: " << p << std::endl;
            continue;
        }
        {
            const std::string algoName = "ERF";

            SubpixelEdgeDetector det;
            det.setCannyThresholds(cfg.cannyLow, cfg.cannyHigh);
            det.setImage(img);
            det.refineEdgesSubpixel(cfg.subpixWindow, cfg.subpixSigma);
            const auto& pts = det.getSubpixelEdges();

            RealEvalResult r_px = evaluateCircleFit(pts, imgName, algoName);

            std::vector<cv::Point2d> undist_px;
            cv::undistortPoints(pts, undist_px, cam.K, cam.dist, cv::noArray(), cam.K);
            RealEvalResult r_ud = evaluateCircleFit(undist_px, imgName, algoName + "_undist");
            double diameter_ud_px = (r_ud.radius > 0) ? (2.0 * r_ud.radius) : 0.0;

            // 根据外参映射到标定板平面，选取与目标直径最接近的外参
            RealEvalResult r_world;
            int used_extrinsic = -1;
            if (!preferPixelScale) {
                double best_err = std::numeric_limits<double>::max();
                const auto& extrinsics = getExtrinsics();
                for (size_t i = 0; i < extrinsics.size(); ++i) {
                    cv::Matx33d R = rvecToMatx(extrinsics[i].rvec);
                    cv::Vec3d t = extrinsics[i].tvec;
                    std::vector<cv::Point2d> world_pts;
                    if (!undistortToWorldPlane(pts, cam.K, cam.dist, R, t, world_pts))
                        continue;
                    RealEvalResult r = evaluateCircleFit(world_pts, imgName, algoName + "_world");
                    if (r.radius <= 0)
                        continue;
                    double diam = 2.0 * r.radius;
                    double err = std::abs(diam - effectiveTarget);
                    if (err < best_err) {
                        best_err = err;
                        r_world = r;
                        used_extrinsic = static_cast<int>(i);
                    }
                }
            }

            double diameter_world = (r_world.radius > 0) ? 2.0 * r_world.radius : 0.0;
            double diameter_phys = 0.0;
            double rmse_phys = 0.0;
            const char* phys_source = "n/a";
            if (used_extrinsic >= 0 && r_world.radius > 0) {
                diameter_phys = diameter_world;
                rmse_phys = r_world.rmse;
                phys_source = "world";
            } else if (pixelScale > 0.0 && r_ud.radius > 0) {
                diameter_phys = diameter_ud_px * pixelScale;
                rmse_phys = r_ud.rmse * pixelScale;
                phys_source = "scale";
            }

            std::cout << "[Eval] image=" << imgName
                      << " algo=" << algoName
                      << " points=" << r_px.points
                      << " | px(R=" << r_px.radius << ",RMSE=" << r_px.rmse << ",Max=" << r_px.maxErr << ")"
                      << " | undist(R=" << r_ud.radius << ",D=" << diameter_ud_px << ",RMSE=" << r_ud.rmse << ")";
            if (used_extrinsic >= 0 && r_world.radius > 0) {
                std::cout << " | phys(src=world,idx=" << used_extrinsic
                          << ",D=" << diameter_world
                          << ",RMSE=" << r_world.rmse << ")";
            } else if (pixelScale > 0.0 && r_ud.radius > 0) {
                std::cout << " | phys(src=scale"
                          << ",D=" << diameter_phys
                          << ",RMSE=" << rmse_phys << ")";
            } else {
                std::cout << " | phys(src=n/a)";
            }
            std::cout << std::endl;

            auto &s = summaries[algoName];
            s.n++;
            s.sum_radius_ud_px += r_ud.radius;
            s.sum_rmse_ud_px += r_ud.rmse;
            if (r_world.radius > 0) {
                s.n_world++;
                s.sum_radius_world += r_world.radius;
                s.sum_rmse_world += r_world.rmse;
            }
            if (diameter_phys > 0) {
                any_phys = true;
                s.n_phys++;
                s.sum_diam_phys += diameter_phys;
                s.sum_rmse_phys += rmse_phys;
            }

            csv << r_px.imageName << "," << r_px.algoName << "," << r_px.points << ","
                << r_px.rmse << "," << r_px.maxErr << "," << r_px.radius << ","
                << r_px.center.x << "," << r_px.center.y << ","
                << r_ud.rmse << "," << r_ud.radius << ","
                << diameter_ud_px << ","
                << r_world.rmse << "," << r_world.radius << "," << diameter_world << ","
                << used_extrinsic << ","
                << diameter_phys << "," << rmse_phys << "," << phys_source << "\n";
        }
    }

    if (!summaries.empty()) {
        std::cout << "[Eval Summary] target_diam=" << effectiveTarget << std::endl;
        for (const auto& kv : summaries) {
            const auto& name = kv.first;
            const auto& s = kv.second;
            double denom = static_cast<double>(std::max(1, s.n));
            double avg_rad = s.sum_radius_ud_px / denom;
            double avg_rmse = s.sum_rmse_ud_px / denom;
            double avg_diam = avg_rad * 2.0;
            double avg_rad_world = (s.n_world > 0) ? (s.sum_radius_world / s.n_world) : 0.0;
            double avg_diam_world = avg_rad_world * 2.0;
            double avg_rmse_world = (s.n_world > 0) ? (s.sum_rmse_world / s.n_world) : 0.0;
            double avg_diam_phys = (s.n_phys > 0) ? (s.sum_diam_phys / s.n_phys) : 0.0;
            double avg_rmse_phys = (s.n_phys > 0) ? (s.sum_rmse_phys / s.n_phys) : 0.0;
            std::cout << "  " << name << std::endl;
            std::cout << "    count=" << s.n
                      << " world_count=" << s.n_world
                      << " phys_count=" << s.n_phys << std::endl;
            std::cout << "    undist_px: R=" << avg_rad
                      << " D=" << avg_diam
                      << " RMSE=" << avg_rmse << std::endl;
            if (s.n_world > 0) {
                std::cout << "    world: R=" << avg_rad_world
                          << " D=" << avg_diam_world
                          << " RMSE=" << avg_rmse_world << std::endl;
            } else {
                std::cout << "    world: n/a" << std::endl;
            }
            if (s.n_phys > 0) {
                std::cout << "    phys: D=" << avg_diam_phys
                          << " RMSE=" << avg_rmse_phys << std::endl;
            } else {
                std::cout << "    phys: n/a" << std::endl;
            }
        }
    }

    if (!any_phys) {
        std::cout << "[Eval] No physical results available. Provide valid extrinsics or --pixel-scale." << std::endl;
    }

    std::cout << "[Eval] Saved: " << csvPath << std::endl;
}

} // namespace stitch
