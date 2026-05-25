/*
==============================================================================
文件：TelecentricCalibrator.cpp
------------------------------------------------------------------------------
【本文件最值得读】—— 这里包含标定的核心数学逻辑。

整体流程（从上到下对应代码结构）：
  1) ListImages()               —— 读取标定图列表（按文件名排序）
  2) CreateBlobDetector()       —— SimpleBlobDetector 参数：针对“白底黑点”的圆点
  3) findCirclesGrid()          —— OpenCV 用 blob 检测网格拓扑（先得到粗中心）
  4) RefineCentersByEllipse()   —— 在每个圆点附近做 ROI + 二值化 + 轮廓 + 椭圆拟合
                                   把中心精修到亚像素（这一步对精度非常关键）
  5) AlignByBestAffineD4()      —— 因为 circle grid 的点序可能存在旋转/翻转（D4 对称群）
                                   这里用“最小仿射 RMS”自动选一个最合理的点序
  6) 初值：FitAffineLS() + InitExtrinsicFromAffine()
                                   用仿射拟合得到外参初值（rvec/txy）
  7) Ceres 优化                 —— 联合优化 intr/dist + 每张图外参，使重投影误差最小
  8) 保存 init 文件             —— 写 intr/dist + 每张图 rvec/txy

阅读建议：
  - 先理解“远心模型”里 intr 的含义：fx/fy/skew/cx/cy 的单位是 px/mm
  - 再看“dist”如何作用于像素平面（pixel-domain 畸变）

==============================================================================
*/

#include "TelecentricCalibrator.h"
#include "CircleEdgeResponse.h"
#include "core/SubpixelEdgeDetector.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <ctime>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace telecentric {

static fs::path Utf8Path(const std::string& text) {
  return fs::u8path(text);
}

static bool LoadGrayImageUtf8(const std::string& path_text, cv::Mat& out) {
  out.release();

  std::ifstream ifs(Utf8Path(path_text), std::ios::binary);
  if (!ifs.is_open()) return false;

  ifs.seekg(0, std::ios::end);
  const std::streamoff size = ifs.tellg();
  if (size <= 0) return false;
  ifs.seekg(0, std::ios::beg);

  std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
  if (!ifs.read(reinterpret_cast<char*>(bytes.data()), size)) return false;

  out = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
  return !out.empty();
}

static std::string TrimCopy(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

static bool ParseKVLine(const std::string& line, std::string& k, std::string& v) {
  const auto p = line.find('=');
  if (p == std::string::npos) return false;
  k = TrimCopy(line.substr(0, p));
  v = TrimCopy(line.substr(p + 1));
  return !k.empty();
}

static std::string nowIsoLocal() {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const std::time_t tt = system_clock::to_time_t(now);
  std::tm tmv{};
#ifdef _WIN32
  localtime_s(&tmv, &tt);
#else
  localtime_r(&tt, &tmv);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

static std::string HashFileFNV1a64(const std::string& path) {
  std::ifstream f(Utf8Path(path), std::ios::binary);
  if (!f.is_open()) return std::string();
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t h = kOffset;
  char buf[4096];
  while (f.good()) {
    f.read(buf, sizeof(buf));
    const std::streamsize n = f.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
      h ^= static_cast<unsigned char>(buf[i]);
      h *= kPrime;
    }
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

static double MedianCopy(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double med = values[mid];
  if ((values.size() % 2) == 0) {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    med = 0.5 * (med + values[mid - 1]);
  }
  return med;
}

static double RelativeDiffPercent(double a, double b) {
  const double denom = 0.5 * (std::abs(a) + std::abs(b));
  if (denom < 1e-12) return 0.0;
  return std::abs(a - b) / denom * 100.0;
}

static std::string JoinStrings(const std::vector<std::string>& items, const std::string& sep) {
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << sep;
    oss << items[i];
  }
  return oss.str();
}

std::string DefaultCalibReportPath(const std::string& init_path) {
  if (init_path.empty()) return std::string("calib_quality_report.txt");
  const std::size_t slash = init_path.find_last_of("/\\");
  const std::string dir = (slash == std::string::npos) ? std::string() : init_path.substr(0, slash + 1);
  const std::string file = (slash == std::string::npos) ? init_path : init_path.substr(slash + 1);
  const std::size_t dot = file.find_last_of('.');
  std::string stem = (dot == std::string::npos) ? file : file.substr(0, dot);
  if (stem.empty()) stem = "telecentric_init";
  return dir + stem + "_calib_quality_report.txt";
}

bool SaveCalibQualityReport(const std::string& path,
                            const CalibQualityReport& report,
                            std::string* err) {
  std::ofstream out(Utf8Path(path));
  if (!out.is_open()) {
    if (err) *err = "cannot open report path: " + path;
    return false;
  }
  out << "# telecentric_calib_quality_report v2\n";
  out << "created_at=" << report.created_at << "\n";
  out << "init_path=" << report.init_path << "\n";
  out << "init_hash_hex=" << report.init_hash_hex << "\n";
  out << "total_images=" << report.total_images << "\n";
  out << "valid_images=" << report.valid_images << "\n";
  out << "reproj_rms_px=" << report.reproj_rms_px << "\n";
  out << "mean_affine_rms_px=" << report.mean_affine_rms_px << "\n";
  out << "image_width=" << report.image_width << "\n";
  out << "image_height=" << report.image_height << "\n";
  out << "valid_ratio_pct=" << report.valid_ratio_pct << "\n";
  out << "fx_px_per_mm=" << report.fx_px_per_mm << "\n";
  out << "fy_px_per_mm=" << report.fy_px_per_mm << "\n";
  out << "fx_fy_diff_pct=" << report.fx_fy_diff_pct << "\n";
  out << "cx_px=" << report.cx_px << "\n";
  out << "cy_px=" << report.cy_px << "\n";
  out << "cx_offset_px=" << report.cx_offset_px << "\n";
  out << "cy_offset_px=" << report.cy_offset_px << "\n";
  out << "intrinsic_model=" << report.intrinsic_model << "\n";
  out << "principal_point_policy=" << report.principal_point_policy << "\n";
  out << "warning_summary=" << report.warning_summary << "\n";
  return true;
}

bool LoadCalibQualityReport(const std::string& path,
                            CalibQualityReport& report,
                            std::string* err) {
  report = CalibQualityReport{};
  std::ifstream in(Utf8Path(path));
  if (!in.is_open()) {
    if (err) *err = "cannot open report path: " + path;
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::string k, v;
    if (!ParseKVLine(line, k, v)) continue;
    try {
      if (k == "created_at") report.created_at = v;
      else if (k == "init_path") report.init_path = v;
      else if (k == "init_hash_hex") report.init_hash_hex = v;
      else if (k == "total_images") report.total_images = std::stoi(v);
      else if (k == "valid_images") report.valid_images = std::stoi(v);
      else if (k == "reproj_rms_px") report.reproj_rms_px = std::stod(v);
      else if (k == "mean_affine_rms_px") report.mean_affine_rms_px = std::stod(v);
      else if (k == "image_width") report.image_width = std::stoi(v);
      else if (k == "image_height") report.image_height = std::stoi(v);
      else if (k == "valid_ratio_pct") report.valid_ratio_pct = std::stod(v);
      else if (k == "fx_px_per_mm") report.fx_px_per_mm = std::stod(v);
      else if (k == "fy_px_per_mm") report.fy_px_per_mm = std::stod(v);
      else if (k == "fx_fy_diff_pct") report.fx_fy_diff_pct = std::stod(v);
      else if (k == "cx_px") report.cx_px = std::stod(v);
      else if (k == "cy_px") report.cy_px = std::stod(v);
      else if (k == "cx_offset_px") report.cx_offset_px = std::stod(v);
      else if (k == "cy_offset_px") report.cy_offset_px = std::stod(v);
      else if (k == "intrinsic_model") report.intrinsic_model = v;
      else if (k == "principal_point_policy") report.principal_point_policy = v;
      else if (k == "warning_summary") report.warning_summary = v;
    } catch (...) {
      // keep parsing other lines
    }
  }
  return true;
}

// ------------------------- Utilities 文件自动排序读取-------------------------
static std::vector<std::string> ListImages(const std::string& folder) {
  std::vector<std::string> files;
  const std::vector<std::string> exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"};
  std::error_code ec;
  for (fs::directory_iterator it(Utf8Path(folder), ec), end; !ec && it != end; it.increment(ec)) {
    const auto& p = *it;
    if (!p.is_regular_file()) continue;
    auto ext = p.path().extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
      files.push_back(p.path().u8string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

static std::vector<cv::Point2f> MakeWorldPts2D(int rows, int cols, double d) {
  std::vector<cv::Point2f> P;
  P.reserve(rows * cols);
  for (int r=0; r<rows; ++r)
    for (int c=0; c<cols; ++c)
      P.emplace_back((float)(c*d), (float)(r*d)); // mm
  return P;
}

// ------------------------- 斑点检测-------------------------
static cv::Ptr<cv::FeatureDetector> CreateBlobDetector() {
  cv::SimpleBlobDetector::Params p;
  p.minThreshold = 5;
  p.maxThreshold = 240;
  p.thresholdStep = 5;

  p.filterByColor = true;
  p.blobColor = 0;

  p.filterByArea = true;
  p.minArea = 9000.0f;
  p.maxArea = 20000.0f;

  p.filterByCircularity = true;
  p.minCircularity = 0.82f;

  p.filterByInertia = true;
  p.minInertiaRatio = 0.5f;

  p.filterByConvexity = true;
  p.minConvexity = 0.8f;

  p.minDistBetweenBlobs = 140.0f;
  return cv::SimpleBlobDetector::create(p);
}

static bool SelectLargestContour(const std::vector<std::vector<cv::Point>>& contours,
                                 std::vector<cv::Point>& bestContour) {
  int bestIndex = -1;
  double bestArea = 0.0;
  for (int k = 0; k < static_cast<int>(contours.size()); ++k) {
    const double area = cv::contourArea(contours[k]);
    if (area > bestArea) {
      bestArea = area;
      bestIndex = k;
    }
  }
  if (bestIndex < 0) return false;
  bestContour = contours[bestIndex];
  return true;
}

static bool ComputeInitialEllipseFromPatch(const cv::Mat& patch,
                                           cv::RotatedRect& ellipse,
                                           std::vector<cv::Point>& contour) {
  cv::Mat bin;
  cv::threshold(patch, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (!SelectLargestContour(contours, contour)) return false;
  if (contour.size() < 30) return false;

  ellipse = cv::fitEllipse(contour);
  return true;
}

static double EllipseNormalizedRadius(const cv::RotatedRect& ellipse, const cv::Point2f& point) {
  const double a = std::max(1e-6, 0.5 * static_cast<double>(ellipse.size.width));
  const double b = std::max(1e-6, 0.5 * static_cast<double>(ellipse.size.height));
  const double theta = ellipse.angle * CV_PI / 180.0;
  const double ct = std::cos(theta);
  const double st = std::sin(theta);

  const double dx = point.x - ellipse.center.x;
  const double dy = point.y - ellipse.center.y;
  const double xr = ct * dx + st * dy;
  const double yr = -st * dx + ct * dy;
  return std::sqrt((xr * xr) / (a * a) + (yr * yr) / (b * b));
}

static void DetectSubpixelEdgePoints(const cv::Mat& patch,
                                     std::vector<cv::Point2f>& edgePoints) {
  edgePoints.clear();

  SubpixelEdgeDetector detector;
  SubpixelEdgeDetector::Params params;
  params.canny_low = 30.0;
  params.canny_high = 90.0;
  params.window_size = 7;
  params.presmooth_sigma = 0.8;
  params.use_scharr = true;
  detector.setParams(params);
  detector.setImage(patch);
  detector.refineEdgesSubpixel();

  const auto& points = detector.getSubpixelPoints();
  edgePoints.reserve(points.size());
  for (const auto& point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.confidence) || !std::isfinite(point.gradient)) {
      continue;
    }
    if (point.confidence < 1.0 || point.gradient < 10.0) {
      continue;
    }
    edgePoints.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
  }
}

static bool RefineEllipseBySubpixelEdges(const cv::Mat& patch,
                                         const cv::RotatedRect& initialEllipse,
                                         const std::vector<cv::Point>& contour,
                                         cv::RotatedRect& refinedEllipse) {
  std::vector<cv::Point2f> edgePoints;
  DetectSubpixelEdgePoints(patch, edgePoints);
  if (edgePoints.size() < 24) return false;

  std::vector<cv::Point2f> candidates;
  candidates.reserve(edgePoints.size());
  const double minAxis = std::max(
      8.0, 0.5 * static_cast<double>(std::min(initialEllipse.size.width, initialEllipse.size.height)));
  const double maxContourDistance = std::clamp(0.08 * minAxis, 1.5, 4.0);
  const double ellipseBand = std::clamp(3.0 / std::max(1.0, minAxis), 0.04, 0.16);

  for (const auto& point : edgePoints) {
    const double contourDistance = std::abs(cv::pointPolygonTest(contour, point, true));
    if (contourDistance > maxContourDistance) continue;

    const double rho = EllipseNormalizedRadius(initialEllipse, point);
    if (std::abs(rho - 1.0) > ellipseBand) continue;

    candidates.push_back(point);
  }

  if (candidates.size() < 20) return false;

  cv::RotatedRect ellipse = cv::fitEllipse(candidates);
  std::vector<cv::Point2f> tightened;
  tightened.reserve(candidates.size());
  const double tightenedBand = std::max(0.03, ellipseBand * 0.6);
  for (const auto& point : candidates) {
    if (std::abs(EllipseNormalizedRadius(ellipse, point) - 1.0) <= tightenedBand) {
      tightened.push_back(point);
    }
  }
  if (tightened.size() >= 16) {
    ellipse = cv::fitEllipse(tightened);
  }

  const double centerShift = cv::norm(ellipse.center - initialEllipse.center);
  if (!std::isfinite(centerShift) || centerShift > 6.0) return false;

  const double widthRatio = ellipse.size.width / std::max(1.0f, initialEllipse.size.width);
  const double heightRatio = ellipse.size.height / std::max(1.0f, initialEllipse.size.height);
  if (!std::isfinite(widthRatio) || !std::isfinite(heightRatio)) return false;
  if (widthRatio < 0.75 || widthRatio > 1.25 || heightRatio < 0.75 || heightRatio > 1.25) {
    return false;
  }

  refinedEllipse = ellipse;
  return true;
}

// ------------------------- Ellipse refinement for subpixel dot centers 优化检测斑点的圆心亚像素坐标精准度-------------------------
static bool RefineCentersByEllipse(const cv::Mat& gray,
                                  std::vector<cv::Point2f>& centers,
                                  int roi_radius_px) {
  if (gray.empty() || gray.type() != CV_8U) return false;

  std::vector<cv::Point2f> refined = centers;
  for (size_t i=0; i<centers.size(); ++i) {
    int cx = (int)std::lround(centers[i].x);
    int cy = (int)std::lround(centers[i].y);
    int r = roi_radius_px;

    int x0 = std::max(0, cx - r);
    int y0 = std::max(0, cy - r);
    int x1 = std::min(gray.cols-1, cx + r);
    int y1 = std::min(gray.rows-1, cy + r);
    if (x1 - x0 < 20 || y1 - y0 < 20) continue;

    cv::Rect roi(x0, y0, x1-x0+1, y1-y0+1);
    cv::Mat patch = gray(roi).clone();

    cv::RotatedRect ell;
    std::vector<cv::Point> contour;
    if (!ComputeInitialEllipseFromPatch(patch, ell, contour)) continue;

    cv::RotatedRect refinedEllipse = ell;
    if (RefineEllipseBySubpixelEdges(patch, ell, contour, refinedEllipse)) {
      ell = refinedEllipse;
    }
    cv::Point2f c = ell.center;
    cv::Point2f roiCenter((float)(cx - x0), (float)(cy - y0));
    if (cv::norm(c - roiCenter) > (float)(0.35 * roi_radius_px)) continue;

    refined[i] = cv::Point2f(c.x + x0, c.y + y0);
  }

  centers.swap(refined);
  return true;
}

// ------------------------- Affine LS fit + RMS (for D4 disambiguation & init) -------------------------
static cv::Mat FitAffineLS(const std::vector<cv::Point2f>& worldXY,
                           const std::vector<cv::Point2f>& imgUV) {
  const int N = (int)worldXY.size();
  CV_Assert(N >= 3 && imgUV.size() == worldXY.size());
  cv::Mat A(2*N, 6, CV_64F), b(2*N, 1, CV_64F);

  for (int i=0;i<N;++i) {
    double X = worldXY[i].x, Y = worldXY[i].y;
    double u = imgUV[i].x,   v = imgUV[i].y;

    A.at<double>(2*i,0)=X; A.at<double>(2*i,1)=Y; A.at<double>(2*i,2)=1;
    A.at<double>(2*i,3)=0; A.at<double>(2*i,4)=0; A.at<double>(2*i,5)=0;
    b.at<double>(2*i,0)=u;

    A.at<double>(2*i+1,0)=0; A.at<double>(2*i+1,1)=0; A.at<double>(2*i+1,2)=0;
    A.at<double>(2*i+1,3)=X; A.at<double>(2*i+1,4)=Y; A.at<double>(2*i+1,5)=1;
    b.at<double>(2*i+1,0)=v;
  }

  cv::Mat x;
  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat H = (cv::Mat_<double>(2,3) <<
    x.at<double>(0), x.at<double>(1), x.at<double>(2),
    x.at<double>(3), x.at<double>(4), x.at<double>(5));
  return H;
}

static double AffineRMS(const cv::Mat& H,
                        const std::vector<cv::Point2f>& worldXY,
                        const std::vector<cv::Point2f>& imgUV) {
  double sse=0.0;
  const int N=(int)worldXY.size();
  for(int i=0;i<N;++i){
    double X=worldXY[i].x, Y=worldXY[i].y;
    double up = H.at<double>(0,0)*X + H.at<double>(0,1)*Y + H.at<double>(0,2);
    double vp = H.at<double>(1,0)*X + H.at<double>(1,1)*Y + H.at<double>(1,2);
    double du = up - imgUV[i].x;
    double dv = vp - imgUV[i].y;
    sse += du*du + dv*dv;
  }
  return std::sqrt(sse / N);
}

// ------------------------- D4 (8 symmetries) for nxn square grid -------------------------
static std::vector<cv::Point2f> ApplyD4(const std::vector<cv::Point2f>& in,
                                        int n, int rotK, bool flipX) {
  std::vector<cv::Point2f> out(n*n);
  for(int ri=0; ri<n; ++ri){
    for(int ci=0; ci<n; ++ci){
      int r=ri, c=ci;
      if (flipX) c = (n-1 - c);

      int ro=r, co=c;
      if(rotK==0){ ro=r;        co=c;       }
      if(rotK==1){ ro=(n-1-c);  co=r;       }
      if(rotK==2){ ro=(n-1-r);  co=(n-1-c); }
      if(rotK==3){ ro=c;        co=(n-1-r); }

      out[ro*n + co] = in[ri*n + ci];
    }
  }
  return out;
}

static std::vector<std::vector<cv::Point2f>>
GenerateD4Permutations(const std::vector<cv::Point2f>& centers, int n) {
  std::vector<std::vector<cv::Point2f>> perms;
  perms.reserve(8);
  for(int k=0;k<4;++k) perms.push_back(ApplyD4(centers, n, k, false));
  for(int k=0;k<4;++k) perms.push_back(ApplyD4(centers, n, k, true));
  return perms;
}

static std::vector<cv::Point2f>
AlignByBestAffineD4(const std::vector<cv::Point2f>& centers,
                    const std::vector<cv::Point2f>& worldXY,
                    int n) {
  auto perms = GenerateD4Permutations(centers, n);
  double best = 1e100;
  std::vector<cv::Point2f> bestC = centers;

  for (auto& cand : perms) {
    cv::Mat H = FitAffineLS(worldXY, cand);
    double rms = AffineRMS(H, worldXY, cand);
    if (rms < best) { best = rms; bestC = cand; }
  }
  return bestC;
}

// ------------------------- Init extrinsic from affine -------------------------
static void OrthonormalizeR(double R[9]) {
  cv::Vec3d r1(R[0],R[1],R[2]);
  cv::Vec3d r2(R[3],R[4],R[5]);

  if (cv::norm(r1) < 1e-9 || cv::norm(r2) < 1e-9) {
    R[0]=1;R[1]=0;R[2]=0;
    R[3]=0;R[4]=1;R[5]=0;
    R[6]=0;R[7]=0;R[8]=1;
    return;
  }

  r1 = r1 / cv::norm(r1);
  r2 = r2 - r1.dot(r2)*r1;
  if (cv::norm(r2) < 1e-9) {
    r2 = cv::Vec3d(-r1[1], r1[0], 0);
  }
  r2 = r2 / cv::norm(r2);

  cv::Vec3d r3 = r1.cross(r2);
  if (r3[2] < 0) { r2 = -r2; r3 = r1.cross(r2); }

  R[0]=r1[0]; R[1]=r1[1]; R[2]=r1[2];
  R[3]=r2[0]; R[4]=r2[1]; R[5]=r2[2];
  R[6]=r3[0]; R[7]=r3[1]; R[8]=r3[2];
}

static void InitExtrinsicFromAffine(const cv::Mat& H,
                                   double fx, double fy,
                                   double skew,
                                   double cx, double cy,
                                   double rvec[3], double txy[2]) {
  (void)skew;
  double a = H.at<double>(0,0), b = H.at<double>(0,1), e = H.at<double>(0,2);
  double c = H.at<double>(1,0), d = H.at<double>(1,1), f = H.at<double>(1,2);

  txy[0] = (e - cx)/fx;
  txy[1] = (f - cy)/fy;

  double r11 = a/fx, r12 = b/fx;
  double r21 = c/fy, r22 = d/fy;

  double r13 = std::sqrt(std::max(0.0, 1.0 - r11*r11 - r12*r12));
  double r23 = std::sqrt(std::max(0.0, 1.0 - r21*r21 - r22*r22));

  double R[9] = {
    r11, r12, r13,
    r21, r22, r23,
    0,   0,   1
  };
  OrthonormalizeR(R);

  cv::Mat Rm(3,3,CV_64F,R);
  cv::Mat rv;
  cv::Rodrigues(Rm, rv);
  rvec[0]=rv.at<double>(0); rvec[1]=rv.at<double>(1); rvec[2]=rv.at<double>(2);
}

static void InitIntrinsicsFromAffines(const std::vector<cv::Mat>& Hs,
                                      int imgW, int imgH,
                                      bool enforce_square_pixels,
                                      double& fx, double& fy,
                                      double& skew,
                                      double& cx, double& cy) {
  cx = 0.5 * (imgW - 1);
  cy = 0.5 * (imgH - 1);
  skew = 0.0;

  std::vector<double> fx_vals;
  std::vector<double> fy_vals;
  fx_vals.reserve(Hs.size());
  fy_vals.reserve(Hs.size());
  double fx_max = 0.0;
  double fy_max = 0.0;
  for (auto& H : Hs) {
    double a=H.at<double>(0,0), b=H.at<double>(0,1);
    double c=H.at<double>(1,0), d=H.at<double>(1,1);
    const double sx = std::sqrt(a*a + b*b);
    const double sy = std::sqrt(c*c + d*d);
    if (std::isfinite(sx) && sx > 1e-9) {
      fx_vals.push_back(sx);
      fx_max = std::max(fx_max, sx);
    }
    if (std::isfinite(sy) && sy > 1e-9) {
      fy_vals.push_back(sy);
      fy_max = std::max(fy_max, sy);
    }
  }

  fx = MedianCopy(fx_vals);
  fy = MedianCopy(fy_vals);
  if (fx < 1e-6) fx = 99.34;
  if (fy < 1e-6) fy = 99.34;

  if (enforce_square_pixels) {
    // 等尺度模型需要保证各图像的行向量归一化后不超过 1，
    // 因此共享尺度初值取一个“可行上界”更稳妥。
    const double shared = std::max(fx_max, fy_max);
    fx = (shared > 1e-6) ? shared : 99.34;
    fy = fx;
  }
}

// ------------------------- Ceres cost: Telecentric orthographic + distortion -------------------------
struct TelecentricReprojError {
  TelecentricReprojError(double X, double Y, double u, double v)
      : X_(X), Y_(Y), u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T* const intr,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  T* residuals) const {
    const T fx   = intr[0];
    const T fy   = intr[1];
    const T skew = intr[2];
    const T cx   = intr[3];
    const T cy   = intr[4];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T Pw[3] = {T(X_), T(Y_), T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    T x = Pc[0];
    T y = Pc[1];

    // project to pixel without distortion
    T u0 = fx*x + skew*y + cx;
    T v0 = fy*y + cy;

    // pixel-domain distortion around principal point
    T dx = u0 - cx;
    T dy = v0 - cy;
    T r2 = dx*dx + dy*dy;
    T radial = T(1) + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
    T x_tan = T(2)*p1*dx*dy + p2*(r2 + T(2)*dx*dx);
    T y_tan = p1*(r2 + T(2)*dy*dy) + T(2)*p2*dx*dy;

    T u_pred = cx + dx*radial + x_tan;
    T v_pred = cy + dy*radial + y_tan;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojError, 2, 5, 5, 3, 2>(
        new TelecentricReprojError(X, Y, u, v)));
  }

  double X_, Y_, u_, v_;
};

struct TelecentricReprojErrorIso {
  TelecentricReprojErrorIso(double X, double Y, double u, double v)
      : X_(X), Y_(Y), u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T* const intr_iso,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  T* residuals) const {
    const T f  = intr_iso[0];
    const T cx = intr_iso[1];
    const T cy = intr_iso[2];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T Pw[3] = {T(X_), T(Y_), T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    const T x = Pc[0];
    const T y = Pc[1];

    // 各向同性远心内参：u=f*x+cx, v=f*y+cy
    const T u0 = f*x + cx;
    const T v0 = f*y + cy;

    const T dx = u0 - cx;
    const T dy = v0 - cy;
    const T r2 = dx*dx + dy*dy;
    const T radial = T(1) + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
    const T x_tan = T(2)*p1*dx*dy + p2*(r2 + T(2)*dx*dx);
    const T y_tan = p1*(r2 + T(2)*dy*dy) + T(2)*p2*dx*dy;

    const T u_pred = cx + dx*radial + x_tan;
    const T v_pred = cy + dy*radial + y_tan;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojErrorIso, 2, 3, 5, 3, 2>(
        new TelecentricReprojErrorIso(X, Y, u, v)));
  }

  double X_, Y_, u_, v_;
};

struct TelecentricReprojErrorBoard {
  TelecentricReprojErrorBoard(double X, double Y, double u, double v)
      : X_(X), Y_(Y), u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T* const intr,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  const T* const board_delta,
                  T* residuals) const {
    const T fx   = intr[0];
    const T fy   = intr[1];
    const T skew = intr[2];
    const T cx   = intr[3];
    const T cy   = intr[4];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T Pw[3] = {T(X_) + board_delta[0], T(Y_) + board_delta[1], T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    const T x = Pc[0];
    const T y = Pc[1];
    const T u0 = fx*x + skew*y + cx;
    const T v0 = fy*y + cy;

    const T dx = u0 - cx;
    const T dy = v0 - cy;
    const T r2 = dx*dx + dy*dy;
    const T radial = T(1) + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
    const T x_tan = T(2)*p1*dx*dy + p2*(r2 + T(2)*dx*dx);
    const T y_tan = p1*(r2 + T(2)*dy*dy) + T(2)*p2*dx*dy;

    const T u_pred = cx + dx*radial + x_tan;
    const T v_pred = cy + dy*radial + y_tan;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojErrorBoard, 2, 5, 5, 3, 2, 2>(
        new TelecentricReprojErrorBoard(X, Y, u, v)));
  }

  double X_, Y_, u_, v_;
};

struct TelecentricReprojErrorIsoBoard {
  TelecentricReprojErrorIsoBoard(double X, double Y, double u, double v)
      : X_(X), Y_(Y), u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T* const intr_iso,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  const T* const board_delta,
                  T* residuals) const {
    const T f  = intr_iso[0];
    const T cx = intr_iso[1];
    const T cy = intr_iso[2];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T Pw[3] = {T(X_) + board_delta[0], T(Y_) + board_delta[1], T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    const T x = Pc[0];
    const T y = Pc[1];
    const T u0 = f*x + cx;
    const T v0 = f*y + cy;

    const T dx = u0 - cx;
    const T dy = v0 - cy;
    const T r2 = dx*dx + dy*dy;
    const T radial = T(1) + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
    const T x_tan = T(2)*p1*dx*dy + p2*(r2 + T(2)*dx*dx);
    const T y_tan = p1*(r2 + T(2)*dy*dy) + T(2)*p2*dx*dy;

    const T u_pred = cx + dx*radial + x_tan;
    const T v_pred = cy + dy*radial + y_tan;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojErrorIsoBoard, 2, 3, 5, 3, 2, 2>(
        new TelecentricReprojErrorIsoBoard(X, Y, u, v)));
  }

  double X_, Y_, u_, v_;
};

template <typename T>
static inline void ApplyBoardWarpCompensation(const T* const board_warp,
                                              const double basis0,
                                              const double basis1,
                                              const double basis2,
                                              T& dx,
                                              T& dy) {
  const T b0 = T(basis0);
  const T b1 = T(basis1);
  const T b2 = T(basis2);
  dx += board_warp[0] * b0 + board_warp[1] * b1 + board_warp[2] * b2;
  dy += board_warp[3] * b0 + board_warp[4] * b1 + board_warp[5] * b2;
}

static constexpr double kImageResidualBasisScalePx = 1000.0;

template <typename T>
static inline void ApplyImageResidualQuadCompensation(const T* const image_residual_quad,
                                                      const T& dx_px,
                                                      const T& dy_px,
                                                      T& du_px,
                                                      T& dv_px) {
  const T scale = T(kImageResidualBasisScalePx);
  const T xn = dx_px / scale;
  const T yn = dy_px / scale;
  const T b0 = xn * xn;
  const T b1 = xn * yn;
  const T b2 = yn * yn;

  du_px += image_residual_quad[0] * b0 + image_residual_quad[1] * b1 + image_residual_quad[2] * b2;
  dv_px += image_residual_quad[3] * b0 + image_residual_quad[4] * b1 + image_residual_quad[5] * b2;
}

struct TelecentricReprojErrorBoardFull {
  TelecentricReprojErrorBoardFull(double X, double Y, double u, double v,
                                  double basis0, double basis1, double basis2)
      : X_(X), Y_(Y), u_(u), v_(v), basis0_(basis0), basis1_(basis1), basis2_(basis2) {}

  template <typename T>
  bool operator()(const T* const intr,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  const T* const image_residual_quad,
                  const T* const board_warp,
                  const T* const board_delta,
                  T* residuals) const {
    const T fx   = intr[0];
    const T fy   = intr[1];
    const T skew = intr[2];
    const T cx   = intr[3];
    const T cy   = intr[4];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T dx_board = board_delta[0];
    T dy_board = board_delta[1];
    ApplyBoardWarpCompensation(board_warp, basis0_, basis1_, basis2_, dx_board, dy_board);

    T Pw[3] = {T(X_) + dx_board, T(Y_) + dy_board, T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    const T x = Pc[0];
    const T y = Pc[1];
    const T u0 = fx * x + skew * y + cx;
    const T v0 = fy * y + cy;

    const T dx = u0 - cx;
    const T dy = v0 - cy;
    const T r2 = dx * dx + dy * dy;
    const T radial = T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    T dist_x = dx * radial + T(2) * p1 * dx * dy + p2 * (r2 + T(2) * dx * dx);
    T dist_y = dy * radial + p1 * (r2 + T(2) * dy * dy) + T(2) * p2 * dx * dy;
    ApplyImageResidualQuadCompensation(image_residual_quad, dist_x, dist_y, dist_x, dist_y);

    const T u_pred = cx + dist_x;
    const T v_pred = cy + dist_y;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v,
                                     double basis0, double basis1, double basis2) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojErrorBoardFull, 2, 5, 5, 3, 2, 6, 6, 2>(
        new TelecentricReprojErrorBoardFull(X, Y, u, v, basis0, basis1, basis2)));
  }

  double X_, Y_, u_, v_;
  double basis0_, basis1_, basis2_;
};

struct TelecentricReprojErrorIsoBoardFull {
  TelecentricReprojErrorIsoBoardFull(double X, double Y, double u, double v,
                                     double basis0, double basis1, double basis2)
      : X_(X), Y_(Y), u_(u), v_(v), basis0_(basis0), basis1_(basis1), basis2_(basis2) {}

  template <typename T>
  bool operator()(const T* const intr_iso,
                  const T* const dist,
                  const T* const rvec,
                  const T* const txy,
                  const T* const image_residual_quad,
                  const T* const board_warp,
                  const T* const board_delta,
                  T* residuals) const {
    const T f  = intr_iso[0];
    const T cx = intr_iso[1];
    const T cy = intr_iso[2];

    const T k1 = dist[0];
    const T k2 = dist[1];
    const T k3 = dist[2];
    const T p1 = dist[3];
    const T p2 = dist[4];

    T dx_board = board_delta[0];
    T dy_board = board_delta[1];
    ApplyBoardWarpCompensation(board_warp, basis0_, basis1_, basis2_, dx_board, dy_board);

    T Pw[3] = {T(X_) + dx_board, T(Y_) + dy_board, T(0)};
    T Pc[3];
    ceres::AngleAxisRotatePoint(rvec, Pw, Pc);

    Pc[0] += txy[0];
    Pc[1] += txy[1];

    const T x = Pc[0];
    const T y = Pc[1];
    const T u0 = f * x + cx;
    const T v0 = f * y + cy;

    const T dx = u0 - cx;
    const T dy = v0 - cy;
    const T r2 = dx * dx + dy * dy;
    const T radial = T(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    T dist_x = dx * radial + T(2) * p1 * dx * dy + p2 * (r2 + T(2) * dx * dx);
    T dist_y = dy * radial + p1 * (r2 + T(2) * dy * dy) + T(2) * p2 * dx * dy;
    ApplyImageResidualQuadCompensation(image_residual_quad, dist_x, dist_y, dist_x, dist_y);

    const T u_pred = cx + dist_x;
    const T v_pred = cy + dist_y;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double X, double Y, double u, double v,
                                     double basis0, double basis1, double basis2) {
    return (new ceres::AutoDiffCostFunction<TelecentricReprojErrorIsoBoardFull, 2, 3, 5, 3, 2, 6, 6, 2>(
        new TelecentricReprojErrorIsoBoardFull(X, Y, u, v, basis0, basis1, basis2)));
  }

  double X_, Y_, u_, v_;
  double basis0_, basis1_, basis2_;
};

struct BoardPointPrior {
  explicit BoardPointPrior(double weight) : weight_(weight) {}

  template <typename T>
  bool operator()(const T* const board_delta, T* residuals) const {
    residuals[0] = T(weight_) * board_delta[0];
    residuals[1] = T(weight_) * board_delta[1];
    return true;
  }

  static ceres::CostFunction* Create(double weight) {
    return (new ceres::AutoDiffCostFunction<BoardPointPrior, 2, 2>(
        new BoardPointPrior(weight)));
  }

  double weight_;
};

struct BoardWarpPrior {
  explicit BoardWarpPrior(double weight) : weight_(weight) {}

  template <typename T>
  bool operator()(const T* const board_warp, T* residuals) const {
    for (int i = 0; i < 6; ++i) {
      residuals[i] = T(weight_) * board_warp[i];
    }
    return true;
  }

  static ceres::CostFunction* Create(double weight) {
    return (new ceres::AutoDiffCostFunction<BoardWarpPrior, 6, 6>(
        new BoardWarpPrior(weight)));
  }

  double weight_;
};

struct BoardPointSmoothness {
  explicit BoardPointSmoothness(double weight) : weight_(weight) {}

  template <typename T>
  bool operator()(const T* const prev_delta,
                  const T* const curr_delta,
                  const T* const next_delta,
                  T* residuals) const {
    residuals[0] = T(weight_) * (prev_delta[0] - T(2) * curr_delta[0] + next_delta[0]);
    residuals[1] = T(weight_) * (prev_delta[1] - T(2) * curr_delta[1] + next_delta[1]);
    return true;
  }

  static ceres::CostFunction* Create(double weight) {
    return (new ceres::AutoDiffCostFunction<BoardPointSmoothness, 2, 2, 2, 2>(
        new BoardPointSmoothness(weight)));
  }

  double weight_;
};

struct PrincipalPointPriorIntr {
  PrincipalPointPriorIntr(double cx0, double cy0, double weight)
      : cx0_(cx0), cy0_(cy0), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const intr, T* residuals) const {
    residuals[0] = T(weight_) * (intr[3] - T(cx0_));
    residuals[1] = T(weight_) * (intr[4] - T(cy0_));
    return true;
  }

  static ceres::CostFunction* Create(double cx0, double cy0, double weight) {
    return (new ceres::AutoDiffCostFunction<PrincipalPointPriorIntr, 2, 5>(
        new PrincipalPointPriorIntr(cx0, cy0, weight)));
  }

  double cx0_;
  double cy0_;
  double weight_;
};

struct PrincipalPointPriorIso {
  PrincipalPointPriorIso(double cx0, double cy0, double weight)
      : cx0_(cx0), cy0_(cy0), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const intr_iso, T* residuals) const {
    residuals[0] = T(weight_) * (intr_iso[1] - T(cx0_));
    residuals[1] = T(weight_) * (intr_iso[2] - T(cy0_));
    return true;
  }

  static ceres::CostFunction* Create(double cx0, double cy0, double weight) {
    return (new ceres::AutoDiffCostFunction<PrincipalPointPriorIso, 2, 3>(
        new PrincipalPointPriorIso(cx0, cy0, weight)));
  }

  double cx0_;
  double cy0_;
  double weight_;
};

static std::vector<std::array<double,3>> BuildBoardWarpBasis(const std::vector<cv::Point2f>& worldXY) {
  std::vector<std::array<double,3>> basis_values;
  basis_values.reserve(worldXY.size());
  if (worldXY.empty()) return basis_values;

  double min_x = worldXY.front().x;
  double max_x = worldXY.front().x;
  double min_y = worldXY.front().y;
  double max_y = worldXY.front().y;
  for (const auto& point : worldXY) {
    min_x = std::min(min_x, static_cast<double>(point.x));
    max_x = std::max(max_x, static_cast<double>(point.x));
    min_y = std::min(min_y, static_cast<double>(point.y));
    max_y = std::max(max_y, static_cast<double>(point.y));
  }

  const double center_x = 0.5 * (min_x + max_x);
  const double center_y = 0.5 * (min_y + max_y);
  const double half_span_x = std::max(1e-9, 0.5 * (max_x - min_x));
  const double half_span_y = std::max(1e-9, 0.5 * (max_y - min_y));

  std::vector<std::array<double,2>> normalized;
  normalized.reserve(worldXY.size());
  double mean_x2 = 0.0;
  double mean_xy = 0.0;
  double mean_y2 = 0.0;
  for (const auto& point : worldXY) {
    const double xn = (static_cast<double>(point.x) - center_x) / half_span_x;
    const double yn = (static_cast<double>(point.y) - center_y) / half_span_y;
    normalized.push_back({xn, yn});
    mean_x2 += xn * xn;
    mean_xy += xn * yn;
    mean_y2 += yn * yn;
  }
  const double inv_count = 1.0 / static_cast<double>(normalized.size());
  mean_x2 *= inv_count;
  mean_xy *= inv_count;
  mean_y2 *= inv_count;

  for (const auto& point : normalized) {
    basis_values.push_back({
        point[0] * point[0] - mean_x2,
        point[0] * point[1] - mean_xy,
        point[1] * point[1] - mean_y2,
    });
  }
  return basis_values;
}

static std::array<double,2> EvaluateBoardWarpOffsetMm(const std::array<double,6>& board_warp,
                                                      const std::array<double,3>& basis) {
  return {
      board_warp[0] * basis[0] + board_warp[1] * basis[1] + board_warp[2] * basis[2],
      board_warp[3] * basis[0] + board_warp[4] * basis[1] + board_warp[5] * basis[2],
  };
}

static std::array<double,2> EvaluateTotalBoardOffsetMm(
    const std::vector<std::array<double,3>>* board_warp_basis,
    const std::array<double,6>* board_warp_coeffs,
    const std::vector<std::array<double,2>>* board_point_deltas,
    const size_t index) {
  std::array<double,2> offset{0.0, 0.0};
  if (board_warp_basis != nullptr && board_warp_coeffs != nullptr && index < board_warp_basis->size()) {
    const auto warp = EvaluateBoardWarpOffsetMm(*board_warp_coeffs, (*board_warp_basis)[index]);
    offset[0] += warp[0];
    offset[1] += warp[1];
  }
  if (board_point_deltas != nullptr && index < board_point_deltas->size()) {
    offset[0] += (*board_point_deltas)[index][0];
    offset[1] += (*board_point_deltas)[index][1];
  }
  return offset;
}

static std::array<double,2> EvaluateImageResidualQuadPx(const std::array<double,6>& image_residual_quad,
                                                        const double dx_px,
                                                        const double dy_px) {
  const double xn = dx_px / kImageResidualBasisScalePx;
  const double yn = dy_px / kImageResidualBasisScalePx;
  const double b0 = xn * xn;
  const double b1 = xn * yn;
  const double b2 = yn * yn;
  return {
      image_residual_quad[0] * b0 + image_residual_quad[1] * b1 + image_residual_quad[2] * b2,
      image_residual_quad[3] * b0 + image_residual_quad[4] * b1 + image_residual_quad[5] * b2,
  };
}

static double ComputeRMSAll(const std::vector<std::vector<cv::Point2f>>& imgPts,
                            const std::vector<cv::Point2f>& worldXY,
                            const double intr[5], const double dist[5],
                            const std::vector<std::array<double,3>>& rvecs,
                            const std::vector<std::array<double,2>>& tvecs,
                            const std::array<double,6>* image_residual_quad = nullptr,
                            const std::vector<std::array<double,3>>* board_warp_basis = nullptr,
                            const std::array<double,6>* board_warp_coeffs = nullptr,
                            const std::vector<std::array<double,2>>* board_point_deltas = nullptr) {
  double sse=0.0;
  size_t count=0;

  for (size_t k=0; k<imgPts.size(); ++k) {
    for (size_t i=0; i<worldXY.size(); ++i) {
      double X = worldXY[i].x;
      double Y = worldXY[i].y;
      const auto board_offset =
          EvaluateTotalBoardOffsetMm(board_warp_basis, board_warp_coeffs, board_point_deltas, i);
      X += board_offset[0];
      Y += board_offset[1];
      double u = imgPts[k][i].x;
      double v = imgPts[k][i].y;

      double Pw[3] = {X,Y,0};
      double Pc[3];
      ceres::AngleAxisRotatePoint(rvecs[k].data(), Pw, Pc);
      Pc[0] += tvecs[k][0];
      Pc[1] += tvecs[k][1];

      double x = Pc[0], y = Pc[1];
      double u0 = intr[0]*x + intr[2]*y + intr[3];
      double v0 = intr[1]*y + intr[4];

      double dx = u0 - intr[3];
      double dy = v0 - intr[4];
      double r2 = dx*dx + dy*dy;
      double radial = 1.0 + dist[0]*r2 + dist[1]*r2*r2 + dist[2]*r2*r2*r2;
      double dist_x = dx*radial + 2.0*dist[3]*dx*dy + dist[4]*(r2 + 2.0*dx*dx);
      double dist_y = dy*radial + dist[3]*(r2 + 2.0*dy*dy) + 2.0*dist[4]*dx*dy;
      if (image_residual_quad != nullptr) {
        const auto image_residual = EvaluateImageResidualQuadPx(*image_residual_quad, dist_x, dist_y);
        dist_x += image_residual[0];
        dist_y += image_residual[1];
      }

      double up = intr[3] + dist_x;
      double vp = intr[4] + dist_y;

      double du = up - u;
      double dv = vp - v;
      sse += du*du + dv*dv;
      count += 1;
    }
  }
  return std::sqrt(sse / std::max<size_t>(1, count));
}

static double ComputeImageResidualFieldRmsPx(const std::vector<std::vector<cv::Point2f>>& imgPts,
                                             const double cx,
                                             const double cy,
                                             const std::array<double,6>& image_residual_quad) {
  double sse = 0.0;
  size_t count = 0;
  for (const auto& points : imgPts) {
    for (const auto& point : points) {
      const auto residual = EvaluateImageResidualQuadPx(image_residual_quad, point.x - cx, point.y - cy);
      sse += residual[0] * residual[0] + residual[1] * residual[1];
      count += 1;
    }
  }
  if (count == 0) return 0.0;
  return std::sqrt(sse / static_cast<double>(count));
}

static double ComputeImageResidualFieldMaxPx(const std::vector<std::vector<cv::Point2f>>& imgPts,
                                             const double cx,
                                             const double cy,
                                             const std::array<double,6>& image_residual_quad) {
  double max_norm = 0.0;
  for (const auto& points : imgPts) {
    for (const auto& point : points) {
      const auto residual = EvaluateImageResidualQuadPx(image_residual_quad, point.x - cx, point.y - cy);
      max_norm = std::max(max_norm, std::sqrt(residual[0] * residual[0] + residual[1] * residual[1]));
    }
  }
  return max_norm;
}

static double ComputeBoardOffsetRmsMm(const std::vector<std::array<double,2>>& offsets) {
  if (offsets.empty()) return 0.0;
  double sse = 0.0;
  for (const auto& delta : offsets) {
    sse += delta[0] * delta[0] + delta[1] * delta[1];
  }
  return std::sqrt(sse / static_cast<double>(offsets.size()));
}

static double ComputeBoardOffsetMaxMm(const std::vector<std::array<double,2>>& offsets) {
  double max_norm = 0.0;
  for (const auto& delta : offsets) {
    max_norm = std::max(max_norm, std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]));
  }
  return max_norm;
}

static double ComputeBoardWarpOffsetRmsMm(const std::vector<std::array<double,3>>& board_warp_basis,
                                          const std::array<double,6>& board_warp_coeffs) {
  if (board_warp_basis.empty()) return 0.0;
  double sse = 0.0;
  for (const auto& basis : board_warp_basis) {
    const auto offset = EvaluateBoardWarpOffsetMm(board_warp_coeffs, basis);
    sse += offset[0] * offset[0] + offset[1] * offset[1];
  }
  return std::sqrt(sse / static_cast<double>(board_warp_basis.size()));
}

static double ComputeBoardWarpOffsetMaxMm(const std::vector<std::array<double,3>>& board_warp_basis,
                                          const std::array<double,6>& board_warp_coeffs) {
  double max_norm = 0.0;
  for (const auto& basis : board_warp_basis) {
    const auto offset = EvaluateBoardWarpOffsetMm(board_warp_coeffs, basis);
    max_norm = std::max(max_norm, std::sqrt(offset[0] * offset[0] + offset[1] * offset[1]));
  }
  return max_norm;
}

static double ComputeTotalBoardOffsetRmsMm(const std::vector<std::array<double,3>>& board_warp_basis,
                                           const std::array<double,6>& board_warp_coeffs,
                                           const std::vector<std::array<double,2>>& board_point_deltas) {
  if (board_warp_basis.empty() && board_point_deltas.empty()) return 0.0;
  const size_t count = std::max(board_warp_basis.size(), board_point_deltas.size());
  if (count == 0) return 0.0;

  double sse = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const auto offset = EvaluateTotalBoardOffsetMm(&board_warp_basis, &board_warp_coeffs, &board_point_deltas, i);
    sse += offset[0] * offset[0] + offset[1] * offset[1];
  }
  return std::sqrt(sse / static_cast<double>(count));
}

static double ComputeTotalBoardOffsetMaxMm(const std::vector<std::array<double,3>>& board_warp_basis,
                                           const std::array<double,6>& board_warp_coeffs,
                                           const std::vector<std::array<double,2>>& board_point_deltas) {
  const size_t count = std::max(board_warp_basis.size(), board_point_deltas.size());
  double max_norm = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const auto offset = EvaluateTotalBoardOffsetMm(&board_warp_basis, &board_warp_coeffs, &board_point_deltas, i);
    max_norm = std::max(max_norm, std::sqrt(offset[0] * offset[0] + offset[1] * offset[1]));
  }
  return max_norm;
}

static bool SaveInit(const std::string& path,
                     const double intr[5], const double dist[5],
                     const std::array<double,6>& image_residual_quad,
                     const std::vector<std::string>& files,
                     const std::vector<std::array<double,3>>& rvecs,
                     const std::vector<std::array<double,2>>& tvecs) {
  std::ofstream out(Utf8Path(path));
  if (!out) return false;
  out << "# telecentric_calib_init v2\n";
  out << std::setprecision(17);
  out << "intr " << intr[0] << " " << intr[1] << " " << intr[2] << " " << intr[3] << " " << intr[4] << "\n";
  out << "dist " << dist[0] << " " << dist[1] << " " << dist[2] << " " << dist[3] << " " << dist[4] << "\n";
  out << "image_residual_quad "
      << image_residual_quad[0] << " " << image_residual_quad[1] << " " << image_residual_quad[2] << " "
      << image_residual_quad[3] << " " << image_residual_quad[4] << " " << image_residual_quad[5] << "\n";
  for (size_t i=0; i<files.size() && i<rvecs.size() && i<tvecs.size(); ++i) {
    out << "image " << files[i] << "\n";
    out << "rvec " << rvecs[i][0] << " " << rvecs[i][1] << " " << rvecs[i][2] << "\n";
    out << "tvec " << tvecs[i][0] << " " << tvecs[i][1] << "\n";
  }
  return true;
}

bool RunTelecentricCalibration(const CalibOptions& opt,
                               CalibParams& out_params,
                               std::string* err,
                               CalibVisCallback vis_cb,
                               CalibQualityReport* quality_out,
                               CalibLogCallback log_cb) {
  const auto emit_log = [&](const std::string& line) {
    if (log_cb) log_cb(line);
  };
  if (opt.image_folder.empty()) {
    if (err) *err = "image_folder is empty";
    return false;
  }
  std::error_code exists_ec;
  if (!fs::exists(Utf8Path(opt.image_folder), exists_ec) || exists_ec) {
    if (err) *err = "image_folder not found: " + opt.image_folder;
    return false;
  }

  auto files = ListImages(opt.image_folder);
  if (files.empty()) {
    if (err) *err = "No images found in: " + opt.image_folder;
    return false;
  }

  const int totalFiles = (int)files.size();
  const cv::Size patternSize(opt.cols, opt.rows);
  auto world2D = MakeWorldPts2D(opt.rows, opt.cols, opt.pitch_mm);
  auto blob = CreateBlobDetector();

  std::vector<std::vector<cv::Point2f>> allImgPts;
  std::vector<cv::Mat> allAffines;
  std::vector<std::string> usedFiles;
  std::vector<double> acceptedAffineRms;

  int imgW=0, imgH=0;

  int fileIdx = 0;
  for (auto& f : files) {
    ++fileIdx;

    cv::Mat gray;
    if (!LoadGrayImageUtf8(f, gray)) {
      emit_log("[skip] " + f + " cannot read image");
      if (vis_cb) {
        cv::Mat vis(200, 900, CV_8UC3, cv::Scalar(20,20,20));
        cv::putText(vis, "SKIP: cannot read image", {20, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
        vis_cb("detect", fileIdx, totalFiles, f, false, vis);
      }
      continue;
    }
    imgW = gray.cols; imgH = gray.rows;

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, {3,3}, 0);

    std::vector<cv::Point2f> centers;
    bool found = cv::findCirclesGrid(
        blur, patternSize, centers,
        cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING,
        blob);

    bool accepted = false;
    std::ostringstream reason;

    if (!found || (int)centers.size() != opt.rows*opt.cols) {
      reason << "detect failed, got " << centers.size();
      std::cout << "[skip] " << f << " " << reason.str() << std::endl;
      emit_log("[skip] " + f + " " + reason.str());

      if (vis_cb) {
        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        cv::putText(vis, ("SKIP: " + reason.str()).c_str(), {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
        vis_cb("detect", fileIdx, totalFiles, f, false, vis);
      }
      continue;
    }

    RefineCentersByEllipse(gray, centers, opt.roi_radius_px);

    if (opt.rows == opt.cols) {
      centers = AlignByBestAffineD4(centers, world2D, opt.rows);
    }

    cv::Mat H = FitAffineLS(world2D, centers);
    double rms_aff = AffineRMS(H, world2D, centers);
    if (rms_aff > opt.affine_rms_thresh) {
      reason << "affine rms too large: " << rms_aff;
      std::cout << "[skip] " << f << " " << reason.str() << std::endl;
      emit_log("[skip] " + f + " " + reason.str());

      if (vis_cb) {
        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(vis, patternSize, centers, true);
        cv::putText(vis, ("SKIP: " + reason.str()).c_str(), {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
        vis_cb("detect", fileIdx, totalFiles, f, false, vis);
      }
      continue;
    }

    accepted = true;
    reason << "affine_rms=" << rms_aff;
    allImgPts.push_back(centers);
    allAffines.push_back(H);
    usedFiles.push_back(f);
    acceptedAffineRms.push_back(rms_aff);

    std::cout << "[ok] " << f << " " << reason.str() << std::endl;
    emit_log("[ok] " + f + " " + reason.str());

    if (vis_cb) {
      cv::Mat vis;
      cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
      cv::drawChessboardCorners(vis, patternSize, centers, true);
      cv::putText(vis, ("ACCEPT: " + reason.str()).c_str(), {20, 40},
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
      vis_cb("detect", fileIdx, totalFiles, f, true, vis);
    }
  }

  if ((int)allImgPts.size() < opt.min_valid_images) {
    if (err) *err = "Not enough valid images. Got " + std::to_string(allImgPts.size()) +
                    ", need >= " + std::to_string(opt.min_valid_images);
    return false;
  }

  double intr[5];
  double dist[5] = {0,0,0,0,0};
  const double imgCx = 0.5 * (imgW - 1.0);
  const double imgCy = 0.5 * (imgH - 1.0);

  InitIntrinsicsFromAffines(allAffines, imgW, imgH, opt.enforce_square_pixels,
                            intr[0], intr[1], intr[2], intr[3], intr[4]);
  intr[2] = 0.0;
  intr[3] = imgCx;
  intr[4] = imgCy;
  double intr_iso[3] = {0.5 * (intr[0] + intr[1]), imgCx, imgCy};
  if (opt.enforce_square_pixels) {
    intr[0] = intr_iso[0];
    intr[1] = intr_iso[0];
  }

  std::vector<std::array<double,3>> rvecs(allImgPts.size());
  std::vector<std::array<double,2>> tvecs(allImgPts.size());
  for (size_t k=0; k<allImgPts.size(); ++k) {
    double rv[3], txy[2];
    InitExtrinsicFromAffine(allAffines[k], intr[0], intr[1], intr[2], intr[3], intr[4], rv, txy);
    rvecs[k] = {rv[0], rv[1], rv[2]};
    tvecs[k] = {txy[0], txy[1]};
  }
  const std::vector<std::array<double,3>> boardWarpBasis = BuildBoardWarpBasis(world2D);
  std::array<double,6> imageResidualQuad = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double,6> boardWarpCoeffs = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<std::array<double,2>> boardPointDeltas(world2D.size(), {0.0, 0.0});

  enum class BoardCompensationMode {
    None,
    WarpOnly,
    WarpAndPoints,
  };

  auto solve_stage = [&](const std::string& name,
                         bool fix_dist,
                         bool fix_k3_p1_p2,
                         bool print_report,
                         bool enable_image_residual = false,
                         BoardCompensationMode board_mode = BoardCompensationMode::None) {
    ceres::Problem problem;
    const bool enable_image_residual_model =
        opt.enable_image_residual_compensation && enable_image_residual;
    const bool enable_board_warp =
        opt.enable_board_warp_compensation && board_mode != BoardCompensationMode::None;
    const bool enable_board_points =
        opt.enable_board_point_compensation && board_mode == BoardCompensationMode::WarpAndPoints;
    const bool any_board_compensation = enable_board_warp || enable_board_points;

    if (opt.enforce_square_pixels) {
      for (size_t k=0; k<allImgPts.size(); ++k) {
        for (int i=0; i<opt.rows*opt.cols; ++i) {
          const auto& Pw = world2D[i];
          const auto& uv = allImgPts[k][i];
          const auto& basis = boardWarpBasis[static_cast<size_t>(i)];
          ceres::CostFunction* cost =
              TelecentricReprojErrorIsoBoardFull::Create(Pw.x, Pw.y, uv.x, uv.y,
                                                         basis[0], basis[1], basis[2]);
          ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
          problem.AddResidualBlock(cost, loss,
                                   intr_iso, dist,
                                   rvecs[k].data(), tvecs[k].data(),
                                   imageResidualQuad.data(),
                                   boardWarpCoeffs.data(),
                                   boardPointDeltas[static_cast<size_t>(i)].data());
        }
      }

      problem.SetParameterLowerBound(intr_iso, 0, 1e-6);
      if (opt.lock_principal_point_to_image_center) {
        std::vector<int> constant_idx = {1, 2};
#if CERES_VERSION_MAJOR >= 2
        ceres::Manifold* m = new ceres::SubsetManifold(3, constant_idx);
        problem.SetManifold(intr_iso, m);
#else
        ceres::LocalParameterization* p = new ceres::SubsetParameterization(3, constant_idx);
        problem.SetParameterization(intr_iso, p);
#endif
      } else {
        problem.SetParameterLowerBound(intr_iso, 1, imgCx - opt.principal_point_max_offset_px);
        problem.SetParameterUpperBound(intr_iso, 1, imgCx + opt.principal_point_max_offset_px);
        problem.SetParameterLowerBound(intr_iso, 2, imgCy - opt.principal_point_max_offset_px);
        problem.SetParameterUpperBound(intr_iso, 2, imgCy + opt.principal_point_max_offset_px);
        if (opt.principal_point_prior_weight > 0.0) {
          problem.AddResidualBlock(
              PrincipalPointPriorIso::Create(imgCx, imgCy, opt.principal_point_prior_weight),
              nullptr,
              intr_iso);
        }
      }
    } else {
      for (size_t k=0; k<allImgPts.size(); ++k) {
        for (int i=0; i<opt.rows*opt.cols; ++i) {
          const auto& Pw = world2D[i];
          const auto& uv = allImgPts[k][i];
          const auto& basis = boardWarpBasis[static_cast<size_t>(i)];
          ceres::CostFunction* cost =
              TelecentricReprojErrorBoardFull::Create(Pw.x, Pw.y, uv.x, uv.y,
                                                      basis[0], basis[1], basis[2]);
          ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
          problem.AddResidualBlock(cost, loss,
                                   intr, dist,
                                   rvecs[k].data(), tvecs[k].data(),
                                   imageResidualQuad.data(),
                                   boardWarpCoeffs.data(),
                                   boardPointDeltas[static_cast<size_t>(i)].data());
        }
      }

      intr[2] = 0.0;
      problem.SetParameterLowerBound(intr, 0, 1e-6);
      problem.SetParameterLowerBound(intr, 1, 1e-6);
      std::vector<int> constant_idx = {2};
      if (opt.lock_principal_point_to_image_center) {
        constant_idx.push_back(3);
        constant_idx.push_back(4);
      } else {
        problem.SetParameterLowerBound(intr, 3, imgCx - opt.principal_point_max_offset_px);
        problem.SetParameterUpperBound(intr, 3, imgCx + opt.principal_point_max_offset_px);
        problem.SetParameterLowerBound(intr, 4, imgCy - opt.principal_point_max_offset_px);
        problem.SetParameterUpperBound(intr, 4, imgCy + opt.principal_point_max_offset_px);
        if (opt.principal_point_prior_weight > 0.0) {
          problem.AddResidualBlock(
              PrincipalPointPriorIntr::Create(imgCx, imgCy, opt.principal_point_prior_weight),
              nullptr,
              intr);
        }
      }
#if CERES_VERSION_MAJOR >= 2
      ceres::Manifold* m = new ceres::SubsetManifold(5, constant_idx);
      problem.SetManifold(intr, m);
#else
      ceres::LocalParameterization* p = new ceres::SubsetParameterization(5, constant_idx);
      problem.SetParameterization(intr, p);
#endif
    }

    if (enable_image_residual_model) {
      const double residualPriorSigma = std::max(1e-6, opt.image_residual_prior_sigma_px);
      const double residualPriorWeight = 1.0 / residualPriorSigma;
      const double residualMaxCoeff = std::max(1e-6, opt.image_residual_max_coeff_px);
      problem.AddResidualBlock(BoardWarpPrior::Create(residualPriorWeight), nullptr, imageResidualQuad.data());
      for (int j = 0; j < 6; ++j) {
        problem.SetParameterLowerBound(imageResidualQuad.data(), j, -residualMaxCoeff);
        problem.SetParameterUpperBound(imageResidualQuad.data(), j,  residualMaxCoeff);
      }
    } else {
      problem.SetParameterBlockConstant(imageResidualQuad.data());
    }

    if (enable_board_warp) {
      const double warpPriorSigma = std::max(1e-6, opt.board_warp_prior_sigma_mm);
      const double warpPriorWeight = 1.0 / warpPriorSigma;
      const double warpMaxOffset = std::max(1e-6, opt.board_warp_max_offset_mm);
      problem.AddResidualBlock(BoardWarpPrior::Create(warpPriorWeight), nullptr, boardWarpCoeffs.data());
      for (int j = 0; j < 6; ++j) {
        problem.SetParameterLowerBound(boardWarpCoeffs.data(), j, -warpMaxOffset);
        problem.SetParameterUpperBound(boardWarpCoeffs.data(), j,  warpMaxOffset);
      }
    } else {
      problem.SetParameterBlockConstant(boardWarpCoeffs.data());
    }

    if (enable_board_points) {
      const double priorSigma = std::max(1e-6, opt.board_point_prior_sigma_mm);
      const double smoothSigma = std::max(1e-6, opt.board_point_smooth_sigma_mm);
      const double priorWeight = 1.0 / priorSigma;
      const double smoothWeight = 1.0 / smoothSigma;
      const double maxOffset = std::max(1e-6, opt.board_point_max_offset_mm);

      for (int i = 0; i < opt.rows * opt.cols; ++i) {
        problem.AddResidualBlock(
            BoardPointPrior::Create(priorWeight),
            nullptr,
            boardPointDeltas[static_cast<size_t>(i)].data());
        problem.SetParameterLowerBound(boardPointDeltas[static_cast<size_t>(i)].data(), 0, -maxOffset);
        problem.SetParameterUpperBound(boardPointDeltas[static_cast<size_t>(i)].data(), 0,  maxOffset);
        problem.SetParameterLowerBound(boardPointDeltas[static_cast<size_t>(i)].data(), 1, -maxOffset);
        problem.SetParameterUpperBound(boardPointDeltas[static_cast<size_t>(i)].data(), 1,  maxOffset);
      }

      for (int r = 0; r < opt.rows; ++r) {
        for (int c = 1; c + 1 < opt.cols; ++c) {
          const int prev = r * opt.cols + (c - 1);
          const int curr = r * opt.cols + c;
          const int next = r * opt.cols + (c + 1);
          problem.AddResidualBlock(
              BoardPointSmoothness::Create(smoothWeight),
              nullptr,
              boardPointDeltas[static_cast<size_t>(prev)].data(),
              boardPointDeltas[static_cast<size_t>(curr)].data(),
              boardPointDeltas[static_cast<size_t>(next)].data());
        }
      }
      for (int c = 0; c < opt.cols; ++c) {
        for (int r = 1; r + 1 < opt.rows; ++r) {
          const int prev = (r - 1) * opt.cols + c;
          const int curr = r * opt.cols + c;
          const int next = (r + 1) * opt.cols + c;
          problem.AddResidualBlock(
              BoardPointSmoothness::Create(smoothWeight),
              nullptr,
              boardPointDeltas[static_cast<size_t>(prev)].data(),
              boardPointDeltas[static_cast<size_t>(curr)].data(),
              boardPointDeltas[static_cast<size_t>(next)].data());
        }
      }
    } else {
      for (auto& delta : boardPointDeltas) {
        problem.SetParameterBlockConstant(delta.data());
      }
    }

    if (fix_dist) {
      problem.SetParameterBlockConstant(dist);
    } else if (fix_k3_p1_p2) {
#if CERES_VERSION_MAJOR >= 2
      std::vector<int> constant_idx = {2,3,4};
      ceres::Manifold* m = new ceres::SubsetManifold(5, constant_idx);
      problem.SetManifold(dist, m);
#else
      (void)fix_k3_p1_p2;
#endif
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = print_report ? 500 : 60;
    options.minimizer_progress_to_stdout = print_report;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (opt.enforce_square_pixels) {
      intr[0] = intr_iso[0];
      intr[1] = intr_iso[0];
      intr[2] = 0.0;
      intr[3] = intr_iso[1];
      intr[4] = intr_iso[2];
    } else {
      intr[2] = 0.0;
    }

    if (print_report) {
      double rms = ComputeRMSAll(
          allImgPts, world2D, intr, dist, rvecs, tvecs,
          enable_image_residual_model ? &imageResidualQuad : nullptr,
          enable_board_warp ? &boardWarpBasis : nullptr,
          enable_board_warp ? &boardWarpCoeffs : nullptr,
          enable_board_points ? &boardPointDeltas : nullptr);
      std::cout << "\n==== " << name << " ====\n";
      std::cout << summary.BriefReport() << "\n";
      std::cout << "RMS(px) = " << rms << "\n";
      std::cout << "intrinsic_model = "
                << (opt.enforce_square_pixels ? "isotropic_telecentric" : "free_fx_fy_telecentric")
                << ", principal_point = "
                << (opt.lock_principal_point_to_image_center ? "locked_image_center" : "bounded_image_center")
                << "\n";
      std::cout << "intr (px/mm): fx=" << intr[0] << " fy=" << intr[1]
                << " skew=" << intr[2] << " cx=" << intr[3] << " cy=" << intr[4] << "\n";
      std::cout << "dist (pixel-domain): k1=" << dist[0] << " k2=" << dist[1] << " k3=" << dist[2]
                << " p1=" << dist[3] << " p2=" << dist[4] << "\n\n";
      if (enable_image_residual_model) {
        std::cout << "image_residual_rms(px) = "
                  << ComputeImageResidualFieldRmsPx(allImgPts, intr[3], intr[4], imageResidualQuad)
                  << " image_residual_max(px) = "
                  << ComputeImageResidualFieldMaxPx(allImgPts, intr[3], intr[4], imageResidualQuad)
                  << "\n";
      }
      if (any_board_compensation) {
        if (enable_board_warp) {
          std::cout << "board_warp_rms(mm) = " << ComputeBoardWarpOffsetRmsMm(boardWarpBasis, boardWarpCoeffs)
                    << " board_warp_max(mm) = " << ComputeBoardWarpOffsetMaxMm(boardWarpBasis, boardWarpCoeffs)
                    << "\n";
        }
        if (enable_board_points) {
          std::cout << "board_point_rms(mm) = " << ComputeBoardOffsetRmsMm(boardPointDeltas)
                    << " board_point_max(mm) = " << ComputeBoardOffsetMaxMm(boardPointDeltas)
                    << "\n";
        }
        std::cout << "board_total_offset_rms(mm) = "
                  << ComputeTotalBoardOffsetRmsMm(boardWarpBasis, boardWarpCoeffs, boardPointDeltas)
                  << " board_total_offset_max(mm) = "
                  << ComputeTotalBoardOffsetMaxMm(boardWarpBasis, boardWarpCoeffs, boardPointDeltas)
                  << "\n\n";
      }

      std::ostringstream stageLog;
      stageLog << name << " | " << summary.BriefReport() << " | RMS(px)=" << rms
               << " | fx=" << intr[0] << " fy=" << intr[1]
               << " | cx=" << intr[3] << " cy=" << intr[4]
               << " | k1=" << dist[0] << " k2=" << dist[1]
               << " | k3=" << dist[2] << " p1=" << dist[3] << " p2=" << dist[4];
      if (enable_image_residual_model) {
        stageLog << " | image_residual_rms(px)="
                 << ComputeImageResidualFieldRmsPx(allImgPts, intr[3], intr[4], imageResidualQuad)
                 << " | image_residual_max(px)="
                 << ComputeImageResidualFieldMaxPx(allImgPts, intr[3], intr[4], imageResidualQuad);
      }
      if (any_board_compensation) {
        if (enable_board_warp) {
          stageLog << " | board_warp_rms(mm)=" << ComputeBoardWarpOffsetRmsMm(boardWarpBasis, boardWarpCoeffs)
                   << " | board_warp_max(mm)=" << ComputeBoardWarpOffsetMaxMm(boardWarpBasis, boardWarpCoeffs);
        }
        if (enable_board_points) {
          stageLog << " | board_point_rms(mm)=" << ComputeBoardOffsetRmsMm(boardPointDeltas)
                   << " | board_point_max(mm)=" << ComputeBoardOffsetMaxMm(boardPointDeltas);
        }
        stageLog << " | board_total_offset_rms(mm)="
                 << ComputeTotalBoardOffsetRmsMm(boardWarpBasis, boardWarpCoeffs, boardPointDeltas)
                 << " | board_total_offset_max(mm)="
                 << ComputeTotalBoardOffsetMaxMm(boardWarpBasis, boardWarpCoeffs, boardPointDeltas);
      }
      emit_log(stageLog.str());
    }
  };

  std::cout << "\nInitial intr (px/mm): fx=" << intr[0] << " fy=" << intr[1]
            << " cx=" << intr[3] << " cy=" << intr[4] << std::endl;
  {
    std::ostringstream initLog;
    initLog << "Initial intr (px/mm): fx=" << intr[0] << " fy=" << intr[1]
            << " cx=" << intr[3] << " cy=" << intr[4];
    emit_log(initLog.str());
  }

  solve_stage("Stage1 (warmup)", true, false, false);
  solve_stage("Stage2 (opt k1,k2)", false, true, true);
  solve_stage("Stage3 (opt k1,k2,k3,p1,p2)", false, false, true);
  if (opt.enable_image_residual_compensation) {
    solve_stage("Stage4 (image residual compensation)", false, false, true, true);
  }
  if (opt.enable_board_warp_compensation) {
    solve_stage(opt.enable_image_residual_compensation
                    ? "Stage5 (board-warp compensation)"
                    : "Stage4 (board-warp compensation)",
                false,
                false,
                true,
                opt.enable_image_residual_compensation,
                BoardCompensationMode::WarpOnly);
  }
  if (opt.enable_board_point_compensation) {
    solve_stage(opt.enable_image_residual_compensation
                    ? (opt.enable_board_warp_compensation
                           ? "Stage6 (board-point residual compensation)"
                           : "Stage5 (board-point compensation)")
                    : (opt.enable_board_warp_compensation
                           ? "Stage5 (board-point residual compensation)"
                           : "Stage4 (board-point compensation)"),
                false,
                false,
                true,
                opt.enable_image_residual_compensation,
                BoardCompensationMode::WarpAndPoints);
  }
  const double final_rms = ComputeRMSAll(
      allImgPts, world2D, intr, dist, rvecs, tvecs,
      opt.enable_image_residual_compensation ? &imageResidualQuad : nullptr,
      opt.enable_board_warp_compensation ? &boardWarpBasis : nullptr,
      opt.enable_board_warp_compensation ? &boardWarpCoeffs : nullptr,
      opt.enable_board_point_compensation ? &boardPointDeltas : nullptr);

  if (!opt.init_path.empty()) {
    if (!SaveInit(opt.init_path, intr, dist, imageResidualQuad, usedFiles, rvecs, tvecs)) {
      if (err) *err = "Failed to save init: " + opt.init_path;
      return false;
    }
  }

  for (int i=0;i<5;++i) out_params.intr[i]=intr[i];
  for (int i=0;i<5;++i) out_params.dist[i]=dist[i];
  for (int i=0;i<6;++i) out_params.image_residual_quad[i]=imageResidualQuad[i];

  // Save a quality sidecar report for data/calibration stage traceability.
  CalibQualityReport q;
  q.total_images = totalFiles;
  q.valid_images = static_cast<int>(allImgPts.size());
  q.reproj_rms_px = final_rms;
  if (!acceptedAffineRms.empty()) {
    const double s = std::accumulate(acceptedAffineRms.begin(), acceptedAffineRms.end(), 0.0);
    q.mean_affine_rms_px = s / static_cast<double>(acceptedAffineRms.size());
  }
  q.image_width = imgW;
  q.image_height = imgH;
  q.valid_ratio_pct = (totalFiles > 0)
      ? (100.0 * static_cast<double>(q.valid_images) / static_cast<double>(totalFiles))
      : 0.0;
  q.fx_px_per_mm = intr[0];
  q.fy_px_per_mm = intr[1];
  q.fx_fy_diff_pct = RelativeDiffPercent(intr[0], intr[1]);
  q.cx_px = intr[3];
  q.cy_px = intr[4];
  q.cx_offset_px = intr[3] - imgCx;
  q.cy_offset_px = intr[4] - imgCy;
  q.intrinsic_model = opt.enforce_square_pixels ? "isotropic_telecentric" : "free_fx_fy_telecentric";
  q.principal_point_policy = opt.lock_principal_point_to_image_center ? "locked_image_center" : "bounded_image_center";

  std::vector<std::string> warnings;
  if (q.valid_ratio_pct >= 0.0 && q.valid_ratio_pct < opt.warn_valid_ratio_pct) {
    std::ostringstream oss;
    oss << "valid image ratio is low (" << q.valid_ratio_pct << "%)";
    warnings.push_back(oss.str());
  }
  if (!opt.enforce_square_pixels && q.fx_fy_diff_pct > opt.warn_fx_fy_diff_pct) {
    std::ostringstream oss;
    oss << "fx/fy differ by " << q.fx_fy_diff_pct << "%";
    warnings.push_back(oss.str());
  }
  if (!opt.lock_principal_point_to_image_center &&
      (std::abs(q.cx_offset_px) > opt.warn_principal_point_offset_px ||
       std::abs(q.cy_offset_px) > opt.warn_principal_point_offset_px)) {
    std::ostringstream oss;
    oss << "principal point offset is large (" << q.cx_offset_px << ", " << q.cy_offset_px << " px)";
    warnings.push_back(oss.str());
  }
  q.warning_summary = JoinStrings(warnings, " | ");
  q.init_path = opt.init_path;
  q.init_hash_hex = HashFileFNV1a64(opt.init_path);
  q.created_at = nowIsoLocal();
  const std::string report_path = DefaultCalibReportPath(opt.init_path);
  std::string report_err;
  if (!SaveCalibQualityReport(report_path, q, &report_err)) {
    std::cout << "[CalibReport] failed: " << report_err << std::endl;
  } else {
    std::cout << "[CalibReport] saved: " << report_path << std::endl;
  }
  if (!q.warning_summary.empty()) {
    std::cout << "[CalibReport] warning: " << q.warning_summary << std::endl;
  }
  if (quality_out) *quality_out = q;

  if (opt.enable_edge_response) {
    EdgeResponseResult edge_out;
    std::string edge_err;
    const bool edge_ok = RunCircleEdgeResponseCalibration(
        usedFiles, allImgPts, cv::Size(imgW, imgH), opt, edge_out, &edge_err);
    if (!edge_ok) {
      std::cout << "[EdgeResponse] failed: " << edge_err << std::endl;
      if (err && err->empty()) *err = "Edge-response calibration failed: " + edge_err;
    } else {
      std::cout << "[EdgeResponse] sigma0_erf_bg=" << edge_out.sigma0_erf_bg
                << " sigma0_erf_integral=" << edge_out.sigma0_erf_integral
                << " valid_bg=" << edge_out.stats_erf_bg.valid_count
                << " valid_integral=" << edge_out.stats_erf_integral.valid_count
                << std::endl;
    }
  }

  return true;
}

} // namespace telecentric
