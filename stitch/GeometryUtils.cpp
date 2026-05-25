#include "GeometryUtils.h"
#include <algorithm>
#include <cmath>

namespace stitch {
using namespace cv;
using namespace std;
using cv::Point2d;
using std::vector;

double computeAngle(const Point2d &pt, const Point2d &center)
{
    return atan2(pt.y - center.y, pt.x - center.x) * 180.0 / CV_PI;
}

double computeLocalCurvature(const std::vector<cv::Point2d>& pts, int idx, int window)
{
    if (pts.size() < static_cast<size_t>(2 * window + 1)) return 0.0;
    idx = std::max(window, std::min<int>(static_cast<int>(pts.size()) - window - 1, idx));
    const Point2d& p1 = pts[idx - window];
    const Point2d& p2 = pts[idx];
    const Point2d& p3 = pts[idx + window];

    double ax = p2.x - p1.x, ay = p2.y - p1.y;
    double bx = p3.x - p2.x, by = p3.y - p2.y;

    double cross = ax * by - ay * bx;
    double a_len = sqrt(ax * ax + ay * ay);
    double b_len = sqrt(bx * bx + by * by);

    if (a_len < 1e-6 || b_len < 1e-6)
        return 0;

    double c_len = sqrt((p3.x - p1.x) * (p3.x - p1.x) + (p3.y - p1.y) * (p3.y - p1.y));
    if (c_len < 1e-6)
        return 0;

    return 2.0 * abs(cross) / (a_len * b_len * c_len);
}

double computeTangentChange(const std::vector<cv::Point2d>& pts, int idx, int window)
{
    if (pts.size() < static_cast<size_t>(2 * window + 1)) return 0.0;
    idx = std::max(window, std::min<int>(static_cast<int>(pts.size()) - window - 1, idx));
    const Point2d& p1 = pts[idx - window];
    const Point2d& p2 = pts[idx];
    const Point2d& p3 = pts[idx + window];

    double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
    double angle2 = atan2(p3.y - p2.y, p3.x - p2.x);

    double diff = angle2 - angle1;
    while (diff > CV_PI) diff -= 2 * CV_PI;
    while (diff < -CV_PI) diff += 2 * CV_PI;

    return abs(diff);
}

void sortContourByX(std::vector<cv::Point2d> &contour)
{
    std::sort(contour.begin(), contour.end(),
              [](const cv::Point2d &a, const cv::Point2d &b)
              { return a.x < b.x; });
}

void sortContourByY(std::vector<cv::Point2d> &contour)
{
    std::sort(contour.begin(), contour.end(),
              [](const cv::Point2d &a, const cv::Point2d &b)
              { return a.y < b.y; });
}

bool getInterpolatedY(const std::vector<cv::Point2d> &contour, double x, double &y_out)
{
    if (contour.empty())
        return false;
    if (x < contour.front().x || x > contour.back().x)
        return false;

    auto it = std::lower_bound(contour.begin(), contour.end(), cv::Point2d(x, -1e9),
                               [](const cv::Point2d &pt, const cv::Point2d &val)
                               { return pt.x < val.x; });

    if (it == contour.begin())
    {
        y_out = it->y;
        return true;
    }

    const auto &p2 = *it;
    const auto &p1 = *(it - 1);

    if (std::abs(p2.x - p1.x) < 1e-6)
        y_out = p1.y;
    else
    {
        double ratio = (x - p1.x) / (p2.x - p1.x);
        y_out = p1.y + ratio * (p2.y - p1.y);
    }
    return true;
}

bool getInterpolatedX(const std::vector<cv::Point2d> &contour, double y, double &x_out)
{
    if (contour.empty())
        return false;
    if (y < contour.front().y || y > contour.back().y)
        return false;

    auto it = std::lower_bound(contour.begin(), contour.end(), cv::Point2d(-1e9, y),
                               [](const cv::Point2d &pt, const cv::Point2d &val)
                               { return pt.y < val.y; });

    if (it == contour.begin())
    {
        x_out = it->x;
        return true;
    }

    const auto &p2 = *it;
    const auto &p1 = *(it - 1);

    if (std::abs(p2.y - p1.y) < 1e-6)
        x_out = p1.x;
    else
    {
        double ratio = (y - p1.y) / (p2.y - p1.y);
        x_out = p1.x + ratio * (p2.x - p1.x);
    }
    return true;
}

cv::Point2d catmullRomInterpolate(const cv::Point2d& p0, const cv::Point2d& p1,
                                   const cv::Point2d& p2, const cv::Point2d& p3,
                                   double t, cv::Point2d* out_tangent)
{
    const double t2 = t * t;
    const double t3 = t2 * t;

    const double x = 0.5 * ((2.0 * p1.x) +
                             (-p0.x + p2.x) * t +
                             (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2 +
                             (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

    const double y = 0.5 * ((2.0 * p1.y) +
                             (-p0.y + p2.y) * t +
                             (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2 +
                             (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

    if (out_tangent != nullptr) {
        const double tx = 0.5 * ((-p0.x + p2.x) +
                                  2.0 * (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t +
                                  3.0 * (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t2);
        const double ty = 0.5 * ((-p0.y + p2.y) +
                                  2.0 * (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t +
                                  3.0 * (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t2);
        *out_tangent = cv::Point2d(tx, ty);
    }

    return cv::Point2d(x, y);
}

void rotatePoints(const std::vector<cv::Point2d> &in, std::vector<cv::Point2d> &out,
                  double angle_deg, cv::Point2d center)
{
    out.resize(in.size());
    double rad = angle_deg * CV_PI / 180.0;
    double cos_a = std::cos(rad);
    double sin_a = std::sin(rad);

    for (size_t i = 0; i < in.size(); ++i)
    {
        double x = in[i].x - center.x;
        double y = in[i].y - center.y;
        out[i].x = x * cos_a - y * sin_a + center.x;
        out[i].y = x * sin_a + y * cos_a + center.y;
    }
}

} // namespace stitch
