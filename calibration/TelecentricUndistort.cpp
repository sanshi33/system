/*
==============================================================================
文件：TelecentricUndistort.cpp
------------------------------------------------------------------------------
这里实现远心模型的畸变与反畸变。

注意：这里的畸变模型定义在**像素域**（pixel-domain），
即畸变直接作用在 (u,v) 像素坐标相对于主点的偏移上。
远心模型的线性投影（fx/fy, skew, cx, cy）用于把物理平面映射到像素，
畸变则在像素平面进行（与 fx/fy 无关）。

==============================================================================
*/

#include "TelecentricUndistort.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>

namespace telecentric {

static constexpr double kImageResidualBasisScalePx = 1000.0;

static inline void ApplyImageResidualQuadPx(const CalibParams& p,
                                            const double x_base,
                                            const double y_base,
                                            double& x_obs,
                                            double& y_obs) {
    const double xn = x_base / kImageResidualBasisScalePx;
    const double yn = y_base / kImageResidualBasisScalePx;
    const double b0 = xn * xn;
    const double b1 = xn * yn;
    const double b2 = yn * yn;

    x_obs += p.image_residual_quad[0] * b0 + p.image_residual_quad[1] * b1 + p.image_residual_quad[2] * b2;
    y_obs += p.image_residual_quad[3] * b0 + p.image_residual_quad[4] * b1 + p.image_residual_quad[5] * b2;
}

static inline void DistortPx(const CalibParams& p, double x, double y, double& xd, double& yd) {
    const double k1 = p.dist[0];
    const double k2 = p.dist[1];
    const double k3 = p.dist[2];
    const double p1 = p.dist[3];
    const double p2 = p.dist[4];

    const double r2 = x*x + y*y;
    const double r4 = r2*r2;
    const double r6 = r4*r2;

    const double radial = 1.0 + k1*r2 + k2*r4 + k3*r6;

    const double xr = x * radial;
    const double yr = y * radial;

    const double x_tan = 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
    const double y_tan = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y;

    xd = xr + x_tan;
    yd = yr + y_tan;
    ApplyImageResidualQuadPx(p, xd, yd, xd, yd);
}

static inline void DistortJacobian(const CalibParams& p, double x, double y,
                                   double& j11, double& j12, double& j21, double& j22) {
    const double k1 = p.dist[0];
    const double k2 = p.dist[1];
    const double k3 = p.dist[2];
    const double p1 = p.dist[3];
    const double p2 = p.dist[4];

    const double r2 = x*x + y*y;
    const double r4 = r2*r2;

    const double radial = 1.0 + k1*r2 + k2*r4 + k3*r4*r2;

    // d(radial)/dx, d(radial)/dy
    const double d_radial_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4;
    const double d_radial_dx  = 2.0*x*d_radial_dr2;
    const double d_radial_dy  = 2.0*y*d_radial_dr2;

    // xr = x*radial; yr = y*radial
    const double dxr_dx = radial + x*d_radial_dx;
    const double dxr_dy = x*d_radial_dy;
    const double dyr_dx = y*d_radial_dx;
    const double dyr_dy = radial + y*d_radial_dy;

    // tangential terms:
    // x_tan = 2p1xy + p2(r2 + 2x^2) = 2p1xy + p2(3x^2 + y^2)
    const double dxtan_dx = 2.0*p1*y + p2*(6.0*x);
    const double dxtan_dy = 2.0*p1*x + p2*(2.0*y);

    // y_tan = p1(r2 + 2y^2) + 2p2xy = p1(x^2 + 3y^2) + 2p2xy
    const double dytan_dx = p1*(2.0*x) + 2.0*p2*y;
    const double dytan_dy = p1*(6.0*y) + 2.0*p2*x;

    const double base_x = x * radial + (2.0*p1*x*y + p2*(3.0*x*x + y*y));
    const double base_y = y * radial + (p1*(x*x + 3.0*y*y) + 2.0*p2*x*y);
    const double j11_base = dxr_dx + dxtan_dx; // d(base_x)/dx
    const double j12_base = dxr_dy + dxtan_dy; // d(base_x)/dy
    const double j21_base = dyr_dx + dytan_dx; // d(base_y)/dx
    const double j22_base = dyr_dy + dytan_dy; // d(base_y)/dy

    const double xn = base_x / kImageResidualBasisScalePx;
    const double yn = base_y / kImageResidualBasisScalePx;
    const double dresx_dbasex =
        (2.0 * p.image_residual_quad[0] * xn + p.image_residual_quad[1] * yn) / kImageResidualBasisScalePx;
    const double dresx_dbasey =
        (p.image_residual_quad[1] * xn + 2.0 * p.image_residual_quad[2] * yn) / kImageResidualBasisScalePx;
    const double dresy_dbasex =
        (2.0 * p.image_residual_quad[3] * xn + p.image_residual_quad[4] * yn) / kImageResidualBasisScalePx;
    const double dresy_dbasey =
        (p.image_residual_quad[4] * xn + 2.0 * p.image_residual_quad[5] * yn) / kImageResidualBasisScalePx;

    j11 = (1.0 + dresx_dbasex) * j11_base + dresx_dbasey * j21_base;
    j12 = (1.0 + dresx_dbasex) * j12_base + dresx_dbasey * j22_base;
    j21 = dresy_dbasex * j11_base + (1.0 + dresy_dbasey) * j21_base;
    j22 = dresy_dbasex * j12_base + (1.0 + dresy_dbasey) * j22_base;
}

static bool UndistortOnePx(const CalibParams& p, double xd, double yd, double& x, double& y, int max_iter) {
    x = xd; y = yd;
    for (int it=0; it<max_iter; ++it) {
        double x_est, y_est;
        DistortPx(p, x, y, x_est, y_est);
        const double ex = xd - x_est;
        const double ey = yd - y_est;

        if (std::abs(ex) + std::abs(ey) < 1e-14)
            return true;

        double j11,j12,j21,j22;
        DistortJacobian(p, x, y, j11,j12,j21,j22);
        const double det = j11*j22 - j12*j21;
        if (std::abs(det) < 1e-20) {
            // fallback: simple additive step
            x += ex;
            y += ey;
            continue;
        }
        // delta = J^{-1} * e
        const double dx = ( j22*ex - j12*ey) / det;
        const double dy = (-j21*ex + j11*ey) / det;
        x += dx;
        y += dy;

        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        if (std::abs(dx) + std::abs(dy) < 1e-14) return true;
    }
    return true;
}

bool UndistortPointsPxToPx(const std::vector<cv::Point2d>& in_px,
                           std::vector<cv::Point2d>& out_px,
                           const CalibParams& p,
                           int max_iter) {
    out_px.clear();
    out_px.reserve(in_px.size());
    const double cx = p.intr[3];
    const double cy = p.intr[4];
    for (const auto& pt : in_px) {
        const double xd = pt.x - cx;
        const double yd = pt.y - cy;

        double x, y;
        if (!UndistortOnePx(p, xd, yd, x, y, max_iter)) {
            out_px.emplace_back(pt); // fallback
            continue;
        }

        out_px.emplace_back(x + cx, y + cy);
    }
    return true;
}

bool BuildUndistortMaps(const cv::Size& image_size,
                        cv::Mat& map_x,
                        cv::Mat& map_y,
                        const CalibParams& p) {
    if (image_size.width <= 0 || image_size.height <= 0) return false;

    map_x.create(image_size, CV_32FC1);
    map_y.create(image_size, CV_32FC1);

    const double cx = p.intr[3];
    const double cy = p.intr[4];
    for (int ypix = 0; ypix < image_size.height; ++ypix) {
        float* mx = map_x.ptr<float>(ypix);
        float* my = map_y.ptr<float>(ypix);
        for (int xpix = 0; xpix < image_size.width; ++xpix) {
            const double u = static_cast<double>(xpix);
            const double v = static_cast<double>(ypix);

            double x_d = 0.0;
            double y_d = 0.0;
            DistortPx(p, u - cx, v - cy, x_d, y_d);
            mx[xpix] = static_cast<float>(x_d + cx);
            my[xpix] = static_cast<float>(y_d + cy);
        }
    }

    return true;
}

bool UndistortImage(const cv::Mat& img, cv::Mat& out, const CalibParams& p,
                    int interp, int border_mode) {
    if (img.empty()) return false;
    out.create(img.size(), img.type());

    cv::Mat map_x;
    cv::Mat map_y;
    if (!BuildUndistortMaps(img.size(), map_x, map_y, p)) {
        return false;
    }

    cv::remap(img, out, map_x, map_y, interp, border_mode);
    return true;
}

} // namespace telecentric
