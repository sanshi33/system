#include "cad_design/DesignProfileFunction.h"

#include <cmath>
#include <limits>

namespace pinjie::cad_design {

namespace {

constexpr double kLinearSegmentEndZMm = 52.958772;
constexpr double kPolynomialSegmentEndZMm = 100.0;
constexpr double kConstantSegment1EndZMm = 119.0;
constexpr double kPolynomialXiScale = 47.041228;

double evalPolynomialRadius(const double xi)
{
    const double xi2 = xi * xi;
    const double xi3 = xi2 * xi;
    const double xi4 = xi3 * xi;
    const double xi5 = xi4 * xi;
    const double xi6 = xi5 * xi;

    return 0.21387322 * xi6 - 0.86957897 * xi5 + 2.12875038 * xi4 - 3.85239806 * xi3 +
           15.01513050 * xi2 - 23.91622723 * xi + 193.21175337;
}

double evalPolynomialDerivativeByXi(const double xi)
{
    const double xi2 = xi * xi;
    const double xi3 = xi2 * xi;
    const double xi4 = xi3 * xi;
    const double xi5 = xi4 * xi;

    return 6.0 * 0.21387322 * xi5 - 5.0 * 0.86957897 * xi4 + 4.0 * 2.12875038 * xi3 -
           3.0 * 3.85239806 * xi2 + 2.0 * 15.01513050 * xi - 23.91622723;
}

} // namespace

bool isDesignProfileDomain(const double zMm)
{
    return std::isfinite(zMm) && zMm >= kDesignProfileMinZMm && zMm <= kDesignProfileMaxZMm;
}

DesignEval evalDesignRadiusOriginal(const double zMm)
{
    DesignEval eval;
    if (!isDesignProfileDomain(zMm)) {
        return eval;
    }

    eval.valid = true;
    eval.z_mm = zMm;
    if (zMm <= kLinearSegmentEndZMm) {
        eval.r_mm = 220.11920702 - 0.50803027 * zMm;
        eval.dr_dz = -0.50803027;
        return eval;
    }

    if (zMm <= kPolynomialSegmentEndZMm) {
        const double xi = (zMm - kLinearSegmentEndZMm) / kPolynomialXiScale;
        eval.r_mm = evalPolynomialRadius(xi);
        eval.dr_dz = evalPolynomialDerivativeByXi(xi) / kPolynomialXiScale;
        return eval;
    }

    if (zMm <= kConstantSegment1EndZMm) {
        eval.r_mm = 181.931189;
        eval.dr_dz = 0.0;
        return eval;
    }

    eval.r_mm = 179.919242;
    eval.dr_dz = 0.0;
    return eval;
}

DesignEval evalDesignRadiusCompare(const double sMm, const bool reverseZ)
{
    if (!std::isfinite(sMm)) {
        return {};
    }

    const double zOriginal = reverseZ ? (kDesignProfileMaxZMm - sMm) : sMm;
    DesignEval eval = evalDesignRadiusOriginal(zOriginal);
    if (!eval.valid) {
        return eval;
    }

    eval.z_mm = sMm;
    if (reverseZ) {
        eval.dr_dz = -eval.dr_dz;
    }
    return eval;
}

double evalDesignRadius(const double zMm)
{
    const DesignEval eval = evalDesignRadiusOriginal(zMm);
    return eval.valid ? eval.r_mm : std::numeric_limits<double>::quiet_NaN();
}

double evalDesignRadiusDerivative(const double zMm)
{
    const DesignEval eval = evalDesignRadiusOriginal(zMm);
    return eval.valid ? eval.dr_dz : std::numeric_limits<double>::quiet_NaN();
}

} // namespace pinjie::cad_design
