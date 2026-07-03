#pragma once

namespace pinjie::cad_design {

constexpr double kDesignProfileMinZMm = 0.0;
constexpr double kDesignProfileMaxZMm = 155.0;
constexpr double kMeasuredPixelSizeMm = 0.010057;

struct DesignEval {
    bool valid{false};
    double z_mm{0.0};
    double r_mm{0.0};
    double dr_dz{0.0};
};

bool isDesignProfileDomain(double zMm);

double evalDesignRadius(double zMm);

double evalDesignRadiusDerivative(double zMm);

DesignEval evalDesignRadiusOriginal(double zMm);

DesignEval evalDesignRadiusCompare(double sMm, bool reverseAxial);

} // namespace pinjie::cad_design
