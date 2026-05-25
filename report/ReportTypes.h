#pragma once

#include "reconstruction/ReconstructionTypes.h"
#include "registration/RegistrationTypes.h"

#include <cstddef>

namespace pinjie {

struct QualitySummary {
    std::size_t totalSteps{0};
    double meanNormalRmse{0.0};
    double meanTangentRmse{0.0};
    double meanTangentCorr{0.0};
    double meanOverlapCoverage{0.0};
    double meanInlierRatio{0.0};
    std::size_t flaggedStepCount{0};
    int worstStepIndex{-1};
    double worstNormalRmse{0.0};
};

} // namespace pinjie
