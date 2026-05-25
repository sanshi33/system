#pragma once

#include "report/ReportTypes.h"

#include <string>
#include <vector>

namespace pinjie {

QualitySummary buildQualitySummary(const std::vector<StitchStepRecord>& steps);

std::string buildQualitySummaryText(const std::vector<StitchStepRecord>& steps,
                                    const QualitySummary& summary);

} // namespace pinjie
