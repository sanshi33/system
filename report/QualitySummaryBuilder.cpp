#include "report/QualitySummaryBuilder.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>

namespace pinjie {

namespace {

const stitch::ResidualStatistics& displayNormalStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
}

const stitch::ResidualStatistics& displayTangentStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
}

double displayTangentCorr(const stitch::AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;
}

bool isFlaggedStep(const StitchStepRecord& step)
{
    const auto& metrics = step.transform.metrics;
    return displayNormalStats(metrics).rmse > 0.2 || metrics.overlapCoverageRatio < 0.3;
}

std::string formatDouble(double value, int precision = 4)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

} // namespace

QualitySummary buildQualitySummary(const std::vector<StitchStepRecord>& steps)
{
    QualitySummary summary;
    summary.totalSteps = steps.size();
    if (steps.empty()) {
        return summary;
    }

    for (const auto& step : steps) {
        const auto& metrics = step.transform.metrics;
        const double normalRmse = displayNormalStats(metrics).rmse;
        const double tangentRmse = displayTangentStats(metrics).rmse;
        const double tangentCorr = displayTangentCorr(metrics);

        summary.meanNormalRmse += normalRmse;
        summary.meanTangentRmse += tangentRmse;
        summary.meanTangentCorr += tangentCorr;
        summary.meanOverlapCoverage += metrics.overlapCoverageRatio;
        summary.meanInlierRatio += metrics.inlierRatio;

        if (summary.worstStepIndex < 0 || normalRmse > summary.worstNormalRmse) {
            summary.worstStepIndex = static_cast<int>(step.stepIndex);
            summary.worstNormalRmse = normalRmse;
        }

        if (isFlaggedStep(step)) {
            ++summary.flaggedStepCount;
        }
    }

    const double stepCount = static_cast<double>(steps.size());
    summary.meanNormalRmse /= stepCount;
    summary.meanTangentRmse /= stepCount;
    summary.meanTangentCorr /= stepCount;
    summary.meanOverlapCoverage /= stepCount;
    summary.meanInlierRatio /= stepCount;
    return summary;
}

std::string buildQualitySummaryText(const std::vector<StitchStepRecord>& steps,
                                    const QualitySummary& summary)
{
    std::ostringstream stream;
    stream << "质量摘要\n"
           << "总步骤数：" << summary.totalSteps << "\n"
           << "平均法向 RMSE：" << formatDouble(summary.meanNormalRmse) << " px\n"
           << "平均切向 RMSE：" << formatDouble(summary.meanTangentRmse) << " px\n"
           << "平均切向相关：" << formatDouble(summary.meanTangentCorr) << "\n"
           << "平均重叠覆盖率：" << formatDouble(summary.meanOverlapCoverage) << "\n"
           << "平均内点率：" << formatDouble(summary.meanInlierRatio) << "\n"
           << "风险步骤数：" << summary.flaggedStepCount << "\n";

    if (summary.worstStepIndex >= 0) {
        stream << "最差步骤：" << summary.worstStepIndex
               << "（法向 RMSE " << formatDouble(summary.worstNormalRmse) << " px）\n";
    }

    if (steps.empty()) {
        return stream.str();
    }

    std::vector<std::tuple<double, double, double, const StitchStepRecord*>> rankedSteps;
    rankedSteps.reserve(steps.size());
    for (const auto& step : steps) {
        const auto& metrics = step.transform.metrics;
        rankedSteps.emplace_back(displayNormalStats(metrics).rmse,
                                 metrics.overlapCoverageRatio,
                                 displayTangentCorr(metrics),
                                 &step);
    }

    std::sort(rankedSteps.begin(), rankedSteps.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (std::get<0>(lhs) != std::get<0>(rhs)) {
                      return std::get<0>(lhs) > std::get<0>(rhs);
                  }
                  return std::get<1>(lhs) < std::get<1>(rhs);
              });

    stream << "\n高风险步骤\n";
    const std::size_t limit = std::min<std::size_t>(3, rankedSteps.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& [normalRmse, overlapCoverage, tangentCorr, stepPtr] = rankedSteps[i];
        stream << "步骤 " << stepPtr->stepIndex
               << "：法向 RMSE " << formatDouble(normalRmse)
               << " px，覆盖率 " << formatDouble(overlapCoverage)
               << "，切向相关 " << formatDouble(tangentCorr)
               << "，dx " << formatDouble(stepPtr->transform.dx, 2)
               << "，dy " << formatDouble(stepPtr->transform.dy, 2)
               << "，角度 " << formatDouble(stepPtr->transform.da, 3)
               << "\n";
    }

    return stream.str();
}

} // namespace pinjie
