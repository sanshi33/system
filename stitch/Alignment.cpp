#include "Alignment.h"
#include "GeometryUtils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace stitch {
using namespace cv;
using namespace std;
using cv::Point2d;
using std::vector;

namespace {

struct SampledPair {
    double primary{0.0};
    double normalErr{0.0};
    double tangentErr{0.0};
    double refSecondary{0.0};
    double targetSecondary{0.0};
};

struct TangentMatchCost {
    double residualCost{0.0};
    double correlationCost{0.0};

    [[nodiscard]] double total() const noexcept
    {
        return residualCost + correlationCost;
    }
};

double computeQuantile(vector<double> values, double q)
{
    if (values.empty())
        return 0.0;

    q = std::clamp(q, 0.0, 1.0);
    const size_t idx = static_cast<size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

double computePrimarySpan(const vector<double>& values)
{
    if (values.size() < 2) {
        return 0.0;
    }

    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    return std::max(0.0, *maxIt - *minIt);
}

double computeReferencePrimarySpan(const vector<Point2d>& refEdges, AlignmentAxis axis)
{
    if (refEdges.size() < 2) {
        return 0.0;
    }

    const double span = (axis == AlignmentAxis::X)
                            ? (refEdges.back().x - refEdges.front().x)
                            : (refEdges.back().y - refEdges.front().y);
    return std::max(0.0, span);
}

double computeCoverageRatio(double span, double referenceSpan)
{
    if (referenceSpan < 1e-9) {
        return 0.0;
    }
    return std::clamp(span / referenceSpan, 0.0, 1.0);
}

double preferredNormalRmse(const AlignmentMetrics& metrics)
{
    const ResidualStatistics& normal = metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
    return normal.valid() ? normal.rmse : std::numeric_limits<double>::infinity();
}

double preferredTangentRmse(const AlignmentMetrics& metrics)
{
    const ResidualStatistics& tangent = metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
    return tangent.valid() ? tangent.rmse : 0.0;
}

double preferredCoverageRatio(const AlignmentMetrics& metrics)
{
    if (metrics.inlierCount > 0) {
        return metrics.inlierCoverageRatio;
    }
    return metrics.overlapCoverageRatio;
}

double preferredTangentCorrelation(const AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;
}

double primaryShiftDeltaPx(AlignmentAxis axis,
                           double dx,
                           double dy,
                           double approxShift,
                           double approxShiftY)
{
    return axis == AlignmentAxis::X ? dx - approxShift : dy - approxShiftY;
}

double perpendicularShiftDeltaPx(AlignmentAxis axis,
                                 double dx,
                                 double dy,
                                 double approxShift,
                                 double approxShiftY)
{
    return axis == AlignmentAxis::X ? dy - approxShiftY : dx - approxShift;
}

double diagnosticSelectionCost(const AlignmentCandidateDiagnostic& candidate,
                               double approxShift,
                               double approxShiftY)
{
    const double normalRmse = preferredNormalRmse(candidate.metrics);
    if (!std::isfinite(normalRmse)) {
        return std::numeric_limits<double>::infinity();
    }

    const double tangentRmse = preferredTangentRmse(candidate.metrics);
    const double primaryDelta =
        primaryShiftDeltaPx(candidate.axis, candidate.dx, candidate.dy, approxShift, approxShiftY);
    const double perpDelta =
        perpendicularShiftDeltaPx(candidate.axis, candidate.dx, candidate.dy, approxShift, approxShiftY);
    const double priorCost =
        (primaryDelta * primaryDelta) / (40.0 * 40.0) +
        (perpDelta * perpDelta) / (16.0 * 16.0);
    const double angleCost = (candidate.da * candidate.da) / (0.2 * 0.2);
    return normalRmse * normalRmse +
           0.05 * tangentRmse * tangentRmse +
           0.01 * priorCost +
           0.008 * angleCost;
}

double transformSelectionCost(const TransformResult& result,
                              double approxShift,
                              double approxShiftY)
{
    const double normalRmse = preferredNormalRmse(result.metrics);
    if (!std::isfinite(normalRmse)) {
        return std::numeric_limits<double>::infinity();
    }

    const double tangentRmse = preferredTangentRmse(result.metrics);
    const double primaryDelta =
        primaryShiftDeltaPx(result.axis, result.dx, result.dy, approxShift, approxShiftY);
    const double perpDelta =
        perpendicularShiftDeltaPx(result.axis, result.dx, result.dy, approxShift, approxShiftY);
    const double priorCost =
        (primaryDelta * primaryDelta) / (40.0 * 40.0) +
        (perpDelta * perpDelta) / (16.0 * 16.0);
    const double angleCost = (result.da * result.da) / (0.2 * 0.2);
    return normalRmse * normalRmse +
           0.05 * tangentRmse * tangentRmse +
           0.01 * priorCost +
           0.008 * angleCost;
}

vector<unsigned char> computeOverlapCoreMask(const vector<double>& primaryValues)
{
    vector<unsigned char> mask(primaryValues.size(), 1);
    if (primaryValues.size() < 20) {
        return mask;
    }

    const auto [minIt, maxIt] = std::minmax_element(primaryValues.begin(), primaryValues.end());
    const double span = *maxIt - *minIt;
    if (!(span > 40.0)) {
        return mask;
    }

    const double margin = std::min(80.0, span * 0.08);
    if (span <= 2.0 * margin + 20.0) {
        return mask;
    }

    const double lower = *minIt + margin;
    const double upper = *maxIt - margin;
    int kept = 0;
    for (size_t i = 0; i < primaryValues.size(); ++i) {
        const bool keep = primaryValues[i] >= lower && primaryValues[i] <= upper;
        mask[i] = keep ? 1 : 0;
        if (keep) {
            ++kept;
        }
    }

    const int minKept = std::max(10, static_cast<int>(std::ceil(primaryValues.size() * 0.55)));
    if (kept < minKept) {
        std::fill(mask.begin(), mask.end(), 1);
    }
    return mask;
}

ResidualStatistics computeResidualStatistics(const vector<double>& signedErrors)
{
    ResidualStatistics stats;
    stats.sampleCount = static_cast<int>(signedErrors.size());
    if (signedErrors.empty()) {
        return stats;
    }

    vector<double> absErrors;
    absErrors.reserve(signedErrors.size());

    double sum = 0.0;
    double sumSq = 0.0;
    double sumAbs = 0.0;
    for (double err : signedErrors) {
        const double absErr = std::abs(err);
        sum += err;
        sumSq += err * err;
        sumAbs += absErr;
        absErrors.push_back(absErr);
    }

    const double sampleCount = static_cast<double>(signedErrors.size());
    stats.bias = sum / sampleCount;
    stats.rmse = std::sqrt(sumSq / sampleCount);
    stats.meanAbs = sumAbs / sampleCount;
    stats.medianAbs = computeQuantile(absErrors, 0.5);
    stats.p95Abs = computeQuantile(absErrors, 0.95);
    stats.maxAbs = absErrors.empty() ? 0.0 : *std::max_element(absErrors.begin(), absErrors.end());
    return stats;
}

double computePearsonCorrelation(const vector<double>& a, const vector<double>& b)
{
    if (a.size() != b.size() || a.size() < 2)
        return 0.0;

    const double meanA = std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size());
    const double meanB = std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size());

    double cov = 0.0;
    double varA = 0.0;
    double varB = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double da = a[i] - meanA;
        const double db = b[i] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
    }

    if (varA < 1e-12 && varB < 1e-12)
        return 1.0;
    if (varA < 1e-12 || varB < 1e-12)
        return 0.0;

    return std::clamp(cov / std::sqrt(varA * varB), -1.0, 1.0);
}

bool collectAlignmentSamples(const vector<Point2d>& refEdges,
                             const vector<Point2d>& rotatedTargetEdges,
                             const TransformResult& res,
                             vector<SampledPair>& outSamples)
{
    outSamples.clear();
    if (refEdges.size() < 2 || rotatedTargetEdges.empty())
        return false;

    const Point2d translation(res.dx, res.dy);

    for (const auto& rawPt : rotatedTargetEdges)
    {
        const Point2d pt = rawPt + translation;

        if (res.axis == TransformResult::AlignAxis::X)
        {
            if (pt.x < refEdges.front().x || pt.x > refEdges.back().x)
                continue;

            auto it = std::lower_bound(refEdges.begin(), refEdges.end(), Point2d(pt.x, -1e9),
                                       [](const Point2d& lhs, const Point2d& rhs)
                                       { return lhs.x < rhs.x; });
            if (it == refEdges.begin() || it == refEdges.end())
                continue;

            const double dxSegment = it->x - (it - 1)->x;
            if (std::abs(dxSegment) < 1e-9)
                continue;

            const double ratio = (pt.x - (it - 1)->x) / dxSegment;

            Point2d refPt;
            Point2d tangent;
            const bool canUseSpline = (it - 1 > refEdges.begin() && it + 1 < refEdges.end());
            if (canUseSpline) {
                refPt = catmullRomInterpolate(*(it - 2), *(it - 1), *it, *(it + 1), ratio, &tangent);
            } else {
                const Point2d& p1 = *(it - 1);
                const Point2d& p2 = *it;
                refPt = p1 + ratio * (p2 - p1);
                tangent = p2 - p1;
            }

            const double tangentNorm = cv::norm(tangent);
            if (tangentNorm < 1e-9)
                continue;

            const Point2d tangentUnit = tangent * (1.0 / tangentNorm);
            const Point2d normalUnit(-tangentUnit.y, tangentUnit.x);
            const Point2d delta = pt - refPt;

            SampledPair sample;
            sample.primary = pt.x;
            sample.normalErr = delta.dot(normalUnit);
            sample.tangentErr = delta.dot(tangentUnit);
            sample.refSecondary = refPt.y;
            sample.targetSecondary = pt.y;
            outSamples.push_back(sample);
        }
        else
        {
            if (pt.y < refEdges.front().y || pt.y > refEdges.back().y)
                continue;

            auto it = std::lower_bound(refEdges.begin(), refEdges.end(), Point2d(-1e9, pt.y),
                                       [](const Point2d& lhs, const Point2d& rhs)
                                       { return lhs.y < rhs.y; });
            if (it == refEdges.begin() || it == refEdges.end())
                continue;

            const double dySegment = it->y - (it - 1)->y;
            if (std::abs(dySegment) < 1e-9)
                continue;

            const double ratio = (pt.y - (it - 1)->y) / dySegment;

            Point2d refPt;
            Point2d tangent;
            const bool canUseSpline = (it - 1 > refEdges.begin() && it + 1 < refEdges.end());
            if (canUseSpline) {
                refPt = catmullRomInterpolate(*(it - 2), *(it - 1), *it, *(it + 1), ratio, &tangent);
            } else {
                const Point2d& p1 = *(it - 1);
                const Point2d& p2 = *it;
                refPt = p1 + ratio * (p2 - p1);
                tangent = p2 - p1;
            }

            const double tangentNorm = cv::norm(tangent);
            if (tangentNorm < 1e-9)
                continue;

            const Point2d tangentUnit = tangent * (1.0 / tangentNorm);
            const Point2d normalUnit(-tangentUnit.y, tangentUnit.x);
            const Point2d delta = pt - refPt;

            SampledPair sample;
            sample.primary = pt.y;
            sample.normalErr = delta.dot(normalUnit);
            sample.tangentErr = delta.dot(tangentUnit);
            sample.refSecondary = refPt.x;
            sample.targetSecondary = pt.x;
            outSamples.push_back(sample);
        }
    }

    return !outSamples.empty();
}

TangentMatchCost computeTangentMatchCost(const vector<Point2d>& refEdges,
                                         const vector<Point2d>& rotatedTargetEdges,
                                         const TransformResult& candidate,
                                         double tangentResidualCostWeight,
                                         double tangentCorrelationCostWeight)
{
    TangentMatchCost cost;
    tangentResidualCostWeight = std::max(0.0, tangentResidualCostWeight);
    tangentCorrelationCostWeight = std::max(0.0, tangentCorrelationCostWeight);
    if (tangentResidualCostWeight <= 0.0 && tangentCorrelationCostWeight <= 0.0)
        return cost;

    vector<SampledPair> samples;
    if (!collectAlignmentSamples(refEdges, rotatedTargetEdges, candidate, samples))
    {
        cost.residualCost = 1e9;
        return cost;
    }

    vector<double> normalErrs;
    vector<double> primaryValues;
    normalErrs.reserve(samples.size());
    primaryValues.reserve(samples.size());
    for (const auto& sample : samples) {
        normalErrs.push_back(sample.normalErr);
        primaryValues.push_back(sample.primary);
    }

    const vector<unsigned char> coreMask(primaryValues.size(), 1);

    double center = 0.0;
    double threshold = std::numeric_limits<double>::infinity();
    if (normalErrs.size() >= 3)
    {
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < normalErrs.size(); ++i) {
            if (!coreMask[i]) {
                continue;
            }
            sum += normalErrs[i];
            ++count;
        }
        center = count > 0 ? sum / static_cast<double>(count) : 0.0;

        vector<double> absDevs;
        absDevs.reserve(normalErrs.size());
        for (size_t i = 0; i < normalErrs.size(); ++i) {
            if (coreMask[i]) {
                absDevs.push_back(std::abs(normalErrs[i] - center));
            }
        }
        threshold = computeQuantile(absDevs, 0.8);
    }

    vector<double> tangentErrs;
    vector<double> refSecondary;
    vector<double> targetSecondary;
    tangentErrs.reserve(samples.size());
    refSecondary.reserve(samples.size());
    targetSecondary.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (!coreMask[i])
            continue;
        if (std::abs(normalErrs[i] - center) > threshold)
            continue;
        tangentErrs.push_back(samples[i].tangentErr);
        refSecondary.push_back(samples[i].refSecondary);
        targetSecondary.push_back(samples[i].targetSecondary);
    }

    if (tangentErrs.empty())
    {
        cost.residualCost = 1e9;
        return cost;
    }

    const ResidualStatistics tangentStats = computeResidualStatistics(tangentErrs);
    const double tangentCorr = computePearsonCorrelation(refSecondary, targetSecondary);

    cost.residualCost = tangentResidualCostWeight * tangentStats.rmse * tangentStats.rmse;
    cost.correlationCost = tangentCorrelationCostWeight * std::max(0.0, 1.0 - tangentCorr);
    return cost;
}

} // namespace

double computeRobustCost(const vector<Point2d>& edges1,
                         const vector<Point2d>& edges2,
                         double offset_primary,
                         double& out_mean_perp,
                         vector<double>* out_inlier_errs,
                         vector<double>* out_inlier_coords,
                         TransformResult::AlignAxis axis)
{
    vector<double> diffs;
    vector<double> coordsMapped;
    diffs.reserve(edges2.size());
    coordsMapped.reserve(edges2.size());

    for (size_t i = 0; i < edges2.size(); ++i)
    {
        if (axis == TransformResult::AlignAxis::X)
        {
            const double x2Mapped = edges2[i].x + offset_primary;
            if (x2Mapped < edges1.front().x || x2Mapped > edges1.back().x)
                continue;

            auto it = std::lower_bound(edges1.begin(), edges1.end(), Point2d(x2Mapped, -1e9),
                                       [](const Point2d& pt, const Point2d& val)
                                       { return pt.x < val.x; });

            if (it != edges1.begin() && it != edges1.end())
            {
                const auto& p1 = *(it - 1);
                const double dxSegment = it->x - p1.x;
                double y1Ref;
                if (std::abs(dxSegment) < 1e-9)
                {
                    y1Ref = p1.y;
                }
                else
                {
                    const double ratio = (x2Mapped - p1.x) / dxSegment;
                    const bool canUseSpline = (it - 1 > edges1.begin() && it + 1 < edges1.end());
                    if (canUseSpline) {
                        const Point2d refPt = catmullRomInterpolate(
                            *(it - 2), *(it - 1), *it, *(it + 1), ratio, nullptr);
                        y1Ref = refPt.y;
                    } else {
                        y1Ref = p1.y + ratio * (it->y - p1.y);
                    }
                }
                diffs.push_back(y1Ref - edges2[i].y);
                coordsMapped.push_back(x2Mapped);
            }
        }
        else
        {
            const double y2Mapped = edges2[i].y + offset_primary;
            if (y2Mapped < edges1.front().y || y2Mapped > edges1.back().y)
                continue;

            auto it = std::lower_bound(edges1.begin(), edges1.end(), Point2d(-1e9, y2Mapped),
                                       [](const Point2d& pt, const Point2d& val)
                                       { return pt.y < val.y; });

            if (it != edges1.begin() && it != edges1.end())
            {
                const auto& p1 = *(it - 1);
                const double dySegment = it->y - p1.y;
                double x1Ref;
                if (std::abs(dySegment) < 1e-9)
                {
                    x1Ref = p1.x;
                }
                else
                {
                    const double ratio = (y2Mapped - p1.y) / dySegment;
                    const bool canUseSpline = (it - 1 > edges1.begin() && it + 1 < edges1.end());
                    if (canUseSpline) {
                        const Point2d refPt = catmullRomInterpolate(
                            *(it - 2), *(it - 1), *it, *(it + 1), ratio, nullptr);
                        x1Ref = refPt.x;
                    } else {
                        x1Ref = p1.x + ratio * (it->x - p1.x);
                    }
                }
                diffs.push_back(x1Ref - edges2[i].x);
                coordsMapped.push_back(y2Mapped);
            }
        }
    }

    if (diffs.size() < 10)
        return 1e9;

    const vector<unsigned char> coreMask(coordsMapped.size(), 1);
    double sumCore = 0.0;
    int countCore = 0;
    for (size_t i = 0; i < diffs.size(); ++i) {
        if (!coreMask[i]) {
            continue;
        }
        sumCore += diffs[i];
        ++countCore;
    }
    if (countCore < 10)
        return 1e9;

    const double mean = sumCore / static_cast<double>(countCore);

    vector<double> absDevs;
    absDevs.reserve(countCore);
    vector<double> allAbsDevs(diffs.size(), std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < diffs.size(); ++i) {
        if (!coreMask[i]) {
            continue;
        }
        const double absDev = std::abs(diffs[i] - mean);
        allAbsDevs[i] = absDev;
        absDevs.push_back(absDev);
    }

    vector<double> sortedDevs = absDevs;
    size_t keepIndex = static_cast<size_t>(std::floor(static_cast<double>(sortedDevs.size() - 1) * 0.8));
    keepIndex = std::min(keepIndex, sortedDevs.size() - 1);
    std::nth_element(sortedDevs.begin(), sortedDevs.begin() + keepIndex, sortedDevs.end());
    const double threshold = sortedDevs[keepIndex];

    if (out_inlier_errs)
        out_inlier_errs->clear();
    if (out_inlier_coords)
        out_inlier_coords->clear();

    double validSum = 0.0;
    double validSqSum = 0.0;
    int validCount = 0;

    for (size_t i = 0; i < diffs.size(); ++i)
    {
        if (coreMask[i] && allAbsDevs[i] <= threshold)
        {
            validSum += diffs[i];
            ++validCount;
        }
    }

    if (validCount == 0)
        return 1e9;

    const double refinedMean = validSum / static_cast<double>(validCount);

    for (size_t i = 0; i < diffs.size(); ++i)
    {
        if (coreMask[i] && allAbsDevs[i] <= threshold)
        {
            const double err = diffs[i] - refinedMean;
            validSqSum += err * err;
            if (out_inlier_errs && out_inlier_coords)
            {
                out_inlier_errs->push_back(err);
                out_inlier_coords->push_back(coordsMapped[i]);
            }
        }
    }

    out_mean_perp = refinedMean;
    return validSqSum / static_cast<double>(validCount);
}

void populateAlignmentMetrics(const vector<Point2d>& ref_edges,
                              const vector<Point2d>& rotated_target_edges,
                              TransformResult& res)
{
    res.metrics = {};
    res.inlierErrors.clear();
    res.inlierCoordinates.clear();
    res.samplePrimaryCoordinates.clear();
    res.sampleRefSecondaryCoordinates.clear();
    res.sampleTargetSecondaryCoordinates.clear();
    res.sampleNormalErrors.clear();
    res.sampleTangentErrors.clear();
    res.sampleInlierFlags.clear();

    if (res.score >= 1e8)
        return;

    vector<SampledPair> samples;
    if (!collectAlignmentSamples(ref_edges, rotated_target_edges, res, samples))
        return;

    vector<double> allPrimary;
    vector<double> allNormalErrs;
    vector<double> allTangentErrs;
    vector<double> allRefSecondary;
    vector<double> allTargetSecondary;
    allPrimary.reserve(samples.size());
    allNormalErrs.reserve(samples.size());
    allTangentErrs.reserve(samples.size());
    allRefSecondary.reserve(samples.size());
    allTargetSecondary.reserve(samples.size());

    for (const auto& sample : samples) {
        allPrimary.push_back(sample.primary);
        allNormalErrs.push_back(sample.normalErr);
        allTangentErrs.push_back(sample.tangentErr);
        allRefSecondary.push_back(sample.refSecondary);
        allTargetSecondary.push_back(sample.targetSecondary);
    }

    res.samplePrimaryCoordinates = allPrimary;
    res.sampleRefSecondaryCoordinates = allRefSecondary;
    res.sampleTargetSecondaryCoordinates = allTargetSecondary;
    res.sampleNormalErrors = allNormalErrs;
    res.sampleTangentErrors = allTangentErrs;
    res.sampleInlierFlags.assign(samples.size(), 0);

    const double referenceSpan = computeReferencePrimarySpan(ref_edges, res.axis);
    res.metrics.overlapCount = static_cast<int>(samples.size());
    res.metrics.overlapSpan = computePrimarySpan(allPrimary);
    res.metrics.overlapCoverageRatio = computeCoverageRatio(res.metrics.overlapSpan, referenceSpan);
    res.metrics.normalAll = computeResidualStatistics(allNormalErrs);
    res.metrics.tangentAll = computeResidualStatistics(allTangentErrs);
    res.metrics.tangentCorrAll = computePearsonCorrelation(allRefSecondary, allTargetSecondary);

    const vector<unsigned char> coreMask = computeOverlapCoreMask(allPrimary);
    vector<double> coreNormalErrs;
    vector<double> coreTangentErrs;
    vector<double> corePrimary;
    vector<double> coreRefSecondary;
    vector<double> coreTargetSecondary;
    coreNormalErrs.reserve(samples.size());
    coreTangentErrs.reserve(samples.size());
    corePrimary.reserve(samples.size());
    coreRefSecondary.reserve(samples.size());
    coreTargetSecondary.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!coreMask[i]) {
            continue;
        }
        coreNormalErrs.push_back(allNormalErrs[i]);
        coreTangentErrs.push_back(allTangentErrs[i]);
        corePrimary.push_back(allPrimary[i]);
        coreRefSecondary.push_back(allRefSecondary[i]);
        coreTargetSecondary.push_back(allTargetSecondary[i]);
    }

    vector<double> trimmedNormalErrs;
    vector<double> trimmedTangentErrs;
    vector<double> trimmedPrimary;
    vector<double> trimmedRefSecondary;
    vector<double> trimmedTargetSecondary;
    trimmedNormalErrs.reserve(coreNormalErrs.size());
    trimmedTangentErrs.reserve(coreTangentErrs.size());
    trimmedPrimary.reserve(corePrimary.size());
    trimmedRefSecondary.reserve(coreRefSecondary.size());
    trimmedTargetSecondary.reserve(coreTargetSecondary.size());

    double coreCenter = 0.0;
    double coreThreshold = std::numeric_limits<double>::infinity();
    if (coreNormalErrs.size() >= 3) {
        coreCenter = std::accumulate(coreNormalErrs.begin(), coreNormalErrs.end(), 0.0) /
                     static_cast<double>(coreNormalErrs.size());
        vector<double> coreAbsDevs;
        coreAbsDevs.reserve(coreNormalErrs.size());
        for (double err : coreNormalErrs) {
            coreAbsDevs.push_back(std::abs(err - coreCenter));
        }
        coreThreshold = computeQuantile(coreAbsDevs, 0.8);
    }

    for (size_t i = 0; i < coreNormalErrs.size(); ++i) {
        if (std::abs(coreNormalErrs[i] - coreCenter) > coreThreshold) {
            continue;
        }
        trimmedNormalErrs.push_back(coreNormalErrs[i]);
        trimmedTangentErrs.push_back(coreTangentErrs[i]);
        trimmedPrimary.push_back(corePrimary[i]);
        trimmedRefSecondary.push_back(coreRefSecondary[i]);
        trimmedTargetSecondary.push_back(coreTargetSecondary[i]);
    }

    res.metrics.trimmedOverlapCount = static_cast<int>(trimmedNormalErrs.size());
    res.metrics.trimmedOverlapRatio =
        samples.empty() ? 0.0 : static_cast<double>(trimmedNormalErrs.size()) / static_cast<double>(samples.size());
    res.metrics.trimmedOverlapSpan = computePrimarySpan(trimmedPrimary);
    res.metrics.trimmedOverlapCoverageRatio = computeCoverageRatio(res.metrics.trimmedOverlapSpan, referenceSpan);
    res.metrics.normalTrimmed = computeResidualStatistics(trimmedNormalErrs);
    res.metrics.tangentTrimmed = computeResidualStatistics(trimmedTangentErrs);
    res.metrics.tangentCorrTrimmed = computePearsonCorrelation(trimmedRefSecondary, trimmedTargetSecondary);

    if (samples.size() < 3) {
        res.inlierErrors = allNormalErrs;
        res.inlierCoordinates = allPrimary;
        std::fill(res.sampleInlierFlags.begin(), res.sampleInlierFlags.end(), 1);
        res.metrics.inlierCount = res.metrics.overlapCount;
        res.metrics.inlierRatio = res.metrics.overlapCount > 0 ? 1.0 : 0.0;
        res.metrics.inlierSpan = res.metrics.overlapSpan;
        res.metrics.inlierCoverageRatio = res.metrics.overlapCoverageRatio;
        res.metrics.normalInlier = res.metrics.normalAll;
        res.metrics.tangentInlier = res.metrics.tangentAll;
        res.metrics.tangentCorrInlier = res.metrics.tangentCorrAll;
        return;
    }

    // 这里保留原有“按 80% 分位截断”的内点定义，
    // 让新的论文级评估与现有搜索行为保持兼容。
    const double center =
        std::accumulate(allNormalErrs.begin(), allNormalErrs.end(), 0.0) / static_cast<double>(allNormalErrs.size());

    vector<double> absDevs;
    absDevs.reserve(allNormalErrs.size());
    for (double err : allNormalErrs)
        absDevs.push_back(std::abs(err - center));

    const double threshold = computeQuantile(absDevs, 0.8);

    vector<double> inlierNormalErrs;
    vector<double> inlierTangentErrs;
    vector<double> inlierPrimary;
    vector<double> inlierRefSecondary;
    vector<double> inlierTargetSecondary;
    inlierNormalErrs.reserve(samples.size());
    inlierTangentErrs.reserve(samples.size());
    inlierPrimary.reserve(samples.size());
    inlierRefSecondary.reserve(samples.size());
    inlierTargetSecondary.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (std::abs(allNormalErrs[i] - center) > threshold)
            continue;

        res.sampleInlierFlags[i] = 1;
        inlierNormalErrs.push_back(allNormalErrs[i]);
        inlierTangentErrs.push_back(allTangentErrs[i]);
        inlierPrimary.push_back(samples[i].primary);
        inlierRefSecondary.push_back(samples[i].refSecondary);
        inlierTargetSecondary.push_back(samples[i].targetSecondary);
    }

    res.metrics.inlierCount = static_cast<int>(inlierNormalErrs.size());
    res.metrics.inlierRatio =
        samples.empty() ? 0.0 : static_cast<double>(res.metrics.inlierCount) / static_cast<double>(samples.size());
    res.metrics.inlierSpan = computePrimarySpan(inlierPrimary);
    res.metrics.inlierCoverageRatio = computeCoverageRatio(res.metrics.inlierSpan, referenceSpan);
    if (res.metrics.inlierCount == 0)
        return;

    res.inlierErrors = inlierNormalErrs;
    res.inlierCoordinates = inlierPrimary;
    res.metrics.normalInlier = computeResidualStatistics(inlierNormalErrs);
    res.metrics.tangentInlier = computeResidualStatistics(inlierTangentErrs);
    res.metrics.tangentCorrInlier = computePearsonCorrelation(inlierRefSecondary, inlierTargetSecondary);
}

TransformResult computeGlobalAlignment(const vector<Point2d>& edges1,
                                       const vector<Point2d>& edges2,
                                       double expected_shift,
                                       double search_range,
                                       double max_perp_threshold,
                                       TransformResult::AlignAxis axis,
                                       double tangent_residual_cost_weight,
                                       double tangent_correlation_cost_weight)
{
    TransformResult res;

    const double rangeMin = expected_shift - search_range;
    const double rangeMax = expected_shift + search_range;
    const double coarseStep = 4.0;

    for (double x = rangeMin; x <= rangeMax; x += coarseStep)
    {
        double meanPerp = 0.0;
        vector<double> tmpErrs;
        vector<double> tmpCoords;
        const double normalCost = computeRobustCost(edges1, edges2, x, meanPerp, &tmpErrs, &tmpCoords, axis);

        if (std::abs(meanPerp) > max_perp_threshold)
            continue;

        TransformResult candidate;
        if (axis == TransformResult::AlignAxis::X)
        {
            candidate.dx = x;
            candidate.dy = meanPerp;
        }
        else
        {
            candidate.dx = meanPerp;
            candidate.dy = x;
        }
        candidate.axis = axis;

        const TangentMatchCost tangentCost =
            computeTangentMatchCost(edges1, edges2, candidate,
                                    tangent_residual_cost_weight,
                                    tangent_correlation_cost_weight);
        const double cost = normalCost + tangentCost.total();

        if (cost < res.score)
        {
            res.score = cost;
            res.normalMatchCost = normalCost;
            res.tangentResidualMatchCost = tangentCost.residualCost;
            res.tangentCorrelationMatchCost = tangentCost.correlationCost;
            res.dx = candidate.dx;
            res.dy = candidate.dy;
            res.axis = axis;
            res.inlierErrors = std::move(tmpErrs);
            res.inlierCoordinates = std::move(tmpCoords);
        }
    }

    if (res.score < 1e8)
    {
        const double fineRange = 4.0;
        const double fineStep = 0.1;
        const double startFine = (axis == TransformResult::AlignAxis::X ? res.dx : res.dy) - fineRange;
        const double endFine = (axis == TransformResult::AlignAxis::X ? res.dx : res.dy) + fineRange;

        for (double x = startFine; x <= endFine; x += fineStep)
        {
            double meanPerp = 0.0;
            vector<double> tmpErrs;
            vector<double> tmpCoords;
            const double normalCost = computeRobustCost(edges1, edges2, x, meanPerp, &tmpErrs, &tmpCoords, axis);

            if (std::abs(meanPerp) > max_perp_threshold)
                continue;

            TransformResult candidate;
            if (axis == TransformResult::AlignAxis::X)
            {
                candidate.dx = x;
                candidate.dy = meanPerp;
            }
            else
            {
                candidate.dx = meanPerp;
                candidate.dy = x;
            }
            candidate.axis = axis;

            const TangentMatchCost tangentCost =
                computeTangentMatchCost(edges1, edges2, candidate,
                                        tangent_residual_cost_weight,
                                        tangent_correlation_cost_weight);
            const double cost = normalCost + tangentCost.total();

            if (cost < res.score)
            {
                res.score = cost;
                res.normalMatchCost = normalCost;
                res.tangentResidualMatchCost = tangentCost.residualCost;
                res.tangentCorrelationMatchCost = tangentCost.correlationCost;
                res.dx = candidate.dx;
                res.dy = candidate.dy;
                res.axis = axis;
                res.inlierErrors = std::move(tmpErrs);
                res.inlierCoordinates = std::move(tmpCoords);
            }
        }

        // --- 抛物线亚像素精化 ---
        // 在精细网格搜索(fineStep=0.1 px)后，对最优候选做三点抛物线插值
        // 使平移精度从 ~0.1 px 提升至 ~0.01 px
        {
            const double xBest = (axis == TransformResult::AlignAxis::X) ? res.dx : res.dy;
            double meanPerpLeft = 0.0;
            double meanPerpRight = 0.0;

            const double normalCostLeft =
                computeRobustCost(edges1, edges2, xBest - fineStep, meanPerpLeft, nullptr, nullptr, axis);
            const double normalCostRight =
                computeRobustCost(edges1, edges2, xBest + fineStep, meanPerpRight, nullptr, nullptr, axis);

            const bool leftValid = normalCostLeft < 1e8 && std::abs(meanPerpLeft) <= max_perp_threshold;
            const bool rightValid = normalCostRight < 1e8 && std::abs(meanPerpRight) <= max_perp_threshold;

            if (leftValid && rightValid) {
                const double denom = 2.0 * (normalCostLeft + normalCostRight - 2.0 * res.normalMatchCost);
                if (denom > 1e-12) {
                    const double delta = fineStep * (normalCostRight - normalCostLeft) / denom;
                    const double deltaClamped = std::clamp(delta, -fineStep, fineStep);
                    const double xRefined = xBest - deltaClamped;

                    if (std::abs(xRefined - xBest) > 1e-9) {
                        double meanPerpRefined = 0.0;
                        vector<double> refinedErrs;
                        vector<double> refinedCoords;
                        const double normalCostRefined =
                            computeRobustCost(edges1, edges2, xRefined, meanPerpRefined,
                                              &refinedErrs, &refinedCoords, axis);

                        if (normalCostRefined < res.normalMatchCost &&
                            std::abs(meanPerpRefined) <= max_perp_threshold) {
                            TransformResult refined;
                            if (axis == TransformResult::AlignAxis::X) {
                                refined.dx = xRefined;
                                refined.dy = meanPerpRefined;
                            } else {
                                refined.dx = meanPerpRefined;
                                refined.dy = xRefined;
                            }
                            refined.axis = axis;

                            const TangentMatchCost refinedTangent =
                                computeTangentMatchCost(edges1, edges2, refined,
                                                        tangent_residual_cost_weight,
                                                        tangent_correlation_cost_weight);
                            const double refinedCost = normalCostRefined + refinedTangent.total();

                            if (refinedCost < res.score) {
                                res.score = refinedCost;
                                res.normalMatchCost = normalCostRefined;
                                res.tangentResidualMatchCost = refinedTangent.residualCost;
                                res.tangentCorrelationMatchCost = refinedTangent.correlationCost;
                                res.dx = refined.dx;
                                res.dy = refined.dy;
                                res.inlierErrors = std::move(refinedErrs);
                                res.inlierCoordinates = std::move(refinedCoords);
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}

TransformResult matchOnePair(const EdgeVariants& prev_edges,
                             const EdgeVariants& next_edges,
                             const Point2d& center,
                             double approx_shift,
                             double approx_shift_y,
                             double approx_angle_deg,
                             bool has_reliable_motion_prior,
                             double base_search_range,
                             MotionPriorDirection direction_constraint,
                             double rotation_search_min_deg,
                             double rotation_search_max_deg,
                             double rotation_search_step_deg,
                             double tangent_residual_cost_weight,
                             double tangent_correlation_cost_weight,
                             bool strict_local_prior_window,
                             double local_primary_search_half_range_px,
                             double local_perp_search_half_range_px,
                             double& search_range_x,
                             double& search_range_y)
{
    TransformResult res;
    res.score = 1e9;
    res.direction = "N/A";

    const double approx_x_local = std::abs(approx_shift);
    const double approx_y_local = std::abs(approx_shift_y);
    const bool hasFixedDirectionPrior = direction_constraint != MotionPriorDirection::Auto;
    const bool fixedDirectionUsesPrimaryX =
        direction_constraint == MotionPriorDirection::XPositive ||
        direction_constraint == MotionPriorDirection::XNegative;
    if (strict_local_prior_window) {
        search_range_x = std::max(1.0, local_primary_search_half_range_px);
        search_range_y = std::max(1.0, local_perp_search_half_range_px);
    } else if (hasFixedDirectionPrior) {
        // 对单向采集的工件，首轮围绕先验位移做有限搜索即可，
        // 避免把 expected shift 误放大成近两倍跨度的大范围暴力扫描。
        const double fixedPrimarySearchRange = std::max(1.0, base_search_range);
        const double fixedPerpSearchRange = std::clamp(base_search_range * 0.30, 12.0, 60.0);
        search_range_x = fixedDirectionUsesPrimaryX ? fixedPrimarySearchRange : fixedPerpSearchRange;
        search_range_y = fixedDirectionUsesPrimaryX ? fixedPerpSearchRange : fixedPrimarySearchRange;
    } else {
        search_range_x = std::max(base_search_range, approx_x_local * 2.0);
        search_range_y = std::max(base_search_range, approx_y_local * 2.0);
    }

    const auto directionAllowed = [&](MotionPriorDirection candidate) {
        return direction_constraint == MotionPriorDirection::Auto || direction_constraint == candidate;
    };

    bool nearStraightSegment = false;

    vector<AlignmentCandidateDiagnostic> candidateDiagnostics;
    const auto captureCandidate = [&](const char* label, const TransformResult& cand)
    {
        if (!cand.hasCandidate()) {
            return;
        }

        AlignmentCandidateDiagnostic diagnostic;
        diagnostic.direction = label;
        diagnostic.dx = cand.dx;
        diagnostic.dy = cand.dy;
        diagnostic.da = cand.da;
        diagnostic.score = cand.score;
        diagnostic.normalMatchCost = cand.normalMatchCost;
        diagnostic.tangentResidualMatchCost = cand.tangentResidualMatchCost;
        diagnostic.tangentCorrelationMatchCost = cand.tangentCorrelationMatchCost;
        diagnostic.directionPenaltyMatchCost = cand.directionPenaltyMatchCost;
        diagnostic.axis = cand.axis;
        diagnostic.metrics = cand.metrics;
        candidateDiagnostics.push_back(std::move(diagnostic));
    };

    double rotationMin = rotation_search_min_deg;
    double rotationMax = rotation_search_max_deg;
    if (!std::isfinite(rotationMin) || !std::isfinite(rotationMax)) {
        rotationMin = -0.5;
        rotationMax = 0.5;
    }
    if (rotationMin > rotationMax) {
        std::swap(rotationMin, rotationMax);
    }

    double rotationStep = rotation_search_step_deg;
    if (!std::isfinite(rotationStep) || rotationStep <= 0.0) {
        rotationStep = 0.01;
    }

    auto apply_direction_penalty = [&](TransformResult& cand)
    {
        double penalty = 0.0;
        if (cand.axis == TransformResult::AlignAxis::X)
        {
            const double expected_sign = (approx_shift >= 0) ? 1.0 : -1.0;
            if (cand.dx * expected_sign < 0)
                penalty += 0.1;
        }
        else
        {
            const double expected_sign = (approx_shift_y >= 0) ? 1.0 : -1.0;
            if (cand.dy * expected_sign < 0)
                penalty += 0.1;
        }
        cand.directionPenaltyMatchCost += penalty;
        cand.score += penalty;
    };

    auto apply_motion_continuity_penalty = [&](TransformResult& cand)
    {
        double penaltyWeight = 0.02;
        double primaryTolerancePx = std::max(40.0, base_search_range * 0.4);
        double perpTolerancePx = std::max(8.0, base_search_range * 0.075);

        if (strict_local_prior_window) {
            penaltyWeight = has_reliable_motion_prior ? 0.08 : 0.06;
            primaryTolerancePx = std::clamp(local_primary_search_half_range_px * 0.25, 3.5, 10.0);
            perpTolerancePx = std::clamp(local_perp_search_half_range_px * 0.45, 2.0, 6.0);
        } else if (has_reliable_motion_prior || hasFixedDirectionPrior) {
            penaltyWeight = has_reliable_motion_prior ? 0.05 : 0.035;
            primaryTolerancePx = std::clamp(base_search_range * 0.06, 8.0, 18.0);
            perpTolerancePx = std::clamp(base_search_range * 0.03, 3.0, 8.0);
        }

        const auto addSoftPenalty = [&](const double deltaPx, const double tolerancePx) {
            const double excessPx = std::max(0.0, std::abs(deltaPx) - tolerancePx);
            if (excessPx <= 0.0) {
                return 0.0;
            }
            const double normalized = excessPx / tolerancePx;
            return penaltyWeight * normalized * normalized;
        };

        double penalty = 0.0;
        if (cand.axis == TransformResult::AlignAxis::X) {
            penalty += addSoftPenalty(cand.dx - approx_shift, primaryTolerancePx);
            penalty += addSoftPenalty(cand.dy - approx_shift_y, perpTolerancePx);
        } else {
            penalty += addSoftPenalty(cand.dy - approx_shift_y, primaryTolerancePx);
            penalty += addSoftPenalty(cand.dx - approx_shift, perpTolerancePx);
        }

        cand.directionPenaltyMatchCost += penalty;
        cand.score += penalty;
    };

    const auto violates_motion_continuity = [&](const TransformResult& cand, double currentAngleDeg) {
        (void)cand;
        (void)currentAngleDeg;
        return false;
    };

    const auto resolveMaxPerpThreshold = [&](TransformResult::AlignAxis axis) {
        if (strict_local_prior_window) {
            return std::max(15.0, local_perp_search_half_range_px);
        }
        const double expectedPerpMagnitude =
            axis == TransformResult::AlignAxis::X ? std::abs(approx_shift_y) : std::abs(approx_shift);
        return std::clamp(expectedPerpMagnitude + 12.0, 15.0, 30.0);
    };

    // 自适应旋转搜索：对近似直线的轮廓段跳过旋转搜索
    {
        double maxCurvature = 0.0;
        for (const EdgeVariants* ev : {&prev_edges, &next_edges}) {
            const std::size_t step = std::max<std::size_t>(1, ev->raw.size() / 50);
            for (std::size_t i = step + 1; i + step < ev->raw.size(); i += step) {
                const double c = computeLocalCurvature(ev->raw, static_cast<int>(i), 5);
                if (std::isfinite(c)) {
                    maxCurvature = std::max(maxCurvature, c);
                }
            }
        }
        if (maxCurvature < 0.001) {
            nearStraightSegment = true;
            rotationMin = 0.0;
            rotationMax = 0.0;
            rotationStep = 1.0; // 单次迭代，角度=0°
        }
    }

    if (nearStraightSegment) {
        const double straightSegmentHalfWindowDeg = 0.03;
        rotationMin = std::max(rotation_search_min_deg, approx_angle_deg - straightSegmentHalfWindowDeg);
        rotationMax = std::min(rotation_search_max_deg, approx_angle_deg + straightSegmentHalfWindowDeg);
        if (rotationMin > rotationMax) {
            rotationMin = approx_angle_deg;
            rotationMax = approx_angle_deg;
        }
        const double baseRotationStep = rotation_search_step_deg > 0.0 ? rotation_search_step_deg : 0.01;
        rotationStep = std::min(baseRotationStep, 0.005);
    }

    for (double ang = rotationMin; ang <= rotationMax + 1e-12; ang += rotationStep)
    {
        vector<Point2d> rotated_edges2;
        rotatePoints(next_edges.raw, rotated_edges2, ang, center);
        sortContourByX(rotated_edges2);

        vector<Point2d> rotated_edges2_y;
        rotatePoints(next_edges.raw, rotated_edges2_y, ang, center);
        sortContourByY(rotated_edges2_y);

        const Point2d centerNegX(-center.x, center.y);
        vector<Point2d> rotated_edges2_negX;
        rotatePoints(next_edges.negX_sorted, rotated_edges2_negX, ang, centerNegX);
        sortContourByX(rotated_edges2_negX);

        const Point2d centerNegY(center.x, -center.y);
        vector<Point2d> rotated_edges2_negY;
        rotatePoints(next_edges.negY_sorted, rotated_edges2_negY, ang, centerNegY);
        sortContourByY(rotated_edges2_negY);

        TransformResult curBest;
        curBest.score = 1e9;
        string winnerLabel = "N/A";

        if (directionAllowed(MotionPriorDirection::XPositive)) {
            const double maxPerpThreshold = resolveMaxPerpThreshold(TransformResult::AlignAxis::X);
            TransformResult cur = computeGlobalAlignment(prev_edges.x_sorted, rotated_edges2,
                                                         approx_x_local, search_range_x, maxPerpThreshold,
                                                         TransformResult::AlignAxis::X,
                                                         tangent_residual_cost_weight,
                                                         tangent_correlation_cost_weight);
            cur.da = ang;
            if (violates_motion_continuity(cur, ang)) {
                cur.score = 1e9;
            }
            populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, cur);
            apply_direction_penalty(cur);
            apply_motion_continuity_penalty(cur);
            captureCandidate("X+", cur);
            if (cur.score < curBest.score)
            {
                curBest = cur;
                winnerLabel = "X+";
            }
        }

        if (directionAllowed(MotionPriorDirection::XNegative)) {
            const double maxPerpThreshold = resolveMaxPerpThreshold(TransformResult::AlignAxis::X);
            TransformResult curXrev = computeGlobalAlignment(prev_edges.negX_sorted, rotated_edges2_negX,
                                                             -approx_x_local, search_range_x, maxPerpThreshold,
                                                             TransformResult::AlignAxis::X,
                                                             tangent_residual_cost_weight,
                                                             tangent_correlation_cost_weight);
            curXrev.dx = -curXrev.dx;
            curXrev.da = ang;
            if (violates_motion_continuity(curXrev, ang)) {
                curXrev.score = 1e9;
            }
            populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, curXrev);
            apply_direction_penalty(curXrev);
            apply_motion_continuity_penalty(curXrev);
            captureCandidate("X-", curXrev);
            if (curXrev.score < curBest.score)
            {
                curBest = curXrev;
                winnerLabel = "X-";
            }
        }

        if (directionAllowed(MotionPriorDirection::YPositive)) {
            const double maxPerpThreshold = resolveMaxPerpThreshold(TransformResult::AlignAxis::Y);
            TransformResult curV = computeGlobalAlignment(prev_edges.y_sorted, rotated_edges2_y,
                                                          approx_y_local, search_range_y, maxPerpThreshold,
                                                          TransformResult::AlignAxis::Y,
                                                          tangent_residual_cost_weight,
                                                          tangent_correlation_cost_weight);
            curV.da = ang;
            if (violates_motion_continuity(curV, ang)) {
                curV.score = 1e9;
            }
            populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, curV);
            apply_direction_penalty(curV);
            apply_motion_continuity_penalty(curV);
            captureCandidate("Y+", curV);
            if (curV.score < curBest.score)
            {
                curBest = curV;
                winnerLabel = "Y+";
            }
        }

        if (directionAllowed(MotionPriorDirection::YNegative)) {
            const double maxPerpThreshold = resolveMaxPerpThreshold(TransformResult::AlignAxis::Y);
            TransformResult curYrev = computeGlobalAlignment(prev_edges.negY_sorted, rotated_edges2_negY,
                                                             -approx_y_local, search_range_y, maxPerpThreshold,
                                                             TransformResult::AlignAxis::Y,
                                                             tangent_residual_cost_weight,
                                                             tangent_correlation_cost_weight);
            curYrev.dy = -curYrev.dy;
            curYrev.da = ang;
            if (violates_motion_continuity(curYrev, ang)) {
                curYrev.score = 1e9;
            }
            populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, curYrev);
            apply_direction_penalty(curYrev);
            apply_motion_continuity_penalty(curYrev);
            captureCandidate("Y-", curYrev);
            if (curYrev.score < curBest.score)
            {
                curBest = curYrev;
                winnerLabel = "Y-";
            }
        }

        if (curBest.score < res.score)
        {
            res = curBest;
            res.direction = winnerLabel;
        }
    }

    if (false && res.score >= 1e8 && has_reliable_motion_prior) {
        const double fallbackSearchRangeX = std::clamp(base_search_range * 0.30, 45.0, 90.0);
        const double fallbackSearchRangeY = std::clamp(base_search_range * 0.18, 20.0, 50.0);
        const double fallbackRotationMin = std::max(rotationMin, approx_angle_deg - 0.01);
        const double fallbackRotationMax = std::min(rotationMax, approx_angle_deg + 0.01);
        const double fallbackRotationStep = std::min(rotationStep, 0.005);

        for (double ang = fallbackRotationMin; ang <= fallbackRotationMax + 1e-12; ang += fallbackRotationStep)
        {
            vector<Point2d> rotated_edges2;
            rotatePoints(next_edges.raw, rotated_edges2, ang, center);
            sortContourByX(rotated_edges2);

            vector<Point2d> rotated_edges2_y;
            rotatePoints(next_edges.raw, rotated_edges2_y, ang, center);
            sortContourByY(rotated_edges2_y);

            const Point2d centerNegX(-center.x, center.y);
            vector<Point2d> rotated_edges2_negX;
            rotatePoints(next_edges.negX_sorted, rotated_edges2_negX, ang, centerNegX);
            sortContourByX(rotated_edges2_negX);

            const Point2d centerNegY(center.x, -center.y);
            vector<Point2d> rotated_edges2_negY;
            rotatePoints(next_edges.negY_sorted, rotated_edges2_negY, ang, centerNegY);
            sortContourByY(rotated_edges2_negY);

            TransformResult curBest;
            curBest.score = 1e9;
            string winnerLabel = "N/A";

            if (directionAllowed(MotionPriorDirection::XPositive)) {
                const double maxPerpThreshold = std::clamp(std::abs(approx_shift_y) + 24.0, 28.0, 42.0);
                TransformResult cur = computeGlobalAlignment(prev_edges.x_sorted, rotated_edges2,
                                                             approx_x_local, fallbackSearchRangeX, maxPerpThreshold,
                                                             TransformResult::AlignAxis::X,
                                                             tangent_residual_cost_weight,
                                                             tangent_correlation_cost_weight);
                cur.da = ang;
                populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, cur);
                apply_direction_penalty(cur);
                apply_motion_continuity_penalty(cur);
                captureCandidate("X+", cur);
                if (cur.score < curBest.score)
                {
                    curBest = cur;
                    winnerLabel = "X+";
                }
            }

            if (directionAllowed(MotionPriorDirection::XNegative)) {
                const double maxPerpThreshold = std::clamp(std::abs(approx_shift_y) + 24.0, 28.0, 42.0);
                TransformResult curXrev = computeGlobalAlignment(prev_edges.negX_sorted, rotated_edges2_negX,
                                                                 -approx_x_local, fallbackSearchRangeX, maxPerpThreshold,
                                                                 TransformResult::AlignAxis::X,
                                                                 tangent_residual_cost_weight,
                                                                 tangent_correlation_cost_weight);
                curXrev.dx = -curXrev.dx;
                curXrev.da = ang;
                populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, curXrev);
                apply_direction_penalty(curXrev);
                apply_motion_continuity_penalty(curXrev);
                captureCandidate("X-", curXrev);
                if (curXrev.score < curBest.score)
                {
                    curBest = curXrev;
                    winnerLabel = "X-";
                }
            }

            if (directionAllowed(MotionPriorDirection::YPositive)) {
                const double maxPerpThreshold = std::clamp(std::abs(approx_shift) + 24.0, 28.0, 42.0);
                TransformResult curV = computeGlobalAlignment(prev_edges.y_sorted, rotated_edges2_y,
                                                              approx_y_local, fallbackSearchRangeY, maxPerpThreshold,
                                                              TransformResult::AlignAxis::Y,
                                                              tangent_residual_cost_weight,
                                                              tangent_correlation_cost_weight);
                curV.da = ang;
                populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, curV);
                apply_direction_penalty(curV);
                apply_motion_continuity_penalty(curV);
                captureCandidate("Y+", curV);
                if (curV.score < curBest.score)
                {
                    curBest = curV;
                    winnerLabel = "Y+";
                }
            }

            if (directionAllowed(MotionPriorDirection::YNegative)) {
                const double maxPerpThreshold = std::clamp(std::abs(approx_shift) + 24.0, 28.0, 42.0);
                TransformResult curYrev = computeGlobalAlignment(prev_edges.negY_sorted, rotated_edges2_negY,
                                                                 -approx_y_local, fallbackSearchRangeY, maxPerpThreshold,
                                                                 TransformResult::AlignAxis::Y,
                                                                 tangent_residual_cost_weight,
                                                                 tangent_correlation_cost_weight);
                curYrev.dy = -curYrev.dy;
                curYrev.da = ang;
                populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, curYrev);
                apply_direction_penalty(curYrev);
                apply_motion_continuity_penalty(curYrev);
                captureCandidate("Y-", curYrev);
                if (curYrev.score < curBest.score)
                {
                    curBest = curYrev;
                    winnerLabel = "Y-";
                }
            }

            if (curBest.score < res.score)
            {
                res = curBest;
                res.direction = winnerLabel;
                search_range_x = fallbackSearchRangeX;
                search_range_y = fallbackSearchRangeY;
            }
        }
    }

    if (res.hasCandidate() && !candidateDiagnostics.empty()) {
        const double currentNormal = preferredNormalRmse(res.metrics);
        const double currentCoverage = preferredCoverageRatio(res.metrics);
        const double currentCorrelation = preferredTangentCorrelation(res.metrics);
        const double currentSelectionCost = transformSelectionCost(res, approx_shift, approx_shift_y);
        const bool loosePromotionMode = currentNormal > 0.35 || res.score > 1.0;
        const double scoreWindow =
            loosePromotionMode ? std::numeric_limits<double>::infinity() : (res.score + 0.15);
        const double maxPrimaryJumpPx = loosePromotionMode ? 320.0 : (has_reliable_motion_prior ? 42.0 : 90.0);
        const double maxPerpJumpPx = loosePromotionMode ? 60.0 : 24.0;
        const double currentPrimaryShift = res.axis == AlignmentAxis::X ? res.dx : res.dy;
        const double currentPerpShift = res.axis == AlignmentAxis::X ? res.dy : res.dx;

        const AlignmentCandidateDiagnostic* promotedCandidate = nullptr;
        double bestNormal = currentNormal;
        double bestCost = currentSelectionCost;
        for (const AlignmentCandidateDiagnostic& candidate : candidateDiagnostics) {
            if (candidate.axis != res.axis) {
                continue;
            }

            const double normalRmse = preferredNormalRmse(candidate.metrics);
            if (!std::isfinite(normalRmse)) {
                continue;
            }

            const double coverage = preferredCoverageRatio(candidate.metrics);
            if (coverage + 0.04 < currentCoverage) {
                continue;
            }

            const double tangentCorrelation = preferredTangentCorrelation(candidate.metrics);
            if (tangentCorrelation + 0.015 < currentCorrelation) {
                continue;
            }

            if (std::isfinite(scoreWindow) && candidate.score > scoreWindow) {
                continue;
            }

            const double candidatePrimaryShift = candidate.axis == AlignmentAxis::X ? candidate.dx : candidate.dy;
            const double candidatePerpShift = candidate.axis == AlignmentAxis::X ? candidate.dy : candidate.dx;
            if (std::abs(candidatePrimaryShift - currentPrimaryShift) > maxPrimaryJumpPx ||
                std::abs(candidatePerpShift - currentPerpShift) > maxPerpJumpPx) {
                continue;
            }

            const double cost = diagnosticSelectionCost(candidate, approx_shift, approx_shift_y);
            const bool betterNormal = normalRmse < bestNormal - 0.005;
            const bool sameNormalButBetterCost =
                std::abs(normalRmse - bestNormal) <= 0.005 && cost < bestCost;
            if (betterNormal || sameNormalButBetterCost) {
                promotedCandidate = &candidate;
                bestNormal = normalRmse;
                bestCost = cost;
            }
        }

        const bool shouldPromoteCandidate =
            promotedCandidate != nullptr &&
            (bestNormal + 0.015 < currentNormal ||
             bestCost < currentSelectionCost * 0.92 ||
             (loosePromotionMode && bestNormal < currentNormal * 0.8));

        if (shouldPromoteCandidate) {
            const auto rebuildCandidateTransform = [&](const AlignmentCandidateDiagnostic& candidate) {
                TransformResult rebuilt;
                rebuilt.score = 1e9;
                const double refineSearchHalfRangePx = 6.0;

                vector<Point2d> rotated_edges2;
                rotatePoints(next_edges.raw, rotated_edges2, candidate.da, center);
                sortContourByX(rotated_edges2);

                vector<Point2d> rotated_edges2_y = rotated_edges2;
                sortContourByY(rotated_edges2_y);

                const Point2d centerNegX(-center.x, center.y);
                vector<Point2d> rotated_edges2_negX;
                rotatePoints(next_edges.negX_sorted, rotated_edges2_negX, candidate.da, centerNegX);
                sortContourByX(rotated_edges2_negX);

                const Point2d centerNegY(center.x, -center.y);
                vector<Point2d> rotated_edges2_negY;
                rotatePoints(next_edges.negY_sorted, rotated_edges2_negY, candidate.da, centerNegY);
                sortContourByY(rotated_edges2_negY);

                if (candidate.direction == "X+") {
                    const double maxPerpThreshold =
                        std::max(resolveMaxPerpThreshold(AlignmentAxis::X), std::abs(candidate.dy) + 6.0);
                    rebuilt = computeGlobalAlignment(prev_edges.x_sorted,
                                                     rotated_edges2,
                                                     candidate.dx,
                                                     refineSearchHalfRangePx,
                                                     maxPerpThreshold,
                                                     TransformResult::AlignAxis::X,
                                                     tangent_residual_cost_weight,
                                                     tangent_correlation_cost_weight);
                    rebuilt.da = candidate.da;
                    populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, rebuilt);
                } else if (candidate.direction == "X-") {
                    const double maxPerpThreshold =
                        std::max(resolveMaxPerpThreshold(AlignmentAxis::X), std::abs(candidate.dy) + 6.0);
                    rebuilt = computeGlobalAlignment(prev_edges.negX_sorted,
                                                     rotated_edges2_negX,
                                                     -candidate.dx,
                                                     refineSearchHalfRangePx,
                                                     maxPerpThreshold,
                                                     TransformResult::AlignAxis::X,
                                                     tangent_residual_cost_weight,
                                                     tangent_correlation_cost_weight);
                    rebuilt.dx = -rebuilt.dx;
                    rebuilt.da = candidate.da;
                    populateAlignmentMetrics(prev_edges.x_sorted, rotated_edges2, rebuilt);
                } else if (candidate.direction == "Y+") {
                    const double maxPerpThreshold =
                        std::max(resolveMaxPerpThreshold(AlignmentAxis::Y), std::abs(candidate.dx) + 6.0);
                    rebuilt = computeGlobalAlignment(prev_edges.y_sorted,
                                                     rotated_edges2_y,
                                                     candidate.dy,
                                                     refineSearchHalfRangePx,
                                                     maxPerpThreshold,
                                                     TransformResult::AlignAxis::Y,
                                                     tangent_residual_cost_weight,
                                                     tangent_correlation_cost_weight);
                    rebuilt.da = candidate.da;
                    populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, rebuilt);
                } else if (candidate.direction == "Y-") {
                    const double maxPerpThreshold =
                        std::max(resolveMaxPerpThreshold(AlignmentAxis::Y), std::abs(candidate.dx) + 6.0);
                    rebuilt = computeGlobalAlignment(prev_edges.negY_sorted,
                                                     rotated_edges2_negY,
                                                     -candidate.dy,
                                                     refineSearchHalfRangePx,
                                                     maxPerpThreshold,
                                                     TransformResult::AlignAxis::Y,
                                                     tangent_residual_cost_weight,
                                                     tangent_correlation_cost_weight);
                    rebuilt.dy = -rebuilt.dy;
                    rebuilt.da = candidate.da;
                    populateAlignmentMetrics(prev_edges.y_sorted, rotated_edges2_y, rebuilt);
                }

                if (rebuilt.hasCandidate()) {
                    apply_direction_penalty(rebuilt);
                    apply_motion_continuity_penalty(rebuilt);
                    rebuilt.direction = candidate.direction;
                }

                return rebuilt;
            };

            TransformResult rebuiltCandidate = rebuildCandidateTransform(*promotedCandidate);
            const double rebuiltNormal = preferredNormalRmse(rebuiltCandidate.metrics);
            const double rebuiltCoverage = preferredCoverageRatio(rebuiltCandidate.metrics);
            const double rebuiltCost = transformSelectionCost(rebuiltCandidate, approx_shift, approx_shift_y);
            if (rebuiltCandidate.hasCandidate() &&
                std::isfinite(rebuiltNormal) &&
                rebuiltCoverage + 0.04 >= currentCoverage &&
                (rebuiltNormal + 0.015 < currentNormal ||
                 rebuiltCost < currentSelectionCost * 0.92)) {
                res = std::move(rebuiltCandidate);
            }
        }
    }

    std::sort(candidateDiagnostics.begin(), candidateDiagnostics.end(),
              [](const AlignmentCandidateDiagnostic& lhs, const AlignmentCandidateDiagnostic& rhs)
              {
                  return lhs.score < rhs.score;
              });
    if (candidateDiagnostics.size() > 20) {
        candidateDiagnostics.resize(20);
    }
    res.candidateDiagnostics = std::move(candidateDiagnostics);

    return res;
}

} // namespace stitch
