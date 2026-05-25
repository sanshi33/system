#include "gui/StepTableModel.h"

#include <QBrush>
#include <QColor>
#include <QLocale>

namespace pinjie::gui {

namespace {

const stitch::ResidualStatistics& normalDisplayStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
}

const stitch::ResidualStatistics& tangentDisplayStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
}

double tangentDisplayCorr(const stitch::AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;
}

bool isRiskStep(const stitch::AlignmentMetrics& metrics)
{
    const auto& normalStats = normalDisplayStats(metrics);
    const double tangentCorr = tangentDisplayCorr(metrics);
    return (normalStats.valid() && normalStats.rmse > 1.0) || tangentCorr < 0.95 || metrics.inlierRatio < 0.7;
}

QString formatDouble(double value, int decimals = 3)
{
    return QLocale().toString(value, 'f', decimals);
}

} // namespace

StepTableModel::StepTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int StepTableModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : steps_.size();
}

int StepTableModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : ColumnCount;
}

QVariant StepTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= steps_.size()) {
        return {};
    }

    const auto& step = steps_.at(index.row());
    const auto& metrics = step.transform.metrics;

    if (role == Qt::DisplayRole) {
        switch (index.column()) {
        case StepColumn:
            return static_cast<int>(step.stepIndex);
        case ImageAColumn:
            return static_cast<int>(step.referenceImageIndex + 1);
        case ImageBColumn:
            return static_cast<int>(step.targetImageIndex + 1);
        case DxColumn:
            return formatDouble(step.transform.dx, 2);
        case DyColumn:
            return formatDouble(step.transform.dy, 2);
        case AngleColumn:
            return formatDouble(step.transform.da, 3);
        case NormalRmseColumn:
            return formatDouble(normalDisplayStats(metrics).rmse, 4);
        case TangentRmseColumn:
            return formatDouble(tangentDisplayStats(metrics).rmse, 4);
        case TangentCorrColumn:
            return formatDouble(tangentDisplayCorr(metrics), 4);
        case CoverageColumn:
            return QStringLiteral("%1 / %2")
                .arg(formatDouble(metrics.overlapCoverageRatio, 3),
                     formatDouble(metrics.inlierCoverageRatio, 3));
        case InlierColumn:
            return QStringLiteral("%1 / %2").arg(metrics.inlierCount).arg(metrics.overlapCount);
        default:
            return {};
        }
    }

    if (role == Qt::ForegroundRole && isRiskStep(metrics)) {
        return QBrush(QColor(182, 62, 55));
    }

    if (role == Qt::ToolTipRole) {
        return QStringLiteral(
                   "方向=%1\n搜索窗口 sx=%2 px，sy=%3 px\nscore=%4\n法向代价=%5，切向 RMSE 代价=%6，切向相关代价=%7，方向惩罚=%8")
            .arg(QString::fromUtf8(step.transform.direction.data(), static_cast<int>(step.transform.direction.size())),
                 formatDouble(step.searchRangeX, 1),
                 formatDouble(step.searchRangeY, 1),
                 formatDouble(step.transform.score, 4),
                 formatDouble(step.transform.normalMatchCost, 4),
                 formatDouble(step.transform.tangentResidualMatchCost, 4),
                 formatDouble(step.transform.tangentCorrelationMatchCost, 4),
                 formatDouble(step.transform.directionPenaltyMatchCost, 4));
    }

    return {};
}

QVariant StepTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole) {
        return QAbstractTableModel::headerData(section, orientation, role);
    }

    switch (section) {
    case StepColumn:
        return QStringLiteral("步号");
    case ImageAColumn:
        return QStringLiteral("参考图");
    case ImageBColumn:
        return QStringLiteral("目标图");
    case DxColumn:
        return QStringLiteral("dx");
    case DyColumn:
        return QStringLiteral("dy");
    case AngleColumn:
        return QStringLiteral("角度");
    case NormalRmseColumn:
        return QStringLiteral("法向 RMSE");
    case TangentRmseColumn:
        return QStringLiteral("切向 RMSE");
    case TangentCorrColumn:
        return QStringLiteral("切向相关");
    case CoverageColumn:
        return QStringLiteral("重叠/内点覆盖");
    case InlierColumn:
        return QStringLiteral("内点/重叠点");
    default:
        return {};
    }
}

void StepTableModel::clear()
{
    beginResetModel();
    steps_.clear();
    endResetModel();
}

void StepTableModel::appendStep(const pinjie::StitchStepRecord& step)
{
    const int row = steps_.size();
    beginInsertRows(QModelIndex(), row, row);
    steps_.push_back(step);
    endInsertRows();
}

const pinjie::StitchStepRecord* StepTableModel::stepAt(int row) const
{
    if (row < 0 || row >= steps_.size()) {
        return nullptr;
    }
    return &steps_.at(row);
}

} // namespace pinjie::gui
