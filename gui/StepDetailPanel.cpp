#include "gui/StepDetailPanel.h"

#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLocale>
#include <QVBoxLayout>

namespace pinjie::gui {

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

bool isRiskStep(const stitch::AlignmentMetrics& metrics)
{
    const auto& normalStats = displayNormalStats(metrics);
    const double tangentCorr = displayTangentCorr(metrics);
    return (normalStats.valid() && normalStats.rmse > 1.0) || tangentCorr < 0.95 || metrics.inlierRatio < 0.7;
}

QString formatDouble(double value, int decimals = 4)
{
    return QLocale().toString(value, 'f', decimals);
}

QLabel* makeValueLabel(QWidget* parent)
{
    auto* label = new QLabel(parent);
    label->setObjectName(QStringLiteral("stepMetricValue"));
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setWordWrap(true);
    return label;
}

QWidget* makeMetricCard(const QString& title, QLabel*& valueLabel, QWidget* parent)
{
    auto* card = new QFrame(parent);
    card->setObjectName(QStringLiteral("stepMetricCard"));

    auto* layout = new QVBoxLayout(card);
    layout->setContentsMargins(12, 10, 12, 10);
    layout->setSpacing(4);

    auto* titleLabel = new QLabel(title, card);
    titleLabel->setObjectName(QStringLiteral("stepMetricTitle"));

    valueLabel = makeValueLabel(card);

    layout->addWidget(titleLabel);
    layout->addWidget(valueLabel);
    layout->addStretch(1);
    return card;
}

QGridLayout* makeTwoColumnGrid(QWidget* parent)
{
    auto* layout = new QGridLayout(parent);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->setHorizontalSpacing(10);
    layout->setVerticalSpacing(10);
    layout->setColumnStretch(0, 1);
    layout->setColumnStretch(1, 1);
    return layout;
}

} // namespace

StepDetailPanel::StepDetailPanel(QWidget* parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("stepDetailPanel"));

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(10);

    auto* basicGroup = new QGroupBox(QStringLiteral("当前配准步"), this);
    auto* basicGrid = makeTwoColumnGrid(basicGroup);
    basicGrid->addWidget(makeMetricCard(QStringLiteral("步号"), stepValue_, basicGroup), 0, 0);
    basicGrid->addWidget(makeMetricCard(QStringLiteral("图像对"), pairValue_, basicGroup), 0, 1);
    basicGrid->addWidget(makeMetricCard(QStringLiteral("刚体变换"), transformValue_, basicGroup), 1, 0);
    basicGrid->addWidget(makeMetricCard(QStringLiteral("主轴 / 方向"), axisValue_, basicGroup), 1, 1);

    auto* searchGroup = new QGroupBox(QStringLiteral("搜索与重叠"), this);
    auto* searchGrid = makeTwoColumnGrid(searchGroup);
    searchGrid->addWidget(makeMetricCard(QStringLiteral("搜索窗口"), searchValue_, searchGroup), 0, 0);
    searchGrid->addWidget(makeMetricCard(QStringLiteral("覆盖率"), coverageValue_, searchGroup), 0, 1);
    searchGrid->addWidget(makeMetricCard(QStringLiteral("样本数量"), sampleValue_, searchGroup), 1, 0, 1, 2);

    auto* qualityGroup = new QGroupBox(QStringLiteral("质量评估"), this);
    auto* qualityGrid = makeTwoColumnGrid(qualityGroup);
    qualityGrid->addWidget(makeMetricCard(QStringLiteral("法向误差"), normalValue_, qualityGroup), 0, 0);
    qualityGrid->addWidget(makeMetricCard(QStringLiteral("切向误差"), tangentValue_, qualityGroup), 0, 1);
    qualityGrid->addWidget(makeMetricCard(QStringLiteral("切向相关"), corrValue_, qualityGroup), 1, 0);
    qualityGrid->addWidget(makeMetricCard(QStringLiteral("状态"), statusValue_, qualityGroup), 1, 1);

    rootLayout->addWidget(basicGroup);
    rootLayout->addWidget(searchGroup);
    rootLayout->addWidget(qualityGroup);
    rootLayout->addStretch(1);

    clearDetails();
}

void StepDetailPanel::clearDetails()
{
    const QString placeholder = QStringLiteral("等待选择配准步");
    stepValue_->setText(placeholder);
    pairValue_->setText(placeholder);
    transformValue_->setText(placeholder);
    axisValue_->setText(placeholder);
    searchValue_->setText(placeholder);
    coverageValue_->setText(placeholder);
    sampleValue_->setText(placeholder);
    normalValue_->setText(placeholder);
    tangentValue_->setText(placeholder);
    corrValue_->setText(placeholder);
    statusValue_->setText(placeholder);
    statusValue_->setStyleSheet(QString());
}

void StepDetailPanel::setStep(const pinjie::StitchStepRecord& step)
{
    const auto& metrics = step.transform.metrics;
    const auto& normalStats = displayNormalStats(metrics);
    const auto& tangentStats = displayTangentStats(metrics);
    const double tangentCorr = displayTangentCorr(metrics);

    const QString axis = step.transform.axis == stitch::AlignmentAxis::X ? QStringLiteral("X")
                                                                         : QStringLiteral("Y");
    const bool flagged = isRiskStep(metrics);

    stepValue_->setText(QString::number(static_cast<int>(step.stepIndex)));
    pairValue_->setText(QStringLiteral("%1 -> %2")
                            .arg(static_cast<int>(step.referenceImageIndex + 1))
                            .arg(static_cast<int>(step.targetImageIndex + 1)));
    transformValue_->setText(QStringLiteral("dx=%1 px, dy=%2 px, a=%3 deg\n匹配得分=%4")
                                 .arg(formatDouble(step.transform.dx, 2))
                                 .arg(formatDouble(step.transform.dy, 2))
                                 .arg(formatDouble(step.transform.da, 3))
                                 .arg(formatDouble(step.transform.score, 4)));
    axisValue_->setText(QStringLiteral("主轴 %1 / 方向 %2")
                            .arg(axis,
                                 QString::fromUtf8(step.transform.direction.data(),
                                                   static_cast<int>(step.transform.direction.size()))));
    searchValue_->setText(QStringLiteral("sx=%1 px, sy=%2 px")
                              .arg(formatDouble(step.searchRangeX, 1))
                              .arg(formatDouble(step.searchRangeY, 1)));
    coverageValue_->setText(QStringLiteral("重叠=%1, 内点覆盖=%2, 内点率=%3")
                                .arg(formatDouble(metrics.overlapCoverageRatio, 3))
                                .arg(formatDouble(metrics.inlierCoverageRatio, 3))
                                .arg(formatDouble(metrics.inlierRatio, 3)));
    sampleValue_->setText(QStringLiteral("重叠点 %1，内点 %2")
                              .arg(metrics.overlapCount)
                              .arg(metrics.inlierCount));
    normalValue_->setText(QStringLiteral("显示=%1 px, 全量=%2 px\n匹配代价=%3")
                              .arg(formatDouble(normalStats.rmse, 4))
                              .arg(formatDouble(metrics.normalAll.rmse, 4))
                              .arg(formatDouble(step.transform.normalMatchCost, 4)));
    tangentValue_->setText(QStringLiteral("显示=%1 px, 全量=%2 px\n匹配代价=%3")
                               .arg(formatDouble(tangentStats.rmse, 4))
                               .arg(formatDouble(metrics.tangentAll.rmse, 4))
                               .arg(formatDouble(step.transform.tangentResidualMatchCost, 4)));
    corrValue_->setText(QStringLiteral("显示=%1, 全量=%2\n匹配代价=%3")
                            .arg(formatDouble(tangentCorr, 4))
                            .arg(formatDouble(metrics.tangentCorrAll, 4))
                            .arg(formatDouble(step.transform.tangentCorrelationMatchCost, 4)));

    if (flagged) {
        statusValue_->setText(QStringLiteral("建议复核：误差或相关性超出稳态阈值"));
        statusValue_->setStyleSheet(QStringLiteral("color: rgb(210, 74, 61); font-weight: 600;"));
    } else {
        statusValue_->setText(QStringLiteral("质量稳定：可作为正常拼接步"));
        statusValue_->setStyleSheet(QStringLiteral("color: rgb(57, 138, 94); font-weight: 600;"));
    }
}

} // namespace pinjie::gui
