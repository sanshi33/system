#include "gui/MainWindow.h"

#include "common/ResultPathUtils.h"
#include "gui/QtImageUtils.h"
#include "report/QualitySummaryBuilder.h"
#include "stitch/DebugVis.h"
#include "stitch/IO.h"

#include <QAbstractItemView>
#include <QDir>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaType>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTableView>
#include <QTabWidget>
#include <QThread>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace pinjie::gui {

namespace {

enum WorkflowStageIndex {
    CalibrationStage = 0,
    AcquisitionStage = 1,
    ProcessingStage = 2,
    RegistrationStage = 3,
    ReportStage = 4
};

constexpr int kStageSidebarMinimumWidth = 260;
constexpr int kStageSidebarDefaultWidth = 280;
constexpr int kStageContentDefaultWidth = 1360;
constexpr int kReportSidebarWidth = 260;
constexpr int kReportMetricRailWidth = 220;
constexpr int kReportViewerMinimumHeight = 460;
constexpr int kReportDetailsMinimumHeight = 90;
constexpr int kRegistrationViewTabsMinimumHeight = 320;
constexpr int kRegistrationContentTopDefaultHeight = 620;
constexpr int kRegistrationContentBottomDefaultHeight = 170;
constexpr int kSidebarSummaryMaximumHeight = 76;

void configureStageSidebar(QWidget* sidebar)
{
    if (!sidebar) {
        return;
    }
    sidebar->setMinimumWidth(kStageSidebarMinimumWidth);
}

void configureStageSplit(QSplitter* split)
{
    if (!split) {
        return;
    }
    split->setChildrenCollapsible(false);
    split->setSizes({kStageSidebarDefaultWidth, kStageContentDefaultWidth});
}

QString stageLabel(const QString& stage)
{
    if (stage == QStringLiteral("load")) {
        return QStringLiteral("加载");
    }
    if (stage == QStringLiteral("preprocess")) {
        return QStringLiteral("预处理");
    }
    if (stage == QStringLiteral("stitch")) {
        return QStringLiteral("拼接");
    }
    return stage;
}

QString stageTitle(int stageIndex)
{
    switch (stageIndex) {
    case CalibrationStage:
        return QStringLiteral("标定");
    case AcquisitionStage:
        return QStringLiteral("图像加载");
    case ProcessingStage:
        return QStringLiteral("轮廓预处理");
    case RegistrationStage:
        return QStringLiteral("拼接配准");
    case ReportStage:
    default:
        return QStringLiteral("结果导出");
    }
}

QString runModeLabel(const pinjie::StitchRunMode runMode)
{
    switch (runMode) {
    case pinjie::StitchRunMode::Acquisition:
        return QStringLiteral("图像加载");
    case pinjie::StitchRunMode::Processing:
        return QStringLiteral("轮廓预处理");
    case pinjie::StitchRunMode::Registration:
        return QStringLiteral("拼接配准");
    case pinjie::StitchRunMode::Report:
        return QStringLiteral("结果导出");
    case pinjie::StitchRunMode::Full:
    default:
        return QStringLiteral("全流程");
    }
}

pinjie::StitchRunMode runModeForStageIndex(int stageIndex)
{
    switch (stageIndex) {
    case CalibrationStage:
    case AcquisitionStage:
        return pinjie::StitchRunMode::Acquisition;
    case ProcessingStage:
        return pinjie::StitchRunMode::Processing;
    case RegistrationStage:
        return pinjie::StitchRunMode::Registration;
    case ReportStage:
    default:
        return pinjie::StitchRunMode::Report;
    }
}

int stageIndexForRunMode(const pinjie::StitchRunMode runMode)
{
    switch (runMode) {
    case pinjie::StitchRunMode::Acquisition:
        return AcquisitionStage;
    case pinjie::StitchRunMode::Processing:
        return ProcessingStage;
    case pinjie::StitchRunMode::Registration:
        return RegistrationStage;
    case pinjie::StitchRunMode::Report:
    case pinjie::StitchRunMode::Full:
    default:
        return ReportStage;
    }
}

QString directionLabel(const pinjie::MotionPriorDirection direction)
{
    switch (direction) {
    case pinjie::MotionPriorDirection::XPositive:
        return QStringLiteral("X+");
    case pinjie::MotionPriorDirection::XNegative:
        return QStringLiteral("X-");
    case pinjie::MotionPriorDirection::YPositive:
        return QStringLiteral("Y+");
    case pinjie::MotionPriorDirection::YNegative:
        return QStringLiteral("Y-");
    case pinjie::MotionPriorDirection::Auto:
    default:
        return QStringLiteral("自动判断");
    }
}

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

QString formatMetric(double value, int decimals, const QString& suffix = QString())
{
    if (!std::isfinite(value)) {
        return QStringLiteral("--");
    }
    return QString::number(value, 'f', decimals) + suffix;
}

bool nearlyEqual(double lhs, double rhs, double tolerance = 1e-9)
{
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= tolerance * scale;
}

bool sameEdgeConfig(const stitch::EdgeDetectConfig& lhs, const stitch::EdgeDetectConfig& rhs)
{
    return nearlyEqual(lhs.cannyLow, rhs.cannyLow) &&
           nearlyEqual(lhs.cannyHigh, rhs.cannyHigh) &&
           lhs.subpixWindow == rhs.subpixWindow &&
           nearlyEqual(lhs.subpixSigma, rhs.subpixSigma) &&
           lhs.minWarnPoints == rhs.minWarnPoints &&
           lhs.enablePointFiltering == rhs.enablePointFiltering &&
           nearlyEqual(lhs.filterConfidenceQuantile, rhs.filterConfidenceQuantile) &&
           nearlyEqual(lhs.filterGradientQuantile, rhs.filterGradientQuantile) &&
           lhs.filterLocalLinearWindowRadius == rhs.filterLocalLinearWindowRadius &&
           nearlyEqual(lhs.filterHampelSigma, rhs.filterHampelSigma) &&
           nearlyEqual(lhs.filterHampelMinScale, rhs.filterHampelMinScale);
}

bool samePipelineConfig(const stitch::StitchPipelineConfig& lhs, const stitch::StitchPipelineConfig& rhs)
{
    return nearlyEqual(lhs.expectedOverlapRatio, rhs.expectedOverlapRatio) &&
           lhs.directionConstraint == rhs.directionConstraint &&
           nearlyEqual(lhs.baseSearchRange, rhs.baseSearchRange) &&
           nearlyEqual(lhs.approxShiftRatio, rhs.approxShiftRatio) &&
           nearlyEqual(lhs.rotationSearchMinDeg, rhs.rotationSearchMinDeg) &&
           nearlyEqual(lhs.rotationSearchMaxDeg, rhs.rotationSearchMaxDeg) &&
           nearlyEqual(lhs.rotationSearchStepDeg, rhs.rotationSearchStepDeg) &&
           nearlyEqual(lhs.tangentResidualCostWeight, rhs.tangentResidualCostWeight) &&
           nearlyEqual(lhs.tangentCorrelationCostWeight, rhs.tangentCorrelationCostWeight) &&
           lhs.generateDebugVisualization == rhs.generateDebugVisualization &&
           lhs.enableDesignComparison == rhs.enableDesignComparison &&
           lhs.designEvaluateProfileForm == rhs.designEvaluateProfileForm &&
           lhs.designReverseZ == rhs.designReverseZ &&
           lhs.designUseLeftEndpointAnchor == rhs.designUseLeftEndpointAnchor &&
           lhs.designAnchorRadialToLeftEndpoint == rhs.designAnchorRadialToLeftEndpoint &&
           lhs.designEnableBestFitTranslation == rhs.designEnableBestFitTranslation &&
           nearlyEqual(lhs.designBestFitDzMinMm, rhs.designBestFitDzMinMm) &&
           nearlyEqual(lhs.designBestFitDzMaxMm, rhs.designBestFitDzMaxMm) &&
           nearlyEqual(lhs.designBestFitDrMinMm, rhs.designBestFitDrMinMm) &&
           nearlyEqual(lhs.designBestFitDrMaxMm, rhs.designBestFitDrMaxMm) &&
           nearlyEqual(lhs.designBestFitStepMm, rhs.designBestFitStepMm) &&
           nearlyEqual(lhs.designPixelSizeMm, rhs.designPixelSizeMm) &&
           lhs.designInvertY == rhs.designInvertY &&
           lhs.designUseUpperEnvelope == rhs.designUseUpperEnvelope &&
           nearlyEqual(lhs.designProfileBinWidthPx, rhs.designProfileBinWidthPx) &&
           lhs.designFilterEndFaceEdges == rhs.designFilterEndFaceEdges &&
           nearlyEqual(lhs.designMaxAbsSlopeForGeneratrix, rhs.designMaxAbsSlopeForGeneratrix) &&
           lhs.designSlopeWindow == rhs.designSlopeWindow &&
           nearlyEqual(lhs.designTrimLeftAfterEndpointMm, rhs.designTrimLeftAfterEndpointMm) &&
           nearlyEqual(lhs.designTrimRightMm, rhs.designTrimRightMm);
}

bool canReuseOutputPaths(const pinjie::StitchRunRequest& previousRequest,
                         const pinjie::StitchRunRequest& currentRequest)
{
    return !previousRequest.resultOutputDir.empty() &&
           previousRequest.imagePaths == currentRequest.imagePaths &&
           sameEdgeConfig(previousRequest.edgeConfig, currentRequest.edgeConfig) &&
           samePipelineConfig(previousRequest.pipelineConfig, currentRequest.pipelineConfig);
}

void copyOutputPaths(const pinjie::StitchRunRequest& source, pinjie::StitchRunRequest& target)
{
    target.resultOutputDir = source.resultOutputDir;
    target.panoramaOutputPath = source.panoramaOutputPath;
    target.csvOutputPath = source.csvOutputPath;
    target.designErrorProfileCsvOutputPath = source.designErrorProfileCsvOutputPath;
    target.designErrorSummaryCsvOutputPath = source.designErrorSummaryCsvOutputPath;
    target.designComparisonOverlayOutputPath = source.designComparisonOverlayOutputPath;
    target.qualityReviewCsvOutputPath = source.qualityReviewCsvOutputPath;
    target.alignmentCandidateDiagnosticsCsvOutputPath = source.alignmentCandidateDiagnosticsCsvOutputPath;
    target.contourPointsCsvOutputPath = source.contourPointsCsvOutputPath;
    target.originContourOverlayCsvOutputPath = source.originContourOverlayCsvOutputPath;
    target.stitchedContourProfileCsvOutputPath = source.stitchedContourProfileCsvOutputPath;
    target.tangentStepCsvOutputPath = source.tangentStepCsvOutputPath;
    target.normalErrorProfileCsvOutputPath = source.normalErrorProfileCsvOutputPath;
    target.tangentProfileCsvOutputPath = source.tangentProfileCsvOutputPath;
    target.originTangentPointMetricsCsvOutputPath = source.originTangentPointMetricsCsvOutputPath;
    target.debugImageOutputDir = source.debugImageOutputDir;
    target.contourOverlayOutputPath = source.contourOverlayOutputPath;
    target.stitchedContourProfilePlotOutputPath = source.stitchedContourProfilePlotOutputPath;
    target.tangentCorrelationAllOutputPath = source.tangentCorrelationAllOutputPath;
    target.tangentCorrelationInlierOutputPath = source.tangentCorrelationInlierOutputPath;
}

QString displayOutputPath(const std::string& value)
{
    return value.empty() ? QStringLiteral("N/A") : QDir::toNativeSeparators(fromUtf8StdString(value));
}

QString readUtf8TextFile(const std::string& path)
{
    if (path.empty()) {
        return {};
    }

    std::ifstream stream(std::filesystem::u8path(path), std::ios::binary);
    if (!stream) {
        return {};
    }

    std::string content((std::istreambuf_iterator<char>(stream)),
                        std::istreambuf_iterator<char>());
    return QString::fromUtf8(content.data(), static_cast<int>(content.size()));
}

QString publicationFigurePngPath(const pinjie::StitchRunRequest& request)
{
    if (request.resultOutputDir.empty()) {
        return {};
    }
    const std::filesystem::path path =
        std::filesystem::u8path(request.resultOutputDir) / "journal_figure.png";
    return QDir::toNativeSeparators(QString::fromStdString(path.u8string()));
}

const stitch::ResidualStatistics& preferredNormalStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.normalInlier.valid() ? metrics.normalInlier : metrics.normalAll;
}

const stitch::ResidualStatistics& preferredTangentStats(const stitch::AlignmentMetrics& metrics)
{
    return metrics.tangentInlier.valid() ? metrics.tangentInlier : metrics.tangentAll;
}

double preferredTangentCorr(const stitch::AlignmentMetrics& metrics)
{
    return metrics.hasInliers() ? metrics.tangentCorrInlier : metrics.tangentCorrAll;
}

QPlainTextEdit* makeReadOnlyTextEdit(QWidget* parent)
{
    auto* edit = new QPlainTextEdit(parent);
    edit->setReadOnly(true);
    edit->setMinimumHeight(52);
    return edit;
}

QScrollArea* makeScrollArea(QWidget* child, QWidget* parent)
{
    auto* area = new QScrollArea(parent);
    area->setWidgetResizable(true);
    area->setFrameShape(QFrame::NoFrame);
    area->setWidget(child);
    return area;
}

QPushButton* makeModuleButton(const QString& text, QWidget* parent)
{
    auto* button = new QPushButton(text, parent);
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    button->setMinimumHeight(22);
    return button;
}

QFrame* makeSurfaceFrame(const QString& objectName, QWidget* parent)
{
    auto* frame = new QFrame(parent);
    frame->setObjectName(objectName);
    frame->setFrameShape(QFrame::NoFrame);
    return frame;
}

QWidget* makeMetricCard(const QString& title, QLabel*& valueLabel, QWidget* parent)
{
    auto* frame = new QFrame(parent);
    frame->setObjectName(QStringLiteral("metricCard"));
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(12, 10, 12, 10);
    layout->setSpacing(4);

    auto* titleLabel = new QLabel(title, frame);
    titleLabel->setObjectName(QStringLiteral("metricCardTitle"));
    valueLabel = new QLabel(QStringLiteral("--"), frame);
    valueLabel->setObjectName(QStringLiteral("metricCardValue"));

    layout->addWidget(titleLabel);
    layout->addWidget(valueLabel);
    frame->setFrameShape(QFrame::StyledPanel);
    return frame;
}

QString imageDisplayName(const QStringList& imagePaths, int oneBasedIndex)
{
    const int zeroBasedIndex = oneBasedIndex - 1;
    if (zeroBasedIndex < 0 || zeroBasedIndex >= imagePaths.size()) {
        return QStringLiteral("图像 %1").arg(oneBasedIndex);
    }

    const QFileInfo info(imagePaths.at(zeroBasedIndex));
    return QStringLiteral("图像 %1：%2").arg(oneBasedIndex).arg(info.fileName());
}

QString calibrationQualityLine(const pinjie::CalibrationResultCache& cache)
{
    if (!cache.valid) {
        return QStringLiteral("未加载标定结果");
    }

    const auto& quality = cache.profile.quality;
    QStringList parts;
    if (quality.fxPixelsPerMm > 0.0 || quality.fyPixelsPerMm > 0.0) {
        parts << QStringLiteral("fx=%1 px/mm, fy=%2 px/mm")
                     .arg(quality.fxPixelsPerMm, 0, 'f', 4)
                     .arg(quality.fyPixelsPerMm, 0, 'f', 4);
    }
    if (quality.reprojectionRmsPx >= 0.0) {
        parts << QStringLiteral("重投影 RMS=%1 px").arg(quality.reprojectionRmsPx, 0, 'f', 4);
    }
    if (quality.validImages > 0 || quality.totalImages > 0) {
        parts << QStringLiteral("有效图像=%1/%2").arg(quality.validImages).arg(quality.totalImages);
    }
    if (parts.isEmpty()) {
        return QStringLiteral("标定结果：%1").arg(fromUtf8StdString(cache.profile.profileName));
    }
    return parts.join(QStringLiteral(" | "));
}

QString calibrationIdentityLine(const pinjie::CalibrationResultCache& cache)
{
    if (!cache.valid) {
        return QStringLiteral("未加载标定结果");
    }

    const QString profileName = fromUtf8StdString(cache.profile.profileName);
    if (cache.createdAt.empty()) {
        return QStringLiteral("已加载标定结果：%1").arg(profileName);
    }

    return QStringLiteral("已加载标定结果：%1 | 创建时间：%2")
        .arg(profileName, fromUtf8StdString(cache.createdAt));
}

QStringList selectedPaperCsvLabels(const RunConfigPanel* configPanel)
{
    if (!configPanel) {
        return {};
    }

    QStringList labels;
    if (configPanel->saveStepSummaryCsv()) {
        labels << QStringLiteral("拼接汇总");
    }
    if (configPanel->saveContourPointsCsv()) {
        labels << QStringLiteral("轮廓叠加点");
    }
    if (configPanel->saveStitchedContourProfileCsv()) {
        labels << QStringLiteral("整体轮廓剖面");
    }
    if (configPanel->saveTangentStepCsv()) {
        labels << QStringLiteral("切向相关性");
    }
    if (configPanel->saveNormalErrorProfileCsv()) {
        labels << QStringLiteral("法向误差剖面");
    }
    if (configPanel->saveTangentProfileCsv()) {
        labels << QStringLiteral("轮廓波动分析");
    }
    if (configPanel->saveAlignmentCandidateDiagnosticsCsv()) {
        labels << QStringLiteral("候选诊断");
    }
    return labels;
}

bool isRiskStep(const pinjie::StitchStepRecord& step)
{
    const auto& metrics = step.transform.metrics;
    const auto& normalStats = preferredNormalStats(metrics);
    const double tangentCorr = preferredTangentCorr(metrics);
    return (normalStats.valid() && normalStats.rmse > 1.0) || tangentCorr < 0.95 || metrics.inlierRatio < 0.7;
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<pinjie::CalibrationResultCache>("pinjie::CalibrationResultCache");
    qRegisterMetaType<pinjie::StitchStepRecord>("pinjie::StitchStepRecord");
    qRegisterMetaType<pinjie::StitchRunCachePtr>("pinjie::StitchRunCachePtr");

    setWindowTitle(QStringLiteral("远心轮廓拼接工作台"));
    resize(1360, 760);

    calibrationPanel_ = new CalibrationConfigPanel(this);
    configPanel_ = new RunConfigPanel(this);

    auto* central = new QWidget(this);
    central->setObjectName(QStringLiteral("appRoot"));
    auto* rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(3, 3, 3, 3);
    rootLayout->setSpacing(3);

    auto* titleLabel = new QLabel(QStringLiteral("远心轮廓拼接工作台"), central);
    titleLabel->setObjectName(QStringLiteral("pageTitleLabel"));

    startButton_ = new QPushButton(QStringLiteral("运行全流程"), central);
    startButton_->setObjectName(QStringLiteral("primaryActionButton"));
    moduleRunButton_ = new QPushButton(QStringLiteral("运行当前阶段"), central);
    moduleRunButton_->setObjectName(QStringLiteral("secondaryActionButton"));
    stopButton_ = new QPushButton(QStringLiteral("停止任务"), central);
    stopButton_->setObjectName(QStringLiteral("dangerActionButton"));
    stopButton_->setEnabled(false);

    progressBar_ = new QProgressBar(central);
    progressBar_->setObjectName(QStringLiteral("workflowProgressBar"));
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setFormat(QStringLiteral("等待任务开始"));
    progressBar_->setTextVisible(false);

    auto* headerPanel = makeSurfaceFrame(QStringLiteral("headerPanel"), central);
    headerPanel->setMaximumHeight(40);
    auto* headerPanelLayout = new QVBoxLayout(headerPanel);
    headerPanelLayout->setContentsMargins(8, 4, 8, 4);
    headerPanelLayout->setSpacing(2);

    auto* headerRow = new QHBoxLayout();
    auto* titleLayout = new QVBoxLayout();
    titleLayout->setSpacing(2);
    auto* compactTitleRow = new QHBoxLayout();
    compactTitleRow->setSpacing(10);
    compactTitleRow->addWidget(titleLabel);
    compactTitleRow->addStretch(1);
    titleLayout->addLayout(compactTitleRow);
    headerRow->addLayout(titleLayout, 1);

    auto* actionLayout = new QVBoxLayout();
    actionLayout->setSpacing(1);
    auto* buttonRow = new QHBoxLayout();
    buttonRow->setSpacing(6);
    buttonRow->addWidget(startButton_);
    buttonRow->addWidget(moduleRunButton_);
    buttonRow->addWidget(stopButton_);
    actionLayout->addLayout(buttonRow);
    actionLayout->addWidget(progressBar_);
    headerRow->addLayout(actionLayout);
    headerPanelLayout->addLayout(headerRow);
    rootLayout->addWidget(headerPanel);

    auto* workflowPanel = makeSurfaceFrame(QStringLiteral("workflowPanel"), central);
    workflowPanel->setMaximumHeight(28);
    auto* workflowPanelLayout = new QHBoxLayout(workflowPanel);
    workflowPanelLayout->setContentsMargins(6, 3, 6, 3);
    workflowPanelLayout->setSpacing(5);

    auto* moduleRow = new QHBoxLayout();
    moduleRow->setSpacing(10);
    const QStringList moduleLabels = {QStringLiteral("1 标定"),
                                      QStringLiteral("2 输入图像"),
                                      QStringLiteral("3 轮廓预处理"),
                                      QStringLiteral("4 配准诊断"),
                                      QStringLiteral("5 导出结果")};
    const QStringList moduleStages = {QStringLiteral("calibration"),
                                      QStringLiteral("acquisition"),
                                      QStringLiteral("processing"),
                                      QStringLiteral("registration"),
                                      QStringLiteral("report")};
    for (int i = 0; i < moduleLabels.size(); ++i) {
        auto* button = makeModuleButton(moduleLabels.at(i), central);
        button->setProperty("stage", moduleStages.at(i));
        moduleButtons_.push_back(button);
    }
    for (int i = 0; i < static_cast<int>(moduleButtons_.size()); ++i) {
        moduleRow->addWidget(moduleButtons_[i]);
        connect(moduleButtons_[i], &QPushButton::clicked, this, [this, i]() { switchToStage(i); });
    }
    moduleRow->addStretch(1);
    workflowPanelLayout->addLayout(moduleRow, 1);

    workflowStateLabel_ = new QLabel(workflowPanel);
    workflowStateLabel_->setObjectName(QStringLiteral("workflowStateLabel"));
    workflowStateLabel_->setWordWrap(false);
    registrationPresetLabel_ = new QLabel(workflowPanel);
    registrationPresetLabel_->setObjectName(QStringLiteral("registrationPresetLabel"));
    registrationPresetLabel_->setWordWrap(false);
    workflowPanelLayout->addWidget(workflowStateLabel_, 2);
    workflowPanelLayout->addWidget(registrationPresetLabel_, 1);
    rootLayout->addWidget(workflowPanel);

    stageStack_ = new QStackedWidget(central);
    rootLayout->addWidget(stageStack_, 1);

    auto* calibrationPage = new QWidget(central);
    calibrationPage->setObjectName(QStringLiteral("calibrationStagePage"));
    calibrationOverviewEdit_ = makeReadOnlyTextEdit(calibrationPage);
    calibrationDetailEdit_ = makeReadOnlyTextEdit(calibrationPage);
    calibrationDetailEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    calibrationPreviewViewer_ = new ImageViewer(calibrationPage);
    auto* calibrationSidebar = new QWidget(calibrationPage);
    configureStageSidebar(calibrationSidebar);
    auto* calibrationSidebarLayout = new QVBoxLayout(calibrationSidebar);
    calibrationSidebarLayout->setContentsMargins(0, 0, 0, 0);
    calibrationOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    calibrationSidebarLayout->addWidget(makeScrollArea(calibrationPanel_, calibrationSidebar), 1);
    calibrationSidebarLayout->addWidget(calibrationOverviewEdit_);

    auto* calibrationRightPane = new QWidget(calibrationPage);
    auto* calibrationRightLayout = new QVBoxLayout(calibrationRightPane);
    calibrationRightLayout->setContentsMargins(0, 0, 0, 0);
    calibrationRightLayout->addWidget(calibrationPreviewViewer_, 1);
    calibrationRightLayout->addWidget(calibrationDetailEdit_);

    auto* calibrationSplit = new QSplitter(Qt::Horizontal, calibrationPage);
    calibrationSplit->addWidget(calibrationSidebar);
    calibrationSplit->addWidget(calibrationRightPane);
    calibrationSplit->setStretchFactor(0, 0);
    calibrationSplit->setStretchFactor(1, 1);
    configureStageSplit(calibrationSplit);

    auto* calibrationLayout = new QVBoxLayout(calibrationPage);
    calibrationLayout->setContentsMargins(0, 0, 0, 0);
    calibrationLayout->setSpacing(6);
    calibrationLayout->addWidget(calibrationSplit, 1);
    stageStack_->addWidget(calibrationPage);

    auto* acquisitionPage = new QWidget(central);
    acquisitionPage->setObjectName(QStringLiteral("acquisitionStagePage"));
    acquisitionOverviewEdit_ = makeReadOnlyTextEdit(acquisitionPage);
    acquisitionPreviewViewer_ = new ImageViewer(acquisitionPage);
    acquisitionPreviewInfoEdit_ = makeReadOnlyTextEdit(acquisitionPage);
    acquisitionPreviewInfoEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    acquisitionPreviewInfoEdit_->setPlainText(QStringLiteral("等待采集预览。"));
    auto* acquisitionSidebar = new QWidget(acquisitionPage);
    configureStageSidebar(acquisitionSidebar);
    auto* acquisitionSidebarLayout = new QVBoxLayout(acquisitionSidebar);
    acquisitionSidebarLayout->setContentsMargins(0, 0, 0, 0);
    acquisitionOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    acquisitionSidebarLayout->addWidget(makeScrollArea(configPanel_->acquisitionSection(), acquisitionSidebar), 1);
    acquisitionSidebarLayout->addWidget(acquisitionOverviewEdit_);

    auto* acquisitionPreviewPane = new QWidget(acquisitionPage);
    auto* acquisitionPreviewLayout = new QVBoxLayout(acquisitionPreviewPane);
    acquisitionPreviewLayout->setContentsMargins(0, 0, 0, 0);
    acquisitionPreviewLayout->addWidget(acquisitionPreviewViewer_, 1);
    acquisitionPreviewLayout->addWidget(acquisitionPreviewInfoEdit_);

    auto* acquisitionSplit = new QSplitter(Qt::Horizontal, acquisitionPage);
    acquisitionSplit->addWidget(acquisitionSidebar);
    acquisitionSplit->addWidget(acquisitionPreviewPane);
    acquisitionSplit->setStretchFactor(0, 0);
    acquisitionSplit->setStretchFactor(1, 1);
    configureStageSplit(acquisitionSplit);
    auto* acquisitionLayout = new QVBoxLayout(acquisitionPage);
    acquisitionLayout->setContentsMargins(0, 0, 0, 0);
    acquisitionLayout->setSpacing(6);
    acquisitionLayout->addWidget(acquisitionSplit, 1);
    stageStack_->addWidget(acquisitionPage);

    auto* processingPage = new QWidget(central);
    processingPage->setObjectName(QStringLiteral("processingStagePage"));
    processingOverviewEdit_ = makeReadOnlyTextEdit(processingPage);
    processingImageList_ = new QListWidget(processingPage);
    processingImageList_->setObjectName(QStringLiteral("previewImageList"));
    processingImageList_->setAlternatingRowColors(true);
    processingImageList_->setMinimumWidth(120);
    processingImageList_->setMaximumWidth(170);
    processingPreviewViewer_ = new ImageViewer(processingPage);
    processingPreviewInfoEdit_ = makeReadOnlyTextEdit(processingPage);
    processingPreviewInfoEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    auto* processingSidebar = new QWidget(processingPage);
    configureStageSidebar(processingSidebar);
    auto* processingSidebarLayout = new QVBoxLayout(processingSidebar);
    processingSidebarLayout->setContentsMargins(0, 0, 0, 0);
    processingOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    processingSidebarLayout->addWidget(makeScrollArea(configPanel_->processingSection(), processingSidebar), 1);
    processingSidebarLayout->addWidget(processingOverviewEdit_);

    auto* processingPreviewPane = new QWidget(processingPage);
    auto* processingPreviewLayout = new QVBoxLayout(processingPreviewPane);
    processingPreviewLayout->setContentsMargins(0, 0, 0, 0);
    processingPreviewLayout->addWidget(processingPreviewViewer_, 1);
    processingPreviewLayout->addWidget(processingPreviewInfoEdit_);

    auto* processingContentSplit = new QSplitter(Qt::Horizontal, processingPage);
    processingContentSplit->addWidget(processingImageList_);
    processingContentSplit->addWidget(processingPreviewPane);
    processingContentSplit->setStretchFactor(0, 0);
    processingContentSplit->setStretchFactor(1, 1);
    processingContentSplit->setSizes({140, 1140});

    auto* processingSplit = new QSplitter(Qt::Horizontal, processingPage);
    processingSplit->addWidget(processingSidebar);
    processingSplit->addWidget(processingContentSplit);
    processingSplit->setStretchFactor(0, 0);
    processingSplit->setStretchFactor(1, 1);
    configureStageSplit(processingSplit);

    auto* processingLayout = new QVBoxLayout(processingPage);
    processingLayout->setContentsMargins(0, 0, 0, 0);
    processingLayout->setSpacing(6);
    processingLayout->addWidget(processingSplit, 1);
    stageStack_->addWidget(processingPage);

    auto* registrationPage = new QWidget(central);
    registrationPage->setObjectName(QStringLiteral("registrationStagePage"));
    registrationOverviewEdit_ = makeReadOnlyTextEdit(registrationPage);
    registrationOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    referenceViewer_ = new ImageViewer(registrationPage);
    targetViewer_ = new ImageViewer(registrationPage);
    debugViewer_ = new ImageViewer(registrationPage);
    viewTabs_ = new QTabWidget(registrationPage);
    viewTabs_->setObjectName(QStringLiteral("imageTabs"));
    viewTabs_->setMinimumHeight(kRegistrationViewTabsMinimumHeight);
    viewTabs_->addTab(referenceViewer_, QStringLiteral("参考图像"));
    viewTabs_->addTab(targetViewer_, QStringLiteral("目标图像"));
    viewTabs_->addTab(debugViewer_, QStringLiteral("调试图像"));

    stepModel_ = new StepTableModel(this);
    stepTable_ = new QTableView(registrationPage);
    stepTable_->setObjectName(QStringLiteral("stepTable"));
    stepTable_->setModel(stepModel_);
    stepTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    stepTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    stepTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    stepTable_->setAlternatingRowColors(true);
    stepTable_->horizontalHeader()->setStretchLastSection(true);
    stepTable_->verticalHeader()->setVisible(false);
    stepTable_->setMinimumHeight(130);

    stepDetailPanel_ = new StepDetailPanel(registrationPage);
    stepDetailPanel_->setMinimumWidth(260);
    focusWorstStepButton_ = new QPushButton(QStringLiteral("定位最差步"), registrationPage);
    focusNextRiskStepButton_ = new QPushButton(QStringLiteral("定位下一风险步"), registrationPage);
    auto* registrationSidebar = new QWidget(registrationPage);
    registrationSidebar->setObjectName(QStringLiteral("registrationSidebarPanel"));
    configureStageSidebar(registrationSidebar);
    auto* registrationSidebarLayout = new QVBoxLayout(registrationSidebar);
    registrationSidebarLayout->setContentsMargins(0, 0, 0, 0);
    registrationSidebarLayout->setSpacing(6);
    registrationSidebarLayout->addWidget(makeScrollArea(configPanel_->registrationSection(), registrationSidebar), 1);
    registrationSidebarLayout->addWidget(registrationOverviewEdit_);

    auto* registrationDiagnosticsPanel = new QWidget(registrationPage);
    registrationDiagnosticsPanel->setObjectName(QStringLiteral("registrationDiagnosticsPanel"));
    auto* registrationDiagnosticsLayout = new QVBoxLayout(registrationDiagnosticsPanel);
    registrationDiagnosticsLayout->setContentsMargins(0, 0, 0, 0);
    registrationDiagnosticsLayout->setSpacing(6);

    auto* registrationTools = new QHBoxLayout();
    registrationTools->setContentsMargins(0, 0, 0, 0);
    registrationTools->setSpacing(6);
    registrationTools->addWidget(focusWorstStepButton_);
    registrationTools->addWidget(focusNextRiskStepButton_);
    registrationTools->addStretch(1);
    registrationDiagnosticsLayout->addLayout(registrationTools);

    auto* registrationDetailSplit = new QSplitter(Qt::Horizontal, registrationDiagnosticsPanel);
    registrationDetailSplit->addWidget(stepTable_);
    registrationDetailSplit->addWidget(stepDetailPanel_);
    registrationDetailSplit->setChildrenCollapsible(false);
    registrationDetailSplit->setStretchFactor(0, 4);
    registrationDetailSplit->setStretchFactor(1, 2);
    registrationDiagnosticsLayout->addWidget(registrationDetailSplit, 1);

    auto* registrationContent = new QSplitter(Qt::Vertical, registrationPage);
    registrationContent->addWidget(viewTabs_);
    registrationContent->addWidget(registrationDiagnosticsPanel);
    registrationContent->setChildrenCollapsible(false);
    registrationContent->setStretchFactor(0, 5);
    registrationContent->setStretchFactor(1, 2);

    auto* registrationSplit = new QSplitter(Qt::Horizontal, registrationPage);
    registrationSplit->addWidget(registrationSidebar);
    registrationSplit->addWidget(registrationContent);
    registrationSplit->setStretchFactor(0, 0);
    registrationSplit->setStretchFactor(1, 1);
    configureStageSplit(registrationSplit);
    registrationContent->setSizes({kRegistrationContentTopDefaultHeight, kRegistrationContentBottomDefaultHeight});
    registrationDetailSplit->setSizes({920, 260});

    auto* registrationLayout = new QVBoxLayout(registrationPage);
    registrationLayout->setContentsMargins(0, 0, 0, 0);
    registrationLayout->setSpacing(6);
    registrationLayout->addWidget(registrationSplit, 1);
    stageStack_->addWidget(registrationPage);

    auto* reportPage = new QWidget(central);
    reportPage->setObjectName(QStringLiteral("reportStagePage"));
    panoramaViewer_ = new ImageViewer(reportPage);
    designCompareViewer_ = new ImageViewer(reportPage);
    publicationFigureViewer_ = new ImageViewer(reportPage);
    summaryEdit_ = makeReadOnlyTextEdit(reportPage);
    qualityReviewEdit_ = makeReadOnlyTextEdit(reportPage);
    designCompareEdit_ = makeReadOnlyTextEdit(reportPage);
    candidateDiagnosticsEdit_ = makeReadOnlyTextEdit(reportPage);
    csvEdit_ = makeReadOnlyTextEdit(reportPage);
    logEdit_ = makeReadOnlyTextEdit(reportPage);
    reportViewTabs_ = new QTabWidget(reportPage);
    reportViewTabs_->setObjectName(QStringLiteral("reportViewTabs"));
    reportViewTabs_->setMinimumHeight(kReportViewerMinimumHeight);
    reportViewTabs_->addTab(panoramaViewer_, QStringLiteral("全景结果"));
    reportViewTabs_->addTab(designCompareViewer_, QStringLiteral("设计比对图"));
    reportViewTabs_->addTab(publicationFigureViewer_, QStringLiteral("误差分析图"));
    bottomTabs_ = new QTabWidget(reportPage);
    bottomTabs_->setObjectName(QStringLiteral("reportTabs"));
    bottomTabs_->setMinimumHeight(kReportDetailsMinimumHeight);
    bottomTabs_->addTab(summaryEdit_, QStringLiteral("摘要"));
    bottomTabs_->addTab(qualityReviewEdit_, QStringLiteral("质量审查"));
    bottomTabs_->addTab(designCompareEdit_, QStringLiteral("设计比对"));
    bottomTabs_->addTab(candidateDiagnosticsEdit_, QStringLiteral("候选诊断"));
    bottomTabs_->addTab(csvEdit_, QStringLiteral("CSV 数据"));
    bottomTabs_->addTab(logEdit_, QStringLiteral("日志"));
    auto* metricRail = new QWidget(reportPage);
    metricRail->setObjectName(QStringLiteral("reportMetricRail"));
    metricRail->setMinimumWidth(kReportMetricRailWidth);
    metricRail->setMaximumWidth(kReportMetricRailWidth + 28);
    auto* metricRailLayout = new QVBoxLayout(metricRail);
    metricRailLayout->setContentsMargins(8, 8, 8, 8);
    metricRailLayout->setSpacing(8);

    auto* metricRailTitle = new QLabel(QStringLiteral("核心指标"), metricRail);
    metricRailTitle->setObjectName(QStringLiteral("reportRailTitleLabel"));
    metricRailLayout->addWidget(metricRailTitle);

    auto* metricGrid = new QGridLayout();
    metricGrid->setContentsMargins(0, 0, 0, 0);
    metricGrid->setHorizontalSpacing(8);
    metricGrid->setVerticalSpacing(8);
    const auto addReportMetricCard = [&](const QString& title, QLabel*& valueLabel, int row, int column) {
        auto* card = makeMetricCard(title, valueLabel, metricRail);
        card->setProperty("compact", true);
        if (auto* cardLayout = card->layout()) {
            cardLayout->setContentsMargins(8, 6, 8, 6);
            cardLayout->setSpacing(2);
        }
        metricGrid->addWidget(card, row, column);
    };
    addReportMetricCard(QStringLiteral("已加载"), metricLoadedImagesValue_, 0, 0);
    addReportMetricCard(QStringLiteral("已预处理"), metricPreprocessedImagesValue_, 0, 1);
    addReportMetricCard(QStringLiteral("拼接步数"), metricStepCountValue_, 1, 0);
    addReportMetricCard(QStringLiteral("风险步数"), metricFlaggedStepsValue_, 1, 1);
    addReportMetricCard(QStringLiteral("法向 RMSE"), metricNormalRmseValue_, 2, 0);
    addReportMetricCard(QStringLiteral("切向 RMSE"), metricTangentRmseValue_, 2, 1);
    addReportMetricCard(QStringLiteral("切向相关"), metricTangentCorrValue_, 3, 0);
    addReportMetricCard(QStringLiteral("最差步"), metricWorstStepValue_, 3, 1);
    addReportMetricCard(QStringLiteral("设计 RMSE"), metricDesignRmseValue_, 4, 0);
    addReportMetricCard(QStringLiteral("设计 P95"), metricDesignP95Value_, 4, 1);
    addReportMetricCard(QStringLiteral("设计平移"), metricDesignOffsetValue_, 5, 0);
    addReportMetricCard(QStringLiteral("有效点数"), metricDesignUsedCountValue_, 5, 1);
    metricRailLayout->addLayout(metricGrid);
    metricRailLayout->addStretch(1);

    reportExportStatusLabel_ = new QLabel(reportPage);
    reportExportStatusLabel_->setObjectName(QStringLiteral("reportExportStatusLabel"));
    reportExportStatusLabel_->setWordWrap(false);
    generateFigureButton_ = new QPushButton(QStringLiteral("生成误差分析图"), reportPage);
    generateFigureButton_->setObjectName(QStringLiteral("secondaryActionButton"));
    exportSelectedCsvButton_ = new QPushButton(QStringLiteral("导出所选 CSV"), reportPage);
    exportSelectedCsvButton_->setObjectName(QStringLiteral("secondaryActionButton"));

    auto* reportToolbarFrame = makeSurfaceFrame(QStringLiteral("reportToolbarFrame"), reportPage);
    auto* reportToolbar = new QHBoxLayout(reportToolbarFrame);
    reportToolbar->setContentsMargins(10, 8, 10, 8);
    reportToolbar->setSpacing(8);

    auto* reportParameterPaneButton = new QToolButton(reportToolbarFrame);
    reportParameterPaneButton->setObjectName(QStringLiteral("paneToggleButton"));
    reportParameterPaneButton->setText(QStringLiteral("参数"));
    reportParameterPaneButton->setCheckable(true);
    reportParameterPaneButton->setChecked(true);
    reportParameterPaneButton->setToolTip(QStringLiteral("显示或隐藏左侧输出参数"));

    auto* reportMetricPaneButton = new QToolButton(reportToolbarFrame);
    reportMetricPaneButton->setObjectName(QStringLiteral("paneToggleButton"));
    reportMetricPaneButton->setText(QStringLiteral("指标"));
    reportMetricPaneButton->setCheckable(true);
    reportMetricPaneButton->setChecked(true);
    reportMetricPaneButton->setToolTip(QStringLiteral("显示或隐藏右侧核心指标"));

    reportToolbar->addWidget(reportParameterPaneButton);
    reportToolbar->addWidget(reportMetricPaneButton);
    reportToolbar->addWidget(reportExportStatusLabel_, 1);

    auto* reportSidebar = new QWidget(reportPage);
    configureStageSidebar(reportSidebar);
    reportSidebar->setMinimumWidth(kReportSidebarWidth);
    reportSidebar->setMaximumWidth(kReportSidebarWidth + 40);
    auto* reportSidebarLayout = new QVBoxLayout(reportSidebar);
    reportSidebarLayout->setContentsMargins(0, 0, 0, 0);
    reportSidebarLayout->addWidget(makeScrollArea(configPanel_->reportSection(), reportSidebar), 1);

    auto* reportSidebarActions = makeSurfaceFrame(QStringLiteral("reportSidebarActionsFrame"), reportSidebar);
    auto* reportSidebarActionsLayout = new QVBoxLayout(reportSidebarActions);
    reportSidebarActionsLayout->setContentsMargins(10, 10, 10, 10);
    reportSidebarActionsLayout->setSpacing(8);
    reportSidebarActionsLayout->addWidget(generateFigureButton_);
    reportSidebarActionsLayout->addWidget(exportSelectedCsvButton_);
    reportSidebarLayout->addWidget(reportSidebarActions);

    auto* reportContent = new QWidget(reportPage);
    auto* reportContentLayout = new QVBoxLayout(reportContent);
    reportContentLayout->setContentsMargins(0, 0, 0, 0);
    reportContentLayout->setSpacing(8);
    reportContentLayout->addWidget(reportToolbarFrame);

    auto* reportResultSplit = new QSplitter(Qt::Vertical, reportContent);
    reportResultSplit->setChildrenCollapsible(false);
    reportResultSplit->addWidget(reportViewTabs_);
    reportResultSplit->addWidget(bottomTabs_);
    reportResultSplit->setStretchFactor(0, 5);
    reportResultSplit->setStretchFactor(1, 2);
    reportResultSplit->setSizes({680, 150});
    reportContentLayout->addWidget(reportResultSplit, 1);

    auto* reportSplit = new QSplitter(Qt::Horizontal, reportPage);
    reportSplit->addWidget(reportSidebar);
    reportSplit->addWidget(reportContent);
    reportSplit->addWidget(metricRail);
    reportSplit->setStretchFactor(0, 0);
    reportSplit->setStretchFactor(1, 1);
    reportSplit->setStretchFactor(2, 0);
    configureStageSplit(reportSplit);
    reportSplit->setSizes({kReportSidebarWidth, 900, kReportMetricRailWidth});

    connect(reportParameterPaneButton, &QToolButton::toggled, reportSidebar, [reportSidebar](bool checked) {
        reportSidebar->setVisible(checked);
    });
    connect(reportMetricPaneButton, &QToolButton::toggled, metricRail, [metricRail](bool checked) {
        metricRail->setVisible(checked);
    });

    auto* reportLayout = new QVBoxLayout(reportPage);
    reportLayout->setContentsMargins(0, 0, 0, 0);
    reportLayout->addWidget(reportSplit);
    stageStack_->addWidget(reportPage);

    setCentralWidget(central);

    connect(startButton_, &QPushButton::clicked, this, [this]() { startRun(); });
    connect(moduleRunButton_, &QPushButton::clicked, this, [this]() { startCurrentModuleRun(); });
    connect(stopButton_, &QPushButton::clicked, this, [this]() { stopRun(); });
    connect(focusWorstStepButton_, &QPushButton::clicked, this, [this]() { jumpToWorstStep(); });
    connect(focusNextRiskStepButton_, &QPushButton::clicked, this, [this]() { jumpToNextRiskStep(); });
    connect(generateFigureButton_, &QPushButton::clicked, this, [this]() { generatePublicationFigure(); });
    connect(exportSelectedCsvButton_, &QPushButton::clicked, this, [this]() { exportSelectedReportCsvs(); });
    connect(processingImageList_, &QListWidget::currentRowChanged, this, [this](int) { onProcessingImageSelectionChanged(); });
    connect(stepTable_->selectionModel(), &QItemSelectionModel::currentRowChanged, this, [this]() { onStepSelectionChanged(); });
    connect(calibrationPanel_, &CalibrationConfigPanel::configChanged, this, [this]() {
        refreshStageSummaries();
        updateWorkflowAccessState();
    });
    connect(configPanel_, &RunConfigPanel::configChanged, this, [this]() {
        refreshStageSummaries();
        refreshReportExportState();
        updateWorkflowAccessState();
    });
    connect(configPanel_, &RunConfigPanel::cameraPreviewUpdated, this, [this](const QImage& image, const QString& summary) {
        onAcquisitionPreviewUpdated(image, summary);
    });

    calibrationOverviewEdit_->setPlainText(QStringLiteral("等待标定参数。"));
    acquisitionOverviewEdit_->setPlainText(QStringLiteral("等待选择工件图像目录或完成在线采集。"));
    calibrationDetailEdit_->setPlainText(QStringLiteral("等待标定结果。"));
    processingPreviewInfoEdit_->setPlainText(QStringLiteral("等待预处理预览。"));
    if (qualityReviewEdit_) {
        qualityReviewEdit_->setPlainText(QStringLiteral("等待拼接完成后生成 quality_review.csv。"));
    }
    if (designCompareEdit_) {
        designCompareEdit_->setPlainText(QStringLiteral("等待拼接完成后生成设计母线比对结果。"));
    }
    if (candidateDiagnosticsEdit_) {
        candidateDiagnosticsEdit_->setPlainText(QStringLiteral("等待拼接完成后生成候选诊断结果。"));
    }
    if (publicationFigureViewer_) {
        publicationFigureViewer_->clearImage();
    }
    statusBar()->showMessage(QStringLiteral("就绪"));

    resetReportMetricCards();
    applyWindowStyle();
    switchToStage(CalibrationStage);
    refreshStageSummaries();
    refreshCalibrationDetailPane();
    refreshRegistrationToolState();
    refreshReportExportState();
}

MainWindow::~MainWindow()
{
    if (calibrationWorker_) {
        calibrationWorker_->requestStop();
    }
    if (worker_) {
        worker_->requestStop();
    }
    if (workerThread_) {
        workerThread_->quit();
        workerThread_->wait(2000);
    }
}

void MainWindow::startRun()
{
    startRun(pinjie::StitchRunMode::Full);
}

void MainWindow::startCurrentModuleRun()
{
    const int currentStageIndex = stageStack_ ? stageStack_->currentIndex() : CalibrationStage;
    if (currentStageIndex == CalibrationStage) {
        startCalibration();
        return;
    }

    startRun(runModeForStageIndex(currentStageIndex));
}

void MainWindow::startCalibration()
{
    if (workerThread_) {
        return;
    }

    pinjie::CameraCalibrationRequest request;
    QString errorMessage;
    if (!calibrationPanel_->buildRequest(request, errorMessage)) {
        QMessageBox::warning(this, QStringLiteral("相机标定"), errorMessage);
        switchToStage(CalibrationStage);
        return;
    }

    lastCalibrationRequest_ = request;
    calibrationPreviewViewer_->clearImage();
    calibrationOverviewEdit_->clear();
    calibrationDetailEdit_->setPlainText(QStringLiteral("正在准备标定任务..."));
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setFormat(QStringLiteral("标定中..."));
    currentTaskSupportsStop_ = false;

    workerThread_ = new QThread(this);
    calibrationWorker_ = new CalibrationWorker(request);
    calibrationWorker_->moveToThread(workerThread_);

    connect(workerThread_, &QThread::started, calibrationWorker_, &CalibrationWorker::run);
    connect(calibrationWorker_, &CalibrationWorker::logMessage, this, [this](const QString& message) { appendLog(message); });
    connect(calibrationWorker_, &CalibrationWorker::progressChanged, this, [this](int current, int total) {
        onCalibrationProgress(current, total);
    });
    connect(calibrationWorker_, &CalibrationWorker::previewReady, this, [this](int index, int total, const QImage& image) {
        onCalibrationPreviewReady(index, total, image);
    });
    connect(calibrationWorker_,
            &CalibrationWorker::runFinished,
            this,
            [this](bool ok, bool loadedFromCache, const QString& message, pinjie::CalibrationResultCache cache) {
                onCalibrationFinished(ok, loadedFromCache, message, std::move(cache));
            });
    connect(calibrationWorker_, &CalibrationWorker::runFinished, workerThread_, &QThread::quit);
    connect(workerThread_, &QThread::finished, calibrationWorker_, &QObject::deleteLater);
    connect(workerThread_, &QThread::finished, workerThread_, &QObject::deleteLater);
    connect(workerThread_, &QThread::finished, this, [this]() { cleanupWorker(); });

    setUiRunning(true);
    appendLog(QStringLiteral("[信息] 开始标定任务"));
    switchToStage(CalibrationStage);
    workerThread_->start();
}

void MainWindow::startRun(pinjie::StitchRunMode runMode)
{
    if (workerThread_) {
        return;
    }

    if (!hasActiveCalibration()) {
        QMessageBox::information(this,
                                 QStringLiteral("需要标定"),
                                 QStringLiteral("请先加载有效的标定结果，再运行当前阶段。"));
        switchToStage(CalibrationStage);
        return;
    }

    pinjie::StitchRunRequest request;
    QString errorMessage;
    if (!configPanel_->buildRequest(request, errorMessage, false)) {
        QMessageBox::warning(this, QStringLiteral("运行配置"), errorMessage);
        switchToStage(AcquisitionStage);
        return;
    }

    const bool reusePreviousOutputPaths =
        runMode != pinjie::StitchRunMode::Full &&
        runMode != pinjie::StitchRunMode::Acquisition &&
        runCache_ &&
        canReuseOutputPaths(lastRequest_, request);
    if (reusePreviousOutputPaths) {
        copyOutputPaths(lastRequest_, request);
    } else if (!configPanel_->buildRequest(request, errorMessage, true)) {
        QMessageBox::warning(this, QStringLiteral("运行配置"), errorMessage);
        switchToStage(AcquisitionStage);
        return;
    }

    request.previousCache = runCache_;
    lastRequest_ = request;
    currentRunMode_ = runMode;
    currentTaskSupportsStop_ = true;
    completedSteps_.clear();
    stepModel_->clear();
    preprocessImages_.clear();
    debugImages_.clear();
    processingImageList_->clear();
    processingPreviewViewer_->clearImage();
    processingPreviewInfoEdit_->setPlainText(QStringLiteral("等待预处理图像。"));
    referenceViewer_->clearImage();
    targetViewer_->clearImage();
    debugViewer_->clearImage();
    panoramaViewer_->clearImage();
    if (designCompareViewer_) {
        designCompareViewer_->clearImage();
    }
    if (reportViewTabs_ && panoramaViewer_) {
        reportViewTabs_->setCurrentWidget(panoramaViewer_);
    }
    stepDetailPanel_->clearDetails();
    summaryEdit_->clear();
    if (designCompareEdit_) {
        designCompareEdit_->clear();
    }
    csvEdit_->clear();
    logEdit_->clear();
    resetReportMetricCards();
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setFormat(QStringLiteral("处理中..."));
    bottomTabs_->setCurrentWidget(logEdit_);
    refreshReportExportState(QStringLiteral("运行结束后可在这里导出所选 CSV。"));

    lastImagePaths_.clear();
    for (const std::string& path : request.imagePaths) {
        lastImagePaths_.push_back(fromUtf8StdString(path));
    }

    workerThread_ = new QThread(this);
    worker_ = new StitchWorker(request, runMode);
    worker_->moveToThread(workerThread_);

    connect(workerThread_, &QThread::started, worker_, &StitchWorker::run);
    connect(worker_, &StitchWorker::logMessage, this, [this](const QString& message) { appendLog(message); });
    connect(worker_, &StitchWorker::progressChanged, this, [this](const QString& stage, int current, int total) {
        onProgress(stage, current, total);
    });
    connect(worker_, &StitchWorker::stepCompleted, this, [this](const pinjie::StitchStepRecord& step) {
        onStepCompleted(step);
    });
    connect(worker_, &StitchWorker::imageReady, this, [this](const QString& stage, int index, int total, const QImage& image) {
        onImageReady(stage, index, total, image);
    });
    connect(worker_, &StitchWorker::runFinished, this,
            [this](bool ok,
                   const QString& message,
                   int loadedImageCount,
                   int preprocessedImageCount,
                   const QImage& panorama,
                   const QString& csvText,
                   pinjie::StitchRunCachePtr cache) {
                onRunFinished(ok, message, loadedImageCount, preprocessedImageCount, panorama, csvText, cache);
            });
    connect(worker_, &StitchWorker::runFinished, workerThread_, &QThread::quit);
    connect(workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(workerThread_, &QThread::finished, workerThread_, &QObject::deleteLater);
    connect(workerThread_, &QThread::finished, this, [this]() { cleanupWorker(); });

    setUiRunning(true);
    refreshStageSummaries();
    refreshRegistrationToolState();
    appendLog(QStringLiteral("[信息] 开始运行：%1").arg(runModeLabel(runMode)));
    switchToStage(runMode == pinjie::StitchRunMode::Full ? AcquisitionStage : stageIndexForRunMode(runMode));
    workerThread_->start();
}

void MainWindow::stopRun()
{
    if (calibrationWorker_ && !currentTaskSupportsStop_) {
        QMessageBox::information(this,
                                 QStringLiteral("无法停止"),
                                 QStringLiteral("当前阶段不支持中止标定。"));
        return;
    }

    if (!worker_) {
        return;
    }

    appendLog(QStringLiteral("[信息] 已请求停止当前任务"));
    worker_->requestStop();
}

void MainWindow::onCalibrationProgress(int current, int total)
{
    if (total <= 0) {
        progressBar_->setRange(0, 0);
        progressBar_->setFormat(QStringLiteral("标定处理中..."));
    } else {
        progressBar_->setRange(0, total);
        progressBar_->setValue(current);
        progressBar_->setFormat(QStringLiteral("标定图像 %1/%2").arg(current).arg(total));
    }

    switchToStage(CalibrationStage);
    statusBar()->showMessage(progressBar_->format());
}

void MainWindow::onCalibrationPreviewReady(int index, int total, const QImage& image)
{
    if (!image.isNull()) {
        calibrationPreviewViewer_->setImage(image);
    }

    calibrationDetailEdit_->setPlainText(
        QStringLiteral("正在预览标定图像 %1 / %2。").arg(index).arg(total));
    statusBar()->showMessage(QStringLiteral("标定图像 %1/%2").arg(index).arg(total));
}

void MainWindow::onCalibrationFinished(bool ok,
                                       bool loadedFromCache,
                                       const QString& message,
                                       pinjie::CalibrationResultCache cache)
{
    if (ok && cache.valid) {
        activeCalibrationCache_ = std::move(cache);
        const QImage preview(fromUtf8StdString(activeCalibrationCache_.paths.previewImagePath));
        if (!preview.isNull()) {
            calibrationPreviewViewer_->setImage(preview);
        }
    }

    appendLog(ok ? QStringLiteral("[信息] 标定完成")
                 : QStringLiteral("[错误] 标定失败"));
    if (!message.isEmpty()) {
        appendLog(message);
    }

    refreshCalibrationDetailPane();
    refreshStageSummaries();

    if (ok && activeCalibrationCache_.valid) {
        calibrationOverviewEdit_->setPlainText(
            loadedFromCache ? QStringLiteral("标定结果已从缓存加载。")
                            : QStringLiteral("标定完成。"));
        switchToStage(AcquisitionStage);
    } else if (!ok) {
        calibrationOverviewEdit_->setPlainText(QStringLiteral("标定失败，请查看日志。"));
        switchToStage(CalibrationStage);
    }
}

void MainWindow::onProgress(const QString& stage, int current, int total)
{
    if (total <= 0) {
        progressBar_->setRange(0, 0);
        progressBar_->setFormat(QStringLiteral("%1...").arg(stageLabel(stage)));
    } else {
        progressBar_->setRange(0, total);
        progressBar_->setValue(current);
        progressBar_->setFormat(QStringLiteral("%1 %2/%3").arg(stageLabel(stage)).arg(current).arg(total));
    }

    if (stage == QStringLiteral("load")) {
        switchToStage(AcquisitionStage);
    } else if (stage == QStringLiteral("preprocess")) {
        switchToStage(ProcessingStage);
    } else if (stage == QStringLiteral("stitch")) {
        switchToStage(RegistrationStage);
    }

    statusBar()->showMessage(QStringLiteral("%1 %2/%3").arg(stageLabel(stage)).arg(current).arg(total));
}

void MainWindow::onStepCompleted(const pinjie::StitchStepRecord& step)
{
    completedSteps_.push_back(step);
    stepModel_->appendStep(step);

    const int row = stepModel_->rowCount() - 1;
    if (row >= 0) {
        selectStepRow(row);
    }

    const pinjie::QualitySummary partialSummary = pinjie::buildQualitySummary(completedSteps_);
    updateReportMetricCards(static_cast<int>(lastImagePaths_.size()), preprocessImages_.size(), &partialSummary, nullptr);
    refreshRegistrationToolState();
}

void MainWindow::onImageReady(const QString& stage, int index, int total, const QImage& image)
{
    Q_UNUSED(total);

    if (stage == QStringLiteral("preprocess_preview")) {
        preprocessImages_.insert(index, image);

        QListWidgetItem* matchedItem = nullptr;
        for (int row = 0; row < processingImageList_->count(); ++row) {
            QListWidgetItem* item = processingImageList_->item(row);
            if (item && item->data(Qt::UserRole).toInt() == index) {
                matchedItem = item;
                break;
            }
        }

        if (!matchedItem) {
            matchedItem = new QListWidgetItem(imageDisplayName(lastImagePaths_, index), processingImageList_);
            matchedItem->setData(Qt::UserRole, index);
        }

        if (!processingImageList_->currentItem()) {
            processingImageList_->setCurrentItem(matchedItem);
        } else if (processingImageList_->currentItem()->data(Qt::UserRole).toInt() == index) {
            onProcessingImageSelectionChanged();
        }

        statusBar()->showMessage(QStringLiteral("已更新预处理图像 %1").arg(index));
        updateReportMetricCards(static_cast<int>(lastImagePaths_.size()), preprocessImages_.size(), nullptr, nullptr);
        return;
    }

    if (stage == QStringLiteral("debug_step")) {
        debugImages_.insert(index, image);
        debugViewer_->setImage(image);
        viewTabs_->setCurrentWidget(debugViewer_);
        statusBar()->showMessage(QStringLiteral("已更新调试图像 %1").arg(index));
    }
}

void MainWindow::onAcquisitionPreviewUpdated(const QImage& image, const QString& summary)
{
    if (acquisitionPreviewViewer_) {
        if (image.isNull()) {
            acquisitionPreviewViewer_->clearImage();
        } else {
            acquisitionPreviewViewer_->setImage(image);
        }
    }

    if (acquisitionPreviewInfoEdit_) {
        acquisitionPreviewInfoEdit_->setPlainText(summary.isEmpty()
                                                      ? QStringLiteral("等待相机采集预览。")
                                                      : summary);
    }

    if (!image.isNull()) {
        switchToStage(AcquisitionStage);
        statusBar()->showMessage(QStringLiteral("已更新采集预览"), 2000);
    }
}

void MainWindow::onRunFinished(bool ok,
                               const QString& message,
                               int loadedImageCount,
                               int preprocessedImageCount,
                               const QImage& panorama,
                               const QString& csvText,
                               pinjie::StitchRunCachePtr cache)
{
    const pinjie::QualitySummary* qualitySummaryPtr = nullptr;
    pinjie::QualitySummary qualitySummary;
    const pinjie::cad_design::DesignErrorSummary* designSummaryPtr = nullptr;
    pinjie::cad_design::DesignAlignmentResult designAlignmentResult;
    QStringList designCompareLines;

    if (cache && (cache->hasLoadedImages() || cache->hasPreprocessedEdges() || cache->hasStitching())) {
        runCache_ = std::move(cache);
    }

    if (completedSteps_.empty() && runCache_ && runCache_->hasStitching()) {
        completedSteps_ = runCache_->stitching.steps;
        stepModel_->clear();
        for (const auto& step : completedSteps_) {
            stepModel_->appendStep(step);
        }
    }

    if (!panorama.isNull()) {
        panoramaViewer_->setImage(panorama);
    }

    appendLog(ok ? QStringLiteral("[信息] %1已完成").arg(runModeLabel(currentRunMode_))
                 : QStringLiteral("[错误] %1失败").arg(runModeLabel(currentRunMode_)));
    if (!message.isEmpty()) {
        appendLog(message);
    }

    csvEdit_->setPlainText(csvText);
    if (qualityReviewEdit_) {
        const QString reviewText = readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath);
        qualityReviewEdit_->setPlainText(
            reviewText.isEmpty() ? QStringLiteral("本次运行未生成 quality_review.csv。") : reviewText);
    }
    if (candidateDiagnosticsEdit_) {
        const QString diagnosticsText =
            completedSteps_.empty()
                ? QString()
                : fromUtf8StdString(stitch::buildAlignmentCandidateDiagnosticsCsv(completedSteps_));
        candidateDiagnosticsEdit_->setPlainText(
            diagnosticsText.isEmpty() ? QStringLiteral("本次运行未生成候选诊断结果。") : diagnosticsText);
    }
    refreshPublicationFigurePreview();

    QStringList summaryLines;
    summaryLines << QStringLiteral("实验结果摘要");
    summaryLines << QString();
    summaryLines << QStringLiteral("运行模式：%1").arg(runModeLabel(currentRunMode_));
    summaryLines << QStringLiteral("运行结果：%1").arg(ok ? QStringLiteral("成功") : QStringLiteral("失败"));
    summaryLines << QStringLiteral("已加载图像：%1").arg(loadedImageCount);
    summaryLines << QStringLiteral("已预处理图像：%1").arg(preprocessedImageCount);
    if (!lastRequest_.resultOutputDir.empty()) {
        summaryLines << QStringLiteral("结果目录：%1")
                            .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir)));
    }
    if (!completedSteps_.empty()) {
        summaryLines << QStringLiteral("拼接步数：%1").arg(completedSteps_.size());
    }
    if (!message.isEmpty()) {
        summaryLines << QStringLiteral("运行消息：%1").arg(message);
    }
    if (!lastRequest_.panoramaOutputPath.empty()) {
        summaryLines << QStringLiteral("全景图：%1")
                            .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.panoramaOutputPath)));
    }
    if (!lastRequest_.csvOutputPath.empty()) {
        summaryLines << QStringLiteral("汇总 CSV：%1")
                            .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.csvOutputPath)));
    }

    if (!completedSteps_.empty()) {
        qualitySummary = pinjie::buildQualitySummary(completedSteps_);
        qualitySummaryPtr = &qualitySummary;
        const QString qualityText =
            fromUtf8StdString(pinjie::buildQualitySummaryText(completedSteps_, qualitySummary));
        if (!qualityText.isEmpty()) {
            summaryLines << QString() << qualityText;
        }
    }

    if (runCache_ && runCache_->hasPreprocessedEdges() && runCache_->hasStitching()) {
        designAlignmentResult = pinjie::cad_design::compareMeasuredProfileToDesign(
            runCache_->preprocessedEdges,
            runCache_->stitching.imageTransforms,
            lastRequest_.pipelineConfig);
        if (designAlignmentResult.ok) {
            designSummaryPtr = &designAlignmentResult.summary;
            designCompareLines << QStringLiteral("设计母线比对结果");
            designCompareLines << QString();
            designCompareLines << QStringLiteral("reverse_z：%1").arg(designSummaryPtr->designReverseZ ? 1 : 0);
            designCompareLines << QStringLiteral("dz：%1 mm").arg(formatMetric(designSummaryPtr->dzMm, 4));
            designCompareLines << QStringLiteral("dr：%1 mm").arg(formatMetric(designSummaryPtr->drMm, 4));
            designCompareLines << QStringLiteral("左端锚点：(%1, %2) px")
                                      .arg(formatMetric(designSummaryPtr->anchorXPx, 3))
                                      .arg(formatMetric(designSummaryPtr->anchorYPx, 3));
            designCompareLines << QStringLiteral("有效点数：%1").arg(static_cast<qulonglong>(designSummaryPtr->usedCount));
            designCompareLines << QStringLiteral("整体尺寸偏置：%1 um")
                                      .arg(formatMetric(designSummaryPtr->meanNormalErrorUm, 3));
            designCompareLines << QStringLiteral("法向 RMSE：%1 um")
                                      .arg(formatMetric(designSummaryPtr->normalStats.rmseUm, 3));
            designCompareLines << QStringLiteral("法向 MAE：%1 um")
                                      .arg(formatMetric(designSummaryPtr->normalStats.maeUm, 3));
            designCompareLines << QStringLiteral("法向 P95：%1 um")
                                      .arg(formatMetric(designSummaryPtr->normalStats.p95AbsUm, 3));
            designCompareLines << QStringLiteral("法向 PV：%1 um")
                                      .arg(formatMetric(designSummaryPtr->normalStats.pvUm, 3));
            designCompareLines << QStringLiteral("轮廓波动 RMS：%1 um")
                                      .arg(formatMetric(designSummaryPtr->profileStats.rmseUm, 3));
            designCompareLines << QStringLiteral("轮廓波动 P95：%1 um")
                                      .arg(formatMetric(designSummaryPtr->profileStats.p95AbsUm, 3));
            designCompareLines << QStringLiteral("轮廓度 PV：%1 um")
                                      .arg(formatMetric(designSummaryPtr->profileStats.pvUm, 3));
            designCompareLines << QString();
            designCompareLines << QStringLiteral("CSV 预览");
            designCompareLines << QStringLiteral("----------------------------------------");
            designCompareLines << fromUtf8StdString(designAlignmentResult.summaryCsvText).trimmed();
            designCompareLines << QString();
            designCompareLines << fromUtf8StdString(designAlignmentResult.profileCsvText.substr(0, 4000));

            summaryLines << QString();
            summaryLines << QStringLiteral("设计母线比对");
            summaryLines << QStringLiteral("dz：%1 mm").arg(formatMetric(designSummaryPtr->dzMm, 4));
            summaryLines << QStringLiteral("dr：%1 mm").arg(formatMetric(designSummaryPtr->drMm, 4));
            summaryLines << QStringLiteral("有效点数：%1").arg(static_cast<qulonglong>(designSummaryPtr->usedCount));
            summaryLines << QStringLiteral("整体尺寸偏置：%1 um")
                                 .arg(formatMetric(designSummaryPtr->meanNormalErrorUm, 3));
            summaryLines << QStringLiteral("轮廓波动 RMS：%1 um")
                                 .arg(formatMetric(designSummaryPtr->profileStats.rmseUm, 3));
            summaryLines << QStringLiteral("轮廓波动 P95：%1 um")
                                 .arg(formatMetric(designSummaryPtr->profileStats.p95AbsUm, 3));
            summaryLines << QStringLiteral("轮廓度 PV：%1 um")
                                 .arg(formatMetric(designSummaryPtr->profileStats.pvUm, 3));
            if (!readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath).trimmed().isEmpty()) {
                summaryLines << QStringLiteral("自动质量审查：详见“质量审查”页签");
            }

            if (designCompareEdit_) {
                designCompareEdit_->setPlainText(designCompareLines.join('\n'));
            }
            if (designCompareViewer_) {
                designCompareViewer_->setImage(
                    cvMatToQImage(pinjie::cad_design::buildDesignComparisonPlot(designAlignmentResult)));
            }
        } else {
            summaryLines << QString();
            summaryLines << QStringLiteral("设计母线比对：%1")
                                .arg(fromUtf8StdString(designAlignmentResult.message));
            if (designCompareEdit_) {
                designCompareEdit_->setPlainText(QStringLiteral("设计母线比对未生成。\n%1")
                                                     .arg(fromUtf8StdString(designAlignmentResult.message)));
            }
            if (designCompareViewer_) {
                designCompareViewer_->clearImage();
            }
        }
    } else {
        if (designCompareEdit_) {
            designCompareEdit_->setPlainText(QStringLiteral("等待拼接结果生成后再进行设计母线比对。"));
        }
        if (designCompareViewer_) {
            designCompareViewer_->clearImage();
        }
    }

    summaryEdit_->setPlainText(summaryLines.join('\n'));
    bottomTabs_->setCurrentWidget(summaryEdit_);
    updateReportMetricCards(loadedImageCount, preprocessedImageCount, qualitySummaryPtr, designSummaryPtr);
    refreshRegistrationToolState();
    refreshReportExportState(ok ? QStringLiteral("归档状态：结果数据已就绪，请选择需要归档的 CSV 类型。")
                                : QStringLiteral("归档状态：运行失败，请先检查日志，再决定是否导出 CSV。"));
    refreshStageSummaries();

    if (currentRunMode_ == pinjie::StitchRunMode::Full || currentRunMode_ == pinjie::StitchRunMode::Report) {
        switchToStage(ReportStage);
    } else {
        switchToStage(stageIndexForRunMode(currentRunMode_));
    }
}

void MainWindow::onStepSelectionChanged()
{
    const QModelIndex index = stepTable_->currentIndex();
    if (!index.isValid()) {
        stepDetailPanel_->clearDetails();
        return;
    }

    const pinjie::StitchStepRecord* step = stepModel_->stepAt(index.row());
    if (!step) {
        stepDetailPanel_->clearDetails();
        return;
    }

    stepDetailPanel_->setStep(*step);
    loadRawImagesForStep(*step);

    const auto debugIt = debugImages_.constFind(static_cast<int>(step->stepIndex));
    if (debugIt != debugImages_.constEnd()) {
        debugViewer_->setImage(debugIt.value());
    }
}

void MainWindow::appendLog(const QString& message)
{
    if (!logEdit_ || message.isEmpty()) {
        return;
    }
    logEdit_->appendPlainText(message);
    statusBar()->showMessage(message, 3000);
}

void MainWindow::setUiRunning(bool running)
{
    if (calibrationPanel_) {
        calibrationPanel_->setRunning(running);
    }
    configPanel_->setRunning(running);
    if (startButton_) {
        startButton_->setEnabled(!running && hasActiveCalibration());
    }
    if (moduleRunButton_) {
        const int currentStage = stageStack_ ? stageStack_->currentIndex() : CalibrationStage;
        moduleRunButton_->setEnabled(!running && (currentStage == CalibrationStage || hasActiveCalibration()));
    }
    if (stopButton_) {
        stopButton_->setEnabled(running && currentTaskSupportsStop_);
    }
    for (auto* button : moduleButtons_) {
        if (button) {
            button->setEnabled(!running);
        }
    }
}

void MainWindow::cleanupWorker()
{
    calibrationWorker_ = nullptr;
    worker_ = nullptr;
    workerThread_ = nullptr;
    currentTaskSupportsStop_ = false;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setFormat(QStringLiteral("等待任务开始"));
    setUiRunning(false);
    refreshReportExportState();
    refreshCalibrationDetailPane();
    refreshRegistrationToolState();
    refreshStageSummaries();
}

void MainWindow::loadRawImagesForStep(const pinjie::StitchStepRecord& step)
{
    if (!runCache_ || !runCache_->hasLoadedImages()) {
        return;
    }

    const auto toZeroBased = [](std::size_t index) -> int {
        return index > 0 ? static_cast<int>(index - 1) : static_cast<int>(index);
    };

    const int refIndex = toZeroBased(step.referenceImageIndex);
    const int targetIndex = toZeroBased(step.targetImageIndex);

    if (refIndex >= 0 && refIndex < static_cast<int>(runCache_->loadedImages.size())) {
        referenceViewer_->setImage(cvMatToQImage(runCache_->loadedImages[refIndex]));
    }
    if (targetIndex >= 0 && targetIndex < static_cast<int>(runCache_->loadedImages.size())) {
        targetViewer_->setImage(cvMatToQImage(runCache_->loadedImages[targetIndex]));
    }
}

void MainWindow::onProcessingImageSelectionChanged()
{
    QListWidgetItem* currentItem = processingImageList_->currentItem();
    if (!currentItem) {
        processingPreviewViewer_->clearImage();
        processingPreviewInfoEdit_->setPlainText(QStringLiteral("等待预处理图像。"));
        return;
    }

    const int imageIndex = currentItem->data(Qt::UserRole).toInt();
    const auto imageIt = preprocessImages_.constFind(imageIndex);
    if (imageIt != preprocessImages_.constEnd()) {
        processingPreviewViewer_->setImage(imageIt.value());
    } else {
        processingPreviewViewer_->clearImage();
    }

    QStringList infoLines;
    infoLines << imageDisplayName(lastImagePaths_, imageIndex);

    if (imageIt != preprocessImages_.constEnd()) {
        const QImage& preview = imageIt.value();
        infoLines << QStringLiteral("预览尺寸：%1 x %2").arg(preview.width()).arg(preview.height());
    }

    if (imageIndex - 1 >= 0 && imageIndex - 1 < lastImagePaths_.size()) {
        const QString path = lastImagePaths_.at(imageIndex - 1);
        const QFileInfo fileInfo(path);
        infoLines << QStringLiteral("文件名：%1").arg(fileInfo.fileName());
        infoLines << QStringLiteral("文件路径：%1").arg(QDir::toNativeSeparators(path));
    }

    infoLines << QStringLiteral("左侧列表可切换结果。");
    processingPreviewInfoEdit_->setPlainText(infoLines.join('\n'));
}

void MainWindow::switchToStage(int stageIndex)
{
    if (!stageStack_) {
        return;
    }

    const int clamped = std::clamp(stageIndex, 0, stageStack_->count() - 1);
    stageStack_->setCurrentIndex(clamped);
    updateStageButtonStates(clamped);
    updateModuleRunButtonText(clamped);
}

void MainWindow::refreshStageSummaries()
{
    pinjie::CameraCalibrationRequest calibrationRequest;
    QString calibrationError;
    const bool calibrationConfigValid = calibrationPanel_ && calibrationPanel_->buildRequest(calibrationRequest, calibrationError);

    pinjie::StitchRunRequest request;
    QString errorMessage;
    const bool valid = configPanel_->buildRequest(request, errorMessage, false);

    if (!hasActiveCalibration()) {
        workflowStateLabel_->setText(QStringLiteral("流程：未加载标定结果，请先完成阶段 1 或加载缓存模型。"));
        registrationPresetLabel_->setText(calibrationConfigValid
                                              ? QStringLiteral("标定：参数已就绪")
                                              : QStringLiteral("标定：参数缺失"));
    } else {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("流程：%1 | %2 张图像可拼接")
                        .arg(calibrationIdentityLine(activeCalibrationCache_))
                        .arg(request.imagePaths.size())
                  : QStringLiteral("流程：%1 | 运行配置待补充")
                        .arg(calibrationIdentityLine(activeCalibrationCache_)));
        registrationPresetLabel_->setText(QStringLiteral("标定：%1").arg(calibrationQualityLine(activeCalibrationCache_)));
    }

    if (!calibrationConfigValid) {
        calibrationOverviewEdit_->setPlainText(QStringLiteral("配置检查：未通过\n原因：%1").arg(calibrationError));
    } else {
        QStringList calibrationLines;
        calibrationLines << QStringLiteral("配置检查：已就绪");
        calibrationLines << QString();
        calibrationLines << QStringLiteral("标定会话：%1").arg(fromUtf8StdString(calibrationRequest.sessionName));
        calibrationLines << QStringLiteral("图像目录：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(calibrationRequest.imageDirectory)));
        if (!calibrationRequest.cacheDir.empty()) {
            calibrationLines << QStringLiteral("缓存目录：%1")
                                    .arg(QDir::toNativeSeparators(fromUtf8StdString(calibrationRequest.cacheDir)));
        }
        if (!calibrationRequest.outputDir.empty()) {
            calibrationLines << QStringLiteral("输出目录：%1")
                                    .arg(QDir::toNativeSeparators(fromUtf8StdString(calibrationRequest.outputDir)));
        }
        calibrationLines << QStringLiteral("标定板规格：%1 x %2，间距 %3 mm")
                                .arg(calibrationRequest.boardSpec.rows)
                                .arg(calibrationRequest.boardSpec.cols)
                                .arg(calibrationRequest.boardSpec.pitchMm, 0, 'f', 4);
        calibrationLines << QStringLiteral("最少有效图像：%1").arg(calibrationRequest.boardSpec.minValidImages);
        calibrationLines << QStringLiteral("ROI 半径：%1 px").arg(calibrationRequest.boardSpec.roiRadiusPx);
        calibrationLines << QStringLiteral("优先复用缓存：%1")
                                .arg(calibrationRequest.preferCachedResult ? QStringLiteral("是") : QStringLiteral("否"));
        calibrationLines << QStringLiteral("写入标定缓存：%1")
                                .arg(calibrationRequest.persistCache ? QStringLiteral("是") : QStringLiteral("否"));
        calibrationOverviewEdit_->setPlainText(calibrationLines.join('\n'));
    }

    if (!valid) {
        acquisitionOverviewEdit_->setPlainText(QStringLiteral("运行配置检查：未通过\n原因：%1").arg(errorMessage));
        processingOverviewEdit_->setPlainText(QStringLiteral("等待图像输入与预处理参数就绪。"));
        registrationOverviewEdit_->setPlainText(QStringLiteral("等待配准参数与结果导出设置就绪。"));
        return;
    }

    QStringList acquisitionLines;
    acquisitionLines << QStringLiteral("配置检查：已就绪");
    acquisitionLines << QString();
    acquisitionLines << QStringLiteral("标定结果：%1")
                            .arg(hasActiveCalibration()
                                     ? fromUtf8StdString(activeCalibrationCache_.profile.profileName)
                                     : QStringLiteral("未加载"));
    acquisitionLines << QStringLiteral("输入图像：%1").arg(request.imagePaths.size());
    if (!request.imagePaths.empty()) {
        acquisitionLines << QStringLiteral("首张图像：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(request.imagePaths.front())));
    }
    acquisitionLines << QStringLiteral("全景图输出：%1").arg(displayOutputPath(request.panoramaOutputPath));
    acquisitionLines << QStringLiteral("汇总 CSV：%1").arg(displayOutputPath(request.csvOutputPath));
    acquisitionOverviewEdit_->setPlainText(acquisitionLines.join('\n'));

    const auto& edge = request.edgeConfig;
    QStringList processingLines;
    processingLines << QStringLiteral("预处理参数概览");
    processingLines << QString();
    processingLines << QStringLiteral("Canny 阈值：%1 / %2").arg(edge.cannyLow, 0, 'f', 1).arg(edge.cannyHigh, 0, 'f', 1);
    processingLines << QStringLiteral("亚像素窗口：%1").arg(edge.subpixWindow);
    processingLines << QStringLiteral("亚像素 Sigma：%1").arg(edge.subpixSigma, 0, 'f', 2);
    processingLines << QStringLiteral("点过滤：%1").arg(edge.enablePointFiltering ? QStringLiteral("开启") : QStringLiteral("关闭"));
    if (edge.enablePointFiltering) {
        processingLines << QStringLiteral("置信度分位数：%1%")
                                .arg(edge.filterConfidenceQuantile * 100.0, 0, 'f', 1);
        processingLines << QStringLiteral("梯度分位数：%1%")
                                .arg(edge.filterGradientQuantile * 100.0, 0, 'f', 1);
        processingLines << QStringLiteral("局部窗口半径：%1").arg(edge.filterLocalLinearWindowRadius);
        processingLines << QStringLiteral("Hampel Sigma：%1").arg(edge.filterHampelSigma, 0, 'f', 2);
        processingLines << QStringLiteral("最小尺度：%1 px").arg(edge.filterHampelMinScale, 0, 'f', 3);
    }
    processingOverviewEdit_->setPlainText(processingLines.join('\n'));

    const auto& pipeline = request.pipelineConfig;
    QStringList registrationLines;
    registrationLines << QStringLiteral("配准参数概览");
    registrationLines << QString();
    registrationLines << QStringLiteral("方向约束：%1").arg(directionLabel(pipeline.directionConstraint));
    registrationLines << QStringLiteral("预计重叠率：%1")
                             .arg(pipeline.expectedOverlapRatio > 0.0
                                      ? QStringLiteral("%1%").arg(pipeline.expectedOverlapRatio * 100.0, 0, 'f', 2)
                                      : QStringLiteral("自动估计"));
    registrationLines << QStringLiteral("基础搜索范围：%1 px").arg(pipeline.baseSearchRange, 0, 'f', 1);
    registrationLines << QStringLiteral("旋转范围：[%1, %2] deg")
                             .arg(pipeline.rotationSearchMinDeg, 0, 'f', 2)
                             .arg(pipeline.rotationSearchMaxDeg, 0, 'f', 2);
    registrationLines << QStringLiteral("旋转步长：%1 deg").arg(pipeline.rotationSearchStepDeg, 0, 'f', 3);
    registrationLines << QStringLiteral("切向残差权重：%1").arg(pipeline.tangentResidualCostWeight, 0, 'f', 3);
    registrationLines << QStringLiteral("切向相关权重：%1").arg(pipeline.tangentCorrelationCostWeight, 0, 'f', 3);
    registrationLines << QStringLiteral("调试可视化：%1")
                             .arg(pipeline.generateDebugVisualization ? QStringLiteral("开启") : QStringLiteral("关闭"));

    const QStringList csvLabels = selectedPaperCsvLabels(configPanel_);
    registrationLines << QStringLiteral("导出项目：%1")
                             .arg(csvLabels.isEmpty() ? QStringLiteral("未选择")
                                                      : csvLabels.join(QStringLiteral(", ")));
    registrationLines << QStringLiteral("结果目录：%1").arg(displayOutputPath(request.resultOutputDir));

    if (hasActiveCalibration()) {
        registrationLines << QString();
        registrationLines << QStringLiteral("标定质量：");
        registrationLines << calibrationQualityLine(activeCalibrationCache_);
    }

    registrationOverviewEdit_->setPlainText(registrationLines.join('\n'));
}

void MainWindow::refreshCalibrationDetailPane()
{
    if (!calibrationDetailEdit_) {
        return;
    }

    QStringList lines;
    if (!hasActiveCalibration()) {
        lines << QStringLiteral("尚未加载标定结果。");

        pinjie::CameraCalibrationRequest request;
        QString errorMessage;
        if (calibrationPanel_ && calibrationPanel_->buildRequest(request, errorMessage)) {
            lines << QString();
            lines << QStringLiteral("当前标定配置已就绪，可直接开始标定。");
            lines << QStringLiteral("会话名称：%1").arg(fromUtf8StdString(request.sessionName));
            lines << QStringLiteral("标定板：%1 x %2，间距 %3 mm")
                            .arg(request.boardSpec.rows)
                            .arg(request.boardSpec.cols)
                            .arg(request.boardSpec.pitchMm, 0, 'f', 4);
            lines << QStringLiteral("最少有效图像：%1").arg(request.boardSpec.minValidImages);
        } else if (!errorMessage.isEmpty()) {
            lines << QString();
            lines << QStringLiteral("当前标定配置未完成：%1").arg(errorMessage);
        }

        calibrationDetailEdit_->setPlainText(lines.join('\n'));
        return;
    }

    const auto& cache = activeCalibrationCache_;
    const auto& quality = cache.profile.quality;

    lines << QStringLiteral("标定结果：%1").arg(fromUtf8StdString(cache.profile.profileName));
    if (!cache.createdAt.empty()) {
        lines << QStringLiteral("创建时间：%1").arg(fromUtf8StdString(cache.createdAt));
    }
    if (!cache.profile.sourcePath.empty()) {
        lines << QStringLiteral("来源文件：%1")
                    .arg(QDir::toNativeSeparators(fromUtf8StdString(cache.profile.sourcePath)));
    }

    lines << QString();
    lines << QStringLiteral("有效图像：%1/%2").arg(quality.validImages).arg(quality.totalImages);
    if (quality.imageWidth > 0 && quality.imageHeight > 0) {
        lines << QStringLiteral("图像尺寸：%1 x %2").arg(quality.imageWidth).arg(quality.imageHeight);
    }
    if (quality.reprojectionRmsPx >= 0.0) {
        lines << QStringLiteral("重投影 RMS：%1 px").arg(formatMetric(quality.reprojectionRmsPx, 4));
    }
    if (quality.meanAffineRmsPx >= 0.0) {
        lines << QStringLiteral("平均单图 RMS：%1 px").arg(formatMetric(quality.meanAffineRmsPx, 4));
    }
    lines << QStringLiteral("fx / fy：%1 / %2 px/mm")
                .arg(formatMetric(quality.fxPixelsPerMm, 4))
                .arg(formatMetric(quality.fyPixelsPerMm, 4));
    lines << QStringLiteral("主点位置：%1, %2")
                .arg(formatMetric(quality.principalPointX, 3))
                .arg(formatMetric(quality.principalPointY, 3));
    lines << QStringLiteral("主点偏移：%1, %2")
                .arg(formatMetric(quality.principalPointOffsetX, 3, QStringLiteral(" px")))
                .arg(formatMetric(quality.principalPointOffsetY, 3, QStringLiteral(" px")));
    if (!quality.intrinsicModel.empty()) {
        lines << QStringLiteral("内参模型：%1").arg(fromUtf8StdString(quality.intrinsicModel));
    }
    if (!quality.principalPointPolicy.empty()) {
        lines << QStringLiteral("主点策略：%1").arg(fromUtf8StdString(quality.principalPointPolicy));
    }
    lines << QStringLiteral("去畸变映射：%1").arg(cache.hasUndistortMaps() ? QStringLiteral("已生成") : QStringLiteral("未生成"));

    if (quality.hasWarning()) {
        lines << QString();
        lines << QStringLiteral("质量警告：%1").arg(fromUtf8StdString(quality.warningSummary));
    }

    lines << QString();
    lines << QStringLiteral("输出目录：%1").arg(QDir::toNativeSeparators(fromUtf8StdString(cache.paths.outputDir)));
    if (!cache.paths.cacheDir.empty()) {
        lines << QStringLiteral("缓存目录：%1").arg(QDir::toNativeSeparators(fromUtf8StdString(cache.paths.cacheDir)));
    }
    if (!cache.paths.profilePath.empty()) {
        lines << QStringLiteral("Profile：%1").arg(QDir::toNativeSeparators(fromUtf8StdString(cache.paths.profilePath)));
    }
    if (!cache.paths.previewImagePath.empty()) {
        lines << QStringLiteral("Preview：%1").arg(QDir::toNativeSeparators(fromUtf8StdString(cache.paths.previewImagePath)));
    }

    calibrationDetailEdit_->setPlainText(lines.join('\n'));
}

bool MainWindow::hasActiveCalibration() const
{
    return activeCalibrationCache_.valid && activeCalibrationCache_.profile.valid;
}

void MainWindow::updateStageButtonStates(int stageIndex)
{
    for (int i = 0; i < static_cast<int>(moduleButtons_.size()); ++i) {
        if (moduleButtons_[i]) {
            moduleButtons_[i]->setChecked(i == stageIndex);
        }
    }
    updateWorkflowAccessState();
}

void MainWindow::updateWorkflowAccessState()
{
    const bool running = workerThread_ != nullptr;
    const int currentStage = stageStack_ ? stageStack_->currentIndex() : CalibrationStage;

    if (startButton_) {
        startButton_->setEnabled(!running && hasActiveCalibration());
    }
    if (moduleRunButton_) {
        moduleRunButton_->setEnabled(!running && (currentStage == CalibrationStage || hasActiveCalibration()));
    }
    if (stopButton_) {
        stopButton_->setEnabled(running && currentTaskSupportsStop_);
    }
    for (int i = 0; i < static_cast<int>(moduleButtons_.size()); ++i) {
        if (moduleButtons_[i]) {
            moduleButtons_[i]->setEnabled(!running);
        }
    }
}

void MainWindow::updateModuleRunButtonText(int stageIndex)
{
    if (!moduleRunButton_) {
        return;
    }

    if (stageIndex == CalibrationStage) {
        moduleRunButton_->setText(QStringLiteral("运行标定"));
        return;
    }

    if (!hasActiveCalibration()) {
        moduleRunButton_->setText(QStringLiteral("请先加载标定结果"));
        return;
    }

    moduleRunButton_->setText(QStringLiteral("运行%1").arg(stageTitle(stageIndex)));
}

void MainWindow::selectStepRow(int row)
{
    if (!stepModel_ || !stepTable_ || row < 0 || row >= stepModel_->rowCount()) {
        return;
    }

    const QModelIndex index = stepModel_->index(row, 0);
    if (!index.isValid()) {
        return;
    }

    stepTable_->setCurrentIndex(index);
    stepTable_->scrollTo(index, QAbstractItemView::PositionAtCenter);
}

void MainWindow::jumpToWorstStep()
{
    const int row = findWorstStepRow();
    if (row >= 0) {
        selectStepRow(row);
        switchToStage(RegistrationStage);
    }
}

void MainWindow::jumpToNextRiskStep()
{
    const int currentRow = stepTable_ ? stepTable_->currentIndex().row() : -1;
    const int row = findNextRiskStepRow(currentRow);
    if (row >= 0) {
        selectStepRow(row);
        switchToStage(RegistrationStage);
    }
}

int MainWindow::findWorstStepRow() const
{
    int bestRow = -1;
    double worstRmse = -1.0;
    for (int row = 0; row < static_cast<int>(completedSteps_.size()); ++row) {
        const auto& stats = preferredNormalStats(completedSteps_[row].transform.metrics);
        if (stats.valid() && stats.rmse > worstRmse) {
            worstRmse = stats.rmse;
            bestRow = row;
        }
    }
    return bestRow;
}

int MainWindow::findNextRiskStepRow(int startRow) const
{
    for (int row = std::max(startRow + 1, 0); row < static_cast<int>(completedSteps_.size()); ++row) {
        if (isRiskStep(completedSteps_[row])) {
            return row;
        }
    }
    return -1;
}

void MainWindow::refreshRegistrationToolState()
{
    const bool hasSteps = stepModel_ && stepModel_->rowCount() > 0;
    if (focusWorstStepButton_) {
        focusWorstStepButton_->setEnabled(hasSteps);
    }
    if (focusNextRiskStepButton_) {
        focusNextRiskStepButton_->setEnabled(hasSteps);
    }
}

void MainWindow::resetReportMetricCards()
{
    if (metricLoadedImagesValue_) {
        metricLoadedImagesValue_->setText(QStringLiteral("0"));
    }
    if (metricPreprocessedImagesValue_) {
        metricPreprocessedImagesValue_->setText(QStringLiteral("0"));
    }
    if (metricStepCountValue_) {
        metricStepCountValue_->setText(QStringLiteral("0"));
    }
    if (metricNormalRmseValue_) {
        metricNormalRmseValue_->setText(QStringLiteral("--"));
    }
    if (metricTangentRmseValue_) {
        metricTangentRmseValue_->setText(QStringLiteral("--"));
    }
    if (metricTangentCorrValue_) {
        metricTangentCorrValue_->setText(QStringLiteral("--"));
    }
    if (metricFlaggedStepsValue_) {
        metricFlaggedStepsValue_->setText(QStringLiteral("0"));
    }
    if (metricWorstStepValue_) {
        metricWorstStepValue_->setText(QStringLiteral("--"));
    }
    if (metricDesignRmseValue_) {
        metricDesignRmseValue_->setText(QStringLiteral("--"));
    }
    if (metricDesignP95Value_) {
        metricDesignP95Value_->setText(QStringLiteral("--"));
    }
    if (metricDesignOffsetValue_) {
        metricDesignOffsetValue_->setText(QStringLiteral("--"));
    }
    if (metricDesignUsedCountValue_) {
        metricDesignUsedCountValue_->setText(QStringLiteral("0"));
    }
}

void MainWindow::updateReportMetricCards(int loadedImageCount,
                                         int preprocessedImageCount,
                                         const pinjie::QualitySummary* qualitySummary,
                                         const pinjie::cad_design::DesignErrorSummary* designSummary)
{
    if (metricLoadedImagesValue_) {
        metricLoadedImagesValue_->setText(QString::number(std::max(loadedImageCount, 0)));
    }
    if (metricPreprocessedImagesValue_) {
        metricPreprocessedImagesValue_->setText(QString::number(std::max(preprocessedImageCount, 0)));
    }

    if (!qualitySummary) {
        if (metricStepCountValue_) {
            metricStepCountValue_->setText(QString::number(completedSteps_.size()));
        }
        if (metricNormalRmseValue_) {
            metricNormalRmseValue_->setText(QStringLiteral("--"));
        }
        if (metricTangentRmseValue_) {
            metricTangentRmseValue_->setText(QStringLiteral("--"));
        }
        if (metricTangentCorrValue_) {
            metricTangentCorrValue_->setText(QStringLiteral("--"));
        }
        if (metricFlaggedStepsValue_) {
            metricFlaggedStepsValue_->setText(QStringLiteral("0"));
        }
        if (metricWorstStepValue_) {
            metricWorstStepValue_->setText(QStringLiteral("--"));
        }
    } else {
        if (metricStepCountValue_) {
            metricStepCountValue_->setText(QString::number(qualitySummary->totalSteps));
        }
        if (metricNormalRmseValue_) {
            metricNormalRmseValue_->setText(formatMetric(qualitySummary->meanNormalRmse, 4, QStringLiteral(" px")));
        }
        if (metricTangentRmseValue_) {
            metricTangentRmseValue_->setText(formatMetric(qualitySummary->meanTangentRmse, 4, QStringLiteral(" px")));
        }
        if (metricTangentCorrValue_) {
            metricTangentCorrValue_->setText(formatMetric(qualitySummary->meanTangentCorr, 4));
        }
        if (metricFlaggedStepsValue_) {
            metricFlaggedStepsValue_->setText(QString::number(qualitySummary->flaggedStepCount));
        }
        if (metricWorstStepValue_) {
            metricWorstStepValue_->setText(qualitySummary->worstStepIndex >= 0
                                               ? QStringLiteral("第 %1 步").arg(qualitySummary->worstStepIndex)
                                               : QStringLiteral("--"));
        }
    }

    if (!designSummary) {
        if (metricDesignRmseValue_) {
            metricDesignRmseValue_->setText(QStringLiteral("--"));
        }
        if (metricDesignP95Value_) {
            metricDesignP95Value_->setText(QStringLiteral("--"));
        }
        if (metricDesignOffsetValue_) {
            metricDesignOffsetValue_->setText(QStringLiteral("--"));
        }
        if (metricDesignUsedCountValue_) {
            metricDesignUsedCountValue_->setText(QStringLiteral("0"));
        }
        return;
    }

    if (metricDesignRmseValue_) {
        metricDesignRmseValue_->setText(
            formatMetric(designSummary->profileStats.rmseUm, 3, QStringLiteral(" um")));
    }
    if (metricDesignP95Value_) {
        metricDesignP95Value_->setText(
            formatMetric(designSummary->profileStats.p95AbsUm, 3, QStringLiteral(" um")));
    }
    if (metricDesignOffsetValue_) {
        metricDesignOffsetValue_->setText(
            QStringLiteral("bias %1\nPV %2")
                .arg(formatMetric(designSummary->meanNormalErrorUm, 3, QStringLiteral(" um")),
                     formatMetric(designSummary->profileStats.pvUm, 3, QStringLiteral(" um"))));
    }
    if (metricDesignUsedCountValue_) {
        metricDesignUsedCountValue_->setText(QString::number(static_cast<qulonglong>(designSummary->usedCount)));
    }
}

void MainWindow::exportSelectedReportCsvs()
{
    if (!runCache_ || !runCache_->hasStitching() || !runCache_->hasPreprocessedEdges()) {
        QMessageBox::information(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("请先运行拼接，再导出 CSV 文件。"));
        refreshReportExportState(QStringLiteral("当前没有可导出的拼接结果"));
        return;
    }

    if (lastRequest_.resultOutputDir.empty()) {
        QMessageBox::warning(this,
                             QStringLiteral("导出 CSV"),
                             QStringLiteral("结果输出目录不可用。"));
        refreshReportExportState(QStringLiteral("结果输出目录不可用"));
        return;
    }

    const bool saveStepSummary = configPanel_ && configPanel_->saveStepSummaryCsv();
    const bool saveContourPoints = configPanel_ && configPanel_->saveContourPointsCsv();
    const bool saveStitchedContourProfile = configPanel_ && configPanel_->saveStitchedContourProfileCsv();
    const bool saveTangentStep = configPanel_ && configPanel_->saveTangentStepCsv();
    const bool saveNormalError = configPanel_ && configPanel_->saveNormalErrorProfileCsv();
    const bool saveTangentProfile = configPanel_ && configPanel_->saveTangentProfileCsv();
    const bool saveCandidateDiagnostics = configPanel_ && configPanel_->saveAlignmentCandidateDiagnosticsCsv();

    if (!saveStepSummary && !saveContourPoints && !saveStitchedContourProfile && !saveTangentStep &&
        !saveNormalError && !saveTangentProfile && !saveCandidateDiagnostics) {
        QMessageBox::information(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("导出前请至少选择一种 CSV 类型。"));
        refreshReportExportState(QStringLiteral("未选择 CSV 类型"));
        return;
    }

    const std::filesystem::path baseDir = std::filesystem::u8path(lastRequest_.resultOutputDir);
    QStringList exportedFiles;

    if (saveStepSummary) {
        const std::filesystem::path path = baseDir / "stitching_data.csv";
        if (!stitch::writeTextFileToPath(path.generic_u8string(), runCache_->csvText)) {
            QMessageBox::warning(this, QStringLiteral("导出 CSV"), QStringLiteral("保存 stitching_data.csv 失败。"));
            refreshReportExportState(QStringLiteral("拼接汇总 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(path.generic_u8string()));
    }

    if (saveContourPoints) {
        const std::filesystem::path detailPath = baseDir / "contour_points.csv";
        const std::string detailContent =
            stitch::buildContourPointCsv(runCache_->preprocessedEdges, runCache_->stitching.imageTransforms);
        if (!stitch::writeTextFileToPath(detailPath.generic_u8string(), detailContent)) {
            QMessageBox::warning(this, QStringLiteral("导出 CSV"), QStringLiteral("保存 contour_points.csv 失败。"));
            refreshReportExportState(QStringLiteral("轮廓点 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(detailPath.generic_u8string()));

        const std::filesystem::path originPath = baseDir / "origin_contour_overlay_points.csv";
        const std::string originContent =
            stitch::buildOriginContourOverlayCsv(runCache_->preprocessedEdges, runCache_->stitching.imageTransforms);
        if (!stitch::writeTextFileToPath(originPath.generic_u8string(), originContent)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 origin_contour_overlay_points.csv 失败。"));
            refreshReportExportState(QStringLiteral("Origin 轮廓叠加 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(originPath.generic_u8string()));
    }

    if (saveStitchedContourProfile) {
        const std::filesystem::path path = baseDir / "stitched_contour_profile.csv";
        const std::string content = stitch::buildStitchedContourProfileCsv(runCache_->preprocessedEdges,
                                                                           runCache_->stitching.imageTransforms);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 stitched_contour_profile.csv 失败。"));
            refreshReportExportState(QStringLiteral("整体轮廓剖面 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(path.generic_u8string()));
    }

    if (saveTangentStep) {
        const std::filesystem::path path = baseDir / "tangent_correlation_by_step.csv";
        const std::string content = stitch::buildTangentCorrelationStepCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 tangent_correlation_by_step.csv 失败。"));
            refreshReportExportState(QStringLiteral("切向相关性 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(path.generic_u8string()));
    }

    if (saveNormalError) {
        const std::filesystem::path path = baseDir / "normal_error_profile.csv";
        const std::string content = stitch::buildNormalErrorProfileCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 normal_error_profile.csv 失败。"));
            refreshReportExportState(QStringLiteral("法向误差剖面 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(path.generic_u8string()));
    }

    if (saveTangentProfile) {
        const std::filesystem::path detailPath = baseDir / "tangent_profile_compare.csv";
        const std::string detailContent = stitch::buildTangentProfileCompareCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(detailPath.generic_u8string(), detailContent)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 tangent_profile_compare.csv 失败。"));
            refreshReportExportState(QStringLiteral("轮廓波动分析 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(detailPath.generic_u8string()));

        const std::filesystem::path originPath = baseDir / "origin_tangent_point_metrics.csv";
        const std::string originContent = stitch::buildOriginTangentPointMetricsCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(originPath.generic_u8string(), originContent)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 origin_tangent_point_metrics.csv 失败。"));
            refreshReportExportState(QStringLiteral("Origin 轮廓波动 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(originPath.generic_u8string()));
    }

    if (saveCandidateDiagnostics) {
        const std::filesystem::path path = baseDir / "alignment_candidate_diagnostics.csv";
        const std::string content = stitch::buildAlignmentCandidateDiagnosticsCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 alignment_candidate_diagnostics.csv 失败。"));
            refreshReportExportState(QStringLiteral("候选诊断 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(path.generic_u8string()));
    }

    appendLog(QStringLiteral("[信息] 已导出 CSV 文件：\n%1").arg(exportedFiles.join('\n')));
    refreshReportExportState(QStringLiteral("已导出 %1 个 CSV 文件").arg(exportedFiles.size()));
    statusBar()->showMessage(QStringLiteral("已导出 %1 个 CSV 文件").arg(exportedFiles.size()));
}

void MainWindow::generatePublicationFigure()
{
    if (!runCache_ || !runCache_->hasStitching()) {
        QMessageBox::information(this,
                                 QStringLiteral("生成误差分析图"),
                                 QStringLiteral("请先运行拼接，再生成误差分析图。"));
        return;
    }
    if (lastRequest_.resultOutputDir.empty()) {
        QMessageBox::warning(this,
                             QStringLiteral("生成误差分析图"),
                             QStringLiteral("结果输出目录不可用。"));
        return;
    }

    const std::filesystem::path projectRoot = pinjie::projectRootPath();
    const std::filesystem::path scriptPath = projectRoot / "report" / "figure_export" / "publication_figure.py";
    if (!std::filesystem::exists(scriptPath)) {
        QMessageBox::warning(this,
                             QStringLiteral("生成误差分析图"),
                             QStringLiteral("未找到 publication_figure.py。"));
        return;
    }

    if (generateFigureButton_) {
        generateFigureButton_->setEnabled(false);
    }
    statusBar()->showMessage(QStringLiteral("正在生成误差分析图..."));
    appendLog(QStringLiteral("[信息] 开始生成误差分析图：%1")
                  .arg(QDir::toNativeSeparators(QString::fromStdString(scriptPath.u8string()))));

    QProcess process(this);
    process.setWorkingDirectory(QString::fromStdString(projectRoot.u8string()));
    process.setProgram(QStringLiteral("python"));
    process.setArguments({
        QString::fromStdString(scriptPath.u8string()),
        QStringLiteral("--result-dir"),
        QDir::fromNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir)),
    });
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start();
    if (!process.waitForStarted(5000)) {
        if (generateFigureButton_) {
            generateFigureButton_->setEnabled(true);
        }
        QMessageBox::warning(this,
                             QStringLiteral("生成误差分析图"),
                             QStringLiteral("无法启动 Python。请确认 `python` 在 PATH 中。"));
        return;
    }

    process.waitForFinished(-1);
    const QString output = QString::fromUtf8(process.readAllStandardOutput());
    if (!output.trimmed().isEmpty()) {
        appendLog(QStringLiteral("[图形脚本]\n%1").arg(output.trimmed()));
    }

    if (generateFigureButton_) {
        generateFigureButton_->setEnabled(true);
    }

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        QMessageBox::warning(this,
                             QStringLiteral("生成误差分析图"),
                             QStringLiteral("误差分析图脚本执行失败，请检查日志和 Python 依赖。"));
        statusBar()->showMessage(QStringLiteral("误差分析图生成失败"));
        return;
    }

    refreshPublicationFigurePreview();
    reportViewTabs_->setCurrentWidget(publicationFigureViewer_);
    statusBar()->showMessage(QStringLiteral("误差分析图已生成"));
}

void MainWindow::refreshPublicationFigurePreview()
{
    if (!publicationFigureViewer_) {
        return;
    }

    const QString path = publicationFigurePngPath(lastRequest_);
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        publicationFigureViewer_->clearImage();
        return;
    }

    const QImage image(path);
    if (image.isNull()) {
        publicationFigureViewer_->clearImage();
        appendLog(QStringLiteral("[警告] 误差分析图存在但无法读取：%1").arg(path));
        return;
    }
    publicationFigureViewer_->setImage(image);
}

void MainWindow::refreshReportExportState(const QString& statusMessage)
{
    if (generateFigureButton_) {
        generateFigureButton_->setEnabled(!workerThread_ && runCache_ && runCache_->hasStitching());
    }
    if (exportSelectedCsvButton_) {
        exportSelectedCsvButton_->setEnabled(!workerThread_ && runCache_ && runCache_->hasStitching());
    }

    if (!reportExportStatusLabel_) {
        return;
    }

    if (!statusMessage.isEmpty()) {
        reportExportStatusLabel_->setText(statusMessage);
        return;
    }

    const QStringList labels = selectedPaperCsvLabels(configPanel_);
    if (labels.isEmpty()) {
        reportExportStatusLabel_->setText(
            QStringLiteral("归档状态：结果已生成，请先勾选需要导出的 CSV 类型 | 自动生成：设计比对、质量审查、候选诊断"));
        return;
    }

    const QString targetDir =
        lastRequest_.resultOutputDir.empty()
            ? QStringLiteral("当前结果目录")
            : QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir));
    reportExportStatusLabel_->setText(
        QStringLiteral("归档状态：可导出 CSV：%1 | 输出目录：%2 | 自动生成：设计比对、质量审查、候选诊断")
            .arg(labels.join(QStringLiteral(", ")), targetDir));
}

void MainWindow::applyWindowStyle()
{
    setStyleSheet(QStringLiteral(R"(
QMainWindow {
    background: #eef3f7;
}
QWidget#appRoot {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7fafc, stop:1 #edf3f7);
}
QWidget {
    font-size: 12px;
    color: #102033;
}
QFrame#headerPanel,
QFrame#workflowPanel,
QFrame#reportToolbarFrame,
QFrame#reportSidebarActionsFrame,
QWidget#reportMetricRail {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d8e2ec;
    border-radius: 8px;
}
QFrame#statusPanel {
    background: transparent;
}
QWidget#reportMetricRail {
    background: #f7faf8;
}
QFrame#reportSidebarActionsFrame {
    background: #ffffff;
}
QLabel#panelTitleLabel {
    color: #12263a;
    font-size: 15px;
    font-weight: 700;
}
QLabel#pageTitleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #102033;
}
QLabel#workflowStateLabel {
    padding: 2px 6px;
    border-radius: 5px;
    background: #f5f9fd;
    border: 1px solid #d7e6f3;
    border-left: 4px solid #1f5f8b;
    color: #163754;
}
QLabel#registrationPresetLabel {
    padding: 2px 6px;
    border-radius: 5px;
    background: #f8fbf5;
    border: 1px solid #d8e8da;
    border-left: 4px solid #2f6c4f;
    color: #214b38;
}
QPushButton {
    min-height: 22px;
    padding: 2px 8px;
    border-radius: 5px;
    border: 1px solid #c6d2de;
    background: #ffffff;
    color: #102033;
    font-weight: 600;
}
QPushButton:hover {
    background: #f6fafc;
    border-color: #8ea7be;
}
QPushButton:disabled {
    background: #f7fafc;
    color: #90a1b2;
    border-color: #dde6ee;
}
QPushButton#primaryActionButton {
    background: #1f5f8b;
    border-color: #184e74;
    color: #ffffff;
}
QPushButton#primaryActionButton:hover {
    background: #184e74;
}
QPushButton#secondaryActionButton {
    background: #f0f6fb;
    border-color: #c7dae9;
    color: #1f5f8b;
}
QPushButton#secondaryActionButton:hover {
    background: #e4f0f8;
}
QPushButton#dangerActionButton {
    background: #fff5f5;
    border-color: #f3c7ca;
    color: #a63d45;
}
QPushButton#dangerActionButton:hover {
    background: #fdebec;
}
QPushButton[stage="calibration"] {
    background: #f7f6fb;
    border-color: #d7d4e8;
    color: #4e4a72;
}
QPushButton[stage="calibration"]:checked {
    background: #4e4a72;
    border-color: #4e4a72;
    color: #ffffff;
}
QPushButton[stage="acquisition"] {
    background: #f3f8fc;
    border-color: #d1dfeb;
    color: #265779;
}
QPushButton[stage="acquisition"]:checked {
    background: #265779;
    border-color: #265779;
    color: #ffffff;
}
QPushButton[stage="processing"] {
    background: #f2faf9;
    border-color: #d0e5e1;
    color: #24655f;
}
QPushButton[stage="processing"]:checked {
    background: #24655f;
    border-color: #24655f;
    color: #ffffff;
}
QPushButton[stage="registration"] {
    background: #fcf7f0;
    border-color: #eadcc9;
    color: #8b6132;
}
QPushButton[stage="registration"]:checked {
    background: #8b6132;
    border-color: #8b6132;
    color: #ffffff;
}
QPushButton[stage="report"] {
    background: #f4faf5;
    border-color: #d4e6d7;
    color: #36684a;
}
QPushButton[stage="report"]:checked {
    background: #36684a;
    border-color: #36684a;
    color: #ffffff;
}
QProgressBar#workflowProgressBar {
    min-height: 6px;
    max-height: 6px;
    border-radius: 3px;
    border: 1px solid #c9d7e2;
    background: #ffffff;
    text-align: center;
    color: #102033;
}
QProgressBar#workflowProgressBar::chunk {
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f5f8b, stop:1 #2c8c84);
}
QPlainTextEdit, QListWidget, QTableView, QTabWidget::pane, QScrollArea, QGraphicsView {
    background: #ffffff;
    border: 1px solid #d7e0e8;
    border-radius: 6px;
}
QPlainTextEdit, QListWidget, QTableView {
    selection-background-color: #dce9f3;
    selection-color: #102033;
    alternate-background-color: #fbfdff;
}
QGraphicsView {
    background: #1f2832;
    border: 1px solid #304252;
}
QTabWidget::pane {
    margin-top: 4px;
}
QTabBar::tab {
    background: #f5f8fb;
    border: 1px solid #d7e0e8;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 3px 7px;
    margin-right: 3px;
    color: #53677b;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1f5f8b;
}
QHeaderView::section {
    background: #f4f8fb;
    color: #56697c;
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid #d7e0e8;
}
QListWidget#previewImageList::item {
    padding: 5px 8px;
    border-bottom: 1px solid #edf2f6;
}
QListWidget#previewImageList::item:selected {
    background: #e6eff6;
    color: #1b4667;
}
QTableView#stepTable {
    gridline-color: #e1e8ef;
}
QTableView#stepTable::item {
    padding: 4px;
}
QFrame#metricCard {
    border: 1px solid #d7e2eb;
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f7fafc);
}
QFrame#metricCard[compact="true"] {
    min-height: 48px;
    background: #ffffff;
}
QFrame#stepMetricCard {
    border: 1px solid #dbe3ea;
    border-radius: 12px;
    background: #fbfdfe;
}
QLabel#metricCardTitle {
    color: #64778a;
    font-size: 11px;
}
QLabel#metricCardValue {
    font-size: 18px;
    font-weight: 700;
    color: #102033;
}
QFrame#metricCard[compact="true"] QLabel#metricCardTitle {
    font-size: 11px;
}
QFrame#metricCard[compact="true"] QLabel#metricCardValue {
    font-size: 17px;
}
QLabel#stepMetricTitle {
    color: #64778a;
    font-size: 11px;
}
QLabel#stepMetricValue {
    color: #102033;
    font-size: 12px;
    font-weight: 600;
}
QLabel#reportExportStatusLabel {
    padding: 4px 8px;
    border-radius: 6px;
    background: #f6fbf7;
    border: 1px solid #d7e7d9;
    color: #244a34;
}
QLabel#reportRailTitleLabel {
    color: #244a34;
    font-size: 13px;
    font-weight: 700;
}
QWidget#registrationSidebarPanel,
QWidget#registrationDiagnosticsPanel,
QWidget#stepDetailPanel {
    background: transparent;
}
QToolButton#configSectionToggle {
    text-align: left;
    padding: 3px 7px;
    border-radius: 5px;
    border: 1px solid #dbe2ea;
    background: #ffffff;
    font-weight: 600;
}
QToolButton#configSectionToggle:hover {
    background: #f6f9fb;
}
QToolButton#paneToggleButton {
    min-height: 20px;
    padding: 1px 7px;
    border-radius: 6px;
    border: 1px solid #cbd8e3;
    background: #ffffff;
    color: #34536c;
    font-weight: 600;
}
QToolButton#paneToggleButton:hover {
    background: #f3f8fb;
    border-color: #9fb7ca;
}
QToolButton#paneToggleButton:checked {
    background: #e8f2f8;
    border-color: #8db1cb;
    color: #1f5f8b;
}
QWidget#calibrationConfigSection QToolButton#configSectionToggle,
QWidget#calibrationConfigSection QGroupBox {
    color: #4e4a72;
}
QWidget#acquisitionConfigSection QToolButton#configSectionToggle,
QWidget#acquisitionConfigSection QGroupBox {
    color: #265779;
}
QWidget#processingConfigSection QToolButton#configSectionToggle,
QWidget#processingConfigSection QGroupBox {
    color: #24655f;
}
QWidget#registrationConfigSection QToolButton#configSectionToggle,
QWidget#registrationConfigSection QGroupBox {
    color: #8b6132;
}
QWidget#reportConfigSection QToolButton#configSectionToggle,
QWidget#reportConfigSection QGroupBox {
    color: #36684a;
}
QGroupBox {
    margin-top: 5px;
    padding: 8px 7px 7px 7px;
    border-radius: 6px;
    border: 1px solid #dce4eb;
    background: rgba(255, 255, 255, 0.94);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background: transparent;
}
QWidget#calibrationConfigSection QGroupBox {
    border-left: 4px solid #6a648f;
    background: #faf9fd;
}
QWidget#acquisitionConfigSection QGroupBox {
    border-left: 4px solid #39688a;
    background: #f7fbfd;
}
QWidget#processingConfigSection QGroupBox {
    border-left: 4px solid #2d6c66;
    background: #f7fbfa;
}
QWidget#registrationConfigSection QGroupBox {
    border-left: 4px solid #9a6f3d;
    background: #fdfaf6;
}
QWidget#reportConfigSection QGroupBox {
    border-left: 4px solid #4c7a5b;
    background: #f7fbf8;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 22px;
    padding: 2px 6px;
    border-radius: 5px;
    border: 1px solid #c9d5df;
    background: #ffffff;
}
QLabel {
    color: #102033;
}
QSplitter::handle {
    background: transparent;
}
QSplitter::handle:hover {
    background: rgba(114, 138, 160, 0.12);
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #dbe4eb;
}
)"));
}

} // namespace pinjie::gui


