#include "gui/MainWindow.h"

#include "common/ResultPathUtils.h"
#include "gui/QtImageUtils.h"
#include "gui/PointCloud3DViewer.h"
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
#include <QRegularExpression>
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
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace pinjie::gui {

namespace {

enum WorkflowStageIndex {
    CalibrationStage = 0,
    AcquisitionStage = 1,
    ProcessingStage = 2,
    RegistrationStage = 3,
    ReportStage = 4,
    CompensationStage = 5
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
constexpr int kReportPanoramaTabIndex = 0;
constexpr int kReportSecondaryImageTabIndex = 1;
constexpr int kBottomSummaryTabIndex = 0;
constexpr int kBottomCoverageTabIndex = 1;
constexpr int kBottomPointDetailTabIndex = 2;
constexpr int kBottomDiagnosticTabIndex = 3;
constexpr int kBottomCsvTabIndex = 4;
constexpr int kBottomAcceptanceTabIndex = 5;

bool isStandardCircleMode(const pinjie::StitchRunRequest& request)
{
    return request.standardCircleConfig.enabled;
}

constexpr double kLegacySingleProfileSyntheticPixelSizeMm = 0.05;
constexpr double kCadSectionSingleSlotPixelSizeMm = 0.0476190476;

double singleSlotTestPixelSizeMm(const pinjie::StitchRunRequest& request)
{
    for (const std::string& imagePath : request.imagePaths) {
        std::string lowerPath = imagePath;
        std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (lowerPath.find("profile_2040_cad_section_input") != std::string::npos ||
            lowerPath.find("profile_2040_cad_section") != std::string::npos) {
            return kCadSectionSingleSlotPixelSizeMm;
        }
        if (lowerPath.find("single_profile_synthetic") != std::string::npos ||
            lowerPath.find("profile_2040_single") != std::string::npos) {
            return kLegacySingleProfileSyntheticPixelSizeMm;
        }
    }
    return 0.0;
}

bool isSingleSlotTestInput(const pinjie::StitchRunRequest& request)
{
    return singleSlotTestPixelSizeMm(request) > 0.0;
}

QString metricKey(const QString& metric, const QString& unit)
{
    return metric + QStringLiteral("|") + unit;
}

QHash<QString, QString> parseMetricCsvByKey(const QString& csvText)
{
    QHash<QString, QString> metrics;
    const QString normalized = csvText;
    const QStringList lines = normalized.split(QRegularExpression(QStringLiteral("[\r\n]+")),
                                               Qt::SkipEmptyParts);
    for (int i = 1; i < lines.size(); ++i) {
        const QString& line = lines.at(i);
        const int firstComma = line.indexOf(',');
        if (firstComma < 0) {
            continue;
        }
        const int secondComma = line.indexOf(',', firstComma + 1);
        if (secondComma < 0) {
            continue;
        }
        const int thirdComma = line.indexOf(',', secondComma + 1);
        if (thirdComma < 0) {
            continue;
        }

        const QString metric = line.left(firstComma).trimmed();
        const QString value =
            line.mid(firstComma + 1, secondComma - firstComma - 1).trimmed();
        const QString unit =
            line.mid(secondComma + 1, thirdComma - secondComma - 1).trimmed();
        if (!metric.isEmpty()) {
            metrics.insert(metricKey(metric, unit), value);
            if (!metrics.contains(metricKey(metric, QString()))) {
                metrics.insert(metricKey(metric, QString()), value);
            }
        }
    }
    return metrics;
}

QString metricValue(const QHash<QString, QString>& metrics,
                    const QString& metric,
                    const QString& unit = QString())
{
    return metrics.value(metricKey(metric, unit));
}

QString cleanCsvCell(QString value)
{
    value = value.trimmed();
    if (value.size() >= 2 && value.startsWith('"') && value.endsWith('"')) {
        value = value.mid(1, value.size() - 2);
        value.replace(QStringLiteral("\"\""), QStringLiteral("\""));
    }
    return value;
}

QHash<QString, QString> parseFirstCsvDataRow(const QString& csvText)
{
    QHash<QString, QString> values;
    const QStringList lines = csvText.split(QRegularExpression(QStringLiteral("[\r\n]+")),
                                            Qt::SkipEmptyParts);
    if (lines.size() < 2) {
        return values;
    }
    const QStringList headers = lines.at(0).split(',');
    const QStringList cells = lines.at(1).split(',');
    const int count = std::min(headers.size(), cells.size());
    for (int index = 0; index < count; ++index) {
        const QString key = cleanCsvCell(headers.at(index));
        if (!key.isEmpty()) {
            values.insert(key, cleanCsvCell(cells.at(index)));
        }
    }
    return values;
}

void setReportTabTexts(QTabWidget* reportTabs, QTabWidget* bottomTabs, const bool standardCircleMode)
{
    if (reportTabs) {
        reportTabs->setTabText(kReportPanoramaTabIndex,
                               standardCircleMode ? QStringLiteral("测量窗口图")
                                                  : QStringLiteral("全景结果"));
        reportTabs->setTabText(kReportSecondaryImageTabIndex,
                                standardCircleMode ? QStringLiteral("边缘叠加图")
                                                   : QStringLiteral("误差分析图"));
    }
    if (bottomTabs) {
        bottomTabs->setTabText(kBottomSummaryTabIndex, QStringLiteral("复检总览"));
        bottomTabs->setTabText(kBottomCoverageTabIndex,
                               standardCircleMode ? QStringLiteral("候选覆盖")
                                                  : QStringLiteral("质量审查"));
        bottomTabs->setTabText(kBottomPointDetailTabIndex,
                               standardCircleMode ? QStringLiteral("25点详情")
                                                  : QStringLiteral("CAD比对"));
        bottomTabs->setTabText(kBottomDiagnosticTabIndex,
                               standardCircleMode ? QStringLiteral("掩模诊断")
                                                  : QStringLiteral("候选诊断"));
        bottomTabs->setTabText(kBottomCsvTabIndex,
                               standardCircleMode ? QStringLiteral("摘要 CSV")
                                                  : QStringLiteral("CSV 数据"));
        bottomTabs->setTabText(kBottomAcceptanceTabIndex,
                               standardCircleMode ? QStringLiteral("国标复检清单")
                                                  : QStringLiteral("表9验收对照"));
    }
}

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
    if (stage == QStringLiteral("report")) {
        return QStringLiteral("结果导出");
    }
    return stage;
}

QString stageTitle(int stageIndex)
{
    switch (stageIndex) {
    case CalibrationStage:
        return QStringLiteral("CAD/目标尺寸");
    case AcquisitionStage:
        return QStringLiteral("数据接收预处理");
    case ProcessingStage:
        return QStringLiteral("特征提取");
    case RegistrationStage:
        return QStringLiteral("模型配准");
    case ReportStage:
        return QStringLiteral("误差分析");
    case CompensationStage:
    default:
        return QStringLiteral("补偿解算");
    }
}

QString runModeLabel(const pinjie::StitchRunMode runMode)
{
    switch (runMode) {
    case pinjie::StitchRunMode::Acquisition:
        return QStringLiteral("数据接收预处理");
    case pinjie::StitchRunMode::Processing:
        return QStringLiteral("特征提取");
    case pinjie::StitchRunMode::Registration:
        return QStringLiteral("模型配准");
    case pinjie::StitchRunMode::Report:
        return QStringLiteral("误差分析");
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
    case CompensationStage:
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

bool sameDesignProfileMetadata(const pinjie::cad_design::DesignProfileMetadata& lhs,
                               const pinjie::cad_design::DesignProfileMetadata& rhs)
{
    return lhs.sourceType == rhs.sourceType &&
           lhs.sourceName == rhs.sourceName &&
           lhs.sourcePath == rhs.sourcePath &&
           lhs.extractionMethod == rhs.extractionMethod &&
           lhs.axialAxis == rhs.axialAxis &&
           lhs.radialAxis == rhs.radialAxis &&
           lhs.sectionNormalAxis == rhs.sectionNormalAxis &&
           nearlyEqual(lhs.sectionCoordinateMm, rhs.sectionCoordinateMm) &&
           nearlyEqual(lhs.cadAxialOriginMm, rhs.cadAxialOriginMm) &&
           nearlyEqual(lhs.cadAxialDirectionSign, rhs.cadAxialDirectionSign) &&
           lhs.sampleCount == rhs.sampleCount &&
           nearlyEqual(lhs.minSMm, rhs.minSMm) &&
           nearlyEqual(lhs.maxSMm, rhs.maxSMm) &&
           nearlyEqual(lhs.minRMm, rhs.minRMm) &&
           nearlyEqual(lhs.maxRMm, rhs.maxRMm) &&
           lhs.hasCadBounds == rhs.hasCadBounds &&
           nearlyEqual(lhs.minCadXMm, rhs.minCadXMm) &&
           nearlyEqual(lhs.minCadYMm, rhs.minCadYMm) &&
           nearlyEqual(lhs.minCadZMm, rhs.minCadZMm) &&
           nearlyEqual(lhs.maxCadXMm, rhs.maxCadXMm) &&
           nearlyEqual(lhs.maxCadYMm, rhs.maxCadYMm) &&
           nearlyEqual(lhs.maxCadZMm, rhs.maxCadZMm);
}

bool sameDesignProfileSamples(const std::vector<pinjie::cad_design::DesignProfileSample>& lhs,
                              const std::vector<pinjie::cad_design::DesignProfileSample>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!nearlyEqual(lhs[i].sMm, rhs[i].sMm) ||
            !nearlyEqual(lhs[i].rMm, rhs[i].rMm) ||
            lhs[i].hasCadPoint != rhs[i].hasCadPoint ||
            !nearlyEqual(lhs[i].cadXMm, rhs[i].cadXMm) ||
            !nearlyEqual(lhs[i].cadYMm, rhs[i].cadYMm) ||
            !nearlyEqual(lhs[i].cadZMm, rhs[i].cadZMm)) {
            return false;
        }
    }
    return true;
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
           lhs.designReverseAxial == rhs.designReverseAxial &&
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
            nearlyEqual(lhs.designTrimRightMm, rhs.designTrimRightMm) &&
            lhs.designUseExternalProfile == rhs.designUseExternalProfile &&
            nearlyEqual(lhs.designTargetSlotWidthMm, rhs.designTargetSlotWidthMm) &&
            nearlyEqual(lhs.designTargetSlotDepthMm, rhs.designTargetSlotDepthMm) &&
            lhs.localSlotImageMode == rhs.localSlotImageMode &&
            nearlyEqual(lhs.localSlotBottomWidthMm, rhs.localSlotBottomWidthMm) &&
            nearlyEqual(lhs.localSlotPixelSizeOverrideMm, rhs.localSlotPixelSizeOverrideMm) &&
            nearlyEqual(lhs.localSlotPixelSizeScale, rhs.localSlotPixelSizeScale) &&
            lhs.localSlotMaxOutputPoints == rhs.localSlotMaxOutputPoints &&
            lhs.designUseCentralSlotImageRoi == rhs.designUseCentralSlotImageRoi &&
            nearlyEqual(lhs.designImageRoiXRatio, rhs.designImageRoiXRatio) &&
            nearlyEqual(lhs.designImageRoiYRatio, rhs.designImageRoiYRatio) &&
            nearlyEqual(lhs.designImageRoiWidthRatio, rhs.designImageRoiWidthRatio) &&
            nearlyEqual(lhs.designImageRoiHeightRatio, rhs.designImageRoiHeightRatio) &&
            sameDesignProfileMetadata(lhs.designProfileMetadata, rhs.designProfileMetadata) &&
            sameDesignProfileSamples(lhs.designExternalProfileSamples, rhs.designExternalProfileSamples);
}

bool canReuseOutputPaths(const pinjie::StitchRunRequest& previousRequest,
                         const pinjie::StitchRunRequest& currentRequest)
{
    if (isStandardCircleMode(previousRequest) || isStandardCircleMode(currentRequest)) {
        return false;
    }
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
    target.design3dErrorCsvOutputPath = source.design3dErrorCsvOutputPath;
    target.designCompensationCsvOutputPath = source.designCompensationCsvOutputPath;
    target.designFeatureCompensationCsvOutputPath = source.designFeatureCompensationCsvOutputPath;
    target.designComparisonOverlayOutputPath = source.designComparisonOverlayOutputPath;
    target.designCompensationPlotOutputPath = source.designCompensationPlotOutputPath;
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

std::string standardCircleRadiiCsvOutputPath(const pinjie::StitchRunRequest& request)
{
    if (request.resultOutputDir.empty()) {
        return {};
    }

    const std::filesystem::path path =
        std::filesystem::u8path(request.resultOutputDir) / "standard_sphere_gbt57_p2d_radii.csv";
    return path.u8string();
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

QString designPointCloudPngPath(const pinjie::StitchRunRequest& request)
{
    if (request.resultOutputDir.empty()) {
        return {};
    }
    const std::filesystem::path path =
        std::filesystem::u8path(request.resultOutputDir) / "design_3d_point_cloud.png";
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

double calibrationPixelSizeMm(const pinjie::CalibrationResultCache& cache)
{
    if (!cache.valid || !cache.profile.valid) {
        return 0.0;
    }

    const auto& quality = cache.profile.quality;
    std::vector<double> pixelSizes;
    if (quality.fxPixelsPerMm > 0.0) {
        pixelSizes.push_back(1.0 / quality.fxPixelsPerMm);
    }
    if (quality.fyPixelsPerMm > 0.0) {
        pixelSizes.push_back(1.0 / quality.fyPixelsPerMm);
    }
    if (pixelSizes.empty()) {
        return 0.0;
    }
    return std::accumulate(pixelSizes.begin(), pixelSizes.end(), 0.0) /
           static_cast<double>(pixelSizes.size());
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

    const bool standardCircleMode = configPanel->standardCircleModeEnabled();
    QStringList labels;
    if (configPanel->saveStepSummaryCsv()) {
        labels << (standardCircleMode ? QStringLiteral("国标摘要") : QStringLiteral("拼接汇总"));
    }
    if (configPanel->saveContourPointsCsv()) {
        labels << (standardCircleMode ? QStringLiteral("25点详情") : QStringLiteral("轮廓叠加点"));
    }
    if (configPanel->saveStitchedContourProfileCsv()) {
        labels << (standardCircleMode ? QStringLiteral("窗口图像") : QStringLiteral("整体轮廓剖面"));
    }
    if (configPanel->saveTangentStepCsv()) {
        labels << (standardCircleMode ? QStringLiteral("边缘叠加") : QStringLiteral("切向相关性"));
    }
    if (configPanel->saveNormalErrorProfileCsv()) {
        labels << (standardCircleMode ? QStringLiteral("主暗区掩模") : QStringLiteral("法向误差剖面"));
    }
    if (configPanel->saveTangentProfileCsv()) {
        labels << (standardCircleMode ? QStringLiteral("支撑差异掩模") : QStringLiteral("轮廓波动分析"));
    }
    if (configPanel->saveAlignmentCandidateDiagnosticsCsv()) {
        labels << (standardCircleMode ? QStringLiteral("圆边清理诊断") : QStringLiteral("候选诊断"));
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

    setWindowTitle(QStringLiteral("在线测量与误差补偿系统"));
    resize(1360, 760);

    calibrationPanel_ = new CalibrationConfigPanel(this);
    designModelPanel_ = new DesignModelPanel(this);
    configPanel_ = new RunConfigPanel(this);

    auto* central = new QWidget(this);
    central->setObjectName(QStringLiteral("appRoot"));
    auto* rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(3, 3, 3, 3);
    rootLayout->setSpacing(3);

    auto* titleLabel = new QLabel(QStringLiteral("在线测量与误差补偿系统"), central);
    titleLabel->setObjectName(QStringLiteral("pageTitleLabel"));

    startButton_ = new QPushButton(QStringLiteral("运行复检全流程"), central);
    startButton_->setObjectName(QStringLiteral("primaryActionButton"));
    moduleRunButton_ = new QPushButton(QStringLiteral("运行当前模块"), central);
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
    const QStringList moduleLabels = {QStringLiteral("1 CAD/目标尺寸"),
                                      QStringLiteral("2 数据接收预处理"),
                                      QStringLiteral("3 特征提取"),
                                      QStringLiteral("4 模型配准"),
                                      QStringLiteral("5 误差分析"),
                                      QStringLiteral("6 补偿解算")};
    const QStringList moduleStages = {QStringLiteral("baseline"),
                                      QStringLiteral("acquisition"),
                                      QStringLiteral("processing"),
                                      QStringLiteral("registration"),
                                      QStringLiteral("analysis"),
                                      QStringLiteral("compensation")};
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
    calibrationPage->setObjectName(QStringLiteral("baselineStagePage"));
    calibrationOverviewEdit_ = makeReadOnlyTextEdit(calibrationPage);
    calibrationOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    designModelOverviewEdit_ = makeReadOnlyTextEdit(calibrationPage);
    designModelOverviewEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    calibrationDetailEdit_ = makeReadOnlyTextEdit(calibrationPage);
    calibrationDetailEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    designModelDetailEdit_ = makeReadOnlyTextEdit(calibrationPage);
    calibrationPreviewViewer_ = new ImageViewer(calibrationPage);
    cadPreviewViewer_ = new ImageViewer(calibrationPage);
    cadInteractiveViewer_ = new PointCloud3DViewer(calibrationPage);

    auto* baselineSidebar = new QWidget(calibrationPage);
    configureStageSidebar(baselineSidebar);
    auto* baselineSidebarLayout = new QVBoxLayout(baselineSidebar);
    baselineSidebarLayout->setContentsMargins(0, 0, 0, 0);
    auto* baselineConfigTabs = new QTabWidget(baselineSidebar);
    baselineConfigTabs->addTab(makeScrollArea(calibrationPanel_, baselineConfigTabs), QStringLiteral("量值标定"));
    baselineConfigTabs->addTab(makeScrollArea(designModelPanel_, baselineConfigTabs), QStringLiteral("CAD基准"));
    baselineSidebarLayout->addWidget(baselineConfigTabs, 1);
    baselineSidebarLayout->addWidget(calibrationOverviewEdit_);
    baselineSidebarLayout->addWidget(designModelOverviewEdit_);

    auto* calibrationEvidencePane = new QWidget(calibrationPage);
    auto* calibrationEvidenceLayout = new QVBoxLayout(calibrationEvidencePane);
    calibrationEvidenceLayout->setContentsMargins(0, 0, 0, 0);
    calibrationEvidenceLayout->addWidget(calibrationPreviewViewer_, 1);
    calibrationEvidenceLayout->addWidget(calibrationDetailEdit_);

    auto* designEvidencePane = new QWidget(calibrationPage);
    auto* designEvidenceLayout = new QVBoxLayout(designEvidencePane);
    designEvidenceLayout->setContentsMargins(0, 0, 0, 0);
    auto* cadViewTabs = new QTabWidget(designEvidencePane);
    cadViewTabs->addTab(cadPreviewViewer_, QStringLiteral("CAD截面静态图"));
    cadViewTabs->addTab(cadInteractiveViewer_, QStringLiteral("CAD零件3D"));
    designEvidenceLayout->addWidget(cadViewTabs, 3);
    designEvidenceLayout->addWidget(designModelDetailEdit_, 2);

    auto* baselineEvidenceTabs = new QTabWidget(calibrationPage);
    baselineEvidenceTabs->addTab(calibrationEvidencePane, QStringLiteral("标定预览"));
    baselineEvidenceTabs->addTab(designEvidencePane, QStringLiteral("CAD详情"));

    auto* baselineSplit = new QSplitter(Qt::Horizontal, calibrationPage);
    baselineSplit->addWidget(baselineSidebar);
    baselineSplit->addWidget(baselineEvidenceTabs);
    baselineSplit->setStretchFactor(0, 0);
    baselineSplit->setStretchFactor(1, 1);
    configureStageSplit(baselineSplit);

    auto* calibrationLayout = new QVBoxLayout(calibrationPage);
    calibrationLayout->setContentsMargins(0, 0, 0, 0);
    calibrationLayout->setSpacing(6);
    calibrationLayout->addWidget(baselineSplit, 1);
    stageStack_->addWidget(calibrationPage);

    auto* acquisitionPage = new QWidget(central);
    acquisitionPage->setObjectName(QStringLiteral("acquisitionStagePage"));
    acquisitionOverviewEdit_ = makeReadOnlyTextEdit(acquisitionPage);
    acquisitionPreviewViewer_ = new ImageViewer(acquisitionPage);
    acquisitionPreviewInfoEdit_ = makeReadOnlyTextEdit(acquisitionPage);
    acquisitionPreviewInfoEdit_->setMaximumHeight(kSidebarSummaryMaximumHeight);
    acquisitionPreviewInfoEdit_->setPlainText(QStringLiteral("等待导入图像或相机采集预览。"));
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
    singleSlotViewer_ = new ImageViewer(reportPage);
    pointCloudViewer_ = new PointCloud3DViewer(reportPage);
    publicationFigureViewer_ = new ImageViewer(reportPage);
    summaryEdit_ = makeReadOnlyTextEdit(reportPage);
    qualityReviewEdit_ = makeReadOnlyTextEdit(reportPage);
    designCompareEdit_ = makeReadOnlyTextEdit(reportPage);
    candidateDiagnosticsEdit_ = makeReadOnlyTextEdit(reportPage);
    acceptanceChecklistEdit_ = makeReadOnlyTextEdit(reportPage);
    csvEdit_ = makeReadOnlyTextEdit(reportPage);
    logEdit_ = makeReadOnlyTextEdit(reportPage);
    reportViewTabs_ = new QTabWidget(reportPage);
    reportViewTabs_->setObjectName(QStringLiteral("reportViewTabs"));
    reportViewTabs_->setMinimumHeight(kReportViewerMinimumHeight);
    reportViewTabs_->addTab(panoramaViewer_, QStringLiteral("全景结果"));
    reportViewTabs_->addTab(designCompareViewer_, QStringLiteral("误差分析图"));
    reportViewTabs_->addTab(singleSlotViewer_, QStringLiteral("单槽放大图"));
    reportViewTabs_->addTab(pointCloudViewer_, QStringLiteral("交互3D点云"));
    reportViewTabs_->addTab(publicationFigureViewer_, QStringLiteral("出版误差图"));
    bottomTabs_ = new QTabWidget(reportPage);
    bottomTabs_->setObjectName(QStringLiteral("reportTabs"));
    bottomTabs_->setMinimumHeight(kReportDetailsMinimumHeight);
    bottomTabs_->addTab(summaryEdit_, QStringLiteral("复检总览"));
    bottomTabs_->addTab(qualityReviewEdit_, QStringLiteral("质量审查"));
    bottomTabs_->addTab(designCompareEdit_, QStringLiteral("CAD比对"));
    bottomTabs_->addTab(candidateDiagnosticsEdit_, QStringLiteral("候选诊断"));
    bottomTabs_->addTab(csvEdit_, QStringLiteral("CSV 数据"));
    bottomTabs_->addTab(acceptanceChecklistEdit_, QStringLiteral("表9验收对照"));
    bottomTabs_->addTab(logEdit_, QStringLiteral("日志"));
    auto* metricRail = new QWidget(reportPage);
    metricRail->setObjectName(QStringLiteral("reportMetricRail"));
    metricRail->setMinimumWidth(kReportMetricRailWidth);
    metricRail->setMaximumWidth(kReportMetricRailWidth + 28);
    auto* metricRailLayout = new QVBoxLayout(metricRail);
    metricRailLayout->setContentsMargins(8, 8, 8, 8);
    metricRailLayout->setSpacing(8);

    auto* metricRailTitle = new QLabel(QStringLiteral("复检指标"), metricRail);
    metricRailTitle->setObjectName(QStringLiteral("reportRailTitleLabel"));
    metricRailLayout->addWidget(metricRailTitle);

    auto* acceptanceRailTitle = new QLabel(QStringLiteral("表9验收对照"), metricRail);
    acceptanceRailTitle->setObjectName(QStringLiteral("reportRailTitleLabel"));
    metricRailLayout->addWidget(acceptanceRailTitle);
    acceptanceOverviewEdit_ = makeReadOnlyTextEdit(metricRail);
    acceptanceOverviewEdit_->setObjectName(QStringLiteral("acceptanceOverviewEdit"));
    acceptanceOverviewEdit_->setMaximumHeight(230);
    acceptanceOverviewEdit_->setPlainText(QStringLiteral("等待运行。"));
    metricRailLayout->addWidget(acceptanceOverviewEdit_);

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
    reportMetricPaneButton->setToolTip(QStringLiteral("显示或隐藏右侧复检指标"));

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

    auto* compensationPage = new QWidget(central);
    compensationPage->setObjectName(QStringLiteral("compensationStagePage"));
    compensationViewer_ = new ImageViewer(compensationPage);
    compensationSummaryEdit_ = makeReadOnlyTextEdit(compensationPage);
    compensationSummaryEdit_->setMinimumHeight(180);
    auto* compensationContent = new QSplitter(Qt::Vertical, compensationPage);
    compensationContent->setChildrenCollapsible(false);
    compensationContent->addWidget(compensationViewer_);
    compensationContent->addWidget(compensationSummaryEdit_);
    compensationContent->setStretchFactor(0, 6);
    compensationContent->setStretchFactor(1, 2);
    compensationContent->setSizes({700, 200});
    auto* compensationLayout = new QVBoxLayout(compensationPage);
    compensationLayout->setContentsMargins(0, 0, 0, 0);
    compensationLayout->setSpacing(6);
    compensationLayout->addWidget(compensationContent, 1);
    stageStack_->addWidget(compensationPage);

    setCentralWidget(central);

    connect(startButton_, &QPushButton::clicked, this, [this]() { startRun(); });
    connect(moduleRunButton_, &QPushButton::clicked, this, [this]() { startCurrentModuleRun(); });
    connect(configPanel_, &RunConfigPanel::standardCircleRunRequested, this, [this]() {
        if (!configPanel_) {
            return;
        }
        configPanel_->setStandardCircleModeEnabled(true);
        switchToStage(RegistrationStage);
        configPanel_->revealStandardCircleConfig();
        startRun(pinjie::StitchRunMode::Full);
    });
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
    connect(designModelPanel_, &DesignModelPanel::configChanged, this, [this]() {
        refreshStageSummaries();
        refreshDesignModelDetailPane();
        updateWorkflowAccessState();
    });
    connect(designModelPanel_, &DesignModelPanel::importRequested, this, [this](const QString&) {
        startDesignModelImport();
    });
    connect(designModelPanel_, &DesignModelPanel::scanCompareRequested, this, [this](const QString& stlFilePath) {
        startScannedStlComparison(stlFilePath);
    });
    connect(configPanel_, &RunConfigPanel::configChanged, this, [this]() {
        refreshStageSummaries();
        refreshReportExportState();
        updateWorkflowAccessState();
    });
    connect(configPanel_, &RunConfigPanel::cameraPreviewUpdated, this, [this](const QImage& image, const QString& summary) {
        onAcquisitionPreviewUpdated(image, summary);
    });

    calibrationOverviewEdit_->setPlainText(QStringLiteral("模块 1 CAD/目标尺寸：等待量值标定参数。"));
    designModelOverviewEdit_->setPlainText(QStringLiteral("模块 1 CAD/目标尺寸：等待选择 STEP/STP/IGES 理论模型，或选择扫描 STL 进行独立对比。"));
    designModelDetailEdit_->setPlainText(QStringLiteral("导入后将在这里显示模型摘要、目标尺寸、坐标配置、截面方法与 CAD 坐标映射；扫描 STL 对比会复用同一结果展示页。"));
    acquisitionOverviewEdit_->setPlainText(QStringLiteral("模块 2 数据接收预处理：等待选择工件图像目录或完成在线采集。"));
    calibrationDetailEdit_->setPlainText(QStringLiteral("模块 1 CAD/目标尺寸：等待标定结果。"));
    processingPreviewInfoEdit_->setPlainText(QStringLiteral("模块 3 特征提取：等待预处理预览和轮廓边界。"));
    if (qualityReviewEdit_) {
        qualityReviewEdit_->setPlainText(QStringLiteral("等待模块 5 误差分析生成 quality_review.csv。"));
    }
    if (designCompareEdit_) {
        designCompareEdit_->setPlainText(QStringLiteral("等待模块 5 误差分析生成 CAD/设计母线比对结果。"));
    }
    if (candidateDiagnosticsEdit_) {
        candidateDiagnosticsEdit_->setPlainText(QStringLiteral("等待模块 4 生成候选诊断结果。"));
    }
    setAcceptanceText(buildInitialTable9AcceptanceText());
    if (compensationSummaryEdit_) {
        compensationSummaryEdit_->setPlainText(
            QStringLiteral("模块 6 补偿解算\n\n"
                           "等待模块 5 误差分析完成后生成补偿量可视化。\n"
                           "这里将显示补偿曲线、CAD 点级补偿、槽特征补偿、补偿图/CSV 路径和关键字段说明。\n"
                           "最终补偿后绝对坐标字段为 `compensated_cad_x/y/z_mm`。"));
    }
    if (publicationFigureViewer_) {
        publicationFigureViewer_->clearImage();
    }
    if (singleSlotViewer_) {
        singleSlotViewer_->clearImage();
    }
    if (pointCloudViewer_) {
        pointCloudViewer_->clearCloud();
    }
    statusBar()->showMessage(QStringLiteral("就绪"));

    resetReportMetricCards();
    applyWindowStyle();
    switchToStage(CalibrationStage);
    refreshStageSummaries();
    refreshCalibrationDetailPane();
    refreshDesignModelDetailPane();
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

void MainWindow::startDesignModelImport()
{
    if (!designModelPanel_) {
        return;
    }

    pinjie::cad_model::DesignModelRequest request;
    QString errorMessage;
    if (!designModelPanel_->buildRequest(request, errorMessage)) {
        QMessageBox::warning(this, QStringLiteral("设计模型"), errorMessage);
        switchToStage(CalibrationStage);
        return;
    }

    lastDesignModelRequest_ = request;
    switchToStage(CalibrationStage);
    if (cadPreviewViewer_) {
        cadPreviewViewer_->clearImage();
    }
    if (cadInteractiveViewer_) {
        cadInteractiveViewer_->clearCloud(QStringLiteral("正在导入 CAD 模型，等待三维采样点。"));
    }
    designModelOverviewEdit_->setPlainText(QStringLiteral("正在导入设计模型，请稍候..."));
    appendLog(QStringLiteral("[信息] 开始导入设计模型：%1")
                  .arg(QDir::toNativeSeparators(fromUtf8StdString(request.cadFilePath))));

    const auto result = pinjie::cad_model::loadCadModelDocument(std::filesystem::u8path(request.cadFilePath), request);
    if (result.ok) {
        activeCadModelDocument_ = result.document;
        activeCadModelDocument_.modelLabel = request.modelName.empty()
                                                 ? activeCadModelDocument_.modelLabel
                                                 : request.modelName;
        activeCadModelDocument_.axialAxis = request.axialAxis;
        activeCadModelDocument_.radialAxis = request.radialAxis;
        designModelPanel_->setDocumentInfo(activeCadModelDocument_);
        if (cadInteractiveViewer_) {
            cadInteractiveViewer_->setCadModelDocument(
                activeCadModelDocument_,
                QStringLiteral("CAD零件3D - %1")
                    .arg(activeCadModelDocument_.modelLabel.empty()
                             ? fromUtf8StdString(activeCadModelDocument_.fileName)
                             : fromUtf8StdString(activeCadModelDocument_.modelLabel)),
                QStringLiteral("显示导入 CAD 零件的三角网格面片；绿色线为 CAD 模型真实剖切截面。"));
        }
        generateCadModelPreview(activeCadModelDocument_);
        designModelOverviewEdit_->setPlainText(
            QStringLiteral("设计模型导入完成。\n文件：%1\n格式：%2\n面/边：%3 / %4\n真实截面预览：%5 点\n误差比对轮廓：%6 点")
                .arg(QDir::toNativeSeparators(fromUtf8StdString(activeCadModelDocument_.sourcePath)))
                .arg(fromUtf8StdString(pinjie::cad_model::cadFileFormatLabel(activeCadModelDocument_.format)))
                .arg(activeCadModelDocument_.faceCount)
                .arg(activeCadModelDocument_.edgeCount)
                .arg(static_cast<qulonglong>(activeCadModelDocument_.sectionSamples.size()))
                .arg(static_cast<qulonglong>(activeCadModelDocument_.profileSamples.size())));
        appendLog(QStringLiteral("[信息] %1").arg(fromUtf8StdString(result.message)));
        statusBar()->showMessage(QStringLiteral("设计模型已导入"), 4000);
    } else {
        activeCadModelDocument_ = {};
        designModelPanel_->setDocumentInfo(result.document);
        if (cadPreviewViewer_) {
            cadPreviewViewer_->clearImage();
        }
        if (cadInteractiveViewer_) {
            cadInteractiveViewer_->clearCloud(QStringLiteral("CAD导入失败，无法显示交互3D模型。"));
        }
        designModelOverviewEdit_->setPlainText(
            QStringLiteral("设计模型导入失败。\n原因：%1").arg(fromUtf8StdString(result.message)));
        appendLog(QStringLiteral("[错误] %1").arg(fromUtf8StdString(result.message)));
        QMessageBox::warning(this,
                             QStringLiteral("设计模型"),
                             QStringLiteral("设计模型导入失败：\n%1").arg(fromUtf8StdString(result.message)));
        statusBar()->showMessage(QStringLiteral("设计模型导入失败"), 4000);
    }

    refreshStageSummaries();
    refreshDesignModelDetailPane();
    updateWorkflowAccessState();
    updateModuleRunButtonText(stageStack_ ? stageStack_->currentIndex() : CalibrationStage);
}

void MainWindow::startScannedStlComparison(const QString& stlFilePath)
{
    if (!designModelPanel_) {
        return;
    }
    if (!activeCadModelDocument_.valid ||
        (!activeCadModelDocument_.hasProfileSamples && !activeCadModelDocument_.hasSectionSamples)) {
        QMessageBox::information(this,
                                 QStringLiteral("扫描 STL 对比"),
                                 QStringLiteral("请先导入理论 CAD 模型，再进行扫描 STL 对比。"));
        switchToStage(CalibrationStage);
        return;
    }

    const QString trimmedPath = stlFilePath.trimmed();
    if (trimmedPath.isEmpty()) {
        QMessageBox::information(this,
                                 QStringLiteral("扫描 STL 对比"),
                                 QStringLiteral("请先选择扫描 STL 文件。"));
        switchToStage(CalibrationStage);
        return;
    }

    const QFileInfo scanInfo(trimmedPath);
    if (!scanInfo.exists() || !scanInfo.isFile()) {
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("所选扫描 STL 文件不存在。"));
        switchToStage(CalibrationStage);
        return;
    }

    pinjie::cad_model::DesignModelRequest designRequest = lastDesignModelRequest_;
    QString panelError;
    pinjie::cad_model::DesignModelRequest panelRequest;
    if (designModelPanel_->buildRequest(panelRequest, panelError)) {
        designRequest.profileSamplingStepMm = panelRequest.profileSamplingStepMm;
        designRequest.targetSlotWidthMm = panelRequest.targetSlotWidthMm;
        designRequest.targetSlotDepthMm = panelRequest.targetSlotDepthMm;
        designRequest.temporaryPixelSizeMm = panelRequest.temporaryPixelSizeMm;
        designRequest.localSlotImageMode = panelRequest.localSlotImageMode;
        designRequest.localSlotBottomWidthMm = panelRequest.localSlotBottomWidthMm;
    }

    pinjie::cad_model::DesignModelRequest scanRequest = designRequest;
    scanRequest.cadFilePath = QDir::fromNativeSeparators(trimmedPath).toUtf8().toStdString();
    scanRequest.modelName = scanInfo.completeBaseName().toUtf8().toStdString();

    switchToStage(CalibrationStage);
    appendLog(QStringLiteral("[信息] 开始扫描 STL 与理论 CAD 对比：%1")
                  .arg(QDir::toNativeSeparators(trimmedPath)));
    statusBar()->showMessage(QStringLiteral("正在导入扫描 STL 并执行对比..."));
    if (designCompareEdit_) {
        designCompareEdit_->setPlainText(QStringLiteral("正在导入扫描 STL 并执行理论 CAD 对比..."));
    }
    if (compensationSummaryEdit_) {
        compensationSummaryEdit_->setPlainText(QStringLiteral("正在计算扫描 STL 的误差与补偿量..."));
    }

    const auto scanLoadResult =
        pinjie::cad_model::loadCadModelDocument(std::filesystem::u8path(scanRequest.cadFilePath), scanRequest);
    if (!scanLoadResult.ok) {
        appendLog(QStringLiteral("[错误] %1").arg(fromUtf8StdString(scanLoadResult.message)));
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("扫描 STL 导入失败：\n%1").arg(fromUtf8StdString(scanLoadResult.message)));
        statusBar()->showMessage(QStringLiteral("扫描 STL 导入失败"), 4000);
        return;
    }

    const pinjie::cad_model::CadModelDocument& scanDocument = scanLoadResult.document;
    if (cadInteractiveViewer_) {
        cadInteractiveViewer_->setCadModelDocument(
            scanDocument,
            QStringLiteral("扫描 STL 点云 - %1").arg(scanInfo.completeBaseName()),
            QStringLiteral("显示扫描 STL 的三维点云；可拖拽旋转、平移与缩放查看。"));
    }
    const bool hasMeasuredProfile = scanDocument.hasProfileSamples && scanDocument.profileSamples.size() >= 2;
    const bool hasMeasuredSection = scanDocument.hasSectionSamples && scanDocument.sectionSamples.size() >= 2;
    if (!hasMeasuredProfile && !hasMeasuredSection) {
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("扫描 STL 已导入，但未生成可用于对比的截面/轮廓样本。"));
        statusBar()->showMessage(QStringLiteral("扫描 STL 缺少可对比轮廓"), 4000);
        return;
    }

    const std::string runName =
        (activeCadModelDocument_.modelLabel.empty() ? std::string("design_cad") : activeCadModelDocument_.modelLabel) +
        "_scan_vs_" + scanInfo.completeBaseName().toUtf8().toStdString();
    const auto resultPaths = pinjie::buildDefaultStitchResultPaths(runName, "scan_vs_cad", false);
    if (!pinjie::ensureStitchResultDirectories(resultPaths)) {
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("结果目录创建失败。"));
        statusBar()->showMessage(QStringLiteral("结果目录创建失败"), 4000);
        return;
    }

    pinjie::StitchRunRequest compareRequest;
    compareRequest.resultOutputDir = resultPaths.runDir.u8string();
    compareRequest.designErrorProfileCsvOutputPath = resultPaths.designErrorProfileCsvPath.u8string();
    compareRequest.designErrorSummaryCsvOutputPath = resultPaths.designErrorSummaryCsvPath.u8string();
    compareRequest.design3dErrorCsvOutputPath = resultPaths.design3dErrorCsvPath.u8string();
    compareRequest.designCompensationCsvOutputPath = resultPaths.designCompensationCsvPath.u8string();
    compareRequest.designFeatureCompensationCsvOutputPath = resultPaths.designFeatureCompensationCsvPath.u8string();
    compareRequest.pipelineConfig.enableDesignComparison = true;
    applyActiveDesignModelToRequest(compareRequest);
    compareRequest.pipelineConfig.enableDesignComparison = true;

    const std::vector<pinjie::cad_design::DesignProfileSample>& measuredSamples =
        compareRequest.pipelineConfig.localSlotImageMode && hasMeasuredSection
            ? scanDocument.sectionSamples
            : (hasMeasuredProfile ? scanDocument.profileSamples : scanDocument.sectionSamples);
    const pinjie::cad_design::DesignAlignmentResult designResult =
        pinjie::cad_design::compareMeasuredCadProfileToDesign(measuredSamples, compareRequest.pipelineConfig);
    if (!designResult.ok) {
        appendLog(QStringLiteral("[错误] %1").arg(fromUtf8StdString(designResult.message)));
        if (designCompareEdit_) {
            designCompareEdit_->setPlainText(
                QStringLiteral("扫描 STL 与理论 CAD 对比失败。\n%1")
                    .arg(fromUtf8StdString(designResult.message)));
        }
        if (designCompareViewer_) {
            designCompareViewer_->clearImage();
        }
        if (compensationViewer_) {
            compensationViewer_->clearImage();
        }
        if (pointCloudViewer_) {
            pointCloudViewer_->clearCloud(QStringLiteral("扫描 STL 对比失败，未生成三维误差点云。"));
        }
        if (compensationSummaryEdit_) {
            compensationSummaryEdit_->setPlainText(
                QStringLiteral("扫描 STL 对比未生成补偿结果。\n%1")
                    .arg(fromUtf8StdString(designResult.message)));
        }
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("对比失败：\n%1").arg(fromUtf8StdString(designResult.message)));
        statusBar()->showMessage(QStringLiteral("扫描 STL 对比失败"), 4000);
        return;
    }

    const auto saveTextArtifact = [&](const std::string& path,
                                      const std::string& content,
                                      const QString& label) -> bool {
        if (path.empty()) {
            return true;
        }
        if (stitch::writeTextFileToPath(path, content)) {
            return true;
        }
        QMessageBox::warning(this,
                             QStringLiteral("扫描 STL 对比"),
                             QStringLiteral("保存 %1 失败：\n%2").arg(label, displayOutputPath(path)));
        return false;
    };
    if (!saveTextArtifact(compareRequest.designErrorProfileCsvOutputPath,
                          designResult.profileCsvText,
                          QStringLiteral("design_error_profile.csv")) ||
        !saveTextArtifact(compareRequest.designErrorSummaryCsvOutputPath,
                          designResult.summaryCsvText,
                          QStringLiteral("design_error_summary.csv")) ||
        !saveTextArtifact(compareRequest.design3dErrorCsvOutputPath,
                          designResult.error3dCsvText,
                          QStringLiteral("design_3d_error_points.csv")) ||
        !saveTextArtifact(compareRequest.designCompensationCsvOutputPath,
                          designResult.compensationCsvText,
                          QStringLiteral("design_compensation.csv")) ||
        !saveTextArtifact(compareRequest.designFeatureCompensationCsvOutputPath,
                          designResult.featureCompensationCsvText,
                          QStringLiteral("design_feature_compensation.csv"))) {
        statusBar()->showMessage(QStringLiteral("扫描 STL 结果保存失败"), 4000);
        return;
    }

    QString plotError;
    QString plotOutput;
    const bool plotOk = generateMatplotlibDesignPlots(compareRequest, true, plotError, &plotOutput);
    if (!plotOutput.isEmpty()) {
        appendLog(QStringLiteral("[图形脚本]\n%1").arg(plotOutput));
    }
    if (!plotOk) {
        appendLog(QStringLiteral("[警告] %1").arg(plotError));
    }

    lastRequest_ = compareRequest;
    currentRunMode_ = pinjie::StitchRunMode::Report;
    runCache_.reset();
    completedSteps_.clear();
    preprocessImages_.clear();
    debugImages_.clear();
    if (stepModel_) {
        stepModel_->clear();
    }
    if (processingImageList_) {
        processingImageList_->clear();
    }
    if (panoramaViewer_) {
        panoramaViewer_->clearImage();
    }
    if (referenceViewer_) {
        referenceViewer_->clearImage();
    }
    if (targetViewer_) {
        targetViewer_->clearImage();
    }
    if (debugViewer_) {
        debugViewer_->clearImage();
    }
    if (qualityReviewEdit_) {
        qualityReviewEdit_->setPlainText(QStringLiteral("扫描 STL 对比分支未生成拼接质量审查 CSV。"));
    }
    if (candidateDiagnosticsEdit_) {
        candidateDiagnosticsEdit_->setPlainText(QStringLiteral("扫描 STL 对比分支未经过拼接候选诊断流程。"));
    }
    if (csvEdit_) {
        csvEdit_->setPlainText(fromUtf8StdString(designResult.summaryCsvText));
    }

    QStringList compareLines;
    compareLines << QStringLiteral("扫描 STL 与理论 CAD 对比完成");
    compareLines << QString();
    compareLines << QStringLiteral("理论 CAD：%1")
                        .arg(fromUtf8StdString(activeCadModelDocument_.modelLabel.empty()
                                                   ? activeCadModelDocument_.fileName
                                                   : activeCadModelDocument_.modelLabel));
    compareLines << QStringLiteral("扫描 STL：%1").arg(QDir::toNativeSeparators(trimmedPath));
    compareLines << QStringLiteral("扫描样本：%1 点（使用 %2 进行误差解算）")
                        .arg(static_cast<qulonglong>(measuredSamples.size()))
                        .arg(hasMeasuredProfile && !(compareRequest.pipelineConfig.localSlotImageMode && hasMeasuredSection)
                                 ? QStringLiteral("扫描轮廓 profileSamples")
                                 : QStringLiteral("扫描截面 sectionSamples"));
    compareLines << QStringLiteral("有效对比点：%1").arg(static_cast<qulonglong>(designResult.summary.usedCount));
    compareLines << QStringLiteral("轮廓 RMS：%1 um")
                        .arg(formatMetric(designResult.summary.profileStats.rmseUm, 3));
    compareLines << QStringLiteral("轮廓 P95：%1 um")
                        .arg(formatMetric(designResult.summary.profileStats.p95AbsUm, 3));
    compareLines << QStringLiteral("补偿偏置：%1 um")
                        .arg(formatMetric(designResult.summary.meanNormalErrorUm, 3));
    compareLines << QStringLiteral("结果说明：%1").arg(fromUtf8StdString(designResult.message));
    compareLines << QStringLiteral("结果目录：%1").arg(displayOutputPath(compareRequest.resultOutputDir));
    if (!plotOk) {
        compareLines << QStringLiteral("图表生成：%1").arg(plotError);
    }
    if (designCompareEdit_) {
        designCompareEdit_->setPlainText(compareLines.join('\n'));
    }

    QImage contourPreview;
    if (!compareRequest.contourOverlayOutputPath.empty()) {
        contourPreview.load(displayOutputPath(compareRequest.contourOverlayOutputPath));
    }
    if (processingPreviewViewer_) {
        if (!contourPreview.isNull()) {
            processingPreviewViewer_->setImage(contourPreview);
        } else {
            processingPreviewViewer_->clearImage();
        }
    }
    if (processingPreviewInfoEdit_) {
        processingPreviewInfoEdit_->setPlainText(
            contourPreview.isNull()
                ? QStringLiteral("扫描 STL 对比分支未生成单槽/特征预览图。")
                : QStringLiteral("扫描 STL 对比分支：已加载单槽/特征预览图。\n输出图：%1")
                      .arg(displayOutputPath(compareRequest.contourOverlayOutputPath)));
    }
    if (singleSlotViewer_) {
        if (!contourPreview.isNull()) {
            singleSlotViewer_->setImage(contourPreview);
        } else {
            singleSlotViewer_->clearImage();
        }
    }

    if (designCompareViewer_) {
        QImage designComparisonPlot;
        if (!compareRequest.designComparisonOverlayOutputPath.empty()) {
            designComparisonPlot.load(displayOutputPath(compareRequest.designComparisonOverlayOutputPath));
        }
        if (!designComparisonPlot.isNull()) {
            designCompareViewer_->setImage(designComparisonPlot);
        } else {
            designCompareViewer_->clearImage();
        }
    }
    if (compensationViewer_) {
        QImage compensationPlot;
        if (!compareRequest.designCompensationPlotOutputPath.empty()) {
            compensationPlot.load(displayOutputPath(compareRequest.designCompensationPlotOutputPath));
        }
        if (!compensationPlot.isNull()) {
            compensationViewer_->setImage(compensationPlot);
        } else {
            compensationViewer_->clearImage();
        }
    }
    if (pointCloudViewer_) {
        const QString pointCloudCsvPath =
            compareRequest.design3dErrorCsvOutputPath.empty()
                ? QString()
                : displayOutputPath(compareRequest.design3dErrorCsvOutputPath);
        QString pointCloudMessage;
        if (!pointCloudCsvPath.isEmpty() &&
            pointCloudViewer_->loadDesignErrorCsv(pointCloudCsvPath, &pointCloudMessage)) {
            if (!pointCloudMessage.isEmpty()) {
                appendLog(QStringLiteral("[信息] %1").arg(pointCloudMessage));
            }
        } else {
            pointCloudViewer_->clearCloud(
                pointCloudCsvPath.isEmpty()
                    ? QStringLiteral("扫描 STL 对比分支未生成三维误差 CSV。")
                    : QStringLiteral("扫描 STL 对比分支未能加载三维误差点云。"));
        }
    }

    if (compensationSummaryEdit_) {
        const std::size_t cadCompensationRows =
            static_cast<std::size_t>(std::count_if(designResult.profilePoints.begin(),
                                                   designResult.profilePoints.end(),
                                                   [](const auto& point) {
                                                       return point.isUsed && point.hasCadCoordinates;
                                                   }));
        QStringList compensationLines;
        compensationLines << QStringLiteral("模块 6 补偿解算");
        compensationLines << QString();
        compensationLines << QStringLiteral("扫描 STL 对比模式：已输出理论 CAD 坐标系下的点级补偿结果。");
        compensationLines << QStringLiteral("有效 CAD 补偿点：%1 / %2")
                                 .arg(static_cast<qulonglong>(cadCompensationRows))
                                 .arg(static_cast<qulonglong>(designResult.summary.usedCount));
        compensationLines << QStringLiteral("CAD 坐标轴：轴向 %1，径向 %2，轴向原点 %3 mm，方向系数 %4")
                                 .arg(fromUtf8StdString(designResult.summary.designProfileMetadata.axialAxis),
                                      fromUtf8StdString(designResult.summary.designProfileMetadata.radialAxis))
                                 .arg(designResult.summary.designProfileMetadata.cadAxialOriginMm, 0, 'f', 6)
                                 .arg(designResult.summary.designProfileMetadata.cadAxialDirectionSign, 0, 'f', 0);
        compensationLines << QStringLiteral("绝对坐标字段：`compensated_cad_x/y/z_mm`");
        compensationLines << QStringLiteral("补偿增量字段：`delta_x/y/z_um`");
        compensationLines << QString();
        compensationLines << QStringLiteral("输出文件");
        compensationLines << QStringLiteral("误差明细 CSV：%1")
                                 .arg(displayOutputPath(compareRequest.designErrorProfileCsvOutputPath));
        compensationLines << QStringLiteral("误差汇总 CSV：%1")
                                 .arg(displayOutputPath(compareRequest.designErrorSummaryCsvOutputPath));
        compensationLines << QStringLiteral("三维误差 CSV：%1")
                                 .arg(displayOutputPath(compareRequest.design3dErrorCsvOutputPath));
        compensationLines << QStringLiteral("CAD 补偿 CSV：%1")
                                 .arg(displayOutputPath(compareRequest.designCompensationCsvOutputPath));
        compensationLines << QStringLiteral("槽特征补偿 CSV：%1")
                                 .arg(displayOutputPath(compareRequest.designFeatureCompensationCsvOutputPath));
        compensationLines << QStringLiteral("补偿图 PNG：%1")
                                 .arg(displayOutputPath(compareRequest.designCompensationPlotOutputPath));
        compensationLines << QStringLiteral("3D 点云 PNG：%1")
                                 .arg(designPointCloudPngPath(compareRequest));
        compensationLines << QString();
        compensationLines << QStringLiteral("槽特征补偿预览");
        compensationLines << QStringLiteral("----------------------------------------");
        compensationLines << (fromUtf8StdString(designResult.featureCompensationCsvText).trimmed().isEmpty()
                                  ? QStringLiteral("本次扫描 STL 对比未生成槽特征补偿表。")
                                  : fromUtf8StdString(designResult.featureCompensationCsvText).trimmed());
        if (!plotOk) {
            compensationLines << QString();
            compensationLines << QStringLiteral("图表脚本提示");
            compensationLines << plotError;
        }
        compensationSummaryEdit_->setPlainText(compensationLines.join('\n'));
    }

    QStringList summaryLines;
    summaryLines << QStringLiteral("扫描 STL 与理论 CAD 对比完成");
    summaryLines << QString();
    summaryLines << QStringLiteral("理论 CAD：%1")
                        .arg(fromUtf8StdString(activeCadModelDocument_.sourcePath));
    summaryLines << QStringLiteral("扫描 STL：%1").arg(QDir::toNativeSeparators(trimmedPath));
    summaryLines << QStringLiteral("扫描导入摘要：%1").arg(fromUtf8StdString(scanLoadResult.message));
    summaryLines << QStringLiteral("对齐结果：%1").arg(fromUtf8StdString(designResult.message));
    summaryLines << QStringLiteral("轮廓 RMS：%1 um")
                        .arg(formatMetric(designResult.summary.profileStats.rmseUm, 3));
    summaryLines << QStringLiteral("轮廓 P95：%1 um")
                        .arg(formatMetric(designResult.summary.profileStats.p95AbsUm, 3));
    summaryLines << QStringLiteral("有效对比点：%1")
                        .arg(static_cast<qulonglong>(designResult.summary.usedCount));
    summaryLines << QStringLiteral("结果目录：%1").arg(displayOutputPath(compareRequest.resultOutputDir));
    if (!plotOk) {
        summaryLines << QStringLiteral("图表生成：%1").arg(plotError);
    }
    if (summaryEdit_) {
        summaryEdit_->setPlainText(summaryLines.join('\n'));
    }

    setAcceptanceText(buildTable9AcceptanceText(true, 0, 0, &designResult, &designResult.summary));
    updateReportMetricCards(0, 0, nullptr, &designResult.summary);
    refreshRegistrationToolState();
    refreshReportExportState(QStringLiteral("归档状态：扫描 STL 与理论 CAD 对比结果已写入结果目录。"));
    refreshStageSummaries();
    bottomTabs_->setCurrentWidget(summaryEdit_);
    switchToStage(ReportStage);
    statusBar()->showMessage(QStringLiteral("扫描 STL 与理论 CAD 对比完成"), 4000);
    appendLog(QStringLiteral("[信息] 扫描 STL 对比结果目录：%1")
                  .arg(displayOutputPath(compareRequest.resultOutputDir)));
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

    pinjie::StitchRunRequest request;
    QString errorMessage;
    if (!configPanel_->buildRequest(request, errorMessage, false)) {
        QMessageBox::warning(this, QStringLiteral("运行配置"), errorMessage);
        switchToStage(AcquisitionStage);
        return;
    }

    const bool standardCircleMode = isStandardCircleMode(request);
    pinjie::cad_model::DesignModelRequest panelDesignRequest;
    QString panelDesignError;
    const bool panelDesignReady =
        designModelPanel_ && designModelPanel_->buildRequest(panelDesignRequest, panelDesignError);
    const bool panelLocalSlotMode = !standardCircleMode && panelDesignReady && panelDesignRequest.localSlotImageMode;
    const auto applyPanelLocalSlotMode = [&](pinjie::StitchRunRequest& target) {
        if (!panelLocalSlotMode) {
            return;
        }
        target.pipelineConfig.localSlotImageMode = true;
        target.pipelineConfig.localSlotBottomWidthMm = panelDesignRequest.localSlotBottomWidthMm;
        target.pipelineConfig.designTargetSlotWidthMm =
            panelDesignRequest.targetSlotWidthMm > 0.0
                ? panelDesignRequest.targetSlotWidthMm
                : panelDesignRequest.localSlotBottomWidthMm;
        target.pipelineConfig.designTargetSlotDepthMm = panelDesignRequest.targetSlotDepthMm;
        const bool hasExternalCadProfile =
            target.pipelineConfig.designUseExternalProfile &&
            target.pipelineConfig.designExternalProfileSamples.size() >= 2;
        if (!hasExternalCadProfile) {
            target.pipelineConfig.designUseExternalProfile = false;
            target.pipelineConfig.designExternalProfileSamples.clear();
            target.pipelineConfig.designProfileMetadata = {};
        }
        target.pipelineConfig.designUseCentralSlotImageRoi = false;
        if (target.imagePaths.size() > 1) {
            target.imagePaths.resize(1);
        }
    };
    if (panelLocalSlotMode) {
        applyPanelLocalSlotMode(request);
        appendLog(QStringLiteral("[信息] 已启用局部槽特征检测并接入 CAD 对比页：槽底标定宽度 %1 mm，仅使用第一张局部单槽图。")
                      .arg(panelDesignRequest.localSlotBottomWidthMm, 0, 'f', 4));
    }
    const auto applyBaselineToRequest = [this](pinjie::StitchRunRequest& target) {
        const double pixelSizeMm = calibrationPixelSizeMm(activeCalibrationCache_);
        if (pixelSizeMm > 0.0) {
            target.pipelineConfig.designPixelSizeMm = pixelSizeMm;
        }
        applyActiveDesignModelToRequest(target);
    };

    if (!standardCircleMode) {
        applyBaselineToRequest(request);
        applyPanelLocalSlotMode(request);
        const double testPixelSizeMm = singleSlotTestPixelSizeMm(request);
        if (testPixelSizeMm > 0.0) {
            appendLog(QStringLiteral("[信息] 检测到 CAD 匹配单槽测试图，已自动使用 %1 mm/px 的测试像素当量。")
                          .arg(testPixelSizeMm, 0, 'f', 10));
        }
    }
    if (!standardCircleMode &&
        !request.pipelineConfig.localSlotImageMode &&
        !hasMeasurementBaseline() &&
        !isSingleSlotTestInput(request)) {
        QMessageBox::information(this,
                                 QStringLiteral("需要标定"),
                                 QStringLiteral("请先加载有效的标定结果；若只是用当前 CAD 与临时/公开图片测试，请在 CAD/目标尺寸模块填写临时像素当量并重新导入 CAD。"));
        switchToStage(CalibrationStage);
        return;
    }
    if (standardCircleMode) {
        runMode = pinjie::StitchRunMode::Full;
    }

    const bool reusePreviousOutputPaths =
        !standardCircleMode &&
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
    } else if (!standardCircleMode) {
        applyBaselineToRequest(request);
        applyPanelLocalSlotMode(request);
    }

    const bool requestsDesignAnalysis =
        !standardCircleMode &&
        (runMode == pinjie::StitchRunMode::Full || runMode == pinjie::StitchRunMode::Report);
    const bool hasUsableCadProfile =
        activeCadModelDocument_.valid &&
        activeCadModelDocument_.hasProfileSamples &&
        activeCadModelDocument_.profileSamples.size() >= 2;
    const bool hasUsableDesignSource = hasUsableCadProfile || request.pipelineConfig.localSlotImageMode;
    const bool allowFullRunWithoutCad =
        !standardCircleMode &&
        runMode == pinjie::StitchRunMode::Full &&
        !request.pipelineConfig.localSlotImageMode &&
        !hasUsableCadProfile;

    request.pipelineConfig.enableDesignComparison = requestsDesignAnalysis && hasUsableDesignSource;
    if (allowFullRunWithoutCad) {
        appendLog(QStringLiteral("[信息] 未导入可用 CAD 设计基准，本次将仅执行工件拼接/常规结果输出，跳过 CAD 误差分析与补偿解算。"));
    } else if (requestsDesignAnalysis && !hasUsableDesignSource) {
        QMessageBox::warning(this,
                             QStringLiteral("CAD误差分析未就绪"),
                             QStringLiteral("全流程、误差分析和补偿解算需要一个可用设计基准：\n"
                                            "1. 导入可用 CAD 模型并生成 CAD X/Y/Z 采样轮廓；或\n"
                                            "2. 在模块 1 启用“局部槽特征检测”，用单槽截图和槽底真实宽度作为基准。\n\n"
                                            "如果只想检查图像预处理或模型配准，请运行模块 2/3/4。"));
        switchToStage(CalibrationStage);
        return;
    }

    request.previousCache = standardCircleMode ? pinjie::StitchRunCachePtr() : runCache_;
    lastRequest_ = request;
    currentRunMode_ = runMode;
    currentTaskSupportsStop_ = true;
    if (standardCircleMode) {
        runCache_.reset();
    }
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
    if (pointCloudViewer_) {
        pointCloudViewer_->clearCloud();
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
    setReportTabTexts(reportViewTabs_, bottomTabs_, standardCircleMode);
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
        switchToStage(CalibrationStage);
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
    } else if (stage == QStringLiteral("report")) {
        switchToStage(ReportStage);
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
                                                      ? QStringLiteral("等待导入图像或相机采集预览。")
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
    const bool standardCircleMode = isStandardCircleMode(lastRequest_);

    if (!standardCircleMode &&
        cache &&
        (cache->hasLoadedImages() || cache->hasPreprocessedEdges() || cache->hasStitching())) {
        runCache_ = std::move(cache);
    } else if (standardCircleMode) {
        runCache_.reset();
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
    } else if (!lastRequest_.panoramaOutputPath.empty()) {
        const QImage panoramaFromFile(displayOutputPath(lastRequest_.panoramaOutputPath));
        if (!panoramaFromFile.isNull()) {
            panoramaViewer_->setImage(panoramaFromFile);
        } else {
            appendLog(QStringLiteral("[警告] 全景结果图未能加载：%1")
                          .arg(displayOutputPath(lastRequest_.panoramaOutputPath)));
        }
    }

    appendLog(ok ? QStringLiteral("[信息] %1已完成").arg(runModeLabel(currentRunMode_))
                 : QStringLiteral("[错误] %1失败").arg(runModeLabel(currentRunMode_)));
    if (!message.isEmpty()) {
        appendLog(message);
    }

    if (standardCircleMode) {
        setReportTabTexts(reportViewTabs_, bottomTabs_, true);
        completedSteps_.clear();
        if (stepModel_) {
            stepModel_->clear();
        }

        if (panorama.isNull() && !lastRequest_.panoramaOutputPath.empty()) {
            const QImage panoramaFromFile(displayOutputPath(lastRequest_.panoramaOutputPath));
            if (!panoramaFromFile.isNull()) {
                panoramaViewer_->setImage(panoramaFromFile);
            }
        }
        if (designCompareViewer_) {
            const QImage edgeOverlay(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
            if (!edgeOverlay.isNull()) {
                designCompareViewer_->setImage(edgeOverlay);
            } else {
                designCompareViewer_->clearImage();
            }
        }
        if (publicationFigureViewer_) {
            publicationFigureViewer_->clearImage();
        }
        if (singleSlotViewer_) {
            singleSlotViewer_->clearImage();
        }
        if (pointCloudViewer_) {
            pointCloudViewer_->clearCloud();
        }

        const QString summaryCsv = readUtf8TextFile(lastRequest_.csvOutputPath);
        const QString coverageCsv = readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath);
        const QString pointCsv = readUtf8TextFile(lastRequest_.contourPointsCsvOutputPath);
        const QString radiiCsv = readUtf8TextFile(standardCircleRadiiCsvOutputPath(lastRequest_));
        const QString edgeCleanupCsv =
            readUtf8TextFile(lastRequest_.alignmentCandidateDiagnosticsCsvOutputPath);
        const QString dominantMaskCsv = readUtf8TextFile(lastRequest_.designErrorProfileCsvOutputPath);
        const QString supportMaskCsv = readUtf8TextFile(lastRequest_.designErrorSummaryCsvOutputPath);
        const QString summaryText = summaryCsv.isEmpty() ? csvText : summaryCsv;
        const QHash<QString, QString> metrics = parseMetricCsvByKey(summaryText);

        csvEdit_->setPlainText(summaryText.isEmpty() ? QStringLiteral("本次运行未生成国标摘要 CSV。")
                                                     : summaryText);
        if (qualityReviewEdit_) {
            qualityReviewEdit_->setPlainText(
                coverageCsv.isEmpty() ? QStringLiteral("本次运行未生成候选覆盖统计。") : coverageCsv);
        }
        if (designCompareEdit_) {
            QStringList detailSections;
            if (!radiiCsv.trimmed().isEmpty()) {
                detailSections << QStringLiteral("25点半径结果") << radiiCsv.trimmed();
            }
            if (!pointCsv.trimmed().isEmpty()) {
                detailSections << QStringLiteral("25点完整详情") << pointCsv.trimmed();
            }
            designCompareEdit_->setPlainText(
                detailSections.isEmpty() ? QStringLiteral("本次运行未生成 25 点详情。")
                                         : detailSections.join(QStringLiteral("\n\n")));
        }
        if (candidateDiagnosticsEdit_) {
            QStringList diagnostics;
            if (!edgeCleanupCsv.trimmed().isEmpty()) {
                diagnostics << QStringLiteral("圆边清理诊断") << edgeCleanupCsv.trimmed();
            }
            if (!dominantMaskCsv.trimmed().isEmpty()) {
                diagnostics << QStringLiteral("主暗区掩模诊断") << dominantMaskCsv.trimmed();
            }
            if (!supportMaskCsv.trimmed().isEmpty()) {
                diagnostics << QStringLiteral("支撑差异掩模诊断") << supportMaskCsv.trimmed();
            }
            candidateDiagnosticsEdit_->setPlainText(
                diagnostics.isEmpty() ? QStringLiteral("本次运行未生成掩模诊断。")
                                      : diagnostics.join(QStringLiteral("\n\n")));
        }
        if (compensationViewer_) {
            compensationViewer_->clearImage();
        }
        if (compensationSummaryEdit_) {
            compensationSummaryEdit_->setPlainText(
                QStringLiteral("国标标准圆检测模式\n\n"
                               "该模式输出 25 视场形状误差和窗口诊断，不执行 CAD 点级补偿或槽特征补偿。\n"
                               "补偿解算模块仅适用于已导入 CAD 基准的工件轮廓检测。"));
        }

        QStringList summaryLines;
        summaryLines << QStringLiteral("国标复检总览");
        summaryLines << QString();
        summaryLines << QStringLiteral("运行结果：%1").arg(ok ? QStringLiteral("成功") : QStringLiteral("失败"));
        summaryLines << QStringLiteral("已加载图像：%1").arg(loadedImageCount);
        summaryLines << QStringLiteral("已预处理图像：%1").arg(preprocessedImageCount);
        if (!lastRequest_.resultOutputDir.empty()) {
            summaryLines << QStringLiteral("结果目录：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir)));
        }
        if (!message.isEmpty()) {
            summaryLines << QStringLiteral("运行消息：%1").arg(message);
        }
        const auto appendMetricLine =
            [&summaryLines](const QString& label, const QString& value, const QString& suffix = QString()) {
                if (!value.isEmpty()) {
                    summaryLines << QStringLiteral("%1：%2%3").arg(label, value, suffix);
                }
            };
        appendMetricLine(QStringLiteral("E_P2D"), metricValue(metrics, QStringLiteral("e_p2d"), QStringLiteral("um")), QStringLiteral(" um"));
        appendMetricLine(QStringLiteral("单点 RMSE"),
                         metricValue(metrics, QStringLiteral("single_point_rmse"), QStringLiteral("um")),
                         QStringLiteral(" um"));
        appendMetricLine(QStringLiteral("像素当量"),
                         metricValue(metrics, QStringLiteral("pixel_size"), QStringLiteral("um/px")),
                         QStringLiteral(" um/px"));
        appendMetricLine(QStringLiteral("25 点数量"),
                         metricValue(metrics, QStringLiteral("selected_point_count"), QStringLiteral("count")));
        appendMetricLine(QStringLiteral("窗口在图内数量"),
                         metricValue(metrics,
                                     QStringLiteral("measurement_window_inside_image_count"),
                                     QStringLiteral("count")));
        appendMetricLine(QStringLiteral("窗口重叠数量"),
                         metricValue(metrics,
                                     QStringLiteral("measurement_window_overlap_count"),
                                     QStringLiteral("count")));
        appendMetricLine(QStringLiteral("窗口越界数量"),
                         metricValue(metrics, QStringLiteral("window_violation_count"), QStringLiteral("count")));
        appendMetricLine(QStringLiteral("目标角最大偏差"),
                         metricValue(metrics, QStringLiteral("target_angle_delta_max"), QStringLiteral("deg")),
                         QStringLiteral(" deg"));
        appendMetricLine(QStringLiteral("角间距最大误差"),
                         metricValue(metrics, QStringLiteral("angle_spacing_max_error"), QStringLiteral("deg")),
                         QStringLiteral(" deg"));
        appendMetricLine(QStringLiteral("全局椭圆轴比"),
                         metricValue(metrics, QStringLiteral("global_ellipse_axis_ratio"), QStringLiteral("ratio")));
        if (metricValue(metrics,
                        QStringLiteral("ellipse_rectified_evaluation_applied"),
                        QStringLiteral("bool")) == QStringLiteral("1")) {
            summaryLines << QStringLiteral("椭圆修正：已应用");
        }
        if (!lastRequest_.panoramaOutputPath.empty()) {
            summaryLines << QStringLiteral("测量窗口图：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.panoramaOutputPath)));
        }
        if (!lastRequest_.designComparisonOverlayOutputPath.empty()) {
            summaryLines << QStringLiteral("边缘叠加图：%1")
                                .arg(QDir::toNativeSeparators(
                                    fromUtf8StdString(lastRequest_.designComparisonOverlayOutputPath)));
        }
        if (!lastRequest_.contourPointsCsvOutputPath.empty()) {
            summaryLines << QStringLiteral("25 点详情 CSV：%1")
                                .arg(QDir::toNativeSeparators(
                                    fromUtf8StdString(lastRequest_.contourPointsCsvOutputPath)));
        }
        const std::string radiiCsvPath = standardCircleRadiiCsvOutputPath(lastRequest_);
        if (!radiiCsvPath.empty()) {
            summaryLines << QStringLiteral("25 点半径 CSV：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(radiiCsvPath)));
        }

        if (acceptanceChecklistEdit_) {
            QStringList acceptanceLines;
            acceptanceLines << QStringLiteral("国标复检清单");
            acceptanceLines << QString();
            acceptanceLines << QStringLiteral("模块 1 CAD/目标尺寸：%1；标准圆直径 %2 mm")
                                   .arg(hasActiveCalibration() ? QStringLiteral("已加载标定结果")
                                                               : QStringLiteral("标准圆模式使用标准圆直径修正像素当量"))
                                   .arg(lastRequest_.standardCircleConfig.sphereDiameterMm, 0, 'f', 5);
            acceptanceLines << QStringLiteral("模块 2 数据接收预处理：已加载 %1 张图像").arg(loadedImageCount);
            acceptanceLines << QStringLiteral("模块 3 特征提取：候选覆盖、主暗区掩模、支撑差异掩模已生成则视为可复核");
            acceptanceLines << QStringLiteral("模块 4 模型配准：标准圆模式按固定 25 视场检测链复核");
            acceptanceLines << QStringLiteral("模块 5 误差分析：%1")
                                   .arg(summaryText.isEmpty() ? QStringLiteral("未生成国标摘要 CSV")
                                                              : QStringLiteral("已生成国标摘要 CSV"));
            acceptanceLines << QStringLiteral("模块 6 补偿解算：标准圆模式不执行 CAD 补偿；%1")
                                   .arg(ok ? QStringLiteral("通过，结果文件已写入目录")
                                           : QStringLiteral("失败，请检查日志"));
            acceptanceLines << QString();
            acceptanceLines << QStringLiteral("证据文件");
            acceptanceLines << QStringLiteral("摘要 CSV：%1").arg(displayOutputPath(lastRequest_.csvOutputPath));
            acceptanceLines << QStringLiteral("25 点详情 CSV：%1").arg(displayOutputPath(lastRequest_.contourPointsCsvOutputPath));
            acceptanceLines << QStringLiteral("候选覆盖/质量 CSV：%1").arg(displayOutputPath(lastRequest_.qualityReviewCsvOutputPath));
            acceptanceLines << QStringLiteral("掩模/候选诊断 CSV：%1").arg(displayOutputPath(lastRequest_.alignmentCandidateDiagnosticsCsvOutputPath));
            setAcceptanceText(acceptanceLines.join('\n'));
        }

        summaryEdit_->setPlainText(summaryLines.join('\n'));
        bottomTabs_->setCurrentWidget(summaryEdit_);
        resetReportMetricCards();
        if (metricLoadedImagesValue_) {
            metricLoadedImagesValue_->setText(QString::number(std::max(loadedImageCount, 0)));
        }
        if (metricPreprocessedImagesValue_) {
            metricPreprocessedImagesValue_->setText(QString::number(std::max(preprocessedImageCount, 0)));
        }
        if (metricStepCountValue_) {
            const QString selectedCount =
                metricValue(metrics, QStringLiteral("selected_point_count"), QStringLiteral("count"));
            metricStepCountValue_->setText(selectedCount.isEmpty() ? QStringLiteral("0") : selectedCount);
        }
        if (metricFlaggedStepsValue_) {
            const int overlapCount =
                metricValue(metrics,
                            QStringLiteral("measurement_window_overlap_count"),
                            QStringLiteral("count"))
                    .toInt();
            const int violationCount =
                metricValue(metrics, QStringLiteral("window_violation_count"), QStringLiteral("count"))
                    .toInt();
            metricFlaggedStepsValue_->setText(QString::number(overlapCount + violationCount));
        }
        if (metricDesignUsedCountValue_) {
            const QString insideCount =
                metricValue(metrics,
                            QStringLiteral("measurement_window_inside_image_count"),
                            QStringLiteral("count"));
            metricDesignUsedCountValue_->setText(insideCount.isEmpty() ? QStringLiteral("0")
                                                                       : insideCount);
        }
        refreshRegistrationToolState();
        refreshReportExportState(ok ? QStringLiteral("标准圆结果已生成，所选 CSV 可直接从结果目录查看。")
                                    : QStringLiteral("标准圆检测失败，请先检查日志和摘要 CSV。"));
        refreshStageSummaries();
        switchToStage(ReportStage);
        return;
    }

    const bool localSlot2dFallbackMode = lastRequest_.pipelineConfig.localSlotImageMode;
    if (localSlot2dFallbackMode) {
        completedSteps_.clear();
        if (stepModel_) {
            stepModel_->clear();
        }
        csvEdit_->setPlainText(csvText.isEmpty() ? readUtf8TextFile(lastRequest_.csvOutputPath) : csvText);
        if (qualityReviewEdit_) {
            const QString reviewText = readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath);
            qualityReviewEdit_->setPlainText(
                reviewText.isEmpty() ? QStringLiteral("本次局部槽运行未生成质量审查 CSV。") : reviewText);
        }
        if (candidateDiagnosticsEdit_) {
            const QString contourText = readUtf8TextFile(lastRequest_.contourPointsCsvOutputPath);
            candidateDiagnosticsEdit_->setPlainText(
                contourText.isEmpty() ? QStringLiteral("本次局部槽运行未生成轮廓点 CSV。") : contourText.left(12000));
        }

        const QImage contourPlot(displayOutputPath(lastRequest_.contourOverlayOutputPath));
        if (processingPreviewViewer_) {
            if (!contourPlot.isNull()) {
                processingPreviewViewer_->setImage(contourPlot);
            } else {
                processingPreviewViewer_->clearImage();
            }
        }
        if (singleSlotViewer_) {
            if (!contourPlot.isNull()) {
                singleSlotViewer_->setImage(contourPlot);
            } else {
                singleSlotViewer_->clearImage();
            }
        }
        if (processingPreviewInfoEdit_) {
            processingPreviewInfoEdit_->setPlainText(
                QStringLiteral("模块 3 特征提取：局部槽特征检测已完成二值化、骨架边缘提取和槽边缘拟合。\n输出图：%1")
                    .arg(displayOutputPath(lastRequest_.contourOverlayOutputPath)));
        }

        const QImage designPlot(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
        if (designCompareViewer_) {
            if (!designPlot.isNull()) {
                designCompareViewer_->setImage(designPlot);
            } else {
                designCompareViewer_->clearImage();
            }
        }
        const QString summaryCsv = readUtf8TextFile(lastRequest_.designErrorSummaryCsvOutputPath);
        const QString profileCsv = readUtf8TextFile(lastRequest_.designErrorProfileCsvOutputPath);
        const QHash<QString, QString> designSummaryMetrics = parseFirstCsvDataRow(summaryCsv);
        const QString localExceedanceThreshold =
            designSummaryMetrics.value(QStringLiteral("local_exceedance_threshold_um"), QStringLiteral("--"));
        const QString localExceedanceCount =
            designSummaryMetrics.value(QStringLiteral("local_exceedance_count"), QStringLiteral("--"));
        const QString localMaxAbsError =
            designSummaryMetrics.value(QStringLiteral("local_max_abs_error_um"), QStringLiteral("--"));
        const QString localMaxExceedance =
            designSummaryMetrics.value(QStringLiteral("local_max_exceedance_um"), QStringLiteral("--"));
        const QString localExceedanceSummary =
            QStringLiteral("局部超差标注：阈值 ±%1 um，超差点 %2 个，最大绝对误差 %3 um，最大超差量 %4 um。")
                .arg(localExceedanceThreshold,
                     localExceedanceCount,
                     localMaxAbsError,
                     localMaxExceedance);
        if (designCompareEdit_) {
            QStringList lines;
            lines << QStringLiteral("模块 5 误差分析：局部槽特征检测");
            lines << QString();
            lines << QStringLiteral("说明：本模式不再执行旧 ROI 或整件 CAD 母线比对；检测槽边缘与拟合理想槽边缘在统一截面内完成对齐。");
            lines << QStringLiteral("误差分析图：%1").arg(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
            lines << QStringLiteral("图中红色线段/红点为超过阈值的局部超差位置。");
            lines << localExceedanceSummary;
            lines << QString();
            lines << QStringLiteral("误差汇总 CSV");
            lines << QStringLiteral("----------------------------------------");
            lines << (summaryCsv.trimmed().isEmpty() ? QStringLiteral("未生成。") : summaryCsv.trimmed());
            lines << QString();
            lines << QStringLiteral("误差明细预览");
            lines << QStringLiteral("----------------------------------------");
            lines << (profileCsv.trimmed().isEmpty() ? QStringLiteral("未生成。") : profileCsv.left(5000));
            designCompareEdit_->setPlainText(lines.join('\n'));
        }

        const QImage compensationPlot(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
        if (compensationViewer_) {
            if (!compensationPlot.isNull()) {
                compensationViewer_->setImage(compensationPlot);
            } else {
                compensationViewer_->clearImage();
            }
        }
        if (pointCloudViewer_) {
            const QString pointCloudCsvPath =
                lastRequest_.design3dErrorCsvOutputPath.empty()
                    ? QString()
                    : displayOutputPath(lastRequest_.design3dErrorCsvOutputPath);
            QString pointCloudMessage;
            if (!pointCloudCsvPath.isEmpty() &&
                pointCloudViewer_->loadDesignErrorCsv(pointCloudCsvPath, &pointCloudMessage)) {
                if (!pointCloudMessage.isEmpty()) {
                    appendLog(QStringLiteral("[信息] %1").arg(pointCloudMessage));
                }
            } else {
                pointCloudViewer_->clearCloud(
                    pointCloudCsvPath.isEmpty()
                        ? QStringLiteral("本次局部槽运行未生成三维误差坐标CSV。")
                        : QStringLiteral("局部槽结果保持统一截面补偿图；当前CSV未形成可交互CAD XYZ点云。"));
            }
        }
        if (compensationSummaryEdit_) {
            const QString featureCsv = readUtf8TextFile(lastRequest_.designFeatureCompensationCsvOutputPath);
            const QString coordinateCsv =
                QDir::toNativeSeparators(QFileInfo(displayOutputPath(lastRequest_.designCompensationPlotOutputPath))
                                             .dir()
                                             .filePath(QStringLiteral("compensated_slot_edge_points.csv")));
            QStringList lines;
            lines << QStringLiteral("模块 6 补偿结算：局部槽特征检测");
            lines << QString();
            lines << QStringLiteral("补偿输出：检测槽边缘曲线、补偿后槽边缘曲线，以及补偿后实际槽边缘点坐标 CSV。");
            lines << QStringLiteral("CAD坐标输出：若已导入 CAD，CSV 中 `compensated_cad_x/y/z_mm` 为 CAD 坐标系下补偿后的绝对 XYZ 坐标。");
            lines << localExceedanceSummary;
            lines << QStringLiteral("补偿图：%1").arg(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
            lines << QStringLiteral("补偿 CSV：%1").arg(displayOutputPath(lastRequest_.designCompensationCsvOutputPath));
            lines << QStringLiteral("补偿后槽边缘点 CSV：%1").arg(coordinateCsv);
            lines << QString();
            lines << QStringLiteral("槽宽与补偿预览");
            lines << QStringLiteral("----------------------------------------");
            lines << (featureCsv.trimmed().isEmpty() ? QStringLiteral("未生成槽特征补偿 CSV。") : featureCsv.trimmed());
            compensationSummaryEdit_->setPlainText(lines.join('\n'));
        }

        QStringList summaryLines;
        summaryLines << QStringLiteral("第三方复检总览");
        summaryLines << QString();
        summaryLines << QStringLiteral("运行模式：局部槽特征检测");
        summaryLines << QStringLiteral("运行结果：%1").arg(ok ? QStringLiteral("成功") : QStringLiteral("失败"));
        summaryLines << QStringLiteral("已加载图像：%1").arg(loadedImageCount);
        summaryLines << QStringLiteral("已预处理图像：%1").arg(preprocessedImageCount);
        summaryLines << QStringLiteral("槽底标定宽度：%1 mm").arg(lastRequest_.pipelineConfig.localSlotBottomWidthMm, 0, 'f', 4);
        summaryLines << QStringLiteral("像素当量修正：%1 倍").arg(lastRequest_.pipelineConfig.localSlotPixelSizeScale, 0, 'f', 4);
        if (!lastRequest_.resultOutputDir.empty()) {
            summaryLines << QStringLiteral("结果目录：%1")
                                .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir)));
        }
        if (!message.isEmpty()) {
            summaryLines << QStringLiteral("运行消息：%1").arg(message);
        }
        summaryLines << QStringLiteral("模型配准自动判断：单张局部槽图，无需拼接；已按槽底特征点建立统一截面对齐。");
        summaryLines << QStringLiteral("特征提取图：%1").arg(displayOutputPath(lastRequest_.contourOverlayOutputPath));
        summaryLines << QStringLiteral("误差分析图：%1").arg(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
        summaryLines << QStringLiteral("补偿结算图：%1").arg(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
        summaryLines << localExceedanceSummary;
        summaryEdit_->setPlainText(summaryLines.join('\n'));

        const QString compensationCsvText = readUtf8TextFile(lastRequest_.designCompensationCsvOutputPath);
        const QString contourCsvText = readUtf8TextFile(lastRequest_.contourPointsCsvOutputPath);
        const QString reviewCsvText = readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath);
        const QString featureCsvText = readUtf8TextFile(lastRequest_.designFeatureCompensationCsvOutputPath);
        const bool cadReady =
            activeCadModelDocument_.valid ||
            (lastRequest_.pipelineConfig.designUseExternalProfile &&
             lastRequest_.pipelineConfig.designExternalProfileSamples.size() >= 2);
        const bool targetReady = lastRequest_.pipelineConfig.localSlotBottomWidthMm > 0.0;
        const bool contourReady = !contourPlot.isNull() && !contourCsvText.trimmed().isEmpty();
        const bool errorReady = !designPlot.isNull() &&
                                !summaryCsv.trimmed().isEmpty() &&
                                !profileCsv.trimmed().isEmpty();
        const bool compensationReady = !compensationPlot.isNull() &&
                                       compensationCsvText.contains(QStringLiteral("compensated_target_r_mm"));
        const bool cadCompensationReady =
            compensationCsvText.contains(QStringLiteral("compensated_cad_x_mm")) &&
            compensationCsvText.contains(QStringLiteral("compensated_cad_y_mm")) &&
            compensationCsvText.contains(QStringLiteral("compensated_cad_z_mm")) &&
            compensationCsvText.contains(QRegularExpression(QStringLiteral("\\n(?:[^,\\n]*,){6}1,")));

        QStringList acceptanceLines;
        acceptanceLines << QStringLiteral("第三方验收功能对齐 - 局部槽特征检测");
        acceptanceLines << QStringLiteral("说明：当前补偿图保持统一截面槽边缘曲线；CSV 同时输出 CAD 坐标系下补偿后 XYZ。");
        acceptanceLines << QString();
        acceptanceLines << QStringLiteral("| 表9测试要点 | 状态 | GUI入口/证据 |");
        acceptanceLines << QStringLiteral("|---|---|---|");
        acceptanceLines << QStringLiteral("| 数据接收与预处理(1) 导入CAD模型和目标尺寸 | %1 | 模块1 CAD/目标尺寸；CAD=%2；目标槽底宽度=%3 mm |")
                               .arg(cadReady && targetReady ? QStringLiteral("通过") : QStringLiteral("待补充"),
                                    cadReady ? QStringLiteral("已导入并生成CAD截面采样") : QStringLiteral("未导入"),
                                    QString::number(lastRequest_.pipelineConfig.localSlotBottomWidthMm, 'f', 4));
        acceptanceLines << QStringLiteral("| 数据接收与预处理(2) 采集或导入在线测量数据 | %1 | 模块2；图像=%2张；局部槽输入图=%3 |")
                               .arg(loadedImageCount > 0 ? QStringLiteral("通过") : QStringLiteral("待运行"),
                                    QString::number(loadedImageCount),
                                    displayOutputPath(lastRequest_.panoramaOutputPath));
        acceptanceLines << QStringLiteral("| 数据接收与预处理(3) 读取/显示/坐标转换/滤波 | %1 | 模块2-3；二值化/骨架边缘/离群过滤已执行；轮廓点CSV=%2；质量审查=%3 |")
                               .arg(preprocessedImageCount > 0 && !reviewCsvText.trimmed().isEmpty()
                                        ? QStringLiteral("通过")
                                        : QStringLiteral("待运行"),
                                    displayOutputPath(lastRequest_.contourPointsCsvOutputPath),
                                    displayOutputPath(lastRequest_.qualityReviewCsvOutputPath));
        acceptanceLines << QStringLiteral("| 模型配准与特征提取(1) 实测数据与数字化模型对齐 | %1 | 模块4/5；单张局部槽无需拼接，按槽底特征点建立统一截面对齐；CAD XYZ映射见 design_3d_error_points.csv |")
                               .arg(ok && errorReady ? QStringLiteral("通过") : QStringLiteral("待运行"));
        acceptanceLines << QStringLiteral("| 模型配准与特征提取(2) 提取槽宽、槽边和轮廓边界 | %1 | 模块3；单槽轮廓图=%2；槽特征CSV=%3 |")
                               .arg(contourReady && !featureCsvText.trimmed().isEmpty()
                                        ? QStringLiteral("通过")
                                        : QStringLiteral("待运行"),
                                    displayOutputPath(lastRequest_.contourOverlayOutputPath),
                                    displayOutputPath(lastRequest_.designFeatureCompensationCsvOutputPath));
        acceptanceLines << QStringLiteral("| 模型配准与特征提取(3) 记录特征参数、配准结果和可视化 | %1 | 模块3-5；误差分析图=%2；轮廓点/匹配点CSV=%3 |")
                               .arg(errorReady ? QStringLiteral("通过") : QStringLiteral("待运行"),
                                    displayOutputPath(lastRequest_.designComparisonOverlayOutputPath),
                                    displayOutputPath(lastRequest_.designErrorProfileCsvOutputPath));
        acceptanceLines << QStringLiteral("| 误差分析与补偿解算(1) 槽宽偏差、轮廓位置偏差和局部超差 | %1 | 模块5；局部超差=%2点，阈值±%3um；误差图=%4；误差明细=%5 |")
                               .arg(errorReady ? QStringLiteral("通过") : QStringLiteral("待运行"),
                                    localExceedanceCount,
                                    localExceedanceThreshold,
                                    displayOutputPath(lastRequest_.designComparisonOverlayOutputPath),
                                    displayOutputPath(lastRequest_.designErrorProfileCsvOutputPath));
        acceptanceLines << QStringLiteral("| 误差分析与补偿解算(2) 在工件模型坐标系下给出补偿量 | %1 | 模块6；补偿图=%2；补偿CSV=%3；补偿后XYZ字段=compensated_cad_x/y/z_mm；CAD补偿点=%4 |")
                               .arg(compensationReady && cadCompensationReady ? QStringLiteral("通过") : QStringLiteral("待补充"),
                                    displayOutputPath(lastRequest_.designCompensationPlotOutputPath),
                                    displayOutputPath(lastRequest_.designCompensationCsvOutputPath),
                                    cadCompensationReady ? QStringLiteral("已生成") : QStringLiteral("未生成"));
        setAcceptanceText(acceptanceLines.join('\n'));

        resetReportMetricCards();
        if (metricLoadedImagesValue_) {
            metricLoadedImagesValue_->setText(QString::number(std::max(loadedImageCount, 0)));
        }
        if (metricPreprocessedImagesValue_) {
            metricPreprocessedImagesValue_->setText(QString::number(std::max(preprocessedImageCount, 0)));
        }
        if (metricStepCountValue_) {
            metricStepCountValue_->setText(QStringLiteral("0"));
        }
        if (metricFlaggedStepsValue_) {
            metricFlaggedStepsValue_->setText(QStringLiteral("0"));
        }
        if (metricDesignUsedCountValue_) {
            const QHash<QString, QString> metrics = parseMetricCsvByKey(csvEdit_->toPlainText());
            const QString usedPoints = metricValue(metrics, QStringLiteral("used_edge_points"), QStringLiteral("count"));
            metricDesignUsedCountValue_->setText(usedPoints.isEmpty() ? QStringLiteral("0") : usedPoints);
        }
        refreshRegistrationToolState();
        refreshReportExportState(ok ? QStringLiteral("局部槽结果已生成，CSV 和图像均保存在结果目录。")
                                    : QStringLiteral("局部槽检测失败，请检查日志和输入图像。"));
        refreshStageSummaries();
        switchToStage(ReportStage);
        return;
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
    summaryLines << QStringLiteral("第三方复检总览");
    summaryLines << QString();
    summaryLines << QStringLiteral("运行模式：%1").arg(runModeLabel(currentRunMode_));
    summaryLines << QStringLiteral("运行结果：%1").arg(ok ? QStringLiteral("成功") : QStringLiteral("失败"));
    summaryLines << QStringLiteral("已加载图像：%1").arg(loadedImageCount);
    summaryLines << QStringLiteral("已预处理图像：%1").arg(preprocessedImageCount);
    summaryLines << QStringLiteral("模型配准自动判断：%1")
                         .arg(loadedImageCount <= 1
                                  ? QStringLiteral("单张图像，无需拼接，直接展示轮廓并进入 CAD 对齐")
                                  : QStringLiteral("多张图像，已执行/准备执行拼接配准"));
    if (!lastRequest_.resultOutputDir.empty()) {
        summaryLines << QStringLiteral("结果目录：%1")
                            .arg(QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir)));
    }
    summaryLines << QStringLiteral("拼接步数：%1").arg(completedSteps_.size());
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

    const bool designComparisonEnabled = lastRequest_.pipelineConfig.enableDesignComparison;
    if (runCache_ && runCache_->hasPreprocessedEdges() && runCache_->hasStitching() && designComparisonEnabled) {
        designAlignmentResult = pinjie::cad_design::compareMeasuredProfileToDesign(
            runCache_->preprocessedEdges,
            runCache_->stitching.imageTransforms,
            lastRequest_.pipelineConfig);
        if (designAlignmentResult.ok) {
            designSummaryPtr = &designAlignmentResult.summary;
            designCompareLines << QStringLiteral("设计母线比对结果");
            designCompareLines << QString();
            designCompareLines << QStringLiteral("设计来源：%1（%2，%3 点）")
                                      .arg(fromUtf8StdString(designSummaryPtr->designProfileMetadata.sourceType),
                                           fromUtf8StdString(designSummaryPtr->designProfileMetadata.sourceName))
                                      .arg(static_cast<qulonglong>(designSummaryPtr->designProfileMetadata.sampleCount));
            designCompareLines << QStringLiteral("CAD 轴向反向展开：%1").arg(designSummaryPtr->designReverseAxial ? QStringLiteral("是") : QStringLiteral("否"));
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
            if (!lastRequest_.designComparisonOverlayOutputPath.empty()) {
                designCompareLines << QStringLiteral("误差分析图：%1")
                                      .arg(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
            }
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
            if (!lastRequest_.designComparisonOverlayOutputPath.empty()) {
                summaryLines << QStringLiteral("误差分析图：%1")
                                .arg(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
            }
            if (!lastRequest_.design3dErrorCsvOutputPath.empty()) {
                summaryLines << QStringLiteral("三维误差坐标 CSV：%1")
                                .arg(displayOutputPath(lastRequest_.design3dErrorCsvOutputPath));
            }
            if (!readUtf8TextFile(lastRequest_.qualityReviewCsvOutputPath).trimmed().isEmpty()) {
                summaryLines << QStringLiteral("自动质量审查：详见“质量审查”页签");
            }

            if (designCompareEdit_) {
                designCompareEdit_->setPlainText(designCompareLines.join('\n'));
            }
            if (designCompareViewer_) {
                QImage designComparisonPlot;
                if (!lastRequest_.designComparisonOverlayOutputPath.empty()) {
                    designComparisonPlot.load(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
                }
                if (!designComparisonPlot.isNull()) {
                    designCompareViewer_->setImage(designComparisonPlot);
                } else {
                    designCompareViewer_->clearImage();
                    if (!lastRequest_.designComparisonOverlayOutputPath.empty()) {
                        appendLog(QStringLiteral("[警告] 误差分析图未能加载：%1")
                                      .arg(displayOutputPath(lastRequest_.designComparisonOverlayOutputPath)));
                    }
                }
            }
            if (processingPreviewViewer_) {
                QImage contourPreview;
                if (!lastRequest_.contourOverlayOutputPath.empty()) {
                    contourPreview.load(displayOutputPath(lastRequest_.contourOverlayOutputPath));
                }
                if (!contourPreview.isNull()) {
                    processingPreviewViewer_->setImage(contourPreview);
                    if (processingPreviewInfoEdit_) {
                        processingPreviewInfoEdit_->setPlainText(
                            QStringLiteral("模块 3 特征提取：已显示槽类特征 ROI/轮廓提取图。\n输出图：%1")
                                .arg(displayOutputPath(lastRequest_.contourOverlayOutputPath)));
                    }
                } else if (!lastRequest_.contourOverlayOutputPath.empty()) {
                        appendLog(QStringLiteral("[警告] 特征提取轮廓图未能加载：%1")
                                  .arg(displayOutputPath(lastRequest_.contourOverlayOutputPath)));
                }
                if (singleSlotViewer_) {
                    if (!contourPreview.isNull()) {
                        singleSlotViewer_->setImage(contourPreview);
                    } else {
                        singleSlotViewer_->clearImage();
                    }
                }
            }
            if (compensationViewer_) {
                QImage compensationPlot;
                if (!lastRequest_.designCompensationPlotOutputPath.empty()) {
                    compensationPlot.load(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
                }
                if (!compensationPlot.isNull()) {
                    compensationViewer_->setImage(compensationPlot);
                } else {
                    compensationViewer_->clearImage();
                    if (!lastRequest_.designCompensationPlotOutputPath.empty()) {
                        appendLog(QStringLiteral("[警告] 补偿结算图未能加载：%1")
                                      .arg(displayOutputPath(lastRequest_.designCompensationPlotOutputPath)));
                    }
                }
            }
            if (pointCloudViewer_) {
                const QString pointCloudCsvPath =
                    lastRequest_.design3dErrorCsvOutputPath.empty()
                        ? QString()
                        : displayOutputPath(lastRequest_.design3dErrorCsvOutputPath);
                QString pointCloudMessage;
                if (!pointCloudCsvPath.isEmpty() &&
                    pointCloudViewer_->loadDesignErrorCsv(pointCloudCsvPath, &pointCloudMessage)) {
                    if (!pointCloudMessage.isEmpty()) {
                        appendLog(QStringLiteral("[信息] %1").arg(pointCloudMessage));
                    }
                } else {
                    pointCloudViewer_->clearCloud(
                        pointCloudCsvPath.isEmpty()
                            ? QStringLiteral("本次运行未生成三维误差坐标CSV，无法显示交互3D点云。")
                            : QStringLiteral("未能加载交互3D点云，请检查三维误差坐标CSV。"));
                    if (!pointCloudCsvPath.isEmpty()) {
                        appendLog(QStringLiteral("[警告] 交互3D点云未能加载：%1%2")
                                      .arg(pointCloudCsvPath,
                                           pointCloudMessage.isEmpty()
                                               ? QString()
                                               : QStringLiteral("；%1").arg(pointCloudMessage)));
                    }
                }
            }
            if (compensationSummaryEdit_) {
                const std::size_t cadCompensationRows =
                    static_cast<std::size_t>(std::count_if(designAlignmentResult.profilePoints.begin(),
                                                           designAlignmentResult.profilePoints.end(),
                                                           [](const auto& point) {
                                                               return point.isUsed && point.hasCadCoordinates;
                                                           }));
                QStringList compensationLines;
                compensationLines << QStringLiteral("模块 6 补偿解算");
                compensationLines << QString();
                compensationLines << QStringLiteral("点级补偿：%1 个有效 CAD 坐标补偿点 / %2 个设计比对有效点")
                                         .arg(static_cast<qulonglong>(cadCompensationRows))
                                         .arg(static_cast<qulonglong>(designSummaryPtr->usedCount));
                compensationLines << QStringLiteral("CAD模型坐标：轴向 %1，径向 %2，轴向原点 %3 mm，方向系数 %4")
                                         .arg(fromUtf8StdString(designSummaryPtr->designProfileMetadata.axialAxis),
                                              fromUtf8StdString(designSummaryPtr->designProfileMetadata.radialAxis))
                                         .arg(designSummaryPtr->designProfileMetadata.cadAxialOriginMm, 0, 'f', 6)
                                         .arg(designSummaryPtr->designProfileMetadata.cadAxialDirectionSign, 0, 'f', 0);
                compensationLines << QStringLiteral("补偿约定：CSV 中 `compensated_cad_x/y/z_mm` 为 CAD 坐标系下补偿后的绝对 XYZ 坐标。");
                compensationLines << QStringLiteral("字段说明：`delta_x/y/z_um` 表示从实测映射点移动到该 CAD 目标坐标的补偿量。");
                compensationLines << QString();
                compensationLines << QStringLiteral("输出文件");
                compensationLines << QStringLiteral("补偿量可视化 PNG：%1")
                                         .arg(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
                compensationLines << QStringLiteral("3D 点云 PNG：%1")
                                         .arg(designPointCloudPngPath(lastRequest_));
                compensationLines << QStringLiteral("CAD 点级补偿 CSV：%1")
                                          .arg(displayOutputPath(lastRequest_.designCompensationCsvOutputPath));
                compensationLines << QStringLiteral("三维误差坐标 CSV：%1")
                                          .arg(displayOutputPath(lastRequest_.design3dErrorCsvOutputPath));
                compensationLines << QStringLiteral("槽特征补偿 CSV：%1")
                                         .arg(displayOutputPath(lastRequest_.designFeatureCompensationCsvOutputPath));
                compensationLines << QStringLiteral("设计误差汇总 CSV：%1")
                                         .arg(displayOutputPath(lastRequest_.designErrorSummaryCsvOutputPath));
                compensationLines << QString();
                compensationLines << QStringLiteral("槽特征补偿预览");
                compensationLines << QStringLiteral("----------------------------------------");
                const QString featureText =
                    fromUtf8StdString(designAlignmentResult.featureCompensationCsvText).trimmed();
                if (featureText.contains(QStringLiteral("single_slot_width_target"))) {
                    compensationLines << QStringLiteral("单槽宽目标解算：已按输入目标槽宽输出宽度误差与补偿量。");
                    compensationLines << QString();
                }
                compensationLines << (featureText.isEmpty()
                                          ? QStringLiteral("本次运行未生成槽特征补偿。")
                                          : featureText);
                compensationSummaryEdit_->setPlainText(compensationLines.join('\n'));
            }
        } else {
            summaryLines << QString();
            summaryLines << QStringLiteral("设计母线比对：%1")
                                .arg(fromUtf8StdString(designAlignmentResult.message));
            if (designCompareEdit_) {
                designCompareEdit_->setPlainText(
                    QStringLiteral("正式 CAD 误差分析失败。\n%1\n\n"
                                   "请检查：CAD 是否已导入、临时像素当量/标定是否有效、检测轮廓是否有足够点与 CAD 轮廓对齐。\n"
                                   "该状态不会生成兜底误差图；修正配置后需要重新运行全流程或误差分析模块。")
                        .arg(fromUtf8StdString(designAlignmentResult.message)));
            }
            if (designCompareViewer_) {
                designCompareViewer_->clearImage();
            }
            if (processingPreviewViewer_) {
                QImage contourPreview;
                if (!lastRequest_.contourOverlayOutputPath.empty()) {
                    contourPreview.load(displayOutputPath(lastRequest_.contourOverlayOutputPath));
                }
                if (!contourPreview.isNull()) {
                    processingPreviewViewer_->setImage(contourPreview);
                    if (processingPreviewInfoEdit_) {
                        processingPreviewInfoEdit_->setPlainText(
                            QStringLiteral("模块 3 特征提取：已显示当前可用轮廓图；CAD 误差分析未通过。\n输出图：%1")
                                .arg(displayOutputPath(lastRequest_.contourOverlayOutputPath)));
                    }
                }
            }
            if (compensationViewer_) {
                compensationViewer_->clearImage();
            }
            if (pointCloudViewer_) {
                pointCloudViewer_->clearCloud();
            }
            if (compensationSummaryEdit_) {
                compensationSummaryEdit_->setPlainText(
                    QStringLiteral("模块 6 补偿解算未生成 CAD 点级补偿。\n%1\n\n"
                                   "原因：正式 CAD 对齐未通过，因此不会输出 design_compensation.csv 或补偿图。\n"
                                   "请先在模块 1 导入 CAD 并确认三维坐标预览，再运行全流程或模块 5 误差分析。")
                        .arg(fromUtf8StdString(designAlignmentResult.message)));
            }
        }
    } else if (runCache_ && runCache_->hasPreprocessedEdges() && runCache_->hasStitching()) {
        summaryLines << QString();
        summaryLines << QStringLiteral("设计母线比对：本次未启用");
        summaryLines << QStringLiteral("说明：已完成工件拼接与常规结果输出；如需误差分析和补偿解算，请先导入 CAD 后运行全流程或误差分析模块。");
        if (designCompareEdit_) {
            designCompareEdit_->setPlainText(
                QStringLiteral("本次运行未启用 CAD 误差分析。\n\n"
                               "当前结果：工件拼接与常规输出已完成。\n"
                               "如需继续生成设计比对图、误差统计和补偿 CSV，请先在模块 1 导入可用 CAD，再运行全流程或模块 5。"));
        }
        if (designCompareViewer_) {
            designCompareViewer_->clearImage();
        }
        if (compensationViewer_) {
            compensationViewer_->clearImage();
        }
        if (pointCloudViewer_) {
            pointCloudViewer_->clearCloud(QStringLiteral("本次未启用 CAD 误差分析，未生成三维误差点云。"));
        }
        if (compensationSummaryEdit_) {
            compensationSummaryEdit_->setPlainText(
                QStringLiteral("模块 6 补偿解算未执行。\n\n"
                               "原因：本次运行未启用 CAD 设计基准，因此不会生成补偿图、design_compensation.csv 或三维误差坐标 CSV。\n"
                               "若需要补偿结果，请先导入 CAD 后重新运行。"));
        }
    } else {
        if (designCompareEdit_) {
            designCompareEdit_->setPlainText(QStringLiteral("等待拼接结果生成后再进行设计母线比对。"));
        }
        if (designCompareViewer_) {
            designCompareViewer_->clearImage();
        }
        if (compensationViewer_) {
            compensationViewer_->clearImage();
        }
        if (pointCloudViewer_) {
            pointCloudViewer_->clearCloud();
        }
        if (compensationSummaryEdit_) {
            compensationSummaryEdit_->setPlainText(QStringLiteral("模块 6 补偿解算：等待拼接结果与误差分析完成。"));
        }
    }

    setAcceptanceText(buildTable9AcceptanceText(ok,
                                                loadedImageCount,
                                                preprocessedImageCount,
                                                designAlignmentResult.ok ? &designAlignmentResult : nullptr,
                                                designSummaryPtr));

    summaryEdit_->setPlainText(summaryLines.join('\n'));
    bottomTabs_->setCurrentWidget(summaryEdit_);
    updateReportMetricCards(loadedImageCount, preprocessedImageCount, qualitySummaryPtr, designSummaryPtr);
    refreshRegistrationToolState();
    refreshReportExportState(ok ? QStringLiteral("归档状态：结果数据已就绪；误差分析与补偿解算证据已分别写入结果目录。")
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
    if (designModelPanel_) {
        designModelPanel_->setRunning(running);
    }
    configPanel_->setRunning(running);
    const bool standardCircleMode = configPanel_ && configPanel_->standardCircleModeEnabled();
    pinjie::StitchRunRequest previewRequest;
    QString previewError;
    const bool syntheticInput =
        configPanel_ && configPanel_->buildRequest(previewRequest, previewError, false) &&
        isSingleSlotTestInput(previewRequest);
    const bool localSlotImageMode = hasLocalSlotImageMode();
    const bool canRun = hasMeasurementBaseline() || standardCircleMode || syntheticInput || localSlotImageMode;
    if (startButton_) {
        startButton_->setEnabled(!running && canRun);
    }
    if (moduleRunButton_) {
        const int currentStage = stageStack_ ? stageStack_->currentIndex() : CalibrationStage;
        moduleRunButton_->setEnabled(!running && (currentStage == CalibrationStage || canRun));
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
    refreshDesignModelDetailPane();
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

    pinjie::cad_model::DesignModelRequest designRequest;
    QString designError;
    const bool designConfigValid = designModelPanel_ && designModelPanel_->buildRequest(designRequest, designError);

    pinjie::StitchRunRequest request;
    QString errorMessage;
    const bool valid = configPanel_->buildRequest(request, errorMessage, false);
    const bool standardCircleMode = valid && isStandardCircleMode(request);
    double temporaryPixelSizeMm =
        activeCadModelDocument_.valid && designConfigValid ? designRequest.temporaryPixelSizeMm
                                                           : lastDesignModelRequest_.temporaryPixelSizeMm;
    if (valid) {
        const double testPixelSizeMm = singleSlotTestPixelSizeMm(request);
        if (testPixelSizeMm > 0.0) {
            temporaryPixelSizeMm = testPixelSizeMm;
        }
    }
    const bool temporaryPixelSizeMode = activeCadModelDocument_.valid && temporaryPixelSizeMm > 0.0;
    const bool localSlotImageMode = designConfigValid && designRequest.localSlotImageMode;

    if (!hasActiveCalibration() && !standardCircleMode && !temporaryPixelSizeMode && !localSlotImageMode) {
        workflowStateLabel_->setText(QStringLiteral("复检流程：未加载标定结果；如仅测试 CAD/图片链路，可在模块 1 填写临时像素当量后导入 CAD。"));
        registrationPresetLabel_->setText(calibrationConfigValid
                                              ? QStringLiteral("标定：参数已就绪")
                                              : QStringLiteral("标定：参数缺失"));
    } else if (localSlotImageMode) {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("复检流程：局部槽特征检测 | %1 张图像 | 槽底宽度 %2 mm 标定")
                        .arg(request.imagePaths.size())
                        .arg(designRequest.localSlotBottomWidthMm, 0, 'f', 4)
                  : QStringLiteral("复检流程：局部槽特征检测 | 运行配置待补充"));
        registrationPresetLabel_->setText(QStringLiteral("局部槽：二值化 + 骨架边缘 + 特征点对齐 + 补偿输出"));
    } else if (standardCircleMode && !hasActiveCalibration()) {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("复检流程：标准圆国标检测模式 | %1 张图像待检测 | 无需预先标定")
                        .arg(request.imagePaths.size())
                  : QStringLiteral("复检流程：标准圆国标检测模式 | 运行配置待补充"));
        registrationPresetLabel_->setText(QStringLiteral("标准圆：掩模板预处理 + 拼接修正 + 椭圆修正 + GB/T 25窗选点"));
    } else if (temporaryPixelSizeMode && !hasActiveCalibration()) {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("复检流程：临时像素当量测试模式 | %1 张图像可预处理/配准")
                        .arg(request.imagePaths.size())
                  : QStringLiteral("复检流程：临时像素当量测试模式 | 运行配置待补充"));
        registrationPresetLabel_->setText(
            QStringLiteral("临时像素当量：%1 mm/px；正式检测请加载真实标定")
                .arg(temporaryPixelSizeMm, 0, 'f', 9));
    } else {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("复检流程：%1 | %2 张图像可拼接")
                        .arg(calibrationIdentityLine(activeCalibrationCache_))
                        .arg(request.imagePaths.size())
                  : QStringLiteral("复检流程：%1 | 运行配置待补充")
                        .arg(calibrationIdentityLine(activeCalibrationCache_)));
        registrationPresetLabel_->setText(QStringLiteral("标定：%1").arg(calibrationQualityLine(activeCalibrationCache_)));
    }

    if (!calibrationConfigValid) {
        calibrationOverviewEdit_->setPlainText(QStringLiteral("配置检查：未通过\n原因：%1").arg(calibrationError));
    } else {
        QStringList calibrationLines;
        calibrationLines << QStringLiteral("模块 1 CAD/目标尺寸：量值标定配置已就绪");
        calibrationLines << QStringLiteral("表9对应：数据接收与预处理(1)(3)，为目标尺寸导入、坐标转换和滤波提供量值基准。");
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

    if (!designConfigValid) {
        designModelOverviewEdit_->setPlainText(QStringLiteral("配置检查：未通过\n原因：%1").arg(designError));
    } else {
        QStringList designLines;
        designLines << QStringLiteral("模块 1 CAD/目标尺寸：CAD基准配置已就绪");
        designLines << QStringLiteral("表9对应：数据接收与预处理(1)，导入典型槽类 CAD 模型和目标尺寸。");
        designLines << QString();
        designLines << QStringLiteral("CAD 文件：%1")
                           .arg(QDir::toNativeSeparators(fromUtf8StdString(designRequest.cadFilePath)));
        designLines << QStringLiteral("模型名称：%1")
                           .arg(designRequest.modelName.empty()
                                    ? QStringLiteral("未命名")
                                    : fromUtf8StdString(designRequest.modelName));
        designLines << QStringLiteral("工件坐标：轴向 %1，径向 %2")
                           .arg(fromUtf8StdString(designRequest.axialAxis))
                           .arg(fromUtf8StdString(designRequest.radialAxis));
        designLines << QStringLiteral("CAD 轴向展开：%1")
                           .arg(designRequest.reverseAxialDirection ? QStringLiteral("反向展开")
                                                                    : QStringLiteral("正向展开（默认）"));
        designLines << QStringLiteral("左端点原点：%1")
                           .arg(designRequest.useLeftEndpointAsOrigin ? QStringLiteral("启用") : QStringLiteral("关闭"));
        designLines << QStringLiteral("上包络母线：%1")
                           .arg(designRequest.extractUpperEnvelope ? QStringLiteral("启用") : QStringLiteral("关闭"));
        designLines << QStringLiteral("截面采样步长：%1 mm").arg(designRequest.profileSamplingStepMm, 0, 'f', 3);
        if (designRequest.targetSlotWidthMm > 0.0) {
            designLines << QStringLiteral("目标槽宽：%1 mm").arg(designRequest.targetSlotWidthMm, 0, 'f', 4);
        }
        if (designRequest.targetSlotDepthMm > 0.0) {
            designLines << QStringLiteral("目标槽深：%1 mm").arg(designRequest.targetSlotDepthMm, 0, 'f', 4);
        }
        designLines << QStringLiteral("导入状态：%1")
                           .arg(activeCadModelDocument_.valid
                                     ? QStringLiteral("已加载 %1，真实截面 %2 点，比对轮廓 %3 点")
                                           .arg(fromUtf8StdString(activeCadModelDocument_.fileName))
                                           .arg(static_cast<qulonglong>(activeCadModelDocument_.sectionSamples.size()))
                                           .arg(static_cast<qulonglong>(activeCadModelDocument_.profileSamples.size()))
                                     : QStringLiteral("待导入"));
        designModelOverviewEdit_->setPlainText(designLines.join('\n'));
    }

    if (!valid) {
        acquisitionOverviewEdit_->setPlainText(QStringLiteral("模块 2 数据接收预处理：配置未通过\n原因：%1").arg(errorMessage));
        processingOverviewEdit_->setPlainText(QStringLiteral("模块 3 特征提取：等待图像输入、预处理参数和轮廓提取参数就绪。"));
        registrationOverviewEdit_->setPlainText(QStringLiteral("模块 4 模型配准：等待配准参数与结果导出设置就绪。"));
        return;
    }

    QStringList acquisitionLines;
    acquisitionLines << QStringLiteral("模块 2 数据接收预处理：配置已就绪");
    acquisitionLines << QStringLiteral("表9对应：数据接收与预处理(2)(3)，采集/导入在线测量数据并准备读取显示。");
    acquisitionLines << QString();
    acquisitionLines << QStringLiteral("标定结果：%1")
                            .arg(standardCircleMode && !hasActiveCalibration()
                                     ? QStringLiteral("标准圆模式不需要")
                                     : (hasActiveCalibration()
                                            ? fromUtf8StdString(activeCalibrationCache_.profile.profileName)
                                            : (temporaryPixelSizeMode
                                                   ? QStringLiteral("临时像素当量 %1 mm/px")
                                                         .arg(temporaryPixelSizeMm, 0, 'f', 9)
                                                   : QStringLiteral("未加载"))));
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
    processingLines << QStringLiteral("模块 3 特征提取参数");
    processingLines << QStringLiteral("表9对应：数据接收与预处理(3)、模型配准与特征提取(2)，输出轮廓边界、槽宽/槽位特征。");
    processingLines << QString();
    processingLines << QStringLiteral("自动判断：%1")
                           .arg(request.pipelineConfig.designTargetSlotWidthMm > 0.0
                                    ? QStringLiteral("槽类特征识别，提取单槽 ROI、槽宽边界和槽底候选。")
                                    : QStringLiteral("普通工件轮廓提取，输出整体轮廓用于后续 CAD 配准。"));
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

    QStringList registrationLines;
    registrationLines << QStringLiteral("模块 4 模型配准参数");
    registrationLines << QStringLiteral("表9对应：模型配准与特征提取(1)(3)，完成实测数据与数字化模型对齐并记录配准结果。");
    registrationLines << QString();
    if (standardCircleMode) {
        const auto& standardCircle = request.standardCircleConfig;
        registrationLines << QStringLiteral("检测模式：标准圆国标 25 视场检测");
        registrationLines << QStringLiteral("图像序列：%1%2%3 ...")
                                 .arg(QString::fromStdString(standardCircle.imagePrefix))
                                 .arg(standardCircle.startIndex)
                                 .arg(QString::fromStdString(standardCircle.imageExtension));
        registrationLines << QStringLiteral("标准圆直径：%1 mm").arg(standardCircle.sphereDiameterMm, 0, 'f', 5);
        registrationLines << QStringLiteral("视场：%1 mm x %2 mm")
                                 .arg(standardCircle.horizontalFieldOfViewMm, 0, 'f', 3)
                                 .arg(standardCircle.verticalFieldOfViewMm, 0, 'f', 3);
        registrationLines << QStringLiteral("采集重叠率：%1%")
                                 .arg(standardCircle.overlapRatio * 100.0, 0, 'f', 2);
        registrationLines << QStringLiteral("测量窗口半尺寸：%1 px")
                                 .arg(standardCircle.windowHalfSizePx, 0, 'f', 1);
        registrationLines << QStringLiteral("环向中值半径：%1")
                                 .arg(standardCircle.circularMedianFilterRadius);
        registrationLines << QStringLiteral("滤波混合系数：%1")
                                 .arg(standardCircle.circularFilterBlend, 0, 'f', 3);
        registrationLines << QStringLiteral("流程说明：调用标准圆专用检测链，不使用当前工件设计比对链。");
    } else {
        const auto& pipeline = request.pipelineConfig;
        registrationLines << QStringLiteral("自动判断：%1")
                                 .arg(request.imagePaths.size() <= 1
                                          ? QStringLiteral("单张图像，无需拼接，直接展示轮廓并进入 CAD 对齐。")
                                          : QStringLiteral("多张图像，运行拼接并输出配准候选、风险步和调试图。"));
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
    }

    const QStringList csvLabels = selectedPaperCsvLabels(configPanel_);
    registrationLines << QStringLiteral("导出项目：%1")
                             .arg(csvLabels.isEmpty() ? QStringLiteral("未选择")
                                                      : csvLabels.join(QStringLiteral(", ")));
    registrationLines << QStringLiteral("结果目录：%1").arg(displayOutputPath(request.resultOutputDir));

    if (hasActiveCalibration()) {
        registrationLines << QString();
        registrationLines << QStringLiteral("标定质量：");
        registrationLines << calibrationQualityLine(activeCalibrationCache_);
    } else if (standardCircleMode) {
        registrationLines << QString();
        registrationLines << QStringLiteral("标定状态：标准圆国标检测模式下可直接用标准圆直径做像素当量修正。");
    } else if (temporaryPixelSizeMode) {
        registrationLines << QString();
        registrationLines << QStringLiteral("标定状态：使用临时像素当量 %1 mm/px；仅建议用于功能测试。")
                                 .arg(temporaryPixelSizeMm, 0, 'f', 9);
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

void MainWindow::refreshDesignModelDetailPane()
{
    if (!designModelDetailEdit_) {
        return;
    }

    QStringList lines;
    lines << QStringLiteral("CAD 后端：%1")
                 .arg(fromUtf8StdString(pinjie::cad_model::occtCadImportBackendSummary()));

    pinjie::cad_model::DesignModelRequest request;
    QString errorMessage;
    const bool requestReady = designModelPanel_ && designModelPanel_->buildRequest(request, errorMessage);

    if (!activeCadModelDocument_.valid) {
        lines << QString();
        lines << QStringLiteral("当前尚未导入设计模型。");
        if (requestReady) {
            lines << QStringLiteral("已选择文件：%1")
                            .arg(QDir::toNativeSeparators(fromUtf8StdString(request.cadFilePath)));
            lines << QStringLiteral("点击“导入设计模型”后，将在这里显示 STEP/STP/IGES 解析摘要、包围盒和后续截面准备信息。");
        } else if (!errorMessage.isEmpty()) {
            lines << QStringLiteral("当前配置未完成：%1").arg(errorMessage);
        }
        designModelDetailEdit_->setPlainText(lines.join('\n'));
        return;
    }

    lines << QString();
    lines << QStringLiteral("模型摘要");
    lines << QStringLiteral("----------------------------------------");
    lines << fromUtf8StdString(pinjie::cad_model::buildCadModelSummaryText(activeCadModelDocument_)).trimmed();
    lines << QString();
    lines << QStringLiteral("当前提取策略");
    lines << QStringLiteral("----------------------------------------");
    lines << QStringLiteral("轴向 / 径向：%1 / %2")
                    .arg(fromUtf8StdString(activeCadModelDocument_.axialAxis))
                    .arg(fromUtf8StdString(activeCadModelDocument_.radialAxis));
    lines << QStringLiteral("CAD 轴向展开：%1")
                    .arg(lastDesignModelRequest_.reverseAxialDirection ? QStringLiteral("反向展开")
                                                                       : QStringLiteral("正向展开（默认）"));
    lines << QStringLiteral("左端点原点：%1")
                    .arg(lastDesignModelRequest_.useLeftEndpointAsOrigin ? QStringLiteral("启用")
                                                                         : QStringLiteral("关闭"));
    lines << QStringLiteral("上包络母线：%1")
                    .arg(lastDesignModelRequest_.extractUpperEnvelope ? QStringLiteral("启用")
                                                                      : QStringLiteral("关闭"));
    lines << QStringLiteral("截面采样步长：%1 mm").arg(lastDesignModelRequest_.profileSamplingStepMm, 0, 'f', 3);
    if (activeCadModelDocument_.hasProfileSamples) {
        if (activeCadModelDocument_.hasSectionSamples) {
            lines << QStringLiteral("CAD真实截面预览：%1 个原始剖切点，静态图和3D页优先显示这组点。")
                            .arg(static_cast<qulonglong>(activeCadModelDocument_.sectionSamples.size()));
        } else {
            lines << QStringLiteral("CAD真实截面预览：当前模型未生成可显示的原始剖切点。");
        }
        lines << QStringLiteral("CAD误差比对轮廓：%1 点，s=%2~%3 mm，r=%4~%5 mm（由真实截面分箱/特征提取得到，仅用于后续误差比对）")
                        .arg(static_cast<qulonglong>(activeCadModelDocument_.profileSamples.size()))
                        .arg(activeCadModelDocument_.profileMinSMm, 0, 'f', 3)
                        .arg(activeCadModelDocument_.profileMaxSMm, 0, 'f', 3)
                        .arg(activeCadModelDocument_.profileMinRMm, 0, 'f', 3)
                        .arg(activeCadModelDocument_.profileMaxRMm, 0, 'f', 3);
        lines << QStringLiteral("CAD 轮廓提取：%1，截面 %2 = %3 mm")
                        .arg(fromUtf8StdString(activeCadModelDocument_.profileMetadata.extractionMethod))
                        .arg(fromUtf8StdString(activeCadModelDocument_.profileMetadata.sectionNormalAxis))
                        .arg(activeCadModelDocument_.profileMetadata.sectionCoordinateMm, 0, 'f', 6);
        lines << QStringLiteral("比对坐标换算：%1 = %2 + (%3) * s，%4 = r；预览主图按 CAD X/Y/Z 原始坐标显示")
                        .arg(fromUtf8StdString(activeCadModelDocument_.profileMetadata.axialAxis))
                        .arg(activeCadModelDocument_.profileMetadata.cadAxialOriginMm, 0, 'f', 6)
                        .arg(activeCadModelDocument_.profileMetadata.cadAxialDirectionSign, 0, 'f', 0)
                        .arg(fromUtf8StdString(activeCadModelDocument_.profileMetadata.radialAxis));
        if (!lastCadPreviewImagePath_.isEmpty()) {
            lines << QStringLiteral("CAD真实截面预览图：%1").arg(lastCadPreviewImagePath_);
        }
    } else {
        lines << QStringLiteral("CAD 设计轮廓：未生成，运行时将回退到内置母线。");
    }
    if (lastDesignModelRequest_.targetSlotWidthMm > 0.0 || lastDesignModelRequest_.targetSlotDepthMm > 0.0) {
        lines << QStringLiteral("目标槽宽 / 槽深：%1 mm / %2 mm")
                        .arg(lastDesignModelRequest_.targetSlotWidthMm, 0, 'f', 4)
                        .arg(lastDesignModelRequest_.targetSlotDepthMm, 0, 'f', 4);
    } else {
        lines << QStringLiteral("目标槽宽 / 槽深：当前未填写，将在后续特征定义阶段继续补充。");
    }
    if (lastDesignModelRequest_.temporaryPixelSizeMm > 0.0) {
        lines << QStringLiteral("临时像素当量：%1 mm/px（测试覆盖模块 1 标定值）")
                        .arg(lastDesignModelRequest_.temporaryPixelSizeMm, 0, 'f', 9);
    } else if (hasActiveCalibration()) {
        const double pixelSizeMm = calibrationPixelSizeMm(activeCalibrationCache_);
        if (pixelSizeMm > 0.0) {
            lines << QStringLiteral("设计比对像素当量：%1 mm/px（来自模块 1 标定）").arg(pixelSizeMm, 0, 'f', 9);
        }
    }
    lines << QString();
    lines << QStringLiteral("当前说明：CAD展示使用真实剖切截面；误差分析使用处理后的比对轮廓，并输出 CAD 模型 XYZ 坐标下的补偿后绝对坐标；单槽宽目标模式可用于复检一个槽宽。");

    designModelDetailEdit_->setPlainText(lines.join('\n'));
}

void MainWindow::applyActiveDesignModelToRequest(pinjie::StitchRunRequest& request) const
{
    const bool hasProfileSamples =
        activeCadModelDocument_.hasProfileSamples && activeCadModelDocument_.profileSamples.size() >= 2;
    const bool hasSectionSamples =
        activeCadModelDocument_.hasSectionSamples && activeCadModelDocument_.sectionSamples.size() >= 2;
    if (!activeCadModelDocument_.valid || (!hasProfileSamples && !hasSectionSamples)) {
        return;
    }

    pinjie::cad_model::DesignModelRequest currentDesignRequest = lastDesignModelRequest_;
    QString designError;
    if (designModelPanel_) {
        pinjie::cad_model::DesignModelRequest panelRequest;
        if (designModelPanel_->buildRequest(panelRequest, designError)) {
            currentDesignRequest.targetSlotWidthMm = panelRequest.targetSlotWidthMm;
            currentDesignRequest.targetSlotDepthMm = panelRequest.targetSlotDepthMm;
            currentDesignRequest.temporaryPixelSizeMm = panelRequest.temporaryPixelSizeMm;
            currentDesignRequest.localSlotImageMode = panelRequest.localSlotImageMode;
            currentDesignRequest.localSlotBottomWidthMm = panelRequest.localSlotBottomWidthMm;
        }
    }

    request.pipelineConfig.designUseExternalProfile = true;
    request.pipelineConfig.designExternalProfileSamples =
        currentDesignRequest.localSlotImageMode && hasSectionSamples
            ? activeCadModelDocument_.sectionSamples
            : (hasProfileSamples ? activeCadModelDocument_.profileSamples : activeCadModelDocument_.sectionSamples);
    request.pipelineConfig.designProfileMetadata = activeCadModelDocument_.profileMetadata;
    request.pipelineConfig.designReverseAxial = lastDesignModelRequest_.reverseAxialDirection;
    request.pipelineConfig.designUseLeftEndpointAnchor = lastDesignModelRequest_.useLeftEndpointAsOrigin;
    request.pipelineConfig.designAnchorRadialToLeftEndpoint = lastDesignModelRequest_.useLeftEndpointAsOrigin;
    request.pipelineConfig.designUseUpperEnvelope = lastDesignModelRequest_.extractUpperEnvelope;
    request.pipelineConfig.designIgnoreStepTransition = false;
    request.pipelineConfig.designTargetSlotWidthMm = currentDesignRequest.targetSlotWidthMm;
    request.pipelineConfig.designTargetSlotDepthMm = currentDesignRequest.targetSlotDepthMm;
    request.pipelineConfig.localSlotImageMode = currentDesignRequest.localSlotImageMode;
    request.pipelineConfig.localSlotBottomWidthMm = currentDesignRequest.localSlotBottomWidthMm;
    if (request.pipelineConfig.localSlotImageMode &&
        request.pipelineConfig.designTargetSlotWidthMm <= 0.0) {
        request.pipelineConfig.designTargetSlotWidthMm = currentDesignRequest.localSlotBottomWidthMm;
    }
    request.pipelineConfig.designUseCentralSlotImageRoi = false;
    if (currentDesignRequest.temporaryPixelSizeMm > 0.0) {
        request.pipelineConfig.designPixelSizeMm = currentDesignRequest.temporaryPixelSizeMm;
        request.pipelineConfig.designFilterEndFaceEdges = false;
    }
    const double testPixelSizeMm = singleSlotTestPixelSizeMm(request);
    if (testPixelSizeMm > 0.0) {
        request.pipelineConfig.designPixelSizeMm = testPixelSizeMm;
        request.pipelineConfig.designFilterEndFaceEdges = false;
    }
}

bool MainWindow::generateMatplotlibDesignPlots(const pinjie::StitchRunRequest& request,
                                               const bool includeContourPreview,
                                               QString& errorMessage,
                                               QString* scriptOutput) const
{
    errorMessage.clear();
    if (scriptOutput) {
        scriptOutput->clear();
    }
    if (request.resultOutputDir.empty()) {
        errorMessage = QStringLiteral("结果输出目录不可用。");
        return false;
    }

    const std::filesystem::path projectRoot = pinjie::projectRootPath();
    const std::filesystem::path scriptPath = projectRoot / "report" / "figure_export" / "gui_design_plots.py";
    if (!std::filesystem::exists(scriptPath)) {
        errorMessage = QStringLiteral("未找到 gui_design_plots.py。");
        return false;
    }

    const QSize comparisonSize = preferredPlotSize(designCompareViewer_, 1480, 960);
    const QSize compensationSize = preferredPlotSize(compensationViewer_, 1680, 1180);
    const QSize contourSize =
        preferredPlotSize(singleSlotViewer_, 1420, 920)
            .expandedTo(preferredPlotSize(processingPreviewViewer_, 1420, 920));
    const QSize pointCloudSize = preferredPlotSize(pointCloudViewer_, 1500, 980);
    const QString resultDir = QDir::fromNativeSeparators(fromUtf8StdString(request.resultOutputDir));
    const int dpi = 180;

    const auto runPlot = [&](const QString& label,
                             const QString& outputFlag,
                             const QString& outputPath,
                             const QSize& plotSize) -> bool {
        if (outputPath.isEmpty()) {
            return true;
        }

        QProcess process;
        process.setWorkingDirectory(QString::fromStdString(projectRoot.u8string()));
        process.setProgram(QStringLiteral("python"));

        QStringList arguments{
            QString::fromStdString(scriptPath.u8string()),
            QStringLiteral("--result-dir"),
            resultDir,
            outputFlag,
            outputPath,
            QStringLiteral("--width-px"),
            QString::number(plotSize.width()),
            QStringLiteral("--height-px"),
            QString::number(plotSize.height()),
            QStringLiteral("--dpi"),
            QString::number(dpi),
        };

        process.setArguments(arguments);
        process.setProcessChannelMode(QProcess::MergedChannels);
        process.start();
        if (!process.waitForStarted(5000)) {
            errorMessage = QStringLiteral("无法启动 Python。请确认 `python` 已加入 PATH。");
            return false;
        }

        process.waitForFinished(-1);
        const QString output = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
        if (scriptOutput && !output.isEmpty()) {
            if (!scriptOutput->isEmpty()) {
                scriptOutput->append(QStringLiteral("\n\n"));
            }
            scriptOutput->append(QStringLiteral("[%1]\n%2").arg(label, output));
        }
        if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
            errorMessage = output.isEmpty()
                               ? QStringLiteral("%1 图表脚本执行失败。").arg(label)
                               : QStringLiteral("%1 图表脚本执行失败：\n%2").arg(label, output);
            return false;
        }
        return true;
    };

    if (!runPlot(QStringLiteral("误差分析图"),
                 QStringLiteral("--comparison-output"),
                 request.designComparisonOverlayOutputPath.empty()
                     ? QString()
                     : QDir::fromNativeSeparators(fromUtf8StdString(request.designComparisonOverlayOutputPath)),
                 comparisonSize)) {
        return false;
    }
    if (!runPlot(QStringLiteral("补偿图"),
                 QStringLiteral("--compensation-output"),
                 request.designCompensationPlotOutputPath.empty()
                     ? QString()
                     : QDir::fromNativeSeparators(fromUtf8StdString(request.designCompensationPlotOutputPath)),
                 compensationSize)) {
        return false;
    }
    if (includeContourPreview &&
        !runPlot(QStringLiteral("单槽图"),
                 QStringLiteral("--contour-output"),
                 request.contourOverlayOutputPath.empty()
                     ? QString()
                     : QDir::fromNativeSeparators(fromUtf8StdString(request.contourOverlayOutputPath)),
                 contourSize)) {
        return false;
    }
    if (!request.design3dErrorCsvOutputPath.empty() &&
        !runPlot(QStringLiteral("3D点云图"),
                 QStringLiteral("--pointcloud-output"),
                 QDir::fromNativeSeparators(designPointCloudPngPath(request)),
                 pointCloudSize)) {
        return false;
    }
    return true;
}

bool MainWindow::generateCadModelPreview(const pinjie::cad_model::CadModelDocument& document)
{
    lastCadPreviewImagePath_.clear();
    if (cadPreviewViewer_) {
        cadPreviewViewer_->clearImage();
    }
    const bool hasSectionPreview = document.hasSectionSamples && document.sectionSamples.size() >= 2;
    const bool hasProfilePreview = document.hasProfileSamples && document.profileSamples.size() >= 2;
    const bool hasMeshPreview = !hasSectionPreview && document.hasMesh && !document.meshVertices.empty();
    const bool hasPreviewSamples = hasSectionPreview || hasProfilePreview || hasMeshPreview;
    if (!document.valid || !hasPreviewSamples) {
        appendLog(QStringLiteral("[警告] CAD 预览未生成：没有可绘制的 CAD 截面点。"));
        return false;
    }

    const std::filesystem::path outputDir =
        std::filesystem::path(PINJIE_PROJECT_ROOT) / "result" / "cad_preview";
    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);
    if (ec) {
        appendLog(QStringLiteral("[错误] CAD 预览目录创建失败：%1").arg(fromUtf8StdString(ec.message())));
        return false;
    }

    std::string baseName;
    if (!document.sourcePath.empty()) {
        baseName = std::filesystem::u8path(document.sourcePath).stem().u8string();
    }
    if (baseName.empty()) {
        baseName = document.modelLabel.empty() ? std::string("cad_model") : document.modelLabel;
    }
    for (char& ch : baseName) {
        const bool validChar =
            (ch >= '0' && ch <= '9') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= 'a' && ch <= 'z') ||
            ch == '_' || ch == '-';
        if (!validChar) {
            ch = '_';
        }
    }

    const std::filesystem::path csvPath = outputDir / (baseName + "_profile.csv");
    const std::filesystem::path imagePath = outputDir / (baseName + "_profile_preview.png");
    {
        std::ofstream stream(csvPath, std::ios::binary);
        if (!stream) {
            appendLog(QStringLiteral("[错误] CAD 预览采样 CSV 写入失败：%1")
                          .arg(QDir::toNativeSeparators(fromUtf8StdString(csvPath.u8string()))));
            return false;
        }
        stream.setf(std::ios::fixed);
        stream.precision(9);
        stream << "index,s_mm,r_mm,has_cad_point,cad_x_mm,cad_y_mm,cad_z_mm\n";
        std::size_t index = 0;
        if (hasSectionPreview || hasProfilePreview) {
            const auto& previewSamples = hasSectionPreview ? document.sectionSamples : document.profileSamples;
            constexpr std::size_t kMaxCadPreviewRows = 120000;
            const std::size_t previewStride =
                previewSamples.size() > kMaxCadPreviewRows
                    ? (previewSamples.size() + kMaxCadPreviewRows - 1) / kMaxCadPreviewRows
                    : 1;
            for (std::size_t sampleIndex = 0; sampleIndex < previewSamples.size(); sampleIndex += previewStride) {
                const auto& sample = previewSamples[sampleIndex];
                stream << (++index) << ","
                       << sample.sMm << ","
                       << sample.rMm << ","
                       << (sample.hasCadPoint ? 1 : 0) << ","
                       << sample.cadXMm << ","
                       << sample.cadYMm << ","
                       << sample.cadZMm << "\n";
            }
        } else if (hasMeshPreview) {
            for (const auto& vertex : document.meshVertices) {
                stream << (++index) << ","
                       << vertex.xMm << ","
                       << vertex.yMm << ","
                       << 1 << ","
                       << vertex.xMm << ","
                       << vertex.yMm << ","
                       << vertex.zMm << "\n";
            }
        }
    }

    const std::filesystem::path scriptPath =
        std::filesystem::path(PINJIE_PROJECT_ROOT) / "report" / "figure_export" / "cad_profile_preview.py";
    if (!std::filesystem::exists(scriptPath)) {
        appendLog(QStringLiteral("[错误] CAD 预览脚本不存在：%1")
                      .arg(QDir::toNativeSeparators(fromUtf8StdString(scriptPath.u8string()))));
        return false;
    }

    QStringList args;
    const QSize cadPreviewSize = preferredPlotSize(cadPreviewViewer_, 1500, 940);
    args << QString::fromStdString(scriptPath.u8string())
         << QStringLiteral("--profile-csv")
         << QString::fromStdString(csvPath.u8string())
         << QStringLiteral("--output")
         << QString::fromStdString(imagePath.u8string())
         << QStringLiteral("--title")
         << QStringLiteral("CAD XYZ 投影预览 - %1")
                 .arg(document.modelLabel.empty() ? fromUtf8StdString(baseName) : fromUtf8StdString(document.modelLabel))
         << QStringLiteral("--axial-axis")
         << fromUtf8StdString(document.profileMetadata.axialAxis)
         << QStringLiteral("--radial-axis")
         << fromUtf8StdString(document.profileMetadata.radialAxis)
         << QStringLiteral("--section-axis")
         << fromUtf8StdString(document.profileMetadata.sectionNormalAxis)
         << QStringLiteral("--section-coordinate")
         << QString::number(document.profileMetadata.sectionCoordinateMm, 'f', 6)
         << QStringLiteral("--width-px")
         << QString::number(cadPreviewSize.width())
         << QStringLiteral("--height-px")
         << QString::number(cadPreviewSize.height())
         << QStringLiteral("--dpi")
         << QStringLiteral("180");

    QProcess process;
    process.start(QStringLiteral("python"), args);
    if (!process.waitForFinished(60000)) {
        process.kill();
        appendLog(QStringLiteral("[错误] CAD 预览生成超时，请检查 Python/matplotlib 环境。"));
        return false;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        appendLog(QStringLiteral("[错误] CAD 预览生成失败：%1")
                      .arg(QString::fromLocal8Bit(process.readAllStandardError()).trimmed()));
        return false;
    }

    QImage preview;
    if (!preview.load(QString::fromStdString(imagePath.u8string())) || preview.isNull()) {
        appendLog(QStringLiteral("[错误] CAD 预览图读取失败：%1")
                      .arg(QDir::toNativeSeparators(fromUtf8StdString(imagePath.u8string()))));
        return false;
    }
    if (cadPreviewViewer_) {
        cadPreviewViewer_->setImage(preview);
    }
    lastCadPreviewImagePath_ = QDir::toNativeSeparators(QString::fromStdString(imagePath.u8string()));
    appendLog(QStringLiteral("[信息] CAD 真实截面预览已生成：%1").arg(lastCadPreviewImagePath_));
    return true;
}

QString MainWindow::buildInitialTable9AcceptanceText() const
{
    QStringList lines;
    lines << QStringLiteral("表9 在线测量与误差补偿软件系统测试内容");
    lines << QStringLiteral("状态：待运行；运行完成后自动填充证据文件。");
    lines << QString();
    lines << QStringLiteral("| 验收项 | GUI入口 | 当前暴露内容 |");
    lines << QStringLiteral("|---|---|---|");
    lines << QStringLiteral("| 数据接收与预处理(1) CAD模型和目标尺寸 | 模块1 CAD/目标尺寸 | STEP/STP/IGES导入、目标槽宽/槽深、真实截面预览、误差比对轮廓 |");
    lines << QStringLiteral("| 数据接收与预处理(2) 在线测量数据 | 模块2 数据接收预处理 | 图像目录/在线采集、图像数量、结果目录 |");
    lines << QStringLiteral("| 数据接收与预处理(3) 读取/显示/坐标转换/滤波 | 模块2-3 | 标定像素当量、预处理预览、亚像素边缘、点过滤参数 |");
    lines << QStringLiteral("| 模型配准与特征提取(1) 实测-数字模型对齐 | 模块4 模型配准 | 重叠率、搜索范围、旋转、配准候选诊断 |");
    lines << QStringLiteral("| 模型配准与特征提取(2) 横宽/槽位/轮廓边界 | 模块3 特征提取 | 单槽ROI、目标槽宽/槽深、轮廓边界预览 |");
    lines << QStringLiteral("| 模型配准与特征提取(3) 参数/配准/可视化记录 | 模块4-5 | 候选诊断CSV、质量审查CSV、单槽ROI识别图 |");
    lines << QStringLiteral("| 误差分析与补偿解算(1) 槽宽/轮廓/超差 | 模块5 误差分析 | 设计误差明细、三维误差坐标、误差统计图 |");
    lines << QStringLiteral("| 误差分析与补偿解算(2) CAD模型坐标补偿量 | 模块6 补偿解算 | CAD点级补偿、`compensated_cad_x/y/z_mm`补偿后绝对坐标、补偿结算图 |");
    return lines.join('\n');
}

QString MainWindow::buildTable9AcceptanceText(bool ok,
                                              int loadedImageCount,
                                              int preprocessedImageCount,
                                              const pinjie::cad_design::DesignAlignmentResult* designResult,
                                              const pinjie::cad_design::DesignErrorSummary* designSummary) const
{
    const auto state = [](bool pass, bool partial = false) {
        if (pass) {
            return QStringLiteral("通过");
        }
        return partial ? QStringLiteral("待补充") : QStringLiteral("待运行");
    };

    const bool cadFromDesignResult =
        designSummary &&
        designSummary->designProfileMetadata.sourceType == "external_cad" &&
        designSummary->designProfileMetadata.sampleCount >= 2;
    const bool cadImported = activeCadModelDocument_.valid || cadFromDesignResult;
    const bool targetSizeReady =
        lastDesignModelRequest_.targetSlotWidthMm > 0.0 ||
        lastDesignModelRequest_.targetSlotDepthMm > 0.0 ||
        (activeCadModelDocument_.valid && activeCadModelDocument_.hasBounds);
    const bool dataLoaded = loadedImageCount > 0;
    const bool dataPreprocessed = preprocessedImageCount > 0;
    const bool modelAligned = designResult && designResult->ok && designSummary;
    const bool featureResolved =
        designResult && designResult->ok &&
        (QString::fromUtf8(designResult->featureCompensationCsvText.data(),
                           static_cast<int>(designResult->featureCompensationCsvText.size()))
             .contains(QStringLiteral(",ok,")) ||
         lastDesignModelRequest_.targetSlotWidthMm > 0.0);
    const std::size_t usedDesignPointCount =
        designResult ? static_cast<std::size_t>(std::count_if(designResult->profilePoints.begin(),
                                                              designResult->profilePoints.end(),
                                                              [](const auto& point) {
                                                                  return point.isUsed;
                                                              }))
                     : 0;
    const std::size_t cadCoordinateCount =
        designResult ? static_cast<std::size_t>(std::count_if(designResult->profilePoints.begin(),
                                                              designResult->profilePoints.end(),
                                                              [](const auto& point) {
                                                                  return point.isUsed && point.hasCadCoordinates;
                                                              }))
                     : 0;
    const bool hasCadCoordinateCompensation =
        modelAligned &&
        cadCoordinateCount > 0 &&
        cadCoordinateCount == usedDesignPointCount &&
        !lastRequest_.designCompensationCsvOutputPath.empty() &&
        !lastRequest_.design3dErrorCsvOutputPath.empty();
    const bool hasErrorAnalysisEvidence =
        modelAligned &&
        !lastRequest_.designErrorProfileCsvOutputPath.empty() &&
        !lastRequest_.designErrorSummaryCsvOutputPath.empty();
    const bool hasVisualizationEvidence =
        modelAligned &&
        (!lastRequest_.designComparisonOverlayOutputPath.empty() ||
         !lastRequest_.qualityReviewCsvOutputPath.empty());
    const int riskStepCount =
        static_cast<int>(std::count_if(completedSteps_.begin(), completedSteps_.end(), isRiskStep));

    QStringList lines;
    lines << QStringLiteral("表9 在线测量与误差补偿软件系统测试内容");
    lines << QStringLiteral("本次运行：%1").arg(ok ? QStringLiteral("成功") : QStringLiteral("失败或未完整完成"));
    lines << QString();
    lines << QStringLiteral("| 表9测试要点 | 状态 | 界面入口/证据 |");
    lines << QStringLiteral("|---|---|---|");
    lines << QStringLiteral("| 数据接收与预处理(1) 导入CAD模型和目标尺寸 | %1 | 模块1；CAD=%2；目标尺寸=%3 |")
                 .arg(state(cadImported && targetSizeReady, cadImported),
                      cadImported ? QStringLiteral("已导入/已参与比对") : QStringLiteral("未导入"),
                      targetSizeReady ? QStringLiteral("已配置或由CAD边界给出") : QStringLiteral("待填写槽宽/槽深"));
    lines << QStringLiteral("| 数据接收与预处理(2) 采集或导入在线测量数据 | %1 | 模块2；图像=%2张；结果目录=%3 |")
                 .arg(state(dataLoaded),
                      QString::number(loadedImageCount),
                      displayOutputPath(lastRequest_.resultOutputDir));
    lines << QStringLiteral("| 数据接收与预处理(3) 读取/显示/坐标转换/滤波 | %1 | 模块2-3；预处理=%2张；标定=%3 |")
                 .arg(state(dataPreprocessed && (hasActiveCalibration() || lastDesignModelRequest_.temporaryPixelSizeMm > 0.0),
                            dataPreprocessed),
                      QString::number(preprocessedImageCount),
                      hasActiveCalibration() ? calibrationQualityLine(activeCalibrationCache_)
                                             : QStringLiteral("未加载，或使用临时像素当量"));
    lines << QStringLiteral("| 模型配准与特征提取(1) 实测数据与数字化模型对齐 | %1 | 模块4/5；有效比对点=%2；候选诊断=%3 |")
                 .arg(state(modelAligned),
                      QString::number(designSummary ? static_cast<int>(designSummary->usedCount) : 0),
                      displayOutputPath(lastRequest_.alignmentCandidateDiagnosticsCsvOutputPath));
    lines << QStringLiteral("| 模型配准与特征提取(2) 提取横宽/槽位/轮廓边界 | %1 | 模块3/5；槽特征=%2；单槽ROI图=%3 |")
                 .arg(state(featureResolved, modelAligned),
                      displayOutputPath(lastRequest_.designFeatureCompensationCsvOutputPath),
                      displayOutputPath(lastRequest_.designComparisonOverlayOutputPath));
    lines << QStringLiteral("| 模型配准与特征提取(3) 记录参数/配准结果/可视化 | %1 | 模块4/5；风险步=%2；质量审查=%3 |")
                 .arg(state(hasVisualizationEvidence, modelAligned),
                      QString::number(riskStepCount),
                      displayOutputPath(lastRequest_.qualityReviewCsvOutputPath));
    lines << QStringLiteral("| 误差分析与补偿解算(1) 槽宽/轮廓偏差/局部超差 | %1 | 模块5；误差明细=%2；三维误差=%3 |")
                 .arg(state(hasErrorAnalysisEvidence, modelAligned),
                      displayOutputPath(lastRequest_.designErrorProfileCsvOutputPath),
                      displayOutputPath(lastRequest_.design3dErrorCsvOutputPath));
    lines << QStringLiteral("| 误差分析与补偿解算(2) CAD模型坐标补偿量 | %1 | 模块6；CAD补偿点=%2/%3；补偿后XYZ字段=compensated_cad_x/y/z_mm；补偿CSV=%4 |")
                 .arg(state(hasCadCoordinateCompensation, modelAligned),
                      QString::number(static_cast<int>(cadCoordinateCount)),
                      QString::number(static_cast<int>(usedDesignPointCount)),
                      displayOutputPath(lastRequest_.designCompensationCsvOutputPath));
    lines << QString();
    lines << QStringLiteral("补偿可视化：%1").arg(displayOutputPath(lastRequest_.designCompensationPlotOutputPath));
    return lines.join('\n');
}

void MainWindow::setAcceptanceText(const QString& text)
{
    if (acceptanceChecklistEdit_) {
        acceptanceChecklistEdit_->setPlainText(text);
    }
    if (acceptanceOverviewEdit_) {
        acceptanceOverviewEdit_->setPlainText(text);
    }
}

bool MainWindow::hasActiveCalibration() const
{
    return activeCalibrationCache_.valid && activeCalibrationCache_.profile.valid;
}

bool MainWindow::hasTemporaryDesignPixelSize() const
{
    if (!activeCadModelDocument_.valid) {
        return false;
    }

    pinjie::cad_model::DesignModelRequest request;
    QString errorMessage;
    if (designModelPanel_ && designModelPanel_->buildRequest(request, errorMessage)) {
        return request.temporaryPixelSizeMm > 0.0;
    }
    return lastDesignModelRequest_.temporaryPixelSizeMm > 0.0;
}

bool MainWindow::hasLocalSlotImageMode() const
{
    pinjie::cad_model::DesignModelRequest request;
    QString errorMessage;
    if (designModelPanel_ && designModelPanel_->buildRequest(request, errorMessage)) {
        return request.localSlotImageMode;
    }
    return lastDesignModelRequest_.localSlotImageMode;
}

bool MainWindow::hasMeasurementBaseline() const
{
    return hasActiveCalibration() || hasTemporaryDesignPixelSize();
}

QSize MainWindow::preferredPlotSize(const QWidget* widget, const int minWidth, const int minHeight) const
{
    QSize size(minWidth, minHeight);
    if (const auto* viewer = dynamic_cast<const ImageViewer*>(widget)) {
        size = viewer->recommendedExportPixelSize();
    } else if (widget) {
        const QSize widgetSize = widget->size();
        size = QSize(std::max(minWidth, widgetSize.width()),
                     std::max(minHeight, widgetSize.height()));
    }
    return size.expandedTo(QSize(minWidth, minHeight));
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
    const bool standardCircleMode = configPanel_ && configPanel_->standardCircleModeEnabled();
    pinjie::StitchRunRequest previewRequest;
    QString previewError;
    const bool syntheticInput =
        configPanel_ && configPanel_->buildRequest(previewRequest, previewError, false) &&
        isSingleSlotTestInput(previewRequest);
    const bool localSlotImageMode = hasLocalSlotImageMode();
    const bool canRun = hasMeasurementBaseline() || standardCircleMode || syntheticInput || localSlotImageMode;

    if (startButton_) {
        startButton_->setEnabled(!running && canRun);
    }
    if (moduleRunButton_) {
        moduleRunButton_->setEnabled(!running && (currentStage == CalibrationStage || canRun));
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
        moduleRunButton_->setText(QStringLiteral("运行量值标定"));
        return;
    }

    if (configPanel_ && configPanel_->standardCircleModeEnabled()) {
        moduleRunButton_->setText(QStringLiteral("运行标准圆检测"));
        return;
    }

    pinjie::StitchRunRequest previewRequest;
    QString previewError;
    const bool syntheticInput =
        configPanel_ && configPanel_->buildRequest(previewRequest, previewError, false) &&
        isSingleSlotTestInput(previewRequest);

    const bool localSlotImageMode = hasLocalSlotImageMode();
    if (!hasMeasurementBaseline() && !syntheticInput && !localSlotImageMode) {
        moduleRunButton_->setText(QStringLiteral("请先加载标定或填写临时像素当量"));
        return;
    }

    moduleRunButton_->setText(
        syntheticInput && !hasActiveCalibration()
            ? QStringLiteral("运行%1模块（单槽测试像素当量）").arg(stageTitle(stageIndex))
        : localSlotImageMode && !hasActiveCalibration() && !hasTemporaryDesignPixelSize()
            ? QStringLiteral("运行%1模块（局部槽照片）").arg(stageTitle(stageIndex))
        : hasTemporaryDesignPixelSize() && !hasActiveCalibration()
            ? QStringLiteral("运行%1模块（临时像素当量）").arg(stageTitle(stageIndex))
            : QStringLiteral("运行%1模块").arg(stageTitle(stageIndex)));
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
    if (isStandardCircleMode(lastRequest_)) {
        const bool saveStepSummary = configPanel_ && configPanel_->saveStepSummaryCsv();
        const bool saveContourPoints = configPanel_ && configPanel_->saveContourPointsCsv();
        const bool saveCandidateDiagnostics = configPanel_ && configPanel_->saveAlignmentCandidateDiagnosticsCsv();

        QStringList exportedFiles;
        if (saveStepSummary && !lastRequest_.csvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.csvOutputPath));
        }
        if (saveContourPoints && !lastRequest_.contourPointsCsvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.contourPointsCsvOutputPath));
        }
        const std::string radiiCsvPath = standardCircleRadiiCsvOutputPath(lastRequest_);
        if (saveContourPoints && !radiiCsvPath.empty() &&
            std::filesystem::exists(std::filesystem::u8path(radiiCsvPath))) {
            exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(radiiCsvPath));
        }
        if (saveCandidateDiagnostics && !lastRequest_.qualityReviewCsvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.qualityReviewCsvOutputPath));
        }
        if (saveCandidateDiagnostics && !lastRequest_.alignmentCandidateDiagnosticsCsvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(
                fromUtf8StdString(lastRequest_.alignmentCandidateDiagnosticsCsvOutputPath));
        }
        if (configPanel_ && configPanel_->saveNormalErrorProfileCsv() &&
            !lastRequest_.designErrorProfileCsvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(
                fromUtf8StdString(lastRequest_.designErrorProfileCsvOutputPath));
        }
        if (configPanel_ && configPanel_->saveTangentProfileCsv() &&
            !lastRequest_.designErrorSummaryCsvOutputPath.empty()) {
            exportedFiles << QDir::toNativeSeparators(
                fromUtf8StdString(lastRequest_.designErrorSummaryCsvOutputPath));
        }

        exportedFiles.removeDuplicates();
        if (exportedFiles.isEmpty()) {
            QMessageBox::information(this,
                                     QStringLiteral("导出 CSV"),
                                     QStringLiteral("标准圆模式下请先勾选至少一种可查看的 CSV 类型。"));
            refreshReportExportState(QStringLiteral("未选择标准圆结果 CSV"));
            return;
        }

        appendLog(QStringLiteral("[信息] 标准圆结果文件：\n%1").arg(exportedFiles.join('\n')));
        refreshReportExportState(QStringLiteral("标准圆结果文件已就绪，共 %1 项").arg(exportedFiles.size()));
        statusBar()->showMessage(QStringLiteral("标准圆结果文件已就绪，共 %1 项").arg(exportedFiles.size()));
        return;
    }

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
    const bool standardCircleMode = isStandardCircleMode(lastRequest_);
    if (generateFigureButton_) {
        generateFigureButton_->setEnabled(!workerThread_ &&
                                         !standardCircleMode &&
                                         runCache_ &&
                                         runCache_->hasStitching());
    }
    if (exportSelectedCsvButton_) {
        const bool canExport =
            !workerThread_ &&
            ((standardCircleMode && !lastRequest_.resultOutputDir.empty()) ||
             (runCache_ && runCache_->hasStitching()));
        exportSelectedCsvButton_->setEnabled(canExport);
    }

    if (!reportExportStatusLabel_) {
        return;
    }

    if (!statusMessage.isEmpty()) {
        reportExportStatusLabel_->setText(statusMessage);
        return;
    }

    const QStringList labels = selectedPaperCsvLabels(configPanel_);
    const bool analysisMode =
        currentRunMode_ == pinjie::StitchRunMode::Full ||
        currentRunMode_ == pinjie::StitchRunMode::Report;
    const QString workpieceAutoText =
        analysisMode
            ? QStringLiteral("自动生成：误差分析图、质量审查、候选诊断；补偿图/CSV 请在模块6查看")
            : QStringLiteral("自动生成：全景图、轮廓图、拼接汇总、候选诊断；误差/补偿需运行全流程或模块5");
    if (labels.isEmpty()) {
        reportExportStatusLabel_->setText(
            standardCircleMode
                ? QStringLiteral("归档状态：标准圆结果已生成，请先勾选需要查看的 CSV 类型 | 自动生成：摘要、25点详情、候选覆盖、掩模诊断")
                : QStringLiteral("归档状态：结果已生成，请先勾选需要导出的 CSV 类型 | %1").arg(workpieceAutoText));
        return;
    }

    const QString targetDir =
        lastRequest_.resultOutputDir.empty()
            ? QStringLiteral("当前结果目录")
            : QDir::toNativeSeparators(fromUtf8StdString(lastRequest_.resultOutputDir));
    reportExportStatusLabel_->setText(
        (standardCircleMode
             ? QStringLiteral("归档状态：可查看 CSV：%1 | 输出目录：%2 | 自动生成：窗口图、边缘叠加、摘要与掩模诊断")
             : QStringLiteral("归档状态：可导出 CSV：%1 | 输出目录：%2 | %3"))
            .arg(labels.join(QStringLiteral(", ")), targetDir, standardCircleMode ? QString() : workpieceAutoText));
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
QPushButton#successActionButton {
    background: #2f7d4a;
    border-color: #25653b;
    color: #ffffff;
}
QPushButton#successActionButton:hover {
    background: #25653b;
}
QPushButton#successActionButton:checked {
    background: #1f5b35;
    border-color: #18472a;
    color: #ffffff;
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
QPlainTextEdit#acceptanceOverviewEdit {
    background: #f8fbf4;
    border-color: #cfe1bd;
    color: #244a34;
    font-size: 11px;
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
QWidget#designModelConfigSection QToolButton#configSectionToggle,
QWidget#designModelConfigSection QGroupBox {
    color: #7a4d2a;
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
QWidget#designModelConfigSection QGroupBox {
    border-left: 4px solid #9a6540;
    background: #fdf9f5;
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


