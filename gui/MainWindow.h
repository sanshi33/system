#pragma once

#include "calibration/CalibrationTypes.h"
#include "cad_model/CadModelLoader.h"
#include "cad_design/DesignProfileAlignment.h"
#include "gui/CalibrationConfigPanel.h"
#include "gui/CalibrationWorker.h"
#include "gui/DesignModelPanel.h"
#include "gui/ImageViewer.h"
#include "gui/RunConfigPanel.h"
#include "gui/StepDetailPanel.h"
#include "gui/StepTableModel.h"
#include "gui/StitchWorker.h"

#include <QHash>
#include <QMainWindow>
#include <QSize>
#include <QStringList>

#include <vector>

class QListWidget;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QProcess;
class QPushButton;
class QScrollArea;
class QStackedWidget;
class QTableView;
class QTabWidget;
class QThread;

namespace pinjie {
struct QualitySummary;
}

namespace pinjie::gui {

class PointCloud3DViewer;

class MainWindow : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    void startCalibration();
    void startDesignModelImport();
    void startScannedStlComparison(const QString& stlFilePath);
    void startRun();
    void startRun(pinjie::StitchRunMode runMode);
    void startCurrentModuleRun();
    void stopRun();
    void onCalibrationProgress(int current, int total);
    void onCalibrationPreviewReady(int index, int total, const QImage& image);
    void onCalibrationFinished(bool ok,
                               bool loadedFromCache,
                               const QString& message,
                               pinjie::CalibrationResultCache cache);
    void onProgress(const QString& stage, int current, int total);
    void onStepCompleted(const pinjie::StitchStepRecord& step);
    void onImageReady(const QString& stage, int index, int total, const QImage& image);
    void onRunFinished(bool ok,
                       const QString& message,
                       int loadedImageCount,
                       int preprocessedImageCount,
                       const QImage& panorama,
                       const QString& csvText,
                       pinjie::StitchRunCachePtr cache);
    void onAcquisitionPreviewUpdated(const QImage& image, const QString& summary);
    void onStepSelectionChanged();
    void appendLog(const QString& message);
    void setUiRunning(bool running);
    void cleanupWorker();
    void loadRawImagesForStep(const pinjie::StitchStepRecord& step);
    void onProcessingImageSelectionChanged();
    void switchToStage(int stageIndex);
    void refreshStageSummaries();
    bool hasActiveCalibration() const;
    bool hasTemporaryDesignPixelSize() const;
    bool hasLocalSlotImageMode() const;
    bool hasMeasurementBaseline() const;
    void updateStageButtonStates(int stageIndex);
    void updateWorkflowAccessState();
    void updateModuleRunButtonText(int stageIndex);
    void selectStepRow(int row);
    void jumpToWorstStep();
    void jumpToNextRiskStep();
    int findWorstStepRow() const;
    int findNextRiskStepRow(int startRow) const;
    void refreshRegistrationToolState();
    void resetReportMetricCards();
    void updateReportMetricCards(int loadedImageCount,
                                 int preprocessedImageCount,
                                 const pinjie::QualitySummary* qualitySummary,
                                 const pinjie::cad_design::DesignErrorSummary* designSummary);
    void exportSelectedReportCsvs();
    void generatePublicationFigure();
    void refreshPublicationFigurePreview();
    void refreshReportExportState(const QString& statusMessage = QString());
    void applyActiveDesignModelToRequest(pinjie::StitchRunRequest& request) const;
    void refreshCalibrationDetailPane();
    void refreshDesignModelDetailPane();
    bool generateCadModelPreview(const pinjie::cad_model::CadModelDocument& document);
    QString buildInitialTable9AcceptanceText() const;
    QString buildTable9AcceptanceText(bool ok,
                                      int loadedImageCount,
                                      int preprocessedImageCount,
                                      const pinjie::cad_design::DesignAlignmentResult* designResult,
                                      const pinjie::cad_design::DesignErrorSummary* designSummary) const;
    QSize preferredPlotSize(const QWidget* widget, int minWidth, int minHeight) const;
    bool generateMatplotlibDesignPlots(const pinjie::StitchRunRequest& request,
                                       bool includeContourPreview,
                                       QString& errorMessage,
                                       QString* scriptOutput = nullptr) const;
    void setAcceptanceText(const QString& text);
    void applyWindowStyle();

    CalibrationConfigPanel* calibrationPanel_{nullptr};
    DesignModelPanel* designModelPanel_{nullptr};
    RunConfigPanel* configPanel_{nullptr};
    QStackedWidget* stageStack_{nullptr};
    QListWidget* processingImageList_{nullptr};
    StepTableModel* stepModel_{nullptr};
    QTableView* stepTable_{nullptr};
    StepDetailPanel* stepDetailPanel_{nullptr};
    ImageViewer* calibrationPreviewViewer_{nullptr};
    ImageViewer* cadPreviewViewer_{nullptr};
    PointCloud3DViewer* cadInteractiveViewer_{nullptr};
    ImageViewer* acquisitionPreviewViewer_{nullptr};
    ImageViewer* processingPreviewViewer_{nullptr};
    ImageViewer* referenceViewer_{nullptr};
    ImageViewer* targetViewer_{nullptr};
    ImageViewer* debugViewer_{nullptr};
    ImageViewer* panoramaViewer_{nullptr};
    ImageViewer* designCompareViewer_{nullptr};
    ImageViewer* singleSlotViewer_{nullptr};
    PointCloud3DViewer* pointCloudViewer_{nullptr};
    ImageViewer* compensationViewer_{nullptr};
    ImageViewer* publicationFigureViewer_{nullptr};
    QTabWidget* viewTabs_{nullptr};
    QTabWidget* reportViewTabs_{nullptr};
    QTabWidget* bottomTabs_{nullptr};
    QPlainTextEdit* calibrationOverviewEdit_{nullptr};
    QPlainTextEdit* calibrationDetailEdit_{nullptr};
    QPlainTextEdit* designModelOverviewEdit_{nullptr};
    QPlainTextEdit* designModelDetailEdit_{nullptr};
    QPlainTextEdit* acquisitionOverviewEdit_{nullptr};
    QPlainTextEdit* acquisitionPreviewInfoEdit_{nullptr};
    QPlainTextEdit* processingOverviewEdit_{nullptr};
    QPlainTextEdit* processingPreviewInfoEdit_{nullptr};
    QPlainTextEdit* registrationOverviewEdit_{nullptr};
    QPlainTextEdit* summaryEdit_{nullptr};
    QPlainTextEdit* qualityReviewEdit_{nullptr};
    QPlainTextEdit* designCompareEdit_{nullptr};
    QPlainTextEdit* candidateDiagnosticsEdit_{nullptr};
    QPlainTextEdit* acceptanceChecklistEdit_{nullptr};
    QPlainTextEdit* acceptanceOverviewEdit_{nullptr};
    QPlainTextEdit* compensationSummaryEdit_{nullptr};
    QPlainTextEdit* csvEdit_{nullptr};
    QPlainTextEdit* logEdit_{nullptr};
    QLabel* workflowStateLabel_{nullptr};
    QLabel* registrationPresetLabel_{nullptr};
    QLabel* metricLoadedImagesValue_{nullptr};
    QLabel* metricPreprocessedImagesValue_{nullptr};
    QLabel* metricStepCountValue_{nullptr};
    QLabel* metricNormalRmseValue_{nullptr};
    QLabel* metricTangentRmseValue_{nullptr};
    QLabel* metricTangentCorrValue_{nullptr};
    QLabel* metricFlaggedStepsValue_{nullptr};
    QLabel* metricWorstStepValue_{nullptr};
    QLabel* metricDesignRmseValue_{nullptr};
    QLabel* metricDesignP95Value_{nullptr};
    QLabel* metricDesignOffsetValue_{nullptr};
    QLabel* metricDesignUsedCountValue_{nullptr};
    QLabel* reportExportStatusLabel_{nullptr};
    QProgressBar* progressBar_{nullptr};
    QPushButton* startButton_{nullptr};
    QPushButton* moduleRunButton_{nullptr};
    QPushButton* stopButton_{nullptr};
    QPushButton* focusWorstStepButton_{nullptr};
    QPushButton* focusNextRiskStepButton_{nullptr};
    QPushButton* exportSelectedCsvButton_{nullptr};
    QPushButton* generateFigureButton_{nullptr};

    QThread* workerThread_{nullptr};
    CalibrationWorker* calibrationWorker_{nullptr};
    StitchWorker* worker_{nullptr};
    bool currentTaskSupportsStop_{false};

    std::vector<QPushButton*> moduleButtons_;
    QStringList lastImagePaths_;
    QString lastCadPreviewImagePath_;
    QHash<int, QImage> preprocessImages_;
    QHash<int, QImage> debugImages_;
    pinjie::CameraCalibrationRequest lastCalibrationRequest_{};
    pinjie::cad_model::DesignModelRequest lastDesignModelRequest_{};
    pinjie::cad_model::CadModelDocument activeCadModelDocument_{};
    pinjie::CalibrationResultCache activeCalibrationCache_{};
    pinjie::StitchRunRequest lastRequest_{};
    pinjie::StitchRunCachePtr runCache_{};
    pinjie::StitchRunMode currentRunMode_{pinjie::StitchRunMode::Full};
    std::vector<pinjie::StitchStepRecord> completedSteps_;
};

} // namespace pinjie::gui
