#pragma once

#include "acquisition/CameraTypes.h"
#include "acquisition/CaptureTypes.h"
#include "app/StitchWorkflowService.h"
#include "registration/RegistrationTypes.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QWidget>

class QPushButton;
class QThread;

namespace pinjie::gui {

class CameraCaptureWorker;
class ImageViewer;

class RunConfigPanel : public QWidget {
    Q_OBJECT

public:
    explicit RunConfigPanel(QWidget* parent = nullptr);
    ~RunConfigPanel() override;

    bool buildRequest(pinjie::StitchRunRequest& request,
                      QString& errorMessage,
                      bool materializeOutputPaths = true) const;
    void setRunning(bool running);
    pinjie::CaptureMode currentCaptureMode() const;
    QString currentCaptureModeLabel() const;
    bool saveStepSummaryCsv() const;
    bool saveContourPointsCsv() const;
    bool saveStitchedContourProfileCsv() const;
    bool saveTangentStepCsv() const;
    bool saveNormalErrorProfileCsv() const;
    bool saveTangentProfileCsv() const;
    bool saveAlignmentCandidateDiagnosticsCsv() const;
    bool standardCircleModeEnabled() const;
    void setStandardCircleModeEnabled(bool enabled);
    void revealStandardCircleConfig();

    QWidget* acquisitionSection() const;
    QWidget* processingSection() const;
    QWidget* registrationSection() const;
    QWidget* reportSection() const;

signals:
    void configChanged();
    void cameraPreviewUpdated(const QImage& image, const QString& summary);
    void standardCircleRunRequested();

private:
    QStringList scanImagePaths() const;
    void updateDetectedCount();
    void updateInputImagePreview();
    pinjie::MotionPriorDirection currentDirection() const;
    void connectChangeSignals();
    bool buildCameraRequest(pinjie::CameraSequenceRequest& request,
                            QString& errorMessage,
                            pinjie::CameraCaptureMode mode) const;
    QString defaultCameraOutputDir() const;
    int nextCameraStartIndex() const;
    void refreshCameraDevices();
    void cancelCameraCapture();
    void startSingleCameraCapture();
    void startManualStepCapture();
    void startCameraCapture();
    void runCameraCaptureTask(const pinjie::CameraSequenceRequest& request, bool advanceManualStepIndex);
    void updateCameraPreview(const QImage& image, const QString& summary);
    void finishCameraTask();
    void setCameraControlsEnabled(bool enabled);
    void updateCameraStepControlsEnabled();

    QWidget* acquisitionSection_{nullptr};
    QWidget* processingSection_{nullptr};
    QWidget* registrationSection_{nullptr};
    QWidget* reportSection_{nullptr};
    QWidget* standardCircleSectionContainer_{nullptr};

    QComboBox* captureModeCombo_{nullptr};
    QLineEdit* inputDirEdit_{nullptr};
    QLabel* detectedCountLabel_{nullptr};
    QSpinBox* imageLimitSpin_{nullptr};
    QComboBox* cameraDeviceCombo_{nullptr};
    QPushButton* refreshCameraButton_{nullptr};
    QLineEdit* cameraOutputDirEdit_{nullptr};
    QPushButton* cameraBrowseButton_{nullptr};
    QLineEdit* cameraSessionNameEdit_{nullptr};
    QSpinBox* cameraFrameCountSpin_{nullptr};
    QSpinBox* cameraIntervalSpin_{nullptr};
    QCheckBox* cameraSetExposureCheck_{nullptr};
    QDoubleSpinBox* cameraExposureSpin_{nullptr};
    QCheckBox* cameraSetGainCheck_{nullptr};
    QDoubleSpinBox* cameraGainSpin_{nullptr};
    QCheckBox* cameraSoftwareTriggerCheck_{nullptr};
    QCheckBox* cameraWriteStagePositionCheck_{nullptr};
    QSpinBox* cameraStepIndexSpin_{nullptr};
    QDoubleSpinBox* cameraStageXSpin_{nullptr};
    QDoubleSpinBox* cameraStageYSpin_{nullptr};
    QLineEdit* cameraNoteEdit_{nullptr};
    QPushButton* cameraSingleCaptureButton_{nullptr};
    QPushButton* cameraManualStepButton_{nullptr};
    QPushButton* cameraCaptureButton_{nullptr};
    QPushButton* cameraCancelButton_{nullptr};
    QLabel* cameraStatusLabel_{nullptr};
    QLabel* cameraPreviewInfoLabel_{nullptr};
    ImageViewer* cameraPreviewViewer_{nullptr};
    QThread* cameraTaskThread_{nullptr};
    CameraCaptureWorker* cameraCaptureWorker_{nullptr};

    QDoubleSpinBox* cannyLowSpin_{nullptr};
    QDoubleSpinBox* cannyHighSpin_{nullptr};
    QSpinBox* subpixWindowSpin_{nullptr};
    QDoubleSpinBox* subpixSigmaSpin_{nullptr};
    QCheckBox* pointFilterCheck_{nullptr};
    QDoubleSpinBox* filterConfidenceSpin_{nullptr};
    QDoubleSpinBox* filterGradientSpin_{nullptr};
    QSpinBox* filterWindowRadiusSpin_{nullptr};
    QDoubleSpinBox* filterHampelSigmaSpin_{nullptr};
    QDoubleSpinBox* filterHampelMinScaleSpin_{nullptr};

    QDoubleSpinBox* overlapSpin_{nullptr};
    QComboBox* directionCombo_{nullptr};
    QDoubleSpinBox* searchRangeSpin_{nullptr};
    QDoubleSpinBox* rotationRangeSpin_{nullptr};
    QDoubleSpinBox* rotationStepSpin_{nullptr};
    QDoubleSpinBox* tangentResidualWeightSpin_{nullptr};
    QDoubleSpinBox* tangentCorrelationWeightSpin_{nullptr};
    QCheckBox* standardCircleCheck_{nullptr};
    QLineEdit* standardCirclePrefixEdit_{nullptr};
    QLineEdit* standardCircleExtensionEdit_{nullptr};
    QSpinBox* standardCircleStartIndexSpin_{nullptr};
    QDoubleSpinBox* standardCircleDiameterSpin_{nullptr};
    QDoubleSpinBox* standardCircleHorizontalFovSpin_{nullptr};
    QDoubleSpinBox* standardCircleVerticalFovSpin_{nullptr};
    QDoubleSpinBox* standardCircleOverlapSpin_{nullptr};
    QDoubleSpinBox* standardCircleWindowHalfSizeSpin_{nullptr};
    QSpinBox* standardCircleMedianRadiusSpin_{nullptr};
    QDoubleSpinBox* standardCircleFilterBlendSpin_{nullptr};
    QPushButton* standardCircleRunButton_{nullptr};

    QLineEdit* runNameEdit_{nullptr};
    QCheckBox* saveDebugCheck_{nullptr};
    QCheckBox* saveStepSummaryCsvCheck_{nullptr};
    QCheckBox* saveContourPointsCsvCheck_{nullptr};
    QCheckBox* saveStitchedContourProfileCsvCheck_{nullptr};
    QCheckBox* saveTangentStepCsvCheck_{nullptr};
    QCheckBox* saveNormalErrorProfileCsvCheck_{nullptr};
    QCheckBox* saveTangentProfileCsvCheck_{nullptr};
    QCheckBox* saveAlignmentCandidateDiagnosticsCsvCheck_{nullptr};
};

} // namespace pinjie::gui
