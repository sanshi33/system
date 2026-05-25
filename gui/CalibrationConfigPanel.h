#pragma once

#include "calibration/CalibrationTypes.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QWidget>

namespace pinjie::gui {

class CalibrationConfigPanel : public QWidget {
    Q_OBJECT

public:
    explicit CalibrationConfigPanel(QWidget* parent = nullptr);

    bool buildRequest(pinjie::CameraCalibrationRequest& request, QString& errorMessage) const;
    void setRunning(bool running);

signals:
    void configChanged();

private:
    QStringList scanImagePaths() const;
    void updateDetectedCount();
    void connectChangeSignals();

    QLineEdit* sessionNameEdit_{nullptr};
    QLineEdit* imageDirEdit_{nullptr};
    QLineEdit* cacheDirEdit_{nullptr};
    QLineEdit* outputDirEdit_{nullptr};
    QLabel* detectedCountLabel_{nullptr};
    QSpinBox* rowsSpin_{nullptr};
    QSpinBox* colsSpin_{nullptr};
    QDoubleSpinBox* pitchSpin_{nullptr};
    QSpinBox* roiRadiusSpin_{nullptr};
    QSpinBox* minValidImagesSpin_{nullptr};
    QCheckBox* preferCachedCheck_{nullptr};
    QCheckBox* persistCacheCheck_{nullptr};
    QCheckBox* enforceSquarePixelsCheck_{nullptr};
    QCheckBox* lockPrincipalPointCheck_{nullptr};
    QCheckBox* enableImageResidualCompensationCheck_{nullptr};
    QDoubleSpinBox* imageResidualPriorSigmaSpin_{nullptr};
    QDoubleSpinBox* imageResidualMaxCoeffSpin_{nullptr};
    QCheckBox* enableBoardWarpCompensationCheck_{nullptr};
    QDoubleSpinBox* boardWarpPriorSigmaSpin_{nullptr};
    QDoubleSpinBox* boardWarpMaxOffsetSpin_{nullptr};
    QCheckBox* enableBoardPointCompensationCheck_{nullptr};
    QDoubleSpinBox* boardPointPriorSigmaSpin_{nullptr};
    QDoubleSpinBox* boardPointSmoothSigmaSpin_{nullptr};
    QDoubleSpinBox* boardPointMaxOffsetSpin_{nullptr};
};

} // namespace pinjie::gui
