#pragma once

#include "cad_model/CadModelLoader.h"

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;

namespace pinjie::gui {

class DesignModelPanel : public QWidget {
    Q_OBJECT

public:
    explicit DesignModelPanel(QWidget* parent = nullptr);

    bool buildRequest(pinjie::cad_model::DesignModelRequest& request, QString& errorMessage) const;
    void setRunning(bool running);
    void setDocumentInfo(const pinjie::cad_model::CadModelDocument& document);
    void clearDocumentInfo();

signals:
    void configChanged();
    void importRequested(const QString& cadFilePath);
    void scanCompareRequested(const QString& stlFilePath);

private:
    void updateBackendStatusLabel();

    QLineEdit* cadFilePathEdit_{nullptr};
    QLineEdit* scannedStlFilePathEdit_{nullptr};
    QLineEdit* modelNameEdit_{nullptr};
    QComboBox* axialAxisCombo_{nullptr};
    QComboBox* radialAxisCombo_{nullptr};
    QCheckBox* reverseAxialCheck_{nullptr};
    QCheckBox* useLeftEndpointOriginCheck_{nullptr};
    QCheckBox* extractUpperEnvelopeCheck_{nullptr};
    QDoubleSpinBox* samplingStepSpin_{nullptr};
    QDoubleSpinBox* targetSlotWidthSpin_{nullptr};
    QDoubleSpinBox* targetSlotDepthSpin_{nullptr};
    QDoubleSpinBox* temporaryPixelSizeSpin_{nullptr};
    QCheckBox* localSlotImageModeCheck_{nullptr};
    QDoubleSpinBox* localSlotBottomWidthSpin_{nullptr};
    QLabel* backendStatusLabel_{nullptr};
    QLabel* importedModelLabel_{nullptr};
    QPushButton* browseCadFileButton_{nullptr};
    QPushButton* importCadFileButton_{nullptr};
    QPushButton* browseScannedStlButton_{nullptr};
    QPushButton* compareScannedStlButton_{nullptr};
};

} // namespace pinjie::gui
