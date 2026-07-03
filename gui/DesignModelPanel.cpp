#include "gui/DesignModelPanel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

namespace pinjie::gui {

namespace {

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

QWidget* makeCollapsibleSection(const QString& title, QWidget* content, bool expanded, QWidget* parent)
{
    auto* container = new QWidget(parent);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    auto* toggleButton = new QToolButton(container);
    toggleButton->setObjectName(QStringLiteral("configSectionToggle"));
    toggleButton->setText(title);
    toggleButton->setCheckable(true);
    toggleButton->setChecked(expanded);
    toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleButton->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);

    content->setVisible(expanded);
    QObject::connect(toggleButton, &QToolButton::toggled, container, [toggleButton, content](const bool checked) {
        toggleButton->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        content->setVisible(checked);
    });

    layout->addWidget(toggleButton);
    layout->addWidget(content);
    return container;
}

} // namespace

DesignModelPanel::DesignModelPanel(QWidget* parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("designModelConfigSection"));

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(12);

    auto* sourceGroup = new QGroupBox(QStringLiteral("CAD 模型输入"), this);
    auto* sourceLayout = new QVBoxLayout(sourceGroup);
    sourceLayout->setSpacing(8);

    cadFilePathEdit_ = new QLineEdit(sourceGroup);
    cadFilePathEdit_->setPlaceholderText(QStringLiteral("选择 STEP/STP/IGES 文件"));
    browseCadFileButton_ = new QPushButton(QStringLiteral("浏览"), sourceGroup);
    importCadFileButton_ = new QPushButton(QStringLiteral("导入设计模型"), sourceGroup);
    importCadFileButton_->setObjectName(QStringLiteral("secondaryActionButton"));

    auto* fileRow = new QHBoxLayout();
    fileRow->addWidget(cadFilePathEdit_, 1);
    fileRow->addWidget(browseCadFileButton_);
    fileRow->addWidget(importCadFileButton_);
    sourceLayout->addLayout(fileRow);

    scannedStlFilePathEdit_ = new QLineEdit(sourceGroup);
    scannedStlFilePathEdit_->setPlaceholderText(QStringLiteral("选择扫描 STL 文件（用于与理论 CAD 独立比对）"));
    browseScannedStlButton_ = new QPushButton(QStringLiteral("浏览"), sourceGroup);
    compareScannedStlButton_ = new QPushButton(QStringLiteral("扫描 STL 对比理论 CAD"), sourceGroup);
    compareScannedStlButton_->setObjectName(QStringLiteral("secondaryActionButton"));

    auto* scanRow = new QHBoxLayout();
    scanRow->addWidget(scannedStlFilePathEdit_, 1);
    scanRow->addWidget(browseScannedStlButton_);
    scanRow->addWidget(compareScannedStlButton_);
    sourceLayout->addLayout(scanRow);

    modelNameEdit_ = new QLineEdit(sourceGroup);
    modelNameEdit_->setPlaceholderText(QStringLiteral("可选：用于报告和特征命名"));

    backendStatusLabel_ = new QLabel(sourceGroup);
    backendStatusLabel_->setWordWrap(true);
    importedModelLabel_ = new QLabel(QStringLiteral("导入状态：尚未加载设计模型。"), sourceGroup);
    importedModelLabel_->setWordWrap(true);

    auto* sourceForm = new QFormLayout();
    sourceForm->addRow(QStringLiteral("模型名称"), modelNameEdit_);
    sourceForm->addRow(QStringLiteral("后端状态"), backendStatusLabel_);
    sourceForm->addRow(QStringLiteral("当前模型"), importedModelLabel_);
    sourceLayout->addLayout(sourceForm);

    auto* coordinateGroup = new QGroupBox(QStringLiteral("CAD 坐标轴与截面策略"), this);
    auto* coordinateForm = new QFormLayout(coordinateGroup);

    axialAxisCombo_ = new QComboBox(coordinateGroup);
    axialAxisCombo_->addItems({QStringLiteral("X"), QStringLiteral("Y"), QStringLiteral("Z")});
    axialAxisCombo_->setCurrentText(QStringLiteral("X"));
    radialAxisCombo_ = new QComboBox(coordinateGroup);
    radialAxisCombo_->addItems({QStringLiteral("Y"), QStringLiteral("X"), QStringLiteral("Z")});
    reverseAxialCheck_ = new QCheckBox(QStringLiteral("采集方向与 CAD 轴向相反时启用"), coordinateGroup);
    reverseAxialCheck_->setChecked(false);
    reverseAxialCheck_->setToolTip(
        QStringLiteral("真实 STEP/STP/IGES 模型默认保持 CAD 原始轴向，不做 Z 轴特殊映射；只有确认采集方向与 CAD 轴向相反时才勾选。"));
    useLeftEndpointOriginCheck_ = new QCheckBox(QStringLiteral("使用左端点作为工件坐标原点"), coordinateGroup);
    useLeftEndpointOriginCheck_->setChecked(true);
    extractUpperEnvelopeCheck_ = new QCheckBox(QStringLiteral("优先提取上包络母线"), coordinateGroup);
    extractUpperEnvelopeCheck_->setChecked(false);

    coordinateForm->addRow(QStringLiteral("轴向"), axialAxisCombo_);
    coordinateForm->addRow(QStringLiteral("径向"), radialAxisCombo_);
    coordinateForm->addRow(QStringLiteral("采集方向校正"), reverseAxialCheck_);
    coordinateForm->addRow(QStringLiteral("原点锚定"), useLeftEndpointOriginCheck_);
    coordinateForm->addRow(QStringLiteral("截面提取"), extractUpperEnvelopeCheck_);
    auto* coordinateHintLabel = new QLabel(
        QStringLiteral("说明：外部 CAD 默认使用原始坐标轴正向；此处不是 Z 轴映射，仅用于现场采集方向与 CAD 轴向相反时校正。"),
        coordinateGroup);
    coordinateHintLabel->setWordWrap(true);
    coordinateHintLabel->setObjectName(QStringLiteral("sectionHelpText"));
    coordinateForm->addRow(coordinateHintLabel);

    auto* featureGroup = new QGroupBox(QStringLiteral("目标尺寸与采样"), this);
    auto* featureForm = new QFormLayout(featureGroup);

    samplingStepSpin_ = new QDoubleSpinBox(featureGroup);
    samplingStepSpin_->setRange(0.001, 5.0);
    samplingStepSpin_->setDecimals(3);
    samplingStepSpin_->setSingleStep(0.005);
    samplingStepSpin_->setValue(0.005);
    samplingStepSpin_->setSuffix(QStringLiteral(" mm"));

    targetSlotWidthSpin_ = new QDoubleSpinBox(featureGroup);
    targetSlotWidthSpin_->setRange(0.0, 500.0);
    targetSlotWidthSpin_->setDecimals(4);
    targetSlotWidthSpin_->setSingleStep(0.01);
    targetSlotWidthSpin_->setValue(0.0500);
    targetSlotWidthSpin_->setSuffix(QStringLiteral(" mm"));
    targetSlotWidthSpin_->setToolTip(
        QStringLiteral("大于 0 时启用单槽宽目标解算；当前金刚石微流槽默认目标宽度为 0.05 mm。"));

    targetSlotDepthSpin_ = new QDoubleSpinBox(featureGroup);
    targetSlotDepthSpin_->setRange(0.0, 500.0);
    targetSlotDepthSpin_->setDecimals(4);
    targetSlotDepthSpin_->setSingleStep(0.01);
    targetSlotDepthSpin_->setSuffix(QStringLiteral(" mm"));
    targetSlotDepthSpin_->setToolTip(QStringLiteral("可选；大于 0 时同步输出槽深误差和补偿量。"));

    temporaryPixelSizeSpin_ = new QDoubleSpinBox(featureGroup);
    temporaryPixelSizeSpin_->setRange(0.0, 10.0);
    temporaryPixelSizeSpin_->setDecimals(9);
    temporaryPixelSizeSpin_->setSingleStep(0.001);
    temporaryPixelSizeSpin_->setSuffix(QStringLiteral(" mm/px"));
    temporaryPixelSizeSpin_->setValue(0.001614);
    temporaryPixelSizeSpin_->setToolTip(
        QStringLiteral("仅用于临时测试；当前微流槽案例默认 0.001614 mm/px，使 2.png 检测槽宽接近 0.05 mm。正式检测请按真实标定填写。"));

    localSlotImageModeCheck_ = new QCheckBox(QStringLiteral("局部槽特征检测"), featureGroup);
    localSlotImageModeCheck_->setToolTip(
        QStringLiteral("启用后，输入图像应是已经截取好的单槽局部图；软件直接做二值化、边缘检测、槽底标定、误差与补偿输出。"));

    localSlotBottomWidthSpin_ = new QDoubleSpinBox(featureGroup);
    localSlotBottomWidthSpin_->setRange(0.001, 500.0);
    localSlotBottomWidthSpin_->setDecimals(4);
    localSlotBottomWidthSpin_->setSingleStep(0.001);
    localSlotBottomWidthSpin_->setValue(0.050);
    localSlotBottomWidthSpin_->setSuffix(QStringLiteral(" mm"));
    localSlotBottomWidthSpin_->setToolTip(QStringLiteral("局部槽图中需要验收的真实槽宽；当前金刚石微流槽默认 0.05 mm。"));

    auto* singleSlotHintLabel = new QLabel(
        QStringLiteral("提示：若只复检一个槽宽，请填写目标槽宽；若输入已是 CAD/相机截取好的局部槽图，请启用局部槽特征检测并填写槽底真实距离。"),
        featureGroup);
    singleSlotHintLabel->setWordWrap(true);
    singleSlotHintLabel->setObjectName(QStringLiteral("sectionHelpText"));

    featureForm->addRow(QStringLiteral("截面采样步长"), samplingStepSpin_);
    featureForm->addRow(QStringLiteral("目标槽宽 / 单槽宽"), targetSlotWidthSpin_);
    featureForm->addRow(QStringLiteral("目标槽深"), targetSlotDepthSpin_);
    featureForm->addRow(QStringLiteral("临时像素当量"), temporaryPixelSizeSpin_);
    featureForm->addRow(QStringLiteral("局部槽特征检测"), localSlotImageModeCheck_);
    featureForm->addRow(QStringLiteral("槽底标定宽度"), localSlotBottomWidthSpin_);
    featureForm->addRow(singleSlotHintLabel);

    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("模型输入"), sourceGroup, true, this));
    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("坐标定义"), coordinateGroup, true, this));
    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("特征参数"), featureGroup, true, this));
    rootLayout->addStretch(1);

    updateBackendStatusLabel();

    connect(browseCadFileButton_, &QPushButton::clicked, this, [this]() {
        const QString filePath = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("选择 CAD 模型"),
            cadFilePathEdit_->text(),
            QStringLiteral("CAD Files (*.step *.stp *.iges *.igs);;STEP/STP Files (*.step *.stp);;IGES Files (*.iges *.igs)"));
        if (filePath.isEmpty()) {
            return;
        }

        cadFilePathEdit_->setText(QDir::toNativeSeparators(filePath));
        if (modelNameEdit_->text().trimmed().isEmpty()) {
            modelNameEdit_->setText(QFileInfo(filePath).completeBaseName());
        }
        emit configChanged();
    });

    connect(browseScannedStlButton_, &QPushButton::clicked, this, [this]() {
        const QString filePath = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("选择扫描 STL"),
            scannedStlFilePathEdit_ ? scannedStlFilePathEdit_->text() : QString(),
            QStringLiteral("STL Files (*.stl)"));
        if (filePath.isEmpty()) {
            return;
        }

        scannedStlFilePathEdit_->setText(QDir::toNativeSeparators(filePath));
        emit configChanged();
    });

    connect(importCadFileButton_, &QPushButton::clicked, this, [this]() {
        emit importRequested(cadFilePathEdit_ ? cadFilePathEdit_->text().trimmed() : QString());
    });

    connect(compareScannedStlButton_, &QPushButton::clicked, this, [this]() {
        emit scanCompareRequested(scannedStlFilePathEdit_ ? scannedStlFilePathEdit_->text().trimmed() : QString());
    });

    connect(cadFilePathEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(scannedStlFilePathEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(modelNameEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(axialAxisCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { emit configChanged(); });
    connect(radialAxisCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { emit configChanged(); });
    connect(reverseAxialCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(useLeftEndpointOriginCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(extractUpperEnvelopeCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(samplingStepSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { emit configChanged(); });
    connect(targetSlotWidthSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { emit configChanged(); });
    connect(targetSlotDepthSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { emit configChanged(); });
    connect(temporaryPixelSizeSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { emit configChanged(); });
    connect(localSlotImageModeCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(localSlotBottomWidthSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { emit configChanged(); });
}

bool DesignModelPanel::buildRequest(pinjie::cad_model::DesignModelRequest& request, QString& errorMessage) const
{
    const QString filePath = cadFilePathEdit_ ? cadFilePathEdit_->text().trimmed() : QString();
    const bool localSlotImageMode = localSlotImageModeCheck_ && localSlotImageModeCheck_->isChecked();
    if (filePath.isEmpty() && !localSlotImageMode) {
        errorMessage = QStringLiteral("请先选择 STEP、STP 或 IGES 设计文件。");
        return false;
    }

    QFileInfo info(filePath);
    if (!filePath.isEmpty() && (!info.exists() || !info.isFile())) {
        errorMessage = QStringLiteral("所选 CAD 文件不存在。");
        return false;
    }

    request = {};
    if (!filePath.isEmpty()) {
        request.cadFilePath = QDir::fromNativeSeparators(filePath).toUtf8().toStdString();
    }
    request.modelName = (modelNameEdit_ && !modelNameEdit_->text().trimmed().isEmpty()
                             ? modelNameEdit_->text().trimmed()
                             : (filePath.isEmpty() ? QStringLiteral("local_slot_image") : info.completeBaseName()))
                            .toUtf8()
                            .toStdString();
    request.axialAxis = axialAxisCombo_ ? axialAxisCombo_->currentText().toStdString() : std::string("X");
    request.radialAxis = radialAxisCombo_ ? radialAxisCombo_->currentText().toStdString() : std::string("Y");
    if (request.axialAxis == request.radialAxis) {
        errorMessage = QStringLiteral("轴向和径向不能选择同一坐标轴。");
        return false;
    }
    request.reverseAxialDirection = reverseAxialCheck_ && reverseAxialCheck_->isChecked();
    request.useLeftEndpointAsOrigin = useLeftEndpointOriginCheck_ && useLeftEndpointOriginCheck_->isChecked();
    request.extractUpperEnvelope = extractUpperEnvelopeCheck_ && extractUpperEnvelopeCheck_->isChecked();
    request.profileSamplingStepMm = samplingStepSpin_ ? samplingStepSpin_->value() : 0.05;
    request.targetSlotWidthMm = targetSlotWidthSpin_ ? targetSlotWidthSpin_->value() : 0.0;
    request.targetSlotDepthMm = targetSlotDepthSpin_ ? targetSlotDepthSpin_->value() : 0.0;
    request.temporaryPixelSizeMm = temporaryPixelSizeSpin_ ? temporaryPixelSizeSpin_->value() : 0.0;
    request.localSlotImageMode = localSlotImageMode;
    request.localSlotBottomWidthMm = localSlotBottomWidthSpin_ ? localSlotBottomWidthSpin_->value() : 1.527;
    if (request.localSlotImageMode && request.targetSlotWidthMm <= 0.0) {
        request.targetSlotWidthMm = request.localSlotBottomWidthMm;
    }
    return true;
}

void DesignModelPanel::setRunning(bool running)
{
    if (cadFilePathEdit_) {
        cadFilePathEdit_->setEnabled(!running);
    }
    if (modelNameEdit_) {
        modelNameEdit_->setEnabled(!running);
    }
    if (axialAxisCombo_) {
        axialAxisCombo_->setEnabled(!running);
    }
    if (radialAxisCombo_) {
        radialAxisCombo_->setEnabled(!running);
    }
    if (reverseAxialCheck_) {
        reverseAxialCheck_->setEnabled(!running);
    }
    if (useLeftEndpointOriginCheck_) {
        useLeftEndpointOriginCheck_->setEnabled(!running);
    }
    if (extractUpperEnvelopeCheck_) {
        extractUpperEnvelopeCheck_->setEnabled(!running);
    }
    if (samplingStepSpin_) {
        samplingStepSpin_->setEnabled(!running);
    }
    if (targetSlotWidthSpin_) {
        targetSlotWidthSpin_->setEnabled(!running);
    }
    if (targetSlotDepthSpin_) {
        targetSlotDepthSpin_->setEnabled(!running);
    }
    if (temporaryPixelSizeSpin_) {
        temporaryPixelSizeSpin_->setEnabled(!running);
    }
    if (localSlotImageModeCheck_) {
        localSlotImageModeCheck_->setEnabled(!running);
    }
    if (localSlotBottomWidthSpin_) {
        localSlotBottomWidthSpin_->setEnabled(!running);
    }
    if (browseCadFileButton_) {
        browseCadFileButton_->setEnabled(!running);
    }
    if (importCadFileButton_) {
        importCadFileButton_->setEnabled(!running);
    }
    if (scannedStlFilePathEdit_) {
        scannedStlFilePathEdit_->setEnabled(!running);
    }
    if (browseScannedStlButton_) {
        browseScannedStlButton_->setEnabled(!running);
    }
    if (compareScannedStlButton_) {
        compareScannedStlButton_->setEnabled(!running);
    }
}

void DesignModelPanel::setDocumentInfo(const pinjie::cad_model::CadModelDocument& document)
{
    updateBackendStatusLabel();
    if (!importedModelLabel_) {
        return;
    }

    importedModelLabel_->setText(
        document.valid
            ? QStringLiteral("导入状态：已加载 %1 | 面 %2 | 边 %3 | 轮廓采样 %4")
                  .arg(fromUtf8StdString(document.fileName))
                  .arg(document.faceCount)
                  .arg(document.edgeCount)
                  .arg(static_cast<qulonglong>(document.profileSamples.size()))
            : QStringLiteral("导入状态：%1")
                  .arg(document.importMessage.empty()
                           ? QStringLiteral("尚未加载设计模型。")
                           : fromUtf8StdString(document.importMessage)));
}

void DesignModelPanel::clearDocumentInfo()
{
    updateBackendStatusLabel();
    if (importedModelLabel_) {
        importedModelLabel_->setText(QStringLiteral("导入状态：尚未加载设计模型。"));
    }
}

void DesignModelPanel::updateBackendStatusLabel()
{
    if (!backendStatusLabel_) {
        return;
    }
    backendStatusLabel_->setText(
        QStringLiteral("%1。当前设计模型支持 STEP/STP/IGES；扫描件独立比对入口支持 STL。")
            .arg(fromUtf8StdString(pinjie::cad_model::occtCadImportBackendSummary())));
}

} // namespace pinjie::gui
