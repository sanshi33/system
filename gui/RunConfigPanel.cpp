#include "gui/RunConfigPanel.h"
#include "gui/CameraCaptureWorker.h"
#include "gui/ImageViewer.h"

#ifdef PINJIE_ENABLE_GALAXY_CAMERA
#include "acquisition/CameraCaptureService.h"
#include "acquisition/GalaxyCameraDevice.h"
#endif

#include "common/ResultPathUtils.h"

#include <QCollator>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMetaObject>
#include <QPushButton>
#include <QRegularExpression>
#include <QThread>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>
#include <exception>
#include <vector>

namespace pinjie::gui {

namespace {

QString defaultWorkpieceImageDir()
{
#ifdef PINJIE_PROJECT_ROOT
    const QString root = QString::fromUtf8(PINJIE_PROJECT_ROOT);
#else
    const QString root = QDir::currentPath();
#endif
    const QString caseDir = QDir(root).filePath(QStringLiteral("test_cases/online_slot_cases/images/test"));
    return QDir(caseDir).exists() ? QDir::toNativeSeparators(caseDir) : QString();
}

QWidget* makeSectionPanel(const QString& objectName, QWidget* parent)
{
    auto* panel = new QWidget(parent);
    panel->setObjectName(objectName);
    return panel;
}

QWidget* makeCollapsibleSection(const QString& title, QWidget* content, bool expanded, QWidget* parent)
{
    auto* container = new QWidget(parent);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    auto* toggleButton = new QToolButton(container);
    toggleButton->setObjectName(QStringLiteral("configSectionToggle"));
    toggleButton->setText(title);
    toggleButton->setCheckable(true);
    toggleButton->setChecked(expanded);
    toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleButton->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);

    content->setVisible(expanded);

    QObject::connect(toggleButton, &QToolButton::toggled, content, [toggleButton, content](bool checked) {
        toggleButton->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        content->setVisible(checked);
    });

    layout->addWidget(toggleButton);
    layout->addWidget(content);
    return container;
}

} // namespace

RunConfigPanel::RunConfigPanel(QWidget* parent)
    : QWidget(parent)
{
    hide();

    acquisitionSection_ = makeSectionPanel(QStringLiteral("acquisitionConfigSection"), this);
    auto* acquisitionLayout = new QVBoxLayout(acquisitionSection_);
    acquisitionLayout->setContentsMargins(0, 0, 0, 0);
    acquisitionLayout->setSpacing(8);

    auto* inputGroup = new QGroupBox(QStringLiteral("工件图像输入"), acquisitionSection_);
    auto* inputLayout = new QVBoxLayout(inputGroup);

    auto* dirRow = new QHBoxLayout();
    inputDirEdit_ = new QLineEdit(defaultWorkpieceImageDir(), inputGroup);
    inputDirEdit_->setPlaceholderText(QStringLiteral("选择图像目录"));
    inputDirEdit_->setToolTip(QStringLiteral("支持 bmp、png、jpg、jpeg、tif、tiff 格式"));
    auto* browseButton = new QPushButton(QStringLiteral("浏览"), inputGroup);
    dirRow->addWidget(inputDirEdit_, 1);
    dirRow->addWidget(browseButton);
    inputLayout->addLayout(dirRow);

    imageLimitSpin_ = new QSpinBox(inputGroup);
    imageLimitSpin_->setRange(0, 9999);
    imageLimitSpin_->setValue(0);
    imageLimitSpin_->setSpecialValueText(QStringLiteral("全部"));
    imageLimitSpin_->setToolTip(QStringLiteral("0 表示使用目录中的全部图像"));

    detectedCountLabel_ = new QLabel(QStringLiteral("已检测 0 张图像"), inputGroup);

    auto* inputForm = new QFormLayout();
    inputForm->addRow(QStringLiteral("使用数量"), imageLimitSpin_);
    inputForm->addRow(QStringLiteral("目录统计"), detectedCountLabel_);
    inputLayout->addLayout(inputForm);
    acquisitionLayout->addWidget(makeCollapsibleSection(QStringLiteral("基础参数"), inputGroup, true, acquisitionSection_));
    auto* cameraGroup = new QGroupBox(QStringLiteral("图像导入与相机采集"), acquisitionSection_);
    auto* cameraLayout = new QVBoxLayout(cameraGroup);

    auto* cameraDeviceRow = new QHBoxLayout();
    cameraDeviceCombo_ = new QComboBox(cameraGroup);
    cameraDeviceCombo_->addItem(QStringLiteral("未刷新设备"), QString());
    refreshCameraButton_ = new QPushButton(QStringLiteral("刷新"), cameraGroup);
    cameraDeviceRow->addWidget(cameraDeviceCombo_, 1);
    cameraDeviceRow->addWidget(refreshCameraButton_);
    cameraLayout->addLayout(cameraDeviceRow);

    auto* cameraOutputRow = new QHBoxLayout();
    cameraOutputDirEdit_ = new QLineEdit(defaultCameraOutputDir(), cameraGroup);
    cameraOutputDirEdit_->setPlaceholderText(QStringLiteral("相机图像保存目录"));
    cameraBrowseButton_ = new QPushButton(QStringLiteral("浏览"), cameraGroup);
    cameraOutputRow->addWidget(cameraOutputDirEdit_, 1);
    cameraOutputRow->addWidget(cameraBrowseButton_);
    cameraLayout->addLayout(cameraOutputRow);

    cameraSessionNameEdit_ = new QLineEdit(QStringLiteral("camera_capture"), cameraGroup);
    cameraSessionNameEdit_->setPlaceholderText(QStringLiteral("采集会话名称"));

    auto* cameraForm = new QFormLayout();
    cameraFrameCountSpin_ = new QSpinBox(cameraGroup);
    cameraFrameCountSpin_->setRange(1, 9999);
    cameraFrameCountSpin_->setValue(1);

    cameraIntervalSpin_ = new QSpinBox(cameraGroup);
    cameraIntervalSpin_->setRange(0, 600000);
    cameraIntervalSpin_->setSuffix(QStringLiteral(" ms"));
    cameraIntervalSpin_->setValue(0);

    cameraSetExposureCheck_ = new QCheckBox(QStringLiteral("设置"), cameraGroup);
    cameraExposureSpin_ = new QDoubleSpinBox(cameraGroup);
    cameraExposureSpin_->setRange(1.0, 10000000.0);
    cameraExposureSpin_->setDecimals(1);
    cameraExposureSpin_->setValue(10000.0);
    cameraExposureSpin_->setSuffix(QStringLiteral(" us"));
    auto* exposureRow = new QHBoxLayout();
    exposureRow->addWidget(cameraSetExposureCheck_);
    exposureRow->addWidget(cameraExposureSpin_, 1);

    cameraSetGainCheck_ = new QCheckBox(QStringLiteral("设置"), cameraGroup);
    cameraGainSpin_ = new QDoubleSpinBox(cameraGroup);
    cameraGainSpin_->setRange(0.0, 48.0);
    cameraGainSpin_->setDecimals(2);
    cameraGainSpin_->setValue(0.0);
    auto* gainRow = new QHBoxLayout();
    gainRow->addWidget(cameraSetGainCheck_);
    gainRow->addWidget(cameraGainSpin_, 1);

    cameraSoftwareTriggerCheck_ = new QCheckBox(QStringLiteral("软触发"), cameraGroup);
    cameraSoftwareTriggerCheck_->setChecked(false);

    cameraWriteStagePositionCheck_ = new QCheckBox(QStringLiteral("写入位置"), cameraGroup);
    cameraWriteStagePositionCheck_->setChecked(false);

    cameraStepIndexSpin_ = new QSpinBox(cameraGroup);
    cameraStepIndexSpin_->setRange(1, 999999);
    cameraStepIndexSpin_->setValue(1);

    cameraStageXSpin_ = new QDoubleSpinBox(cameraGroup);
    cameraStageXSpin_->setRange(-1000000.0, 1000000.0);
    cameraStageXSpin_->setDecimals(3);
    cameraStageXSpin_->setSuffix(QStringLiteral(" mm"));

    cameraStageYSpin_ = new QDoubleSpinBox(cameraGroup);
    cameraStageYSpin_->setRange(-1000000.0, 1000000.0);
    cameraStageYSpin_->setDecimals(3);
    cameraStageYSpin_->setSuffix(QStringLiteral(" mm"));

    cameraNoteEdit_ = new QLineEdit(cameraGroup);
    cameraNoteEdit_->setPlaceholderText(QStringLiteral("当前帧备注（可选）"));

    cameraForm->addRow(QStringLiteral("采集数量"), cameraFrameCountSpin_);
    cameraForm->addRow(QStringLiteral("采集间隔"), cameraIntervalSpin_);
    cameraForm->addRow(QStringLiteral("曝光时间"), exposureRow);
    cameraForm->addRow(QStringLiteral("增益"), gainRow);
    cameraForm->addRow(QStringLiteral("触发模式"), cameraSoftwareTriggerCheck_);
    cameraForm->addRow(QStringLiteral("会话名称"), cameraSessionNameEdit_);
    cameraForm->addRow(QStringLiteral("手动步号"), cameraStepIndexSpin_);
    cameraForm->addRow(QStringLiteral("位置元数据"), cameraWriteStagePositionCheck_);
    cameraForm->addRow(QStringLiteral("当前位置 X"), cameraStageXSpin_);
    cameraForm->addRow(QStringLiteral("当前位置 Y"), cameraStageYSpin_);
    cameraForm->addRow(QStringLiteral("当前备注"), cameraNoteEdit_);
    cameraLayout->addLayout(cameraForm);

    cameraCaptureButton_ = new QPushButton(QStringLiteral("开始采集"), cameraGroup);
    cameraStatusLabel_ = new QLabel(QStringLiteral("相机采集未开始"), cameraGroup);
    cameraStatusLabel_->setWordWrap(true);
    cameraSingleCaptureButton_ = new QPushButton(QStringLiteral("采一张"), cameraGroup);
    cameraManualStepButton_ = new QPushButton(QStringLiteral("采当前步"), cameraGroup);
    cameraLayout->addWidget(cameraCaptureButton_);
    cameraLayout->addWidget(cameraSingleCaptureButton_);
    cameraLayout->addWidget(cameraManualStepButton_);
    cameraCancelButton_ = new QPushButton(QStringLiteral("取消采集"), cameraGroup);
    cameraCancelButton_->setEnabled(false);
    cameraLayout->addWidget(cameraCancelButton_);
    cameraLayout->addWidget(cameraStatusLabel_);
    cameraPreviewInfoLabel_ = new QLabel(QStringLiteral("导入图像或相机采集预览会显示在下方。"), cameraGroup);
    cameraPreviewInfoLabel_->setWordWrap(true);
    cameraPreviewViewer_ = new ImageViewer(cameraGroup);
    cameraPreviewViewer_->setMinimumHeight(220);
    cameraLayout->addWidget(cameraPreviewInfoLabel_);
    cameraLayout->addWidget(cameraPreviewViewer_);

#ifndef PINJIE_ENABLE_GALAXY_CAMERA
    setCameraControlsEnabled(false);
    cameraStatusLabel_->setText(QStringLiteral("当前构建未启用 Galaxy 相机 SDK；仍可预览导入的工件图像。"));
#endif

    acquisitionLayout->addWidget(makeCollapsibleSection(QStringLiteral("图像导入与相机采集"), cameraGroup, true, acquisitionSection_));
    acquisitionLayout->addStretch(1);

    processingSection_ = makeSectionPanel(QStringLiteral("processingConfigSection"), this);
    auto* processingLayout = new QVBoxLayout(processingSection_);
    processingLayout->setContentsMargins(0, 0, 0, 0);
    processingLayout->setSpacing(8);

    auto* edgeGroup = new QGroupBox(QStringLiteral("边缘检测"), processingSection_);
    auto* edgeForm = new QFormLayout(edgeGroup);

    cannyLowSpin_ = new QDoubleSpinBox(edgeGroup);
    cannyLowSpin_->setRange(0.0, 255.0);
    cannyLowSpin_->setDecimals(1);
    cannyLowSpin_->setValue(50.0);

    cannyHighSpin_ = new QDoubleSpinBox(edgeGroup);
    cannyHighSpin_->setRange(0.0, 255.0);
    cannyHighSpin_->setDecimals(1);
    cannyHighSpin_->setValue(150.0);

    subpixWindowSpin_ = new QSpinBox(edgeGroup);
    subpixWindowSpin_->setRange(3, 31);
    subpixWindowSpin_->setSingleStep(2);
    subpixWindowSpin_->setValue(7);

    subpixSigmaSpin_ = new QDoubleSpinBox(edgeGroup);
    subpixSigmaSpin_->setRange(0.1, 10.0);
    subpixSigmaSpin_->setDecimals(2);
    subpixSigmaSpin_->setSingleStep(0.1);
    subpixSigmaSpin_->setValue(1.0);

    edgeForm->addRow(QStringLiteral("Canny 低阈值"), cannyLowSpin_);
    edgeForm->addRow(QStringLiteral("Canny 高阈值"), cannyHighSpin_);
    edgeForm->addRow(QStringLiteral("亚像素窗口"), subpixWindowSpin_);
    edgeForm->addRow(QStringLiteral("亚像素 Sigma"), subpixSigmaSpin_);

    auto* filterGroup = new QGroupBox(QStringLiteral("点过滤"), processingSection_);
    auto* filterForm = new QFormLayout(filterGroup);

    pointFilterCheck_ = new QCheckBox(QStringLiteral("启用"), filterGroup);
    pointFilterCheck_->setChecked(true);

    filterConfidenceSpin_ = new QDoubleSpinBox(filterGroup);
    filterConfidenceSpin_->setRange(0.0, 100.0);
    filterConfidenceSpin_->setDecimals(1);
    filterConfidenceSpin_->setValue(15.0);
    filterConfidenceSpin_->setSuffix(QStringLiteral(" %"));

    filterGradientSpin_ = new QDoubleSpinBox(filterGroup);
    filterGradientSpin_->setRange(0.0, 100.0);
    filterGradientSpin_->setDecimals(1);
    filterGradientSpin_->setValue(15.0);
    filterGradientSpin_->setSuffix(QStringLiteral(" %"));

    filterWindowRadiusSpin_ = new QSpinBox(filterGroup);
    filterWindowRadiusSpin_->setRange(1, 99);
    filterWindowRadiusSpin_->setValue(5);

    filterHampelSigmaSpin_ = new QDoubleSpinBox(filterGroup);
    filterHampelSigmaSpin_->setRange(0.1, 20.0);
    filterHampelSigmaSpin_->setDecimals(2);
    filterHampelSigmaSpin_->setValue(3.0);

    filterHampelMinScaleSpin_ = new QDoubleSpinBox(filterGroup);
    filterHampelMinScaleSpin_->setRange(0.001, 20.0);
    filterHampelMinScaleSpin_->setDecimals(3);
    filterHampelMinScaleSpin_->setValue(0.05);
    filterHampelMinScaleSpin_->setSuffix(QStringLiteral(" px"));

    filterForm->addRow(QStringLiteral("启用过滤"), pointFilterCheck_);
    filterForm->addRow(QStringLiteral("置信度分位"), filterConfidenceSpin_);
    filterForm->addRow(QStringLiteral("梯度分位"), filterGradientSpin_);
    filterForm->addRow(QStringLiteral("窗口半径"), filterWindowRadiusSpin_);
    filterForm->addRow(QStringLiteral("Hampel Sigma"), filterHampelSigmaSpin_);
    filterForm->addRow(QStringLiteral("最小尺度"), filterHampelMinScaleSpin_);

    processingLayout->addWidget(makeCollapsibleSection(QStringLiteral("基础参数"), edgeGroup, true, processingSection_));
    processingLayout->addWidget(makeCollapsibleSection(QStringLiteral("高级参数"), filterGroup, false, processingSection_));
    processingLayout->addStretch(1);

    registrationSection_ = makeSectionPanel(QStringLiteral("registrationConfigSection"), this);
    auto* registrationLayout = new QVBoxLayout(registrationSection_);
    registrationLayout->setContentsMargins(0, 0, 0, 0);
    registrationLayout->setSpacing(8);

    auto* stitchGroup = new QGroupBox(QStringLiteral("拼接搜索"), registrationSection_);
    auto* stitchForm = new QFormLayout(stitchGroup);

    overlapSpin_ = new QDoubleSpinBox(stitchGroup);
    overlapSpin_->setRange(0.0, 100.0);
    overlapSpin_->setDecimals(2);
    overlapSpin_->setValue(87.5);
    overlapSpin_->setSuffix(QStringLiteral(" %"));

    directionCombo_ = new QComboBox(stitchGroup);
    directionCombo_->addItem(QStringLiteral("X+"), static_cast<int>(pinjie::MotionPriorDirection::XPositive));
    directionCombo_->addItem(QStringLiteral("X-"), static_cast<int>(pinjie::MotionPriorDirection::XNegative));
    directionCombo_->addItem(QStringLiteral("Y+"), static_cast<int>(pinjie::MotionPriorDirection::YPositive));
    directionCombo_->addItem(QStringLiteral("Y-"), static_cast<int>(pinjie::MotionPriorDirection::YNegative));
    directionCombo_->insertSeparator(directionCombo_->count());
    directionCombo_->addItem(QStringLiteral("自动"), static_cast<int>(pinjie::MotionPriorDirection::Auto));
    directionCombo_->setCurrentIndex(0);
    directionCombo_->setToolTip(QStringLiteral("采集方向已知时，固定方向先验可以减少搜索时间"));

    searchRangeSpin_ = new QDoubleSpinBox(stitchGroup);
    searchRangeSpin_->setRange(1.0, 100000.0);
    searchRangeSpin_->setDecimals(1);
    searchRangeSpin_->setValue(200.0);
    searchRangeSpin_->setSuffix(QStringLiteral(" px"));
    searchRangeSpin_->setToolTip(QStringLiteral("方向和重叠率确定后，适当缩小搜索范围通常会更快"));

    rotationRangeSpin_ = new QDoubleSpinBox(stitchGroup);
    rotationRangeSpin_->setRange(0.0, 30.0);
    rotationRangeSpin_->setDecimals(2);
    rotationRangeSpin_->setSingleStep(0.1);
    rotationRangeSpin_->setValue(0.5);
    rotationRangeSpin_->setSuffix(QStringLiteral(" deg"));
    rotationRangeSpin_->setToolTip(QStringLiteral("夹具越稳定，旋转搜索范围可以设置得越小"));

    rotationStepSpin_ = new QDoubleSpinBox(stitchGroup);
    rotationStepSpin_->setRange(0.001, 5.0);
    rotationStepSpin_->setDecimals(3);
    rotationStepSpin_->setSingleStep(0.01);
    rotationStepSpin_->setValue(0.01);
    rotationStepSpin_->setSuffix(QStringLiteral(" deg"));

    tangentResidualWeightSpin_ = new QDoubleSpinBox(stitchGroup);
    tangentResidualWeightSpin_->setRange(0.0, 10.0);
    tangentResidualWeightSpin_->setDecimals(3);
    tangentResidualWeightSpin_->setSingleStep(0.01);
    tangentResidualWeightSpin_->setValue(0.05);
    tangentResidualWeightSpin_->setToolTip(QStringLiteral("将切向 RMSE^2 按该权重加入匹配代价；设为 0 可关闭"));

    tangentCorrelationWeightSpin_ = new QDoubleSpinBox(stitchGroup);
    tangentCorrelationWeightSpin_->setRange(0.0, 10.0);
    tangentCorrelationWeightSpin_->setDecimals(3);
    tangentCorrelationWeightSpin_->setSingleStep(0.05);
    tangentCorrelationWeightSpin_->setValue(0.25);
    tangentCorrelationWeightSpin_->setToolTip(QStringLiteral("将 1 - 切向相关性按该权重加入匹配代价；设为 0 可关闭"));

    stitchForm->addRow(QStringLiteral("重叠率"), overlapSpin_);
    stitchForm->addRow(QStringLiteral("方向先验"), directionCombo_);
    stitchForm->addRow(QStringLiteral("搜索范围"), searchRangeSpin_);
    stitchForm->addRow(QStringLiteral("旋转范围"), rotationRangeSpin_);
    stitchForm->addRow(QStringLiteral("旋转步长"), rotationStepSpin_);
    stitchForm->addRow(QStringLiteral("切向残差权重"), tangentResidualWeightSpin_);
    stitchForm->addRow(QStringLiteral("切向相关权重"), tangentCorrelationWeightSpin_);

    registrationLayout->addWidget(makeCollapsibleSection(QStringLiteral("基础参数"), stitchGroup, true, registrationSection_));

    auto* standardCircleGroup = new QGroupBox(QStringLiteral("标准圆国标检测"), registrationSection_);
    auto* standardCircleLayout = new QVBoxLayout(standardCircleGroup);
    standardCircleLayout->setContentsMargins(8, 10, 8, 8);
    standardCircleLayout->setSpacing(6);

    standardCircleCheck_ = new QCheckBox(QStringLiteral("启用 GB/T 24762 5.7 标准圆检测流程"), standardCircleGroup);
    standardCircleCheck_->setChecked(false);
    standardCircleLayout->addWidget(standardCircleCheck_);

    auto* standardCircleHintLabel = new QLabel(
        QStringLiteral("该模式直接调用已验证的标准圆流程：掩模板预处理、全局拼接、椭圆修正、25 个测量窗口选点与 E_P2D 输出。"),
        standardCircleGroup);
    standardCircleHintLabel->setWordWrap(true);
    standardCircleLayout->addWidget(standardCircleHintLabel);

    auto* standardCircleForm = new QFormLayout();

    standardCirclePrefixEdit_ = new QLineEdit(QStringLiteral("Pic_"), standardCircleGroup);
    standardCircleExtensionEdit_ = new QLineEdit(QStringLiteral(".bmp"), standardCircleGroup);
    standardCircleStartIndexSpin_ = new QSpinBox(standardCircleGroup);
    standardCircleStartIndexSpin_->setRange(1, 999999);
    standardCircleStartIndexSpin_->setValue(1);

    standardCircleDiameterSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleDiameterSpin_->setRange(0.001, 100000.0);
    standardCircleDiameterSpin_->setDecimals(5);
    standardCircleDiameterSpin_->setValue(19.99995);
    standardCircleDiameterSpin_->setSuffix(QStringLiteral(" mm"));

    standardCircleHorizontalFovSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleHorizontalFovSpin_->setRange(0.001, 100000.0);
    standardCircleHorizontalFovSpin_->setDecimals(3);
    standardCircleHorizontalFovSpin_->setValue(40.0);
    standardCircleHorizontalFovSpin_->setSuffix(QStringLiteral(" mm"));

    standardCircleVerticalFovSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleVerticalFovSpin_->setRange(0.001, 100000.0);
    standardCircleVerticalFovSpin_->setDecimals(3);
    standardCircleVerticalFovSpin_->setValue(30.0);
    standardCircleVerticalFovSpin_->setSuffix(QStringLiteral(" mm"));

    standardCircleOverlapSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleOverlapSpin_->setRange(0.0, 100.0);
    standardCircleOverlapSpin_->setDecimals(2);
    standardCircleOverlapSpin_->setValue(70.0);
    standardCircleOverlapSpin_->setSuffix(QStringLiteral(" %"));

    standardCircleWindowHalfSizeSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleWindowHalfSizeSpin_->setRange(1.0, 100000.0);
    standardCircleWindowHalfSizeSpin_->setDecimals(1);
    standardCircleWindowHalfSizeSpin_->setValue(60.0);
    standardCircleWindowHalfSizeSpin_->setSuffix(QStringLiteral(" px"));

    standardCircleMedianRadiusSpin_ = new QSpinBox(standardCircleGroup);
    standardCircleMedianRadiusSpin_->setRange(0, 99);
    standardCircleMedianRadiusSpin_->setValue(10);

    standardCircleFilterBlendSpin_ = new QDoubleSpinBox(standardCircleGroup);
    standardCircleFilterBlendSpin_->setRange(0.0, 1.0);
    standardCircleFilterBlendSpin_->setDecimals(3);
    standardCircleFilterBlendSpin_->setSingleStep(0.005);
    standardCircleFilterBlendSpin_->setValue(0.950);

    standardCircleForm->addRow(QStringLiteral("文件前缀"), standardCirclePrefixEdit_);
    standardCircleForm->addRow(QStringLiteral("文件后缀"), standardCircleExtensionEdit_);
    standardCircleForm->addRow(QStringLiteral("起始序号"), standardCircleStartIndexSpin_);
    standardCircleForm->addRow(QStringLiteral("标准圆直径"), standardCircleDiameterSpin_);
    standardCircleForm->addRow(QStringLiteral("水平视场"), standardCircleHorizontalFovSpin_);
    standardCircleForm->addRow(QStringLiteral("垂直视场"), standardCircleVerticalFovSpin_);
    standardCircleForm->addRow(QStringLiteral("采集重叠率"), standardCircleOverlapSpin_);
    standardCircleForm->addRow(QStringLiteral("测量窗口半尺寸"), standardCircleWindowHalfSizeSpin_);
    standardCircleForm->addRow(QStringLiteral("环向中值半径"), standardCircleMedianRadiusSpin_);
    standardCircleForm->addRow(QStringLiteral("滤波混合系数"), standardCircleFilterBlendSpin_);
    standardCircleLayout->addLayout(standardCircleForm);

    standardCircleRunButton_ = new QPushButton(QStringLiteral("运行标准圆检测"), standardCircleGroup);
    standardCircleRunButton_->setObjectName(QStringLiteral("successActionButton"));
    standardCircleRunButton_->setMinimumHeight(28);
    standardCircleLayout->addWidget(standardCircleRunButton_);

    standardCircleSectionContainer_ =
        makeCollapsibleSection(QStringLiteral("标准圆检测"), standardCircleGroup, false, registrationSection_);
    registrationLayout->addWidget(standardCircleSectionContainer_);
    registrationLayout->addStretch(1);

    reportSection_ = makeSectionPanel(QStringLiteral("reportConfigSection"), this);
    auto* reportLayout = new QVBoxLayout(reportSection_);
    reportLayout->setContentsMargins(0, 0, 0, 0);
    reportLayout->setSpacing(8);

    auto* outputGroup = new QGroupBox(QStringLiteral("结果输出"), reportSection_);
    auto* outputLayout = new QVBoxLayout(outputGroup);
    outputLayout->setContentsMargins(8, 10, 8, 8);
    outputLayout->setSpacing(6);

    runNameEdit_ = new QLineEdit(QStringLiteral("gui"), outputGroup);
    runNameEdit_->setPlaceholderText(QStringLiteral("例如 workpiece_gui"));

    saveDebugCheck_ = new QCheckBox(QStringLiteral("保存调试图"), outputGroup);
    saveDebugCheck_->setChecked(true);

    saveStepSummaryCsvCheck_ = new QCheckBox(QStringLiteral("拼接汇总 CSV"), outputGroup);
    saveStepSummaryCsvCheck_->setChecked(true);
    saveStepSummaryCsvCheck_->setToolTip(QStringLiteral("导出每一步的位移、RMSE、覆盖率和切向相关性汇总"));

    saveContourPointsCsvCheck_ = new QCheckBox(QStringLiteral("轮廓叠加点 CSV"), outputGroup);
    saveContourPointsCsvCheck_->setChecked(false);
    saveContourPointsCsvCheck_->setToolTip(
        QStringLiteral("导出 contour_points.csv 明细，以及可直接在 Origin 叠加各轮廓的 origin_contour_overlay_points.csv"));

    saveTangentStepCsvCheck_ = new QCheckBox(QStringLiteral("切向相关性分步 CSV"), outputGroup);
    saveTangentStepCsvCheck_->setChecked(false);
    saveTangentStepCsvCheck_->setToolTip(QStringLiteral("导出每一步的切向相关性与代表性误差指标"));

    saveNormalErrorProfileCsvCheck_ = new QCheckBox(QStringLiteral("法向误差剖面 CSV"), outputGroup);
    saveNormalErrorProfileCsvCheck_->setChecked(false);
    saveNormalErrorProfileCsvCheck_->setToolTip(QStringLiteral("导出每一步内点法向误差沿主轴坐标变化的数据"));

    saveTangentProfileCsvCheck_ = new QCheckBox(QStringLiteral("轮廓波动分析 CSV"), outputGroup);
    saveTangentProfileCsvCheck_->setChecked(false);
    saveTangentProfileCsvCheck_->setToolTip(
        QStringLiteral("导出每段重叠轮廓的参考波动、目标波动和点级切向/法向误差；包含 tangent_profile_compare.csv 明细与可直接在 Origin 画图的 origin_tangent_point_metrics.csv"));

    saveAlignmentCandidateDiagnosticsCsvCheck_ =
        new QCheckBox(QStringLiteral("候选诊断 CSV"), outputGroup);
    saveAlignmentCandidateDiagnosticsCsvCheck_->setChecked(true);
    saveAlignmentCandidateDiagnosticsCsvCheck_->setToolTip(
        QStringLiteral("导出每一步 Top-N 配准候选的 score、位移、角度、覆盖率和误差指标，便于定位错误谷值。"));

    saveStitchedContourProfileCsvCheck_ = new QCheckBox(QStringLiteral("整体轮廓剖面 CSV"), outputGroup);
    saveStitchedContourProfileCsvCheck_->setChecked(false);
    saveStitchedContourProfileCsvCheck_->setToolTip(QStringLiteral("导出拼接后整体轮廓的主轴坐标、次轴坐标、平滑基线和波动残差，可直接用于 Origin 画图"));

    auto* suffixForm = new QFormLayout();
    suffixForm->setContentsMargins(0, 0, 0, 0);
    suffixForm->setSpacing(6);
    suffixForm->addRow(QStringLiteral("结果后缀"), runNameEdit_);
    outputLayout->addLayout(suffixForm);

    outputLayout->addWidget(saveDebugCheck_);

    auto* basicCsvPanel = new QWidget(outputGroup);
    auto* basicCsvLayout = new QGridLayout(basicCsvPanel);
    basicCsvLayout->setContentsMargins(0, 0, 0, 0);
    basicCsvLayout->setHorizontalSpacing(6);
    basicCsvLayout->setVerticalSpacing(4);
    basicCsvLayout->addWidget(saveStepSummaryCsvCheck_, 0, 0);
    basicCsvLayout->addWidget(saveAlignmentCandidateDiagnosticsCsvCheck_, 1, 0);
    outputLayout->addWidget(makeCollapsibleSection(QStringLiteral("常用 CSV"), basicCsvPanel, true, outputGroup));

    auto* advancedCsvPanel = new QWidget(outputGroup);
    auto* advancedCsvLayout = new QGridLayout(advancedCsvPanel);
    advancedCsvLayout->setContentsMargins(0, 0, 0, 0);
    advancedCsvLayout->setHorizontalSpacing(6);
    advancedCsvLayout->setVerticalSpacing(4);
    advancedCsvLayout->addWidget(saveContourPointsCsvCheck_, 0, 0);
    advancedCsvLayout->addWidget(saveStitchedContourProfileCsvCheck_, 1, 0);
    advancedCsvLayout->addWidget(saveTangentStepCsvCheck_, 2, 0);
    advancedCsvLayout->addWidget(saveNormalErrorProfileCsvCheck_, 3, 0);
    advancedCsvLayout->addWidget(saveTangentProfileCsvCheck_, 4, 0);
    outputLayout->addWidget(makeCollapsibleSection(QStringLiteral("高级 CSV"), advancedCsvPanel, false, outputGroup));

    reportLayout->addWidget(makeCollapsibleSection(QStringLiteral("输出参数"), outputGroup, true, reportSection_));
    reportLayout->addStretch(1);

    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString startDir =
            inputDirEdit_->text().trimmed().isEmpty() ? defaultWorkpieceImageDir()
                                                      : inputDirEdit_->text().trimmed();
        const QString filePath = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("选择工件图像"),
            startDir,
            QStringLiteral("Image Files (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)"));
        if (!filePath.isEmpty()) {
            inputDirEdit_->setText(QDir::toNativeSeparators(QFileInfo(filePath).absolutePath()));
        }
    });

    connect(inputDirEdit_, &QLineEdit::textChanged, this, [this]() {
        updateDetectedCount();
        updateInputImagePreview();
        emit configChanged();
    });

    connect(cameraBrowseButton_, &QPushButton::clicked, this, [this]() {
        const QString dir =
            QFileDialog::getExistingDirectory(this, QStringLiteral("选择相机图像保存目录"), cameraOutputDirEdit_->text());
        if (!dir.isEmpty()) {
            cameraOutputDirEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });
    connect(refreshCameraButton_, &QPushButton::clicked, this, [this]() {
        refreshCameraDevices();
    });
    connect(cameraWriteStagePositionCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        updateCameraStepControlsEnabled();
        emit configChanged();
    });
    connect(cameraSessionNameEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(cameraStepIndexSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(cameraStageXSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(cameraStageYSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(cameraNoteEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(cameraSingleCaptureButton_, &QPushButton::clicked, this, [this]() {
        startSingleCameraCapture();
    });
    connect(cameraManualStepButton_, &QPushButton::clicked, this, [this]() {
        startManualStepCapture();
    });
    connect(cameraCancelButton_, &QPushButton::clicked, this, [this]() {
        cancelCameraCapture();
    });
    connect(cameraCaptureButton_, &QPushButton::clicked, this, [this]() {
        startCameraCapture();
    });
    connect(standardCircleRunButton_, &QPushButton::clicked, this, [this]() {
        setStandardCircleModeEnabled(true);
        revealStandardCircleConfig();
        emit standardCircleRunRequested();
    });

    connectChangeSignals();
    updateDetectedCount();
    updateInputImagePreview();
    updateCameraStepControlsEnabled();
}

RunConfigPanel::~RunConfigPanel()
{
    if (cameraCaptureWorker_) {
        cameraCaptureWorker_->requestStop();
    }
    if (cameraTaskThread_) {
        cameraTaskThread_->quit();
        cameraTaskThread_->wait();
    }
}

bool RunConfigPanel::buildRequest(pinjie::StitchRunRequest& request,
                                  QString& errorMessage,
                                  const bool materializeOutputPaths) const
{
    const QString inputDir = inputDirEdit_->text().trimmed();
    if (inputDir.isEmpty()) {
        errorMessage = QStringLiteral("请先选择图像目录。");
        return false;
    }

    QDir dir(inputDir);
    if (!dir.exists()) {
        errorMessage = QStringLiteral("图像目录不存在。");
        return false;
    }

    const QStringList imagePaths = scanImagePaths();
    if (imagePaths.isEmpty()) {
        errorMessage = QStringLiteral("目录中未找到支持的图像文件。");
        return false;
    }

    int useCount = imageLimitSpin_->value();
    if (useCount <= 0 || useCount > imagePaths.size()) {
        useCount = imagePaths.size();
    }

    request = {};
    const QString suffix =
        runNameEdit_->text().trimmed().isEmpty()
            ? QStringLiteral("workpiece_gui")
            : runNameEdit_->text().trimmed();
    const bool standardCircleMode = standardCircleModeEnabled();

    if (standardCircleMode) {
        const QString prefix =
            standardCirclePrefixEdit_ ? standardCirclePrefixEdit_->text().trimmed() : QStringLiteral("Pic_");
        QString extension =
            standardCircleExtensionEdit_ ? standardCircleExtensionEdit_->text().trimmed() : QStringLiteral(".bmp");
        if (prefix.isEmpty()) {
            errorMessage = QStringLiteral("标准圆检测模式需要有效的文件前缀。");
            return false;
        }
        if (extension.isEmpty()) {
            errorMessage = QStringLiteral("标准圆检测模式需要有效的文件后缀。");
            return false;
        }
        if (!extension.startsWith('.')) {
            extension.prepend('.');
        }

        const int startIndex =
            standardCircleStartIndexSpin_ ? standardCircleStartIndexSpin_->value() : 1;
        request.imagePaths.reserve(static_cast<std::size_t>(useCount));
        for (int i = 0; i < useCount; ++i) {
            const QString fileName = QStringLiteral("%1%2%3")
                                         .arg(prefix)
                                         .arg(startIndex + i)
                                         .arg(extension);
            const QString absolutePath = dir.absoluteFilePath(fileName);
            if (!QFileInfo::exists(absolutePath)) {
                errorMessage = QStringLiteral("标准圆序列文件不存在：%1")
                                   .arg(QDir::toNativeSeparators(absolutePath));
                return false;
            }
            request.imagePaths.push_back(QDir::fromNativeSeparators(absolutePath).toUtf8().toStdString());
        }

        request.standardCircleConfig.enabled = true;
        request.standardCircleConfig.startIndex = startIndex;
        request.standardCircleConfig.imagePrefix = prefix.toUtf8().toStdString();
        request.standardCircleConfig.imageExtension = extension.toUtf8().toStdString();
        request.standardCircleConfig.sphereDiameterMm =
            standardCircleDiameterSpin_ ? standardCircleDiameterSpin_->value() : 19.99995;
        request.standardCircleConfig.horizontalFieldOfViewMm =
            standardCircleHorizontalFovSpin_ ? standardCircleHorizontalFovSpin_->value() : 40.0;
        request.standardCircleConfig.verticalFieldOfViewMm =
            standardCircleVerticalFovSpin_ ? standardCircleVerticalFovSpin_->value() : 30.0;
        request.standardCircleConfig.overlapRatio =
            standardCircleOverlapSpin_ ? standardCircleOverlapSpin_->value() / 100.0 : 0.70;
        request.standardCircleConfig.windowHalfSizePx =
            standardCircleWindowHalfSizeSpin_ ? standardCircleWindowHalfSizeSpin_->value() : 60.0;
        request.standardCircleConfig.windowHalfAngleDeg = 7.0;
        request.standardCircleConfig.circularMedianFilterRadius =
            standardCircleMedianRadiusSpin_ ? standardCircleMedianRadiusSpin_->value() : 10;
        request.standardCircleConfig.circularFilterBlend =
            standardCircleFilterBlendSpin_ ? standardCircleFilterBlendSpin_->value() : 0.95;
    } else {
        request.imagePaths.reserve(static_cast<std::size_t>(useCount));
        for (int i = 0; i < useCount; ++i) {
            request.imagePaths.push_back(QDir::fromNativeSeparators(imagePaths.at(i)).toUtf8().toStdString());
        }
    }

    const double overlapRatio = overlapSpin_->value() / 100.0;

    request.edgeConfig.cannyLow = cannyLowSpin_->value();
    request.edgeConfig.cannyHigh = cannyHighSpin_->value();
    request.edgeConfig.subpixWindow = subpixWindowSpin_->value();
    request.edgeConfig.subpixSigma = subpixSigmaSpin_->value();
    request.edgeConfig.enablePointFiltering = pointFilterCheck_->isChecked();
    request.edgeConfig.filterConfidenceQuantile = filterConfidenceSpin_->value() / 100.0;
    request.edgeConfig.filterGradientQuantile = filterGradientSpin_->value() / 100.0;
    request.edgeConfig.filterLocalLinearWindowRadius = filterWindowRadiusSpin_->value();
    request.edgeConfig.filterHampelSigma = filterHampelSigmaSpin_->value();
    request.edgeConfig.filterHampelMinScale = filterHampelMinScaleSpin_->value();

    request.pipelineConfig.baseSearchRange = searchRangeSpin_->value();
    request.pipelineConfig.approxShiftRatio = std::max(0.0, 1.0 - overlapRatio);
    request.pipelineConfig.expectedOverlapRatio = overlapRatio;
    request.pipelineConfig.directionConstraint = currentDirection();
    request.pipelineConfig.rotationSearchMinDeg = -rotationRangeSpin_->value();
    request.pipelineConfig.rotationSearchMaxDeg = rotationRangeSpin_->value();
    request.pipelineConfig.rotationSearchStepDeg = rotationStepSpin_->value();
    request.pipelineConfig.tangentResidualCostWeight = tangentResidualWeightSpin_->value();
    request.pipelineConfig.tangentCorrelationCostWeight = tangentCorrelationWeightSpin_->value();
    request.pipelineConfig.generateDebugVisualization = saveDebugCheck_->isChecked();
    request.saveStepSummaryCsv = saveStepSummaryCsv();
    request.saveContourPointsCsv = saveContourPointsCsv();
    request.saveStitchedContourProfileCsv = saveStitchedContourProfileCsv();
    request.saveTangentStepCsv = saveTangentStepCsv();
    request.saveNormalErrorProfileCsv = saveNormalErrorProfileCsv();
    request.saveTangentProfileCsv = saveTangentProfileCsv();

    if (!materializeOutputPaths) {
        return true;
    }

    const std::string resultCategory = standardCircleMode ? "standard_sphere_loop" : "workpiece";
    const pinjie::StitchResultPathSet resultPaths =
        pinjie::buildDefaultStitchResultPaths(suffix.toUtf8().toStdString(), resultCategory, saveDebugCheck_->isChecked());
    if (!pinjie::ensureStitchResultDirectories(resultPaths)) {
        errorMessage = QStringLiteral("创建结果输出目录失败。");
        return false;
    }

    request.resultOutputDir = pinjie::genericUtf8String(resultPaths.runDir);
    if (standardCircleMode) {
        const auto runDir = resultPaths.runDir;
        request.panoramaOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_gbt57_p2d_window_overlay.png");
        request.csvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_gbt57_p2d_summary.csv");
        request.designComparisonOverlayOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_gbt57_p2d_edge_overlay.png");
        request.qualityReviewCsvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_gbt57_p2d_candidate_coverage.csv");
        request.contourPointsCsvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_gbt57_p2d_points.csv");
        request.alignmentCandidateDiagnosticsCsvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_circle_edge_cleanup.csv");
        request.designErrorProfileCsvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_dominant_dark_component_mask.csv");
        request.designErrorSummaryCsvOutputPath =
            pinjie::genericUtf8String(runDir / "standard_sphere_support_change_mask.csv");
    } else {
        request.panoramaOutputPath = pinjie::genericUtf8String(resultPaths.panoramaPath);
        request.csvOutputPath = pinjie::genericUtf8String(resultPaths.csvPath);
        request.designErrorProfileCsvOutputPath = pinjie::genericUtf8String(resultPaths.designErrorProfileCsvPath);
        request.designErrorSummaryCsvOutputPath = pinjie::genericUtf8String(resultPaths.designErrorSummaryCsvPath);
        request.design3dErrorCsvOutputPath = pinjie::genericUtf8String(resultPaths.design3dErrorCsvPath);
        request.designCompensationCsvOutputPath = pinjie::genericUtf8String(resultPaths.designCompensationCsvPath);
        request.designFeatureCompensationCsvOutputPath =
            pinjie::genericUtf8String(resultPaths.designFeatureCompensationCsvPath);
        request.designComparisonOverlayOutputPath =
            pinjie::genericUtf8String(resultPaths.designComparisonOverlayPath);
        request.designCompensationPlotOutputPath =
            pinjie::genericUtf8String(resultPaths.designCompensationPlotPath);
        request.qualityReviewCsvOutputPath = pinjie::genericUtf8String(resultPaths.qualityReviewCsvPath);
        request.contourPointsCsvOutputPath = pinjie::genericUtf8String(resultPaths.contourPointsCsvPath);
        request.originContourOverlayCsvOutputPath = pinjie::genericUtf8String(resultPaths.originContourOverlayCsvPath);
        request.stitchedContourProfileCsvOutputPath =
            pinjie::genericUtf8String(resultPaths.stitchedContourProfileCsvPath);
        request.tangentStepCsvOutputPath = pinjie::genericUtf8String(resultPaths.tangentStepCsvPath);
        request.normalErrorProfileCsvOutputPath = pinjie::genericUtf8String(resultPaths.normalErrorProfileCsvPath);
        request.tangentProfileCsvOutputPath = pinjie::genericUtf8String(resultPaths.tangentProfileCsvPath);
        request.originTangentPointMetricsCsvOutputPath =
            pinjie::genericUtf8String(resultPaths.originTangentPointMetricsCsvPath);
        if (saveAlignmentCandidateDiagnosticsCsv()) {
            request.alignmentCandidateDiagnosticsCsvOutputPath =
                pinjie::genericUtf8String(resultPaths.alignmentCandidateDiagnosticsCsvPath);
        }
        if (saveDebugCheck_->isChecked()) {
            request.debugImageOutputDir = pinjie::genericUtf8String(resultPaths.debugDir);
        }
    }

    return true;
}

void RunConfigPanel::setRunning(bool running)
{
    if (acquisitionSection_) {
        acquisitionSection_->setEnabled(!running);
    }
    if (processingSection_) {
        processingSection_->setEnabled(!running);
    }
    if (registrationSection_) {
        registrationSection_->setEnabled(!running);
    }
    if (reportSection_) {
        reportSection_->setEnabled(!running);
    }
}

pinjie::CaptureMode RunConfigPanel::currentCaptureMode() const
{
    return pinjie::CaptureMode::Workpiece;
}

QString RunConfigPanel::currentCaptureModeLabel() const
{
    return QStringLiteral("工件采集");
}

bool RunConfigPanel::saveStepSummaryCsv() const
{
    return saveStepSummaryCsvCheck_ && saveStepSummaryCsvCheck_->isChecked();
}

bool RunConfigPanel::saveContourPointsCsv() const
{
    return saveContourPointsCsvCheck_ && saveContourPointsCsvCheck_->isChecked();
}

bool RunConfigPanel::saveStitchedContourProfileCsv() const
{
    return saveStitchedContourProfileCsvCheck_ && saveStitchedContourProfileCsvCheck_->isChecked();
}

bool RunConfigPanel::saveTangentStepCsv() const
{
    return saveTangentStepCsvCheck_ && saveTangentStepCsvCheck_->isChecked();
}

bool RunConfigPanel::saveNormalErrorProfileCsv() const
{
    return saveNormalErrorProfileCsvCheck_ && saveNormalErrorProfileCsvCheck_->isChecked();
}

bool RunConfigPanel::saveTangentProfileCsv() const
{
    return saveTangentProfileCsvCheck_ && saveTangentProfileCsvCheck_->isChecked();
}

bool RunConfigPanel::saveAlignmentCandidateDiagnosticsCsv() const
{
    return saveAlignmentCandidateDiagnosticsCsvCheck_ &&
           saveAlignmentCandidateDiagnosticsCsvCheck_->isChecked();
}

bool RunConfigPanel::standardCircleModeEnabled() const
{
    return standardCircleCheck_ && standardCircleCheck_->isChecked();
}

void RunConfigPanel::setStandardCircleModeEnabled(const bool enabled)
{
    if (!standardCircleCheck_ || standardCircleCheck_->isChecked() == enabled) {
        return;
    }
    standardCircleCheck_->setChecked(enabled);
}

void RunConfigPanel::revealStandardCircleConfig()
{
    if (!standardCircleSectionContainer_) {
        return;
    }
    if (auto* toggleButton =
            standardCircleSectionContainer_->findChild<QToolButton*>(QStringLiteral("configSectionToggle"))) {
        toggleButton->setChecked(true);
    }
}

QWidget* RunConfigPanel::acquisitionSection() const
{
    return acquisitionSection_;
}

QWidget* RunConfigPanel::processingSection() const
{
    return processingSection_;
}

QWidget* RunConfigPanel::registrationSection() const
{
    return registrationSection_;
}

QWidget* RunConfigPanel::reportSection() const
{
    return reportSection_;
}

QStringList RunConfigPanel::scanImagePaths() const
{
    const QString inputDir = inputDirEdit_->text().trimmed();
    QDir dir(inputDir);
    if (!dir.exists()) {
        return {};
    }

    QStringList filters;
    filters << QStringLiteral("*.bmp")
            << QStringLiteral("*.png")
            << QStringLiteral("*.jpg")
            << QStringLiteral("*.jpeg")
            << QStringLiteral("*.tif")
            << QStringLiteral("*.tiff");

    QFileInfoList entries = dir.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::Name);

    QCollator collator;
    collator.setNumericMode(true);
    std::sort(entries.begin(), entries.end(), [&collator](const QFileInfo& lhs, const QFileInfo& rhs) {
        return collator.compare(lhs.fileName(), rhs.fileName()) < 0;
    });

    QStringList paths;
    for (const QFileInfo& info : entries) {
        paths.push_back(QDir::toNativeSeparators(info.absoluteFilePath()));
    }
    return paths;
}

void RunConfigPanel::updateDetectedCount()
{
    const QStringList imagePaths = scanImagePaths();
    detectedCountLabel_->setText(QStringLiteral("已检测 %1 张图像").arg(imagePaths.size()));
}

void RunConfigPanel::updateInputImagePreview()
{
    const QStringList imagePaths = scanImagePaths();
    if (imagePaths.isEmpty()) {
        updateCameraPreview(QImage(), QStringLiteral("等待导入工件图像或相机采集预览。"));
        return;
    }

    const QString firstImagePath = imagePaths.first();
    const QImage image(firstImagePath);
    const QFileInfo imageInfo(firstImagePath);
    if (image.isNull()) {
        updateCameraPreview(
            QImage(),
            QStringLiteral("已检测 %1 张工件图像，但首张图像读取失败：\n%2")
                .arg(imagePaths.size())
                .arg(QDir::toNativeSeparators(firstImagePath)));
        return;
    }

    const QString summary =
        QStringLiteral("工件图像导入预览\n来源：本地图像目录\n文件：%1\n图像尺寸：%2 x %3 px\n目录图像数：%4 张")
            .arg(imageInfo.fileName())
            .arg(image.width())
            .arg(image.height())
            .arg(imagePaths.size());
    updateCameraPreview(image, summary);
}

bool RunConfigPanel::buildCameraRequest(pinjie::CameraSequenceRequest& request,
                                        QString& errorMessage,
                                        const pinjie::CameraCaptureMode mode) const
{
#ifdef PINJIE_ENABLE_GALAXY_CAMERA
    const QString outputDir = cameraOutputDirEdit_ ? cameraOutputDirEdit_->text().trimmed() : QString();
    if (outputDir.isEmpty()) {
        errorMessage = QStringLiteral("请先选择相机图像保存目录。");
        return false;
    }

    const QString serialNumber = cameraDeviceCombo_ ? cameraDeviceCombo_->currentData().toString().trimmed() : QString();
    if (serialNumber.isEmpty()) {
        errorMessage = QStringLiteral("请先刷新并选择相机设备。");
        return false;
    }

    request = {};
    request.mode = mode;
    request.sessionName = (cameraSessionNameEdit_ && !cameraSessionNameEdit_->text().trimmed().isEmpty()
                               ? cameraSessionNameEdit_->text().trimmed()
                               : QStringLiteral("camera_capture"))
                              .toStdString();
    request.serialNumber = serialNumber.toStdString();
    request.outputDir = QDir::fromNativeSeparators(outputDir).toStdString();
    request.frameCount = mode == pinjie::CameraCaptureMode::Sequence
                             ? (cameraFrameCountSpin_ ? cameraFrameCountSpin_->value() : 1)
                             : 1;
    request.startIndex = mode == pinjie::CameraCaptureMode::ManualStepScan
                             ? (cameraStepIndexSpin_ ? cameraStepIndexSpin_->value() : 1)
                             : nextCameraStartIndex();
    request.timeoutMs = 3000;
    request.intervalMs = mode == pinjie::CameraCaptureMode::Sequence
                             ? (cameraIntervalSpin_ ? cameraIntervalSpin_->value() : 0)
                             : 0;
    request.stepIndex = cameraStepIndexSpin_ ? static_cast<std::size_t>(cameraStepIndexSpin_->value()) : 0;
    request.hasStagePosition = cameraWriteStagePositionCheck_ && cameraWriteStagePositionCheck_->isChecked();
    request.stageX = cameraStageXSpin_ ? cameraStageXSpin_->value() : 0.0;
    request.stageY = cameraStageYSpin_ ? cameraStageYSpin_->value() : 0.0;
    request.note = cameraNoteEdit_ ? cameraNoteEdit_->text().trimmed().toStdString() : std::string();
    request.config.setExposureTime = cameraSetExposureCheck_ && cameraSetExposureCheck_->isChecked();
    request.config.exposureTimeUs = cameraExposureSpin_ ? cameraExposureSpin_->value() : 0.0;
    request.config.setGain = cameraSetGainCheck_ && cameraSetGainCheck_->isChecked();
    request.config.gain = cameraGainSpin_ ? cameraGainSpin_->value() : 0.0;
    request.config.triggerEnabled = cameraSoftwareTriggerCheck_ && cameraSoftwareTriggerCheck_->isChecked();
    request.config.triggerSource = "Software";
    return true;
#else
    Q_UNUSED(request);
    Q_UNUSED(mode);
    errorMessage = QStringLiteral("当前构建未启用 Galaxy 相机 SDK。");
    return false;
#endif
}

int RunConfigPanel::nextCameraStartIndex() const
{
    const QString outputDir = cameraOutputDirEdit_ ? cameraOutputDirEdit_->text().trimmed() : QString();
    QDir dir(outputDir);
    if (!dir.exists()) {
        return 1;
    }

    const QRegularExpression pattern(QStringLiteral("^Pic_(\\d+)(?:\\.[^.]+)?$"));
    const QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::Readable, QDir::Name);

    int maxIndex = 0;
    for (const QFileInfo& entry : entries) {
        const QRegularExpressionMatch match = pattern.match(entry.fileName());
        if (match.hasMatch()) {
            maxIndex = std::max(maxIndex, match.captured(1).toInt());
        }
    }

    return maxIndex + 1;
}

QString RunConfigPanel::defaultCameraOutputDir() const
{
#ifdef PINJIE_PROJECT_ROOT
    const QString root = QString::fromUtf8(PINJIE_PROJECT_ROOT);
#else
    const QString root = QDir::currentPath();
#endif
    return QDir::toNativeSeparators(QDir(root).filePath(
        QStringLiteral("result/camera/capture_%1").arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss")))));
}

void RunConfigPanel::setCameraControlsEnabled(const bool enabled)
{
    if (cameraDeviceCombo_) {
        cameraDeviceCombo_->setEnabled(enabled);
    }
    if (refreshCameraButton_) {
        refreshCameraButton_->setEnabled(enabled);
    }
    if (cameraOutputDirEdit_) {
        cameraOutputDirEdit_->setEnabled(enabled);
    }
    if (cameraBrowseButton_) {
        cameraBrowseButton_->setEnabled(enabled);
    }
    if (cameraSessionNameEdit_) {
        cameraSessionNameEdit_->setEnabled(enabled);
    }
    if (cameraFrameCountSpin_) {
        cameraFrameCountSpin_->setEnabled(enabled);
    }
    if (cameraIntervalSpin_) {
        cameraIntervalSpin_->setEnabled(enabled);
    }
    if (cameraSetExposureCheck_) {
        cameraSetExposureCheck_->setEnabled(enabled);
    }
    if (cameraExposureSpin_) {
        cameraExposureSpin_->setEnabled(enabled);
    }
    if (cameraSetGainCheck_) {
        cameraSetGainCheck_->setEnabled(enabled);
    }
    if (cameraGainSpin_) {
        cameraGainSpin_->setEnabled(enabled);
    }
    if (cameraSoftwareTriggerCheck_) {
        cameraSoftwareTriggerCheck_->setEnabled(enabled);
    }
    if (cameraWriteStagePositionCheck_) {
        cameraWriteStagePositionCheck_->setEnabled(enabled);
    }
    if (cameraStepIndexSpin_) {
        cameraStepIndexSpin_->setEnabled(enabled);
    }
    if (cameraStageXSpin_) {
        cameraStageXSpin_->setEnabled(enabled);
    }
    if (cameraStageYSpin_) {
        cameraStageYSpin_->setEnabled(enabled);
    }
    if (cameraNoteEdit_) {
        cameraNoteEdit_->setEnabled(enabled);
    }
    if (cameraCaptureButton_) {
        cameraCaptureButton_->setEnabled(enabled);
    }
    if (cameraSingleCaptureButton_) {
        cameraSingleCaptureButton_->setEnabled(enabled);
    }
    if (cameraManualStepButton_) {
        cameraManualStepButton_->setEnabled(enabled);
    }
    if (cameraCancelButton_) {
        cameraCancelButton_->setEnabled(!enabled && cameraCaptureWorker_ != nullptr);
    }
    updateCameraStepControlsEnabled();
}

void RunConfigPanel::updateCameraStepControlsEnabled()
{
    const bool allowStagePosition = cameraWriteStagePositionCheck_ &&
                                    cameraWriteStagePositionCheck_->isEnabled() &&
                                    cameraWriteStagePositionCheck_->isChecked();
    if (cameraStageXSpin_) {
        cameraStageXSpin_->setEnabled(allowStagePosition);
    }
    if (cameraStageYSpin_) {
        cameraStageYSpin_->setEnabled(allowStagePosition);
    }
}

void RunConfigPanel::updateCameraPreview(const QImage& image, const QString& summary)
{
    if (cameraPreviewViewer_) {
        if (image.isNull()) {
            cameraPreviewViewer_->clearImage();
        } else {
            cameraPreviewViewer_->setImage(image);
        }
    }
    if (cameraPreviewInfoLabel_) {
        cameraPreviewInfoLabel_->setText(summary.isEmpty() ? QStringLiteral("等待导入图像或相机采集预览。") : summary);
    }
    emit cameraPreviewUpdated(image, summary);
}

void RunConfigPanel::cancelCameraCapture()
{
    if (!cameraCaptureWorker_) {
        return;
    }

    cameraCaptureWorker_->requestStop();
    if (cameraCancelButton_) {
        cameraCancelButton_->setEnabled(false);
    }
    if (cameraStatusLabel_) {
        cameraStatusLabel_->setText(QStringLiteral("正在取消采集，请等待当前帧完成..."));
    }
}

void RunConfigPanel::runCameraCaptureTask(const pinjie::CameraSequenceRequest& request, const bool advanceManualStepIndex)
{
#ifdef PINJIE_ENABLE_GALAXY_CAMERA
    QString capturePendingMessage;
    switch (request.mode) {
    case pinjie::CameraCaptureMode::Single:
        capturePendingMessage = QStringLiteral("正在采集单张图像...");
        break;
    case pinjie::CameraCaptureMode::ManualStepScan:
        capturePendingMessage = QStringLiteral("正在采集当前步图像...");
        break;
    case pinjie::CameraCaptureMode::Sequence:
    default:
        capturePendingMessage = QStringLiteral("正在采集相机图像...");
        break;
    }

    cameraStatusLabel_->setText(capturePendingMessage);
    updateCameraPreview(QImage(), QStringLiteral("等待新的采集结果..."));
    setCameraControlsEnabled(false);

    cameraTaskThread_ = new QThread(this);
    cameraCaptureWorker_ = new CameraCaptureWorker(request);
    cameraCaptureWorker_->moveToThread(cameraTaskThread_);
    if (cameraCancelButton_) {
        cameraCancelButton_->setEnabled(true);
    }

    connect(cameraTaskThread_, &QThread::started, cameraCaptureWorker_, &CameraCaptureWorker::run);
    connect(cameraCaptureWorker_, &CameraCaptureWorker::logMessage, this, [this](const QString& message) {
        if (cameraStatusLabel_ && !message.isEmpty()) {
            cameraStatusLabel_->setText(message);
        }
    });
    connect(cameraCaptureWorker_, &CameraCaptureWorker::progressChanged, this, [this](int current, int total) {
        if (cameraStatusLabel_ && total > 0) {
            cameraStatusLabel_->setText(QStringLiteral("采集中 %1/%2").arg(current).arg(total));
        }
    });
    connect(cameraCaptureWorker_,
            &CameraCaptureWorker::previewReady,
            this,
            [this](int current, int total, const QImage& image, const QString& summary) {
                updateCameraPreview(
                    image,
                    total > 0 ? summary + QStringLiteral("\n进度：%1/%2").arg(current).arg(total) : summary);
            });
    connect(cameraCaptureWorker_,
            &CameraCaptureWorker::runFinished,
            this,
            [this, advanceManualStepIndex, mode = request.mode](bool ok,
                                                                bool cancelled,
                                                                const QString& message,
                                                                int savedCount,
                                                                const QString& manifestPath,
                                                                const QString& outputDir) {
                if (ok || cancelled) {
                    inputDirEdit_->setText(QDir::toNativeSeparators(outputDir));
                    updateDetectedCount();
                    emit configChanged();
                }

                if (ok && advanceManualStepIndex && cameraStepIndexSpin_) {
                    cameraStepIndexSpin_->setValue(cameraStepIndexSpin_->value() + 1);
                }

                QString statusMessage;
                if (cancelled) {
                    statusMessage = QStringLiteral("采集已取消，已保存 %1 张图像").arg(savedCount);
                } else if (ok) {
                    switch (mode) {
                    case pinjie::CameraCaptureMode::Single:
                        statusMessage = QStringLiteral("单张采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    case pinjie::CameraCaptureMode::ManualStepScan:
                        statusMessage = QStringLiteral("当前步采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    case pinjie::CameraCaptureMode::Sequence:
                    default:
                        statusMessage = QStringLiteral("连续采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    }
                } else {
                    statusMessage = QStringLiteral("采集失败：%1").arg(message);
                }

                if ((ok || cancelled) && !manifestPath.isEmpty()) {
                    statusMessage += QStringLiteral("。Manifest：%1").arg(QDir::toNativeSeparators(manifestPath));
                }
                cameraStatusLabel_->setText(statusMessage);
            });
    connect(cameraCaptureWorker_, &CameraCaptureWorker::runFinished, cameraTaskThread_, &QThread::quit);
    connect(cameraTaskThread_, &QThread::finished, cameraCaptureWorker_, &QObject::deleteLater);
    connect(cameraTaskThread_, &QThread::finished, cameraTaskThread_, &QObject::deleteLater);
    connect(cameraTaskThread_, &QThread::finished, this, [this]() { finishCameraTask(); });
    cameraTaskThread_->start();
    return;

    const QString outputDir = QDir::toNativeSeparators(QString::fromStdString(request.outputDir));

    QString pendingMessage;
    switch (request.mode) {
    case pinjie::CameraCaptureMode::Single:
        pendingMessage = QStringLiteral("正在采集单张图像...");
        break;
    case pinjie::CameraCaptureMode::ManualStepScan:
        pendingMessage = QStringLiteral("正在采集当前步图像...");
        break;
    case pinjie::CameraCaptureMode::Sequence:
    default:
        pendingMessage = QStringLiteral("正在采集相机图像...");
        break;
    }

    cameraStatusLabel_->setText(pendingMessage);
    setCameraControlsEnabled(false);

    cameraTaskThread_ = QThread::create([this, request, outputDir, advanceManualStepIndex]() {
        pinjie::CameraCaptureService service;
        pinjie::CameraSequenceResult result;

        switch (request.mode) {
        case pinjie::CameraCaptureMode::Single:
            result = service.captureSingle(request);
            break;
        case pinjie::CameraCaptureMode::ManualStepScan:
            result = service.captureManualStep(request);
            break;
        case pinjie::CameraCaptureMode::Sequence:
        default:
            result = service.captureSequence(request);
            break;
        }

        const int savedCount = static_cast<int>(result.frames.size());
        const QString message = QString::fromStdString(result.message);
        const QString manifestPath = QString::fromStdString(result.manifestPath);
        const QString lastFileName =
            savedCount > 0 ? QString::fromStdString(result.frames.back().fileName) : QString();

        QMetaObject::invokeMethod(
            this,
            [this,
             ok = result.ok,
             savedCount,
             message,
             outputDir,
             manifestPath,
             lastFileName,
             advanceManualStepIndex,
             mode = request.mode]() {
                if (ok) {
                    inputDirEdit_->setText(outputDir);
                    updateDetectedCount();
                    emit configChanged();

                    if (advanceManualStepIndex && cameraStepIndexSpin_) {
                        cameraStepIndexSpin_->setValue(cameraStepIndexSpin_->value() + 1);
                    }

                    QString successMessage;
                    switch (mode) {
                    case pinjie::CameraCaptureMode::Single:
                        successMessage = QStringLiteral("单张采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    case pinjie::CameraCaptureMode::ManualStepScan:
                        successMessage = QStringLiteral("当前步采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    case pinjie::CameraCaptureMode::Sequence:
                    default:
                        successMessage = QStringLiteral("采集完成，已保存 %1 张图像").arg(savedCount);
                        break;
                    }

                    if (!lastFileName.isEmpty()) {
                        successMessage += QStringLiteral("，最后文件：%1").arg(lastFileName);
                    }
                    if (!manifestPath.isEmpty()) {
                        successMessage += QStringLiteral("。Manifest：%1")
                                              .arg(QDir::toNativeSeparators(manifestPath));
                    }
                    cameraStatusLabel_->setText(successMessage);
                } else {
                    cameraStatusLabel_->setText(QStringLiteral("采集失败：%1").arg(message));
                }
            },
            Qt::QueuedConnection);
    });
    connect(cameraTaskThread_, &QThread::finished, this, &RunConfigPanel::finishCameraTask);
    cameraTaskThread_->start();
#else
    Q_UNUSED(request);
    Q_UNUSED(advanceManualStepIndex);
    cameraStatusLabel_->setText(QStringLiteral("当前构建未启用 Galaxy 相机 SDK。"));
#endif
}

void RunConfigPanel::finishCameraTask()
{
    cameraCaptureWorker_ = nullptr;
    if (cameraTaskThread_) {
        cameraTaskThread_->deleteLater();
        cameraTaskThread_ = nullptr;
    }
    if (cameraCancelButton_) {
        cameraCancelButton_->setEnabled(false);
    }
    setCameraControlsEnabled(true);
}

void RunConfigPanel::refreshCameraDevices()
{
    if (cameraTaskThread_) {
        return;
    }

#ifdef PINJIE_ENABLE_GALAXY_CAMERA
    cameraStatusLabel_->setText(QStringLiteral("正在刷新相机设备..."));
    setCameraControlsEnabled(false);

    cameraTaskThread_ = QThread::create([this]() {
        bool ok = false;
        QString message;
        std::vector<pinjie::CameraDeviceInfo> devices;

        try {
            pinjie::GalaxyCameraDevice camera;
            devices = camera.enumerateDevices(1000);
            ok = true;
            message = QStringLiteral("发现 %1 台相机").arg(static_cast<int>(devices.size()));
        } catch (const std::exception& error) {
            message = QString::fromLocal8Bit(error.what());
        }

        QMetaObject::invokeMethod(
            this,
            [this, ok, message, devices]() {
                cameraDeviceCombo_->clear();
                if (ok) {
                    for (const pinjie::CameraDeviceInfo& device : devices) {
                        const QString serial = QString::fromStdString(device.serialNumber);
                        QString label = QString::fromStdString(device.modelName);
                        if (!serial.isEmpty()) {
                            label += QStringLiteral(" SN=") + serial;
                        }
                        if (!device.ipAddress.empty()) {
                            label += QStringLiteral(" IP=") + QString::fromStdString(device.ipAddress);
                        }
                        if (!device.readableAndWritable) {
                            label += QStringLiteral(" [访问受限]");
                        }
                        cameraDeviceCombo_->addItem(label, serial);
                    }
                    if (devices.empty()) {
                        cameraDeviceCombo_->addItem(QStringLiteral("未发现相机"), QString());
                    }
                } else {
                    cameraDeviceCombo_->addItem(QStringLiteral("刷新失败"), QString());
                }
                cameraStatusLabel_->setText(message);
            },
            Qt::QueuedConnection);
    });
    connect(cameraTaskThread_, &QThread::finished, this, &RunConfigPanel::finishCameraTask);
    cameraTaskThread_->start();
#else
    cameraStatusLabel_->setText(QStringLiteral("当前构建未启用 Galaxy 相机 SDK。"));
#endif
}

void RunConfigPanel::startSingleCameraCapture()
{
    if (cameraTaskThread_) {
        return;
    }

    pinjie::CameraSequenceRequest request;
    QString errorMessage;
    if (!buildCameraRequest(request, errorMessage, pinjie::CameraCaptureMode::Single)) {
        cameraStatusLabel_->setText(errorMessage);
        return;
    }

    runCameraCaptureTask(request, false);
}

void RunConfigPanel::startManualStepCapture()
{
    if (cameraTaskThread_) {
        return;
    }

    pinjie::CameraSequenceRequest request;
    QString errorMessage;
    if (!buildCameraRequest(request, errorMessage, pinjie::CameraCaptureMode::ManualStepScan)) {
        cameraStatusLabel_->setText(errorMessage);
        return;
    }

    runCameraCaptureTask(request, true);
}

void RunConfigPanel::startCameraCapture()
{
    if (cameraTaskThread_) {
        return;
    }

    pinjie::CameraSequenceRequest captureRequest;
    QString errorMessage;
    if (!buildCameraRequest(captureRequest, errorMessage, pinjie::CameraCaptureMode::Sequence)) {
        cameraStatusLabel_->setText(errorMessage);
        return;
    }

    runCameraCaptureTask(captureRequest, false);
    return;

#ifdef PINJIE_ENABLE_GALAXY_CAMERA
    const QString outputDir = cameraOutputDirEdit_->text().trimmed();
    if (outputDir.isEmpty()) {
        cameraStatusLabel_->setText(QStringLiteral("请先选择相机图像保存目录。"));
        return;
    }

    pinjie::CameraSequenceRequest request;
    request.serialNumber = cameraDeviceCombo_->currentData().toString().toStdString();
    request.outputDir = QDir::fromNativeSeparators(outputDir).toStdString();
    request.frameCount = cameraFrameCountSpin_->value();
    request.intervalMs = cameraIntervalSpin_->value();
    request.timeoutMs = 3000;
    request.config.setExposureTime = cameraSetExposureCheck_->isChecked();
    request.config.exposureTimeUs = cameraExposureSpin_->value();
    request.config.setGain = cameraSetGainCheck_->isChecked();
    request.config.gain = cameraGainSpin_->value();
    request.config.triggerEnabled = cameraSoftwareTriggerCheck_->isChecked();
    request.config.triggerSource = "Software";

    cameraStatusLabel_->setText(QStringLiteral("正在采集相机图像..."));
    setCameraControlsEnabled(false);

    cameraTaskThread_ = QThread::create([this, request, outputDir]() {
        pinjie::CameraCaptureService service;
        const pinjie::CameraSequenceResult result = service.captureSequence(request);
        const int savedCount = static_cast<int>(result.frames.size());
        const QString message = QString::fromStdString(result.message);

        QMetaObject::invokeMethod(
            this,
            [this, ok = result.ok, savedCount, message, outputDir]() {
                if (ok) {
                    inputDirEdit_->setText(QDir::toNativeSeparators(outputDir));
                    updateDetectedCount();
                    emit configChanged();
                    cameraStatusLabel_->setText(QStringLiteral("采集完成，已保存 %1 张图像。").arg(savedCount));
                } else {
                    cameraStatusLabel_->setText(QStringLiteral("采集失败：%1").arg(message));
                }
            },
            Qt::QueuedConnection);
    });
    connect(cameraTaskThread_, &QThread::finished, this, &RunConfigPanel::finishCameraTask);
    cameraTaskThread_->start();
#else
    cameraStatusLabel_->setText(QStringLiteral("当前构建未启用 Galaxy 相机 SDK。"));
#endif
}

pinjie::MotionPriorDirection RunConfigPanel::currentDirection() const
{
    return static_cast<pinjie::MotionPriorDirection>(directionCombo_->currentData().toInt());
}

void RunConfigPanel::connectChangeSignals()
{
    connect(imageLimitSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &RunConfigPanel::configChanged);

    connect(cannyLowSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(cannyHighSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(subpixWindowSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(subpixSigmaSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(pointFilterCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(filterConfidenceSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(filterGradientSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(filterWindowRadiusSpin_,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(filterHampelSigmaSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(filterHampelMinScaleSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);

    connect(overlapSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(directionCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        emit configChanged();
    });
    connect(searchRangeSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &RunConfigPanel::configChanged);
    connect(rotationRangeSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(rotationStepSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(tangentResidualWeightSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(tangentCorrelationWeightSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(standardCirclePrefixEdit_, &QLineEdit::textChanged, this, [this](const QString&) {
        emit configChanged();
    });
    connect(standardCircleExtensionEdit_, &QLineEdit::textChanged, this, [this](const QString&) {
        emit configChanged();
    });
    connect(standardCircleStartIndexSpin_,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleDiameterSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleHorizontalFovSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleVerticalFovSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleOverlapSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleWindowHalfSizeSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleMedianRadiusSpin_,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);
    connect(standardCircleFilterBlendSpin_,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &RunConfigPanel::configChanged);

    connect(runNameEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(saveDebugCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveStepSummaryCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveContourPointsCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveStitchedContourProfileCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveTangentStepCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveNormalErrorProfileCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveTangentProfileCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
    connect(saveAlignmentCandidateDiagnosticsCsvCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) {
        emit configChanged();
    });
}

} // namespace pinjie::gui
