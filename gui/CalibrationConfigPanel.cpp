#include "gui/CalibrationConfigPanel.h"

#include <QCollator>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>

namespace pinjie::gui {

namespace {

QWidget* makeCollapsibleSection(const QString& title, QWidget* content, bool expanded, QWidget* parent)
{
    auto* container = new QWidget(parent);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

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

CalibrationConfigPanel::CalibrationConfigPanel(QWidget* parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("calibrationConfigSection"));

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(12);

    auto* inputGroup = new QGroupBox(QStringLiteral("标定输入"), this);
    auto* inputLayout = new QVBoxLayout(inputGroup);
    auto* inputForm = new QFormLayout();

    sessionNameEdit_ = new QLineEdit(QStringLiteral("camera_calibration"), inputGroup);
    sessionNameEdit_->setPlaceholderText(QStringLiteral("例如 telecentric_camera"));
    inputForm->addRow(QStringLiteral("标定会话"), sessionNameEdit_);
    inputLayout->addLayout(inputForm);

    imageDirEdit_ = new QLineEdit(inputGroup);
    imageDirEdit_->setPlaceholderText(QStringLiteral("选择标定图像目录"));
    imageDirEdit_->setToolTip(QStringLiteral("支持 bmp、png、jpg、jpeg、tif、tiff 格式"));
    auto* imageDirRow = new QHBoxLayout();
    auto* browseImageDirButton = new QPushButton(QStringLiteral("浏览"), inputGroup);
    imageDirRow->addWidget(imageDirEdit_, 1);
    imageDirRow->addWidget(browseImageDirButton);
    inputLayout->addLayout(imageDirRow);

    detectedCountLabel_ = new QLabel(QStringLiteral("已检测 0 张标定图像"), inputGroup);
    inputForm = new QFormLayout();
    inputForm->addRow(QStringLiteral("图像统计"), detectedCountLabel_);
    inputLayout->addLayout(inputForm);

    cacheDirEdit_ = new QLineEdit(inputGroup);
    cacheDirEdit_->setPlaceholderText(QStringLiteral("可选：已有标定缓存目录"));
    auto* cacheDirRow = new QHBoxLayout();
    auto* browseCacheDirButton = new QPushButton(QStringLiteral("浏览"), inputGroup);
    cacheDirRow->addWidget(cacheDirEdit_, 1);
    cacheDirRow->addWidget(browseCacheDirButton);
    inputLayout->addLayout(cacheDirRow);

    outputDirEdit_ = new QLineEdit(inputGroup);
    outputDirEdit_->setPlaceholderText(QStringLiteral("可选：标定结果输出目录"));
    auto* outputDirRow = new QHBoxLayout();
    auto* browseOutputDirButton = new QPushButton(QStringLiteral("浏览"), inputGroup);
    outputDirRow->addWidget(outputDirEdit_, 1);
    outputDirRow->addWidget(browseOutputDirButton);
    inputLayout->addLayout(outputDirRow);

    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("输入参数"), inputGroup, true, this));

    auto* boardGroup = new QGroupBox(QStringLiteral("标定板参数"), this);
    auto* boardForm = new QFormLayout(boardGroup);

    rowsSpin_ = new QSpinBox(boardGroup);
    rowsSpin_->setRange(2, 99);
    rowsSpin_->setValue(7);

    colsSpin_ = new QSpinBox(boardGroup);
    colsSpin_->setRange(2, 99);
    colsSpin_->setValue(7);

    pitchSpin_ = new QDoubleSpinBox(boardGroup);
    pitchSpin_->setRange(0.001, 1000.0);
    pitchSpin_->setDecimals(4);
    pitchSpin_->setValue(2.5);
    pitchSpin_->setSuffix(QStringLiteral(" mm"));

    roiRadiusSpin_ = new QSpinBox(boardGroup);
    roiRadiusSpin_->setRange(5, 2000);
    roiRadiusSpin_->setValue(90);
    roiRadiusSpin_->setSuffix(QStringLiteral(" px"));

    minValidImagesSpin_ = new QSpinBox(boardGroup);
    minValidImagesSpin_->setRange(1, 999);
    minValidImagesSpin_->setValue(3);

    boardForm->addRow(QStringLiteral("圆点阵行数"), rowsSpin_);
    boardForm->addRow(QStringLiteral("圆点阵列数"), colsSpin_);
    boardForm->addRow(QStringLiteral("圆点间距"), pitchSpin_);
    boardForm->addRow(QStringLiteral("ROI 半径"), roiRadiusSpin_);
    boardForm->addRow(QStringLiteral("最少有效图像"), minValidImagesSpin_);

    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("几何参数"), boardGroup, true, this));

    auto* compensationGroup = new QGroupBox(QStringLiteral("标定补偿参数"), this);
    auto* compensationForm = new QFormLayout(compensationGroup);

    enableImageResidualCompensationCheck_ = new QCheckBox(QStringLiteral("启用系统残差补偿"), compensationGroup);
    enableImageResidualCompensationCheck_->setChecked(true);

    imageResidualPriorSigmaSpin_ = new QDoubleSpinBox(compensationGroup);
    imageResidualPriorSigmaSpin_->setRange(0.0001, 1.0);
    imageResidualPriorSigmaSpin_->setDecimals(4);
    imageResidualPriorSigmaSpin_->setSingleStep(0.001);
    imageResidualPriorSigmaSpin_->setValue(0.02);
    imageResidualPriorSigmaSpin_->setSuffix(QStringLiteral(" px"));

    imageResidualMaxCoeffSpin_ = new QDoubleSpinBox(compensationGroup);
    imageResidualMaxCoeffSpin_->setRange(0.0001, 5.0);
    imageResidualMaxCoeffSpin_->setDecimals(4);
    imageResidualMaxCoeffSpin_->setSingleStep(0.01);
    imageResidualMaxCoeffSpin_->setValue(0.2);
    imageResidualMaxCoeffSpin_->setSuffix(QStringLiteral(" px"));

    enableBoardWarpCompensationCheck_ = new QCheckBox(QStringLiteral("启用全局板形变补偿"), compensationGroup);
    enableBoardWarpCompensationCheck_->setChecked(true);

    boardWarpPriorSigmaSpin_ = new QDoubleSpinBox(compensationGroup);
    boardWarpPriorSigmaSpin_->setRange(0.0001, 1.0);
    boardWarpPriorSigmaSpin_->setDecimals(4);
    boardWarpPriorSigmaSpin_->setSingleStep(0.001);
    boardWarpPriorSigmaSpin_->setValue(0.02);
    boardWarpPriorSigmaSpin_->setSuffix(QStringLiteral(" mm"));

    boardWarpMaxOffsetSpin_ = new QDoubleSpinBox(compensationGroup);
    boardWarpMaxOffsetSpin_->setRange(0.0001, 5.0);
    boardWarpMaxOffsetSpin_->setDecimals(4);
    boardWarpMaxOffsetSpin_->setSingleStep(0.005);
    boardWarpMaxOffsetSpin_->setValue(0.05);
    boardWarpMaxOffsetSpin_->setSuffix(QStringLiteral(" mm"));

    enableBoardPointCompensationCheck_ = new QCheckBox(QStringLiteral("启用逐点残差补偿"), compensationGroup);
    enableBoardPointCompensationCheck_->setChecked(true);

    boardPointPriorSigmaSpin_ = new QDoubleSpinBox(compensationGroup);
    boardPointPriorSigmaSpin_->setRange(0.0001, 1.0);
    boardPointPriorSigmaSpin_->setDecimals(4);
    boardPointPriorSigmaSpin_->setSingleStep(0.001);
    boardPointPriorSigmaSpin_->setValue(0.005);
    boardPointPriorSigmaSpin_->setSuffix(QStringLiteral(" mm"));

    boardPointSmoothSigmaSpin_ = new QDoubleSpinBox(compensationGroup);
    boardPointSmoothSigmaSpin_->setRange(0.0001, 1.0);
    boardPointSmoothSigmaSpin_->setDecimals(4);
    boardPointSmoothSigmaSpin_->setSingleStep(0.001);
    boardPointSmoothSigmaSpin_->setValue(0.003);
    boardPointSmoothSigmaSpin_->setSuffix(QStringLiteral(" mm"));

    boardPointMaxOffsetSpin_ = new QDoubleSpinBox(compensationGroup);
    boardPointMaxOffsetSpin_->setRange(0.0001, 5.0);
    boardPointMaxOffsetSpin_->setDecimals(4);
    boardPointMaxOffsetSpin_->setSingleStep(0.005);
    boardPointMaxOffsetSpin_->setValue(0.03);
    boardPointMaxOffsetSpin_->setSuffix(QStringLiteral(" mm"));

    compensationForm->addRow(QStringLiteral("系统残差"), enableImageResidualCompensationCheck_);
    compensationForm->addRow(QStringLiteral("残差先验 sigma"), imageResidualPriorSigmaSpin_);
    compensationForm->addRow(QStringLiteral("残差系数上限"), imageResidualMaxCoeffSpin_);
    compensationForm->addRow(QStringLiteral("全局形变"), enableBoardWarpCompensationCheck_);
    compensationForm->addRow(QStringLiteral("形变先验 sigma"), boardWarpPriorSigmaSpin_);
    compensationForm->addRow(QStringLiteral("形变最大偏移"), boardWarpMaxOffsetSpin_);
    compensationForm->addRow(QStringLiteral("逐点残差"), enableBoardPointCompensationCheck_);
    compensationForm->addRow(QStringLiteral("点先验 sigma"), boardPointPriorSigmaSpin_);
    compensationForm->addRow(QStringLiteral("点平滑 sigma"), boardPointSmoothSigmaSpin_);
    compensationForm->addRow(QStringLiteral("点最大偏移"), boardPointMaxOffsetSpin_);

    const auto updateBoardCompensationWidgets = [this]() {
        const bool imageResidualEnabled = enableImageResidualCompensationCheck_->isChecked();
        imageResidualPriorSigmaSpin_->setEnabled(imageResidualEnabled);
        imageResidualMaxCoeffSpin_->setEnabled(imageResidualEnabled);

        const bool warpEnabled = enableBoardWarpCompensationCheck_->isChecked();
        boardWarpPriorSigmaSpin_->setEnabled(warpEnabled);
        boardWarpMaxOffsetSpin_->setEnabled(warpEnabled);

        const bool pointEnabled = enableBoardPointCompensationCheck_->isChecked();
        boardPointPriorSigmaSpin_->setEnabled(pointEnabled);
        boardPointSmoothSigmaSpin_->setEnabled(pointEnabled);
        boardPointMaxOffsetSpin_->setEnabled(pointEnabled);
    };
    updateBoardCompensationWidgets();
    connect(enableImageResidualCompensationCheck_,
            &QCheckBox::checkStateChanged,
            this,
            [updateBoardCompensationWidgets](Qt::CheckState) { updateBoardCompensationWidgets(); });
    connect(enableBoardWarpCompensationCheck_,
            &QCheckBox::checkStateChanged,
            this,
            [updateBoardCompensationWidgets](Qt::CheckState) { updateBoardCompensationWidgets(); });
    connect(enableBoardPointCompensationCheck_,
            &QCheckBox::checkStateChanged,
            this,
            [updateBoardCompensationWidgets](Qt::CheckState) { updateBoardCompensationWidgets(); });

    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("补偿参数"), compensationGroup, true, this));

    auto* policyGroup = new QGroupBox(QStringLiteral("运行策略"), this);
    auto* policyForm = new QFormLayout(policyGroup);

    preferCachedCheck_ = new QCheckBox(QStringLiteral("优先复用缓存"), policyGroup);
    preferCachedCheck_->setChecked(true);

    persistCacheCheck_ = new QCheckBox(QStringLiteral("保存缓存结果"), policyGroup);
    persistCacheCheck_->setChecked(true);

    enforceSquarePixelsCheck_ = new QCheckBox(QStringLiteral("约束 fx/fy 一致"), policyGroup);
    enforceSquarePixelsCheck_->setChecked(true);

    lockPrincipalPointCheck_ = new QCheckBox(QStringLiteral("锁定主点到图像中心"), policyGroup);
    lockPrincipalPointCheck_->setChecked(true);

    policyForm->addRow(QStringLiteral("缓存"), preferCachedCheck_);
    policyForm->addRow(QStringLiteral("写回"), persistCacheCheck_);
    policyForm->addRow(QStringLiteral("像元约束"), enforceSquarePixelsCheck_);
    policyForm->addRow(QStringLiteral("主点策略"), lockPrincipalPointCheck_);

    rootLayout->addWidget(makeCollapsibleSection(QStringLiteral("执行策略"), policyGroup, true, this));
    rootLayout->addStretch(1);

    connect(browseImageDirButton, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("选择标定图像目录"), imageDirEdit_->text());
        if (!dir.isEmpty()) {
            imageDirEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });

    connect(browseCacheDirButton, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("选择标定缓存目录"), cacheDirEdit_->text());
        if (!dir.isEmpty()) {
            cacheDirEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });

    connect(browseOutputDirButton, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("选择标定结果输出目录"), outputDirEdit_->text());
        if (!dir.isEmpty()) {
            outputDirEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });

    connect(imageDirEdit_, &QLineEdit::textChanged, this, [this]() {
        updateDetectedCount();
        emit configChanged();
    });

    connect(cacheDirEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(outputDirEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });

    connectChangeSignals();
    updateDetectedCount();
}

bool CalibrationConfigPanel::buildRequest(pinjie::CameraCalibrationRequest& request, QString& errorMessage) const
{
    const QString imageDir = imageDirEdit_->text().trimmed();
    const QString cacheDir = cacheDirEdit_->text().trimmed();
    const QString outputDir = outputDirEdit_->text().trimmed();

    if (imageDir.isEmpty() && cacheDir.isEmpty()) {
        errorMessage = QStringLiteral("请至少提供标定图像目录或已有标定缓存目录。");
        return false;
    }

    if (!imageDir.isEmpty()) {
        QDir dir(imageDir);
        if (!dir.exists()) {
            errorMessage = QStringLiteral("标定图像目录不存在。");
            return false;
        }

        if (scanImagePaths().isEmpty()) {
            errorMessage = QStringLiteral("标定图像目录中未找到支持的图像文件。");
            return false;
        }
    }

    request = {};
    request.sessionName = sessionNameEdit_->text().trimmed().isEmpty()
                              ? QStringLiteral("camera_calibration").toUtf8().toStdString()
                              : sessionNameEdit_->text().trimmed().toUtf8().toStdString();
    request.imageDirectory = QDir::fromNativeSeparators(imageDir).toUtf8().toStdString();
    request.outputDir = QDir::fromNativeSeparators(outputDir).toUtf8().toStdString();
    request.cacheDir = QDir::fromNativeSeparators(cacheDir).toUtf8().toStdString();
    request.boardSpec.rows = rowsSpin_->value();
    request.boardSpec.cols = colsSpin_->value();
    request.boardSpec.pitchMm = pitchSpin_->value();
    request.boardSpec.roiRadiusPx = roiRadiusSpin_->value();
    request.boardSpec.minValidImages = minValidImagesSpin_->value();
    request.boardSpec.enforceSquarePixels = enforceSquarePixelsCheck_->isChecked();
    request.boardSpec.lockPrincipalPointToImageCenter = lockPrincipalPointCheck_->isChecked();
    request.boardSpec.enableImageResidualCompensation = enableImageResidualCompensationCheck_->isChecked();
    request.boardSpec.imageResidualPriorSigmaPx = imageResidualPriorSigmaSpin_->value();
    request.boardSpec.imageResidualMaxCoeffPx = imageResidualMaxCoeffSpin_->value();
    request.boardSpec.enableBoardWarpCompensation = enableBoardWarpCompensationCheck_->isChecked();
    request.boardSpec.boardWarpPriorSigmaMm = boardWarpPriorSigmaSpin_->value();
    request.boardSpec.boardWarpMaxOffsetMm = boardWarpMaxOffsetSpin_->value();
    request.boardSpec.enableBoardPointCompensation = enableBoardPointCompensationCheck_->isChecked();
    request.boardSpec.boardPointPriorSigmaMm = boardPointPriorSigmaSpin_->value();
    request.boardSpec.boardPointSmoothSigmaMm = boardPointSmoothSigmaSpin_->value();
    request.boardSpec.boardPointMaxOffsetMm = boardPointMaxOffsetSpin_->value();
    request.preferCachedResult = preferCachedCheck_->isChecked();
    request.persistCache = persistCacheCheck_->isChecked();
    return true;
}

void CalibrationConfigPanel::setRunning(bool running)
{
    setEnabled(!running);
}

QStringList CalibrationConfigPanel::scanImagePaths() const
{
    const QString inputDir = imageDirEdit_->text().trimmed();
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

void CalibrationConfigPanel::updateDetectedCount()
{
    detectedCountLabel_->setText(QStringLiteral("已检测 %1 张标定图像").arg(scanImagePaths().size()));
}

void CalibrationConfigPanel::connectChangeSignals()
{
    connect(sessionNameEdit_, &QLineEdit::textChanged, this, [this](const QString&) { emit configChanged(); });
    connect(rowsSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(colsSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(pitchSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(roiRadiusSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(minValidImagesSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(preferCachedCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(persistCacheCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(enforceSquarePixelsCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(lockPrincipalPointCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(enableImageResidualCompensationCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(imageResidualPriorSigmaSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(imageResidualMaxCoeffSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(enableBoardWarpCompensationCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(boardWarpPriorSigmaSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(boardWarpMaxOffsetSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(enableBoardPointCompensationCheck_, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { emit configChanged(); });
    connect(boardPointPriorSigmaSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(boardPointSmoothSigmaSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
    connect(boardPointMaxOffsetSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &CalibrationConfigPanel::configChanged);
}

} // namespace pinjie::gui
