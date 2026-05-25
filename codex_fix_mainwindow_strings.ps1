$path = 'gui/MainWindow.cpp'
$content = Get-Content $path -Encoding utf8 -Raw

function Replace-Block([string]$text, [string]$pattern, [string]$replacement, [string]$name) {
    $regex = [regex]::new($pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $regex.IsMatch($text)) {
        throw "Failed to replace block: $name"
    }
    return $regex.Replace($text, $replacement, 1)
}

$content = Replace-Block $content 'void MainWindow::startRun\(pinjie::StitchRunMode runMode\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::stopRun\(\))' @'
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
    processingPreviewInfoEdit_->setPlainText(QStringLiteral("请选择一张预处理图像进行预览。"));
    referenceViewer_->clearImage();
    targetViewer_->clearImage();
    debugViewer_->clearImage();
    panoramaViewer_->clearImage();
    stepDetailPanel_->clearDetails();
    summaryEdit_->clear();
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
'@ 'startRun'

$content = Replace-Block $content 'void MainWindow::onImageReady\(const QString& stage, int index, int total, const QImage& image\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::onRunFinished)' @'
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
        updateReportMetricCards(static_cast<int>(lastImagePaths_.size()), preprocessImages_.size(), nullptr);
        return;
    }

    if (stage == QStringLiteral("debug_step")) {
        debugImages_.insert(index, image);
        debugViewer_->setImage(image);
        viewTabs_->setCurrentWidget(debugViewer_);
        statusBar()->showMessage(QStringLiteral("已更新调试图像 %1").arg(index));
    }
}
'@ 'onImageReady'

$content = Replace-Block $content 'void MainWindow::onRunFinished\(bool ok,\s*const QString& message,\s*int loadedImageCount,\s*int preprocessedImageCount,\s*const QImage& panorama,\s*const QString& csvText,\s*pinjie::StitchRunCachePtr cache\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::onStepSelectionChanged)' @'
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

    QStringList summaryLines;
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
        const QString qualityText = QString::fromStdString(pinjie::buildQualitySummaryText(completedSteps_, qualitySummary));
        if (!qualityText.isEmpty()) {
            summaryLines << QString() << qualityText;
        }
    }

    summaryEdit_->setPlainText(summaryLines.join('\n'));
    bottomTabs_->setCurrentWidget(summaryEdit_);
    updateReportMetricCards(loadedImageCount, preprocessedImageCount, qualitySummaryPtr);
    refreshRegistrationToolState();
    refreshReportExportState(ok ? QStringLiteral("结果数据已就绪，请选择要导出的 CSV 类型。")
                                : QStringLiteral("运行失败，请先检查日志，再决定是否导出 CSV。"));
    refreshStageSummaries();

    if (currentRunMode_ == pinjie::StitchRunMode::Full || currentRunMode_ == pinjie::StitchRunMode::Report) {
        switchToStage(ReportStage);
    } else {
        switchToStage(stageIndexForRunMode(currentRunMode_));
    }
}
'@ 'onRunFinished'

$content = Replace-Block $content 'void MainWindow::cleanupWorker\(\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::loadRawImagesForStep)' @'
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
'@ 'cleanupWorker'

$content = Replace-Block $content 'void MainWindow::onProcessingImageSelectionChanged\(\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::switchToStage)' @'
void MainWindow::onProcessingImageSelectionChanged()
{
    QListWidgetItem* currentItem = processingImageList_->currentItem();
    if (!currentItem) {
        processingPreviewViewer_->clearImage();
        processingPreviewInfoEdit_->setPlainText(QStringLiteral("请选择一张预处理图像进行预览。"));
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

    infoLines << QStringLiteral("提示：可在左侧列表中切换其他预处理结果。");
    processingPreviewInfoEdit_->setPlainText(infoLines.join('\n'));
}
'@ 'onProcessingImageSelectionChanged'

$content = Replace-Block $content 'void MainWindow::refreshStageSummaries\(\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::refreshCalibrationDetailPane)' @'
void MainWindow::refreshStageSummaries()
{
    pinjie::CameraCalibrationRequest calibrationRequest;
    QString calibrationError;
    const bool calibrationConfigValid = calibrationPanel_ && calibrationPanel_->buildRequest(calibrationRequest, calibrationError);

    pinjie::StitchRunRequest request;
    QString errorMessage;
    const bool valid = configPanel_->buildRequest(request, errorMessage, false);

    if (!hasActiveCalibration()) {
        workflowStateLabel_->setText(QStringLiteral("请先完成或加载标定结果，然后再运行后续阶段。"));
        registrationPresetLabel_->setText(calibrationConfigValid
                                              ? QStringLiteral("标定配置已就绪")
                                              : QStringLiteral("标定配置未完成"));
    } else {
        workflowStateLabel_->setText(
            valid ? QStringLiteral("%1，可用于 %2 张图像的拼接流程。")
                        .arg(calibrationIdentityLine(activeCalibrationCache_))
                        .arg(request.imagePaths.size())
                  : QStringLiteral("%1，但当前运行配置仍需补充。")
                        .arg(calibrationIdentityLine(activeCalibrationCache_)));
        registrationPresetLabel_->setText(calibrationQualityLine(activeCalibrationCache_));
    }

    if (!calibrationConfigValid) {
        calibrationOverviewEdit_->setPlainText(QStringLiteral("标定配置未完成：%1").arg(calibrationError));
    } else {
        QStringList calibrationLines;
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
        acquisitionOverviewEdit_->setPlainText(QStringLiteral("运行配置未完成：%1").arg(errorMessage));
        processingOverviewEdit_->setPlainText(QStringLiteral("请先完成图像输入和预处理参数配置。"));
        registrationOverviewEdit_->setPlainText(QStringLiteral("请先完成拼接与导出参数配置。"));
        return;
    }

    QStringList acquisitionLines;
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
'@ 'refreshStageSummaries'

$content = Replace-Block $content 'void MainWindow::refreshCalibrationDetailPane\(\)\s*\{.*?\n\}(?=\r?\n\r?\nbool MainWindow::hasActiveCalibration\(\) const)' @'
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
'@ 'refreshCalibrationDetailPane'

$content = Replace-Block $content 'void MainWindow::updateReportMetricCards\(int loadedImageCount,\s*int preprocessedImageCount,\s*const pinjie::QualitySummary\* qualitySummary\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::exportSelectedReportCsvs)' @'
void MainWindow::updateReportMetricCards(int loadedImageCount,
                                         int preprocessedImageCount,
                                         const pinjie::QualitySummary* qualitySummary)
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
        return;
    }

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
'@ 'updateReportMetricCards'

$content = Replace-Block $content 'void MainWindow::exportSelectedReportCsvs\(\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::refreshReportExportState)' @'
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

    if (!saveStepSummary && !saveContourPoints && !saveStitchedContourProfile && !saveTangentStep &&
        !saveNormalError && !saveTangentProfile) {
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
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
    }

    if (saveContourPoints) {
        const std::filesystem::path path = baseDir / "contour_points.csv";
        const std::string content =
            stitch::buildContourPointCsv(runCache_->preprocessedEdges, runCache_->stitching.imageTransforms);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this, QStringLiteral("导出 CSV"), QStringLiteral("保存 contour_points.csv 失败。"));
            refreshReportExportState(QStringLiteral("轮廓点 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
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
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
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
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
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
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
    }

    if (saveTangentProfile) {
        const std::filesystem::path path = baseDir / "tangent_profile_compare.csv";
        const std::string content = stitch::buildTangentProfileCompareCsv(runCache_->stitching.steps);
        if (!stitch::writeTextFileToPath(path.generic_u8string(), content)) {
            QMessageBox::warning(this,
                                 QStringLiteral("导出 CSV"),
                                 QStringLiteral("保存 tangent_profile_compare.csv 失败。"));
            refreshReportExportState(QStringLiteral("切向轮廓对比 CSV 导出失败"));
            return;
        }
        exportedFiles << QDir::toNativeSeparators(QString::fromStdString(path.generic_u8string()));
    }

    appendLog(QStringLiteral("[信息] 已导出 CSV 文件：\n%1").arg(exportedFiles.join('\n')));
    refreshReportExportState(QStringLiteral("已导出 %1 个 CSV 文件").arg(exportedFiles.size()));
    statusBar()->showMessage(QStringLiteral("已导出 %1 个 CSV 文件").arg(exportedFiles.size()));
}
'@ 'exportSelectedReportCsvs'

$content = Replace-Block $content 'void MainWindow::refreshReportExportState\(const QString& statusMessage\)\s*\{.*?\n\}(?=\r?\n\r?\nvoid MainWindow::applyWindowStyle)' @'
void MainWindow::refreshReportExportState(const QString& statusMessage)
{
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
        reportExportStatusLabel_->setText(QStringLiteral("结果数据已就绪，请先选择要导出的 CSV 类型。"));
        return;
    }

    const QString targetDir =
        lastRequest_.resultOutputDir.empty()
            ? QStringLiteral("当前结果目录")
            : QDir::toNativeSeparators(QString::fromStdString(lastRequest_.resultOutputDir));
    reportExportStatusLabel_->setText(
        QStringLiteral("可导出 CSV：%1\n输出目录：%2").arg(labels.join(QStringLiteral(", ")), targetDir));
}
'@ 'refreshReportExportState'

Set-Content $path -Encoding utf8 $content
