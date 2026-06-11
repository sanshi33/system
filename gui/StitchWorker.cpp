#include "gui/StitchWorker.h"

#include "gui/QtImageUtils.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QRegularExpression>

namespace pinjie::gui {

namespace {

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

bool isStandardCircleMode(const pinjie::StitchRunRequest& request)
{
    return request.standardCircleConfig.enabled;
}

QString readUtf8TextFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QString::fromUtf8(file.readAll());
}

QString standardCircleExecutablePath()
{
    const QDir appDir(QCoreApplication::applicationDirPath());
#ifdef Q_OS_WIN
    const QString candidate = appDir.filePath(QStringLiteral("standard_sphere_loop_eval.exe"));
#else
    const QString candidate = appDir.filePath(QStringLiteral("standard_sphere_loop_eval"));
#endif
    return QFileInfo::exists(candidate) ? candidate : QString();
}

QStringList buildStandardCircleArguments(const pinjie::StitchRunRequest& request)
{
    QStringList arguments;
    if (request.imagePaths.empty()) {
        return arguments;
    }

    const QFileInfo firstImageInfo(fromUtf8StdString(request.imagePaths.front()));
    const auto& config = request.standardCircleConfig;
    arguments << QDir::toNativeSeparators(firstImageInfo.absolutePath())
              << QString::number(static_cast<int>(request.imagePaths.size()))
              << QStringLiteral("--out-dir")
              << QDir::toNativeSeparators(fromUtf8StdString(request.resultOutputDir))
              << QStringLiteral("--start-index")
              << QString::number(config.startIndex)
              << QStringLiteral("--prefix")
              << QString::fromStdString(config.imagePrefix)
              << QStringLiteral("--ext")
              << QString::fromStdString(config.imageExtension)
              << QStringLiteral("--sphere-diameter")
              << QString::number(config.sphereDiameterMm, 'f', 5)
              << QStringLiteral("--horizontal-fov")
              << QString::number(config.horizontalFieldOfViewMm, 'f', 6)
              << QStringLiteral("--vertical-fov")
              << QString::number(config.verticalFieldOfViewMm, 'f', 6)
              << QStringLiteral("--overlap")
              << QString::number(config.overlapRatio * 100.0, 'f', 6)
              << QStringLiteral("--gbt57-mask-standard-circle-profile")
              << QStringLiteral("--gui-progress")
              << QStringLiteral("--gbt57-window-half-size")
              << QString::number(config.windowHalfSizePx, 'f', 3)
              << QStringLiteral("--gbt57-window-half-angle")
              << QString::number(config.windowHalfAngleDeg, 'f', 3)
              << QStringLiteral("--gbt57-circular-median-filter-radius")
              << QString::number(config.circularMedianFilterRadius)
              << QStringLiteral("--gbt57-circular-filter-blend")
              << QString::number(config.circularFilterBlend, 'f', 6);
    return arguments;
}

} // namespace

StitchWorker::StitchWorker(pinjie::StitchRunRequest request,
                           pinjie::StitchRunMode runMode,
                           QObject* parent)
    : QObject(parent)
    , request_(std::move(request))
    , runMode_(runMode)
{
}

void StitchWorker::run()
{
    if (isStandardCircleMode(request_)) {
        emit progressChanged(QStringLiteral("load"), 0, 0);

        const QString executablePath = standardCircleExecutablePath();
        if (executablePath.isEmpty()) {
            emit runFinished(false,
                             QStringLiteral("未找到 standard_sphere_loop_eval 可执行程序。"),
                             0,
                             0,
                             QImage(),
                             QString(),
                             {});
            return;
        }

        QProcess process;
        process.setProgram(executablePath);
        process.setArguments(buildStandardCircleArguments(request_));
        process.setProcessChannelMode(QProcess::SeparateChannels);
        process.start();
        if (!process.waitForStarted()) {
            emit runFinished(false,
                             QStringLiteral("标准圆检测进程启动失败。"),
                             0,
                             0,
                             QImage(),
                             QString(),
                             {});
            return;
        }

        const QString guiProgressDir =
            QDir(QDir::fromNativeSeparators(fromUtf8StdString(request_.resultOutputDir)))
                .filePath(QStringLiteral("gui_progress"));
        QByteArray stdOutBuffer;
        QByteArray stdErrBuffer;
        const QString guiProgressPrefix = QStringLiteral("__GUI_PROGRESS__|");
        const QString guiImagePrefix = QStringLiteral("__GUI_IMAGE__|");
        const auto handleProcessLine = [this, guiProgressDir, guiProgressPrefix, guiImagePrefix](const QString& line) {
            if (line.startsWith(guiProgressPrefix)) {
                const QStringList parts = line.split(QLatin1Char('|'));
                bool currentOk = false;
                bool totalOk = false;
                const int current = parts.size() > 2 ? parts.at(2).toInt(&currentOk) : 0;
                const int total = parts.size() > 3 ? parts.at(3).toInt(&totalOk) : 0;
                if (parts.size() >= 4 && currentOk && totalOk) {
                    emit progressChanged(parts.at(1), current, total);
                    return;
                }
            }

            if (line.startsWith(guiImagePrefix)) {
                const QStringList parts = line.split(QLatin1Char('|'));
                bool indexOk = false;
                bool totalOk = false;
                const int index = parts.size() > 2 ? parts.at(2).toInt(&indexOk) : 0;
                const int total = parts.size() > 3 ? parts.at(3).toInt(&totalOk) : 0;
                if (parts.size() >= 5 && indexOk && totalOk) {
                    const QString imagePath =
                        QDir(guiProgressDir).filePath(parts.at(4));
                    const QImage image(imagePath);
                    if (!image.isNull()) {
                        emit imageReady(parts.at(1), index, total, image);
                    } else {
                        emit logMessage(QStringLiteral("[警告] 过程图加载失败：%1").arg(imagePath));
                    }
                    return;
                }
            }

            emit logMessage(line);
        };
        const auto processChunk = [&handleProcessLine](QByteArray& buffer, const QByteArray& chunk) {
            if (chunk.isEmpty()) {
                return;
            }

            buffer += chunk;
            while (true) {
                const int newlineIndex = buffer.indexOf('\n');
                if (newlineIndex < 0) {
                    break;
                }

                QByteArray lineBytes = buffer.left(newlineIndex);
                buffer.remove(0, newlineIndex + 1);
                if (!lineBytes.isEmpty() && lineBytes.endsWith('\r')) {
                    lineBytes.chop(1);
                }
                if (!lineBytes.isEmpty()) {
                    handleProcessLine(QString::fromLocal8Bit(lineBytes));
                }
            }
        };
        const auto flushBuffer = [&handleProcessLine](QByteArray& buffer) {
            if (!buffer.isEmpty()) {
                if (buffer.endsWith('\r')) {
                    buffer.chop(1);
                }
                if (!buffer.isEmpty()) {
                    handleProcessLine(QString::fromLocal8Bit(buffer));
                }
                buffer.clear();
            }
        };

        while (process.state() != QProcess::NotRunning) {
            if (stopRequested_.load()) {
                process.kill();
                process.waitForFinished(3000);
                emit runFinished(false,
                                 QStringLiteral("标准圆检测已取消。"),
                                 0,
                                 0,
                                 QImage(),
                                 QString(),
                                 {});
                return;
            }

            process.waitForFinished(100);
            processChunk(stdOutBuffer, process.readAllStandardOutput());
            processChunk(stdErrBuffer, process.readAllStandardError());
        }

        processChunk(stdOutBuffer, process.readAllStandardOutput());
        processChunk(stdErrBuffer, process.readAllStandardError());
        flushBuffer(stdOutBuffer);
        flushBuffer(stdErrBuffer);

        emit progressChanged(QStringLiteral("report"), 1, 1);
        const QString summaryCsv = readUtf8TextFile(
            QDir::toNativeSeparators(fromUtf8StdString(request_.csvOutputPath)));
        const QImage panorama(
            QDir::toNativeSeparators(fromUtf8StdString(request_.panoramaOutputPath)));
        const bool ok = process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0;
        const QString finishMessage =
            ok ? QStringLiteral("标准圆国标检测完成。")
               : QStringLiteral("标准圆国标检测失败，请检查日志和结果目录。");
        emit runFinished(ok,
                         finishMessage,
                         static_cast<int>(request_.imagePaths.size()),
                         static_cast<int>(request_.imagePaths.size()),
                         panorama,
                         summaryCsv,
                         {});
        return;
    }

    const pinjie::StitchWorkflowService workflow;
    pinjie::TaskCallbacks callbacks;

    callbacks.onLog = [this](const std::string& message) {
        emit logMessage(fromUtf8StdString(message));
    };

    callbacks.onProgress = [this](const std::string& stage, std::size_t current, std::size_t total) {
        emit progressChanged(fromUtf8StdString(stage),
                             static_cast<int>(current),
                             static_cast<int>(total));
    };

    callbacks.onStepFinished = [this](const stitch::StitchStepRecord& step) {
        emit stepCompleted(step);
    };

    callbacks.onImage = [this](const std::string& stage,
                               std::size_t index,
                               std::size_t total,
                               const cv::Mat& image) {
        emit imageReady(fromUtf8StdString(stage),
                        static_cast<int>(index),
                        static_cast<int>(total),
                        cvMatToQImage(image));
    };

    callbacks.isCancelled = [this]() {
        return stopRequested_.load();
    };

    const pinjie::StitchRunResult result = workflow.run(request_, runMode_, callbacks);
    const QImage panorama = result.ok ? cvMatToQImage(result.stitching.canvas) : QImage();

    emit runFinished(result.ok,
                     fromUtf8StdString(result.message),
                     static_cast<int>(result.loadedImageCount),
                     static_cast<int>(result.preprocessedImageCount),
                     panorama,
                     fromUtf8StdString(result.csvText),
                     result.cache);
}

void StitchWorker::requestStop()
{
    stopRequested_.store(true);
}

} // namespace pinjie::gui
