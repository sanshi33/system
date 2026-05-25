#include "gui/CameraCaptureWorker.h"

#include "acquisition/CameraCaptureService.h"
#include "gui/QtImageUtils.h"

#include <QStringList>

#include <utility>

namespace pinjie::gui {

namespace {

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

QString buildPreviewSummary(const pinjie::ImageFrameInfo& frame)
{
    QStringList lines;
    lines << QStringLiteral("文件：%1").arg(fromUtf8StdString(frame.fileName));
    lines << QStringLiteral("编号：%1").arg(static_cast<qulonglong>(frame.index));
    lines << QStringLiteral("步号：%1").arg(static_cast<qulonglong>(frame.stepIndex));
    if (!frame.captureTimeIso.empty()) {
        lines << QStringLiteral("时间：%1").arg(fromUtf8StdString(frame.captureTimeIso));
    }
    lines << QStringLiteral("曝光：%1 us").arg(frame.exposure, 0, 'f', 1);
    lines << QStringLiteral("增益：%1").arg(frame.gain, 0, 'f', 2);
    if (frame.hasStagePosition) {
        lines << QStringLiteral("位置：X=%1, Y=%2")
                     .arg(frame.stageX, 0, 'f', 3)
                     .arg(frame.stageY, 0, 'f', 3);
    }
    if (!frame.note.empty()) {
        lines << QStringLiteral("备注：%1").arg(fromUtf8StdString(frame.note));
    }
    return lines.join('\n');
}

} // namespace

CameraCaptureWorker::CameraCaptureWorker(pinjie::CameraSequenceRequest request, QObject* parent)
    : QObject(parent)
    , request_(std::move(request))
{
}

void CameraCaptureWorker::run()
{
    pinjie::CameraCaptureService service;
    std::size_t previewCount = 0;
    request_.callbacks.onLog = [this](const std::string& message) {
        emit logMessage(fromUtf8StdString(message));
    };
    request_.callbacks.onProgress = [this](const std::size_t current, const std::size_t total) {
        emit progressChanged(static_cast<int>(current), static_cast<int>(total));
    };
    request_.callbacks.onFrameCaptured = [this, &previewCount](const pinjie::ImageFrameInfo& info, const cv::Mat& image) {
        ++previewCount;
        emit previewReady(static_cast<int>(previewCount),
                          request_.frameCount,
                          cvMatToQImage(image),
                          buildPreviewSummary(info));
    };
    request_.callbacks.isCancelled = [this]() {
        return stopRequested_.load();
    };

    pinjie::CameraSequenceResult result;
    switch (request_.mode) {
    case pinjie::CameraCaptureMode::Single:
        result = service.captureSingle(request_);
        break;
    case pinjie::CameraCaptureMode::ManualStepScan:
        result = service.captureManualStep(request_);
        break;
    case pinjie::CameraCaptureMode::Sequence:
    default:
        result = service.captureSequence(request_);
        break;
    }

    emit runFinished(result.ok,
                     result.cancelled,
                     fromUtf8StdString(result.message),
                     static_cast<int>(result.frames.size()),
                     fromUtf8StdString(result.manifestPath),
                     fromUtf8StdString(request_.outputDir));
}

void CameraCaptureWorker::requestStop()
{
    stopRequested_.store(true);
}

} // namespace pinjie::gui
