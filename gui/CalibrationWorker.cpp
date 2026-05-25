#include "gui/CalibrationWorker.h"

#include "gui/QtImageUtils.h"

namespace pinjie::gui {

namespace {

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

} // namespace

CalibrationWorker::CalibrationWorker(pinjie::CameraCalibrationRequest request, QObject* parent)
    : QObject(parent)
    , request_(std::move(request))
{
}

void CalibrationWorker::run()
{
    if (stopRequested_.load()) {
        pinjie::CalibrationResultCache cache;
        emit runFinished(false, false, QStringLiteral("标定任务已取消。"), cache);
        return;
    }

    auto service = pinjie::createCameraCalibrationService();

    request_.callbacks.logMessage = [this](const std::string& message) {
        emit logMessage(fromUtf8StdString(message));
    };
    request_.callbacks.progressChanged = [this](const int current, const int total) {
        emit progressChanged(current, total);
    };
    request_.callbacks.previewReady = [this](const int index, const int total, const cv::Mat& image) {
        emit previewReady(index, total, cvMatToQImage(image));
    };

    const pinjie::CameraCalibrationResult result = service->runCalibration(request_);
    emit runFinished(result.ok,
                     result.loadedFromCache,
                     fromUtf8StdString(result.message),
                     result.cache);
}

void CalibrationWorker::requestStop()
{
    stopRequested_.store(true);
}

} // namespace pinjie::gui
