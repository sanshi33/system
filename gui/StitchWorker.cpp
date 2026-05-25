#include "gui/StitchWorker.h"

#include "gui/QtImageUtils.h"

namespace pinjie::gui {

namespace {

QString fromUtf8StdString(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
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
