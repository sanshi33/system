#pragma once

#include "acquisition/CameraTypes.h"

#include <QImage>
#include <QObject>

#include <atomic>

namespace pinjie::gui {

class CameraCaptureWorker : public QObject {
    Q_OBJECT

public:
    explicit CameraCaptureWorker(pinjie::CameraSequenceRequest request, QObject* parent = nullptr);

public slots:
    void run();
    void requestStop();

signals:
    void logMessage(const QString& message);
    void progressChanged(int current, int total);
    void previewReady(int current, int total, const QImage& image, const QString& summary);
    void runFinished(bool ok,
                     bool cancelled,
                     const QString& message,
                     int savedCount,
                     const QString& manifestPath,
                     const QString& outputDir);

private:
    pinjie::CameraSequenceRequest request_{};
    std::atomic_bool stopRequested_{false};
};

} // namespace pinjie::gui
