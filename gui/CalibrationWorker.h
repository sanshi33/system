#pragma once

#include "calibration/CalibrationService.h"

#include <QImage>
#include <QObject>

#include <atomic>

Q_DECLARE_METATYPE(pinjie::CalibrationResultCache)

namespace pinjie::gui {

class CalibrationWorker : public QObject {
    Q_OBJECT

public:
    explicit CalibrationWorker(pinjie::CameraCalibrationRequest request, QObject* parent = nullptr);

public slots:
    void run();
    void requestStop();

signals:
    void logMessage(const QString& message);
    void progressChanged(int current, int total);
    void previewReady(int index, int total, const QImage& image);
    void runFinished(bool ok, bool loadedFromCache, const QString& message, pinjie::CalibrationResultCache cache);

private:
    pinjie::CameraCalibrationRequest request_{};
    std::atomic_bool stopRequested_{false};
};

} // namespace pinjie::gui
