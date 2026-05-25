#pragma once

#include "app/StitchWorkflowService.h"
#include "reconstruction/ReconstructionTypes.h"

#include <QImage>
#include <QObject>

#include <atomic>

Q_DECLARE_METATYPE(pinjie::StitchStepRecord)
Q_DECLARE_METATYPE(pinjie::StitchRunCachePtr)

namespace pinjie::gui {

class StitchWorker : public QObject {
    Q_OBJECT

public:
    explicit StitchWorker(pinjie::StitchRunRequest request,
                          pinjie::StitchRunMode runMode,
                          QObject* parent = nullptr);

public slots:
    void run();
    void requestStop();

signals:
    void logMessage(const QString& message);
    void progressChanged(const QString& stage, int current, int total);
    void stepCompleted(pinjie::StitchStepRecord step);
    void imageReady(const QString& stage, int index, int total, const QImage& image);
    void runFinished(bool ok,
                     const QString& message,
                     int loadedImageCount,
                     int preprocessedImageCount,
                     const QImage& panorama,
                     const QString& csvText,
                     pinjie::StitchRunCachePtr cache);

private:
    pinjie::StitchRunRequest request_;
    pinjie::StitchRunMode runMode_{pinjie::StitchRunMode::Full};
    std::atomic_bool stopRequested_{false};
};

} // namespace pinjie::gui
