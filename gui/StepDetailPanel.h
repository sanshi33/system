#pragma once

#include "reconstruction/ReconstructionTypes.h"

#include <QLabel>
#include <QWidget>

namespace pinjie::gui {

class StepDetailPanel : public QWidget {
public:
    explicit StepDetailPanel(QWidget* parent = nullptr);

    void clearDetails();
    void setStep(const pinjie::StitchStepRecord& step);

private:
    QLabel* stepValue_{nullptr};
    QLabel* pairValue_{nullptr};
    QLabel* transformValue_{nullptr};
    QLabel* axisValue_{nullptr};
    QLabel* searchValue_{nullptr};
    QLabel* coverageValue_{nullptr};
    QLabel* sampleValue_{nullptr};
    QLabel* normalValue_{nullptr};
    QLabel* tangentValue_{nullptr};
    QLabel* corrValue_{nullptr};
    QLabel* statusValue_{nullptr};
};

} // namespace pinjie::gui
