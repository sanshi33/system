#pragma once

#include "reconstruction/ReconstructionTypes.h"

#include <QAbstractTableModel>
#include <QVector>

namespace pinjie::gui {

class StepTableModel : public QAbstractTableModel {
public:
    explicit StepTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

    void clear();
    void appendStep(const pinjie::StitchStepRecord& step);
    const pinjie::StitchStepRecord* stepAt(int row) const;

private:
    enum Column {
        StepColumn,
        ImageAColumn,
        ImageBColumn,
        DxColumn,
        DyColumn,
        AngleColumn,
        NormalRmseColumn,
        TangentRmseColumn,
        TangentCorrColumn,
        CoverageColumn,
        InlierColumn,
        ColumnCount
    };

    QVector<pinjie::StitchStepRecord> steps_;
};

} // namespace pinjie::gui
