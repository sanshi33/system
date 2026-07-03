#pragma once

#include "cad_design/DesignProfileTypes.h"
#include "cad_model/CadModelLoader.h"

#include <QColor>
#include <QPointF>
#include <QString>
#include <QVector>
#include <QWidget>

#include <vector>

class QMouseEvent;
class QPaintEvent;
class QWheelEvent;

namespace pinjie::gui {

struct PointCloud3DPoint {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double value{0.0};
    bool hasValue{false};
};

class PointCloud3DViewer : public QWidget {
public:
    explicit PointCloud3DViewer(QWidget* parent = nullptr);

    void clearCloud(const QString& message = QString());
    void setCadModelDocument(const pinjie::cad_model::CadModelDocument& document,
                             const QString& title,
                             const QString& subtitle);
    void setCadProfileSamples(const std::vector<pinjie::cad_design::DesignProfileSample>& samples,
                              const QString& title,
                              const QString& subtitle);
    bool loadDesignErrorCsv(const QString& csvPath, QString* message = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    struct Series {
        QString label;
        QColor color;
        QVector<PointCloud3DPoint> points;
        bool drawLine{true};
        bool drawPoints{true};
    };

    struct Link {
        PointCloud3DPoint from;
        PointCloud3DPoint to;
        QColor color;
    };

    struct MeshTriangle {
        PointCloud3DPoint a;
        PointCloud3DPoint b;
        PointCloud3DPoint c;
        QColor fill;
        QColor edge;
    };

    struct ProjectedPoint {
        QPointF screen;
        double depth{0.0};
    };

    void resetView();
    void updateBounds();
    void setSeries(QVector<Series> series, QVector<Link> links, QString title, QString subtitle);
    void setDocumentPointCloud(const pinjie::cad_model::CadModelDocument& document,
                               const QString& title,
                               const QString& subtitle,
                               const QString& pointLabel,
                               const QString& profileLabel,
                               const QString& statusPrefix);
    [[nodiscard]] ProjectedPoint project(const PointCloud3DPoint& point) const;
    [[nodiscard]] QPointF projectRaw(double x, double y, double z) const;
    [[nodiscard]] bool hasRenderablePoints() const;

    QVector<Series> series_;
    QVector<Link> links_;
    QVector<MeshTriangle> meshTriangles_;
    QString title_{QStringLiteral("交互3D点云")};
    QString subtitle_{QStringLiteral("等待点云数据。")};
    QString statusText_;
    PointCloud3DPoint center_{};
    double radius_{1.0};
    double yawDeg_{-42.0};
    double pitchDeg_{24.0};
    double zoom_{1.0};
    QPointF pan_{0.0, 0.0};
    QPoint lastMousePos_;
    bool rotating_{false};
    bool panning_{false};
};

} // namespace pinjie::gui
