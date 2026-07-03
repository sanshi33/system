#pragma once

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QSize>

class QMouseEvent;
class QPainter;
class QResizeEvent;
class QWheelEvent;

namespace pinjie::gui {

class ImageViewer : public QGraphicsView {
public:
    explicit ImageViewer(QWidget* parent = nullptr);

    void setImage(const QImage& image);
    void clearImage();
    [[nodiscard]] QSize recommendedExportPixelSize() const;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void drawForeground(QPainter* painter, const QRectF& rect) override;

private:
    void fitImageToView();

    QGraphicsScene scene_;
    QGraphicsPixmapItem* pixmapItem_{nullptr};
    bool hasImage_{false};
    double zoomFactor_{1.0};
};

} // namespace pinjie::gui
