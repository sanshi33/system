#include "gui/ImageViewer.h"

#include <QFrame>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace pinjie::gui {

ImageViewer::ImageViewer(QWidget* parent)
    : QGraphicsView(parent)
{
    setScene(&scene_);
    pixmapItem_ = scene_.addPixmap(QPixmap());
    setRenderHint(QPainter::Antialiasing, false);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setRenderHint(QPainter::TextAntialiasing, true);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
    setBackgroundBrush(QColor(31, 40, 50));
    setFrameShape(QFrame::StyledPanel);
    setAlignment(Qt::AlignCenter);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
}

void ImageViewer::setImage(const QImage& image)
{
    if (image.isNull()) {
        clearImage();
        return;
    }

    pixmapItem_->setPixmap(QPixmap::fromImage(image));
    scene_.setSceneRect(pixmapItem_->boundingRect());
    hasImage_ = true;
    zoomFactor_ = 1.0;
    fitImageToView();
}

void ImageViewer::clearImage()
{
    pixmapItem_->setPixmap(QPixmap());
    scene_.setSceneRect(QRectF());
    hasImage_ = false;
    zoomFactor_ = 1.0;
    resetTransform();
    viewport()->update();
}

QSize ImageViewer::recommendedExportPixelSize() const
{
    const QSize viewportSize = viewport() ? viewport()->size() : size();
    const double dpr = viewport() ? viewport()->devicePixelRatioF() : devicePixelRatioF();
    const double factor = std::clamp(dpr * 1.15, 1.15, 1.90);
    const int width = std::clamp(static_cast<int>(std::lround(viewportSize.width() * factor)), 960, 2400);
    const int height = std::clamp(static_cast<int>(std::lround(viewportSize.height() * factor)), 620, 1680);
    return QSize(width, height);
}

void ImageViewer::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    if (hasImage_) {
        fitImageToView();
    }
}

void ImageViewer::wheelEvent(QWheelEvent* event)
{
    if (!hasImage_) {
        QGraphicsView::wheelEvent(event);
        return;
    }

    const double factor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    const double nextZoomFactor = zoomFactor_ * factor;
    if (nextZoomFactor < 0.2 || nextZoomFactor > 12.0) {
        event->accept();
        return;
    }

    zoomFactor_ = nextZoomFactor;
    scale(factor, factor);
    event->accept();
}

void ImageViewer::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (hasImage_ && event->button() == Qt::LeftButton) {
        fitImageToView();
        event->accept();
        return;
    }

    QGraphicsView::mouseDoubleClickEvent(event);
}

void ImageViewer::drawForeground(QPainter* painter, const QRectF& rect)
{
    QGraphicsView::drawForeground(painter, rect);

    if (hasImage_) {
        return;
    }

    painter->save();
    painter->resetTransform();
    painter->setPen(QColor(195, 207, 220));

    QFont titleFont = painter->font();
    titleFont.setPointSize(std::max(titleFont.pointSize(), 10));
    titleFont.setBold(false);
    painter->setFont(titleFont);
    painter->drawText(viewport()->rect().adjusted(24, 0, -24, -18),
                      Qt::AlignCenter,
                      QStringLiteral("等待图像"));
    painter->restore();
}

void ImageViewer::fitImageToView()
{
    if (!hasImage_) {
        return;
    }

    resetTransform();
    fitInView(pixmapItem_, Qt::KeepAspectRatio);
    zoomFactor_ = 1.0;
}

} // namespace pinjie::gui
