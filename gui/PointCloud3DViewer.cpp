#include "gui/PointCloud3DViewer.h"

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPolygonF>
#include <QStringConverter>
#include <QTextStream>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>

namespace pinjie::gui {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool finite(double value)
{
    return std::isfinite(value);
}

double toDouble(const QString& value)
{
    bool ok = false;
    const double parsed = value.trimmed().toDouble(&ok);
    return ok ? parsed : std::numeric_limits<double>::quiet_NaN();
}

int toInt(const QString& value, int defaultValue = 0)
{
    bool ok = false;
    const double parsed = value.trimmed().toDouble(&ok);
    return ok ? static_cast<int>(std::lround(parsed)) : defaultValue;
}

QStringList splitCsvLine(const QString& line)
{
    QStringList columns;
    QString current;
    bool inQuotes = false;
    for (int i = 0; i < line.size(); ++i) {
        const QChar ch = line.at(i);
        if (ch == QLatin1Char('"')) {
            if (inQuotes && i + 1 < line.size() && line.at(i + 1) == QLatin1Char('"')) {
                current.append(ch);
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch == QLatin1Char(',') && !inQuotes) {
            columns.append(current);
            current.clear();
        } else {
            current.append(ch);
        }
    }
    columns.append(current);
    return columns;
}

QColor withAlpha(QColor color, int alpha)
{
    color.setAlpha(alpha);
    return color;
}

int samplingStride(const int itemCount, const int budget)
{
    if (itemCount <= 0 || budget <= 0) {
        return 1;
    }
    return std::max(1, (itemCount + budget - 1) / budget);
}

} // namespace

PointCloud3DViewer::PointCloud3DViewer(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(320);
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setAutoFillBackground(false);
}

void PointCloud3DViewer::clearCloud(const QString& message)
{
    series_.clear();
    links_.clear();
    meshTriangles_.clear();
    title_ = QStringLiteral("交互3D点云");
    subtitle_ = message.isEmpty() ? QStringLiteral("等待点云数据。") : message;
    statusText_.clear();
    resetView();
    update();
}

void PointCloud3DViewer::setDocumentPointCloud(
    const pinjie::cad_model::CadModelDocument& document,
    const QString& title,
    const QString& subtitle,
    const QString& pointLabel,
    const QString& profileLabel,
    const QString& statusPrefix)
{
    if (document.meshVertices.empty()) {
        setCadProfileSamples(document.hasSectionSamples ? document.sectionSamples : document.profileSamples,
                             title,
                             subtitle);
        return;
    }

    Series cloudSeries;
    cloudSeries.label = pointLabel;
    cloudSeries.color = QColor(38, 139, 210);
    cloudSeries.drawLine = false;
    cloudSeries.drawPoints = true;

    const int pointCount = static_cast<int>(document.meshVertices.size());
    const int stride = std::max(1, pointCount / 25000);
    for (int index = 0; index < pointCount; index += stride) {
        const auto& point = document.meshVertices[index];
        cloudSeries.points.push_back({point.xMm, point.yMm, point.zMm, 0.0, false});
    }

    QVector<Series> series;
    series.push_back(cloudSeries);

    Series profileSeries;
    profileSeries.label = profileLabel;
    profileSeries.color = QColor(27, 158, 119);
    profileSeries.drawLine = false;
    profileSeries.drawPoints = true;
    const auto& previewSamples =
        document.hasSectionSamples ? document.sectionSamples : document.profileSamples;
    for (const auto& sample : previewSamples) {
        if (!sample.hasCadPoint ||
            !finite(sample.cadXMm) ||
            !finite(sample.cadYMm) ||
            !finite(sample.cadZMm)) {
            continue;
        }
        profileSeries.points.push_back({sample.cadXMm, sample.cadYMm, sample.cadZMm, sample.sMm, true});
    }
    if (profileSeries.points.size() >= 2) {
        series.push_back(profileSeries);
    }

    setSeries(series, {}, title, subtitle);
    statusText_ = QStringLiteral("%1: %2 / %3 points | left drag rotate, right/middle drag pan, wheel zoom")
                      .arg(statusPrefix)
                      .arg(cloudSeries.points.size())
                      .arg(document.meshVertices.size());
}

void PointCloud3DViewer::setCadModelDocument(const pinjie::cad_model::CadModelDocument& document,
                                             const QString& title,
                                             const QString& subtitle)
{
    if (document.format == pinjie::cad_model::CadFileFormat::Stl && !document.meshVertices.empty()) {
        setDocumentPointCloud(document,
                              title,
                              subtitle,
                              QStringLiteral("STL point cloud"),
                              QStringLiteral("STL section/profile"),
                              QStringLiteral("STL point cloud"));
        return;
    }

    if ((!document.hasMesh || document.meshTriangles.empty()) && !document.meshVertices.empty()) {
        setDocumentPointCloud(document,
                              title,
                              subtitle,
                              QStringLiteral("CAD XYZ points"),
                              QStringLiteral("CAD profile"),
                              QStringLiteral("CAD point cloud"));
        return;
    }

    if (!document.hasMesh || document.meshTriangles.empty() || document.meshVertices.empty()) {
        setCadProfileSamples(document.hasSectionSamples ? document.sectionSamples : document.profileSamples,
                             title,
                             subtitle);
        return;
    }

    QVector<MeshTriangle> meshTriangles;
    meshTriangles.reserve(static_cast<int>(document.meshTriangles.size()));
    const QColor fillColor(88, 143, 172, 92);
    const QColor edgeColor(50, 87, 110, 105);
    for (const auto& triangle : document.meshTriangles) {
        if (triangle.a >= document.meshVertices.size() ||
            triangle.b >= document.meshVertices.size() ||
            triangle.c >= document.meshVertices.size()) {
            continue;
        }
        const auto& a = document.meshVertices[triangle.a];
        const auto& b = document.meshVertices[triangle.b];
        const auto& c = document.meshVertices[triangle.c];
        meshTriangles.push_back({
            {a.xMm, a.yMm, a.zMm, 0.0, false},
            {b.xMm, b.yMm, b.zMm, 0.0, false},
            {c.xMm, c.yMm, c.zMm, 0.0, false},
            fillColor,
            edgeColor,
        });
    }

    Series profileSeries;
    profileSeries.label = QStringLiteral("CAD真实截面线");
    profileSeries.color = QColor(27, 158, 119);
    profileSeries.drawLine = false;
    profileSeries.drawPoints = true;
    const auto& previewSamples =
        document.hasSectionSamples ? document.sectionSamples : document.profileSamples;
    for (const auto& sample : previewSamples) {
        if (!sample.hasCadPoint ||
            !finite(sample.cadXMm) ||
            !finite(sample.cadYMm) ||
            !finite(sample.cadZMm)) {
            continue;
        }
        profileSeries.points.push_back({sample.cadXMm, sample.cadYMm, sample.cadZMm, sample.sMm, true});
    }

    series_.clear();
    if (profileSeries.points.size() >= 2) {
        series_.push_back(profileSeries);
    }
    links_.clear();
    meshTriangles_ = std::move(meshTriangles);
    title_ = title;
    subtitle_ = subtitle;
    statusText_ = QStringLiteral("CAD网格：%1 顶点 / %2 面片 | 左键旋转，右键/中键平移，滚轮缩放，双击复位")
                      .arg(document.meshVertices.size())
                      .arg(document.meshTriangles.size());
    resetView();
    updateBounds();
    update();
}

void PointCloud3DViewer::setCadProfileSamples(const std::vector<pinjie::cad_design::DesignProfileSample>& samples,
                                              const QString& title,
                                              const QString& subtitle)
{
    Series cadSeries;
    cadSeries.label = QStringLiteral("CAD采样点");
    cadSeries.color = QColor(27, 158, 119);
    cadSeries.drawLine = false;
    cadSeries.drawPoints = true;

    for (const auto& sample : samples) {
        if (!sample.hasCadPoint ||
            !finite(sample.cadXMm) ||
            !finite(sample.cadYMm) ||
            !finite(sample.cadZMm)) {
            continue;
        }
        cadSeries.points.push_back({sample.cadXMm, sample.cadYMm, sample.cadZMm, sample.sMm, true});
    }

    if (cadSeries.points.size() < 2) {
        clearCloud(QStringLiteral("CAD采样点不足，无法显示交互3D模型预览。"));
        return;
    }

    setSeries({cadSeries}, {}, title, subtitle);
    statusText_ = QStringLiteral("点数：%1 | 鼠标左键旋转，右键/中键平移，滚轮缩放，双击复位")
                      .arg(cadSeries.points.size());
}

bool PointCloud3DViewer::loadDesignErrorCsv(const QString& csvPath, QString* message)
{
    QFile file(csvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QString text = QStringLiteral("无法读取三维误差坐标CSV：%1").arg(csvPath);
        clearCloud(text);
        if (message) {
            *message = text;
        }
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    if (stream.atEnd()) {
        const QString text = QStringLiteral("三维误差坐标CSV为空：%1").arg(csvPath);
        clearCloud(text);
        if (message) {
            *message = text;
        }
        return false;
    }

    const QStringList headers = splitCsvLine(stream.readLine());
    QHash<QString, int> indexByName;
    for (int i = 0; i < headers.size(); ++i) {
        indexByName.insert(headers.at(i).trimmed(), i);
    }

    const auto valueAt = [&indexByName](const QStringList& row, const QString& key) -> QString {
        const int index = indexByName.value(key, -1);
        return index >= 0 && index < row.size() ? row.at(index) : QString();
    };
    const auto firstNumber = [&valueAt](const QStringList& row, std::initializer_list<const char*> keys) -> double {
        for (const char* key : keys) {
            const double value = toDouble(valueAt(row, QString::fromLatin1(key)));
            if (finite(value)) {
                return value;
            }
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    Series designSeries;
    designSeries.label = QStringLiteral("CAD目标XYZ");
    designSeries.color = QColor(27, 158, 119);
    designSeries.drawLine = false;
    designSeries.drawPoints = true;

    Series measuredSeries;
    measuredSeries.label = QStringLiteral("检测映射XYZ");
    measuredSeries.color = QColor(11, 95, 165);
    measuredSeries.drawLine = false;
    measuredSeries.drawPoints = true;

    Series sectionDesignSeries;
    sectionDesignSeries.label = QStringLiteral("CAD截面轮廓");
    sectionDesignSeries.color = QColor(27, 158, 119);
    sectionDesignSeries.drawLine = false;
    sectionDesignSeries.drawPoints = true;

    Series sectionMeasuredSeries;
    sectionMeasuredSeries.label = QStringLiteral("检测截面点");
    sectionMeasuredSeries.color = QColor(11, 95, 165);
    sectionMeasuredSeries.drawLine = false;

    int usedRows = 0;
    int cadCoordinateRows = 0;
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        if (line.trimmed().isEmpty()) {
            continue;
        }
        const QStringList row = splitCsvLine(line);
        const bool used = toInt(valueAt(row, QStringLiteral("is_used")), 1) == 1;
        if (!used) {
            continue;
        }
        ++usedRows;

        const bool hasCad = toInt(valueAt(row, QStringLiteral("has_cad_coordinates")), 0) == 1;
        const double s = firstNumber(row, {"s_aligned_mm", "design_s_mm"});
        if (hasCad) {
            const double dx = firstNumber(row, {"design_x_mm", "cad_design_x_mm", "nearest_design_cad_x_mm"});
            const double dy = firstNumber(row, {"design_y_mm", "cad_design_y_mm", "nearest_design_cad_y_mm"});
            const double dz = firstNumber(row, {"design_z_mm", "cad_design_z_mm", "nearest_design_cad_z_mm"});
            const double mx = firstNumber(row, {"measured_x_mm", "cad_measured_x_mm", "measured_cad_x_mm"});
            const double my = firstNumber(row, {"measured_y_mm", "cad_measured_y_mm", "measured_cad_y_mm"});
            const double mz = firstNumber(row, {"measured_z_mm", "cad_measured_z_mm", "measured_cad_z_mm"});
            const double error3d = firstNumber(row, {"error_3d_um", "normal_error_um", "profile_error_um"});
            if (finite(dx) && finite(dy) && finite(dz) && finite(mx) && finite(my) && finite(mz)) {
                const PointCloud3DPoint designPoint{dx, dy, dz, s, finite(s)};
                const PointCloud3DPoint measuredPoint{mx, my, mz, error3d, finite(error3d)};
                designSeries.points.push_back(designPoint);
                measuredSeries.points.push_back(measuredPoint);
                ++cadCoordinateRows;
            }
        }

        const double measuredR = firstNumber(row, {"measured_r_mm", "r_aligned_mm"});
        const double designS = firstNumber(row, {"design_s_mm", "s_aligned_mm"});
        const double designR = firstNumber(row, {"design_r_mm", "r_design_mm", "nearest_design_r_mm"});
        const double sectionError = firstNumber(row, {"normal_error_um", "radial_error_um", "profile_error_um"});
        if (finite(s) && finite(measuredR)) {
            sectionMeasuredSeries.points.push_back({s, measuredR, 0.0, sectionError, finite(sectionError)});
        }
        if (finite(designS) && finite(designR)) {
            sectionDesignSeries.points.push_back({designS, designR, 0.0, s, finite(s)});
        }
    }

    if (designSeries.points.size() >= 2 && measuredSeries.points.size() >= 2) {
        setSeries({designSeries, measuredSeries},
                  {},
                  QStringLiteral("交互3D点云 - CAD目标点与检测映射点"),
                  QStringLiteral("数据源：%1；坐标基准为导入 CAD 的模型坐标。")
                      .arg(QFileInfo(csvPath).fileName()));
        statusText_ = QStringLiteral("有效XYZ点：%1 / 使用点：%2 | 鼠标左键旋转，右键/中键平移，滚轮缩放，双击复位")
                          .arg(cadCoordinateRows)
                          .arg(usedRows);
        if (message) {
            *message = QStringLiteral("已加载交互3D点云：%1 个 CAD XYZ 对齐点").arg(cadCoordinateRows);
        }
        return true;
    }

    if (sectionMeasuredSeries.points.size() >= 2) {
        QVector<Series> fallbackSeries;
        if (sectionDesignSeries.points.size() >= 2) {
            fallbackSeries.push_back(sectionDesignSeries);
        }
        fallbackSeries.push_back(sectionMeasuredSeries);
        setSeries(fallbackSeries,
                  {},
                  QStringLiteral("交互2.5D截面点云 - 未形成CAD XYZ映射"),
                  QStringLiteral("当前CSV没有有效 has_cad_coordinates=1 点，因此只显示统一截面点云；这不是完整 CAD 三维点云。"));
        statusText_ = QStringLiteral("截面点：%1 | 完整3D需要 design_x/y/z_mm 与 measured_x/y/z_mm 有效")
                          .arg(sectionMeasuredSeries.points.size());
        if (message) {
            *message = QStringLiteral("未形成完整CAD XYZ映射，已显示2.5D截面点云");
        }
        return true;
    }

    const QString text = QStringLiteral("CSV中没有足够的三维点或截面点：%1").arg(csvPath);
    clearCloud(text);
    if (message) {
        *message = text;
    }
    return false;
}

void PointCloud3DViewer::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    const bool interactiveMode = rotating_ || panning_;
    painter.setRenderHint(QPainter::Antialiasing, !interactiveMode);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.fillRect(rect(), QColor(247, 250, 252));

    painter.setPen(QColor(20, 32, 51));
    QFont titleFont = painter.font();
    titleFont.setPointSize(std::max(11, titleFont.pointSize() + 2));
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.drawText(QRect(18, 12, width() - 36, 24), Qt::AlignLeft | Qt::AlignVCenter, title_);

    QFont bodyFont = painter.font();
    bodyFont.setBold(false);
    bodyFont.setPointSize(std::max(9, bodyFont.pointSize() - 1));
    painter.setFont(bodyFont);
    painter.setPen(QColor(66, 84, 102));
    painter.drawText(QRect(18, 38, width() - 36, 38), Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap, subtitle_);

    if (!hasRenderablePoints()) {
        painter.setPen(QColor(107, 121, 138));
        painter.drawText(rect().adjusted(24, 80, -24, -36),
                         Qt::AlignCenter | Qt::TextWordWrap,
                         subtitle_.isEmpty() ? QStringLiteral("等待点云数据。") : subtitle_);
        return;
    }

    const QRect plotRect = rect().adjusted(20, 84, -20, -48);
    painter.save();
    painter.setClipRect(plotRect.adjusted(-10, -10, 10, 10));

    const double x0 = center_.x - radius_;
    const double x1 = center_.x + radius_;
    const double y0 = center_.y - radius_;
    const double y1 = center_.y + radius_;
    const double z0 = center_.z - radius_;
    const double z1 = center_.z + radius_;

    painter.setPen(QPen(QColor(214, 219, 224), 1.0));
    for (int i = 0; i <= 4; ++i) {
        const double t = static_cast<double>(i) / 4.0;
        const double x = x0 + (x1 - x0) * t;
        const double y = y0 + (y1 - y0) * t;
        painter.drawLine(projectRaw(x, y0, z0), projectRaw(x, y1, z0));
        painter.drawLine(projectRaw(x0, y, z0), projectRaw(x1, y, z0));
    }

    struct DrawTriangle {
        QPolygonF polygon;
        double depth{0.0};
        QColor fill;
        QColor edge;
    };
    QVector<DrawTriangle> drawTriangles;
    const int triangleCount = static_cast<int>(meshTriangles_.size());
    const int maxTriangleBudget = interactiveMode ? 900 : 18000;
    const int triangleStride = samplingStride(triangleCount, maxTriangleBudget);
    drawTriangles.reserve(std::min(triangleCount, maxTriangleBudget));
    for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += triangleStride) {
        const MeshTriangle& triangle = meshTriangles_.at(triangleIndex);
        const ProjectedPoint a = project(triangle.a);
        const ProjectedPoint b = project(triangle.b);
        const ProjectedPoint c = project(triangle.c);
        QPolygonF polygon;
        polygon << a.screen << b.screen << c.screen;
        drawTriangles.push_back({
            polygon,
            (a.depth + b.depth + c.depth) / 3.0,
            triangle.fill,
            triangle.edge,
        });
    }
    std::sort(drawTriangles.begin(), drawTriangles.end(), [](const DrawTriangle& lhs, const DrawTriangle& rhs) {
        return lhs.depth < rhs.depth;
    });
    for (const DrawTriangle& triangle : drawTriangles) {
        painter.setPen(QPen(triangle.edge, 0.6));
        painter.setBrush(triangle.fill);
        painter.drawPolygon(triangle.polygon);
    }

    const auto drawAxis = [&](double ex, double ey, double ez, const QColor& color, const QString& label) {
        const QPointF start = projectRaw(center_.x, center_.y, center_.z);
        const QPointF end = projectRaw(ex, ey, ez);
        painter.setPen(QPen(color, 2.0));
        painter.drawLine(start, end);
        painter.setPen(color.darker(130));
        painter.drawText(end + QPointF(6.0, -4.0), label);
    };
    drawAxis(x1, center_.y, center_.z, QColor(198, 99, 52), QStringLiteral("X"));
    drawAxis(center_.x, y1, center_.z, QColor(90, 143, 32), QStringLiteral("Y"));
    drawAxis(center_.x, center_.y, z1, QColor(49, 85, 201), QStringLiteral("Z"));

    painter.setPen(QPen(withAlpha(QColor(107, 114, 128), 110), 1.0));
    const int linkCount = static_cast<int>(links_.size());
    const int linkStride = std::max(1, linkCount / 160);
    for (int i = 0; i < linkCount; i += linkStride) {
        painter.drawLine(project(links_.at(i).from).screen, project(links_.at(i).to).screen);
    }

    for (const Series& series : series_) {
        if (series.drawLine && series.points.size() >= 2) {
            painter.setPen(QPen(withAlpha(series.color, 180), 1.5));
            const int seriesPointCount = static_cast<int>(series.points.size());
            const int lineStride = samplingStride(seriesPointCount - 1, interactiveMode ? 1200 : 8000);
            for (int i = lineStride; i < seriesPointCount; i += lineStride) {
                painter.drawLine(project(series.points.at(i - lineStride)).screen,
                                 project(series.points.at(i)).screen);
            }
        }
    }

    struct DrawPoint {
        QPointF screen;
        double depth{0.0};
        QColor color;
    };
    QVector<DrawPoint> points;
    int totalSourcePoints = 0;
    for (const Series& series : series_) {
        if (series.drawPoints) {
            totalSourcePoints += static_cast<int>(series.points.size());
        }
    }
    const int maxPointBudget = interactiveMode ? 7000 : 70000;
    for (const Series& series : series_) {
        if (!series.drawPoints) {
            continue;
        }
        const int seriesPointCount = static_cast<int>(series.points.size());
        const int seriesBudget =
            totalSourcePoints > 0
                ? std::max(800, maxPointBudget * seriesPointCount / totalSourcePoints)
                : maxPointBudget;
        const int pointStride = samplingStride(seriesPointCount, seriesBudget);
        for (int pointIndex = 0; pointIndex < seriesPointCount; pointIndex += pointStride) {
            const PointCloud3DPoint& point = series.points.at(pointIndex);
            const ProjectedPoint projected = project(point);
            points.push_back({projected.screen, projected.depth, series.color});
        }
    }
    std::sort(points.begin(), points.end(), [](const DrawPoint& lhs, const DrawPoint& rhs) {
        return lhs.depth < rhs.depth;
    });
    const double pointRadius = interactiveMode ? 2.2 : 3.3;
    for (const DrawPoint& point : points) {
        painter.setPen(QPen(point.color.darker(135), 0.7));
        painter.setBrush(point.color);
        painter.drawEllipse(point.screen, pointRadius, pointRadius);
    }

    painter.restore();

    int legendX = 22;
    const int legendY = height() - 36;
    for (const Series& series : series_) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(series.color);
        painter.drawEllipse(QPointF(legendX + 6, legendY + 7), 5, 5);
        painter.setPen(QColor(32, 48, 64));
        painter.drawText(QPoint(legendX + 18, legendY + 12), series.label);
        legendX += painter.fontMetrics().horizontalAdvance(series.label) + 44;
    }

    painter.setPen(QColor(82, 98, 116));
    painter.drawText(QRect(18, height() - 24, width() - 36, 20),
                     Qt::AlignRight | Qt::AlignVCenter,
                     statusText_.isEmpty()
                         ? QStringLiteral("鼠标左键旋转，右键/中键平移，滚轮缩放，双击复位")
                         : statusText_);
}

void PointCloud3DViewer::wheelEvent(QWheelEvent* event)
{
    const double factor = event->angleDelta().y() > 0 ? 1.12 : 1.0 / 1.12;
    zoom_ = std::clamp(zoom_ * factor, 0.18, 28.0);
    update();
    event->accept();
}

void PointCloud3DViewer::mousePressEvent(QMouseEvent* event)
{
    lastMousePos_ = event->pos();
    rotating_ = event->button() == Qt::LeftButton;
    panning_ = event->button() == Qt::RightButton || event->button() == Qt::MiddleButton;
    event->accept();
}

void PointCloud3DViewer::mouseMoveEvent(QMouseEvent* event)
{
    const QPoint delta = event->pos() - lastMousePos_;
    lastMousePos_ = event->pos();
    if (rotating_) {
        yawDeg_ += delta.x() * 0.45;
        pitchDeg_ = std::clamp(pitchDeg_ + delta.y() * 0.35, -88.0, 88.0);
        update();
    } else if (panning_) {
        pan_ += QPointF(delta);
        update();
    }
    event->accept();
}

void PointCloud3DViewer::mouseReleaseEvent(QMouseEvent* event)
{
    bool stateChanged = false;
    if (event->button() == Qt::LeftButton) {
        rotating_ = false;
        stateChanged = true;
    }
    if (event->button() == Qt::RightButton || event->button() == Qt::MiddleButton) {
        panning_ = false;
        stateChanged = true;
    }
    if (stateChanged) {
        update();
    }
    event->accept();
}

void PointCloud3DViewer::mouseDoubleClickEvent(QMouseEvent* event)
{
    resetView();
    update();
    event->accept();
}

void PointCloud3DViewer::resetView()
{
    yawDeg_ = -42.0;
    pitchDeg_ = 24.0;
    zoom_ = 1.0;
    pan_ = QPointF(0.0, 0.0);
}

void PointCloud3DViewer::updateBounds()
{
    bool hasPoint = false;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    const auto observe = [&](const PointCloud3DPoint& point) {
        if (!finite(point.x) || !finite(point.y) || !finite(point.z)) {
            return;
        }
        hasPoint = true;
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        minZ = std::min(minZ, point.z);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
        maxZ = std::max(maxZ, point.z);
    };

    for (const Series& series : series_) {
        for (const PointCloud3DPoint& point : series.points) {
            observe(point);
        }
    }
    for (const Link& link : links_) {
        observe(link.from);
        observe(link.to);
    }
    for (const MeshTriangle& triangle : meshTriangles_) {
        observe(triangle.a);
        observe(triangle.b);
        observe(triangle.c);
    }

    if (!hasPoint) {
        center_ = {};
        radius_ = 1.0;
        return;
    }

    center_.x = 0.5 * (minX + maxX);
    center_.y = 0.5 * (minY + maxY);
    center_.z = 0.5 * (minZ + maxZ);
    const double spanX = maxX - minX;
    const double spanY = maxY - minY;
    const double spanZ = maxZ - minZ;
    radius_ = std::max({spanX, spanY, spanZ, 1.0}) * 0.58;
}

void PointCloud3DViewer::setSeries(QVector<Series> series, QVector<Link> links, QString title, QString subtitle)
{
    series_ = std::move(series);
    links_ = std::move(links);
    meshTriangles_.clear();
    title_ = std::move(title);
    subtitle_ = std::move(subtitle);
    resetView();
    updateBounds();
    update();
}

PointCloud3DViewer::ProjectedPoint PointCloud3DViewer::project(const PointCloud3DPoint& point) const
{
    const double yaw = yawDeg_ * kPi / 180.0;
    const double pitch = pitchDeg_ * kPi / 180.0;
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);

    const double x = point.x - center_.x;
    const double y = point.y - center_.y;
    const double z = point.z - center_.z;
    const double x1 = cy * x - sy * y;
    const double y1 = sy * x + cy * y;
    const double z1 = z;
    const double y2 = cp * y1 - sp * z1;
    const double z2 = sp * y1 + cp * z1;

    const QRect plotRect = rect().adjusted(20, 84, -20, -48);
    const double scale = std::min(plotRect.width(), plotRect.height()) * 0.42 * zoom_ / std::max(radius_, 1e-9);
    return {
        QPointF(plotRect.center().x() + pan_.x() + x1 * scale,
                plotRect.center().y() + pan_.y() - y2 * scale),
        z2,
    };
}

QPointF PointCloud3DViewer::projectRaw(double x, double y, double z) const
{
    return project({x, y, z, 0.0, false}).screen;
}

bool PointCloud3DViewer::hasRenderablePoints() const
{
    for (const Series& series : series_) {
        if (!series.points.isEmpty()) {
            return true;
        }
    }
    return !meshTriangles_.isEmpty();
}

} // namespace pinjie::gui
