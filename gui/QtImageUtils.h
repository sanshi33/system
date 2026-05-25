#pragma once

#include <QImage>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace pinjie::gui {

inline QImage cvMatToQImage(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }

    switch (image.type()) {
    case CV_8UC1: {
        QImage view(image.data,
                    image.cols,
                    image.rows,
                    static_cast<int>(image.step),
                    QImage::Format_Grayscale8);
        return view.copy();
    }
    case CV_8UC3: {
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        QImage view(rgb.data,
                    rgb.cols,
                    rgb.rows,
                    static_cast<int>(rgb.step),
                    QImage::Format_RGB888);
        return view.copy();
    }
    case CV_8UC4: {
        cv::Mat rgba;
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
        QImage view(rgba.data,
                    rgba.cols,
                    rgba.rows,
                    static_cast<int>(rgba.step),
                    QImage::Format_RGBA8888);
        return view.copy();
    }
    default: {
        cv::Mat normalized;
        cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX);
        cv::Mat u8;
        normalized.convertTo(u8, CV_8U);
        return cvMatToQImage(u8);
    }
    }
}

} // namespace pinjie::gui
