#include "IO.h"

#include "DebugVis.h"
#include "EdgeProcessing.h"

#include <opencv2/imgcodecs.hpp>

#include "../core/SubpixelEdgeDetector.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace stitch {

namespace {

bool isCancelled(const StitchCallbacks& callbacks)
{
    return callbacks.isCancelled && callbacks.isCancelled();
}

void emitLog(const StitchCallbacks& callbacks, const std::string& message)
{
    if (callbacks.onLog) {
        callbacks.onLog(message);
    }
}

void emitProgress(const StitchCallbacks& callbacks,
                  const std::string& stage,
                  std::size_t current,
                  std::size_t total)
{
    if (callbacks.onProgress) {
        callbacks.onProgress(stage, current, total);
    }
}

std::filesystem::path pathFromUtf8(const std::string& pathUtf8)
{
#if defined(_WIN32)
    return std::filesystem::u8path(pathUtf8);
#else
    return std::filesystem::path(pathUtf8);
#endif
}

cv::Mat loadImageFromPath(const std::string& pathUtf8, const int flags)
{
    const std::filesystem::path filePath = pathFromUtf8(pathUtf8);
    std::ifstream stream(filePath, std::ios::binary);
    if (!stream.is_open()) {
        return {};
    }

    const std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(stream)),
                                            std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        return {};
    }

    return cv::imdecode(buffer, flags);
}

std::string normalizedImageExtension(const std::filesystem::path& filePath)
{
    std::string extension = filePath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    return extension.empty() ? std::string(".png") : extension;
}

} // namespace

std::vector<cv::Mat> loadInputImages(const std::vector<std::string>& imagePaths,
                                     const StitchCallbacks& callbacks)
{
    std::vector<cv::Mat> images;
    images.reserve(imagePaths.size());

    for (std::size_t i = 0; i < imagePaths.size(); ++i) {
        if (isCancelled(callbacks)) {
            emitLog(callbacks, "[信息] 图像加载已取消。");
            return {};
        }

        emitProgress(callbacks, "load", i + 1, imagePaths.size());

        const std::string& path = imagePaths[i];
        cv::Mat image = loadImageFromPath(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            emitLog(callbacks, "[错误] 图像加载失败：" + path);
            return {};
        }

        images.push_back(image);
    }

    emitLog(callbacks, "[信息] 已加载 " + std::to_string(images.size()) + " 张图像。");
    return images;
}

std::vector<EdgeVariants> preprocessAllImages(const std::vector<cv::Mat>& images,
                                              const EdgeDetectConfig& cfg,
                                              const StitchCallbacks& callbacks)
{
    std::vector<EdgeVariants> edgesList;
    edgesList.reserve(images.size());

    SubpixelEdgeDetector detector;
    detector.setCannyThresholds(cfg.cannyLow, cfg.cannyHigh);

    for (std::size_t i = 0; i < images.size(); ++i) {
        if (isCancelled(callbacks)) {
            emitLog(callbacks, "[信息] 预处理已取消。");
            return {};
        }

        emitProgress(callbacks, "preprocess", i + 1, images.size());
        emitLog(callbacks,
                "[预处理] 图像 " + std::to_string(i + 1) + "/" + std::to_string(images.size()));
        const auto preprocessBegin = std::chrono::steady_clock::now();

        EdgeVariants variants = buildEdgeVariants(detector, images[i], cfg);
        const auto preprocessEnd = std::chrono::steady_clock::now();
        const double preprocessSeconds =
            std::chrono::duration<double>(preprocessEnd - preprocessBegin).count();
        if (variants.raw.size() < cfg.minWarnPoints) {
            emitLog(callbacks,
                    "    [警告] 边缘点数量过少：" + std::to_string(variants.raw.size()));
        }
        emitLog(callbacks,
                "    [Timing] preprocess image " + std::to_string(i + 1) +
                    " runtime = " + std::to_string(preprocessSeconds) + " s");
        emitLog(callbacks,
                "    [PreprocessDetail] " + variants.preprocessingMode);

        edgesList.push_back(std::move(variants));
    }

    return edgesList;
}

bool saveImageToPath(const std::string& pathUtf8, const cv::Mat& image)
{
    if (image.empty()) {
        return false;
    }

    const std::filesystem::path filePath = pathFromUtf8(pathUtf8);
    if (filePath.has_parent_path()) {
        std::filesystem::create_directories(filePath.parent_path());
    }

    std::vector<unsigned char> encoded;
    if (!cv::imencode(normalizedImageExtension(filePath), image, encoded)) {
        return false;
    }

    std::ofstream stream(filePath, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    stream.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    return stream.good();
}

bool writeTextFileToPath(const std::string& pathUtf8, const std::string& content)
{
    const std::filesystem::path filePath = pathFromUtf8(pathUtf8);
    if (filePath.has_parent_path()) {
        std::filesystem::create_directories(filePath.parent_path());
    }

    std::ofstream stream(filePath, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    stream << content;
    return stream.good();
}

} // namespace stitch
