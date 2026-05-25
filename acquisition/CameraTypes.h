#pragma once

#include "common/ImageFrameInfo.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pinjie {

enum class CameraCaptureMode {
    Single,
    Sequence,
    ManualStepScan
};

struct CameraDeviceInfo {
    std::string vendorName;
    std::string modelName;
    std::string serialNumber;
    std::string displayName;
    std::string userId;
    std::string ipAddress;
    std::string macAddress;
    std::string deviceClass;
    bool readableAndWritable{false};
};

struct CameraCaptureConfig {
    double exposureTimeUs{0.0};
    double gain{0.0};
    bool setExposureTime{false};
    bool setGain{false};
    bool triggerEnabled{false};
    std::string triggerSource{"Software"};
    std::string acquisitionMode{"Continuous"};
};

struct CameraFrame {
    cv::Mat image;
    std::uint64_t frameId{0};
    std::uint64_t timestamp{0};
    std::uint64_t width{0};
    std::uint64_t height{0};
    std::string pixelFormat;
    ImageFrameInfo info;
};

using CameraCaptureLogCallback = std::function<void(const std::string&)>;
using CameraCaptureProgressCallback = std::function<void(std::size_t current, std::size_t total)>;
using CameraCaptureFrameCallback = std::function<void(const ImageFrameInfo&, const cv::Mat&)>;
using CameraCaptureCancelCallback = std::function<bool()>;

struct CameraCaptureCallbacks {
    CameraCaptureLogCallback onLog;
    CameraCaptureProgressCallback onProgress;
    CameraCaptureFrameCallback onFrameCaptured;
    CameraCaptureCancelCallback isCancelled;
};

struct CameraSequenceRequest {
    CameraCaptureMode mode{CameraCaptureMode::Sequence};
    std::string sessionName{"camera_capture"};
    std::string serialNumber;
    std::string outputDir;
    std::string filePrefix{"Pic_"};
    std::string extension{".bmp"};
    int frameCount{1};
    int startIndex{1};
    int timeoutMs{1000};
    int intervalMs{0};
    std::size_t stepIndex{0};
    double stageX{0.0};
    double stageY{0.0};
    bool hasStagePosition{false};
    bool writeManifest{true};
    std::string manifestFileName{"manifest.csv"};
    std::string note;
    CameraCaptureConfig config;
    CameraCaptureCallbacks callbacks;
};

struct CameraSequenceResult {
    bool ok{false};
    bool cancelled{false};
    std::string message;
    std::string manifestPath;
    std::vector<ImageFrameInfo> frames;
};

} // namespace pinjie
