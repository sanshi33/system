#include "acquisition/CameraCaptureService.h"

#include "acquisition/GalaxyCameraDevice.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pinjie {
namespace {

bool isCancelled(const CameraCaptureCallbacks& callbacks)
{
    return callbacks.isCancelled && callbacks.isCancelled();
}

void emitLog(const CameraCaptureCallbacks& callbacks, const std::string& message)
{
    if (callbacks.onLog) {
        callbacks.onLog(message);
    }
}

void emitProgress(const CameraCaptureCallbacks& callbacks, const std::size_t current, const std::size_t total)
{
    if (callbacks.onProgress) {
        callbacks.onProgress(current, total);
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

std::string genericUtf8String(const std::filesystem::path& path)
{
    return path.generic_u8string();
}

std::string normalizedExtension(std::string extension)
{
    if (extension.empty()) {
        return ".bmp";
    }
    if (extension.front() != '.') {
        extension.insert(extension.begin(), '.');
    }
    return extension;
}

std::string currentCaptureTimeIso()
{
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y-%m-%dT%H:%M:%S")
           << '.'
           << std::setw(3)
           << std::setfill('0')
           << millis.count();
    return stream.str();
}

std::filesystem::path framePath(const CameraSequenceRequest& request, const int index)
{
    std::ostringstream name;
    name << request.filePrefix << index << normalizedExtension(request.extension);
    return pathFromUtf8(request.outputDir) / name.str();
}

std::filesystem::path manifestPath(const CameraSequenceRequest& request)
{
    return pathFromUtf8(request.outputDir) / request.manifestFileName;
}

bool saveImageToPath(const std::filesystem::path& path, const cv::Mat& image)
{
    if (image.empty()) {
        return false;
    }

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::vector<unsigned char> encoded;
    if (!cv::imencode(path.extension().string(), image, encoded)) {
        return false;
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    stream.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    return stream.good();
}

std::string csvEscape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

std::string formatCsvDouble(const double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

bool appendManifestRow(const std::filesystem::path& path, const ImageFrameInfo& frame)
{
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    const bool exists = std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::app | std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }

    if (!exists) {
        stream << "index,step_index,file_name,file_path,capture_time,serial_number,session_name,stage_x,stage_y,"
                  "has_stage_position,exposure_us,gain,note\n";
    }

    stream << frame.index << ','
           << frame.stepIndex << ','
           << csvEscape(frame.fileName) << ','
           << csvEscape(frame.filePath) << ','
           << csvEscape(frame.captureTimeIso) << ','
           << csvEscape(frame.serialNumber) << ','
           << csvEscape(frame.sessionName) << ','
           << formatCsvDouble(frame.stageX) << ','
           << formatCsvDouble(frame.stageY) << ','
           << (frame.hasStagePosition ? 1 : 0) << ','
           << formatCsvDouble(frame.exposure) << ','
           << formatCsvDouble(frame.gain) << ','
           << csvEscape(frame.note) << '\n';
    return stream.good();
}

void populateFrameInfo(CameraFrame& frame,
                       const CameraSequenceRequest& request,
                       const int fileIndex,
                       const std::filesystem::path& outputPath)
{
    frame.info.index = static_cast<std::size_t>(fileIndex);
    frame.info.filePath = genericUtf8String(outputPath);
    frame.info.fileName = outputPath.filename().generic_u8string();
    frame.info.stageX = request.stageX;
    frame.info.stageY = request.stageY;
    frame.info.hasStagePosition = request.hasStagePosition;
    frame.info.gain = request.config.setGain ? request.config.gain : frame.info.gain;
    if (request.config.setExposureTime) {
        frame.info.exposure = request.config.exposureTimeUs;
    }
    frame.info.stepIndex =
        request.mode == CameraCaptureMode::ManualStepScan && request.stepIndex > 0 ? request.stepIndex
                                                                                   : static_cast<std::size_t>(fileIndex);
    frame.info.captureTimeIso = currentCaptureTimeIso();
    frame.info.serialNumber = request.serialNumber;
    frame.info.sessionName = request.sessionName;
    frame.info.note = request.note;
}

CameraSequenceResult captureSequenceImpl(const CameraSequenceRequest& request)
{
    CameraSequenceResult result;

    if (request.outputDir.empty()) {
        result.message = "camera capture output directory is empty";
        return result;
    }
    if (request.frameCount <= 0) {
        result.message = "camera capture frame count must be positive";
        return result;
    }

    try {
        GalaxyCameraDevice camera;
        camera.open(request.serialNumber);
        camera.applyConfig(request.config);
        camera.startGrabbing();
        emitLog(request.callbacks, "camera capture started");

        const std::filesystem::path manifestOutputPath = manifestPath(request);
        result.manifestPath = genericUtf8String(manifestOutputPath);
        result.frames.reserve(static_cast<std::size_t>(request.frameCount));
        for (int i = 0; i < request.frameCount; ++i) {
            if (isCancelled(request.callbacks)) {
                result.cancelled = true;
                result.message = "camera capture cancelled";
                emitLog(request.callbacks, result.message);
                break;
            }

            if (request.config.triggerEnabled && request.config.triggerSource == "Software") {
                camera.softwareTrigger();
            }

            CameraFrame frame = camera.grabFrame(request.timeoutMs);
            const int fileIndex = request.startIndex + i;
            const std::filesystem::path outputPath = framePath(request, fileIndex);
            if (!saveImageToPath(outputPath, frame.image)) {
                throw std::runtime_error("failed to save captured frame: " + genericUtf8String(outputPath));
            }

            populateFrameInfo(frame, request, fileIndex, outputPath);
            if (request.writeManifest && !appendManifestRow(manifestOutputPath, frame.info)) {
                throw std::runtime_error("failed to update capture manifest: " + genericUtf8String(manifestOutputPath));
            }

            if (request.callbacks.onFrameCaptured) {
                request.callbacks.onFrameCaptured(frame.info, frame.image);
            }
            result.frames.push_back(std::move(frame.info));
            emitProgress(request.callbacks, result.frames.size(), static_cast<std::size_t>(request.frameCount));

            if (request.intervalMs > 0 && i + 1 < request.frameCount) {
                std::this_thread::sleep_for(std::chrono::milliseconds(request.intervalMs));
            }
        }

        camera.stopGrabbing();
        camera.close();

        result.ok = !result.cancelled;
        switch (request.mode) {
        case CameraCaptureMode::Single:
            if (!result.cancelled) {
                result.message = "single camera capture completed";
            }
            break;
        case CameraCaptureMode::ManualStepScan:
            if (!result.cancelled) {
                result.message = "manual step camera capture completed";
            }
            break;
        case CameraCaptureMode::Sequence:
        default:
            if (!result.cancelled) {
                result.message = "camera capture completed";
            }
            break;
        }
        emitLog(request.callbacks, result.message);
        return result;
    } catch (std::exception& error) {
        result.ok = false;
        result.message = error.what();
        emitLog(request.callbacks, std::string("camera capture failed: ") + result.message);
        return result;
    }
}

} // namespace

CameraSequenceResult CameraCaptureService::captureSingle(const CameraSequenceRequest& request) const
{
    CameraSequenceRequest singleRequest = request;
    singleRequest.mode = CameraCaptureMode::Single;
    singleRequest.frameCount = 1;
    singleRequest.intervalMs = 0;
    return captureSequenceImpl(singleRequest);
}

CameraSequenceResult CameraCaptureService::captureManualStep(const CameraSequenceRequest& request) const
{
    CameraSequenceRequest singleRequest = request;
    singleRequest.mode = CameraCaptureMode::ManualStepScan;
    singleRequest.frameCount = 1;
    singleRequest.intervalMs = 0;
    if (singleRequest.stepIndex > 0) {
        singleRequest.startIndex = static_cast<int>(singleRequest.stepIndex);
    }
    return captureSequenceImpl(singleRequest);
}

CameraSequenceResult CameraCaptureService::captureSequence(const CameraSequenceRequest& request) const
{
    CameraSequenceRequest sequenceRequest = request;
    sequenceRequest.mode = CameraCaptureMode::Sequence;
    return captureSequenceImpl(sequenceRequest);
}

} // namespace pinjie
