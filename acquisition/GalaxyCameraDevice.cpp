#include "acquisition/GalaxyCameraDevice.h"

#include <GalaxyIncludes.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pinjie {
namespace {

#ifndef PINJIE_GALAXY_SDK_ROOT
#define PINJIE_GALAXY_SDK_ROOT "E:/GalaxySDK"
#endif

std::string gxToString(const GxIAPICPP::gxstring& value)
{
    return value.c_str() ? std::string(value.c_str()) : std::string();
}

std::string deviceClassName(const GX_DEVICE_CLASS_LIST deviceClass)
{
    switch (deviceClass) {
    case GX_DEVICE_CLASS_USB2:
        return "USB2";
    case GX_DEVICE_CLASS_GEV:
        return "GigE";
    case GX_DEVICE_CLASS_U3V:
        return "USB3";
    case GX_DEVICE_CLASS_SMART:
        return "Smart";
    case GX_DEVICE_CLASS_CXP:
        return "CXP";
    default:
        return "Unknown";
    }
}

std::string pixelFormatName(const GX_PIXEL_FORMAT_ENTRY format)
{
    switch (format) {
    case GX_PIXEL_FORMAT_MONO8:
        return "Mono8";
    case GX_PIXEL_FORMAT_MONO10:
        return "Mono10";
    case GX_PIXEL_FORMAT_MONO12:
        return "Mono12";
    case GX_PIXEL_FORMAT_MONO14:
        return "Mono14";
    case GX_PIXEL_FORMAT_MONO16:
        return "Mono16";
    case GX_PIXEL_FORMAT_BGR8:
        return "BGR8";
    case GX_PIXEL_FORMAT_RGB8:
        return "RGB8";
    case GX_PIXEL_FORMAT_BAYER_GR8:
        return "BayerGR8";
    case GX_PIXEL_FORMAT_BAYER_RG8:
        return "BayerRG8";
    case GX_PIXEL_FORMAT_BAYER_GB8:
        return "BayerGB8";
    case GX_PIXEL_FORMAT_BAYER_BG8:
        return "BayerBG8";
    default:
        return "0x" + std::to_string(static_cast<unsigned int>(format));
    }
}

CameraDeviceInfo toCameraDeviceInfo(const CGXDeviceInfo& info)
{
    CameraDeviceInfo result;
    result.vendorName = gxToString(info.GetVendorName());
    result.modelName = gxToString(info.GetModelName());
    result.serialNumber = gxToString(info.GetSN());
    result.displayName = gxToString(info.GetDisplayName());
    result.userId = gxToString(info.GetUserID());
    result.ipAddress = gxToString(info.GetIP());
    result.macAddress = gxToString(info.GetMAC());
    result.deviceClass = deviceClassName(info.GetDeviceClass());
    result.readableAndWritable = info.GetAccessStatus() == GX_ACCESS_STATUS_READWRITE;
    return result;
}

std::runtime_error galaxyError(const std::string& context, CGalaxyException& error)
{
    return std::runtime_error(context + ": [" + std::to_string(error.GetErrorCode()) + "] " + error.what());
}

std::string summarizeDevices(const std::vector<CameraDeviceInfo>& devices)
{
    if (devices.empty()) {
        return "none";
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        const CameraDeviceInfo& device = devices[index];
        if (index > 0) {
            stream << "; ";
        }

        stream << (device.modelName.empty() ? "UnknownModel" : device.modelName)
               << " SN=" << (device.serialNumber.empty() ? "<empty>" : device.serialNumber)
               << " access=" << (device.readableAndWritable ? "rw" : "limited");
    }
    return stream.str();
}

double clampToFeatureRange(CFloatFeaturePointer feature, const double value)
{
    return std::max(feature->GetMin(), std::min(feature->GetMax(), value));
}

void setEnumIfWritable(CGXFeatureControlPointer featureControl,
                       const char* featureName,
                       const std::string& value)
{
    if (!featureControl->IsImplemented(featureName) || !featureControl->IsWritable(featureName)) {
        return;
    }
    featureControl->GetEnumFeature(featureName)->SetValue(value.c_str());
}

void setFloatIfWritable(CGXFeatureControlPointer featureControl,
                        const char* featureName,
                        const double value)
{
    if (!featureControl->IsImplemented(featureName) || !featureControl->IsWritable(featureName)) {
        return;
    }
    CFloatFeaturePointer feature = featureControl->GetFloatFeature(featureName);
    feature->SetValue(clampToFeatureRange(feature, value));
}

class GalaxySdkRuntime {
public:
    GalaxySdkRuntime()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (refCount_ == 0) {
            configureGalaxyProcessEnvironment();
            IGXFactory::GetInstance().Init();
        }
        ++refCount_;
    }

    ~GalaxySdkRuntime()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (refCount_ > 0) {
            --refCount_;
        }
        if (refCount_ == 0) {
            try {
                IGXFactory::GetInstance().Uninit();
            } catch (...) {
            }
        }
    }

private:
    static void prependEnvPath(const char* name, const std::string& path)
    {
        if (path.empty()) {
            return;
        }

        char* current = nullptr;
        std::size_t length = 0;
        std::string value = path;
        if (_dupenv_s(&current, &length, name) == 0 && current != nullptr && length > 0) {
            const std::string existing(current);
            if (existing.find(path) == std::string::npos) {
                value += ";" + existing;
            } else {
                value = existing;
            }
        }
        if (current != nullptr) {
            free(current);
        }
        _putenv_s(name, value.c_str());
    }

    static void configureGalaxyProcessEnvironment()
    {
        const std::string root = PINJIE_GALAXY_SDK_ROOT;
        prependEnvPath("PATH", root + "/APIDll/Win64");
        prependEnvPath("PATH", root + "/GenICam/bin/Win64_x64");
        prependEnvPath("GENICAM_GENTL64_PATH", root + "/GenTL/Win64");
        _putenv_s("GALAXY_GENICAM_ROOT", (root + "/GenICam").c_str());
    }

    static std::mutex mutex_;
    static int refCount_;
};

std::mutex GalaxySdkRuntime::mutex_;
int GalaxySdkRuntime::refCount_ = 0;

} // namespace

class GalaxyCameraDevice::Impl {
public:
    Impl()
        : runtime_(std::make_shared<GalaxySdkRuntime>())
    {
    }

    std::vector<CameraDeviceInfo> enumerateDevices(const int timeoutMs)
    {
        try {
            GxIAPICPP::gxdeviceinfo_vector devices;
            IGXFactory::GetInstance().UpdateDeviceList(static_cast<uint32_t>(std::max(0, timeoutMs)), devices);

            std::vector<CameraDeviceInfo> result;
            result.reserve(devices.size());
            for (const auto& device : devices) {
                result.push_back(toCameraDeviceInfo(device));
            }
            return result;
        } catch (CGalaxyException& error) {
            throw galaxyError("enumerate Galaxy cameras", error);
        } catch (std::exception& error) {
            throw std::runtime_error(std::string("enumerate Galaxy cameras: ") + error.what());
        }
    }

    void open(const std::string& serialNumber)
    {
        close();

        try {
            const std::string selectedSerial = resolveSerialNumber(serialNumber);
            try {
                device_ = IGXFactory::GetInstance().OpenDeviceBySN(selectedSerial.c_str(), GX_ACCESS_EXCLUSIVE);
            } catch (CGalaxyException&) {
                device_ = IGXFactory::GetInstance().OpenDeviceBySN(selectedSerial.c_str(), GX_ACCESS_CONTROL);
            }
            remoteControl_ = device_->GetRemoteFeatureControl();
            currentInfo_ = toCameraDeviceInfo(device_->GetDeviceInfo());

            if (device_->GetStreamCount() == 0) {
                throw std::runtime_error("camera has no image stream");
            }

            stream_ = device_->OpenStream(0);
            streamControl_ = stream_->GetFeatureControl();

            try {
                setEnumIfWritable(remoteControl_, "UserSetSelector", "Default");
                if (remoteControl_->IsImplemented("UserSetLoad") && remoteControl_->IsWritable("UserSetLoad")) {
                    remoteControl_->GetCommandFeature("UserSetLoad")->Execute();
                }
            } catch (...) {
            }

            setEnumIfWritable(remoteControl_, "AcquisitionMode", "Continuous");
            setEnumIfWritable(remoteControl_, "TriggerMode", "Off");
        } catch (CGalaxyException& error) {
            close();
            std::vector<CameraDeviceInfo> devices;
            try {
                devices = enumerateDevices(1000);
            } catch (...) {
            }
            throw std::runtime_error("open Galaxy camera failed. serial=" + serialNumber +
                                     ", available devices: " + summarizeDevices(devices) + ", sdk error: [" +
                                     std::to_string(error.GetErrorCode()) + "] " + error.what());
        } catch (...) {
            close();
            throw;
        }
    }

    void close() noexcept
    {
        stopGrabbing();

        try {
            if (!stream_.IsNull()) {
                stream_->Close();
            }
        } catch (...) {
        }

        try {
            if (!device_.IsNull()) {
                device_->Close();
            }
        } catch (...) {
        }

        stream_ = CGXStreamPointer();
        streamControl_ = CGXFeatureControlPointer();
        remoteControl_ = CGXFeatureControlPointer();
        device_ = CGXDevicePointer();
        currentInfo_ = {};
    }

    bool isOpen() const noexcept
    {
        return !device_.IsNull();
    }

    CameraDeviceInfo currentDeviceInfo() const
    {
        requireOpen();
        return currentInfo_;
    }

    void applyConfig(const CameraCaptureConfig& config)
    {
        requireOpen();

        try {
            if (!config.acquisitionMode.empty()) {
                setEnumIfWritable(remoteControl_, "AcquisitionMode", config.acquisitionMode);
            }

            if (remoteControl_->IsImplemented("TriggerMode") && remoteControl_->IsWritable("TriggerMode")) {
                remoteControl_->GetEnumFeature("TriggerMode")->SetValue(config.triggerEnabled ? "On" : "Off");
            }
            if (config.triggerEnabled && !config.triggerSource.empty()) {
                setEnumIfWritable(remoteControl_, "TriggerSource", config.triggerSource);
            }

            if (config.setExposureTime) {
                setFloatIfWritable(remoteControl_, "ExposureTime", config.exposureTimeUs);
            }
            if (config.setGain) {
                setFloatIfWritable(remoteControl_, "Gain", config.gain);
            }
        } catch (CGalaxyException& error) {
            throw galaxyError("apply Galaxy camera config", error);
        }
    }

    void startGrabbing()
    {
        requireOpen();
        if (grabbing_) {
            return;
        }

        try {
            if (!streamControl_.IsNull()) {
                try {
                    setEnumIfWritable(streamControl_, "StreamBufferHandlingMode", "OldestFirst");
                } catch (...) {
                }
            }

            stream_->StartGrab();
            remoteControl_->GetCommandFeature("AcquisitionStart")->Execute();
            grabbing_ = true;
        } catch (CGalaxyException& error) {
            throw galaxyError("start Galaxy camera acquisition", error);
        }
    }

    void stopGrabbing() noexcept
    {
        if (!grabbing_) {
            return;
        }

        try {
            if (!remoteControl_.IsNull() && remoteControl_->IsImplemented("AcquisitionStop")) {
                remoteControl_->GetCommandFeature("AcquisitionStop")->Execute();
            }
        } catch (...) {
        }

        try {
            if (!stream_.IsNull()) {
                stream_->StopGrab();
            }
        } catch (...) {
        }

        grabbing_ = false;
    }

    void softwareTrigger()
    {
        requireOpen();
        try {
            remoteControl_->GetCommandFeature("TriggerSoftware")->Execute();
        } catch (CGalaxyException& error) {
            throw galaxyError("execute Galaxy software trigger", error);
        }
    }

    CameraFrame grabFrame(const int timeoutMs)
    {
        requireOpen();
        if (!grabbing_) {
            startGrabbing();
        }

        CImageDataPointer imageData;
        try {
            imageData = stream_->DQBuf(static_cast<uint32_t>(std::max(0, timeoutMs)));
            if (imageData->GetStatus() != GX_FRAME_STATUS_SUCCESS) {
                stream_->QBuf(imageData);
                throw std::runtime_error("camera returned incomplete frame");
            }

            CameraFrame frame;
            frame.frameId = imageData->GetFrameID();
            frame.timestamp = imageData->GetTimeStamp();
            frame.width = imageData->GetWidth();
            frame.height = imageData->GetHeight();
            frame.pixelFormat = pixelFormatName(imageData->GetPixelFormat());
            frame.image = convertFrame(imageData);
            frame.info.index = static_cast<std::size_t>(frame.frameId);
            frame.info.exposure = readFloatFeature("ExposureTime");
            frame.info.note = frame.pixelFormat;

            stream_->QBuf(imageData);
            return frame;
        } catch (CGalaxyException& error) {
            if (!imageData.IsNull()) {
                try {
                    stream_->QBuf(imageData);
                } catch (...) {
                }
            }
            throw galaxyError("grab Galaxy camera frame", error);
        } catch (...) {
            if (!imageData.IsNull()) {
                try {
                    stream_->QBuf(imageData);
                } catch (...) {
                }
            }
            throw;
        }
    }

private:
    std::string resolveSerialNumber(const std::string& serialNumber)
    {
        std::vector<CameraDeviceInfo> devices = enumerateDevices(1000);
        if (devices.empty()) {
            throw std::runtime_error("no Galaxy camera was found");
        }

        if (!serialNumber.empty()) {
            const auto selected = std::find_if(devices.begin(), devices.end(), [&serialNumber](const CameraDeviceInfo& device) {
                return device.serialNumber == serialNumber;
            });
            if (selected == devices.end()) {
                throw std::runtime_error("selected Galaxy camera was not found. serial=" + serialNumber +
                                         ", available devices: " + summarizeDevices(devices));
            }
            return selected->serialNumber;
        }

        const auto preferred = std::find_if(devices.begin(), devices.end(), [](const CameraDeviceInfo& device) {
            return !device.serialNumber.empty();
        });
        if (preferred != devices.end()) {
            return preferred->serialNumber;
        }

        throw std::runtime_error("Galaxy cameras were detected, but no valid serial number was available. available devices: " +
                                 summarizeDevices(devices));
    }

    void requireOpen() const
    {
        if (device_.IsNull() || remoteControl_.IsNull() || stream_.IsNull()) {
            throw std::runtime_error("Galaxy camera is not open");
        }
    }

    double readFloatFeature(const char* featureName)
    {
        try {
            if (!remoteControl_.IsNull() && remoteControl_->IsImplemented(featureName) &&
                remoteControl_->IsReadable(featureName)) {
                return remoteControl_->GetFloatFeature(featureName)->GetValue();
            }
        } catch (...) {
        }
        return 0.0;
    }

    cv::Mat convertFrame(CImageDataPointer imageData)
    {
        const auto width = static_cast<int>(imageData->GetWidth());
        const auto height = static_cast<int>(imageData->GetHeight());
        const GX_PIXEL_FORMAT_ENTRY format = imageData->GetPixelFormat();

        if (format == GX_PIXEL_FORMAT_MONO8) {
            cv::Mat view(height, width, CV_8UC1, imageData->GetBuffer());
            return view.clone();
        }

        if (format == GX_PIXEL_FORMAT_MONO10 || format == GX_PIXEL_FORMAT_MONO12 ||
            format == GX_PIXEL_FORMAT_MONO14 || format == GX_PIXEL_FORMAT_MONO16 ||
            format == GX_PIXEL_FORMAT_MONO10_P || format == GX_PIXEL_FORMAT_MONO12_P ||
            format == GX_PIXEL_FORMAT_MONO10_PACKED || format == GX_PIXEL_FORMAT_MONO12_PACKED) {
            void* raw8 = imageData->ConvertToRaw8(GX_BIT_4_11);
            cv::Mat view(height, width, CV_8UC1, raw8);
            return view.clone();
        }

        if (format == GX_PIXEL_FORMAT_BGR8) {
            cv::Mat view(height, width, CV_8UC3, imageData->GetBuffer());
            return view.clone();
        }

        if (format == GX_PIXEL_FORMAT_RGB8) {
            cv::Mat view(height, width, CV_8UC3, imageData->GetBuffer());
            cv::Mat bgr;
            cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
            return bgr;
        }

        CGXImageFormatConvertPointer converter = IGXFactory::GetInstance().CreateImageFormatConvert();
        converter->SetDstFormat(GX_PIXEL_FORMAT_BGR8);
        converter->SetInterpolationType(GX_RAW2RGB_NEIGHBOUR);
        converter->SetValidBits(GX_BIT_4_11);

        std::vector<unsigned char> buffer(static_cast<std::size_t>(converter->GetBufferSizeForConversion(imageData)));
        converter->Convert(imageData, buffer.data(), buffer.size(), false);

        cv::Mat view(height, width, CV_8UC3, buffer.data());
        return view.clone();
    }

    std::shared_ptr<GalaxySdkRuntime> runtime_;
    CGXDevicePointer device_;
    CGXFeatureControlPointer remoteControl_;
    CGXStreamPointer stream_;
    CGXFeatureControlPointer streamControl_;
    CameraDeviceInfo currentInfo_;
    bool grabbing_{false};
};

GalaxyCameraDevice::GalaxyCameraDevice()
    : impl_(std::make_unique<Impl>())
{
}

GalaxyCameraDevice::~GalaxyCameraDevice() = default;

std::vector<CameraDeviceInfo> GalaxyCameraDevice::enumerateDevices(const int timeoutMs)
{
    return impl_->enumerateDevices(timeoutMs);
}

void GalaxyCameraDevice::open(const std::string& serialNumber)
{
    impl_->open(serialNumber);
}

void GalaxyCameraDevice::close() noexcept
{
    impl_->close();
}

bool GalaxyCameraDevice::isOpen() const noexcept
{
    return impl_->isOpen();
}

CameraDeviceInfo GalaxyCameraDevice::currentDeviceInfo() const
{
    return impl_->currentDeviceInfo();
}

void GalaxyCameraDevice::applyConfig(const CameraCaptureConfig& config)
{
    impl_->applyConfig(config);
}

void GalaxyCameraDevice::startGrabbing()
{
    impl_->startGrabbing();
}

void GalaxyCameraDevice::stopGrabbing() noexcept
{
    impl_->stopGrabbing();
}

void GalaxyCameraDevice::softwareTrigger()
{
    impl_->softwareTrigger();
}

CameraFrame GalaxyCameraDevice::grabFrame(const int timeoutMs)
{
    return impl_->grabFrame(timeoutMs);
}

} // namespace pinjie
