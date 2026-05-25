#pragma once

#include "acquisition/CameraTypes.h"

#include <string>
#include <vector>

namespace pinjie {

class ICameraDevice {
public:
    virtual ~ICameraDevice() = default;

    virtual std::vector<CameraDeviceInfo> enumerateDevices(int timeoutMs = 1000) = 0;
    virtual void open(const std::string& serialNumber) = 0;
    virtual void close() noexcept = 0;
    virtual bool isOpen() const noexcept = 0;

    virtual CameraDeviceInfo currentDeviceInfo() const = 0;
    virtual void applyConfig(const CameraCaptureConfig& config) = 0;
    virtual void startGrabbing() = 0;
    virtual void stopGrabbing() noexcept = 0;
    virtual void softwareTrigger() = 0;
    virtual CameraFrame grabFrame(int timeoutMs = 1000) = 0;
};

} // namespace pinjie
