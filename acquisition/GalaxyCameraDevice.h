#pragma once

#include "acquisition/ICameraDevice.h"

#include <memory>

namespace pinjie {

class GalaxyCameraDevice final : public ICameraDevice {
public:
    GalaxyCameraDevice();
    ~GalaxyCameraDevice() override;

    GalaxyCameraDevice(const GalaxyCameraDevice&) = delete;
    GalaxyCameraDevice& operator=(const GalaxyCameraDevice&) = delete;

    std::vector<CameraDeviceInfo> enumerateDevices(int timeoutMs = 1000) override;
    void open(const std::string& serialNumber) override;
    void close() noexcept override;
    bool isOpen() const noexcept override;

    CameraDeviceInfo currentDeviceInfo() const override;
    void applyConfig(const CameraCaptureConfig& config) override;
    void startGrabbing() override;
    void stopGrabbing() noexcept override;
    void softwareTrigger() override;
    CameraFrame grabFrame(int timeoutMs = 1000) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pinjie
