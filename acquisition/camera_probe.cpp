#include "acquisition/CameraCaptureService.h"
#include "acquisition/GalaxyCameraDevice.h"

#include <iostream>
#include <string>

namespace {

void printUsage()
{
    std::cout << "Usage:\n"
              << "  pinjie_camera_probe\n"
              << "  pinjie_camera_probe --grab <output_dir> [count]\n"
              << "  pinjie_camera_probe --grab <output_dir> [count] [serial]\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        pinjie::GalaxyCameraDevice camera;
        const auto devices = camera.enumerateDevices(1000);

        std::cout << "Galaxy camera count: " << devices.size() << "\n";
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            std::cout << "[" << i << "] "
                      << device.vendorName << " "
                      << device.modelName << " SN=" << device.serialNumber
                      << " class=" << device.deviceClass
                      << " access=" << (device.readableAndWritable ? "rw" : "limited");
            if (!device.ipAddress.empty()) {
                std::cout << " ip=" << device.ipAddress;
            }
            std::cout << "\n";
        }

        if (argc >= 3 && std::string(argv[1]) == "--grab") {
            pinjie::CameraSequenceRequest request;
            request.outputDir = argv[2];
            request.frameCount = argc >= 4 ? std::stoi(argv[3]) : 1;
            request.serialNumber = argc >= 5 ? argv[4] : std::string();
            request.timeoutMs = 2000;

            pinjie::CameraCaptureService service;
            const pinjie::CameraSequenceResult result = service.captureSequence(request);
            std::cout << result.message << "\n";
            for (const auto& frame : result.frames) {
                std::cout << "saved: " << frame.filePath << "\n";
            }
            return result.ok ? 0 : 2;
        }

        if (argc > 1) {
            printUsage();
        }
        return 0;
    } catch (std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
