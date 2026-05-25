#pragma once

#include "calibration/CalibrationTypes.h"

#include <memory>
#include <string>

namespace pinjie {

class CameraCalibrationService {
public:
    virtual ~CameraCalibrationService() = default;

    virtual CameraCalibrationResult runCalibration(const CameraCalibrationRequest& request) = 0;
    virtual bool loadCachedResult(const std::string& cacheDir, CalibrationResultCache& cache) = 0;
    virtual bool saveCachedResult(const CalibrationResultCache& cache, const std::string& cacheDir) = 0;
};

using CameraCalibrationServicePtr = std::shared_ptr<CameraCalibrationService>;

CameraCalibrationServicePtr createCameraCalibrationService();

} // namespace pinjie
