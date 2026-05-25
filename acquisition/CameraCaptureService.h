#pragma once

#include "acquisition/CameraTypes.h"

namespace pinjie {

class CameraCaptureService {
public:
    CameraSequenceResult captureSingle(const CameraSequenceRequest& request) const;
    CameraSequenceResult captureManualStep(const CameraSequenceRequest& request) const;
    CameraSequenceResult captureSequence(const CameraSequenceRequest& request) const;
};

} // namespace pinjie
