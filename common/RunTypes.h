#pragma once

#include "common/TaskCallbacks.h"
#include "stitch/StitchService.h"
#include "stitch/StitchTypes.h"

#include <memory>

namespace pinjie {

using StitchRunRequest = stitch::StitchRunRequest;
using StitchRunResult = stitch::StitchRunResult;
using StitchRunMode = stitch::StitchRunMode;
using StitchRunCache = stitch::StitchRunCache;
using StitchRunCachePtr = std::shared_ptr<StitchRunCache>;
using ConstStitchRunCachePtr = std::shared_ptr<const StitchRunCache>;

} // namespace pinjie
