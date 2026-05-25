#pragma once

#include "common/RunTypes.h"

namespace pinjie {

class StitchWorkflowService {
public:
    StitchRunResult run(const StitchRunRequest& request,
                        StitchRunMode runMode = StitchRunMode::Full,
                        const TaskCallbacks& callbacks = {}) const;
};

} // namespace pinjie
