#include "app/StitchWorkflowService.h"

namespace pinjie {

StitchRunResult StitchWorkflowService::run(const StitchRunRequest& request,
                                           StitchRunMode runMode,
                                           const TaskCallbacks& callbacks) const
{
    return stitch::runStitching(request, runMode, callbacks);
}

} // namespace pinjie
