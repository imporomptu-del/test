// Diagnostic annotations only: no CUDA calls, state or algorithm changes.
#include <nvtx3/nvToolsExt.h>
extern "C" int seaqr_trace_push(const char* name) { return nvtxRangePushA(name); }
extern "C" int seaqr_trace_pop() { return nvtxRangePop(); }
