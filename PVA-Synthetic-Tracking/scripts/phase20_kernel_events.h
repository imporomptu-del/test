// Diagnostic only. Events bracket launches on the existing default stream;
// no per-kernel synchronize and no changes to device code or launch dimensions.
#include <cuda_runtime.h>
#include <cstdint>

namespace seaqr_kernel_probe {
constexpr int capacity = 1024;
struct Sample { cudaEvent_t first=nullptr, last=nullptr; int id=-1; };
Sample samples[capacity];
int used=0, active=-1;
bool enabled=false, initialized=false;

int begin(int id) {
    if (!enabled) return 0;
    if (!initialized || used>=capacity || active>=0) return int(cudaErrorInvalidValue);
    active=used; samples[used].id=id;
    return int(cudaEventRecord(samples[used].first));
}
int end() {
    if (!enabled) return 0;
    if (active!=used) return int(cudaErrorInvalidValue);
    auto error=cudaEventRecord(samples[used].last);
    active=-1; ++used;
    return int(error);
}
}

extern "C" int seaqr_kernel_probe_enable(int value) {
    using namespace seaqr_kernel_probe;
    if (value!=0 && value!=1) return int(cudaErrorInvalidValue);
    if (value && !initialized) {
        for (auto& sample:samples) {
            auto error=cudaEventCreate(&sample.first);
            if (error==cudaSuccess) error=cudaEventCreate(&sample.last);
            if (error!=cudaSuccess) {
                for (auto& s:samples) {
                    if(s.first) cudaEventDestroy(s.first);
                    if(s.last) cudaEventDestroy(s.last);
                    s.first=s.last=nullptr;
                }
                return int(error);
            }
        }
        initialized=true;
    }
    enabled=value==1;
    if(enabled) { used=0; active=-1; }
    return 0;
}
extern "C" int seaqr_kernel_probe_read(int* ids, float* milliseconds, int length) {
    using namespace seaqr_kernel_probe;
    if(enabled || active>=0 || !ids || !milliseconds || length<used) return -1;
    for(int i=0;i<used;++i) {
        auto error=cudaEventSynchronize(samples[i].last);
        if(error==cudaSuccess) error=cudaEventElapsedTime(milliseconds+i,samples[i].first,samples[i].last);
        if(error!=cudaSuccess) return -1;
        ids[i]=samples[i].id;
    }
    return used;
}
extern "C" void seaqr_kernel_probe_close() {
    using namespace seaqr_kernel_probe;
    enabled=false;
    for(auto& s:samples) {
        if(s.first) cudaEventDestroy(s.first);
        if(s.last) cudaEventDestroy(s.last);
        s.first=s.last=nullptr;
    }
    used=0; active=-1; initialized=false;
}
