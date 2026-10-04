// Diagnostic-only build. Original kernels stay byte-for-byte source-identical.
// Events add overhead: these are copy intervals and host API waits, NOT FPS.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace {
constexpr int capacity = 64;
struct Record {
    int source=0, line=0, kind=0;
    uint64_t calls=0, bytes=0;
    double host_ms=0, event_ms=0;
};
Record records[capacity];
int used=0;
bool enabled=false;
int source_id(const char* file) {
    if (std::strstr(file,"phase20_cuda_warp_exact.cu")) return 1;
    if (std::strstr(file,"phase20_cuda_resident.cu")) return 2;
    if (std::strstr(file,"phase20_cuda_integrated.cu")) return 3;
    if (std::strstr(file,"phase20_cuda_median.cu")) return 4;
    return 0;
}
Record* slot(const char* file, int line, int kind) {
    const int source=source_id(file);
    for (int i=0; i<used; ++i) if (records[i].source==source && records[i].line==line && records[i].kind==kind) return &records[i];
    if (!source || used==capacity) return nullptr;
    auto& r=records[used++]; r.source=source; r.line=line; r.kind=kind;
    return &r;
}
cudaError_t probe_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, const char* file, int line) {
    if (!enabled) return cudaMemcpy(dst,src,bytes,kind);
    auto* r=slot(file,line,int(kind));
    if (!r) return cudaErrorInvalidValue;
    cudaEvent_t start, end;
    auto e=cudaEventCreate(&start); if (e!=cudaSuccess) return e;
    e=cudaEventCreate(&end);
    if (e!=cudaSuccess) { cudaEventDestroy(start); return e; }
    e=cudaEventRecord(start);
    if (e==cudaSuccess) {
        auto t=std::chrono::steady_clock::now();
        e=cudaMemcpy(dst,src,bytes,kind);
        r->host_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
        ++r->calls; r->bytes+=bytes;
    }
    if (e==cudaSuccess) e=cudaEventRecord(end);
    if (e==cudaSuccess) e=cudaEventSynchronize(end);
    float duration=0;
    if (e==cudaSuccess) e=cudaEventElapsedTime(&duration,start,end);
    r->event_ms+=duration;
    cudaEventDestroy(start); cudaEventDestroy(end);
    return e;
}
cudaError_t probe_sync(const char* file, int line) {
    if (!enabled) return cudaDeviceSynchronize();
    auto* r=slot(file,line,-1);
    if (!r) return cudaErrorInvalidValue;
    auto t=std::chrono::steady_clock::now();
    auto e=cudaDeviceSynchronize();
    r->host_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    ++r->calls;
    return e;
}
}

#define cudaMemcpy(dst,src,bytes,kind) probe_copy(dst,src,bytes,kind,__FILE__,__LINE__)
#define cudaDeviceSynchronize() probe_sync(__FILE__,__LINE__)
#include "phase20_cuda_integrated.cu"
#undef cudaMemcpy
#undef cudaDeviceSynchronize

extern "C" void seaqr_transfer_probe_enable(int value) {
    enabled=value==1;
    if (enabled) { used=0; for (auto& r : records) r=Record{}; }
}
extern "C" int seaqr_transfer_probe_read(int* metadata, uint64_t* calls, uint64_t* bytes, double* host, double* events) {
    if (!metadata || !calls || !bytes || !host || !events) return -1;
    for (int i=0; i<used; ++i) {
        metadata[3*i]=records[i].source; metadata[3*i+1]=records[i].line; metadata[3*i+2]=records[i].kind;
        calls[i]=records[i].calls; bytes[i]=records[i].bytes; host[i]=records[i].host_ms; events[i]=records[i].event_ms;
    }
    return used;
}
