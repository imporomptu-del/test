// Instrumented laboratory build only: timings add synchronization/overhead.
#define seaqr_resident_prepare unprofiled_prepare
#define seaqr_resident_select unprofiled_select
#define seaqr_resident_patches unprofiled_patches
#define seaqr_resident_finish unprofiled_finish
#include "phase20_cuda_resident.cu"
#undef seaqr_resident_prepare
#undef seaqr_resident_select
#undef seaqr_resident_patches
#undef seaqr_resident_finish
#include <chrono>
#undef CHECK
#define CHECK(x) do {int error=int(x);if(error)return error;}while(0)

static double host_ms[16]={},event_ms[16]={};
static int calls[16]={};
template<class F> int timed(int id,F operation) {
    cudaEvent_t a,b;
    CHECK(cudaEventCreate(&a));
    cudaError_t e=cudaEventCreate(&b);
    if(e!=cudaSuccess){cudaEventDestroy(a);return int(e);}
    auto start=std::chrono::steady_clock::now();
    e=cudaEventRecord(a);
    int result=e==cudaSuccess?operation():int(e);
    if(!result)result=int(cudaEventRecord(b));
    if(!result)result=int(cudaEventSynchronize(b));
    float elapsed=0;
    if(!result)result=int(cudaEventElapsedTime(&elapsed,a,b));
    host_ms[id]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    event_ms[id]+=elapsed;calls[id]++;
    cudaEventDestroy(a);cudaEventDestroy(b);return result;
}
extern "C" void seaqr_profile_reset(){for(int i=0;i<16;i++){host_ms[i]=event_ms[i]=0;calls[i]=0;}}
extern "C" void seaqr_profile_read(double* host,double* events,int* count){
    for(int i=0;i<16;i++){host[i]=host_ms[i];events[i]=event_ms[i];count[i]=calls[i];}
}
extern "C" int seaqr_resident_prepare(void* ptr,const float* image,const float* blur,const unsigned char* support,int reset,float floor2,float* samples) {
    if(!ptr)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);size_t bytes=size_t(s.n)*4;
    CHECK(timed(0,[&](){return int(cudaMemcpy(s.image,image,bytes,cudaMemcpyHostToDevice));}));
    CHECK(timed(1,[&](){return int(cudaMemcpy(s.blur,blur,bytes,cudaMemcpyHostToDevice));}));
    CHECK(timed(2,[&](){return int(cudaMemcpy(s.support,support,s.n,cudaMemcpyHostToDevice));}));
    CHECK(timed(3,[&](){median5<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(s.image,s.median,s.h,s.w);return int(cudaGetLastError());}));
    CHECK(timed(4,[&](){residual_prepare<<<(s.n+255)/256,256>>>(s,reset,floor2);return int(cudaGetLastError());}));
    CHECK(timed(5,[&](){gather_samples<<<(s.ns+255)/256,256>>>(s);return int(cudaGetLastError());}));
    CHECK(timed(6,[&](){return int(cudaMemcpy(samples,s.samples,s.ns*4,cudaMemcpyDeviceToHost));}));return 0;
}
extern "C" int seaqr_resident_select(void* ptr,const float* stats,int k,int ready,float threshold,float spatial_threshold,Peak* peaks,int* counts) {
    if(!ptr || k<1 || k>16)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(timed(7,[&](){return int(cudaMemcpy(s.stats,stats,s.tiles*2*4,cudaMemcpyHostToDevice));}));
    CHECK(timed(8,[&](){select_peaks<<<s.tiles*2,256>>>(s,k,ready,threshold,spatial_threshold);return int(cudaGetLastError());}));
    CHECK(timed(9,[&](){return int(cudaMemcpy(peaks,s.peaks,s.tiles*2*k*sizeof(Peak),cudaMemcpyDeviceToHost));}));
    CHECK(timed(10,[&](){return int(cudaMemcpy(counts,s.counts,s.tiles*2*sizeof(int),cudaMemcpyDeviceToHost));}));return 0;
}
extern "C" int seaqr_resident_patches(void* ptr,const int* seeds,int count,float* output) {
    if(!ptr || count<1 || count>512)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(timed(11,[&](){return int(cudaMemcpy(s.seeds,seeds,count*2*sizeof(int),cudaMemcpyHostToDevice));}));
    CHECK(timed(12,[&](){gather_patches<<<(count*289+255)/256,256>>>(s,count);return int(cudaGetLastError());}));
    CHECK(timed(13,[&](){return int(cudaMemcpy(output,s.patches,count*289*4,cudaMemcpyDeviceToHost));}));return 0;
}
extern "C" int seaqr_resident_finish(void* ptr,const unsigned char* learn,float alpha,float noise_alpha,float clip2,float floor2,int variance_only) {
    if(!ptr)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(timed(14,[&](){return int(cudaMemcpy(s.learn,learn,s.n,cudaMemcpyHostToDevice));}));
    CHECK(timed(15,[&](){finish_state<<<(s.n+255)/256,256>>>(s,alpha,noise_alpha,clip2,floor2,variance_only);return int(cudaGetLastError());}));return 0;
}
