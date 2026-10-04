// Isolated prototype: exact order-statistic median on finite float32 pixels.
// Replicate border matches cv::medianBlur; no quantization, fast math or FMA.
#include <cuda_runtime.h>
#include <cstddef>
#include <new>
struct Workspace { float *input=nullptr, *output=nullptr; size_t size=0; };
__global__ void median5(const float* src, float* dst, int h, int w) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    float a[25];
    #pragma unroll
    for(int dy=-2;dy<=2;dy++) {
        #pragma unroll
        for(int dx=-2;dx<=2;dx++) {
            int xx=max(0,min(w-1,x+dx)), yy=max(0,min(h-1,y+dy));
            a[(dy+2)*5+dx+2]=src[yy*w+xx];
        }
    }
    // Fixed odd-even sorting network: comparisons select existing float bits.
    #pragma unroll
    for(int p=0;p<25;p++) {
        #pragma unroll
        for(int j=(p&1);j<24;j+=2) {
            float lo=fminf(a[j],a[j+1]), hi=fmaxf(a[j],a[j+1]);
            a[j]=lo; a[j+1]=hi;
        }
    }
    dst[y*w+x]=a[12];
}
extern "C" void* seaqr_create() { return new(std::nothrow) Workspace; }
extern "C" int seaqr_median(void* ptr, const float* src, float* dst, int h, int w) {
    if(!ptr || !src || !dst || h<=0 || w<=0) return (int)cudaErrorInvalidValue;
    auto *p=static_cast<Workspace*>(ptr);
    size_t bytes=size_t(h)*size_t(w)*sizeof(float);
    cudaError_t e;
    if(p->size != bytes) {
        if(p->input) cudaFree(p->input);
        if(p->output) cudaFree(p->output);
        p->input=nullptr;p->output=nullptr;p->size=0;
        e=cudaMalloc(&p->input,bytes); if(e!=cudaSuccess) return e;
        e=cudaMalloc(&p->output,bytes); if(e!=cudaSuccess) return e;
        p->size=bytes;
    }
    e=cudaMemcpy(p->input,src,bytes,cudaMemcpyHostToDevice); if(e!=cudaSuccess) return e;
    median5<<<dim3((w+31)/32,(h+7)/8),dim3(32,8)>>>(p->input,p->output,h,w);
    e=cudaGetLastError(); if(e!=cudaSuccess) return e;
    // Blocking transfer synchronizes the kernel; benchmark includes both copies.
    return cudaMemcpy(dst,p->output,bytes,cudaMemcpyDeviceToHost);
}
extern "C" void seaqr_destroy(void* ptr) {
    auto *p=static_cast<Workspace*>(ptr);
    if(!p)return;
    if(p->input)cudaFree(p->input);
    if(p->output)cudaFree(p->output);
    delete p;
}
