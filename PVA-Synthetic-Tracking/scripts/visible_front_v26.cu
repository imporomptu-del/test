// Additive ABI: original integrated kernels and their arithmetic are unchanged.
#include "phase20_cuda_integrated.cu"
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cstdint>

struct FrontV26 {
    Resident* core=nullptr;
    unsigned char* temporary=nullptr;
    double* sigmas=nullptr;
    int* metrics=nullptr; // searchable count, numerical error
    uint32_t* protection=nullptr;
    int2 *points=nullptr,*offsets=nullptr;
    int stride=0;
    void release() {
        seaqr_resident_destroy(core);cudaFree(temporary);cudaFree(sigmas);
        cudaFree(metrics);cudaFree(protection);cudaFree(points);cudaFree(offsets);
    }
};
constexpr int FRONT_POINTS=512*1024,FRONT_OFFSETS=33*33;
extern "C" int seaqr_front_v26_abi(){return 1;}
extern "C" void seaqr_front_v26_destroy(void* ptr) {
    auto*p=static_cast<FrontV26*>(ptr);if(p){p->release();delete p;}
}
extern "C" void* seaqr_front_v26_create(int h,int w,int tile,int stride) {
    if(h<1 || w<1 || h>=32767 || w>=32767 || int64_t(h)*w>32000000 ||
       tile<1 || tile>256 || stride<1 || stride>256 ||
       ((tile+stride-1)/stride)*((tile+stride-1)/stride)>4096)return nullptr;
    auto*p=new(std::nothrow)FrontV26;if(!p)return nullptr;
    // The old gather-samples ABI is not used; retain one valid constructor index.
    int zero=0;p->core=static_cast<Resident*>(seaqr_resident_create(h,w,tile,1,&zero));
    if(!p->core){delete p;return nullptr;}p->stride=stride;auto&s=*p->core;
    #define ALLOC26(field,count) if(cudaMalloc(&p->field,size_t(count)*sizeof(*p->field))!=cudaSuccess){seaqr_front_v26_destroy(p);return nullptr;}
    ALLOC26(temporary,s.n);ALLOC26(sigmas,s.tiles);ALLOC26(metrics,2);
    ALLOC26(protection,(s.n+31)/32);ALLOC26(points,FRONT_POINTS);ALLOC26(offsets,FRONT_OFFSETS);
    #undef ALLOC26
    return p;
}
extern "C" void* seaqr_front_v26_core(void* ptr) {
    return ptr?static_cast<FrontV26*>(ptr)->core:nullptr;
}

__global__ void front_erode(const unsigned char* source,unsigned char* dest,int h,int w,int radius,int axis) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*w)return;
    int y=i/w,x=i%w,p=axis?y:x,n=axis?h:w,step=axis?w:1;
    bool valid=p>=radius && p+radius<n;
    if(valid)for(int k=-radius;k<=radius && valid;k++)valid=source[i+k*step]!=0;
    dest[i]=valid;
}
__global__ void front_eligible(Resident s,int ready,int* metrics) {
    using Reduce=cub::BlockReduce<int,256>;
    __shared__ Reduce::TempStorage temp;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int valid=i<s.n && ready && s.support[i] && s.previous[i];
    if(i<s.n)s.learn[i]=valid; // Reused for the immutable host shape mask until finish.
    int count=Reduce(temp).Sum(valid);
    if(threadIdx.x==0)atomicAdd(metrics,count);
}
__device__ float front_median(const float* sorted,int count) {
    if(!count)return 0.f;
    float high=sorted[count/2];
    if(count&1)return __fadd_rn(0.f,high);
    return __fdiv_rn(__fadd_rn(__fadd_rn(0.f,sorted[count/2-1]),high),2.f);
}
__global__ void front_noise(Resident s,int stride,double floor,double* sigmas,int* metrics) {
    using Sort=cub::BlockRadixSort<float,256,16>;
    using Reduce=cub::BlockReduce<int,256>;
    __shared__ union {typename Sort::TempStorage sort;typename Reduce::TempStorage reduce;float ordered[4096];} tmp;
    __shared__ int count;
    __shared__ float center;
    int tile=blockIdx.x,tid=threadIdx.x,nx=(s.w+s.tile-1)/s.tile;
    int x0=tile%nx*s.tile,y0=tile/nx*s.tile;
    int sw=(min(s.tile,s.w-x0)+stride-1)/stride;
    int sh=(min(s.tile,s.h-y0)+stride-1)/stride;
    float values[16];int used=0;
    #pragma unroll
    for(int k=0;k<16;k++) {
        int j=tid*16+k;float v=CUDART_INF_F;
        if(j<sw*sh) {
            int i=(y0+(j/sw)*stride)*s.w+x0+(j%sw)*stride;
            if(s.support[i]) {
                v=s.temporal[i];++used;
                if(!isfinite(v)){atomicExch(metrics+1,1);v=0.f;}
            }
        }
        values[k]=v;
    }
    int total=Reduce(tmp.reduce).Sum(used);
    if(tid==0)count=total;
    __syncthreads();Sort(tmp.sort).Sort(values);__syncthreads();
    #pragma unroll
    for(int k=0;k<16;k++)tmp.ordered[tid*16+k]=values[k];
    __syncthreads();
    if(tid==0){center=front_median(tmp.ordered,count);s.stats[2*tile]=center;}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<16;k++)values[k]=(tid*16+k<count)?fabsf(__fsub_rn(values[k],center)):CUDART_INF_F;
    Sort(tmp.sort).Sort(values);__syncthreads();
    #pragma unroll
    for(int k=0;k<16;k++)tmp.ordered[tid*16+k]=values[k];
    __syncthreads();
    if(tid==0) {
        float mad=front_median(tmp.ordered,count);
        double sigma=fmax(floor,__dmul_rn(1.4826,double(mad)));
        sigmas[tile]=sigma;s.stats[2*tile+1]=__double2float_rn(sigma);
    }
}

static int front_prepare(FrontV26& f,Resident view,const unsigned char* mask,int radius,
    int reset,int ready,float floor2,double floor,int k,float threshold,float spatial_threshold,
    unsigned char* eligible,double* sigmas,Peak* peaks,int* counts,int* searchable) {
    auto&s=*f.core;
    if(radius<0 || radius>22 || (reset!=0 && reset!=1) || (ready!=0 && ready!=1) ||
       k<1 || k>16 || !isfinite(floor2) || floor2<=0 || !isfinite(floor) || floor<=0 ||
       !eligible || !sigmas || !peaks || !counts || !searchable)return cudaErrorInvalidValue;
    CHECK(cudaMemset(f.metrics,0,2*sizeof(int)));
    front_erode<<<(s.n+255)/256,256>>>(mask,f.temporary,s.h,s.w,radius,0);CHECK(cudaGetLastError());
    front_erode<<<(s.n+255)/256,256>>>(f.temporary,s.support,s.h,s.w,radius,1);CHECK(cudaGetLastError());
    median5<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(view.image,view.median,s.h,s.w);CHECK(cudaGetLastError());
    residual_prepare<<<(s.n+255)/256,256>>>(view,reset,floor2);CHECK(cudaGetLastError());
    front_eligible<<<(s.n+255)/256,256>>>(s,ready,f.metrics);CHECK(cudaGetLastError());
    front_noise<<<s.tiles,256>>>(s,f.stride,floor,f.sigmas,f.metrics);CHECK(cudaGetLastError());
    select_peaks<<<s.tiles*2,256>>>(s,k,ready,threshold,spatial_threshold);CHECK(cudaGetLastError());
    int status[2];CHECK(cudaMemcpy(status,f.metrics,sizeof(status),cudaMemcpyDeviceToHost));
    if(status[1])return cudaErrorInvalidValue;
    CHECK(cudaMemcpy(eligible,s.learn,s.n,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sigmas,f.sigmas,s.tiles*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(peaks,s.peaks,2*s.tiles*k*sizeof(Peak),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(counts,s.counts,2*s.tiles*sizeof(int),cudaMemcpyDeviceToHost));
    *searchable=status[0];return 0;
}
extern "C" int seaqr_front_v26_prepare_warp(void* ptr,void* warp,int erosion,int reset,int ready,
    float floor2,double floor,int k,float threshold,float spatial_threshold,
    unsigned char* eligible,double* sigmas,Peak* peaks,int* counts,int* searchable) {
    if(!ptr || !warp || erosion<0 || erosion>16)return cudaErrorInvalidValue;
    auto&f=*static_cast<FrontV26*>(ptr);auto&s=*f.core;auto&w=*static_cast<WarpWorkspace*>(warp);
    if(s.h!=w.h || s.w!=w.w)return cudaErrorInvalidValue;
    Resident view=s;view.image=w.output;view.blur=w.blur;
    return front_prepare(f,view,w.mask_out,erosion+6,reset,ready,floor2,floor,k,threshold,spatial_threshold,
                         eligible,sigmas,peaks,counts,searchable);
}
extern "C" int seaqr_front_v26_prepare_host(void* ptr,const float* image,const float* blur,
    const unsigned char* valid,int reset,int ready,float floor2,double floor,int k,
    float threshold,float spatial_threshold,unsigned char* eligible,double* sigmas,Peak* peaks,int* counts,int* searchable) {
    if(!ptr || !image || !blur || !valid)return cudaErrorInvalidValue;
    auto&f=*static_cast<FrontV26*>(ptr);auto&s=*f.core;
    CHECK(cudaMemcpy(s.image,image,size_t(s.n)*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.blur,blur,size_t(s.n)*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.support,valid,s.n,cudaMemcpyHostToDevice));
    return front_prepare(f,s,s.support,6,reset,ready,floor2,floor,k,threshold,spatial_threshold,
                         eligible,sigmas,peaks,counts,searchable);
}
__global__ void front_protect(uint32_t* bits,const int2* points,int n,const int2* offsets,int no,int h,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*no)return;
    int2 p=points[i/no],d=offsets[i%no];int x=p.x+d.x,y=p.y+d.y;
    if(x>=0 && x<w && y>=0 && y<h){int pixel=y*w+x;atomicOr(bits+pixel/32,uint32_t(1)<<(pixel%32));}
}
__global__ void front_learn(Resident s,const uint32_t* bits) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n)s.learn[i]=s.support[i] && !(bits[i/32]&(uint32_t(1)<<(i%32)));
}
extern "C" int seaqr_front_v26_finish(void* ptr,const int2* points,int n,const int2* offsets,int no,
    float alpha,float noise_alpha,float clip2,float floor2,int variance_only) {
    if(!ptr || n<0 || n>FRONT_POINTS || no<1 || no>FRONT_OFFSETS || (n && !points) || !offsets ||
       (variance_only!=0 && variance_only!=1))return cudaErrorInvalidValue;
    auto&f=*static_cast<FrontV26*>(ptr);auto&s=*f.core;
    CHECK(cudaMemset(f.protection,0,((s.n+31)/32)*sizeof(uint32_t)));
    if(n) {
        CHECK(cudaMemcpy(f.points,points,n*sizeof(int2),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(f.offsets,offsets,no*sizeof(int2),cudaMemcpyHostToDevice));
        front_protect<<<(n*no+255)/256,256>>>(f.protection,f.points,n,f.offsets,no,s.h,s.w);CHECK(cudaGetLastError());
    }
    front_learn<<<(s.n+255)/256,256>>>(s,f.protection);CHECK(cudaGetLastError());
    finish_state<<<(s.n+255)/256,256>>>(s,alpha,noise_alpha,clip2,floor2,variance_only);CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());return 0;
}
// Diagnostic-only transfers: never called by the timed front end.
extern "C" int seaqr_front_v26_debug(void* ptr,unsigned char* support,unsigned char* learn,float* stats,double* sigmas) {
    if(!ptr || !support || !learn || !stats || !sigmas)return cudaErrorInvalidValue;
    auto&f=*static_cast<FrontV26*>(ptr);auto&s=*f.core;
    CHECK(cudaMemcpy(support,s.support,s.n,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(learn,s.learn,s.n,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(stats,s.stats,s.tiles*2*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sigmas,f.sigmas,s.tiles*sizeof(double),cudaMemcpyDeviceToHost));return 0;
}
extern "C" int seaqr_front_v26_noise_probe(void* ptr,const float* temporal,const unsigned char* support,
    double floor,float* stats,double* sigmas) {
    if(!ptr || !temporal || !support || !stats || !sigmas || !isfinite(floor) || floor<=0)return cudaErrorInvalidValue;
    auto&f=*static_cast<FrontV26*>(ptr);auto&s=*f.core;
    CHECK(cudaMemcpy(s.temporal,temporal,s.n*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.support,support,s.n,cudaMemcpyHostToDevice));CHECK(cudaMemset(f.metrics,0,2*sizeof(int)));
    front_noise<<<s.tiles,256>>>(s,f.stride,floor,f.sigmas,f.metrics);CHECK(cudaGetLastError());
    int error;CHECK(cudaMemcpy(&error,f.metrics+1,sizeof(int),cudaMemcpyDeviceToHost));
    if(error)return cudaErrorInvalidValue;
    CHECK(cudaMemcpy(stats,s.stats,s.tiles*2*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sigmas,f.sigmas,s.tiles*sizeof(double),cudaMemcpyDeviceToHost));return 0;
}
