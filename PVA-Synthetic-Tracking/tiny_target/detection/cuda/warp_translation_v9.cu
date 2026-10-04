// Reference arithmetic: OpenCV 4.10 imgwarp.cpp (Apache-2.0), Jetson NEON build.
// This is translation-only, float32 cubic + uint8 nearest mask, zero border.
#include <cuda_runtime.h>
#include <cstddef>
#include <new>

struct WarpWorkspace {
    int h, w;
    float *image=nullptr, *output=nullptr, *weights=nullptr;
    unsigned char *mask=nullptr, *outmask=nullptr;
};

__device__ int short_clamp(int x) { return max(-32768,min(32767,x)); }
__device__ int rounded(double x) {
    x=fmax(-2147483648.,fmin(2147483647.,x));
    return __double2int_rn(x);
}
__global__ void translation(const float *src,const unsigned char *mask,float *dst,
    unsigned char *outmask,const float *table,int h,int w,double tx,double ty,int bw) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    // Match warpPerspective's horizontal block origin + within-block addition.
    int origin=(x/bw)*bw;
    double fx=__dadd_rn(__dadd_rn(double(origin),tx),double(x-origin));
    double fy=__dadd_rn(double(y),ty);
    int nx=short_clamp(rounded(fx)),ny=short_clamp(rounded(fy));
    outmask[y*w+x]=(nx>=0 && nx<w && ny>=0 && ny<h)?mask[ny*w+nx]:0;
    int ix=rounded(__dmul_rn(fx,32.)),iy=rounded(__dmul_rn(fy,32.));
    int sx=short_clamp(ix>>5)-1,sy=short_clamp(iy>>5)-1;
    const float *weights=table+((iy&31)*32+(ix&31))*16;
    float sum=0.f;
    if(sx>=0 && sx<w-3 && sy>=0 && sy<h-3) {
        // Installed compiler starts at term 1, fuses term 0, then terms 2..15.
        sum=__fmul_rn(src[sy*w+sx+1],weights[1]);
        sum=__fmaf_rn(src[sy*w+sx],weights[0],sum);
        #pragma unroll
        for(int k=2;k<16;++k) sum=__fmaf_rn(src[(sy+k/4)*w+sx+k%4],weights[k],sum);
    } else {
        // CPU's border branch skips nonexistent pixels, starting at zero.
        #pragma unroll
        for(int k=0;k<16;++k) {
            int xx=sx+k%4,yy=sy+k/4;
            if(xx>=0 && xx<w && yy>=0 && yy<h)
                sum=__fmaf_rn(src[yy*w+xx],weights[k],sum);
        }
    }
    dst[y*w+x]=sum;
}

extern "C" int seaqr_warp_v9_abi() { return 1; }
extern "C" const char *seaqr_warp_v9_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
extern "C" void seaqr_warp_v9_destroy(WarpWorkspace *p) {
    if(!p) return;
    cudaFree(p->image);cudaFree(p->output);cudaFree(p->mask);cudaFree(p->outmask);
    cudaFree(p->weights);delete p;
}
extern "C" int seaqr_warp_v9_create(int h,int w,const float *table,WarpWorkspace **out) {
    if(!out || !table || h<1 || w<1 || h>=32767 || w>=32767 || size_t(h)*w>32000000)
        return cudaErrorInvalidValue;
    *out=nullptr;
    auto *p=new(std::nothrow) WarpWorkspace;
    if(!p) return cudaErrorMemoryAllocation;
    p->h=h;p->w=w;size_t n=size_t(h)*w;
    auto status=cudaMalloc(&p->image,n*sizeof(float));
    if(status==cudaSuccess) status=cudaMalloc(&p->output,n*sizeof(float));
    if(status==cudaSuccess) status=cudaMalloc(&p->mask,n);
    if(status==cudaSuccess) status=cudaMalloc(&p->outmask,n);
    if(status==cudaSuccess) status=cudaMalloc(&p->weights,1024*16*sizeof(float));
    if(status==cudaSuccess) status=cudaMemcpy(p->weights,table,1024*16*sizeof(float),cudaMemcpyHostToDevice);
    if(status!=cudaSuccess){seaqr_warp_v9_destroy(p);return status;}
    *out=p;return cudaSuccess;
}
extern "C" int seaqr_warp_v9_run(WarpWorkspace *p,const float *src,const unsigned char *mask,
    double tx,double ty,float *dst,unsigned char *outmask) {
    if(!p || !src || !mask || !dst || !outmask || !isfinite(tx) || !isfinite(ty)
        || fabs(tx)>1000000. || fabs(ty)>1000000.) return cudaErrorInvalidValue;
    size_t n=size_t(p->h)*p->w;
    auto status=cudaMemcpy(p->image,src,n*sizeof(float),cudaMemcpyHostToDevice);
    if(status==cudaSuccess) status=cudaMemcpy(p->mask,mask,n,cudaMemcpyHostToDevice);
    if(status!=cudaSuccess) return status;
    int bw=min(1024/min(16,p->h),p->w);
    translation<<<dim3((p->w+31)/32,(p->h+7)/8),dim3(32,8)>>>(
        p->image,p->mask,p->output,p->outmask,p->weights,p->h,p->w,tx,ty,bw);
    status=cudaGetLastError();
    if(status==cudaSuccess) status=cudaMemcpy(dst,p->output,n*sizeof(float),cudaMemcpyDeviceToHost);
    if(status==cudaSuccess) status=cudaMemcpy(outmask,p->outmask,n,cudaMemcpyDeviceToHost);
    return status;
}
