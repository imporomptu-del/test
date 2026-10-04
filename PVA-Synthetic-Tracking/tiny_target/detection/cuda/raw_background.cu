// Opt-in RAW16 temporal background + 9x9 support, with explicit FP32 rounding.
// The production path deliberately retains OpenCV's FFT-based point filter.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <climits>
#include <new>
#include <type_traits>

struct State {
    int h=0,w=0,n=0,warmup=0,required=0;
    float floor2=0,rate=0,decay=0,out_rate=0,out_decay=0,exclude=0,clip=0;
    float *image=nullptr,*location=nullptr,*variance=nullptr,*white=nullptr,*response=nullptr,*kernel=nullptr;
    uint16_t *history=nullptr;
    uint8_t *valid=nullptr,*detect=nullptr,*horizontal=nullptr,*mask=nullptr;
    void release() {
        cudaFree(image);cudaFree(location);cudaFree(variance);cudaFree(white);
        cudaFree(response);cudaFree(kernel);cudaFree(history);cudaFree(valid);
        cudaFree(detect);cudaFree(horizontal);cudaFree(mask);
    }
};
static_assert(std::is_trivially_copyable<State>::value,"Kernel argument copies must not own buffers");
#define CHECK(op) do {cudaError_t e=(op);if(e!=cudaSuccess)return int(e);}while(0)

__global__ void temporal(State s,int reset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    float image=s.image[i];bool valid=s.valid[i]!=0;
    if(reset) {
        s.location[i]=valid?image:0.f;s.variance[i]=s.floor2;
        s.history[i]=valid?1:0;s.white[i]=0.f;s.detect[i]=0;s.mask[i]=0;
        return;
    }
    float location=s.location[i],variance=s.variance[i];unsigned history=s.history[i];
    float sigma=__fsqrt_rn(fmaxf(variance,s.floor2));
    bool ready=history>=s.warmup;
    float white=(valid && history>0)?__fdiv_rn(__fsub_rn(image,location),sigma):0.f;
    s.white[i]=white;s.detect[i]=valid && ready;
    bool unseen=valid && history==0;
    if(unseen) {location=image;variance=s.floor2;}
    if(valid && !unseen) {
        bool protected_outlier=ready && fabsf(white)>=s.exclude;
        float innovation=__fsub_rn(image,location);
        float limit=__fmul_rn(s.clip,sigma);
        float clipped=fminf(fmaxf(innovation,-limit),limit);
        float rate=protected_outlier?s.out_rate:s.rate;
        float decay=protected_outlier?s.out_decay:s.decay;
        // Mirror distinct NumPy ufunc operations; never fuse state arithmetic.
        location=__fadd_rn(location,__fmul_rn(clipped,rate));
        float product=__fmul_rn(variance,decay);
        float square=__fmul_rn(clipped,clipped);
        variance=__fadd_rn(product,__fmul_rn(square,rate));
    }
    s.location[i]=location;s.variance[i]=variance;
    s.history[i]=history+unsigned(valid && history<65535);
}

__global__ void support_horizontal(State s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    int x=i%s.w,total=0;
    #pragma unroll
    for(int dx=-4;dx<=4;dx++)if(x+dx>=0 && x+dx<s.w)total+=s.detect[i+dx]!=0;
    s.horizontal[i]=uint8_t(total);
}

__global__ void support_vertical(State s,int ready) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    int x=i%s.w,y=i/s.w,total=0;
    #pragma unroll
    for(int dy=-4;dy<=4;dy++)if(y+dy>=0 && y+dy<s.h)total+=s.horizontal[i+dy*s.w];
    s.mask[i]=ready && s.detect[i] && total>=s.required && x>=4 && y>=4 && x<s.w-4 && y<s.h-4;
}

// Diagnostic only: establishes whether direct sequential FMA convolution is
// identical to the installed CPU implementation. Never called by production.
__global__ void direct_point_probe(State s,float normalizer) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    int x=i%s.w,y=i/s.w;float sum=0.f;
    #pragma unroll
    for(int ky=0;ky<9;ky++) {
        #pragma unroll
        for(int kx=0;kx<9;kx++) {
            float coefficient=s.kernel[ky*9+kx];
            int xx=x+kx-4,yy=y+ky-4;
            float value=(xx>=0 && xx<s.w && yy>=0 && yy<s.h)?s.white[yy*s.w+xx]:0.f;
            if(coefficient!=0.f)sum=__fmaf_rn(value,coefficient,sum);
        }
    }
    s.response[i]=__fdiv_rn(sum,normalizer);
}

// ABI 2 requires the Jetson CPU's flush-to-zero arithmetic mode. Compile this
// translation unit with --ftz=true; the host checks its own mode before use.
extern "C" int seaqr_raw_background_abi(){return 2;}
extern "C" const char* seaqr_raw_background_error(int status){return cudaGetErrorString(cudaError_t(status));}
extern "C" int seaqr_raw_background_create(int h,int w,int warmup,int required,const float* params,void** result) {
    if(!result)return int(cudaErrorInvalidValue);*result=nullptr;
    if(h<1 || w<1 || (int64_t)h*w>32000000 || warmup<1 || required<1 || required>81 || !params)
        return int(cudaErrorInvalidValue);
    for(int i=0;i<7;i++)if(!std::isfinite(params[i]) || params[i]<0)return int(cudaErrorInvalidValue);
    if(params[0]<=0 || params[1]<=0 || params[1]>1 || params[3]<=0 || params[3]>params[1]
            || params[2]>1 || params[4]>1 || params[5]<=0 || params[6]<=0)return int(cudaErrorInvalidValue);
    State*s=new(std::nothrow)State;if(!s)return int(cudaErrorMemoryAllocation);
    s->h=h;s->w=w;s->n=h*w;s->warmup=warmup;s->required=required;
    s->floor2=params[0];s->rate=params[1];s->decay=params[2];s->out_rate=params[3];
    s->out_decay=params[4];s->exclude=params[5];s->clip=params[6];
    #define ALLOC(field,count) do {cudaError_t e=cudaMalloc(&s->field,size_t(count)*sizeof(*s->field));if(e!=cudaSuccess){s->release();delete s;return int(e);}}while(0)
    ALLOC(image,s->n);ALLOC(location,s->n);ALLOC(variance,s->n);ALLOC(white,s->n);
    ALLOC(history,s->n);ALLOC(valid,s->n);
    ALLOC(detect,s->n);ALLOC(horizontal,s->n);ALLOC(mask,s->n);
    #undef ALLOC
    *result=s;return 0;
}
extern "C" void seaqr_raw_background_destroy(void*ptr){State*s=static_cast<State*>(ptr);if(s){s->release();delete s;}}
extern "C" int seaqr_raw_background_step(void*ptr,const float*image,const uint8_t*valid,int reset,int ready,float*white,uint8_t*mask) {
    if(!ptr || !image || !valid || !white || !mask || (reset!=0 && reset!=1) || (ready!=0 && ready!=1))return int(cudaErrorInvalidValue);
    State&s=*static_cast<State*>(ptr);size_t bytes=size_t(s.n)*sizeof(float);
    CHECK(cudaMemcpy(s.image,image,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.valid,valid,s.n,cudaMemcpyHostToDevice));
    temporal<<<(s.n+255)/256,256>>>(s,reset);CHECK(cudaGetLastError());
    if(!reset) {
        support_horizontal<<<(s.n+255)/256,256>>>(s);CHECK(cudaGetLastError());
        support_vertical<<<(s.n+255)/256,256>>>(s,ready);CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(white,s.white,bytes,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(mask,s.mask,s.n,cudaMemcpyDeviceToHost));
    } else {CHECK(cudaDeviceSynchronize());}
    return 0;
}
extern "C" int seaqr_raw_background_debug(void*ptr,float*location,float*variance,uint16_t*history,float*white,uint8_t*detect) {
    if(!ptr || !location || !variance || !history || !white || !detect)return int(cudaErrorInvalidValue);
    State&s=*static_cast<State*>(ptr);size_t bytes=size_t(s.n)*sizeof(float);
    CHECK(cudaMemcpy(location,s.location,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(variance,s.variance,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(history,s.history,size_t(s.n)*sizeof(uint16_t),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(white,s.white,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(detect,s.detect,s.n,cudaMemcpyDeviceToHost));return 0;
}
extern "C" int seaqr_raw_background_set_state(void*ptr,const float*location,const float*variance,const uint16_t*history) {
    if(!ptr || !location || !variance || !history)return int(cudaErrorInvalidValue);
    State&s=*static_cast<State*>(ptr);size_t bytes=size_t(s.n)*sizeof(float);
    CHECK(cudaMemcpy(s.location,location,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.variance,variance,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.history,history,size_t(s.n)*sizeof(uint16_t),cudaMemcpyHostToDevice));return 0;
}
extern "C" int seaqr_raw_background_point_probe(void*ptr,const float*white,const float*kernel,float normalizer,float*response) {
    if(!ptr || !white || !kernel || !response || !std::isfinite(normalizer) || normalizer<=0)return int(cudaErrorInvalidValue);
    State&s=*static_cast<State*>(ptr);size_t bytes=size_t(s.n)*sizeof(float);
    // Diagnostic buffers are never allocated in the production temporal path.
    if(!s.response)CHECK(cudaMalloc(&s.response,bytes));
    if(!s.kernel)CHECK(cudaMalloc(&s.kernel,81*sizeof(float)));
    CHECK(cudaMemcpy(s.white,white,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.kernel,kernel,81*sizeof(float),cudaMemcpyHostToDevice));
    direct_point_probe<<<(s.n+255)/256,256>>>(s,normalizer);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(response,s.response,bytes,cudaMemcpyDeviceToHost));return 0;
}
