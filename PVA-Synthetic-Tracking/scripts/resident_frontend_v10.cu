// Generated-data feasibility ONLY: direct point filtering is not CPU-FFT exact.
#include "../tiny_target/detection/cuda/raw_background.cu"
#include "../tiny_target/detection/cuda/warp_translation_v9.cu"
#include "../tiny_target/detection/cuda/point_filter_v8.cu"

extern "C" int seaqr_ring_v10_push_device(void *,const float *,const uint8_t *);

struct FrontV10 {
    State *background=nullptr;
    WarpWorkspace *warp=nullptr;
    uint16_t *raw=nullptr;
    float *response=nullptr,*point=nullptr;
    float norm=0, dark=0, saturated=0;
    int count=0;
    cudaEvent_t start=nullptr,end=nullptr;
};

__global__ void convert_raw_v10(const uint16_t *raw,float *image,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) image[i]=float(raw[i]);
}
__global__ void prepare_valid_v10(const float *image,const uint8_t *warped,uint8_t *valid,
    int h,int w,float dark,float saturated) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=h*w)return;
    int y=i/w,x=i%w;
    bool good=x>=2 && y>=2 && x<w-2 && y<h-2;
    if(good)for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)
        good=good && warped[(y+dy)*w+x+dx]!=0;
    valid[i]=good && image[i]>dark && image[i]<saturated;
}

extern "C" int seaqr_front_v10_abi(){return 1;}
extern "C" const char *seaqr_front_v10_error(int code){return cudaGetErrorString(cudaError_t(code));}
extern "C" void seaqr_front_v10_destroy(FrontV10 *p) {
    if(!p)return;
    seaqr_raw_background_destroy(p->background);seaqr_warp_v9_destroy(p->warp);
    cudaFree(p->raw);cudaFree(p->response);cudaFree(p->point);
    if(p->start)cudaEventDestroy(p->start);
    if(p->end)cudaEventDestroy(p->end);
    delete p;
}
extern "C" int seaqr_front_v10_create(int h,int w,int warmup,int required,const float *params,
    const float *table,const float *kernel,float norm,float dark,float saturated,FrontV10 **out) {
    if(!out)return cudaErrorInvalidValue;
    *out=nullptr;
    if(!params || !table || !kernel || !isfinite(norm) || norm<=0 || !isfinite(dark)
       || !isfinite(saturated) || saturated<=dark || warmup!=4)return cudaErrorInvalidValue;
    auto *p=new(std::nothrow)FrontV10;
    if(!p)return cudaErrorMemoryAllocation;
    int status=seaqr_raw_background_create(h,w,warmup,required,params,reinterpret_cast<void**>(&p->background));
    if(!status)status=seaqr_warp_v9_create(h,w,table,&p->warp);
    size_t n=size_t(h)*w;
    if(!status)status=cudaMalloc(&p->raw,n*2);
    if(!status)status=cudaMalloc(&p->response,n*4);
    if(!status)status=cudaMalloc(&p->point,81*4);
    if(!status)status=cudaMemcpy(p->point,kernel,81*4,cudaMemcpyHostToDevice);
    if(!status)status=cudaMemset(p->warp->mask,1,n);
    if(!status)status=cudaEventCreate(&p->start);
    if(!status)status=cudaEventCreate(&p->end);
    if(status){seaqr_front_v10_destroy(p);return status;}
    p->norm=norm;p->dark=dark;p->saturated=saturated;*out=p;return 0;
}
extern "C" int seaqr_front_v10_reset(FrontV10 *p) {
    if(!p)return cudaErrorInvalidValue;
    p->count=0;
    // Reset returns the source mask to the constructor's all-valid contract.
    CHECK(cudaMemset(p->warp->mask,1,size_t(p->background->n)));
    CHECK(cudaDeviceSynchronize());return 0;
}
extern "C" int seaqr_front_v10_step(FrontV10 *p,const uint16_t *raw,const uint8_t *mask,
    double tx,double ty,void *ring,int *emitted,float *compute_ms) {
    if(!p || !raw || !ring || !emitted || !compute_ms || !isfinite(tx) || !isfinite(ty)
       || fabs(tx)>1000000. || fabs(ty)>1000000. || p->count==INT_MAX)return cudaErrorInvalidValue;
    *emitted=0;
    State s=*p->background;WarpWorkspace &w=*p->warp;int blocks=(s.n+255)/256;
    CHECK(cudaMemcpy(p->raw,raw,size_t(s.n)*2,cudaMemcpyHostToDevice));
    if(mask)CHECK(cudaMemcpy(w.mask,mask,s.n,cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(p->start));
    convert_raw_v10<<<blocks,256>>>(p->raw,w.image,s.n);CHECK(cudaGetLastError());
    if(tx==0. && ty==0.) {
        CHECK(cudaMemcpy(w.output,w.image,size_t(s.n)*4,cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(w.outmask,w.mask,s.n,cudaMemcpyDeviceToDevice));
    } else {
        int bw=min(1024/min(16,s.h),s.w);
        translation<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(
            w.image,w.mask,w.output,w.outmask,w.weights,s.h,s.w,tx,ty,bw);
        CHECK(cudaGetLastError());
    }
    s.image=w.output; // Non-owning kernel view; background owns its original buffer.
    prepare_valid_v10<<<blocks,256>>>(s.image,w.outmask,s.valid,s.h,s.w,p->dark,p->saturated);
    CHECK(cudaGetLastError());
    temporal<<<blocks,256>>>(s,p->count==0);CHECK(cudaGetLastError());
    if(p->count) {
        support_horizontal<<<blocks,256>>>(s);CHECK(cudaGetLastError());
        support_vertical<<<blocks,256>>>(s,p->count>=s.warmup);CHECK(cudaGetLastError());
        correlate<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(s.white,p->point,p->response,s.h,s.w,p->norm);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaEventRecord(p->end));CHECK(cudaEventSynchronize(p->end));
    CHECK(cudaEventElapsedTime(compute_ms,p->start,p->end));
    if(p->count>=s.warmup) {
        int result=seaqr_ring_v10_push_device(ring,p->response,s.mask);
        if(result)return result;
        *emitted=1;
    }
    ++p->count;return 0;
}
extern "C" int seaqr_front_v10_debug(FrontV10 *p,float *warped,uint8_t *input_valid,
    float *white,float *response,uint8_t *filter_valid) {
    if(!p || p->count<2 || !warped || !input_valid || !white || !response || !filter_valid)
        return cudaErrorInvalidValue;
    State &s=*p->background;size_t bytes=size_t(s.n)*4;
    CHECK(cudaMemcpy(warped,p->warp->output,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(input_valid,s.valid,s.n,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(white,s.white,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(response,p->response,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(filter_valid,s.mask,s.n,cudaMemcpyDeviceToHost));return 0;
}
