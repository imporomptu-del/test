// V56 additive diagnostics. Include only the hash-pinned archived V26 source.
// None of the included production kernels or exports is replaced or modified.
#include "visible_front_v26.cu"

constexpr int V56_FLOATS=20,V56_FLAGS=13,V56_MAX_PIXELS=600000;
extern "C" int seaqr_accuracy_v56_abi(){return 1;}

// This kernel reads production state and writes only separate capture buffers.
// Predicate operations deliberately match archived select_peaks, including the
// raw, uncentered, absolute 5x5 comparison across unsupported neighboring pixels.
__global__ void accuracy_v56_snapshot(Resident s,const float* image,const float* blur,
    const double* sigmas,int x0,int y0,int width,int height,int ready,
    float threshold,float spatial_threshold,float* values,unsigned char* flags,
    double* precise_sigmas) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=width*height)return;
    int x=x0+j%width,y=y0+j/width,i=y*s.w+x;
    int nx=(s.w+s.tile-1)/s.tile,tile=(y/s.tile)*nx+x/s.tile;
    float r=s.temporal[i],spatial=s.spatial[i],center=s.stats[2*tile];
    float sigma=s.stats[2*tile+1],noise=fmaxf(sigma,__fsqrt_rn(s.variance[i]));
    float centered=__fsub_rn(r,center),positive=__fmul_rn(1.f,centered);
    float negative=__fmul_rn(-1.f,centered);
    float positive_spatial=__fmul_rn(1.f,spatial),negative_spatial=__fmul_rn(-1.f,spatial);
    float temporal_limit=__fmul_rn(threshold,noise);
    float spatial_limit=__fmul_rn(spatial_threshold,noise);
    float absolute=fabsf(r),maximum=absolute;bool peak=true,finite_neighbors=true;
    for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++) {
        int xx=x+dx,yy=y+dy;
        if(xx>=0 && xx<s.w && yy>=0 && yy<s.h) {
            float neighbor=fabsf(s.temporal[yy*s.w+xx]);
            if(neighbor>absolute)peak=false;
            if(!isfinite(neighbor))finite_neighbors=false;
            maximum=fmaxf(maximum,neighbor);
        }
    }
    bool eligible=ready && s.support[i] && s.previous[i];
    bool pt=!(positive<temporal_limit),nt=!(negative<temporal_limit);
    bool ps=!(positive_spatial<spatial_limit),ns=!(negative_spatial<spatial_limit);
    float* out=values+size_t(j)*V56_FLOATS;
    out[0]=image[i];out[1]=blur[i];out[2]=s.median[i];out[3]=spatial;
    out[4]=s.background[i];out[5]=r;out[6]=s.variance[i];out[7]=center;
    out[8]=sigma;out[9]=noise;out[10]=centered;out[11]=temporal_limit;
    out[12]=spatial_limit;out[13]=__fdiv_rn(positive,noise);
    out[14]=__fdiv_rn(negative,noise);out[15]=maximum;
    out[16]=positive;out[17]=negative;out[18]=positive_spatial;out[19]=negative_spatial;
    unsigned char* bits=flags+size_t(j)*V56_FLAGS;
    bits[0]=s.support[i]!=0;bits[1]=s.previous[i]!=0;bits[2]=s.learn[i]!=0;
    bits[3]=ready;bits[4]=eligible;bits[5]=pt;bits[6]=nt;bits[7]=ps;
    bits[8]=ns;bits[9]=peak;bits[10]=eligible && pt && ps && peak;
    bits[11]=eligible && nt && ns && peak;bits[12]=finite_neighbors;
    precise_sigmas[j]=sigmas[tile];
}

// Must be called immediately after a successful prepare and before finish. For
// warp preparation the borrowed current image/blur reside in WarpWorkspace, not
// the unused image/blur allocations owned by Resident.
extern "C" int seaqr_accuracy_v56_capture(void* ptr,void* warp,int x0,int y0,
    int width,int height,int ready,float threshold,float spatial_threshold,
    float* host_values,unsigned char* host_flags,double* host_sigmas) {
    if(!ptr || !host_values || !host_flags || !host_sigmas || x0<0 || y0<0 ||
       width<1 || height<1 || int64_t(width)*height>V56_MAX_PIXELS ||
       (ready!=0 && ready!=1) || !isfinite(threshold) || !isfinite(spatial_threshold))
        return cudaErrorInvalidValue;
    auto& f=*static_cast<FrontV26*>(ptr);if(!f.core)return cudaErrorInvalidValue;
    auto& s=*f.core;
    if(int64_t(x0)+width>s.w || int64_t(y0)+height>s.h)return cudaErrorInvalidValue;
    const float* image=s.image;const float* blur=s.blur;
    if(warp) {
        auto& w=*static_cast<WarpWorkspace*>(warp);
        if(w.h!=s.h || w.w!=s.w)return cudaErrorInvalidValue;
        image=w.output;blur=w.blur;
    }
    float* values=nullptr;unsigned char* flags=nullptr;double* sigmas=nullptr;
    size_t count=size_t(width)*height;
    auto capture=[&]()->int {
        CHECK(cudaMalloc(&values,count*V56_FLOATS*sizeof(float)));
        CHECK(cudaMalloc(&flags,count*V56_FLAGS));
        CHECK(cudaMalloc(&sigmas,count*sizeof(double)));
        accuracy_v56_snapshot<<<(count+255)/256,256>>>(s,image,blur,f.sigmas,
            x0,y0,width,height,ready,threshold,spatial_threshold,values,flags,sigmas);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(host_values,values,count*V56_FLOATS*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(host_flags,flags,count*V56_FLAGS,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(host_sigmas,sigmas,count*sizeof(double),cudaMemcpyDeviceToHost));
        return 0;
    };
    int result=capture();
    cudaError_t a=cudaFree(values),b=cudaFree(flags),c=cudaFree(sigmas);
    if(result)return result;
    return a!=cudaSuccess?int(a):(b!=cudaSuccess?int(b):int(c));
}
