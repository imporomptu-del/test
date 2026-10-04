// Resident residual/variance/peak pipeline; explicit float32 rounding, no fast math.
#include "phase20_cuda_median.cu"
#include <climits>
#include <type_traits>
struct Peak { int x,y; float score,response,noise; };
static_assert(sizeof(Peak)==20,"ABI layout");
struct Resident {
    int h,w,n,ns,tiles,tile;
    float *image=nullptr,*blur=nullptr,*median=nullptr,*spatial=nullptr;
    float *background=nullptr,*variance=nullptr,*temporal=nullptr,*samples=nullptr;
    float *stats=nullptr,*patches=nullptr;
    unsigned char *support=nullptr,*previous=nullptr,*learn=nullptr;
    int *indices=nullptr,*counts=nullptr,*seeds=nullptr;
    Peak *peaks=nullptr;
    // Kernel arguments must be trivially copyable: never free buffers from an
    // argument-copy destructor. Only the owning host handle calls release().
    void release() {
        cudaFree(image);cudaFree(blur);cudaFree(median);cudaFree(spatial);
        cudaFree(background);cudaFree(variance);cudaFree(temporal);cudaFree(samples);
        cudaFree(stats);cudaFree(patches);cudaFree(support);cudaFree(previous);
        cudaFree(learn);cudaFree(indices);cudaFree(counts);cudaFree(seeds);cudaFree(peaks);
    }
};
static_assert(std::is_trivially_copyable<Resident>::value,"Kernel view must not own argument copies");
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess)return int(e);}while(0)
__global__ void residual_prepare(Resident s,int reset,float floor2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    float spatial=__fsub_rn(s.blur[i],s.median[i]);s.spatial[i]=spatial;
    if(reset || (s.support[i] && !s.previous[i])) {
        s.background[i]=spatial;s.variance[i]=floor2;
    }
    if(reset)s.previous[i]=s.support[i];
    s.temporal[i]=__fsub_rn(spatial,s.background[i]);
}
__global__ void gather_samples(Resident s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.ns)s.samples[i]=s.temporal[s.indices[i]];
}
__device__ bool better(Peak a,Peak b) {
    if(a.x<0)return false;if(b.x<0)return true;
    return a.score>b.score || (a.score==b.score && (a.y<b.y || (a.y==b.y && a.x<b.x)));
}
__global__ void select_peaks(Resident s,int k,int ready,float threshold,float spatial_threshold) {
    int cell=blockIdx.x,tid=threadIdx.x,polarity=cell%2,tile=cell/2;
    int nx=(s.w+s.tile-1)/s.tile,x0=(tile%nx)*s.tile,y0=(tile/nx)*s.tile;
    int tw=min(s.tile,s.w-x0),th=min(s.tile,s.h-y0);
    float center=s.stats[tile*2],sigma=s.stats[tile*2+1],sign=polarity==0?1.f:-1.f;
    Peak local[16];int kept=0,total=0;
    if(ready)for(int j=tid;j<tw*th;j+=256) {
        int x=x0+j%tw,y=y0+j/tw,i=y*s.w+x;
        if(!s.support[i] || !s.previous[i])continue;
        float r=s.temporal[i],absolute=fabsf(r);bool peak=true;
        for(int dy=-2;dy<=2 && peak;dy++)for(int dx=-2;dx<=2;dx++) {
            int xx=x+dx,yy=y+dy;
            if(xx>=0 && xx<s.w && yy>=0 && yy<s.h && fabsf(s.temporal[yy*s.w+xx])>absolute) {peak=false;break;}
        }
        if(!peak)continue;
        float noise=fmaxf(sigma,__fsqrt_rn(s.variance[i]));
        float signed_r=__fmul_rn(sign,__fsub_rn(r,center));
        if(signed_r<__fmul_rn(threshold,noise) || __fmul_rn(sign,s.spatial[i])<__fmul_rn(spatial_threshold,noise))continue;
        total++;
        Peak p={x,y,__fdiv_rn(signed_r,noise),r,noise};
        int pos=kept;
        for(int q=0;q<kept;q++)if(better(p,local[q])){pos=q;break;}
        if(pos<k) {
            for(int q=min(kept,k-1);q>pos;q--)local[q]=local[q-1];
            local[pos]=p;kept=min(kept+1,k);
        }
    }
    __shared__ Peak heads[256];__shared__ int totals[256],winner;
    totals[tid]=total;__syncthreads();
    if(tid==0) {int sum=0;for(int j=0;j<256;j++)sum+=totals[j];s.counts[cell]=sum;}
    for(int rank=0;rank<k;rank++) {
        heads[tid]=kept?local[0]:Peak{-1,-1,0,0,0};__syncthreads();
        if(tid==0) {
            Peak best={-1,-1,0,0,0};winner=-1;
            for(int j=0;j<256;j++)if(better(heads[j],best)){best=heads[j];winner=j;}
            s.peaks[cell*k+rank]=best;
        }
        __syncthreads();
        if(tid==winner) {for(int j=1;j<kept;j++)local[j-1]=local[j];kept--;}
        __syncthreads();
    }
}
__global__ void gather_patches(Resident s,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*289)return;
    int seed=i/289,pos=i%289,x=s.seeds[2*seed]+pos%17-8,y=s.seeds[2*seed+1]+pos/17-8;
    s.patches[i]=(x>=0 && x<s.w && y>=0 && y<s.h)?s.spatial[y*s.w+x]:0.f;
}
__global__ void finish_state(Resident s,float alpha,float noise_alpha,float clip2,float floor2,int variance_only) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    float t=s.temporal[i],v=s.variance[i];
    if(variance_only?s.support[i]:s.learn[i])s.background[i]=__fadd_rn(s.background[i],__fmul_rn(alpha,t));
    float observed=fminf(__fmul_rn(t,t),__fmul_rn(v,clip2));
    if(s.learn[i])v=__fadd_rn(v,__fmul_rn(noise_alpha,__fsub_rn(observed,v)));
    s.variance[i]=fmaxf(v,floor2);s.previous[i]=s.support[i];
}
extern "C" void* seaqr_resident_create(int h,int w,int tile,int ns,const int* indices) {
    if(h<1 || w<1 || tile<1 || tile>256 || (long long)h*w>INT_MAX || ns<1 || ns>h*w)return nullptr;
    auto*s=new(std::nothrow)Resident;if(!s)return nullptr;
    s->h=h;s->w=w;s->n=h*w;s->ns=ns;s->tile=tile;s->tiles=((w+tile-1)/tile)*((h+tile-1)/tile);
    #define ALLOC(field,count) if(cudaMalloc(&s->field,size_t(count)*sizeof(*s->field))!=cudaSuccess){s->release();delete s;return nullptr;}
    ALLOC(image,s->n);ALLOC(blur,s->n);ALLOC(median,s->n);ALLOC(spatial,s->n);
    ALLOC(background,s->n);ALLOC(variance,s->n);ALLOC(temporal,s->n);
    ALLOC(support,s->n);ALLOC(previous,s->n);ALLOC(learn,s->n);
    ALLOC(indices,ns);ALLOC(samples,ns);ALLOC(stats,2*s->tiles);
    ALLOC(counts,2*s->tiles);ALLOC(peaks,2*s->tiles*16);ALLOC(seeds,1024);ALLOC(patches,512*289);
    #undef ALLOC
    if(cudaMemcpy(s->indices,indices,ns*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){s->release();delete s;return nullptr;}
    return s;
}
extern "C" void seaqr_resident_destroy(void* ptr){auto*s=static_cast<Resident*>(ptr);if(s){s->release();delete s;}}
extern "C" int seaqr_resident_prepare(void* ptr,const float* image,const float* blur,const unsigned char* support,int reset,float floor2,float* samples) {
    if(!ptr)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);size_t bytes=size_t(s.n)*4;
    CHECK(cudaMemcpy(s.image,image,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.blur,blur,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.support,support,s.n,cudaMemcpyHostToDevice));
    median5<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(s.image,s.median,s.h,s.w);
    CHECK(cudaGetLastError());
    residual_prepare<<<(s.n+255)/256,256>>>(s,reset,floor2);CHECK(cudaGetLastError());
    gather_samples<<<(s.ns+255)/256,256>>>(s);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(samples,s.samples,s.ns*4,cudaMemcpyDeviceToHost));return 0;
}
extern "C" int seaqr_resident_select(void* ptr,const float* stats,int k,int ready,float threshold,float spatial_threshold,Peak* peaks,int* counts) {
    if(!ptr || k<1 || k>16)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(s.stats,stats,s.tiles*2*4,cudaMemcpyHostToDevice));
    select_peaks<<<s.tiles*2,256>>>(s,k,ready,threshold,spatial_threshold);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(peaks,s.peaks,s.tiles*2*k*sizeof(Peak),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(counts,s.counts,s.tiles*2*sizeof(int),cudaMemcpyDeviceToHost));return 0;
}
extern "C" int seaqr_resident_patches(void* ptr,const int* seeds,int count,float* output) {
    if(!ptr || count<1 || count>512)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(s.seeds,seeds,count*2*sizeof(int),cudaMemcpyHostToDevice));
    gather_patches<<<(count*289+255)/256,256>>>(s,count);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output,s.patches,count*289*4,cudaMemcpyDeviceToHost));return 0;
}
extern "C" int seaqr_resident_finish(void* ptr,const unsigned char* learn,float alpha,float noise_alpha,float clip2,float floor2,int variance_only) {
    if(!ptr)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(s.learn,learn,s.n,cudaMemcpyHostToDevice));
    finish_state<<<(s.n+255)/256,256>>>(s,alpha,noise_alpha,clip2,floor2,variance_only);CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());return 0;
}
extern "C" int seaqr_resident_debug(void* ptr,float* background,float* variance) {
    if(!ptr)return cudaErrorInvalidValue;auto&s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(background,s.background,size_t(s.n)*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(variance,s.variance,size_t(s.n)*4,cudaMemcpyDeviceToHost));return 0;
}
