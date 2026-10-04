// Translation-only cubic conformance prototype. Coefficients come from the
// installed CPU reference, not a different interpolation formula.
#include <cuda_runtime.h>
#include <new>
#include <climits>
struct WarpWorkspace {
    int h=0,w=0;
    float *source=nullptr,*output=nullptr,*table=nullptr,*row=nullptr,*blur=nullptr;
    unsigned char *mask=nullptr,*mask_out=nullptr;
    int *xm=nullptr,*ym=nullptr;
    ~WarpWorkspace(){cudaFree(source);cudaFree(output);cudaFree(table);cudaFree(row);cudaFree(blur);cudaFree(mask);cudaFree(mask_out);cudaFree(xm);cudaFree(ym);}
};
__device__ float add_product(float sum,float pixel,float weight,int mode) {
    return mode?__fmaf_rn(pixel,weight,sum):__fadd_rn(sum,__fmul_rn(pixel,weight));
}
template<class T> __global__ void warp_cubic(const T* source,const unsigned char* mask,
        float* output,unsigned char* valid,const float* table,const int* xm,const int* ym,int h,int w,int mode) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h)return;
    int sx=xm[3*x]-1,sy=ym[3*y]-1,phase=ym[3*y+1]*32+xm[3*x+1];
    const float* weights=table+phase*16;
    float sum=0;
    bool inside=sx>=0 && sy>=0 && sx+3<w && sy+3<h;
    if(inside && mode==3) {
        // Installed OpenCV 4.10 ARM64 contraction order, verified independently
        // with synthetic conformance cases. Do not reassociate these operations.
        int base=sy*w+sx;
        sum=__fmul_rn(float(source[base+1]),weights[1]);
        sum=__fmaf_rn(float(source[base]),weights[0],sum);
        for(int i=2;i<16;i++)sum=__fmaf_rn(float(source[(sy+i/4)*w+sx+i%4]),weights[i],sum);
    } else if(inside) {
        for(int row=0;row<4;row++) {
            int base=(sy+row)*w+sx;
            float a=float(source[base]),b=float(source[base+1]),c=float(source[base+2]),d=float(source[base+3]);
            const float* k=weights+4*row;
            float line;
            if(mode==2)line=__fmaf_rn(a,k[0],__fmul_rn(b,k[1]));
            else line=add_product(__fmul_rn(a,k[0]),b,k[1],mode);
            line=add_product(line,c,k[2],mode);line=add_product(line,d,k[3],mode);
            sum=row?__fadd_rn(sum,line):line;
        }
    } else {
        for(int row=0;row<4;row++)for(int col=0;col<4;col++) {
            int xx=sx+col,yy=sy+row;
            if(xx>=0 && xx<w && yy>=0 && yy<h)sum=add_product(sum,float(source[yy*w+xx]),weights[row*4+col],mode);
        }
    }
    output[y*w+x]=sum;
    int nx=xm[3*x+2],ny=ym[3*y+2];
    valid[y*w+x]=(nx>=0 && nx<w && ny>=0 && ny<h)?mask[ny*w+nx]:0;
}
extern "C" void* seaqr_warp_create(int h,int w,const float* table) {
    if(h<1 || w<1 || h>=32767 || w>=32767 || !table)return nullptr;
    auto*p=new(std::nothrow)WarpWorkspace;if(!p)return nullptr;p->h=h;p->w=w;
    #define ALLOC(field,count) if(cudaMalloc(&p->field,size_t(count)*sizeof(*p->field))!=cudaSuccess){delete p;return nullptr;}
    ALLOC(source,h*w);ALLOC(output,h*w);ALLOC(row,h*w);ALLOC(blur,h*w);ALLOC(table,32*32*16);ALLOC(mask,h*w);ALLOC(mask_out,h*w);ALLOC(xm,w*3);ALLOC(ym,h*3);
    #undef ALLOC
    if(cudaMemcpy(p->table,table,32*32*16*4,cudaMemcpyHostToDevice)!=cudaSuccess){delete p;return nullptr;}
    return p;
}
__device__ int reflect101(int p,int n) {
    if(n==1)return 0;
    while(p<0 || p>=n)p=p<0?-p:2*n-p-2;
    return p;
}
__global__ void gaussian5(const float* in,float* out,int h,int w,int axis,float k0,float k1,float k2,int mode) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h)return;
    int p=axis?y:x,n=axis?h:w,stride=axis?w:1,base=axis?x:y*w;
    if(n==1){out[y*w+x]=in[y*w+x];return;}
    float v[3]={in[y*w+x],
        __fadd_rn(in[base+reflect101(p-1,n)*stride],in[base+reflect101(p+1,n)*stride]),
        __fadd_rn(in[base+reflect101(p-2,n)*stride],in[base+reflect101(p+2,n)*stride])};
    float k[3]={k0,k1,k2};
    const int order[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    int m=axis?0:mode;
    // OpenCV uses four-lane SIMD, then a two-lane tail, then one scalar.
    if(m==6)m=x<(w/4)*4?2:(x<(w/2)*2?4:0);
    float sum=__fmul_rn(v[order[m][0]],k[order[m][0]]);
    sum=__fmaf_rn(v[order[m][1]],k[order[m][1]],sum);
    out[y*w+x]=__fmaf_rn(v[order[m][2]],k[order[m][2]],sum);
}
extern "C" int seaqr_warp_gaussian(void* ptr,float k0,float k1,float k2,int mode,float* output) {
    if(!ptr || mode<0 || mode>6)return int(cudaErrorInvalidValue);
    auto&p=*static_cast<WarpWorkspace*>(ptr);dim3 t(32,8),b((p.w+31)/32,(p.h+7)/8);
    gaussian5<<<b,t>>>(p.output,p.row,p.h,p.w,0,k0,k1,k2,mode);
    cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    gaussian5<<<b,t>>>(p.row,p.blur,p.h,p.w,1,k0,k1,k2,mode);
    e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    return output?int(cudaMemcpy(output,p.blur,size_t(p.h)*p.w*4,cudaMemcpyDeviceToHost)):int(cudaDeviceSynchronize());
}
extern "C" void seaqr_warp_destroy(void* ptr){delete static_cast<WarpWorkspace*>(ptr);}
extern "C" int seaqr_warp_run(void* ptr,const void* source,const unsigned char* mask,const int* xm,const int* ym,
                              int kind,int mode,float* output,unsigned char* valid) {
    if(!ptr || !source || !mask || !xm || !ym || !valid || kind<0 || kind>1 || mode<0 || mode>3)return int(cudaErrorInvalidValue);
    auto&p=*static_cast<WarpWorkspace*>(ptr);
    #define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess)return int(e);}while(0)
    CHECK(cudaMemcpy(p.source,source,size_t(p.h)*p.w*(kind?1:4),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.mask,mask,size_t(p.h)*p.w,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.xm,xm,p.w*3*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p.ym,ym,p.h*3*4,cudaMemcpyHostToDevice));
    dim3 threads(32,8),blocks((p.w+31)/32,(p.h+7)/8);
    if(kind)warp_cubic<<<blocks,threads>>>(reinterpret_cast<unsigned char*>(p.source),p.mask,p.output,p.mask_out,p.table,p.xm,p.ym,p.h,p.w,mode);
    else warp_cubic<<<blocks,threads>>>(p.source,p.mask,p.output,p.mask_out,p.table,p.xm,p.ym,p.h,p.w,mode);
    CHECK(cudaGetLastError());
    if(output)CHECK(cudaMemcpy(output,p.output,size_t(p.h)*p.w*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid,p.mask_out,size_t(p.h)*p.w,cudaMemcpyDeviceToHost));return 0;
    #undef CHECK
}
extern "C" int seaqr_warp_download(void* ptr,float* output,float* blur) {
    if(!ptr || (!output && !blur))return int(cudaErrorInvalidValue);
    auto&p=*static_cast<WarpWorkspace*>(ptr);cudaError_t e;
    if(output){e=cudaMemcpy(output,p.output,size_t(p.h)*p.w*4,cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return int(e);}
    return blur?int(cudaMemcpy(blur,p.blur,size_t(p.h)*p.w*4,cudaMemcpyDeviceToHost)):0;
}
