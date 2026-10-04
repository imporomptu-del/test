// Persistent storage only: the production arithmetic kernels are unchanged.
#include "../tiny_target/detection/cuda/synthetic_tracking.cu"
#include <new>

struct RingV10 {
    int h, w, frames, velocities, minimum, batch, threads;
    size_t pixels, count=0;
    bool output_ready=false;
    float *data=nullptr, *score=nullptr;
    uint8_t *masks=nullptr, *valid=nullptr;
    uint16_t *velocity=nullptr, *support=nullptr;
    float2 *grid=nullptr, *displacements=nullptr;
    double *offsets=nullptr;
    cudaEvent_t start=nullptr, end=nullptr;
};

#define RV_CHECK(op) do { cudaError_t e=(op); if(e!=cudaSuccess) return int(e); } while(0)
extern "C" int seaqr_ring_v10_abi() { return 1; }
extern "C" const char *seaqr_ring_v10_error(int code) { return cudaGetErrorString(cudaError_t(code)); }
extern "C" void seaqr_ring_v10_destroy(RingV10 *p) {
    if(!p) return;
    cudaFree(p->data); cudaFree(p->masks); cudaFree(p->score); cudaFree(p->valid);
    cudaFree(p->velocity); cudaFree(p->support); cudaFree(p->grid);
    cudaFree(p->displacements); cudaFree(p->offsets);
    if(p->start) cudaEventDestroy(p->start);
    if(p->end) cudaEventDestroy(p->end);
    delete p;
}
extern "C" int seaqr_ring_v10_create(int h, int w, int frames, int velocities,
    int minimum, int batch, int threads, const float *grid, RingV10 **out) {
    if(!out) return cudaErrorInvalidValue;
    *out=nullptr;
    if(!grid || h<1 || w<1 || size_t(h)*w>32000000 || frames!=16 || velocities!=48
       || minimum!=12 || batch!=32 || threads!=256) return cudaErrorInvalidValue;
    size_t n=size_t(h)*w, free_bytes=0, total_bytes=0;
    RV_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    size_t required=n*(size_t(frames)*2*5+9)+velocities*8+frames*8+frames*velocities*8;
    if(required>free_bytes || free_bytes-required<512ULL*1024*1024) return cudaErrorMemoryAllocation;
    auto *p=new(std::nothrow) RingV10;
    if(!p) return cudaErrorMemoryAllocation;
    p->h=h; p->w=w; p->pixels=n; p->frames=frames; p->velocities=velocities;
    p->minimum=minimum; p->batch=batch; p->threads=threads;
    cudaError_t status=cudaSuccess;
    #define RV_ALLOC(field,count) if(status==cudaSuccess) status=cudaMalloc(&p->field,size_t(count)*sizeof(*p->field))
    RV_ALLOC(data,2*frames*n); RV_ALLOC(masks,2*frames*n);
    RV_ALLOC(score,n); RV_ALLOC(valid,n); RV_ALLOC(velocity,n); RV_ALLOC(support,n);
    RV_ALLOC(grid,velocities); RV_ALLOC(displacements,frames*velocities); RV_ALLOC(offsets,frames);
    #undef RV_ALLOC
    if(status==cudaSuccess) status=cudaMemcpy(p->grid,grid,velocities*sizeof(float2),cudaMemcpyHostToDevice);
    if(status==cudaSuccess) status=cudaEventCreate(&p->start);
    if(status==cudaSuccess) status=cudaEventCreate(&p->end);
    if(status!=cudaSuccess) { seaqr_ring_v10_destroy(p); return int(status); }
    *out=p; return 0;
}
extern "C" int seaqr_ring_v10_reset(RingV10 *p) {
    if(!p) return cudaErrorInvalidValue;
    p->count=0; p->output_ready=false; return 0;
}
// Double-mapping by explicit device copy makes the chronological window contiguous.
// Its duplication traffic and extra device memory are included in the report.
static int ring_push(RingV10 *p, const float *data, const uint8_t *mask, cudaMemcpyKind kind) {
    if(!p || !data || !mask || p->count==SIZE_MAX) return cudaErrorInvalidValue;
    p->output_ready=false;
    size_t slot=p->count%p->frames, offset=slot*p->pixels, bytes=p->pixels*sizeof(float);
    RV_CHECK(cudaMemcpy(p->data+offset,data,bytes,kind));
    RV_CHECK(cudaMemcpy(p->masks+offset,mask,p->pixels,kind));
    RV_CHECK(cudaMemcpy(p->data+offset+p->frames*p->pixels,p->data+offset,bytes,cudaMemcpyDeviceToDevice));
    RV_CHECK(cudaMemcpy(p->masks+offset+p->frames*p->pixels,p->masks+offset,p->pixels,cudaMemcpyDeviceToDevice));
    RV_CHECK(cudaDeviceSynchronize());
    ++p->count; return 0;
}
extern "C" int seaqr_ring_v10_push(RingV10 *p,const float *data,const uint8_t *mask) {
    return ring_push(p,data,mask,cudaMemcpyHostToDevice);
}
extern "C" int seaqr_ring_v10_push_device(RingV10 *p,const float *data,const uint8_t *mask) {
    return ring_push(p,data,mask,cudaMemcpyDeviceToDevice);
}
extern "C" int seaqr_ring_v10_run(RingV10 *p,const double *offsets,int polarity,float *kernel_ms) {
    if(!p || !offsets || !kernel_ms || p->count<size_t(p->frames) || (polarity!=1 && polarity!=-1))
        return cudaErrorInvalidValue;
    p->output_ready=false;
    RV_CHECK(cudaMemcpy(p->offsets,offsets,p->frames*sizeof(double),cudaMemcpyHostToDevice));
    RV_CHECK(cudaEventRecord(p->start));
    int displacement_blocks=(p->frames*p->velocities+p->threads-1)/p->threads;
    precompute_displacements_kernel<<<displacement_blocks,p->threads>>>(
        p->offsets,p->grid,p->displacements,p->frames,p->velocities);
    RV_CHECK(cudaGetLastError());
    int blocks=int((p->pixels+p->threads-1)/p->threads);
    initialize_outputs_kernel<<<blocks,p->threads>>>(p->score,p->velocity,p->support,p->valid,p->pixels);
    RV_CHECK(cudaGetLastError());
    size_t offset=(p->count%p->frames)*p->pixels;
    for(int start=0;start<p->velocities;start+=p->batch) {
        int count=min(p->batch,p->velocities-start);
        shift_and_stack_batch_kernel<<<blocks,p->threads>>>(p->data+offset,p->masks+offset,
            p->displacements,p->frames,start,count,p->w,p->h,p->minimum,polarity,
            p->score,p->velocity,p->support);
        RV_CHECK(cudaGetLastError());
    }
    finalize_outputs_kernel<<<blocks,p->threads>>>(p->score,p->valid,p->pixels);
    RV_CHECK(cudaGetLastError());
    RV_CHECK(cudaEventRecord(p->end)); RV_CHECK(cudaEventSynchronize(p->end));
    RV_CHECK(cudaEventElapsedTime(kernel_ms,p->start,p->end));
    p->output_ready=true; return 0;
}
extern "C" int seaqr_ring_v10_download(RingV10 *p,float *score,uint16_t *velocity,uint16_t *support,uint8_t *valid) {
    if(!p || !p->output_ready || !score || !velocity || !support || !valid) return cudaErrorInvalidValue;
    RV_CHECK(cudaMemcpy(score,p->score,p->pixels*4,cudaMemcpyDeviceToHost));
    RV_CHECK(cudaMemcpy(velocity,p->velocity,p->pixels*2,cudaMemcpyDeviceToHost));
    RV_CHECK(cudaMemcpy(support,p->support,p->pixels*2,cudaMemcpyDeviceToHost));
    RV_CHECK(cudaMemcpy(valid,p->valid,p->pixels,cudaMemcpyDeviceToHost));
    return 0;
}
