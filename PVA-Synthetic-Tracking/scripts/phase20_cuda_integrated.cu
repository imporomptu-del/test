// One compiled library owns the warp and detector ABIs. No foreign device pointer
// is exposed to Python; the caller validates handle lifetime and frame generation.
#include "phase20_cuda_warp_exact.cu"
#include "phase20_cuda_resident.cu"
extern "C" int seaqr_resident_prepare_warp(void* ptr,void* warp,const unsigned char* support,
                                          int reset,float floor2,float* samples) {
    if(!ptr || !warp || !support || !samples)return int(cudaErrorInvalidValue);
    auto&s=*static_cast<Resident*>(ptr);auto&p=*static_cast<WarpWorkspace*>(warp);
    if(s.h!=p.h || s.w!=p.w)return int(cudaErrorInvalidValue);
    // The owning Resident retains its own allocations; only this trivially
    // copyable launch view borrows the warp buffers for the duration of prepare.
    Resident view=s;view.image=p.output;view.blur=p.blur;
    CHECK(cudaMemcpy(s.support,support,s.n,cudaMemcpyHostToDevice));
    median5<<<dim3((s.w+31)/32,(s.h+7)/8),dim3(32,8)>>>(view.image,view.median,s.h,s.w);
    CHECK(cudaGetLastError());
    residual_prepare<<<(s.n+255)/256,256>>>(view,reset,floor2);CHECK(cudaGetLastError());
    gather_samples<<<(s.ns+255)/256,256>>>(view);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(samples,s.samples,s.ns*4,cudaMemcpyDeviceToHost));return 0;
}
