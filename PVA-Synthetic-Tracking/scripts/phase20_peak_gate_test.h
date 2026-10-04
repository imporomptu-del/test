// Test-only bounded state seeding/timer; absent from the candidate runtime ABI.
extern "C" int seaqr_test_seed(void* ptr,const float* temporal,const float* spatial,
        const float* variance,const unsigned char* support,const unsigned char* previous) {
    if(!ptr || !temporal || !spatial || !variance || !support || !previous) return int(cudaErrorInvalidValue);
    auto& s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(s.temporal,temporal,size_t(s.n)*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.spatial,spatial,size_t(s.n)*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.variance,variance,size_t(s.n)*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.support,support,s.n,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.previous,previous,s.n,cudaMemcpyHostToDevice));
    return 0;
}
extern "C" int seaqr_test_kernel_time(void* ptr,const float* stats,int k,int ready,
        float threshold,float spatial_threshold,int iterations,float* elapsed) {
    if(!ptr || !stats || !elapsed || k<1 || k>16 || iterations<1 || iterations>100) return int(cudaErrorInvalidValue);
    auto& s=*static_cast<Resident*>(ptr);
    CHECK(cudaMemcpy(s.stats,stats,s.tiles*2*4,cudaMemcpyHostToDevice));
    cudaEvent_t begin,end;
    CHECK(cudaEventCreate(&begin));
    auto error=cudaEventCreate(&end);
    if(error!=cudaSuccess) {cudaEventDestroy(begin);return int(error);}
    error=cudaEventRecord(begin);
    for(int i=0;i<iterations && error==cudaSuccess;++i) {
        select_peaks<<<s.tiles*2,256>>>(s,k,ready,threshold,spatial_threshold);
        error=cudaGetLastError();
    }
    if(error==cudaSuccess)error=cudaEventRecord(end);
    if(error==cudaSuccess)error=cudaEventSynchronize(end);
    if(error==cudaSuccess)error=cudaEventElapsedTime(elapsed,begin,end);
    cudaEventDestroy(begin);cudaEventDestroy(end);
    return int(error);
}
