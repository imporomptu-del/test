// Diagnostic-only ABI. Neither this timer nor its overhead is in the runtime library.
extern "C" int seaqr_median_event(void* ptr,int h,int w,int iterations,float* elapsed) {
    if(!ptr || !elapsed || h<1 || w<1 || iterations<1 || iterations>100)return int(cudaErrorInvalidValue);
    auto& p=*static_cast<Workspace*>(ptr);
    if(!p.input || p.size!=size_t(h)*size_t(w)*4)return int(cudaErrorInvalidValue);
    cudaEvent_t begin,end;
    cudaError_t error=cudaEventCreate(&begin);if(error!=cudaSuccess)return int(error);
    error=cudaEventCreate(&end);
    if(error!=cudaSuccess){cudaEventDestroy(begin);return int(error);}
    error=cudaEventRecord(begin);
    for(int i=0;i<iterations && error==cudaSuccess;++i) {
        median5<<<dim3((w+31)/32,(h+7)/8),dim3(32,8)>>>(p.input,p.output,h,w);
        error=cudaGetLastError();
    }
    if(error==cudaSuccess)error=cudaEventRecord(end);
    if(error==cudaSuccess)error=cudaEventSynchronize(end);
    if(error==cudaSuccess)error=cudaEventElapsedTime(elapsed,begin,end);
    cudaEventDestroy(begin);cudaEventDestroy(end);return int(error);
}
