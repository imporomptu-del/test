// Experimental float32 9x9 correlation. Not an OpenCV-FFT bit-exact replacement.
#include <cuda_runtime.h>
#include <cstddef>
#include <new>

struct Workspace {
    int h, w;
    float *input = nullptr, *output = nullptr, *kernel = nullptr;
    float normalizer;
};

__global__ void correlate(const float *src, const float *kernel, float *dst,
                          int h, int w, float normalizer) {
    constexpr int bx = 32, by = 8, tw = bx + 8, th = by + 8;
    __shared__ float tile[tw * th];
    int lane = threadIdx.y * bx + threadIdx.x;
    for (int i = lane; i < tw * th; i += bx * by) {
        int x = blockIdx.x * bx + i % tw - 4;
        int y = blockIdx.y * by + i / tw - 4;
        tile[i] = x >= 0 && x < w && y >= 0 && y < h ? src[y * w + x] : 0.f;
    }
    __syncthreads();
    int x = blockIdx.x * bx + threadIdx.x, y = blockIdx.y * by + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.f;
    #pragma unroll
    for (int ky = 0; ky < 9; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 9; ++kx)
            sum = __fmaf_rn(tile[(threadIdx.y + ky) * tw + threadIdx.x + kx],
                            kernel[ky * 9 + kx], sum);
    }
    dst[y * w + x] = __fdiv_rn(sum, normalizer);
}

extern "C" int seaqr_point_v8_abi() { return 1; }
extern "C" const char *seaqr_point_v8_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
extern "C" void seaqr_point_v8_destroy(Workspace *p) {
    if (!p) return;
    cudaFree(p->input); cudaFree(p->output); cudaFree(p->kernel); delete p;
}
extern "C" int seaqr_point_v8_create(int h, int w, const float *kernel,
                                     float norm, Workspace **out) {
    if (!out || !kernel || h < 1 || w < 1 || size_t(h)*w > 32000000 || !(norm > 0))
        return cudaErrorInvalidValue;
    *out = nullptr;
    auto *p = new (std::nothrow) Workspace;
    if (!p) return cudaErrorMemoryAllocation;
    p->h = h; p->w = w; p->normalizer = norm;
    cudaError_t status = cudaMalloc(&p->input, size_t(h)*w*sizeof(float));
    if (status == cudaSuccess) status = cudaMalloc(&p->output, size_t(h)*w*sizeof(float));
    if (status == cudaSuccess) status = cudaMalloc(&p->kernel, 81*sizeof(float));
    if (status == cudaSuccess) status = cudaMemcpy(p->kernel, kernel, 81*sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { seaqr_point_v8_destroy(p); return status; }
    *out = p;
    return cudaSuccess;
}
extern "C" int seaqr_point_v8_run(Workspace *p, const float *src, float *dst) {
    if (!p || !src || !dst) return cudaErrorInvalidValue;
    size_t bytes = size_t(p->h)*p->w*sizeof(float);
    auto status = cudaMemcpy(p->input, src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    correlate<<<dim3((p->w+31)/32, (p->h+7)/8), dim3(32,8)>>>(
        p->input, p->kernel, p->output, p->h, p->w, p->normalizer);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    // Blocking download also surfaces execution errors before the host returns.
    return cudaMemcpy(dst, p->output, bytes, cudaMemcpyDeviceToHost);
}
