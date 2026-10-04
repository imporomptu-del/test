#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kAbiVersion = 1;
constexpr int kTimingCount = 5;
constexpr int kCounterCount = 7;
constexpr int kLaunchMetricCount = 12;

void set_error(char *buffer, std::size_t capacity, const char *operation,
               cudaError_t status) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(buffer, capacity, "%s: %s", operation,
                  cudaGetErrorString(status));
}

void set_message(char *buffer, std::size_t capacity, const char *message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(buffer, capacity, "%s", message);
}

__global__ void precompute_displacements_kernel(
    const double *time_offsets_s, const float2 *velocities_xy,
    float2 *displacements_xy, int frame_count, int velocity_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = frame_count * velocity_count;
    if (index >= count) {
        return;
    }
    const int velocity_index = index / frame_count;
    const int frame_index = index - velocity_index * frame_count;
    const float2 velocity = velocities_xy[velocity_index];
    const double offset = time_offsets_s[frame_index];
    displacements_xy[index] = make_float2(
        static_cast<float>(static_cast<double>(velocity.x) * offset),
        static_cast<float>(static_cast<double>(velocity.y) * offset));
}

__global__ void initialize_outputs_kernel(
    float *best_score, std::uint16_t *best_velocity,
    std::uint16_t *best_support, std::uint8_t *valid, std::size_t pixel_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    best_score[index] = -CUDART_INF_F;
    best_velocity[index] = 0;
    best_support[index] = 0;
    valid[index] = 0;
}

__device__ bool bilinear_sample(
    const float *frame, const std::uint8_t *mask, int width, int height,
    int output_x, int output_y, float shift_x, float shift_y, float *sample) {
    const int base_x = static_cast<int>(floorf(shift_x));
    const int base_y = static_cast<int>(floorf(shift_y));
    const float fraction_x = shift_x - static_cast<float>(base_x);
    const float fraction_y = shift_y - static_cast<float>(base_y);
    const int offset_x[4] = {base_x, base_x + 1, base_x, base_x + 1};
    const int offset_y[4] = {base_y, base_y, base_y + 1, base_y + 1};
    const float weight[4] = {
        (1.0F - fraction_x) * (1.0F - fraction_y),
        fraction_x * (1.0F - fraction_y),
        (1.0F - fraction_x) * fraction_y,
        fraction_x * fraction_y,
    };
    float value = 0.0F;
    bool used_neighbor = false;
    for (int neighbor = 0; neighbor < 4; ++neighbor) {
        if (weight[neighbor] <= 1.0e-7F) {
            continue;
        }
        used_neighbor = true;
        const int source_x = output_x + offset_x[neighbor];
        const int source_y = output_y + offset_y[neighbor];
        if (source_x < 0 || source_x >= width || source_y < 0 ||
            source_y >= height) {
            return false;
        }
        const std::size_t source_index =
            static_cast<std::size_t>(source_y) * width + source_x;
        if (mask[source_index] == 0) {
            return false;
        }
        value += weight[neighbor] * frame[source_index];
    }
    if (!used_neighbor) {
        return false;
    }
    *sample = value;
    return true;
}

__global__ void shift_and_stack_batch_kernel(
    const float *frames, const std::uint8_t *masks,
    const float2 *displacements_xy, int frame_count, int velocity_start,
    int velocity_count, int width, int height, int minimum_support,
    int polarity_sign, float *best_score, std::uint16_t *best_velocity,
    std::uint16_t *best_support) {
    const std::size_t pixel_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * height;
    if (pixel_index >= pixel_count) {
        return;
    }
    const int output_x = static_cast<int>(pixel_index % width);
    const int output_y = static_cast<int>(pixel_index / width);
    float local_best_score = best_score[pixel_index];
    std::uint16_t local_best_velocity = best_velocity[pixel_index];
    std::uint16_t local_best_support = best_support[pixel_index];
    for (int batch_velocity = 0; batch_velocity < velocity_count;
         ++batch_velocity) {
        const int velocity_index = velocity_start + batch_velocity;
        float accumulator = 0.0F;
        int support = 0;
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const std::size_t frame_offset =
                static_cast<std::size_t>(frame_index) * pixel_count;
            const float2 displacement =
                displacements_xy[velocity_index * frame_count + frame_index];
            float sample = 0.0F;
            if (bilinear_sample(
                    frames + frame_offset, masks + frame_offset, width, height,
                    output_x, output_y, displacement.x, displacement.y,
                    &sample)) {
                accumulator += static_cast<float>(polarity_sign) * sample;
                ++support;
            }
        }
        if (support >= minimum_support) {
            const float score = accumulator / sqrtf(static_cast<float>(support));
            if (score > local_best_score) {
                local_best_score = score;
                local_best_velocity =
                    static_cast<std::uint16_t>(velocity_index);
                local_best_support = static_cast<std::uint16_t>(support);
            }
        }
    }
    best_score[pixel_index] = local_best_score;
    best_velocity[pixel_index] = local_best_velocity;
    best_support[pixel_index] = local_best_support;
}

__global__ void finalize_outputs_kernel(
    float *best_score, std::uint8_t *valid, std::size_t pixel_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    const bool is_valid = isfinite(best_score[index]);
    valid[index] = is_valid ? 1 : 0;
    if (!is_valid) {
        best_score[index] = 0.0F;
    }
}

}  // namespace

extern "C" int tt_cuda_abi_version() { return kAbiVersion; }

extern "C" int tt_synthetic_track(
    const float *host_frames, const std::uint8_t *host_masks,
    const double *host_time_offsets_s, const float *host_velocities_xy,
    int frame_count, int velocity_count, int height, int width,
    int minimum_support, int polarity_sign, int velocity_batch_size,
    int threads_per_block, int device_index, float *host_best_score,
    std::uint16_t *host_best_velocity, std::uint16_t *host_best_support,
    std::uint8_t *host_valid, float *host_timings_ms,
    std::uint64_t *host_counters, std::int32_t *host_launch_metrics,
    char *error_buffer, std::size_t error_capacity) {
    float *device_frames = nullptr;
    std::uint8_t *device_masks = nullptr;
    double *device_time_offsets = nullptr;
    float2 *device_velocities = nullptr;
    float2 *device_displacements = nullptr;
    float *device_best_score = nullptr;
    std::uint16_t *device_best_velocity = nullptr;
    std::uint16_t *device_best_support = nullptr;
    std::uint8_t *device_valid = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t displacement_start = nullptr;
    cudaEvent_t displacement_end = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_end = nullptr;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};
    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    std::size_t free_after_allocations = 0;
    int active_blocks_per_sm = 0;
    int result = -1;

#define TT_CUDA_TRY(operation)                                                \
    do {                                                                      \
        const cudaError_t status = (operation);                               \
        if (status != cudaSuccess) {                                          \
            set_error(error_buffer, error_capacity, #operation, status);      \
            goto cleanup;                                                     \
        }                                                                     \
    } while (false)

    if (error_buffer != nullptr && error_capacity > 0) {
        error_buffer[0] = '\0';
    }
    if (host_timings_ms == nullptr || host_counters == nullptr ||
        host_launch_metrics == nullptr) {
        set_message(error_buffer, error_capacity, "metrics output is null");
        return -1;
    }
    std::memset(host_timings_ms, 0, sizeof(float) * kTimingCount);
    std::memset(host_counters, 0, sizeof(std::uint64_t) * kCounterCount);
    std::memset(host_launch_metrics, 0,
                sizeof(std::int32_t) * kLaunchMetricCount);
    if (host_frames == nullptr || host_masks == nullptr ||
        host_time_offsets_s == nullptr || host_velocities_xy == nullptr ||
        host_best_score == nullptr || host_best_velocity == nullptr ||
        host_best_support == nullptr || host_valid == nullptr) {
        set_message(error_buffer, error_capacity, "input or output pointer is null");
        return -1;
    }
    if (frame_count <= 0 || velocity_count <= 0 || height <= 0 || width <= 0 ||
        minimum_support <= 0 || minimum_support > frame_count ||
        velocity_batch_size <= 0 || threads_per_block <= 0 ||
        threads_per_block > 1024 || (threads_per_block % 32) != 0 ||
        (polarity_sign != 1 && polarity_sign != -1)) {
        set_message(error_buffer, error_capacity, "invalid scalar argument");
        return -1;
    }
    if (frame_count > INT_MAX / velocity_count) {
        set_message(error_buffer, error_capacity,
                    "frame_count times velocity_count exceeds kernel index capacity");
        return -1;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const std::size_t frame_values =
        pixel_count * static_cast<std::size_t>(frame_count);
    const std::size_t displacement_count =
        static_cast<std::size_t>(frame_count) * velocity_count;
    const std::size_t frames_bytes = frame_values * sizeof(float);
    const std::size_t masks_bytes = frame_values * sizeof(std::uint8_t);
    const std::size_t offsets_bytes =
        static_cast<std::size_t>(frame_count) * sizeof(double);
    const std::size_t velocities_bytes =
        static_cast<std::size_t>(velocity_count) * sizeof(float2);
    const std::size_t displacements_bytes = displacement_count * sizeof(float2);
    const std::size_t score_bytes = pixel_count * sizeof(float);
    const std::size_t velocity_index_bytes =
        pixel_count * sizeof(std::uint16_t);
    const std::size_t support_bytes = pixel_count * sizeof(std::uint16_t);
    const std::size_t valid_bytes = pixel_count * sizeof(std::uint8_t);
    const int pixel_blocks = static_cast<int>(
        (pixel_count + threads_per_block - 1) / threads_per_block);
    const int displacement_blocks = static_cast<int>(
        (displacement_count + threads_per_block - 1) / threads_per_block);
    const int batch_count =
        (velocity_count + velocity_batch_size - 1) / velocity_batch_size;

    TT_CUDA_TRY(cudaSetDevice(device_index));
    TT_CUDA_TRY(cudaGetDeviceProperties(&properties, device_index));
    if (threads_per_block > properties.maxThreadsPerBlock) {
        set_message(error_buffer, error_capacity,
                    "threads_per_block exceeds device capability");
        goto cleanup;
    }
    TT_CUDA_TRY(cudaFuncGetAttributes(&attributes,
                                      shift_and_stack_batch_kernel));
    TT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, shift_and_stack_batch_kernel,
        threads_per_block, 0));
    TT_CUDA_TRY(cudaMemGetInfo(&free_before, &total_memory));

    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_frames),
                           frames_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_masks), masks_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_time_offsets),
                           offsets_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_velocities),
                           velocities_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_displacements),
                           displacements_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_best_score),
                           score_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_best_velocity),
                           velocity_index_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_best_support),
                           support_bytes));
    TT_CUDA_TRY(cudaMalloc(reinterpret_cast<void **>(&device_valid), valid_bytes));
    TT_CUDA_TRY(cudaMemGetInfo(&free_after_allocations, &total_memory));

    TT_CUDA_TRY(cudaEventCreate(&h2d_start));
    TT_CUDA_TRY(cudaEventCreate(&h2d_end));
    TT_CUDA_TRY(cudaEventCreate(&displacement_start));
    TT_CUDA_TRY(cudaEventCreate(&displacement_end));
    TT_CUDA_TRY(cudaEventCreate(&kernel_start));
    TT_CUDA_TRY(cudaEventCreate(&kernel_end));
    TT_CUDA_TRY(cudaEventCreate(&d2h_start));
    TT_CUDA_TRY(cudaEventCreate(&d2h_end));

    TT_CUDA_TRY(cudaEventRecord(h2d_start));
    TT_CUDA_TRY(cudaMemcpy(device_frames, host_frames, frames_bytes,
                           cudaMemcpyHostToDevice));
    TT_CUDA_TRY(cudaMemcpy(device_masks, host_masks, masks_bytes,
                           cudaMemcpyHostToDevice));
    TT_CUDA_TRY(cudaMemcpy(device_time_offsets, host_time_offsets_s,
                           offsets_bytes, cudaMemcpyHostToDevice));
    TT_CUDA_TRY(cudaMemcpy(device_velocities, host_velocities_xy,
                           velocities_bytes, cudaMemcpyHostToDevice));
    TT_CUDA_TRY(cudaEventRecord(h2d_end));

    TT_CUDA_TRY(cudaEventRecord(displacement_start));
    precompute_displacements_kernel<<<displacement_blocks, threads_per_block>>>(
        device_time_offsets, device_velocities, device_displacements,
        frame_count, velocity_count);
    TT_CUDA_TRY(cudaGetLastError());
    TT_CUDA_TRY(cudaEventRecord(displacement_end));

    TT_CUDA_TRY(cudaEventRecord(kernel_start));
    initialize_outputs_kernel<<<pixel_blocks, threads_per_block>>>(
        device_best_score, device_best_velocity, device_best_support,
        device_valid, pixel_count);
    TT_CUDA_TRY(cudaGetLastError());
    for (int batch = 0; batch < batch_count; ++batch) {
        const int velocity_start = batch * velocity_batch_size;
        const int remaining = velocity_count - velocity_start;
        const int count =
            velocity_batch_size < remaining ? velocity_batch_size : remaining;
        shift_and_stack_batch_kernel<<<pixel_blocks, threads_per_block>>>(
            device_frames, device_masks, device_displacements, frame_count,
            velocity_start, count, width, height, minimum_support,
            polarity_sign, device_best_score, device_best_velocity,
            device_best_support);
        TT_CUDA_TRY(cudaGetLastError());
    }
    finalize_outputs_kernel<<<pixel_blocks, threads_per_block>>>(
        device_best_score, device_valid, pixel_count);
    TT_CUDA_TRY(cudaGetLastError());
    TT_CUDA_TRY(cudaEventRecord(kernel_end));
    TT_CUDA_TRY(cudaEventSynchronize(kernel_end));

    TT_CUDA_TRY(cudaEventRecord(d2h_start));
    TT_CUDA_TRY(cudaMemcpy(host_best_score, device_best_score, score_bytes,
                           cudaMemcpyDeviceToHost));
    TT_CUDA_TRY(cudaMemcpy(host_best_velocity, device_best_velocity,
                           velocity_index_bytes, cudaMemcpyDeviceToHost));
    TT_CUDA_TRY(cudaMemcpy(host_best_support, device_best_support, support_bytes,
                           cudaMemcpyDeviceToHost));
    TT_CUDA_TRY(cudaMemcpy(host_valid, device_valid, valid_bytes,
                           cudaMemcpyDeviceToHost));
    TT_CUDA_TRY(cudaEventRecord(d2h_end));
    TT_CUDA_TRY(cudaEventSynchronize(d2h_end));

    TT_CUDA_TRY(cudaEventElapsedTime(&host_timings_ms[0], h2d_start, h2d_end));
    TT_CUDA_TRY(cudaEventElapsedTime(&host_timings_ms[1], displacement_start,
                                     displacement_end));
    TT_CUDA_TRY(cudaEventElapsedTime(&host_timings_ms[2], kernel_start,
                                     kernel_end));
    TT_CUDA_TRY(cudaEventElapsedTime(&host_timings_ms[3], d2h_start, d2h_end));
    TT_CUDA_TRY(cudaEventElapsedTime(&host_timings_ms[4], h2d_start, d2h_end));

    host_counters[0] = frames_bytes + masks_bytes + offsets_bytes + velocities_bytes;
    host_counters[1] = score_bytes + velocity_index_bytes + support_bytes + valid_bytes;
    host_counters[2] = frames_bytes + masks_bytes + offsets_bytes + velocities_bytes +
                       displacements_bytes + score_bytes + velocity_index_bytes +
                       support_bytes + valid_bytes;
    host_counters[3] = pixel_count * static_cast<std::uint64_t>(frame_count) *
                       static_cast<std::uint64_t>(velocity_count);
    host_counters[4] = displacements_bytes;
    host_counters[5] = free_before;
    host_counters[6] = free_after_allocations;
    host_launch_metrics[0] = device_index;
    host_launch_metrics[1] = properties.major;
    host_launch_metrics[2] = properties.minor;
    host_launch_metrics[3] = properties.multiProcessorCount;
    host_launch_metrics[4] = threads_per_block;
    host_launch_metrics[5] = active_blocks_per_sm;
    host_launch_metrics[6] = attributes.numRegs;
    host_launch_metrics[7] = static_cast<std::int32_t>(attributes.sharedSizeBytes);
    host_launch_metrics[8] = static_cast<std::int32_t>(attributes.localSizeBytes);
    host_launch_metrics[9] = properties.maxThreadsPerMultiProcessor;
    host_launch_metrics[10] = batch_count;
    host_launch_metrics[11] = batch_count + 3;
    result = 0;

cleanup:
    if (d2h_end != nullptr) cudaEventDestroy(d2h_end);
    if (d2h_start != nullptr) cudaEventDestroy(d2h_start);
    if (kernel_end != nullptr) cudaEventDestroy(kernel_end);
    if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
    if (displacement_end != nullptr) cudaEventDestroy(displacement_end);
    if (displacement_start != nullptr) cudaEventDestroy(displacement_start);
    if (h2d_end != nullptr) cudaEventDestroy(h2d_end);
    if (h2d_start != nullptr) cudaEventDestroy(h2d_start);
    if (device_valid != nullptr) cudaFree(device_valid);
    if (device_best_support != nullptr) cudaFree(device_best_support);
    if (device_best_velocity != nullptr) cudaFree(device_best_velocity);
    if (device_best_score != nullptr) cudaFree(device_best_score);
    if (device_displacements != nullptr) cudaFree(device_displacements);
    if (device_velocities != nullptr) cudaFree(device_velocities);
    if (device_time_offsets != nullptr) cudaFree(device_time_offsets);
    if (device_masks != nullptr) cudaFree(device_masks);
    if (device_frames != nullptr) cudaFree(device_frames);
    return result;

#undef TT_CUDA_TRY
}
