// Restricted finite float32 path; no approximation or floating-point reassociation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static float median(std::vector<float>& values) {
    const auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2) return *mid;
    const float low = *std::max_element(values.begin(), mid);
    const float sum = low + *mid; // NumPy float32 mean of the two central elements.
    return sum / 2.0f;
}

static uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

extern "C" int seaqr_noise_v18(const float* samples, int64_t n,
        const uint8_t* support, int64_t pixels, const int64_t* indices,
        const int64_t* boundaries, int64_t tiles, double floor,
        float* stats, double* sigmas) noexcept {
    if (!samples || !support || !indices || !boundaries || !stats || !sigmas ||
        n <= 0 || n > 32000000 || pixels <= 0 || pixels > 32000000 ||
        tiles <= 0 || tiles > n || !std::isfinite(floor) || floor < 0 ||
        floor > 1e15 || std::signbit(floor)) return -1;
    if (boundaries[0] != 0 || boundaries[tiles] != n) return -2;
    for (int64_t t = 0; t < tiles; ++t)
        if (boundaries[t] < 0 || boundaries[t+1] <= boundaries[t] || boundaries[t+1] > n)
            return -3;
    // Positive status requests the explicitly unchanged numeric fallback.
    // Conservative bounds exclude overflow, subnormals and signed-zero ties.
    for (int64_t k = 0; k < n; ++k) {
        if (indices[k] < 0 || indices[k] >= pixels) return -4;
        if (!support[indices[k]]) continue;
        // Inspect bits: the host FP mode can flush a subnormal comparison to
        // zero. It must still take the unchanged numeric fallback, not pass.
        const uint32_t raw = bits(samples[k]), magnitude = raw & 0x7fffffffU;
        if (magnitude >= 0x7f800000U || raw == 0x80000000U ||
            magnitude > bits(1e15f) || (magnitude && magnitude < bits(1e-15f))) return 1;
    }
    try {
        std::vector<float> values;
        for (int64_t t = 0; t < tiles; ++t) {
            values.clear();
            for (int64_t k = boundaries[t]; k < boundaries[t+1]; ++k)
                if (support[indices[k]]) values.push_back(samples[k]);
            float center = 0.0f;
            double sigma = std::max(floor, 0.0);
            if (!values.empty()) {
                center = median(values);
                for (float& x : values) x = std::fabs(x - center);
                sigma = std::max(floor, 1.4826 * static_cast<double>(median(values)));
            }
            stats[2*t] = center;
            stats[2*t+1] = static_cast<float>(sigma);
            sigmas[t] = sigma;
        }
        return 0;
    } catch (...) { return -5; }
}
