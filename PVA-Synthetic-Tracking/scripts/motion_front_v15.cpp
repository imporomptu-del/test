// Fused original per-pixel mapping + measured VPI half-scale 2x2 interpolation.
// No source mutation, detector input conversion or changed motion geometry.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <thread>

static uint32_t mapped(uint16_t value, float scale, float offset) {
    float x = static_cast<float>(value) * scale;
    x = x + offset; // compile with fp contraction disabled: matches NumPy ops
    x = std::min(65535.0f, std::max(0.0f, x));
    return static_cast<uint32_t>(std::nearbyint(x));
}

template<class T> static void rows(const T *input, T *output, int width,
                                    int begin, int end, float scale, float offset) {
    for (int y = begin; y < end; ++y) {
        const T *a = input + static_cast<int64_t>(2*y)*width;
        const T *b = a + width;
        for (int x = 0; x < width/2; ++x) {
            uint32_t sum;
            if constexpr(sizeof(T) == 1) {
                sum = static_cast<uint32_t>(a[2*x])+a[2*x+1]+b[2*x]+b[2*x+1];
            } else {
                sum = mapped(a[2*x], scale, offset)+mapped(a[2*x+1], scale, offset)
                      +mapped(b[2*x], scale, offset)+mapped(b[2*x+1], scale, offset);
            }
            output[static_cast<int64_t>(y)*(width/2)+x] = static_cast<T>((sum+2)/4);
        }
    }
}

extern "C" int seaqr_motion_front_v15(const void *input, void *output, int height,
                                      int width, int bytes, float scale, float offset) {
    if (!input || !output || input == output || height <= 0 || width <= 0 ||
        height % 2 || width % 2 || static_cast<int64_t>(height)*width > 32000000 ||
        (bytes != 1 && bytes != 2) || !std::isfinite(scale) || !std::isfinite(offset) ||
        scale < 0 || std::fegetround() != FE_TONEAREST) return 1;
    try {
        const int middle = height/4;
        if (bytes == 1) {
            auto fn = [&](int begin, int end) { rows(static_cast<const uint8_t *>(input),
                static_cast<uint8_t *>(output), width, begin, end, scale, offset); };
            std::thread worker(fn, 0, middle);
            fn(middle, height/2);
            worker.join();
        } else {
            auto fn = [&](int begin, int end) { rows(static_cast<const uint16_t *>(input),
                static_cast<uint16_t *>(output), width, begin, end, scale, offset); };
            std::thread worker(fn, 0, middle);
            fn(middle, height/2);
            worker.join();
        }
    } catch (...) { return 2; }
    return 0;
}
