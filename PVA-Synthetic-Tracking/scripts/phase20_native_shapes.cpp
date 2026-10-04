// Bounded CPU shape bookkeeping. No CUDA, labels, tracking state or reductions.
// Compile without fast-math; floating centroid reductions stay in NumPy.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

namespace {
constexpr int radius = 8, side = 17, area = side * side, limit = 512;
using Pixels = std::vector<int64_t>;
bool contains(const Pixels& p, int64_t k) {
    return std::binary_search(p.begin(), p.end(), k);
}
}

extern "C" int seaqr_shapes_abi() { return 1; }

// Outputs have capacities n+1 (offsets), n (members/states), n*289 (pixels/
// values), 2 (summary). Invalid arguments/errors fail closed; no exceptions
// cross the C boundary. Inputs are immutable and every call owns its scratch.
extern "C" int seaqr_shapes_v1(
    int h, int w, int n, const int32_t* seeds, const int32_t* polarity,
    const float* patches, const uint8_t* eligible, int32_t* group_offsets,
    int32_t* members, int32_t* states, int32_t* region_offsets,
    int64_t* pixels, float* values, int32_t* summary) {
    if (h < 1 || w < 1 || n < 0 || n > limit || !seeds || !polarity ||
        !patches || !eligible || !group_offsets || !members || !states ||
        !region_offsets || !pixels || !values || !summary) return 1;
    try {
        std::vector<int64_t> keys(n);
        std::vector<bool> bounded(n), accepted(n, false);
        std::vector<Pixels> footprints(n);
        std::map<int64_t, int> last_window;
        int rejected = 0;
        for (int i = 0; i < n; ++i) {
            const int x = seeds[2*i], y = seeds[2*i+1];
            if (x < 0 || x >= w || y < 0 || y >= h ||
                (polarity[i] != 0 && polarity[i] != 1)) return 2;
            keys[i] = int64_t(y)*w+x;
            bounded[i] = x >= radius && y >= radius && x < w-radius && y < h-radius;
            if (bounded[i]) last_window[keys[i]] = i;
        }
        for (int i = 0; i < n; ++i) {
            if (!bounded[i]) { ++rejected; continue; }
            const int x = seeds[2*i], y = seeds[2*i+1];
            const float sign = polarity[i] == 0 ? 1.f : -1.f;
            // SparseSpatial's duplicate windows use the LAST supplied patch.
            const float* patch = patches + area*last_window.at(keys[i]);
            const float height = sign * patch[radius*side+radius];
            if (height <= 0 || std::isnan(height) || !eligible[keys[i]]) {
                ++rejected; continue;
            }
            const float threshold = 0.5f * height;
            std::array<uint8_t, area> visited{};
            std::array<int, area> queue{};
            int head = 0, tail = 1;
            queue[0] = radius*side+radius;
            visited[queue[0]] = 1;
            bool good = true;
            while (head < tail) {
                const int k = queue[head++], py = k/side, px = k%side;
                const int64_t global = int64_t(y+py-radius)*w + x+px-radius;
                if (py == 0 || py == side-1 || px == 0 || px == side-1 || !eligible[global]) {
                    good = false; break;
                }
                footprints[i].push_back(global);
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                    const int q = (py+dy)*side + px+dx;
                    if (!visited[q] && sign*patch[q] >= threshold) {
                        visited[q] = 1; queue[tail++] = q;
                    }
                }
            }
            if (good) {
                accepted[i] = true;
                std::sort(footprints[i].begin(), footprints[i].end());
            } else { footprints[i].clear(); ++rejected; }
        }
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (polarity[a] != polarity[b]) return polarity[a] < polarity[b];
            return keys[a] < keys[b];
        });
        std::vector<std::vector<int>> groups;
        for (int i : order) {
            int selected = -1;
            if (accepted[i]) for (int g = 0; g < int(groups.size()); ++g) {
                bool compatible = true;
                for (int j : groups[g]) {
                    if (polarity[i] != polarity[j] || !accepted[j] ||
                        !contains(footprints[i], keys[j]) || !contains(footprints[j], keys[i])) {
                        compatible = false; break;
                    }
                }
                if (compatible) { selected = g; break; }
            }
            if (selected < 0) groups.push_back({i});
            else groups[selected].push_back(i);
        }
        std::stable_sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
            return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end());
        });
        int member_count = 0, pixel_count = 0;
        group_offsets[0] = region_offsets[0] = 0;
        for (int g = 0; g < int(groups.size()); ++g) {
            const auto& group = groups[g];
            for (int i : group) members[member_count++] = i;
            group_offsets[g+1] = member_count;
            states[g] = 0;  // Original, unsupported/unbounded singleton.
            Pixels region;
            if (accepted[group[0]]) {
                for (int i : group) region.insert(region.end(), footprints[i].begin(), footprints[i].end());
                std::sort(region.begin(), region.end());
                region.erase(std::unique(region.begin(), region.end()), region.end());
                states[g] = 2;
                std::array<bool, limit> in_group{};
                for (int i : group) in_group[i] = true;
                for (int i = 0; i < n; ++i) if (!in_group[i] && polarity[i] == polarity[group[0]] && contains(region, keys[i])) {
                    states[g] = 1; break;  // Asymmetric neighbor: preserve every member.
                }
                if (states[g] == 2) {
                    int xmin = w, xmax = -1;
                    const int ymin = region.front()/w, ymax = region.back()/w;
                    for (auto k : region) { xmin = std::min(xmin, int(k%w)); xmax = std::max(xmax, int(k%w)); }
                    std::vector<int> windows;
                    // Coordinate reads use the FIRST supplied overlapping patch.
                    // Only windows intersecting this union need be considered.
                    for (int i = 0; i < n; ++i) if (bounded[i] &&
                        seeds[2*i]+radius >= xmin && seeds[2*i]-radius <= xmax &&
                        seeds[2*i+1]+radius >= ymin && seeds[2*i+1]-radius <= ymax) windows.push_back(i);
                    for (auto k : region) {
                        const int x = k%w, y = k/w;
                        bool found = false;
                        for (int i : windows) {
                            const int px = x-seeds[2*i]+radius, py = y-seeds[2*i+1]+radius;
                            if (px >= 0 && px < side && py >= 0 && py < side) {
                                pixels[pixel_count] = k;
                                values[pixel_count++] = patches[area*i + py*side+px];
                                found = true; break;
                            }
                        }
                        if (!found) return 3;
                    }
                }
            }
            region_offsets[g+1] = pixel_count;
        }
        summary[0] = int(groups.size()); summary[1] = rejected;
        return 0;
    } catch (...) { return 4; }
}
