// Standalone memory/UB stress test. Not a numerical oracle or deployment binary.
#include "../../scripts/phase20_native_shapes.cpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

template<class T> struct Guarded {
    static constexpr int margin = 32;
    std::vector<T> storage;
    explicit Guarded(size_t count): storage(count+2*margin, T(123)) {}
    T* data() { return storage.data()+margin; }
    void check() const {
        for (int i=0; i<margin; ++i) {
            assert(storage[i] == T(123));
            assert(storage[storage.size()-1-i] == T(123));
        }
    }
};

int main() {
    std::mt19937 rng(418290);
    for (int trial=0; trial<800; ++trial) {
        const int h=1+rng()%130, w=1+rng()%180;
        const int n=trial%5 == 0 ? 512 : trial%5 == 1 ? 0 : rng()%80;
        std::vector<int32_t> seeds(std::max(1,2*n)), polarities(std::max(1,n));
        std::vector<float> patches(std::max(1,n*289));
        std::vector<uint8_t> mask(h*w);
        for (int i=0; i<n; ++i) {
            seeds[2*i]=rng()%w; seeds[2*i+1]=rng()%h; polarities[i]=rng()%2;
            if (i && trial%3 == 0) { seeds[2*i]=seeds[0]; seeds[2*i+1]=seeds[1]; }
        }
        for (auto& v : patches) v=(int(rng()%27)-13)*.5f;
        for (auto& v : mask) v=trial%3 ? rng()%100 > 0 : 1;
        const auto saved_seeds=seeds, saved_polarities=polarities;
        const auto saved_patches=patches;
        const auto saved_mask=mask;
        Guarded<int32_t> go(n+1), members(n), states(n), ro(n+1), summary(2);
        Guarded<int64_t> pixels(n*289);
        Guarded<float> values(n*289);
        const int code=seaqr_shapes_v1(h,w,n,seeds.data(),polarities.data(),patches.data(),mask.data(),
            go.data(),members.data(),states.data(),ro.data(),pixels.data(),values.data(),summary.data());
        assert(code == 0);
        assert(seeds == saved_seeds && polarities == saved_polarities && mask == saved_mask);
        assert(std::memcmp(patches.data(), saved_patches.data(), patches.size()*sizeof(float)) == 0);
        const int groups=summary.data()[0];
        assert(groups >= 0 && groups <= n && go.data()[groups] == n);
        assert(go.data()[0] == 0 && ro.data()[0] == 0 && ro.data()[groups] <= n*289);
        std::vector<int> seen(n,0);
        for (int g=0; g<groups; ++g) {
            assert(go.data()[g] < go.data()[g+1]);
            assert(ro.data()[g] <= ro.data()[g+1]);
            assert(states.data()[g] >= 0 && states.data()[g] <= 2);
            for (int j=go.data()[g]; j<go.data()[g+1]; ++j) {
                const int i=members.data()[j];
                assert(i >= 0 && i < n); ++seen[i];
            }
            for (int j=ro.data()[g]; j<ro.data()[g+1]; ++j) {
                assert(pixels.data()[j] >= 0 && pixels.data()[j] < int64_t(h)*w);
                if (j != ro.data()[g]) assert(pixels.data()[j-1] < pixels.data()[j]);
            }
        }
        for (int count : seen) assert(count == 1);
        go.check(); members.check(); states.check(); ro.check(); summary.check(); pixels.check(); values.check();
        assert(seaqr_shapes_v1(h,w,513,seeds.data(),polarities.data(),patches.data(),mask.data(),
            go.data(),members.data(),states.data(),ro.data(),pixels.data(),values.data(),summary.data()) != 0);
    }
    std::cout << "800 bounded cases passed with input immutability and output canaries\n";
}
