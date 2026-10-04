// Exact union of clipped integer disk footprints; no floating-point arithmetic.
#include <cstdint>
#include <cstdlib>
extern "C" int seaqr_learning_mask_v17(const unsigned char* support,
        unsigned char* output, int h, int w, const int64_t* xy, int n,
        const int64_t* offsets, int k) {
    if (!support || !output || support == output || h < 1 || w < 1 ||
        int64_t(h)*w > 32000000 || n < 0 || n > 32000000 || k < 1 || k > 1089 ||
        (n && !xy) || !offsets) return 1;
    for (int j=0; j<n; ++j)
        if (xy[2*j]<0 || xy[2*j]>=w || xy[2*j+1]<0 || xy[2*j+1]>=h) return 2;
    for (int j=0; j<k; ++j)
        if (offsets[2*j]<-16 || offsets[2*j]>16 || offsets[2*j+1]<-16 || offsets[2*j+1]>16) return 3;
    // Canonicalize bool bytes exactly like support & True, including unusual
    // NumPy bool buffers with noncanonical nonzero representations.
    for (int64_t i=0; i<int64_t(h)*w; ++i) output[i] = support[i] != 0;
    for (int j=0; j<n; ++j) {
        const int64_t x=xy[2*j], y=xy[2*j+1];
        for (int d=0; d<k; ++d) {
            const int64_t xx=x+offsets[2*d+1], yy=y+offsets[2*d];
            if (xx>=0 && xx<w && yy>=0 && yy<h) output[yy*w+xx]=0;
        }
    }
    return 0;
}
