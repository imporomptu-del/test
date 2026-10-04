// One bounded, reentrant translation hypothesis-scoring call. No Python API.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int seaqr_translation_score_v25(const double* previous,const double* current,
    int n,const int64_t* samples,int m,double threshold,unsigned char* best_mask,
    int64_t* result,double* median_out) {
    if(!previous || !current || !samples || !best_mask || !result || !median_out ||
       n<1 || n>4096 || m<1 || m>1024)return 2;
    if(!std::isfinite(threshold) || threshold<=0)return 1;
    for(int i=0;i<n*2;++i)for(double v:{previous[i],current[i]}) {
        if(!std::isfinite(v) || v<0 || v>65536 || (v==0 && std::signbit(v)) ||
           double(float(v))!=v)return 1;
    }
    for(int h=0;h<m;++h)if(samples[h]<0 || samples[h]>=n)return 2;
    try {
        std::vector<double> inliers;inliers.reserve(n);
        std::vector<unsigned char> mask(n);
        int best_count=-1;int64_t best_index=-1;
        double best_median=std::numeric_limits<double>::infinity();
        for(int h=0;h<m;++h) {
            const int64_t sample=samples[h];
            const double dx=current[2*sample]-previous[2*sample];
            const double dy=current[2*sample+1]-previous[2*sample+1];
            inliers.clear();
            for(int i=0;i<n;++i) {
                // Same (previous + delta) - current, not delta - (current - previous).
                const double rx=(previous[2*i]+dx)-current[2*i];
                const double ry=(previous[2*i+1]+dy)-current[2*i+1];
                const double x2=rx*rx,y2=ry*ry;
                const double residual=std::sqrt((0.0+x2)+y2);
                mask[i]=residual<=threshold;
                if(mask[i])inliers.push_back(residual);
            }
            const int count=int(inliers.size());
            if(count<best_count)continue; // Cannot win the primary integer score.
            double median=std::numeric_limits<double>::infinity();
            if(count) {
                const int middle=count/2;
                std::nth_element(inliers.begin(),inliers.begin()+middle,inliers.end());
                median=inliers[middle];
                if(count%2==0) {
                    const double lower=*std::max_element(inliers.begin(),inliers.begin()+middle);
                    median=((0.0+lower)+median)/2.0;
                }
            }
            if(count>best_count || median<best_median ||
               (median==best_median && (best_index<0 || sample<best_index))) {
                best_count=count;best_index=sample;best_median=median;
                std::copy(mask.begin(),mask.end(),best_mask);
            }
        }
        result[0]=best_index;result[1]=best_count;*median_out=best_median;
        return 0;
    } catch(...) {return 3;}
}
