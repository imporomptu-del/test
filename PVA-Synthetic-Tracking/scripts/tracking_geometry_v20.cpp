// Exact ordinary-float64 residual geometry only. No association policy here.
#include <cmath>
#include <cstdint>
extern "C" int seaqr_tracking_geometry_v20(const double* candidates,const double* mean,
    int n,int d,double position_limit,double velocity_limit,double* residual,
    double* position,double* velocity,unsigned char* position_pass,
    unsigned char* velocity_pass,int64_t* counts) {
    if(n<0 || n>100000 || (d!=2 && d!=4))return 2;
    if(!std::isfinite(position_limit) || !std::isfinite(velocity_limit))return 1;
    // Preserve NumPy special-value/overflow/underflow behavior via its reference
    // path. Complete this check before producing a successful native result.
    for(int j=0;j<d;++j)if(!std::isfinite(mean[j]))return 1;
    for(int i=0;i<n;++i)for(int j=0;j<d;++j) {
        double x=candidates[4*i+j]-mean[j];
        if(!std::isfinite(x) || std::abs(x)>1e150 || (x!=0 && std::abs(x)<1e-140))return 1;
        residual[d*i+j]=x;
    }
    counts[0]=counts[1]=counts[2]=0;
    for(int i=0;i<n;++i) {
        const double* r=residual+d*i;
        // Explicit scalar products then ordered two-element reduction; no FMA.
        double p0=r[0]*r[0],p1=r[1]*r[1];
        position[i]=std::sqrt((0.0+p0)+p1);
        if(d==4) {
            double v0=r[2]*r[2],v1=r[3]*r[3];
            velocity[i]=std::sqrt((0.0+v0)+v1);
        } else velocity[i]=0.0; // NumPy sum of the empty velocity slice.
        bool p=position[i]<=position_limit,v=velocity[i]<=velocity_limit;
        position_pass[i]=p;velocity_pass[i]=v;
        counts[0]+=!p;counts[1]+=p&&!v;counts[2]+=p&&v;
    }
    return 0;
}
