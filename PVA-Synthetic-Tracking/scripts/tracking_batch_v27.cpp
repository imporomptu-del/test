// Batch the frozen scalar geometry without changing a floating-point operation.
#include "tracking_geometry_v20.cpp"
extern "C" int seaqr_tracking_batch_v27(const double* candidates,const double* means,
    int tracks,int n,int d,double pl,double vl,double* residual,double* position,
    double* velocity,unsigned char* pp,unsigned char* vp,int64_t* counts) {
    if(tracks<0 || tracks>512 || n<0 || n>1024 || int64_t(tracks)*n>262144 ||
       (d!=2 && d!=4) || !candidates || !means || !residual || !position ||
       !velocity || !pp || !vp || !counts)return 2;
    for(int t=0;t<tracks;++t) {
        int status=seaqr_tracking_geometry_v20(candidates,means+4*t,n,d,pl,vl,
            residual+int64_t(t)*n*d,position+int64_t(t)*n,velocity+int64_t(t)*n,
            pp+int64_t(t)*n,vp+int64_t(t)*n,counts+int64_t(t)*3);
        if(status)return status;
    }
    return 0;
}
