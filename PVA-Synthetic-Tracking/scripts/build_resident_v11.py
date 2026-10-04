"""Generate an auditable interpolation-stencil variant of the frozen v10 ring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TRACK_SHA='9246bcbe49a8d26ce5da1beedf844985d0be7dda11a7cec83685cc75ef6679f4'
RING_SHA='fe5b5952814c72c94fc4c51d142f84b7994103ff5a32d8bb7e28ccb04c118270'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Frozen transformation anchor changed: '+old)
    return source.replace(old,new,1)


def transformed():
    track=ROOT/'tiny_target/detection/cuda/synthetic_tracking.cu'
    ring=ROOT/'scripts/resident_tracking_v10.cu'
    if sha(track)!=TRACK_SHA or sha(ring)!=RING_SHA:raise ValueError('Frozen CUDA sources changed')
    code=track.read_text()
    sample=code[code.index('__device__ bool bilinear_sample('):code.index('__global__ void shift_and_stack_batch_kernel(')]
    kernel=code[code.index('__global__ void shift_and_stack_batch_kernel('):code.index('__global__ void finalize_outputs_kernel(')]
    sample=once(sample,'bilinear_sample(','bilinear_stencil_v11(')
    sample=once(sample,'float shift_x, float shift_y, float *sample','const StencilV11 &stencil, float *sample')
    start=sample.index('    const int base_x =');end=sample.index('    float value =')
    sample=sample[:start]+'''    const int base_x = stencil.x, base_y = stencil.y;
    const int offset_x[4] = {base_x, base_x + 1, base_x, base_x + 1};
    const int offset_y[4] = {base_y, base_y, base_y + 1, base_y + 1};
    const float *weight = stencil.weight;
'''+sample[end:]
    kernel=once(kernel,'shift_and_stack_batch_kernel(','stack_stencil_v11(')
    kernel=once(kernel,'const float2 *displacements_xy','const StencilV11 *displacements_xy')
    kernel=once(kernel,'const float2 displacement =','const StencilV11 &displacement =')
    kernel=once(kernel,'bilinear_sample(','bilinear_stencil_v11(')
    kernel=once(kernel,'output_x, output_y, displacement.x, displacement.y,','output_x, output_y, displacement,')
    ringcode=ring.read_text()
    ringcode=once(ringcode,'#include "../tiny_target/detection/cuda/synthetic_tracking.cu"','')
    ringcode=once(ringcode,'float2 *grid=nullptr, *displacements=nullptr;','float2 *grid=nullptr; StencilV11 *displacements=nullptr;')
    ringcode=once(ringcode,'frames*velocities*8;','frames*velocities*sizeof(StencilV11);')
    ringcode=once(ringcode,'precompute_displacements_kernel<<<','precompute_stencil_v11<<<')
    ringcode=once(ringcode,'shift_and_stack_batch_kernel<<<','stack_stencil_v11<<<')
    precompute='''
struct StencilV11 { int x,y; float weight[4]; };
__global__ void precompute_stencil_v11(const double *offsets,const float2 *velocities,
    StencilV11 *out,int frames,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=frames*count)return;
    int v=i/frames,f=i-v*frames;
    float sx=static_cast<float>(static_cast<double>(velocities[v].x)*offsets[f]);
    float sy=static_cast<float>(static_cast<double>(velocities[v].y)*offsets[f]);
    int bx=static_cast<int>(floorf(sx)),by=static_cast<int>(floorf(sy));
    float fx=sx-static_cast<float>(bx),fy=sy-static_cast<float>(by);
    StencilV11 s;
    s.x=bx;s.y=by;
    s.weight[0]=(1.0F-fx)*(1.0F-fy);
    s.weight[1]=fx*(1.0F-fy);
    s.weight[2]=(1.0F-fx)*fy;
    s.weight[3]=fx*fy;
    out[i]=s;
}
'''
    return '#include "'+str(track)+'"\n'+precompute+sample+kernel+ringcode


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'generated.cu';source.write_text(transformed())
    library=output/'libresident_tracking_v11.so'
    command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-lineinfo','--shared',
        '-Xcompiler=-fPIC','-gencode=arch=compute_87,code=sm_87',str(source),'-o',str(library)]
    p=subprocess.run(command,capture_output=True,text=True)
    record=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        generated_only=True,production_approved=False,builder_sha256=sha(__file__),
        plan_sha256=sha(ROOT/'docs/raw16_execution_v11_plan.md'),
        reference_tracker_source_sha256=TRACK_SHA,reference_ring_source_sha256=RING_SHA,
        generated_source_sha256=sha(source),library_sha256=sha(library) if library.exists() else None)
    (output/'build.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record,indent=2));return p.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(build(p.parse_args().output))
