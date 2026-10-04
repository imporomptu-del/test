"""Exact execution-only change: reject below-threshold pixels before local maxima."""
import argparse
import json
from pathlib import Path
import subprocess

from build_phase20_kernel_probe import SITES, digest

REFERENCE_LIBRARY_SHA256='de9f204e1ce74d025c5e43eef752c55052f0e75f42eb789085a3092edb398f21'

BEFORE = '''        float r=s.temporal[i],absolute=fabsf(r);bool peak=true;
        for(int dy=-2;dy<=2 && peak;dy++)for(int dx=-2;dx<=2;dx++) {
            int xx=x+dx,yy=y+dy;
            if(xx>=0 && xx<s.w && yy>=0 && yy<s.h && fabsf(s.temporal[yy*s.w+xx])>absolute) {peak=false;break;}
        }
        if(!peak)continue;
        float noise=fmaxf(sigma,__fsqrt_rn(s.variance[i]));
        float signed_r=__fmul_rn(sign,__fsub_rn(r,center));
        if(signed_r<__fmul_rn(threshold,noise) || __fmul_rn(sign,s.spatial[i])<__fmul_rn(spatial_threshold,noise))continue;
'''
AFTER = '''        float r=s.temporal[i];
        float noise=fmaxf(sigma,__fsqrt_rn(s.variance[i]));
        float signed_r=__fmul_rn(sign,__fsub_rn(r,center));
        if(signed_r<__fmul_rn(threshold,noise) || __fmul_rn(sign,s.spatial[i])<__fmul_rn(spatial_threshold,noise))continue;
        float absolute=fabsf(r);bool peak=true;
        for(int dy=-2;dy<=2 && peak;dy++)for(int dx=-2;dx<=2;dx++) {
            int xx=x+dx,yy=y+dy;
            if(xx>=0 && xx<s.w && yy>=0 && yy<s.h && fabsf(s.temporal[yy*s.w+xx])>absolute) {peak=false;break;}
        }
        if(!peak)continue;
'''


def transform(source):
    if source.count(BEFORE) != 1:
        raise ValueError('Expected exactly one frozen peak predicate block')
    return source.replace(BEFORE, AFTER, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-freeze', type=Path, required=True)
    parser.add_argument('--kernel-freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-reference', action='store_true', help='Build unchanged kernels plus test-only seed/timer ABI')
    parser.add_argument('--test-candidate', action='store_true', help='Build reordered kernels plus test-only seed/timer ABI')
    args = parser.parse_args()
    if args.test_reference and args.test_candidate:
        raise ValueError('Choose at most one test library type')
    here=Path(__file__).resolve().parent
    output=args.output.resolve()
    metadata=output.with_suffix('.so.build.json')
    generated=output.parent/(output.stem+'_sources')
    if output.exists() or metadata.exists() or generated.exists():
        raise ValueError('Never overwrite an experiment build')
    if digest(args.kernel_freeze)!=args.kernel_freeze_sha256:
        raise ValueError('Kernel freeze hash changed')
    frozen=json.loads(args.kernel_freeze.read_text())
    contents={}
    for name in SITES:
        if digest(here/name)!=frozen['files_sha256']['scripts/'+name]:
            raise ValueError('Reference kernel source changed: '+name)
        contents[name]=(here/name).read_text()
    if not args.test_reference:
        contents['phase20_cuda_resident.cu']=transform(contents['phase20_cuda_resident.cu'])
    generated.mkdir()
    for name, value in contents.items():
        with (generated/name).open('x') as f:
            f.write(value)
    driver=generated/'phase20_cuda_integrated.cu'
    if args.test_reference or args.test_candidate:
        driver=generated/'conformance.cu'
        with driver.open('x') as f:
            f.write('#include "phase20_cuda_integrated.cu"\n#include "'+str(here/'phase20_peak_gate_test.h')+'"\n')
    command=['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_87','-Xptxas=-v',
        '-Xcompiler','-fPIC','-shared',str(driver),'-o',str(output)]
    subprocess.run(command,check=True)
    record=dict(schema='seaqr.cuda-peak-gate-build.v1',
        diagnostic_only=args.test_reference or args.test_candidate,
        variant='reference' if args.test_reference else 'threshold_before_local_max',
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        algorithm_policy_changed=False, kernel_freeze_sha256=digest(args.kernel_freeze),
        library_sha256=digest(output), command=command,
        compiler_version=subprocess.check_output([command[0],'--version'],text=True),
        sources_sha256={n:digest(here/n) for n in (*SITES,'build_phase20_peak_gate.py',
            'build_phase20_kernel_probe.py','phase20_peak_gate_test.h','verify_phase20_peak_gate.py')},
        generated_sha256={p.name:digest(p) for p in generated.iterdir()},
        change='Move unchanged noise/threshold predicates before read-only 5x5 maximum search; all arithmetic, neighborhoods, tie rules, counts and launch geometry retained.')
    with metadata.open('x') as f:
        json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
