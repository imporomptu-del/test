"""Separate event-instrumented build of a verified candidate's generated kernels."""
import argparse
import json
from pathlib import Path
import subprocess

from build_phase20_kernel_probe import SITES,digest,instrument


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();here=Path(__file__).resolve().parent
    candidate=a.candidate_library.resolve();build_path=candidate.with_suffix('.so.build.json')
    build=json.loads(build_path.read_text())
    if (build['schema']!='seaqr.cuda-peak-gate-build.v1' or build['diagnostic_only'] is not False
            or build['variant']!='threshold_before_local_max' or digest(candidate)!=build['library_sha256']):
        raise ValueError('Candidate build/binary mismatch')
    for name,expected in build['sources_sha256'].items():
        if digest(here/name)!=expected:raise ValueError('Candidate source changed')
    source_dir=candidate.parent/(candidate.stem+'_sources')
    if set(build['generated_sha256'])!=set(SITES):raise ValueError('Unexpected candidate source set')
    for name,expected in build['generated_sha256'].items():
        if digest(source_dir/name)!=expected:raise ValueError('Generated candidate source changed')
    output=a.output.resolve();metadata=output.with_suffix('.so.build.json')
    generated=output.parent/(output.stem+'_sources')
    if output.exists() or metadata.exists() or generated.exists():raise ValueError('Never overwrite a probe')
    generated.mkdir()
    for name in SITES:
        with (generated/name).open('x') as f:f.write(instrument(name,(source_dir/name).read_text()))
    driver=generated/'probe.cu'
    with driver.open('x') as f:
        f.write('#include "'+str(here/'phase20_kernel_events.h')+'"\n#include "phase20_cuda_integrated.cu"\n')
    command=['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_87','-Xptxas=-v',
        '-Xcompiler','-fPIC','-shared',str(driver),'-o',str(output)]
    subprocess.run(command,check=True)
    record=dict(diagnostic_only=True,kernel_bodies_unchanged_from_candidate=True,
        candidate_library_sha256=digest(candidate),candidate_build_sha256=digest(build_path),
        library_sha256=digest(output),command=command,
        compiler_version=subprocess.check_output([command[0],'--version'],text=True),
        sites={str(site):label for sites in SITES.values() for _,site,label in sites},
        sources_sha256={n:digest(here/n) for n in (*SITES,'phase20_kernel_events.h',
            'build_phase20_kernel_probe.py','build_phase20_candidate_probe.py','profile_phase20_kernels.py')},
        generated_sha256={p.name:digest(p) for p in generated.iterdir()})
    with metadata.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
