"""Hash-checked one-kernel transformation, exhaustive network proof and CUDA builds."""
import argparse
from pathlib import Path
import subprocess
from profile_visible_v17 import sha,write
from median_v19 import SOURCE_SHA,REFERENCE_LIBRARY_SHA,network,transform,proof_source


def proof(output):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'proof.cpp';source.write_text(proof_source())
    cmd=['c++','-O3','-std=c++17',str(source),'-o',str(output/'proof')]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    result=subprocess.run([str(output/'proof')],capture_output=True,text=True,check=True)
    record=dict(passed=True,cases=1<<25,minmax_nodes=len(network()[0]),stdout=result.stdout,
                source_sha256=sha(source),generator_sha256=sha(Path(__file__).with_name('median_v19.py')),command=cmd)
    write(output/'proof.json',record)
    return record


def build(sources,reference,output):
    if sha(reference)!=REFERENCE_LIBRARY_SHA:
        raise ValueError('Unknown baseline GPU library')
    contents={}
    for name,digest in SOURCE_SHA.items():
        if sha(sources/name)!=digest:raise ValueError('Frozen CUDA source changed: '+name)
        contents[name]=(sources/name).read_text()
    output.mkdir(parents=True,exist_ok=False)
    p=proof(output/'proof')
    builds={}
    for variant in ('candidate','reference_probe','candidate_probe'):
        directory=output/variant;directory.mkdir()
        for name,value in contents.items():
            if name=='phase20_cuda_median.cu' and variant!='reference_probe':value=transform(value)
            (directory/name).write_text(value)
        driver=directory/'phase20_cuda_integrated.cu'
        if variant.endswith('_probe'):
            header=Path(__file__).with_name('median_probe_v19.h')
            (directory/header.name).write_text(header.read_text())
            driver=directory/'probe.cu'
            driver.write_text('#include "phase20_cuda_integrated.cu"\n#include "median_probe_v19.h"\n')
        library=output/(variant+'.so')
        cmd=['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_87','-Xptxas=-v',
             '-Xcompiler','-fPIC','-shared',str(driver),'-o',str(library)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        record=dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    library_sha256=sha(library) if not result.returncode else None,
                    generated_sha256={f.name:sha(f) for f in directory.iterdir()})
        write(output/(variant+'.build.json'),record)
        if result.returncode:raise RuntimeError(result.stderr)
        builds[variant]=record
    here=Path(__file__).resolve().parent
    record=dict(passed=True,proof=p,builds=builds,reference_library_sha256=REFERENCE_LIBRARY_SHA,
        original_sources_sha256=SOURCE_SHA,source_sha256={n:sha(here/n) for n in
          ('median_v19.py','build_median_v19.py','median_probe_v19.h')},
        compiler_version=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True))
    write(output/'build.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--proof-only',action='store_true')
    a=p.parse_args()
    if a.proof_only:print(proof(a.output))
    else:build(a.sources,a.reference,a.output)
