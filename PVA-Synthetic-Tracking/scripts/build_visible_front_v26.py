"""Additive CUDA front ABI, frozen original kernels and strict compilation."""
import argparse
from pathlib import Path
import shutil
import subprocess
from profile_visible_v17 import sha,write
from visible_front_v26 import CUDA_SOURCES,REFERENCE_LIBRARY_SHA


def build(sources,reference,output):
    if sha(reference)!=REFERENCE_LIBRARY_SHA:raise ValueError('Changed original CUDA library')
    for n,d in CUDA_SOURCES.items():
        if sha(sources/n)!=d:raise ValueError('Changed frozen CUDA source '+n)
    output.mkdir(parents=True,exist_ok=False);source_dir=output/'source';source_dir.mkdir()
    for n in CUDA_SOURCES:shutil.copyfile(sources/n,source_dir/n)
    driver=Path(__file__).with_name('visible_front_v26.cu');shutil.copyfile(driver,source_dir/driver.name)
    library=output/'candidate.so'
    command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','--fmad=false','--ftz=false',
        '--prec-div=true','--prec-sqrt=true','-arch=sm_87','-Xptxas=-v','-Xcompiler','-fPIC',
        '-shared',str(source_dir/driver.name),'-o',str(library)]
    r=subprocess.run(command,capture_output=True,text=True)
    write(output/'build.json',dict(passed=r.returncode==0,returncode=r.returncode,command=command,
        stdout=r.stdout,stderr=r.stderr,library_sha256=sha(library) if not r.returncode else None,
        reference_library_sha256=REFERENCE_LIBRARY_SHA,original_sources_sha256=CUDA_SOURCES,
        source_sha256={n:sha(source_dir/n) for n in (*CUDA_SOURCES,driver.name)},
        builder_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('visible_front_v26.py')),
        compiler_version=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)))
    if r.returncode:raise RuntimeError(r.stderr)
    print(library,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.sources,a.reference,a.output)
