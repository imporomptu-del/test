"""Exclusive CUDA exact-warp build; no system installation."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from profile_raw16_efficiency import sha,write_json

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'tiny_target/detection/cuda/warp_translation_v9.cu'
    library=output/'libwarp_translation_v9.so'
    command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-lineinfo','--shared','-Xcompiler=-fPIC',
        '--fmad=false','--prec-div=true','--ftz=true','-gencode=arch=compute_87,code=sm_87',str(source),'-o',str(library)]
    result=subprocess.run(command,capture_output=True,text=True)
    record=dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        source_sha256=sha(source),builder_sha256=sha(__file__),library_sha256=sha(library) if library.exists() else None)
    write_json(output/'build.json',record);print(record,flush=True)
    return result.returncode

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
