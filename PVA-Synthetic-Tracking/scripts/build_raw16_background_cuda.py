"""Build the separate opt-in RAW16 GPU library; never overwrite an existing build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tiny_target/detection/cuda/raw_background.cu'


def build(directory):
    directory.mkdir(parents=True,exist_ok=False)
    library=directory/'libraw16_background_v7.so'
    command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-lineinfo','--shared',
             '-Xcompiler=-fPIC','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=true',
             '-gencode=arch=compute_87,code=sm_87',str(SOURCE),'-o',str(library)]
    result=subprocess.run(command,text=True,capture_output=True)
    record=dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None)
    with (directory/'build.json').open('x') as handle:json.dump(record,handle,indent=2)
    print(json.dumps(record,indent=2))
    return result.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    raise SystemExit(build(parser.parse_args().output_directory))
