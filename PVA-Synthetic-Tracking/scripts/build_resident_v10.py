"""Build the isolated persistent tracker with unchanged reference CUDA arithmetic."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from profile_raw16_efficiency import sha,write_json


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    source = ROOT/'scripts/resident_tracking_v10.cu'
    library = output/'libresident_tracking_v10.so'
    command = ['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-lineinfo','--shared',
        '-Xcompiler=-fPIC','-gencode=arch=compute_87,code=sm_87',str(source),'-o',str(library)]
    result = subprocess.run(command,capture_output=True,text=True)
    record = dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source,ROOT/'tiny_target/detection/cuda/synthetic_tracking.cu')},
        builder_sha256=sha(__file__),library_sha256=sha(library) if library.exists() else None,
        arithmetic='Same compiler flags as frozen synthetic tracker; no fast-math/FTZ override')
    write_json(output/'build.json',record)
    print(json.dumps(record,indent=2))
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(build(parser.parse_args().output))
