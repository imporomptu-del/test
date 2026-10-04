"""Separate CUDA arithmetic units: exact tracking and diagnostic FTZ RAW front end."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from profile_raw16_efficiency import sha,write_json


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    nvcc='/usr/local/cuda/bin/nvcc'
    common=[nvcc,'-O3','-std=c++17','-lineinfo','-Xcompiler=-fPIC','-gencode=arch=compute_87,code=sm_87']
    ring=ROOT/'scripts/resident_tracking_v10.cu';front=ROOT/'scripts/resident_frontend_v10.cu'
    commands=[common+['-c',str(ring),'-o',str(output/'ring.o')],
        common+['--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=true','-c',str(front),'-o',str(output/'front.o')],
        [nvcc,'--shared',str(output/'ring.o'),str(output/'front.o'),'-o',str(output/'libraw_probe_v10.so')]]
    sources=[ring,front]+[ROOT/'tiny_target/detection/cuda'/name for name in (
        'synthetic_tracking.cu','raw_background.cu','warp_translation_v9.cu','point_filter_v8.cu')]
    record=dict(generated_data_only=True,production_approved=False,builder_sha256=sha(__file__),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},steps=[])
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True)
        record['steps'].append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:break
    record['passed']=all(s['returncode']==0 for s in record['steps']) and len(record['steps'])==3
    library=output/'libraw_probe_v10.so'
    record['library_sha256']=sha(library) if library.exists() else None
    write_json(output/'build.json',record);print(json.dumps(record,indent=2))
    return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
