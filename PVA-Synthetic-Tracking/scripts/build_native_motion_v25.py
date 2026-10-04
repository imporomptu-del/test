"""Build an isolated strict-arithmetic, GIL-releasing CPU helper; never install."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('native_motion_v25.cpp');library=output/'libnative_motion_v25.so'
    command=['c++','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library)]
    done=subprocess.run(command,capture_output=True,text=True)
    receipt=dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,
        source_sha256=sha(source),builder_sha256=sha(__file__),library_sha256=sha(library) if done.returncode==0 else None)
    with (output/'build.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    if done.returncode:raise RuntimeError(done.stderr)
    return library

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(build(p.parse_args().output))
