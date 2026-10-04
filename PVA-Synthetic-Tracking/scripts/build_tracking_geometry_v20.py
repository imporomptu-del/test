"""Build one isolated strict-arithmetic CPU helper, never install."""
import argparse
from pathlib import Path
import subprocess
from profile_visible_v17 import sha,write


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('tracking_geometry_v20.cpp');library=output/'libtracking_geometry_v20.so'
    cmd=['c++','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    write(output/'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        source_sha256=sha(source),builder_sha256=sha(__file__),library_sha256=sha(library) if r.returncode==0 else None))
    if r.returncode:raise RuntimeError(r.stderr)
    return library


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(build(p.parse_args().output))
