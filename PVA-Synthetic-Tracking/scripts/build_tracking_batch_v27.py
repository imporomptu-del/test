"""Strict build: include unchanged frozen scalar geometry in batch ABI."""
import argparse
from pathlib import Path
import shutil
import subprocess
from profile_visible_v17 import sha,write

GEOMETRY_SHA='4bfb91cfa7023f0239205bbacfcd9f2b48792bdccdc61daaf05770a2832dc416'

def build(output):
    here=Path(__file__).resolve().parent
    if sha(here/'tracking_geometry_v20.cpp')!=GEOMETRY_SHA:raise ValueError('Changed original geometry source')
    output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    for n in ('tracking_geometry_v20.cpp','tracking_batch_v27.cpp'):shutil.copyfile(here/n,source/n)
    library=output/'libtracking_batch_v27.so'
    command=['g++','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',
             str(source/'tracking_batch_v27.cpp'),'-o',str(library)]
    done=subprocess.run(command,text=True,capture_output=True)
    write(output/'build.json',dict(passed=done.returncode==0,returncode=done.returncode,command=command,
        stdout=done.stdout,stderr=done.stderr,source_sha256={n:sha(source/n) for n in ('tracking_geometry_v20.cpp','tracking_batch_v27.cpp')},
        library_sha256=sha(library) if done.returncode==0 else None,builder_sha256=sha(__file__)))
    if done.returncode:raise RuntimeError(done.stderr)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
