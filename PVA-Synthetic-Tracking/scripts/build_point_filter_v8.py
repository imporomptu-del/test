"""Build an exclusive experimental CUDA point-filter artifact."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'tiny_target/detection/cuda/point_filter_v8.cu'
    library = output/'libpoint_filter_v8.so'
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-lineinfo', '--shared',
               '-Xcompiler=-fPIC', '--fmad=false', '--prec-div=true', '--ftz=true',
               '-gencode=arch=compute_87,code=sm_87', str(source), '-o', str(library)]
    completed = subprocess.run(command, capture_output=True, text=True)
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(command=command, returncode=completed.returncode, stdout=completed.stdout,
                  stderr=completed.stderr, source_sha256=sha(source), builder_sha256=sha(Path(__file__)),
                  library_sha256=sha(library) if library.exists() else None)
    with (output/'build.json').open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2), flush=True)
    return completed.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(build(parser.parse_args().output))
