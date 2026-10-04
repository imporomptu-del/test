"""Build explicit isolated CPU helper; no installation or runtime replacement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('learning_mask_v17.cpp')
    library = output/'liblearning_mask_v17.so'
    command = ['c++', '-O3', '-std=c++17', '-fno-fast-math', '-shared', '-fPIC', str(source), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    record = dict(command=command, returncode=result.returncode, stdout=result.stdout,
                  stderr=result.stderr, source_sha256=sha(source), builder_sha256=sha(__file__),
                  library_sha256=sha(library) if result.returncode == 0 else None)
    with (output/'build.json').open('x') as f:
        json.dump(record, f, indent=2)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return library


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(build(p.parse_args().output))
