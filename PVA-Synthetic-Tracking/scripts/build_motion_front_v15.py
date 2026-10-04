"""Build the additive CPU motion-preparation experiment; no installs/defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix('.cpp').with_name('motion_front_v15.cpp')
    library = output/'libmotion_front_v15.so'
    command = ['c++', '-O3', '-std=c++17', '-ffp-contract=off', '-fno-fast-math',
               '-shared', '-fPIC', '-pthread', str(source), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    report = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  source_sha256=sha(source), builder_sha256=sha(__file__),
                  library_sha256=sha(library) if result.returncode == 0 else None,
                  production_approved=False)
    with (output/'build.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(build(parser.parse_args().output))
