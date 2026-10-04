"""Build an isolated opt-in CPU library; never install or replace a runtime."""
import argparse
from pathlib import Path
import subprocess
from profile_visible_v17 import sha, write


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('noise_v18.cpp')
    library = output/'libnoise_v18.so'
    cmd = ['c++', '-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off',
           '-shared', '-fPIC', str(source), '-o', str(library)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    write(output/'build.json', dict(command=cmd, returncode=result.returncode,
          stdout=result.stdout, stderr=result.stderr, source_sha256=sha(source),
          builder_sha256=sha(__file__), library_sha256=sha(library) if not result.returncode else None))
    if result.returncode:
        raise RuntimeError(result.stderr)
    return library


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(build(p.parse_args().output))
