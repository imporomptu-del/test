"""Explicit compiler invocation for the optional CPU shape helper; no installs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def build(output):
    output = Path(output).resolve()
    metadata = output.with_suffix(output.suffix + '.build.json')
    if output.exists() or metadata.exists():
        raise ValueError('Never overwrite a compiled experiment')
    source = Path(__file__).resolve().with_name('phase20_native_shapes.cpp')
    compiler = shutil.which('c++')
    if compiler is None:
        raise RuntimeError('An existing C++ compiler is required; no package installation')
    command = [compiler, '-std=c++17', '-O3', '-shared', '-fPIC', '-fno-fast-math',
        '-ffp-contract=off', str(source), '-o', str(output)]
    subprocess.run(command, check=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record = dict(command=command, compiler_version=subprocess.check_output([compiler, '--version'], text=True),
        machine=platform.machine(), source_sha256=digest(source), library_sha256=digest(output),
        builder_sha256=digest(__file__), no_fast_math=True, abi=1)
    with metadata.open('x') as handle:
        json.dump(record, handle, indent=2)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
