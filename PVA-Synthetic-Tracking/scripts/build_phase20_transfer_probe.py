"""Build a separate timing-only CUDA library from hash-checked frozen kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = ('phase20_cuda_integrated.cu', 'phase20_cuda_warp_exact.cu',
    'phase20_cuda_resident.cu', 'phase20_cuda_median.cu')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-freeze', type=Path, required=True)
    parser.add_argument('--kernel-freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    build_path = output.with_suffix('.so.build.json')
    if output.exists() or build_path.exists():
        raise ValueError('Never overwrite a profiling build')
    if digest(args.kernel_freeze) != args.kernel_freeze_sha256:
        raise ValueError('Kernel freeze hash changed')
    frozen = json.loads(args.kernel_freeze.read_text())
    for name in SOURCES:
        if digest(here/name) != frozen['files_sha256']['scripts/'+name]:
            raise ValueError('Frozen kernel source changed: '+name)
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '--fmad=false', '-arch=sm_87',
        '-Xcompiler', '-fPIC', '-shared', str(here/'phase20_cuda_transfer_probe.cu'), '-o', str(output)]
    subprocess.run(command, check=True)
    record = dict(diagnostic_only=True, kernel_freeze_sha256=digest(args.kernel_freeze),
        library_sha256=digest(output), command=command,
        compiler_version=subprocess.check_output([command[0], '--version'], text=True),
        sources_sha256={n: digest(here/n) for n in (*SOURCES, 'phase20_cuda_transfer_probe.cu',
            'build_phase20_transfer_probe.py', 'probe_phase20_cuda_transfers.py')})
    with build_path.open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
