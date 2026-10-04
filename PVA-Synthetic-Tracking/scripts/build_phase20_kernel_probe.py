"""Bracket host launches only, from a hash-checked frozen CUDA source set."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SITES = {
    'phase20_cuda_warp_exact.cu': [('gaussian5', 0, 'gaussian_horizontal'),
        ('gaussian5', 1, 'gaussian_vertical'), ('warp_cubic', 2, 'warp_u8'),
        ('warp_cubic', 3, 'warp_f32')],
    'phase20_cuda_median.cu': [('median5', 4, 'median5')],
    'phase20_cuda_resident.cu': [('median5', 4, 'median5'),
        ('residual_prepare', 5, 'residual_prepare'), ('gather_samples', 6, 'gather_samples'),
        ('select_peaks', 7, 'select_peaks'), ('gather_patches', 8, 'gather_patches'),
        ('finish_state', 9, 'finish_state')],
    'phase20_cuda_integrated.cu': [('median5', 4, 'median5'),
        ('residual_prepare', 5, 'residual_prepare'), ('gather_samples', 6, 'gather_samples')],
}
LAUNCH = re.compile(r'\b([a-zA-Z_]\w*)<<<[^;]+;')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(name, source):
    """Use do/while to retain single-statement if/else and early-error behavior."""
    matches = list(LAUNCH.finditer(source))
    sites = SITES[name]
    if [m.group(1) for m in matches] != [s[0] for s in sites]:
        raise ValueError('Unexpected kernel launch topology: ' + name)
    result = source
    for match, (_, site, _) in reversed(list(zip(matches, sites))):
        replacement = ('do { int probe_error=seaqr_kernel_probe::begin(' + str(site) + '); '
            'if(probe_error)return probe_error; ' + match.group(0) +
            ' probe_error=seaqr_kernel_probe::end(); if(probe_error)return probe_error; } while(0);')
        result = result[:match.start()] + replacement + result[match.end():]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-freeze', type=Path, required=True)
    parser.add_argument('--kernel-freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    build_path = output.with_suffix('.so.build.json')
    generated = output.parent / (output.stem + '_sources')
    if output.exists() or build_path.exists() or generated.exists():
        raise ValueError('Never overwrite a profiling build')
    if digest(args.kernel_freeze) != args.kernel_freeze_sha256:
        raise ValueError('Kernel freeze hash changed')
    frozen = json.loads(args.kernel_freeze.read_text())
    derivatives = {}
    for name in SITES:
        if digest(here/name) != frozen['files_sha256']['scripts/'+name]:
            raise ValueError('Frozen kernel source changed: '+name)
        derivatives[name] = instrument(name, (here/name).read_text())
    generated.mkdir()
    for name, contents in derivatives.items():
        with (generated/name).open('x') as f:
            f.write(contents)
    driver = generated/'probe.cu'
    with driver.open('x') as f:
        f.write('#include "'+str(here/'phase20_kernel_events.h')+'"\n#include "phase20_cuda_integrated.cu"\n')
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '--fmad=false', '-arch=sm_87',
        '-Xptxas=-v', '-Xcompiler', '-fPIC', '-shared', str(driver), '-o', str(output)]
    subprocess.run(command, check=True)
    record = dict(diagnostic_only=True, kernel_bodies_unchanged=True,
        instrumentation='Only host launches wrapped in default-stream CUDA events; no per-kernel synchronization',
        kernel_freeze_sha256=digest(args.kernel_freeze), library_sha256=digest(output), command=command,
        compiler_version=subprocess.check_output([command[0], '--version'], text=True),
        sites={str(site):label for sites in SITES.values() for _,site,label in sites},
        sources_sha256={n:digest(here/n) for n in (*SITES, 'phase20_kernel_events.h',
            'build_phase20_kernel_probe.py', 'profile_phase20_kernels.py')},
        generated_sha256={p.name:digest(p) for p in generated.iterdir()})
    with build_path.open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
