"""One-worker diagnostics; clean and traced runs are not speedup candidates."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V20 = Path('/tmp/seaqr_visible_speed_v20_XUf1LR')
V21 = Path('/tmp/seaqr_architecture_v21_XMExvL')
HEADERS = Path('/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nvtx/include')
HOST_SHA = '8c00ed3af86ed04e320868fc938d90a82f1983e14d0c06a41b5d00e854f415ab'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = HERE/'batch_01.json'
    if output.exists() or (HERE/'build.log').exists():
        raise FileExistsError('Do not overwrite or restart a completed/partial diagnostic batch')
    record = dict(passed=False, error=None, rows=[], raw16_accessed=False, defaults_changed=False,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  sources={n:sha(HERE/n) for n in ('batch_gpu_timeline_v22.py',
                           'profile_gpu_timeline_v22.py', 'nvtx_bridge_v22.cpp')},
                  headers={str(p.relative_to(HEADERS)):sha(p) for p in sorted(HEADERS.rglob('*.h'))})
    def execute(command, log):
        start = time.monotonic()
        with (HERE/log).open('x') as handle:
            done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
        row = dict(command=command, log=log, returncode=done.returncode,
                   wall_s=time.monotonic()-start, log_sha256=sha(HERE/log))
        record['rows'].append(row)
        print(json.dumps(row), flush=True)
        if done.returncode:
            raise RuntimeError('Diagnostic command failed: '+log)
    try:
        if sha(V21/'profile_visible_architecture_v21.py') != HOST_SHA:
            raise ValueError('Frozen host wrapper changed')
        execute(['nsys', '--version'], 'nsys_version.log')
        execute(['c++', '-O2', '-shared', '-fPIC', '-I'+str(HEADERS),
                 str(HERE/'nvtx_bridge_v22.cpp'), '-ldl', '-o', str(HERE/'libnvtx_bridge.so')], 'build.log')
        record['bridge_sha256'] = sha(HERE/'libnvtx_bridge.so')
        for clip, mode in [('0126','clean'), ('0126','trace'), ('0082','trace'), ('0082','clean')]:
            prefix = HERE/f'{mode}_{clip}'
            if mode == 'clean':
                command = [sys.executable, str(V20/'run_visible_v20.py'), '--clip', clip,
                           '--mode', 'candidate', '--frames', '128', '--output', str(prefix)]
            else:
                command = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                    '--cpuctxsw=process-tree', '--osrt-threshold=100000', '--cuda-flush-interval=0',
                    '--kill=none', '--force-overwrite=false', '--export=sqlite',
                    '--output='+str(HERE/f'gpu_{clip}'), sys.executable,
                    str(HERE/'profile_gpu_timeline_v22.py'), '--clip', clip,
                    '--output', str(prefix), '--library', str(HERE/'libnvtx_bridge.so'), '--host-sha', HOST_SHA]
            execute(command, f'{mode}_{clip}.log')
            receipt = json.loads(prefix.with_suffix('.v20.json').read_text())
            if not receipt['passed'] or receipt['processed_frames'] != 128:
                raise AssertionError('Frozen candidate output parity failed')
            record['rows'][-1].update(clip=clip, mode=mode,
                receipt_sha256=sha(prefix.with_suffix('.v20.json')), fps=receipt['fps'])
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with output.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
