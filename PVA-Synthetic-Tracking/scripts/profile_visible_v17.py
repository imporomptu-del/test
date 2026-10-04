"""Profile frozen 8-bit pipeline only, keeping complete archived journal parity."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
V13 = Path('/tmp/seaqr_video_v13_IyK7eQ')
VISIBLE = Path('/tmp/seaqr_phase20_decode_v10_Iz9RSF')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def verify(clip):
    if clip not in ('0126', '0082'):
        raise ValueError('Profile scope is two existing 8-bit prefixes only')
    reference = V13/'results'/f'visible_{clip}_full_reuse'
    launch = read(reference/'launch.json')
    for name, digest in launch['package_sha256'].items():
        if sha(RUNTIME/'tiny_target'/name) != digest:
            raise ValueError('Frozen visible runtime changed: '+name)
    if sha(VISIBLE/'config.json') != launch['config_sha256']:
        raise ValueError('Frozen visible configuration changed')
    config = launch['configuration']
    if (sha(config['cuda_median_library']) != launch['exact_cuda_stabilization']['library_sha256']
            or sha(config['native_shape_library']) != config['native_shape_library_sha256']):
        raise ValueError('Accelerator identity changed')
    old = read(reference.with_suffix('.execution.json'))
    if sha(V13/'motion_reuse_v12.py') != old['adapter_sha256']:
        raise ValueError('Motion adapter changed')
    return reference, launch


def run(args):
    reference, launch = verify(args.clip)
    if args.output.exists() or args.output.with_suffix('.profile.json').exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(V13), str(RUNTIME), str(RUNTIME/'scripts')]
    from run_motion_video_v13 import run as baseline
    from batch_motion_video_v13 import check_visible
    from tiny_target import visible_baseline as visible, visible_resident as resident
    from tiny_target import visible_warp_exact as warp, visible_shapes_native as shapes
    rows = {}
    profile = cProfile.Profile()
    record = dict(passed=False, error=None, clip=args.clip, frames=128,
                  script_sha256=sha(__file__), plan_sha256=sha(HERE/'visible_speed_v17_plan.md'),
                  baseline_launch_sha256=sha(reference/'launch.json'),
                  raw16_accessed=False, production_changed=False,
                  warning='Instrumented call timings include native work/waits. Nested calls overlap; '
                          'decode runs on a different thread. Not clean FPS or camera-to-alert latency.')
    try:
        with ExitStack() as stack:
            def wrap(owner, name, label):
                original = getattr(owner, name)
                def timed(*a, **kw):
                    start = time.perf_counter()
                    try:
                        return original(*a, **kw)
                    finally:
                        rows.setdefault(label, []).append(1000*(time.perf_counter()-start))
                stack.enter_context(patch.object(owner, name, timed))
            original_init = resident.VisibleCudaResident.__init__
            def init(instance, *a, **kw):
                original_init(instance, *a, **kw)
                for name in ('prepare', 'select', 'patches', 'finish'):
                    wrap(instance.lib, 'seaqr_resident_'+name, 'native.'+name)
            stack.enter_context(patch.object(resident.VisibleCudaResident, '__init__', init))
            for owner, name, label in (
                (resident.VisibleCudaResident, 'update', 'detector.total'),
                (resident, 'tile_noise_statistics', 'detector.noise'),
                (resident, 'shape_learning_mask', 'detector.learning_mask'),
                (resident, 'decode_peak_cells', 'detector.decode_peaks'),
                (shapes.NativeShapes, 'consolidate', 'detector.shapes'),
                (warp.CudaWarpFrame, 'prepare', 'native.prepare_warp'),
                (warp.CudaCubicTranslation, '__call__', 'warp.total'),
                (warp.CudaCubicTranslation, 'gaussian', 'warp.gaussian'),
                (visible.PvaMotion, 'update', 'motion_and_warp.total'),
                (visible.VisibleTracks, 'update', 'tracks.total'),
                (visible.VisibleTracks, 'learning_centers', 'tracks.learning_centers')):
                wrap(owner, name, label)
            profile.enable()
            try:
                baseline(argparse.Namespace(branch='visible', clip=args.clip, frames=128,
                    injected=False, mode='reuse', output=args.output))
            finally:
                profile.disable()
        record['comparison'] = check_visible(reference, args.output, 128, False)
        old = read(reference.with_suffix('.execution.json'))['motion'][:127]
        new = read(args.output.with_suffix('.execution.json'))['motion']
        if [(r['frame'], r['identity']) for r in old] != [(r['frame'], r['identity']) for r in new]:
            raise AssertionError('Non-timing motion identities changed')
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        profile.disable()
        record['calls'] = {k: dict(calls=len(v), total_ms=sum(v), mean_ms=sum(v)/len(v),
                                  maximum_ms=max(v), samples_ms=v) for k, v in rows.items()}
        record['python_profile'] = sorted([dict(file=k[0], line=k[1], function=k[2],
            calls=v[1], self_ms=v[2]*1000, cumulative_ms=v[3]*1000)
            for k, v in pstats.Stats(profile).stats.items()], key=lambda r: -r['self_ms'])
        write(args.output.with_suffix('.profile.json'), record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
