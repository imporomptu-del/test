"""Four frozen 8-bit arms with unchanged serial execution and exact gates."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

from profile_visible_v17 import read, sha, write
from combined_v29_protocol import arm_flags, validate_scope

HERE = Path(__file__).resolve().parent
V26 = Path('/tmp/seaqr_visible_front_v26_retry_24sEU7')
V28 = Path('/tmp/seaqr_tracking_v28_Nn629D')
V27 = Path('/tmp/seaqr_tracking_v27_pmUXGZ')
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
SOURCES = ('run_visible_combined_v29.py', 'batch_visible_combined_v29.py',
    'combined_v29_protocol.py', 'visible_combined_v29_plan.md', 'test_combined_v29.py',
    'profile_visible_v17.py', 'tracking_stage_v28.py', 'tracking_batch_v27.py',
    'tracking_geometry_v20.py', 'replay_tracking_v27.py', 'combined_v29_state.py')
BATCH_LIBRARY = V27 / 'build_01/libtracking_batch_v27.so'


def dependencies():
    sys.path[:0] = [str(V26), str(RUNTIME), str(V28)]
    import run_visible_front_v26 as front
    f, g = front.verify_freeze()
    frozen = read(V28 / 'freeze_01.json')
    for path, expected in frozen['files'].items():
        if sha(path) != expected:
            raise ValueError('Changed v28 dependency: ' + path)
    manifest = read(V28 / 'post_run_01.json')
    for name, expected in manifest['files'].items():
        if sha(V28 / name) != expected:
            raise ValueError('Changed v28 evidence: ' + name)
    gate = read(V28 / 'replays_01.json')
    if not gate['passed'] or gate['error'] is not None or len(gate['replays']) != 12:
        raise ValueError('Completed v28 gate required')
    for name in ('tracking_stage_v28.py', 'tracking_batch_v27.py', 'tracking_geometry_v20.py', 'replay_tracking_v27.py'):
        if sha(HERE / name) != sha(V28 / name):
            raise ValueError('Copied adapter differs: ' + name)
    return front, f, g


def prepare():
    dependencies()
    if any((HERE / n).exists() for n in ('freeze.json', 'unit_gate.json', 'unit_gate.log')):
        raise FileExistsError('Fresh v29 freeze required')
    with (HERE / 'unit_gate.log').open('x') as log:
        done = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_combined_v29'],
                              cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    write(HERE / 'unit_gate.json', dict(passed=done.returncode == 0, returncode=done.returncode,
        log_sha256=sha(HERE / 'unit_gate.log'), test_sha256=sha(HERE / 'test_combined_v29.py')))
    if done.returncode:
        raise RuntimeError('v29 harness unit gate failed')
    write(HERE / 'freeze.json', dict(pre_run=True, created_ns=time.time_ns(),
        files={n: sha(HERE / n) for n in SOURCES},
        v26_freeze_sha256=sha(V26 / 'freeze.json'), v28_freeze_sha256=sha(V28 / 'freeze_01.json'),
        v28_manifest_sha256=sha(V28 / 'post_run_01.json'),
        v28_gate_sha256=sha(V28 / 'replays_01.json'), library_sha256=sha(BATCH_LIBRARY),
        baseline_profiles={c: sha(V27 / f'profile_{c}_01.json') for c in ('0126', '0082')},
        unit_gate_sha256=sha(HERE / 'unit_gate.json')))


def verify_freeze():
    front, _, g = dependencies()
    f = read(HERE / 'freeze.json')
    if not f['pre_run'] or set(f['files']) != set(SOURCES) or any(sha(HERE/n) != v for n, v in f['files'].items()):
        raise ValueError('Changed v29 source freeze')
    for key, path in (('v26_freeze', V26/'freeze.json'), ('v28_freeze', V28/'freeze_01.json'),
                      ('v28_manifest', V28/'post_run_01.json'), ('v28_gate', V28/'replays_01.json'),
                      ('library', BATCH_LIBRARY), ('unit_gate', HERE/'unit_gate.json')):
        if f[key+'_sha256'] != sha(path):
            raise ValueError('Changed frozen dependency ' + key)
    for c, value in f['baseline_profiles'].items():
        if sha(V27 / f'profile_{c}_01.json') != value:
            raise ValueError('Changed state baseline')
    unit = read(HERE / 'unit_gate.json')
    if not unit['passed'] or unit['returncode'] or unit['log_sha256'] != sha(HERE/'unit_gate.log'):
        raise ValueError('Failed unit gate')
    return front, f, g


def run(args):
    validate_scope(args.clip, args.arm, args.frames, args.state_audit)
    if args.output.exists() or args.output.with_suffix('.v29.json').exists():
        raise FileExistsError(args.output)
    front, frozen, generated = verify_freeze()
    from run_visible_v17 import verify as verify_baseline
    reference, _ = verify_baseline(args.clip)
    from run_motion_video_v13 import run as baseline
    from visible_stage_v24 import StageBinding, install
    from run_visible_stage_v24 import validate_snapshot
    from learning_mask_v17 import LearningMaskV17
    from tracking_geometry_v20 import GeometryV20
    from tracking_stage_v28 import TrackingStageV28
    from tiny_target.tracking.kalman import KalmanTrackManager
    from tiny_target import visible_baseline as visible, visible_resident as resident, visible_warp_exact as warp
    from visible_front_v26 import ResidentFrontV26, attach_warp
    from video_checks_v19 import check
    from combined_v29_state import StateAudit

    gpu, tracking = arm_flags(args.arm)
    mask = LearningMaskV17(front.V17/'build/liblearning_mask_v17.so')
    geometry = GeometryV20(front.V20/'build/libtracking_geometry_v20.so')
    method = geometry.adapter(KalmanTrackManager.update)
    optimized = TrackingStageV28(BATCH_LIBRARY) if tracking else None
    if optimized is not None:
        method = optimized.adapt(method)
    config = V26/'candidate_config.json' if gpu else front.VISIBLE/'config.json'
    old_run = visible.run
    def configured(source, original_config, output, motion_config, frames):
        if original_config != front.VISIBLE/'config.json':
            raise ValueError('Unexpected configuration interception')
        return old_run(source, config, output, motion_config, frames)
    binding = StageBinding('reference')
    instances, intervals, last = [], [], None
    audit = StateAudit(read(V27/f'profile_{args.clip}_01.json')['replay']['digests']) if args.state_audit else None
    class CapturedFront(ResidentFrontV26):
        def __init__(self, cfg):
            super().__init__(cfg)
            instances.append(self)
    receipt = dict(schema='seaqr.visible-combined-v29.v1', passed=False, error=None,
        clip=args.clip, arm=args.arm, frames=args.frames, state_audit=args.state_audit,
        source_sha256=frozen['files'], freeze_sha256=sha(HERE/'freeze.json'),
        gpu_front=gpu, tracking_stage=tracking, execution_policy='serial_reference',
        config_sha256=sha(config), library_sha256=sha(read(config)['cuda_median_library']),
        geometry_library_sha256=sha(front.V20/'build/libtracking_geometry_v20.so'),
        batch_library_sha256=sha(BATCH_LIBRARY) if tracking else None,
        tracking_transformed_sha256=optimized.transformed_sha256 if tracking else geometry.transformed_sha256,
        raw16_accessed=False, defaults_changed=False, production_approved=False,
        new_accuracy_validated=False, staged_v24_enabled=False, native_motion_v25_enabled=False)
    try:
        with ExitStack() as stack:
            install(stack, binding)
            from tiny_target.visible_decode import VisibleFrameReader
            old_read, old_close = VisibleFrameReader.read, VisibleFrameReader.close
            def read_frame(reader):
                nonlocal last
                now = time.perf_counter()
                if last is not None:
                    intervals.append(1000*(now-last))
                last = now
                value = old_read(reader)
                if value[0] is None:
                    last = None
                return value
            def close_reader(reader):
                nonlocal last
                if last is not None:
                    intervals.append(1000*(time.perf_counter()-last))
                    last = None
                return old_close(reader)
            stack.enter_context(patch.object(VisibleFrameReader, 'read', read_frame))
            stack.enter_context(patch.object(VisibleFrameReader, 'close', close_reader))
            stack.enter_context(patch.object(visible, 'run', configured))
            stack.enter_context(patch.object(resident, 'shape_learning_mask', mask))
            stack.enter_context(patch.object(KalmanTrackManager, 'update', method))
            if gpu:
                stack.enter_context(patch.object(resident, 'VisibleCudaResident', CapturedFront))
                stack.enter_context(patch.object(warp.CudaCubicTranslation, '__call__', attach_warp(warp.CudaCubicTranslation.__call__)))
            if audit is not None:
                audit.install(stack)
            baseline(argparse.Namespace(branch='visible', clip=args.clip, frames=args.frames,
                       injected=False, mode='reuse', output=args.output))
        report = read(args.output/'report.json')
        count = report['frames']
        receipt['comparison'] = check(reference, args.output, count, args.frames is None,
            'candidate' if gpu else 'reference', read(config), sha(config), read(V26/'transition.json'))
        validate_snapshot(binding.snapshot(), count)
        if len(intervals) != count or geometry.fallbacks:
            raise AssertionError('Missing timing or unexpected geometry fallback')
        if tracking:
            if (geometry.calls or not optimized.geometry.calls or optimized.geometry.fallbacks
                    or optimized.innovation_fallbacks or optimized.innovation_tracks != optimized.geometry.track_rows):
                raise AssertionError('Tracking batch path incomplete')
        elif geometry.calls <= 0:
            raise AssertionError('Missing original geometry')
        if gpu:
            if (len(instances) != 1 or instances[0].calls != count or instances[0].device_calls != count
                    or instances[0].finish_calls != count or instances[0].host_calls or instances[0].handle
                    or instances[0].front or mask.calls):
                raise AssertionError('Incomplete GPU front/learning lifecycle')
        elif instances or mask.calls <= 0:
            raise AssertionError('Reference front changed')
        if audit is not None:
            audit.finish(count)
        receipt.update(passed=True, processed_frames=count, fps=report['processed_fps'], wall_s=report['elapsed_seconds'])
    except BaseException as exc:
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt.update(consumer_frame_ms=intervals, execution=binding.snapshot(),
            geometry_calls=geometry.calls, geometry_fallbacks=geometry.fallbacks,
            optimized_tracking=(dict(geometry_batches=optimized.geometry.calls, geometry_tracks=optimized.geometry.track_rows,
                geometry_fallbacks=optimized.geometry.fallbacks, innovation_batches=optimized.innovation_batches,
                innovation_tracks=optimized.innovation_tracks, innovation_fallbacks=optimized.innovation_fallbacks)
                if optimized else None),
            private_state_digests=audit.rows if audit is not None else None,
            native_mask_calls=mask.calls,
            fronts=[dict(calls=x.calls, device_calls=x.device_calls, host_calls=x.host_calls, finish_calls=x.finish_calls,
                         learning_points=x.learning_points, closed=x.front is None and x.handle is None) for x in instances],
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output.with_suffix('.v29.json'), receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--clip'); parser.add_argument('--arm'); parser.add_argument('--frames', type=int)
    parser.add_argument('--state-audit', action='store_true'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    prepare() if args.prepare else run(args)
