"""Isolated Jetson tracker shadow with unchanged, audited native detector input.

Only the three explicitly exposed8-bit sources in plan.json may be opened.
All writes are exclusive inside this freshly packaged experiment directory.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LEGACY = Path('/tmp/seaqr_accuracy_v56_GFs5W1')
LEGACY_RUNNER_SHA = '474fe81b8e8f920228ad7f86132b3d8813ed15e9105f8f365871b5055671a678'
BRIDGE_SHA = '19ec2b84b7786c8fb95a95547de0b60bb0f925992134e9f70c66593702c56baa'
ALLOWED = {
    '0029': (687, '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359'),
    '0126': (674, 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'),
    '0055': (689, 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), 'Nonfinite JSON number')
        return parsed
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def scheduled(spec, frame):
    return any(first <= frame <= last for first, last in spec['weak_windows_inclusive'])


def validate_plan(plan):
    require(plan['schema'] == 'seaqr.weak-continuation-shadow.plan.v1' and set(plan['clips']) == set(ALLOWED), 'Plan scope differs')
    require(plan['full_causal_replay'] is True and plan['concurrent_workers'] == 1, 'Causal serial replay required')
    require(plan['capture']['max_total_capture_pixels_per_frame'] == 8000000, 'Resource bound changed')
    for clip, (frames, digest) in ALLOWED.items():
        spec = plan['clips'][clip]
        require(spec['frames'] == frames and spec['source_sha256'] == digest, 'Media identity differs')
        last = -1
        for pair in spec['weak_windows_inclusive']:
            require(isinstance(pair, list) and len(pair) == 2 and all(type(x) is int for x in pair), 'Invalid windows')
            require(last < pair[0] <= pair[1] < frames, 'Overlapping/invalid windows')
            last = pair[1]
    require(plan['shadow']['weak_budget_per_strong_gap'] == 1
            and plan['shadow']['weak_measurement_covariance_multiplier'] == 2.0, 'Weak hypothesis changed')


def load_dependencies():
    require(os.geteuid() != 0, 'Never run as root')
    path = LEGACY/'run_accuracy_v56_diagnostic.py'
    require(sha(path) == LEGACY_RUNNER_SHA and sha(LEGACY/'build/capture.so') == BRIDGE_SHA, 'Legacy dependency changed')
    spec = importlib.util.spec_from_file_location('verified_v56_dependencies', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    frozen = legacy.verify()
    baseline, runtime_info = legacy.dependencies()
    # New helper modules take precedence, but the production package remains the
    # original dependency already imported by legacy.verify()/baseline.freeze.
    sys.path.insert(0, str(HERE))
    return legacy, baseline, runtime_info, frozen


def verify_transfer():
    manifest = read(HERE/'transfer_manifest.json')
    require(manifest['schema'] == 'seaqr.weak-continuation-shadow.transfer.v1', 'Transfer schema differs')
    for name, digest in manifest['files_sha256'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Nonliteral bundle path')
        require(sha(HERE/name) == digest, 'Bundle changed: '+name)
    plan = read(HERE/'plan.json')
    validate_plan(plan)
    return manifest, plan


def prepare():
    require(not (HERE/'freeze.json').exists(), 'Existing freeze preserved')
    manifest, plan = verify_transfer()
    legacy, _, runtime_info, old = load_dependencies()
    runtime = runtime_info()
    legacy.runtime_check(runtime, old['runtime_reference'])
    tests = ['test_weak_continuation_information_v1', 'test_weak_continuation_shadow_v1',
             'test_run_weak_continuation_shadow_v1', 'test_audit_weak_continuation_shadow_v1',
             'test_weak_shadow_lifecycle_v1', 'test_summarize_weak_continuation_shadow_v1']
    with (HERE/'generated_tests.log').open('x') as log:
        done = subprocess.run([sys.executable, '-W', 'error::ResourceWarning', '-m', 'unittest', *tests, '-q'],
                              cwd=HERE, env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
                              stdout=log, stderr=subprocess.STDOUT)
    require(done.returncode == 0, 'Generated tests failed; no media opened')
    verify_transfer()
    write(HERE/'freeze.json', dict(schema='seaqr.weak-continuation-shadow.freeze.v1', pre_run=True,
        files_sha256=manifest['files_sha256'], transfer_sha256=sha(HERE/'transfer_manifest.json'),
        legacy_freeze_sha256=sha(LEGACY/'freeze.json'), bridge_sha256=BRIDGE_SHA,
        plan_sha256=sha(HERE/'plan.json'), runtime_reference=runtime,
        generated_tests_sha256=sha(HERE/'generated_tests.log')))


def verify():
    manifest, plan = verify_transfer()
    frozen = read(HERE/'freeze.json')
    require(frozen['pre_run'] is True and frozen['files_sha256'] == manifest['files_sha256'], 'Freeze changed')
    for key, path in [('transfer_sha256', HERE/'transfer_manifest.json'), ('plan_sha256', HERE/'plan.json'),
                      ('legacy_freeze_sha256', LEGACY/'freeze.json'), ('generated_tests_sha256', HERE/'generated_tests.log')]:
        require(sha(path) == frozen[key], 'Frozen dependency changed: '+key)
    require(frozen['bridge_sha256'] == BRIDGE_SHA, 'Bridge differs')
    return frozen, plan


def rectangle_key(rectangle):
    return tuple(rectangle['capture_bounds_exclusive_xyxy'])+tuple(rectangle['tile_bounds_exclusive_xyxy'])


class Experiment:
    def __init__(self, arm, clip, spec, output, legacy, bridge, config, fps):
        self.arm, self.clip, self.spec, self.output = arm, clip, spec, output
        self.legacy, self.bridge = legacy, bridge
        self.context = self.active = self.learning = None
        self.pending, self.forecasts, self.rows = {}, [], []
        self.before = self.after = None
        self.capture_count = 0
        self.trace = None
        self.cost_ms = dict(capture=0.0, shadow=0.0, snapshot_write=0.0)
        self.shadow = None
        if arm == 'shadow':
            from weak_continuation_shadow_v1 import WeakContinuationShadow
            self.shadow = WeakContinuationShadow(config, fps)

    def install(self, stack):
        import numpy as np
        from tiny_target import visible_baseline as visible
        import visible_front_v26 as front
        from combined_v29_state import state_of
        from replay_tracking_v27 import digest
        from accuracy_v56_capture import capture_prepared, tile_rectangle
        old_motion, old_detector = visible.PvaMotion.update, visible.VisiblePointDetector.update
        old_front, old_decode, old_track = front.ResidentFrontV26.update, front.decode_peak_cells, visible.VisibleTracks.update
        old_learning = visible.VisibleTracks.learning_centers

        def learning_centers(instance, timestamp, segment):
            require(self.shadow is None or instance is not self.shadow.tracker,
                    'Shadow must never supply detector learning inputs')
            return old_learning(instance, timestamp, segment)

        def motion(instance, gray, index, timestamp):
            result = old_motion(instance, gray, index, timestamp)
            require(index == len(self.rows), 'Frame sequence differs')
            self.context = dict(frame=index, timestamp_ns=timestamp, segment=result[3], matrix=result[2].copy())
            if self.shadow:
                self.forecasts = self.shadow.prepare(index, timestamp, result[3])
            return result

        def detector(instance, image, valid, segment, learning_centers=()):
            require(self.learning is None and self.context is not None, 'Detector lifecycle differs')
            self.learning = digest(learning_centers)
            return old_detector(instance, image, valid, segment, learning_centers)

        def update_front(instance, image, valid, segment, learning_centers=()):
            require(self.active is None, 'Reentrant native front')
            self.active = (instance, image.owner.handle)
            try:
                return old_front(instance, image, valid, segment, learning_centers)
            finally:
                self.active = None

        def decode(peaks):
            index = self.context['frame']
            self.before = self.after = None
            if self.shadow and scheduled(self.spec, index):
                require(not self.pending and self.active is not None, 'Capture lifecycle differs')
                instance, warp = self.active
                rectangles, owners = {}, {}
                for forecast in self.forecasts:
                    if not forecast['query_eligible']:
                        continue
                    x, y = forecast['reference_xy']
                    if not (0 <= x < instance.shape[1] and 0 <= y < instance.shape[0]):
                        continue  # Provider returns explicit unavailable; never a zero-peak observation.
                    rectangle = tile_rectangle(instance.shape, instance.config.tile_size, [x, y], 45)
                    key = rectangle_key(rectangle)
                    rectangles.setdefault(key, rectangle)
                    owners[forecast['identity']] = key
                pixels = sum((r['capture_bounds_exclusive_xyxy'][2]-r['capture_bounds_exclusive_xyxy'][0])*
                             (r['capture_bounds_exclusive_xyxy'][3]-r['capture_bounds_exclusive_xyxy'][1]) for r in rectangles.values())
                require(pixels <= 8000000, 'Capture resource bound exceeded; no tracks silently dropped')
                if rectangles:
                    self.before = self.legacy.native_state(instance)
                    snapshots = {}
                    started = time.perf_counter()
                    for key, rectangle in rectangles.items():
                        snapshots[key] = capture_prepared(instance, self.bridge, rectangle=rectangle,
                            ready=instance.count > instance.config.warmup_frames, warp_handle=warp)
                        snapshots[key]['metadata'].update(frame=index, segment=self.context['segment'], prelearning=True)
                        for value in snapshots[key].values():
                            if isinstance(value, np.ndarray):
                                value.flags.writeable = False
                    self.cost_ms['capture'] += 1000*(time.perf_counter()-started)
                    self.after = self.legacy.native_state(instance)
                    require(self.before == self.after, 'Native capture changed baseline state')
                    self.pending = {identity: snapshots[key] for identity, key in owners.items()}
            return old_decode(peaks)

        def tracking(instance, proposals, index, timestamp, segment, matrix, shape):
            # The shadow independently calls the original strong update. Never
            # intercept it as if it were the baseline detector's owner.
            if self.shadow and instance is self.shadow.tracker:
                return old_track(instance, proposals, index, timestamp, segment, matrix, shape)
            require(index == len(self.rows) and self.learning is not None, 'Baseline tracker lifecycle differs')
            original_proposals = deepcopy(proposals)
            result = old_track(instance, proposals, index, timestamp, segment, matrix, shape)
            baseline_state = digest(state_of(instance))
            self.rows.append(dict(frame=index, output=digest(list(result)), state=baseline_state, learning=self.learning))
            self.learning = None
            if self.shadow:
                self.output.mkdir(exist_ok=True)
                if self.trace is None:
                    self.trace = stack.enter_context((self.output/'shadow_trace.jsonl').open('x'))
                capture_files, archived = [], {}

                def provider(forecast):
                    snapshot = self.pending.get(forecast['identity'])
                    if snapshot is None:
                        return None
                    key = rectangle_key(snapshot['metadata']['rectangle'])
                    if key not in archived:
                        (self.output/'captures').mkdir(exist_ok=True)
                        basename = f'frame_{index:06d}_capture_{len(archived):03d}'
                        npz = self.output/'captures'/(basename+'.npz')
                        metadata = npz.with_suffix('.json')
                        started = time.perf_counter()
                        with npz.open('xb') as stream:
                            np.savez_compressed(stream, **{k:v for k,v in snapshot.items() if k != 'metadata'})
                        meta = deepcopy(snapshot['metadata'])
                        meta.update(frame=index, prelearning=True, baseline_learning_unchanged=True)
                        write(metadata, meta)
                        archived[key] = dict(path=str(npz.relative_to(self.output)), sha256=sha(npz),
                            metadata_path=str(metadata.relative_to(self.output)), metadata_sha256=sha(metadata))
                        self.capture_count += 1
                        self.cost_ms['snapshot_write'] += 1000*(time.perf_counter()-started)
                    capture_files.append(dict(identity=forecast['identity'], **archived[key]))
                    return {k:snapshot[k] for k in ('values', 'flags', 'metadata')}

                prior_write_ms = self.cost_ms['snapshot_write']
                started = time.perf_counter()
                records, metrics = self.shadow.step(deepcopy(original_proposals), matrix, shape, provider)
                self.cost_ms['shadow'] += 1000*(time.perf_counter()-started)-(self.cost_ms['snapshot_write']-prior_write_ms)
                require(digest(state_of(instance)) == baseline_state and proposals == original_proposals,
                        'Shadow changed baseline state or proposals')
                row = dict(frame_index=index, timestamp_ns=timestamp, segment=segment,
                    capture_scheduled=scheduled(self.spec, index), prior_forecasts=self.forecasts,
                    strong_proposals=original_proposals, records=records, metrics=metrics,
                    capture_files=capture_files, native_state_before=self.before, native_state_after=self.after)
                self.trace.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                self.trace.flush()
                self.pending.clear()
            if index % 100 == 0:
                print(self.clip, self.arm, index+1, '/', self.spec['frames'], flush=True)
            return result

        for obj, name, method in ((visible.PvaMotion,'update',motion), (visible.VisiblePointDetector,'update',detector),
                                  (front.ResidentFrontV26,'update',update_front), (front,'decode_peak_cells',decode),
                                  (visible.VisibleTracks,'update',tracking), (visible.VisibleTracks,'learning_centers',learning_centers)):
            stack.enter_context(patch.object(obj, name, method))


def run(clip, arm):
    require(clip in ALLOWED and arm in ('clean', 'shadow'), 'Outside experiment scope')
    frozen, plan = verify()
    legacy, baseline, runtime_info, old = load_dependencies()
    before = runtime_info()
    legacy.runtime_check(before, frozen['runtime_reference'])
    spec = plan['clips'][clip]
    reference = read(HERE/f'reference_{clip}.json')
    media = Path('/home/serg/project/camera_reader_sky/srcsky/chunks')/f'chunk_{clip}.avi'
    require(media.is_file() and not media.is_symlink() and sha(media) == spec['source_sha256'], 'Source identity mismatch')
    output = HERE/clip/arm
    require(not output.exists() and not output.with_suffix('.shadow.json').exists(), 'Existing evidence preserved')
    output.parent.mkdir(exist_ok=True)
    from tiny_target.visible_baseline import VisibleConfig
    config = VisibleConfig(**reference['configuration'])
    import tiny_target
    root = Path(tiny_target.__file__).parent
    require(all(sha(root/p) == h for p,h in reference['package_sha256'].items()), 'Original production package differs')
    bridge = ctypes.CDLL(str(LEGACY/'build/capture.so'), mode=ctypes.RTLD_LOCAL) if arm == 'shadow' else None
    experiment = Experiment(arm, clip, spec, output, legacy, bridge, config, reference['fps'])
    receipt = dict(schema='seaqr.weak-continuation-shadow.run.v1', arm=arm, clip=clip, passed=False, error=None,
        expected_frames=spec['frames'], production_changed=False, weak_learning_enabled=False,
        freeze_sha256=sha(HERE/'freeze.json'), plan_sha256=sha(HERE/'plan.json'), source_sha256=spec['source_sha256'],
        configuration_sha256=reference['config_sha256'], runtime=dict(before=before), trace_sha256=None,
        optimized_counter_scope='Global v28 adapter counters include baseline and shadow tracking in shadow arm; not baseline-only or speed evidence.')
    try:
        with ExitStack() as stack:
            experiment.install(stack)
            baseline.run(argparse.Namespace(clip=clip, arm='combined', frames=None, state_audit=False, output=output))
        after = runtime_info()
        legacy.runtime_check(after, frozen['runtime_reference'], after=True)
        require(len(experiment.rows) == spec['frames'] and experiment.learning is None and not experiment.pending,
                'Incomplete causal replay')
        launch = read(output/'launch.json')
        for key in ('configuration', 'package_sha256', 'code_sha256', 'source_sha256', 'config_sha256', 'motion_config_sha256'):
            require(launch[key] == reference[key], 'Baseline launch differs: '+key)
        require(sha(media) == spec['source_sha256'], 'Source changed during replay')
        verify()
        receipt.update(passed=True, runtime=dict(before=before, after=after))
    except BaseException as exc:
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt.update(processed_frames=len(experiment.rows), baseline_digests=experiment.rows,
                       capture_count=experiment.capture_count, diagnostic_cost_ms=experiment.cost_ms)
        if (output/'shadow_trace.jsonl').is_file():
            receipt['trace_sha256'] = sha(output/'shadow_trace.jsonl')
        write(output.with_suffix('.shadow.json'), receipt)


def batch():
    frozen, plan = verify()
    require(not (HERE/'batch_status.json').exists(), 'Existing batch receipt preserved')
    # Refuse all existing outputs before starting even the first clip.
    require(not any((HERE/clip).exists() for clip in plan['execution_order']), 'Fresh clip directories required')
    rows, passed, error = [], False, None
    try:
        for clip in plan['execution_order']:
            for arm in ('clean', 'shadow', 'audit'):
                command = ([sys.executable, str(HERE/'run_weak_continuation_shadow_v1.py'), arm, '--clip', clip]
                    if arm != 'audit' else [sys.executable, str(HERE/'audit_weak_continuation_shadow_v1.py'),
                        '--directory', str(HERE/clip), '--clip', clip, '--freeze-sha256', sha(HERE/'freeze.json'),
                        '--plan-sha256', frozen['plan_sha256'], '--output', str(HERE/clip/'independent_audit.json')])
                print('Starting', clip, arm, flush=True)
                with (HERE/f'{clip}_{arm}.log').open('x') as log:
                    done = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                rows.append(dict(clip=clip, arm=arm, command=command, returncode=done.returncode))
                write(HERE/f'{clip}_{arm}_completed.json', rows[-1])
                require(done.returncode == 0, clip+' '+arm+' failed; all evidence retained')
                print('Completed', clip, arm, flush=True)
        passed = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(HERE/'batch_status.json', dict(schema='seaqr.weak-continuation-shadow.batch.v1',
            passed=passed, error=error, runs=rows, concurrent_workers=1, production_changed=False))
    command = [sys.executable, str(HERE/'summarize_weak_continuation_shadow_v1.py'),
               '--directory', str(HERE), '--freeze-sha256', sha(HERE/'freeze.json'),
               '--plan-sha256', frozen['plan_sha256'], '--output', str(HERE/'summary.json')]
    with (HERE/'summary.log').open('x') as log:
        done = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    write(HERE/'summary_stage.json', dict(command=command, returncode=done.returncode))
    require(done.returncode == 0, 'Summary failed; completed run evidence retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'batch', 'clean', 'shadow'))
    parser.add_argument('--clip', choices=tuple(ALLOWED))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'batch':
        batch()
    else:
        run(args.clip, args.action)
