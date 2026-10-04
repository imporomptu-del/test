"""V56 read-only probes over the untouched, full-clip V29 combined replay.

Run in a new isolated directory on Jetson. Never modifies archived dependencies,
detector configuration, clocks, media, or existing evidence. Timing is diagnostic.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
V29 = Path('/tmp/seaqr_visible_combined_v29_s8XhL1')
V26 = Path('/tmp/seaqr_visible_front_v26_retry_24sEU7')
V30 = Path('/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ')
FRAMES = list(range(213, 219))
NATIVE = {
    'phase20_cuda_resident.cu': 'cf64594508ef47c599e30c6c41fa2c0ee2888c3afa2cdbf37df783113dc8261a',
    'phase20_cuda_median.cu': '452e1cfe96636cb8dad74dd74471507a1318992ec1c62fa43767e2876ebf05f7',
    'phase20_cuda_warp_exact.cu': '7eb54309d9d0e504ec8d0fd86532346d4549a0ce2d7d2b1d8e0943bda45c8374',
    'phase20_cuda_integrated.cu': 'ea607056117051b9655c28ecd2ee7bf444fc53f344e629a732a1c51cd88a8366',
    'visible_front_v26.cu': 'c6238e263be225889b82443a1686f8f0f318b4b7c2fcad0484171a6d2201ada2',
}
FILES = ('run_accuracy_v56_diagnostic.py', 'batch_accuracy_v56.py', 'accuracy_v56_capture.py',
         'accuracy_v56_capture.cu', 'audit_accuracy_v56_replay.py',
         'test_accuracy_v56_capture.py', 'test_accuracy_v56_replay.py',
         'plan.md', 'probes.json', 'runtime_reference.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def runtime_check(info, expected, after=False):
    keys = ('blas', 'affinity', 'numpy', 'opencv', 'thread_environment', 'clock_ticks')
    require(all(info[k] == expected[k] for k in keys), 'Numerical runtime differs from V34')
    require(len(info['blas']) == 1 and info['blas'][0]['threads'] == 12,
            'Expected original 12-thread BLAS policy')
    require(all(v is None for v in info['thread_environment'].values()), 'Thread environment changed')
    require(not after or info['opencv_threads'] == 2, 'OpenCV post-run workers changed')


def dependencies():
    require(os.geteuid() != 0, 'Never run this replay as root')
    sys.path[:0] = [str(V29), str(V30)]
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    from profile_visible_interaction_v30 import runtime_info
    return baseline, runtime_info


def prepare():
    require(not (HERE/'freeze.json').exists() and not (HERE/'build').exists(), 'Fresh directory required')
    transfer = read(HERE/'transfer_manifest.json')
    require(set(transfer['files_sha256']) == set(FILES), 'Incomplete transfer manifest')
    for name, digest in transfer['files_sha256'].items():
        require(sha(HERE/name) == digest, 'Transferred file mismatch: ' + name)
    baseline, runtime_info = dependencies()
    runtime = runtime_info()
    runtime_check(runtime, read(HERE/'runtime_reference.json'))
    source = V26/'build_03/source'
    for name, digest in NATIVE.items():
        require(sha(source/name) == digest, 'Archived native source mismatch: ' + name)
    build = HERE/'build'
    build.mkdir()
    for name in NATIVE:
        shutil.copyfile(source/name, build/name)
    shutil.copyfile(HERE/'accuracy_v56_capture.cu', build/'accuracy_v56_capture.cu')
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '--fmad=false', '--ftz=false',
               '--prec-div=true', '--prec-sqrt=true', '-arch=sm_87', '-Xptxas=-v',
               '-Xcompiler', '-fPIC', '-shared', str(build/'accuracy_v56_capture.cu'),
               '-o', str(build/'capture.so')]
    done = subprocess.run(command, capture_output=True, text=True)
    write(build/'build.json', dict(command=command, returncode=done.returncode,
          stdout=done.stdout, stderr=done.stderr, passed=done.returncode == 0,
          native_sources_sha256=NATIVE,
          compiler_version=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True)))
    require(done.returncode == 0, 'Diagnostic bridge compilation failed: ' + done.stderr)
    with (HERE/'generated_cuda_smoke.log').open('x') as log:
        smoke = subprocess.run([sys.executable, str(HERE/'test_accuracy_v56_capture.py'),
            '--cuda-smoke', str(V26/'build_03/candidate.so'), str(build/'capture.so')],
            stdout=log, stderr=subprocess.STDOUT)
    require(smoke.returncode == 0, 'Generated CUDA smoke failed; see generated_cuda_smoke.log')
    write(HERE/'freeze.json', dict(schema='seaqr.accuracy-v56-freeze.v1', pre_run=True,
          files_sha256={name: sha(HERE/name) for name in FILES},
          bridge_sha256=sha(build/'capture.so'), plan_sha256=sha(HERE/'plan.md'),
          runner_sha256=sha(__file__), auditor_sha256=sha(HERE/'audit_accuracy_v56_replay.py'),
          runtime_reference=runtime, native_sources_sha256=NATIVE,
          v29_freeze_sha256=sha(V29/'freeze.json'),
          original_library_sha256=sha(V26/'build_03/candidate.so'),
          generated_cuda_smoke_sha256=sha(HERE/'generated_cuda_smoke.log'),
          build_sha256=sha(build/'build.json')))


def verify():
    f = read(HERE/'freeze.json')
    require(f['pre_run'] and set(f['files_sha256']) == set(FILES), 'Incomplete source freeze')
    for name, digest in f['files_sha256'].items():
        require(sha(HERE/name) == digest, 'Frozen file changed: ' + name)
    for name, digest in NATIVE.items():
        require(sha(HERE/'build'/name) == digest, 'Bridge native layout source changed')
    require(sha(HERE/'build/capture.so') == f['bridge_sha256'], 'Bridge changed')
    require(sha(HERE/'build/build.json') == f['build_sha256'], 'Build record changed')
    require(sha(HERE/'build/accuracy_v56_capture.cu') == f['files_sha256']['accuracy_v56_capture.cu'],
            'Compiled diagnostic source differs from frozen source')
    require(sha(HERE/'generated_cuda_smoke.log') == f['generated_cuda_smoke_sha256'], 'Generated gate changed')
    require(sha(V26/'build_03/candidate.so') == f['original_library_sha256'] ==
            'fd689b653175eb9baf3e84259ccccd431ba3ec8aa6c6fa4378a596eb03f8a027', 'Original library changed')
    require(sha(V29/'freeze.json') == f['v29_freeze_sha256'], 'V29 dependency freeze changed')
    return f


def native_state(detector):
    """Hash full B/V/support/eligibility/tile statistics and host selector buffers."""
    import numpy as np
    b = np.empty(detector.shape, np.float32)
    v = np.empty_like(b)
    detector._check(detector.lib.seaqr_resident_debug(detector.handle, b.ctypes.data, v.ctypes.data))
    support = np.empty(detector.shape, np.bool_)
    learn = np.empty_like(support)
    stats = np.empty((len(detector.sigmas), 2), np.float32)
    sigmas = np.empty_like(detector.sigmas)
    detector._check(detector.lib.seaqr_front_v26_debug(
        detector.front, *(a.ctypes.data for a in (support, learn, stats, sigmas))))
    return {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in
            dict(background=b, variance=v, support=support, eligible=learn, stats=stats,
                 sigmas=sigmas, peaks=detector.peaks, counts=detector.counts,
                 searchable=detector.searchable, host_eligible=detector.eligible).items()}


class Audit:
    def __init__(self, arm, probes, bridge):
        self.arm, self.probes, self.bridge = arm, probes, bridge
        self.rows, self.captures, self.context, self.pending = [], [], None, None
        self.learning, self.active = None, None

    def install(self, stack):
        import numpy as np
        from tiny_target import visible_baseline as visible
        from tiny_target.visible_shapes_native import NativeShapes
        import visible_front_v26 as front
        from combined_v29_state import state_of
        from replay_tracking_v27 import digest
        from accuracy_v56_capture import capture_prepared, tile_rectangle
        old_motion, old_detector = visible.PvaMotion.update, visible.VisiblePointDetector.update
        old_front, old_decode = front.ResidentFrontV26.update, front.decode_peak_cells
        old_shape, old_track = NativeShapes.consolidate, visible.VisibleTracks.update

        def motion(instance, gray, index, timestamp):
            result = old_motion(instance, gray, index, timestamp)
            require(index == len(self.rows), 'Causal frame order changed')
            self.context = dict(frame=index, timestamp_ns=timestamp,
                                matrix=result[2].copy(), segment=result[3])
            return result

        def detector(instance, image, valid, segment, learning_centers=()):
            require(self.learning is None and self.context is not None, 'Repeated detector input')
            self.learning = digest(learning_centers)
            return old_detector(instance, image, valid, segment, learning_centers)

        def update_front(instance, image, valid, segment, learning_centers=()):
            require(self.active is None, 'Reentrant resident front')
            self.active = (instance, image.owner.handle)
            try:
                return old_front(instance, image, valid, segment, learning_centers)
            finally:
                self.active = None

        def decode(peaks):
            frame = self.context['frame']
            if self.arm == 'probe' and frame in self.probes:
                require(self.pending is None and self.active is not None, 'Capture lifecycle changed')
                instance, warp = self.active
                spec = self.probes[frame]
                xy = np.asarray([*spec['source_xy'], 1.0], dtype=np.float64)
                ref = self.context['matrix'] @ xy
                point = (ref[:2]/ref[2]).tolist()
                rectangle = tile_rectangle(instance.shape, instance.config.tile_size, point, 7)
                before = native_state(instance)
                snapshot = capture_prepared(instance, self.bridge, rectangle=rectangle,
                    ready=instance.count > instance.config.warmup_frames, warp_handle=warp)
                after = native_state(instance)
                require(before == after, 'Capture mutated exposed native state or host buffers')
                metadata = snapshot.pop('metadata')
                metadata.update(frame=frame, source_probe=spec, source_to_reference=self.context['matrix'].tolist(),
                    segment=self.context['segment'], prelearning=True, native_state_before=before,
                    native_state_after=after, full_exposed_state_unchanged=True,
                    original_library_sha256=sha(instance.lib._name))
                self.pending = dict(snapshot=snapshot, metadata=metadata)
            cells = old_decode(peaks)
            if self.pending is not None:
                self.pending['metadata']['decoded_pre_frame_quota_cells'] = deepcopy(cells)
            return cells

        def shape(instance, proposals, image_shape, seeds, patches, eligible, *, include_support=False):
            if self.pending is not None:
                self.pending['metadata']['pre_shape_post_frame_quota'] = deepcopy(proposals)
            result = old_shape(instance, proposals, image_shape, seeds, patches, eligible,
                               include_support=include_support)
            if self.pending is not None:
                self.pending['metadata']['post_shape'] = deepcopy(result[0])
                self.pending['metadata']['shape_metrics'] = deepcopy(result[1])
            return result

        def tracking(instance, proposals, index, timestamp_ns, segment, matrix, image_shape):
            require(index == len(self.rows) and self.learning is not None, 'Tracking/learning order changed')
            result = old_track(instance, proposals, index, timestamp_ns, segment, matrix, image_shape)
            self.rows.append(dict(frame=index, output=digest(list(result)), state=digest(state_of(instance)), learning=self.learning))
            self.learning = None
            if self.pending is not None:
                metadata = self.pending['metadata']
                metadata.update(tracks=deepcopy(result[0]), tracking_metrics=deepcopy(result[1]))
                npz = HERE/f'captures/frame_{index:06d}.npz'
                meta = npz.with_suffix('.json')
                with npz.open('xb') as stream:
                    np.savez_compressed(stream, **self.pending['snapshot'])
                write(meta, metadata)
                self.captures.append(dict(frame=index, npz_path=str(npz.relative_to(HERE)),
                    metadata_path=str(meta.relative_to(HERE)), npz_sha256=sha(npz), metadata_sha256=sha(meta)))
                self.pending = None
                print('Captured pre-learning frame', index, flush=True)
            if index % 100 == 0:
                print(self.arm, 'completed', index+1, '/674', flush=True)
            return result

        for obj, name, method in ((visible.PvaMotion, 'update', motion),
                (visible.VisiblePointDetector, 'update', detector), (front.ResidentFrontV26, 'update', update_front),
                (front, 'decode_peak_cells', decode), (NativeShapes, 'consolidate', shape),
                (visible.VisibleTracks, 'update', tracking)):
            stack.enter_context(patch.object(obj, name, method))


def run(arm):
    require(arm in ('clean', 'probe'), 'Only the two frozen arms are allowed')
    frozen = verify()
    baseline, runtime_info = dependencies()
    before = runtime_info()
    runtime_check(before, frozen['runtime_reference'])
    output = HERE/arm
    require(not output.exists() and not output.with_suffix('.v56.json').exists(), 'Existing run evidence')
    spec = read(HERE/'probes.json')
    media = Path(spec['source_media']['path'])
    require(str(media) == '/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi'
            and media.is_file() and not media.is_symlink(), 'Source outside frozen V56 scope')
    require(sha(media) == spec['source_media']['sha256'] ==
            'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344',
            'V56 source identity mismatch before decode')
    probes = {r['frame_index']: r for r in spec['reference_probes']}
    require(sorted(probes) == FRAMES, 'Changed fixed capture frames')
    bridge = ctypes.CDLL(str(HERE/'build/capture.so'), mode=ctypes.RTLD_LOCAL) if arm == 'probe' else None
    if arm == 'probe':
        (HERE/'captures').mkdir(exist_ok=False)
    audit = Audit(arm, probes, bridge)
    receipt = dict(schema='seaqr.accuracy-v56-replay.v1', arm=arm, passed=False, error=None,
        identities={key: frozen[key] for key in ('runner_sha256', 'bridge_sha256', 'plan_sha256')},
        frozen_code=frozen['files_sha256'], runtime=dict(before=before), raw16_accessed=False,
        sealed_holdout_accessed=False, algorithm_changed=False, timing_diagnostic_only=True)
    receipt['identities']['freeze_sha256'] = sha(HERE/'freeze.json')
    try:
        with ExitStack() as stack:
            audit.install(stack)
            baseline.run(argparse.Namespace(clip='0126', arm='combined', frames=None, state_audit=False, output=output))
        after = runtime_info()
        runtime_check(after, frozen['runtime_reference'], after=True)
        require(len(audit.rows) == 674 and audit.pending is None and audit.learning is None, 'Incomplete diagnostic lifecycle')
        require([r['frame'] for r in audit.captures] == (FRAMES if arm == 'probe' else []), 'Incomplete captures')
        verify()
        receipt.update(passed=True, runtime=dict(before=before, after=after))
    except BaseException as exc:
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt.update(processed_frames=len(audit.rows), capture_frames=[r['frame'] for r in audit.captures],
                       captures=audit.captures, private_state_digests=audit.rows)
        write(output.with_suffix('.v56.json'), receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'clean', 'probe'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run(args.action)
