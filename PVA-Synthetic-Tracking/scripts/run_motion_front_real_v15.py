"""Strict generated-gated motion-only replay of two authorized RAW prefixes."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
ARCHIVE = Path('/tmp/seaqr_video_v13_IyK7eQ/results')
sys.path[:0] = [str(HERE), str(RUNTIME), str(RUNTIME/'scripts')]
import cv2
import numpy as np
from motion_front_v15 import DirectMotionV15
from motion_front_pixels_v15 import logical_identity
from motion_reuse_v12 import ReuseMotionV12
from profile_raw16_efficiency import source_paths, sha, compact
from raw16_speed_v8_common import estimate_identity
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion import pva_pyrlk as pva
from tiny_target.motion import GlobalMotionConfig, fit_global_motion

CONFIG = RUNTIME/'configs/evaluation/raw16_motion_v6.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verify_gate(path):
    gate = read(path)
    if (not gate['passed'] or gate['error'] is not None or gate['real_media_read']
            or len(gate['pixels']) != 78 or len(gate['independent']) != 48
            or len(gate['sequences']) != 16 or len(gate['timings']) != 16):
        raise ValueError('Complete generated gate required before source access')
    for name, digest in gate['source_sha256'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Generated gate implementation changed: '+name)
    if sha(CONFIG) != gate['config_sha256'] or sha(Path(pva.__file__)) != gate['runtime_motion_sha256']:
        raise ValueError('Frozen runtime or configuration changed')
    if sha(HERE/'build/libmotion_front_v15.so') != gate['library_sha256']:
        raise ValueError('Native library changed')
    return gate


def run(output, gate_path):
    if output.exists():
        raise FileExistsError(output)
    gate = verify_gate(gate_path)
    output.mkdir()
    cv2.setNumThreads(2)
    cfg = read(CONFIG)
    motion = pva.PvaMotionConfig.from_mapping(cfg['motion'])
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    record = dict(schema='seaqr.motion-front-v15-real.v1', passed=False, error=None,
                  gate_sha256=sha(gate_path), script_sha256=sha(__file__),
                  pipeline_benchmark=False, production_approved=False,
                  scope={'clips': ['0029', '0040'], 'frames_each': 64}, trials=[],
                  timing_boundary='Estimator call only; decode, Frame construction, hashes, fitting, '
                                  'verification and output excluded. No detection/tracking pipeline FPS.')
    try:
        for repeat in range(2):
            for clip in ('0029', '0040'):
                archive = ARCHIVE/f'raw_{clip}_repeat0_reuse'
                archived_motion = read(archive.with_suffix('.execution.json'))['motion']
                archived_frames = read(archive/'source_frames.json')
                archived_fits = read(archive/'global_fit_identities.json')
                if len(archived_frames) != 64 or len(archived_motion) != 63 or len(archived_fits) != 63:
                    raise ValueError('Incomplete archived reference')
                video, stamps = source_paths(clip)  # fixed allowlist, no enumeration
                if video.is_symlink() or stamps.is_symlink():
                    raise ValueError('Redirected source')
                modes = ('reference', 'candidate') if repeat == 0 else ('candidate', 'reference')
                for mode in modes:
                    name = f'raw_{clip}_repeat{repeat}_{mode}'
                    trial = dict(name=name, clip=clip, repeat=repeat, mode=mode, frames=[], pairs=[],
                                 source_path=str(video), timestamps_sha256=sha(stamps), passed=False,
                                 archive_sha256={p.name: sha(p) for p in (archive/'source_frames.json',
                                     archive/'global_fit_identities.json', archive.with_suffix('.execution.json'))},
                                 closed=False, error=None)
                    estimator = (ReuseMotionV12 if mode == 'reference' else DirectMotionV15)(motion)
                    source = FfmpegVideoSource(video, timestamp_csv=stamps, timestamp_policy='require_sidecar',
                                               bit_depth=16, max_frames=64)
                    previous = None
                    try:
                        with closing(iter(source)) as frames:
                            for current in frames:
                                index = len(trial['frames'])
                                observed = dict(frame_index=current.frame_index, pixel_sha256=current.pixel_sha256(),
                                                source_timestamp_ns=current.source_timestamp_ns, timestamp_ns=current.timestamp_ns)
                                if (index >= 64 or current.image.dtype != np.uint16 or current.shape != (3190, 4784)
                                        or observed != archived_frames[index]):
                                    raise AssertionError('Source pixels, order, geometry or timestamps changed')
                                trial['frames'].append(observed)
                                if previous is not None:
                                    started = time.perf_counter()
                                    pairs = estimator.estimate(previous, current)
                                    elapsed = 1000*(time.perf_counter()-started)
                                    fitted = fit_global_motion(pairs, fitter, execution='translation_batched_exact_v1')
                                    identity = compact(pairs)
                                    fit_identity = estimate_identity(fitted)
                                    same = (logical_identity(identity) == logical_identity(archived_motion[index-1]['identity'])
                                            and fit_identity == archived_fits[index-1])
                                    trial['pairs'].append(dict(frame=index, estimator_ms=elapsed, identity=identity,
                                                               fit_identity=fit_identity, substage_ms=pairs.timings_ms,
                                                               archived_exact=same))
                                    if not same:
                                        raise AssertionError('Real prefix point/fit identity changed')
                                    if index % 16 == 0:
                                        print(name, 'frame', index, 'exact', flush=True)
                                if previous is not None and previous.pixel_sha256() != trial['frames'][-2]['pixel_sha256']:
                                    raise AssertionError('Previous source was modified')
                                if current.pixel_sha256() != observed['pixel_sha256']:
                                    raise AssertionError('Current source was modified')
                                previous = current
                        if len(trial['frames']) != 64 or len(trial['pairs']) != 63 or estimator.hits != 62:
                            raise AssertionError('Incomplete prefix or missing cache reuse')
                        trial['passed'] = True
                    except BaseException as exc:
                        trial['error'] = repr(exc)
                        raise
                    finally:
                        estimator.close()
                        trial['closed'] = estimator.closed
                        trial['reuse_hits'], trial['reuse_misses'] = estimator.hits, estimator.misses
                        write(output/(name+'.json'), trial)
                    row = dict(name=name, clip=clip, repeat=repeat, mode=mode, passed=trial['passed'],
                               pairs=len(trial['pairs']), evidence_sha256=sha(output/(name+'.json')),
                               mean_estimator_ms=statistics.mean(p['estimator_ms'] for p in trial['pairs']),
                               median_warm_estimator_ms=statistics.median(p['estimator_ms'] for p in trial['pairs'][1:]),
                               cold_estimator_ms=trial['pairs'][0]['estimator_ms'])
                    record['trials'].append(row)
                    print(json.dumps(row), flush=True)
        record['passed'] = len(record['trials']) == 8 and all(t['passed'] for t in record['trials'])
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        write(output/'batch.json', record)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.gate))
