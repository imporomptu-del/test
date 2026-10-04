"""Generated and saved-journal v28 checks; no camera or media access."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from profile_visible_v17 import read, sha, write
from replay_tracking_v27 import execute, digest
from tracking_geometry_v20 import GeometryV20
from tracking_batch_v27 import BatchGeometryV27
from tracking_stage_v28 import TrackingStageV28


def generated(library, geometry):
    from tiny_target.detection import CandidateBatch, CandidateRecord
    from tiny_target.tracking.kalman import KalmanTrackManager, KalmanTrackingConfig

    helper = GeometryV20(geometry)
    stage = TrackingStageV28(library)
    method = stage.adapt(helper.adapter(KalmanTrackManager.update))
    rows = []
    for scenario in range(36):
        rng = np.random.default_rng(28000 + scenario)
        position_only = scenario % 2 == 0
        gaussian = position_only or scenario % 3 == 0
        appearance = ('none', 'log_response', 'log_response_coast')[(scenario // 2) % 3] if position_only else 'none'
        cascade = 'confirmed_first' if scenario % 6 == 1 else 'none'
        cfg = KalmanTrackingConfig(
            position_measurement_sigma_px=1., velocity_measurement_sigma_px_s=1.,
            acceleration_process_sigma_px_s2=1., initial_position_sigma_px=2.,
            initial_velocity_sigma_px_s=2., mahalanobis_gate_squared=16.,
            maximum_position_residual_px=6., maximum_velocity_residual_px_s=3.,
            confirmation_independent_hits=2, max_missed_windows=3,
            maximum_timestamp_gap_s=2., measurement_noise_source='synthetic_characterization',
            max_active_tracks=24, measurement_model='position_only' if position_only else 'position_velocity',
            association_cost='gaussian_nll' if gaussian else 'mahalanobis',
            birth_policy='spatial_fair' if scenario % 3 else 'input_order', birth_cell_size_px=8.,
            association_cascade=cascade,
            association_assignment='global_min_cost' if scenario % 6 == 0 else 'greedy',
            association_prior='hit_maturity' if gaussian and cascade == 'none' and scenario % 4 == 0 else 'none',
            association_appearance=appearance)
        reference, candidate = KalmanTrackManager(cfg), KalmanTrackManager(cfg)
        hashes = []
        for frame in range(40):
            points = []
            if frame % 11 != 8:
                for i in range(36):
                    if (i + frame) % 7 == 0:
                        continue
                    x, y = rng.normal(0, .2, 2) + np.array([(i % 8) * 4 + frame * .15, (i // 8) * 5])
                    if scenario % 4 == 1:
                        x, y = 10., 10.
                    score = 9. + i % 3
                    points.append(CandidateRecord(
                        candidate_index=len(points), x_px=float(x), y_px=float(y), velocity_index=0,
                        velocity_xy_px_s=(1.5, 0.), normalized_score_snr=score, raw_sum_score=2*score,
                        supporting_frame_count=1, support_weight=1., peak_neighbor_max_score_snr=None,
                        peak_contrast_snr=None, peak_to_neighbor_ratio=None, distance_to_border_px=10,
                        distance_to_invalid_chebyshev_px=None, distance_to_invalid_is_lower_bound=True))
            batch = CandidateBatch(candidates=tuple(points), frame_indices=(frame,),
                reference_timestamp_ns=frame*100000000 + (3000000000 if frame >= 25 else 0),
                segment_index=int(frame >= 17), metrics={}, timings_ms={})
            a = reference.update(batch, include_quality_evidence=bool(scenario % 2)).to_dict()
            b = method(candidate, batch, include_quality_evidence=bool(scenario % 2)).to_dict()
            a.pop('timings_ms'); b.pop('timings_ms')
            if digest(a) != digest(b) or digest(vars(reference)) != digest(vars(candidate)):
                raise AssertionError(f'Generated output/private state changed: {scenario=} {frame=}')
            hashes.append(digest([a, vars(reference)]))
        rows.append(dict(scenario=scenario, frames=40, exact=True, digest=digest(hashes)))
    return dict(scenarios=rows, innovation_batches=stage.innovation_batches,
        innovation_tracks=stage.innovation_tracks, innovation_fallbacks=stage.innovation_fallbacks)


def counters(adapter):
    if adapter is None:
        return {}
    geometry = adapter.geometry if isinstance(adapter, TrackingStageV28) else adapter
    result = dict(geometry_batches=geometry.calls, geometry_tracks=geometry.track_rows,
                  geometry_fallbacks=geometry.fallbacks)
    if isinstance(adapter, TrackingStageV28):
        result.update(innovation_batches=adapter.innovation_batches,
                      innovation_tracks=adapter.innovation_tracks,
                      innovation_fallbacks=adapter.innovation_fallbacks)
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = read(args.freeze)
    for path, expected in freeze['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen dependency changed: ' + path)
    record = dict(passed=False, error=None, media_read=False, defaults_changed=False,
        full_pipeline_tested=False, freeze_sha256=sha(args.freeze), started_ns=time.time_ns(),
        python=platform.python_version(), numpy=np.__version__, generated=None, replays=[], profiles=[])
    try:
        record['generated'] = generated(args.library, args.geometry)
        print('36 generated output/state scenarios passed', flush=True)
        for clip in ('0126', '0082'):
            parent = args.parents / (clip + '_repeat0_reference')
            baseline = read(args.baselines / f'profile_{clip}_01.json')['replay']
            for repeat in range(2):
                modes = ('v20', 'v27', 'v28') if repeat == 0 else ('v28', 'v27', 'v20')
                for mode in modes:
                    adapter = (TrackingStageV28(args.library) if mode == 'v28' else
                               BatchGeometryV27(args.library) if mode == 'v27' else None)
                    replay = execute(parent, args.geometry, False, adapter)
                    if replay['digests'] != baseline['digests'] or replay['populations'] != baseline['populations']:
                        raise AssertionError('Replay state/learning/population changed')
                    replay.update(mode=mode, repeat=repeat, counters=counters(adapter))
                    record['replays'].append(replay)
                    print(clip, repeat, mode, float(np.mean(replay['tracking_ms'])), flush=True)
            adapter = TrackingStageV28(args.library)
            profile = execute(parent, args.geometry, True, adapter)
            if profile['digests'] != baseline['digests']:
                raise AssertionError('Profiled state/learning changed')
            profile['counters'] = counters(adapter)
            record['profiles'].append(profile)
        for path, expected in freeze['files'].items():
            if sha(path) != expected:
                raise ValueError('Dependency changed during run: ' + path)
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['finished_ns'] = time.time_ns()
        write(args.output, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'geometry', 'parents', 'baselines', 'freeze', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
