"""Saved-development-journal preflight. No source-media or sealed-data access."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from streaming_schedule_v14 import ScheduleConfig, Scheduler, queue_model

EVIDENCE = ROOT/'results/tiny_target/motion_video_v13_20260916'
IDS = ('0029', '0126', '0055', '0082')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def transient_opportunities(config):
    """Enumerate a complete cyclic no-seed schedule, not image sensitivity."""
    # RR with capacity c has period tiles/gcd(tiles,c) window advances. Run a
    # longer integer multiple so every relative onset/tile phase is represented.
    period = config.tiles*config.stride
    first = config.warmup+config.window+period
    last = first+period-1
    scheduler = Scheduler(config)
    windows = []
    for i in range(last+64+config.window):
        row = scheduler.update(i, i*100000000, 0, [])
        if row:
            windows.append(row)
    cases = []
    for duration in (16, 32, 64):
        counts = np.zeros((config.tiles, period+1), np.int32)
        earliest = np.full((config.tiles, period), np.iinfo(np.int32).max, np.int32)
        full = np.zeros(period+1, np.int32)
        for row in windows:
            end = row['frame']
            # >=12 active observations in [end-15,end] intersect [onset,onset+L-1].
            lo = max(first, end-config.window+12-duration+1)
            hi = min(last, end-12+1)
            if hi < lo:
                continue
            a, b = lo-first, hi-first+1
            counts[row['tiles'], a] += 1
            counts[row['tiles'], b] -= 1
            earliest[row['tiles'], a:b] = np.minimum(earliest[row['tiles'], a:b], end)
            full[a] += 1
            full[b] -= 1
        counts = np.cumsum(counts[:, :-1], axis=1)
        full = np.broadcast_to(np.cumsum(full[:-1]), counts.shape)
        eligible = full >= 1
        three = full >= 3
        lost = eligible & (counts == 0)
        # Optimistically assume the first searched window detects the target,
        # then its ROI is pinned for the next two advances at NO extra cost.
        # This is not simulated evidence of detection or a resource-safe policy.
        ideal_three = earliest.astype(np.int64)+2*config.stride <= np.arange(first, last+1)+duration+config.window-12-1
        example = np.argwhere(lost)
        cases.append(dict(visibility_frames=duration, cases=int(counts.size),
                          full_search_at_least_one=int(eligible.sum()),
                          selective_at_least_one=int((counts >= 1).sum()),
                          full_search_at_least_three=int(three.sum()),
                          selective_three_without_feedback=int((counts >= 3).sum()),
                          selective_three_with_ideal_first_hit_feedback=int(ideal_three.sum()),
                          lost_all_opportunities_vs_full=int(lost.sum()),
                          lost_three_even_with_ideal_feedback=int((three & ~ideal_three).sum()),
                          example_omission=None if not len(example) else dict(
                              tile=int(example[0, 0]), onset_frame=first+int(example[0, 1]),
                              full_windows=int(full[tuple(example[0])]),
                              selective_windows=int(counts[tuple(example[0])]))))
    return dict(no_seed=True, period_frames=period, labels_used=False,
                interpretation='Scheduling opportunities only, not image detections or recall. '
                'Full 16-frame history is optimistically available for every selected ROI.',
                cases=cases)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    cfg = ScheduleConfig()
    verified_path = EVIDENCE/'summary_verified_local.json'
    verified = read(verified_path)
    if verified.get('verified') is not True:
        raise ValueError('Missing verified v13 evidence')
    provenance = {str(verified_path.relative_to(ROOT)): sha(verified_path)}
    result = dict(schema='seaqr.streaming-preflight-v14.v1', configuration=asdict(cfg),
                  stage='architectural preflight, NOT a new end-to-end pipeline',
                  real_media_read=False, labels_used=False, production_approved=False,
                  defaults_changed=False, real_airborne_accuracy_validated=False,
                  sources=provenance, visible={}, raw_motion_only={})
    data = EVIDENCE/'evidence/results'
    for cid in IDS:
        name = 'visible_'+cid+'_full_reuse'
        directory = data/name
        execution_path = data/(name+'.execution.json')
        if sha(execution_path) != verified['trials'][name]['execution_sha256']:
            raise ValueError('Archived execution changed')
        execution, report = read(execution_path), read(directory/'report.json')
        if not execution['passed'] or not execution['closed'] or not report['completed']:
            raise ValueError('Incomplete reference')
        scheduler = Scheduler(cfg)
        windows, timings = [], []
        first_time, last_time = None, None
        with (directory/'frames.jsonl').open() as handle:
            for count, line in enumerate(handle):
                row = json.loads(line)
                if row['frame_index'] != count or row['coverage']['full_shape_hw'] != [cfg.height, cfg.width]:
                    raise ValueError('Wrong extent/order')
                decision = scheduler.update(count, row['timestamp_ns'], row['segment'], row['candidates'])
                if decision:
                    windows.append(decision)
                timings.append(row['timings_ms'])
                first_time = row['timestamp_ns'] if first_time is None else first_time
                last_time = row['timestamp_ns']
        if len(timings) != report['frames'] or len(timings) != execution['processed_frames']:
            raise ValueError('Incomplete journal')
        for path in (execution_path, directory/'report.json', directory/'launch.json', directory/'frames.jsonl'):
            provenance[str(path.relative_to(ROOT))] = sha(path)
        motion = [t['motion_and_warp'] for t in timings]
        serial = [sum(t[k] for k in ('motion_and_warp', 'detection', 'tracking', 'decode_wait')) for t in timings]
        gaps = [g for w in windows for g in w['revisit_gaps_frames']]
        result['visible'][cid] = dict(
            frames=len(timings), archived_whole_pipeline_fps=report['processed_fps'],
            playback_timestamp_fps=(len(timings)-1)*1e9/(last_time-first_time),
            physical_sensor_fps_verified=False,
            motion_only_optimistic_lower_bound=queue_model(motion),
            archived_serial_stages_model=queue_model(serial),
            queue_exclusions=['journal/output overhead', 'startup/teardown', 'new selective temporal search'],
            decode_note='Decode/gray overlap is not added; observed decode_wait is included.',
            mean_requested_tiles=float(np.mean([w['requested'] for w in windows])),
            mean_deferred_tiles=float(np.mean([w['requested_deferred'] for w in windows])),
            windows_with_deferred_requests=sum(w['requested_deferred'] > 0 for w in windows),
            mean_core_fraction=float(np.mean([w['core_pixels']/(cfg.height*cfg.width) for w in windows])),
            mean_halo_work_fraction=float(np.mean([w['halo_pixels_including_overlap']/(cfg.height*cfg.width) for w in windows])),
            maximum_observed_revisit_gap_frames=max(gaps, default=None),
            windows=windows,
            accuracy_claim='Unchanged saved proposals are a workload proxy; not a new detector trial.')
    for cid in ('0029', '0040'):
        for repeat in (0, 1):
            name = f'raw_{cid}_repeat{repeat}_reuse'
            path = data/(name+'.execution.json')
            if sha(path) != verified['trials'][name]['execution_sha256']:
                raise ValueError('RAW timing evidence changed')
            execution = read(path)
            if (execution['processed_frames'] != 64 or not execution['passed'] or not execution['closed']
                    or [r['frame'] for r in execution['motion']] != list(range(1, 64))):
                raise ValueError('Invalid RAW prefix')
            provenance[str(path.relative_to(ROOT))] = sha(path)
            result['raw_motion_only'][name] = queue_model(
                [0.]+[1000*r['estimator_s'] for r in execution['motion']])
    result['no_seed_transients'] = transient_opportunities(cfg)
    result['full_coverage_scheduler_control'] = transient_opportunities(replace(cfg, cap=cfg.tiles, blind=cfg.tiles))
    result['gates'] = dict(
        no_added_scheduling_omissions=all(c['lost_all_opportunities_vs_full'] == 0 and
                                        c['lost_three_even_with_ideal_feedback'] == 0
                                        for c in result['no_seed_transients']['cases']),
        current_raw_motion_within_total_100ms_budget=all(not v['mean_exceeds_budget']
                                                       for v in result['raw_motion_only'].values()),
        new_full_pipeline_measured=False, pixel_level_quality_measured=False,
        deploy=False)
    for path in (Path(__file__), ROOT/'scripts/streaming_schedule_v14.py', ROOT/'docs/streaming_feasibility_v14_plan.md'):
        provenance[str(path.relative_to(ROOT))] = sha(path)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), gates=result['gates'],
                          no_seed_transients=result['no_seed_transients']['cases']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
