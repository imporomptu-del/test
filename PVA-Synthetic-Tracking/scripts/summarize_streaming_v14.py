"""Validate compact v14 evidence and report independent, non-deployment gates."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from benchmark_selective_v14 import LIBRARY_SHA, tile_set, sha
from check_streaming_feasibility_v14 import ROOT
from streaming_schedule_v14 import ScheduleConfig


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(directory):
    pre = json.loads((directory/'preflight_02.json').read_text())
    gpu = json.loads((directory/'gpu_01.json').read_text())
    cfg = ScheduleConfig()
    require(pre['configuration'] == gpu['configuration'] == asdict(cfg), 'Changed configuration')
    require(gpu['library_sha256'] == LIBRARY_SHA and gpu['error'] is None, 'Library/run failure')
    require(not gpu['real_media_read'] and not gpu['pipeline_benchmark'] and not gpu['production_approved'], 'Invalid GPU claim')
    require(not pre['real_media_read'] and not pre['labels_used'] and not pre['production_approved'], 'Invalid preflight claim')
    for path, digest in pre['sources'].items():
        require(sha(ROOT/path) == digest, 'Changed preflight input '+path)
    for path, digest in gpu['source_sha256'].items():
        require(sha(ROOT/'scripts'/path) == digest, 'Changed GPU source '+path)
    schedule = [(polarity, repeat, mode) for polarity in ('bright', 'dark') for repeat in range(2)
                for mode in (('dense', 'roi8', 'roi32') if repeat == 0 else ('roi32', 'roi8', 'dense'))]
    require([(t['polarity'], t['repeat'], t['mode']) for t in gpu['trials']] == schedule, 'Incomplete trial schedule')
    comparisons = 0
    for trial in gpu['trials']:
        require([s['cycle'] for s in trial['samples']] == [0, 1, 2], 'Missing cycles')
        for sample in trial['samples']:
            require(all(math.isfinite(sample[k]) and sample[k] > 0 for k in ('elapsed_ms', 'kernel_ms')), 'Invalid timing')
            require(sample['exact'] is True, 'Numerical parity failed')
            if trial['mode'] == 'dense':
                continue
            require([p['tile'] for p in sample['parts']] == tile_set(int(trial['mode'][3:]), cfg), 'Wrong region set')
            require(sample['elapsed_ms'] == sum(p['elapsed_ms'] for p in sample['parts']), 'Bad timing accounting')
            for part in sample['parts']:
                require(part['core'] == list(cfg.rect(part['tile'])) and
                        part['padded'] == list(cfg.rect(part['tile'], True)), 'Incorrect crop geometry')
                c = part['comparison']
                require(c['exact'] and c['reference_sha256'] == c['observed_sha256'] and
                        set(c['mismatched_pixels']) == {'score', 'velocity', 'support', 'valid'} and
                        all(v == 0 for v in c['mismatched_pixels'].values()) and
                        c['max_abs_score_difference'] == 0, 'ROI core mismatch')
                comparisons += 1
    require(gpu['passed'] and comparisons == 480, 'Incomplete comparison count')
    require(all(c['lost_all_opportunities_vs_full'] == 0 and c['lost_three_even_with_ideal_feedback'] == 0
                for c in pre['full_coverage_scheduler_control']['cases']), 'Full-coverage control failed')
    timings = {}
    for mode in ('dense', 'roi8', 'roi32'):
        samples = [s for t in gpu['trials'] if t['mode'] == mode for s in t['samples']]
        timings[mode] = dict(samples=len(samples), median_host_ms=statistics.median(s['elapsed_ms'] for s in samples),
                             minimum_host_ms=min(s['elapsed_ms'] for s in samples),
                             maximum_host_ms=max(s['elapsed_ms'] for s in samples),
                             median_kernel_ms=statistics.median(s['kernel_ms'] for s in samples))
    return dict(schema='seaqr.streaming-v14-summary.v1', evidence_verified=True,
                evidence_sha256={p: sha(directory/p) for p in ('preflight_02.json', 'gpu_01.json')},
                summarizer_sha256=sha(__file__), compared_roi_windows=comparisons,
                generated_gpu_parity_passed=True, gpu_component_timings=timings,
                source_history_bytes=gpu['source_history_bytes'],
                no_seed_scheduling=pre['no_seed_transients']['cases'],
                workload={cid: {k: v[k] for k in ('frames', 'mean_requested_tiles', 'mean_deferred_tiles',
                                                'mean_core_fraction', 'mean_halo_work_fraction',
                                                'maximum_observed_revisit_gap_frames')}
                          for cid, v in pre['visible'].items()},
                gates=pre['gates'],
                decision='Reject this fixed rotating-ROI schedule for promotion: blind transient omissions and '
                         'unchanged RAW motion cost fail necessary gates. Retain verified ROI component result only.',
                new_end_to_end_pipeline_measured=False, real_airborne_accuracy_validated=False,
                defaults_changed=False, production_approved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
