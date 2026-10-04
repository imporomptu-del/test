"""Post-score report retaining all V40 proximity alternatives, without new fits.

The frozen runner's reference_evidence is keyed to the original actual-stage
assignment. That can be an unqualified nearer ID even when a different qualified
ID supplies the strict hit. This additive report preserves both stage assignments
and every gated measured alternative rather than selecting a favorable fit.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/tiny_target/accuracy_v41_20260925'
SCOPES = ('grid_0126_late_r1c1', 'grid_0082_early_r2c1', 'grid_0082_middle_r2c1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compact(state):
    lags = []
    for lag in state['lags']:
        zero = next((v for v in lag['variants'] if v['shift'] == [0, 0]), None)
        lags.append(dict(lag=lag['lag'], pair_available=lag['pair_available'],
            pair_reasons=lag['reasons'], zero_offset_preference=lag['zero_offset_preference'],
            all_nine_available=lag['all_nine_available'], offset_sign_agreement=lag['offset_sign_agreement'],
            zero_offset_details=None if zero is None else {k:zero.get(k) for k in
                ('available','reasons','geometry','mse_stationary','mse_transported','mse_plane',
                 'advantage_stationary_minus_transported','common_support_count')}))
    return dict(segment=state['segment'], track_id=state['track_id'],
                qualified_moving=state['qualified_moving'], actual_source_xy=state['actual_source_xy'], lags=lags)


def run(directory, output):
    if output.exists():
        raise FileExistsError('Fresh additive report required')
    receipt_path = directory/'completion_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['completed'] is not True:
        raise ValueError('Completed diagnostic required')
    for path, expected in receipt['files_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Changed input '+path)
    reference_path = ROOT/'results/tiny_target/accuracy_v40_20260925/coverage_workload_01/visible_reference_evidence.json'
    references = json.loads(reference_path.read_text())
    states = {}
    scope_counts = {scope:Counter() for scope in SCOPES}
    strict_zero = {str(lag):Counter() for lag in (1, 2, 4, 8)}
    for line in (directory/'states.jsonl').open():
        state = json.loads(line)
        key = (state['window_id'], state['frame_index'], str(state['segment'])+'/'+state['track_id'])
        if key in states:
            raise ValueError('Duplicate state')
        states[key] = state
        if state['qualified_moving']:
            for lag in state['lags']:
                strict_zero[str(lag['lag'])][lag['zero_offset_preference'] or 'unavailable'] += 1
            if state['window_id'] in SCOPES:
                counts = scope_counts[state['window_id']]
                counts['strict_states'] += 1
                for lag in state['lags']:
                    counts[str(lag['lag'])+':zero:'+(lag['zero_offset_preference'] or 'unavailable')] += 1
                    counts[str(lag['lag'])+':nine:'+(lag['offset_sign_agreement'] or 'unavailable')] += 1
    if len(states) != 1367 or len(references) != 11:
        raise ValueError('Wrong denominator')
    result, unsafe_counterexamples = [], []
    for ref in references:
        stage = ref['stages']
        alternatives = []
        for tid in stage['actual_measurement']['all_gated_ids']:
            state = states[(ref['window_id'], ref['frame_index'], tid)]
            alternatives.append(dict(identity=tid, **compact(state)))
        strict_ids = stage['strict_qualified_measurement']['all_gated_ids']
        if {a['identity'] for a in alternatives if a['qualified_moving']} != set(strict_ids):
            raise ValueError('Strict alternatives differ from original scoring')
        result.append(dict(reference=ref, original_actual_assignment=stage['actual_measurement']['assigned_id'],
            original_strict_assignment=stage['strict_qualified_measurement']['assigned_id'],
            measured_alternatives=alternatives))
        if len(strict_ids) == 1:
            s = next(a for a in alternatives if a['identity'] == strict_ids[0])
            if s['lags'][0]['zero_offset_preference'] == 'stationary':
                unsafe_counterexamples.append(dict(clip=ref['clip_id'], frame=ref['frame_index'],
                    window_id=ref['window_id'], sole_strict_matching_identity=strict_ids[0],
                    lag1_all_nine_offset_sign_agreement=s['lags'][0]['offset_sign_agreement'],
                    interpretation='A literal reject-on-lag1-zero-offset-stationary preference would discard the sole near-reference strict measurement. This is a post-score counterexample, not a proposed or validated filter.'))
    paths = [Path(__file__).resolve(), receipt_path, directory/'states.jsonl', directory/'summary.json', reference_path]
    hashes = {str(path.resolve()):digest(path) for path in paths}
    report = dict(schema='seaqr.accuracy-v41-postscore-report.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), post_score_analysis=True, new_fits_computed=False,
        source_hashes=hashes, all_diagnostic_receipt_hashes_verified=True,
        actual_state_count=len(states), strict_zero_offset_by_lag={k:dict(v) for k,v in strict_zero.items()},
        highlighted_strict_scope_counts={k:dict(v) for k,v in scope_counts.items()},
        reference_count=11, reference_assignment_caveat='Nearest actual-stage and strict-stage assignments can differ; every originally gated measured alternative is retained without selecting the best V41 result.',
        reference_alternatives=result, unsafe_simple_rule_counterexamples=unsafe_counterexamples,
        no_production_changes=True, no_new_target_or_false_positive_labels=True)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Changed during additive report')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(reference_count=11, alternatives=sum(len(r['measured_alternatives']) for r in result),
                         unsafe_simple_rule_counterexamples=unsafe_counterexamples), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=BASE/'transport_01')
    parser.add_argument('--output', type=Path, default=BASE/'postscore_report_01.json')
    args = parser.parse_args()
    run(args.directory, args.output)
