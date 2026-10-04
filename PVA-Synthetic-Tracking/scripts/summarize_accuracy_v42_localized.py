"""Post-score descriptive accounting; never a learned suppression rule.

All warning-free counts mean no implemented warning fired, not identified motion
or reliable evidence. Reference panels overlap and do not establish class truth.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    return None if not values else dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'max'),
        np.quantile(values, [0, .25, .5, .75, .9, 1]).tolist()))


def scored(record):
    return record['localized'] is not None and record['localized']['available']


def compact(record):
    core = record['localized']
    return dict(clip=record['clip'], frame_index=record['frame_index'], track_id=record['track_id'],
        qualified_moving=record['qualified_moving'], reference_sample_indices=record['reference_samples'],
        geometry_available=record['geometry']['available'],
        forecast_to_current_measurement_px=record['predicted_to_actual_distance_px'],
        localized_available=scored(record),
        ambiguity_reasons=None if core is None else core['ambiguity_reasons'],
        unavailable_reasons=record['geometry']['reasons'] if core is None else core['reasons'],
        stationary_mse=None if core is None else core['mse_stationary'],
        augmented_mse=None if core is None else core['mse_augmented'],
        advantage=None if core is None else core['advantage_stationary_minus_augmented'])


def group(records):
    geometry = [r for r in records if r['geometry']['available']]
    available = [r for r in records if scored(r)]
    warning_free = [r for r in available if not r['localized']['ambiguous']]
    deltas = [r['localized']['advantage_stationary_minus_augmented'] for r in available]
    return dict(states=len(records), geometry_available=len(geometry), scores_available=len(available),
        insufficient_prior_history=len(records)-len(geometry),
        geometry_but_core_unavailable=len(geometry)-len(available),
        scores_with_recorded_ambiguity=len(available)-len(warning_free),
        scores_without_recorded_ambiguity=len(warning_free),
        # Numerical reporting epsilon only, not a detector/class threshold.
        score_sign_at_reporting_epsilon_1e_9=dict(Counter(
            'positive' if d > 1e-9 else 'negative' if d < -1e-9 else 'near_zero' for d in deltas)),
        stationary_mse_dn2=quantiles([r['localized']['mse_stationary'] for r in available]),
        augmented_mse_dn2=quantiles([r['localized']['mse_augmented'] for r in available]),
        advantage_dn2=quantiles(deltas),
        forecast_to_current_measurement_px=quantiles([r['predicted_to_actual_distance_px'] for r in geometry]),
        persistent_anchor_count_before_cap=quantiles([
            r['localized']['components']['persistent_anchor_count_before_cap'] for r in geometry]),
        measured_prior_counts=dict(Counter(str(r['geometry']['geometry']['measured_prior_count']) for r in records)))


def summarize(records, references):
    groups = dict(all=group(records), strict=group([r for r in records if r['qualified_moving']]),
        grid=group([r for r in records if r['grid_windows']]),
        grid_strict=group([r for r in records if r['grid_windows'] and r['qualified_moving']]),
        reference_union=group([r for r in records if r['reference_samples']]))
    panels = {}
    for panel, original_counts in references['panels'].items():
        samples = [s for s in references['samples'] if s['original']['panel'] == panel]
        counts = Counter(samples=len(samples))
        for sample in samples:
            alternatives = sample['measured_alternatives']
            strict = [a for a in alternatives if a['qualified_moving']]
            counts['all_actual_alternatives'] += len(alternatives)
            counts['all_strict_alternatives'] += len(strict)
            for prefix, subset in [('actual', alternatives), ('strict', strict)]:
                available = [a for a in subset if scored(a)]
                clear = [a for a in available if not a['localized']['ambiguous']]
                counts[prefix+'_samples_no_original_alternative'] += int(not subset)
                counts[prefix+'_samples_with_any_available_score'] += int(bool(available))
                counts[prefix+'_samples_with_any_score_without_recorded_ambiguity'] += int(bool(clear))
                counts[prefix+'_samples_with_alternatives_but_no_available_score'] += int(bool(subset) and not available)
                # Keeping every alternative, this is audit coverage, not choosing an ID.
                counts[prefix+'_available_alternatives'] += len(available)
                counts[prefix+'_available_alternatives_without_recorded_ambiguity'] += len(clear)
        panels[panel] = dict(original_counts=original_counts, evidence_coverage=dict(counts))
    available = [r for r in records if scored(r)]
    return dict(schema='seaqr.accuracy-v42-descriptive-analysis.v1', groups=groups, panels=panels,
        known_v41_counterexamples=[compact(r) for r in records if r['clip']=='0029'
            and r['frame_index'] in (346, 347) and r['track_id']=='bright:2641'],
        posthoc_largest_negative_gains=[compact(r) for r in sorted(available,
            key=lambda r:r['localized']['advantage_stationary_minus_augmented'])[:5]],
        posthoc_reference_states_with_negative_gain_and_no_recorded_ambiguity=[compact(r)
            for r in available if r['reference_samples'] and not r['localized']['ambiguous']
            and r['localized']['advantage_stationary_minus_augmented'] < -1e-9],
        caveats=[
            'No suppression, classifier or production change; warning-free is not reliable or airborne.',
            'Panels overlap. Sample/alternative counts are not independent encounter counts.',
            'Any-score counts describe coverage only; no favorable alternative replaces original assignment.',
            'Prediction error is against a saved tracker measurement, not independent physical truth.',
            'Current imagery estimates global geometry/annulus photometry but never places the forecast.',
            'Worst negative gains are post-score diagnostic selections, not prespecified performance tests.',
            'Missing verification is unknown; existing misses remain misses.'])


def main(run, output):
    if output.exists():
        raise FileExistsError('Use a fresh analysis path')
    receipt = json.loads((run/'completion_receipt.json').read_text())
    inputs = {}
    for name in ('states.jsonl', 'reference_evidence.json'):
        path = run/name
        actual = sha(path)
        if receipt['files_sha256'][str(path.resolve())] != actual:
            raise ValueError('Completed evidence changed')
        inputs[str(path.resolve())] = actual
    records = [json.loads(line) for line in (run/'states.jsonl').read_text().splitlines()]
    result = summarize(records, json.loads((run/'reference_evidence.json').read_text()))
    inputs[str((run/'completion_receipt.json').resolve())] = sha(run/'completion_receipt.json')
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    result['inputs_sha256'] = inputs
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['groups'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
