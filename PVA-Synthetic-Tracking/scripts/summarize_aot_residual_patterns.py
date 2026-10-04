#!/usr/bin/env python3
"""Compact fixed residual-analysis results without recomputing fits or statistics."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'outputs/seaqr_aot_pilot_20260927/residual_patterns_01'


def compact_row(row):
    keys = ('ordinal', 'case_id', 'source_kind', 'previous_index', 'current_index',
        'expected_shift_xy', 'saved_candidate_translation', 'residual_available',
        'selected_count', 'accepted_count', 'lost_count', 'acceptance_fraction',
        'displacement_summary', 'accepted_displacement_fraction_above_sqrt20',
        'candidate_residual_summary', 'saved_inlier_residual_summary', 'saved_outlier_residual_summary')
    out = {k: row[k] for k in keys}
    out['saved_fit'] = {k: row['original_fit'][k] for k in ('quality_status', 'rejection_reasons', 'metrics')}
    out['grid_supported_cells'] = sum(c['median_vector'] is not None for c in row['grid']['cells'])
    out['coherence'] = {k: v for k, v in row['coherence'].items() if k not in ('anchors', 'neighbors', 'reference_values')}
    out['score_strata'] = {k: v for k, v in row['score_strata'].items() if k != 'bins'}
    out['score_strata']['bins'] = []
    for b in row['score_strata']['bins']:
        cb = {k: v for k, v in b.items() if k not in ('selected_grid_coverage', 'accepted_grid_coverage')}
        cb['selected_occupied_cells'] = b['selected_grid_coverage']['occupied_cells']
        cb['accepted_occupied_cells'] = b['accepted_grid_coverage']['occupied_cells']
        out['score_strata']['bins'].append(cb)
    return out


def count_totals(rows):
    bins = []
    for index in range(4):
        selected = sum(r['score_strata']['bins'][index]['selected_count'] for r in rows)
        accepted = sum(r['score_strata']['bins'][index]['accepted_count'] for r in rows)
        value = dict(bin=index, selected_count=selected, accepted_count=accepted,
            lost_count=selected-accepted, acceptance_fraction=accepted/selected if selected else None)
        truth = [r['score_strata']['bins'][index]['fixed_support_truth'] for r in rows
                 if 'fixed_support_truth' in r['score_strata']['bins'][index]]
        if truth:
            totals = {k: sum(t[k] for t in truth) for k in ('selected_count', 'accepted_count', 'lost_count',
                      'accepted_within_0_1_count', 'accepted_within_0_5_count')}
            totals['within_0_5_fraction_of_selected'] = totals['accepted_within_0_5_count']/totals['selected_count'] if totals['selected_count'] else None
            totals['acceptance_fraction'] = totals['accepted_count']/totals['selected_count'] if totals['selected_count'] else None
            value['fixed_support_truth'] = totals
        bins.append(value)
    return dict(case_count=len(rows), bins=bins,
                caution='Repeated selected features/pairs, not independent recordings; counts are review denominators, not population estimates.')


def main():
    raw = (ROOT / 'result.json').read_bytes()
    source = json.loads(raw)
    assert source['passed'] and len(source['rows']) == 64
    rows = [compact_row(r) for r in source['rows']]
    summary = dict(schema='seaqr.aot.residual-patterns-summary.v1', source_result_sha256=hashlib.sha256(raw).hexdigest(),
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_integrity_passed=source['passed'], design=source['design'], source_hashes=source['hashes_before'],
        rows=rows, source_invariance=source['source_invariance'], limitations=source['limitations'],
        actual_score_counts=count_totals(rows[:8]),
        nonzero_synthetic_score_counts=count_totals([r for r in rows[8:] if r['expected_shift_xy'] != [0., 0.]]),
        new_fit=False, detector_run=False, media_accessed=False, production_changed=False)
    with (ROOT / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
