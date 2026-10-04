#!/usr/bin/env python3
"""Compact, independently recomputed regional-motion comparison accounting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('global_translation', 'local_translation', 'local_affine')
COMPARISONS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                p90=float(np.quantile(values, .9)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def cohort_summary(indices, references, predictions):
    indices = list(map(int, indices))
    assert len(indices) == len(set(indices))
    reference_arrays = {name: np.asarray(values, dtype=float).reshape(-1, 2) for name, values in references.items()}
    assert all(len(values) == len(indices) and np.isfinite(values).all() for values in reference_arrays.values())
    errors, available = {}, {}
    for arm in ARMS:
        saved = predictions[arm]
        assert len(saved['available']) == len(saved['predicted_displacement_xy'])
        assert all(bool(ok) == (value is not None) for ok, value in zip(saved['available'], saved['predicted_displacement_xy']))
        available[arm] = np.array([saved['available'][i] for i in indices], dtype=bool)
        values = np.array([saved['predicted_displacement_xy'][i] if saved['available'][i] else [0, 0] for i in indices], dtype=float).reshape(-1, 2)
        assert np.isfinite(values).all()
        # Temporary zeros are never scored or serialized as missing predictions.
        errors[arm] = {name: np.linalg.norm(values-ref, axis=1) for name, ref in reference_arrays.items()}
    arms = {arm: dict(available_count=int(available[arm].sum()), unavailable_count=len(indices)-int(available[arm].sum()),
        conditional_errors={name: stats(values[available[arm]]) for name, values in errors[arm].items()}) for arm in ARMS}
    paired = {}
    for left, right in COMPARISONS:
        common = available[left] & available[right]
        paired[f'{left}__{right}'] = dict(common_count=int(common.sum()), missing_count=len(indices)-int(common.sum()),
            common_query_indices=[index for index, keep in zip(indices, common) if keep],
            references={name: dict(left=stats(errors[left][name][common]), right=stats(errors[right][name][common]),
                paired_error_reduction_left_minus_right=stats((errors[left][name]-errors[right][name])[common]),
                right_lower_error_count=int((errors[right][name][common] < errors[left][name][common]).sum()))
                for name in references})
    common = np.logical_and.reduce([available[arm] for arm in ARMS])
    return dict(total_count=len(indices), arms=arms, pairwise=paired,
        all_three_common=dict(count=int(common.sum()), missing_count=len(indices)-int(common.sum()),
            query_indices=[index for index, keep in zip(indices, common) if keep],
            references={name: {arm: stats(errors[arm][name][common]) for arm in ARMS} for name in references}),
        interpretation='Conditional errors require coverage context; unavailability is not zero error or correctness.')


def summarize_case(row, patch_rows):
    previous = np.asarray(row['points']['accepted_previous_xy'], dtype=float)
    current = np.asarray(row['points']['accepted_current_xy'], dtype=float)
    assert previous.shape == current.shape == (row['accepted_count'], 2)
    patches = [p for p in patch_rows if p['source_kind'] == 'actual_residual_extremum' and p['previous_index'] == row['previous_index']]
    for p in patches:
        index = p['accepted_index']
        assert previous[index].tolist() == p['previous_xy'] and current[index].tolist() == p['current_xy']
    independently_supported = [p for p in patches if p['comparison']['two_sizes_qualified_and_consistent']]
    indices = [p['accepted_index'] for p in patches]
    independent_indices = [p['accepted_index'] for p in independently_supported]
    all_accepted = cohort_summary(range(len(previous)), {'saved_lk': current-previous}, row['predictions'])
    sampled = cohort_summary(indices, {'saved_lk': (current-previous)[indices]}, row['predictions'])
    independent = cohort_summary(independent_indices,
        {f'ncc{size}': [p['scales'][i]['best_offset_xy'] for p in independently_supported] for i, size in enumerate((33, 65))}, row['predictions'])
    local_gates = {}
    for arm in ARMS:
        gates = Counter()
        for cell in row['cells']:
            gates.update(cell['arms'][arm]['gate_reasons'])
        local_gates[arm] = dict(sorted(gates.items()))
    return dict(case_id=row['case_id'], previous_index=row['previous_index'], current_index=row['current_index'],
        selected_count=row['selected_count'], accepted_count=row['accepted_count'], lost_count=row['lost_count'],
        all_accepted_lk=all_accepted, all_patch_samples=sampled, independent_patch_references=independent,
        unresolved_patch_evidence_count=len(patches)-len(independently_supported),
        gate_failure_cell_counts_nonexclusive=local_gates)


def compact_generated(generated):
    result = {key: value for key, value in generated.items()
              if key not in ('target_preservation', 'training_contamination')}
    for group in ('target_preservation', 'training_contamination'):
        cases = []
        for case in generated[group]['cases']:
            compact = {key: value for key, value in case.items() if key != 'cells'}
            compact['cell_fit_status'] = [dict(cell_id=cell['cell_id'], arms={arm: {
                key: cell['arms'][arm].get(key) for key in ('training_eligible', 'gate_reasons', 'train_count',
                    'train_occupied_cells', 'coherent_count', 'coherent_occupied_cells', 'coherent_base_mass_fraction')}
                for arm in ARMS}) for cell in case['cells']]
            cases.append(compact)
        result[group] = dict(cases=cases, paired_checks=generated[group]['paired_checks'])
    result['full_solver_receipts_location'] = 'Original result.json generated cases; compact copy omits full cell fit arrays only.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--patch-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = {str(p): sha(p) for p in (args.result, args.patch_result)}
    assert before[str(args.patch_result)] == '6fd7beb2e83ae8915e2ec9f5d96ae9f7fa956df66a3cee4e5a2c6146f0a82f3c'
    result, patches = json.loads(args.result.read_text()), json.loads(args.patch_result.read_text())
    assert result['passed'] and patches['passed']
    assert result['hashes_before'] == result['hashes_after']
    assert result['hashes_before']['patch_result_sha256'] == before[str(args.patch_result)]
    assert [r['previous_index'] for r in result['rows']] == [0, 42, 85, 127, 170, 212, 255, 298]
    summary = dict(schema='seaqr.aot.regional-motion-summary.v1', result_sha256=before[str(args.result)],
        patch_result_sha256=before[str(args.patch_result)], script_sha256=sha(__file__),
        rows=[summarize_case(row, patches['rows']) for row in result['rows']], generated=compact_generated(result['generated']),
        interpretation='Offline development prediction checks; not target recall, calibrated confidence, or production promotion.')
    assert sum(row['all_patch_samples']['total_count'] for row in summary['rows']) == 758
    assert sum(row['independent_patch_references']['total_count'] for row in summary['rows']) == 172
    assert before == {str(p): sha(p) for p in (args.result, args.patch_result)}
    with args.output.open('x') as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), pairs=len(summary['rows']))))


if __name__ == '__main__':
    main()
