"""Frozen coordinate-only controls; no image, file, model selection, or detector IO."""
from __future__ import annotations

import numpy as np


def lattice():
    return np.array([(x, y) for y in range(24, 2048, 48) for x in range(24, 2448, 48)], dtype=float)


def background(points, kind):
    points = np.asarray(points, dtype=float)
    if kind == 'translation':
        return np.tile([3., -2.], (len(points), 1))
    if kind != 'affine':
        raise ValueError('unplanned analytic background')
    x, y = points[:, 0]-1224, points[:, 1]-1024
    return np.column_stack((3+.008*x+.004*y, -2-.003*x+.006*y))


def cluster(center, size):
    center = np.asarray(center, dtype=float)
    if size == 1:
        return center[None, :].copy()
    if size != 9:
        raise ValueError('unplanned compact cluster size')
    return np.array([center+[x, y] for y in (-1, 0, 1) for x in (-1, 0, 1)])


def evaluate_queries(helper, evaluate_cell, arms, previous, current, query_indices, kind,
                     target_offset, check_clean_recovery=False):
    query_indices = list(map(int, query_indices))
    assignments = helper.native_cells(previous[query_indices])
    cells = [evaluate_cell(helper, previous, current, int(cell)) for cell in sorted(set(assignments.tolist()))]
    cell_lookup = {cell['cell_id']: cell for cell in cells}
    queries = previous[query_indices]
    truth = background(queries, kind)
    results = {arm: [] for arm in arms}
    for j, (index, cell_id) in enumerate(zip(query_indices, assignments)):
        cell = cell_lookup[int(cell_id)]
        position = cell['test_indices'].index(index)
        for arm in arms:
            fitted = cell['arms'][arm]
            # All compact query points must be excluded for every query, even
            # when the cluster straddles a native cell boundary.
            if set(fitted['train_indices']) & set(query_indices):
                raise ValueError('target cluster leaked into guarded training')
            saved = fitted['queries']
            available = saved['available'][position]
            value = saved['predicted_displacement_xy'][position]
            entry = dict(query_index=index, cell_id=int(cell_id), available=available,
                predicted_background_xy=value, unavailable_reason=saved['unavailable_reason'][position],
                compensated_target_xy=None, background_error_xy=None,
                target_error_from_injected_xy=None, background_error_px=None)
            if available:
                value = np.array(value)
                residual = current[index]-previous[index]-value
                background_error = truth[j]-value
                if np.linalg.norm(residual-(np.asarray(target_offset)+background_error)) > 1e-8:
                    raise ValueError('compensated vector accounting failed')
                error = float(np.linalg.norm(background_error))
                correctly_specified = arm == 'local_affine' or (arm == 'local_translation' and kind == 'translation')
                if check_clean_recovery and correctly_specified and error > 1e-8:
                    raise ValueError('clean correctly specified local model failed analytic recovery')
                entry.update(compensated_target_xy=residual.tolist(), background_error_xy=background_error.tolist(),
                    target_error_from_injected_xy=(residual-target_offset).tolist(), background_error_px=error)
            results[arm].append(entry)
    return dict(query_indices=query_indices, query_xy=queries.tolist(),
        analytic_background_displacement_xy=truth.tolist(), target_offset_xy=list(target_offset),
        query_results=results, cells=cells,
        interpretation='Sparse-vector background subtraction only; unavailable queries are not preservation passes.')


def case_counts(case, arms):
    return {arm: dict(total=len(case['query_results'][arm]),
        available=sum(row['available'] for row in case['query_results'][arm]),
        unavailable=sum(not row['available'] for row in case['query_results'][arm]),
        max_background_error_px=max((row['background_error_px'] for row in case['query_results'][arm]
                                      if row['available']), default=None)) for arm in arms}


def target_checks(baseline, moving, arms):
    # Entire solver/support/weight receipts must match, not only the winning
    # model or a rounded prediction. The query endpoints are never fit inputs.
    if baseline['cells'] != moving['cells']:
        raise ValueError('excluded target motion changed a fit, support gate, or query prediction')
    delta = np.asarray(moving['target_offset_xy'], dtype=float)
    result = {}
    for arm in arms:
        differences, unavailable = [], 0
        for left, right in zip(baseline['query_results'][arm], moving['query_results'][arm]):
            if left['available'] != right['available']:
                raise ValueError('target perturbation changed eligibility')
            if not left['available']:
                unavailable += 1
                continue
            error = float(np.linalg.norm(np.array(right['compensated_target_xy'])-
                                         left['compensated_target_xy']-delta))
            if error > 1e-8:
                raise ValueError('independent target vector was changed by self-leakage')
            differences.append(error)
        result[arm] = dict(total=len(baseline['query_results'][arm]), supported_preservation_checks=len(differences),
            unavailable_not_passed=unavailable, max_vector_difference_error_px=max(differences, default=None))
    return dict(baseline_case_id=baseline['id'], moving_case_id=moving['id'],
        identical_fits_support_weights_predictions=True, arms=result)


def contamination_checks(clean, contaminated, arms):
    result = {}
    target = np.asarray(contaminated['target_offset_xy'], dtype=float)
    for arm in arms:
        rows = []
        for left, right in zip(clean['query_results'][arm], contaminated['query_results'][arm]):
            entry = dict(query_index=left['query_index'], clean_available=left['available'],
                contaminated_available=right['available'], background_prediction_drift_xy=None,
                background_prediction_drift_px=None, compensated_target_change_xy=None,
                clean_target_residual_xy=left['compensated_target_xy'],
                contaminated_target_residual_xy=right['compensated_target_xy'],
                contaminated_target_error_px=None, contaminated_target_direction_fraction=None)
            if left['available'] and right['available']:
                drift = np.array(right['predicted_background_xy'])-left['predicted_background_xy']
                change = np.array(right['compensated_target_xy'])-left['compensated_target_xy']
                if np.linalg.norm(drift+change) > 1e-8:
                    raise ValueError('contamination residual accounting failed')
                residual = np.array(right['compensated_target_xy'])
                entry.update(background_prediction_drift_xy=drift.tolist(),
                    background_prediction_drift_px=float(np.linalg.norm(drift)),
                    compensated_target_change_xy=change.tolist(),
                    contaminated_target_error_px=float(np.linalg.norm(residual-target)),
                    contaminated_target_direction_fraction=float(residual@target/(target@target)))
            rows.append(entry)
        result[arm] = rows
    return dict(clean_case_id=clean['id'], contaminated_case_id=contaminated['id'], arms=result,
        interpretation='Fixed neighboring-training stress, not guaranteed exclusion; no outcome-derived acceptance threshold.')


def run_generated(helper, evaluate_cell, arms):
    if tuple(arms) != ('global_translation', 'local_translation', 'local_affine'):
        raise ValueError('unexpected model inventory')
    base = lattice()
    cy = 2048*3.5/6
    placements = [('center_single', [1071, cy], 1), ('center_3x3', [1071, cy], 9),
                  ('right_boundary_3x3', [1224, cy], 9)]
    target_cases, target_paired = [], []
    for kind in ('translation', 'affine'):
        for placement, center, size in placements:
            queries = cluster(center, size)
            previous = np.vstack((base, queries))
            indices = np.arange(len(base), len(previous))
            baseline = None
            for offset in ([0, 0], [1, 0], [4, -2]):
                current = previous+background(previous, kind)
                current[indices] += offset
                case = dict(id=f'{kind}__{placement}__target_{offset[0]:+d}_{offset[1]:+d}',
                    background=kind, placement=placement, target_size_points=size,
                    **evaluate_queries(helper, evaluate_cell, arms, previous, current, indices, kind, offset, True))
                case['counts'] = case_counts(case, arms)
                target_cases.append(case)
                if baseline is None:
                    baseline = case
                else:
                    target_paired.append(target_checks(baseline, case, arms))
    nuisance_cases, nuisance_paired = [], []
    for kind in ('translation', 'affine'):
        for size in (1, 9):
            nuisance = cluster([1377, cy], size)
            previous = np.vstack((base, nuisance, [[1071, cy]]))
            nuisance_indices = np.arange(len(base), len(base)+size)
            query_indices = [len(previous)-1]
            clean = None
            for offset in ([0, 0], [20, -10]):
                current = previous+background(previous, kind)
                current[nuisance_indices] += offset
                current[query_indices] += [4, -2]
                case = dict(id=f'{kind}__nuisance_{size}__offset_{offset[0]:+d}_{offset[1]:+d}',
                    background=kind, nuisance_size_points=size, nuisance_xy=nuisance.tolist(),
                    nuisance_indices=nuisance_indices.tolist(), nuisance_offset_xy=offset,
                    **evaluate_queries(helper, evaluate_cell, arms, previous, current, query_indices,
                                       kind, [4, -2], offset == [0, 0]))
                for cell in case['cells']:
                    for arm in arms:
                        if not set(nuisance_indices.tolist()) <= set(cell['arms'][arm]['train_indices']):
                            raise ValueError('nuisance stress cluster did not enter intended training set')
                case['counts'] = case_counts(case, arms)
                nuisance_cases.append(case)
                if clean is None:
                    clean = case
                else:
                    nuisance_paired.append(contamination_checks(clean, case, arms))
    if (len(target_cases), len(target_paired), len(nuisance_cases), len(nuisance_paired)) != (18, 12, 8, 4):
        raise ValueError('incomplete generated case inventory')
    return dict(schema='seaqr.aot.regional-generated.v1', passed=True, source_images_used=False,
        analytic_recovery_tolerance_px=1e-8,
        target_preservation=dict(cases=target_cases, paired_checks=target_paired),
        training_contamination=dict(cases=nuisance_cases, paired_checks=nuisance_paired),
        interpretation='Generated coordinate-level invariance and contamination tests, not photometric target preservation or detection recall.')
