"""Independent saved-patch and least-squares audit of frozen v36 context_01.

Does not import the feature/runner implementation and does not read video files.
This checks saved patches and their journal selection, not video-to-patch decoding.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / 'results/tiny_target/accuracy_v36_20260924/context_01'
SHADOW = ROOT / 'results/tiny_target/accuracy_v36_20260924/shadow_01'
SHADOW_AUDIT = ROOT / 'results/tiny_target/accuracy_v36_20260924/independent_audit_01.json'
CONTEXT_FREEZE = 'cd1ca1c38653a8758810aa2f4048e2071cb50b2ed996a472b31661b88a9e93a8'
SHADOW_FREEZE = '9f1439c9eb9c63c820dfaea571cc7d3b8894cfb28ef6463f93c198602c6c6b70'
SHADOW_AUDIT_SHA = '80359935892e7ab201fe91cafc6e8abb36ac23e65db6bfe07d2c30aff5b48ceb'
COUNTS = {'0029': 687, '0126': 674}
SOURCE_HASHES = {
    '0029': '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359',
    '0126': 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344',
}
IMPLEMENTATION = ('scripts/accuracy_v36_context.py', 'scripts/diagnose_accuracy_v36_context.py',
    'tests/unit/test_accuracy_v36_context.py', 'tests/unit/test_accuracy_v36_context_runner.py',
    'docs/accuracy_v36_context_plan.md')
FEATURE_NAMES = ('point_minus_edge_fraction', 'point_gain_fraction', 'edge_gain_fraction',
    'point_gain_after_edge_fraction', 'point_amplitude_dn', 'point_after_edge_amplitude_dn', 'residual_rms_dn')
LIMITATIONS = [
    'Only selected development observations, no full-output gate replay',
    'Known points have unknown physical class; nuisance controls are provisional and selected',
    'Overlapping references and adjacent frames are not independent samples',
    'Unequal template searches and correlated pixels: no calibrated confidence',
    'Native source pixels differ from warped temporally filtered detector domain',
    'Single-frame evidence alone does not prove independent object motion',
]
# Numerical validation tolerances, not object-acceptance thresholds. Direct LS
# and the recorded orthogonal-projection formulation have different roundoff.
RTOL = 2e-10
ATOL = 2e-8


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular non-symlink file required: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(token):
        raise ValueError('Invalid JSON constant ' + token)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def exact(actual, expected, label):
    options = dict(sort_keys=True, allow_nan=False, separators=(',', ':'))
    require(json.dumps(actual, **options) == json.dumps(expected, **options), 'Mismatch: ' + label)


def close_tree(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'Fields differ: ' + label)
        for key in expected:
            close_tree(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List differs: ' + label)
        for index, (a, e) in enumerate(zip(actual, expected)):
            close_tree(a, e, label + '/' + str(index))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                'Independent LS numeric mismatch: ' + label + ': ' + str((actual, expected)))
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value differs: ' + label)


class Integrity:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.resolve().is_relative_to(ROOT.resolve()), 'Audit may not open source media/out-of-scope files')
        require(path.suffix in ('.json', '.jsonl', '.py', '.md', '.log', '.npz'), 'Disallowed audit input type')
        current = digest(path)
        require(expected is None or current == expected, 'Hash mismatch: ' + str(path))
        require(self.files.setdefault(str(path), current) == current, 'Changed input: ' + str(path))
        return current

    def recheck(self):
        for path, expected in self.files.items():
            require(digest(path) == expected, 'Input changed during audit: ' + path)


def verify_inputs(integrity):
    integrity.bind(CONTEXT / 'freeze.json', CONTEXT_FREEZE)
    integrity.bind(SHADOW / 'freeze.json', SHADOW_FREEZE)
    integrity.bind(SHADOW_AUDIT, SHADOW_AUDIT_SHA)
    frozen, shadow, shadow_audit = read(CONTEXT / 'freeze.json'), read(SHADOW / 'freeze.json'), read(SHADOW_AUDIT)
    require(shadow_audit['verified'] is True and shadow_audit['schema'] == 'seaqr.accuracy-v36-independent-audit.v1'
            and shadow_audit['experiment_freeze_sha256'] == SHADOW_FREEZE, 'Independent shadow audit missing')
    expected_inputs = {**shadow_audit['checked_files_sha256'], str(SHADOW_AUDIT): SHADOW_AUDIT_SHA}
    exact(frozen['inputs_sha256'], expected_inputs, 'complete context input inventory')
    for path, expected in expected_inputs.items():
        integrity.bind(path, expected)
    require(frozen['pre_extraction'] is True and frozen['raw16_accessed'] is False
            and frozen['classifier_promoted'] is False and frozen['native_pixels'] is True
            and frozen['patch_size'] == 25 and frozen['rounding'] == 'floor(source_coordinate+0.5)'
            and frozen['zero_margin_ablation'] == 'informative and point_minus_edge_fraction > 0', 'Context scope changed')
    require(set(frozen['source_videos']) == set(COUNTS), 'Source clip scope changed')
    for clip in COUNTS:
        metadata = frozen['source_videos'][clip]
        expected_path = ROOT.parent / 'outputs/jetson_review_clips_20260913' / ('chunk_' + clip + '.avi')
        require(metadata['path'] == str(expected_path) and metadata['sha256'] == SOURCE_HASHES[clip],
                'Source-video metadata changed')
    require(set(frozen['implementation_sha256']) == set(IMPLEMENTATION), 'Implementation inventory changed')
    for relative in IMPLEMENTATION:
        integrity.bind(ROOT / relative, frozen['implementation_sha256'][relative])
        integrity.bind(CONTEXT / 'implementation' / relative, frozen['implementation_sha256'][relative])
    integrity.bind(CONTEXT / 'selection.json', frozen['selection_sha256'])
    integrity.bind(CONTEXT / 'unit.log', frozen['unit_log_sha256'])
    require(re.search(r'Ran 39 tests in [0-9.]+s\s+OK\s*$', (CONTEXT / 'unit.log').read_text()) is not None,
            'Frozen 39-test successful log required')
    integrity.bind(CONTEXT / 'summary.json')
    summary = read(CONTEXT / 'summary.json')
    require(summary['completed'] is True and summary['freeze_sha256'] == CONTEXT_FREEZE,
            'Completed context summary required')
    require(set(summary['outputs_sha256']) == {'native_patches.npz', 'observations.json'}, 'Output inventory changed')
    for name, expected in summary['outputs_sha256'].items():
        integrity.bind(CONTEXT / name, expected)
    return frozen, shadow, summary


def rebuild_selection(shadow):
    controls = read(shadow['references']['controls'])['controls']
    groups, observations = [], []
    require(len(controls) == 7, 'Control denominator changed')
    for clip in COUNTS:
        baseline = read(SHADOW / (clip + '_results.json'))['arms']['baseline']
        for kind in ('dense', 'pilot'):
            for window in baseline['references'][kind]['positive_windows']:
                for sample in window['evidence']:
                    frame = sample['frame_index']
                    groups.append(dict(kind=kind, clip=clip, window=window['window_id'], frame=frame,
                        keys=[f'{clip}/{frame}/{identity}' for identity in sample['all_gated_same_polarity_ids']],
                        baseline_assigned_id=sample['assigned_track_id']))
        for anchor in baseline['required_anchor_evidence']:
            groups.append(dict(kind='anchor', clip=clip, window=anchor['event_id'], frame=anchor['frame'],
                keys=[f'{clip}/{anchor["frame"]}/{identity}' for identity in anchor['ids']], baseline_assigned_id=None))
        wanted = {key for group in groups if group['clip'] == clip for key in group['keys']}
        covered = set()
        with (Path(shadow['inputs'][clip]['path']) / 'frames.jsonl').open() as stream:
            frames = 0
            for line in stream:
                row = json.loads(line)
                require(row['frame_index'] == frames and row['timestamp_ns'] == frames * 100000000,
                        'Unexpected original journal sequence')
                frames += 1
                for track in row['tracks']:
                    if not (track['measured'] and track['qualified_moving']):
                        continue
                    identity = str(track['segment']) + '/' + track['track_id']
                    key = f'{clip}/{row["frame_index"]}/{identity}'
                    xy = track['measurement_source_xy']
                    control_ids = []
                    if clip == '0126':
                        for index, control in enumerate(controls):
                            first, last = control['frames_inclusive']
                            x, y, width, height = control['crop_xywh']
                            if first <= row['frame_index'] <= last and x <= xy[0] < x + width and y <= xy[1] < y + height:
                                control_ids.append(index)
                    if key in wanted or control_ids:
                        require(key not in covered, 'Duplicate selected observation')
                        covered.add(key)
                        observations.append(dict(key=key, clip=clip, frame=row['frame_index'], identity=identity,
                            measurement_source_xy=xy, polarity=track['track_id'].split(':')[0],
                            provisional_control_indices=control_ids))
            require(frames == COUNTS[clip] and wanted <= covered, 'Dropped selected reference observation')
    for kind, count, hits in (('dense', 285, 284), ('pilot', 28, 28), ('anchor', 24, 24)):
        selected = [group for group in groups if group['kind'] == kind]
        require(len(selected) == count and sum(bool(g['keys']) for g in selected) == hits, 'Reference denominator changed')
    require(len(observations) == len({o['key'] for o in observations}) == 358, 'Expected 358 distinct selected observations')
    require(sum(len(o['provisional_control_indices']) for o in observations) == 70, 'Expected 70 control memberships')
    selection = dict(observations=observations, groups=groups, controls=controls)
    exact(read(CONTEXT / 'selection.json'), selection, 'independent complete selection')
    return selection


class DirectLeastSquares:
    """Independent full design-matrix fits; no QR/template projection reuse."""
    def __init__(self):
        y, x = np.indices((25, 25), dtype=np.float64)
        x, y = x - 12, y - 12
        u, v = x.ravel() / 12, y.ravel() / 12
        self.background = np.column_stack([np.ones(625), u, v, u * u, u * v, v * v])
        self.points, self.point_params = [], []
        self.edges, self.edge_params = [], []
        for sigma in (1.0, 2.0, 3.0):
            for dx in (-1.0, 0.0, 1.0):
                for dy in (-1.0, 0.0, 1.0):
                    self.points.append(np.exp(-((x - dx)**2 + (y - dy)**2) / (2 * sigma**2)).ravel())
                    self.point_params.append((sigma, dx, dy))
        for width in (1.0, 2.0, 4.0):
            for k in range(8):
                theta = k * math.pi / 8
                for offset in (-2.0, 0.0, 2.0):
                    self.edges.append(np.tanh((x * math.cos(theta) + y * math.sin(theta) - offset) / width).ravel())
                    self.edge_params.append((width, theta, offset))
        self.point_models = [np.column_stack((self.background, t)) for t in self.points]
        self.edge_models = [np.column_stack((self.background, t)) for t in self.edges]

    @staticmethod
    def fit(design, values):
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = values - design @ coefficients
        return float(residual @ residual), float(coefficients[-1])

    def best(self, models, values, base_energy, sign=None):
        gains, amplitudes, energies = [], [], []
        for model in models:
            energy, amplitude = self.fit(model, values)
            if sign is not None and sign * amplitude < 0:
                energy, amplitude = base_energy, 0.0
            gains.append(max(0.0, min(base_energy, base_energy - energy)))
            amplitudes.append(amplitude)
            energies.append(energy)
        index = int(np.argmax(gains))
        return index, gains[index], amplitudes[index], energies[index]

    def measure(self, patch, polarity):
        require(polarity in ('bright', 'dark'), 'Invalid observation polarity')
        values = np.asarray(patch, dtype=np.float64)
        require(values.shape == (25, 25) and np.isfinite(values).all(), 'Finite 25x25 patch required')
        values = values.ravel() - values.mean()
        base, _ = self.fit(self.background, values)
        guard = 1e-24 * max(1.0, float(values @ values))
        informative = base > guard
        sign = 1 if polarity == 'bright' else -1
        pi, point_gain, pa, _ = self.best(self.point_models, values, base, sign)
        ei, edge_gain, ea, edge_energy = self.best(self.edge_models, values, base)
        nested = [np.column_stack((self.edge_models[ei], template)) for template in self.points]
        ci, conditional_gain, ca, _ = self.best(nested, values, edge_energy, sign)
        conditional_informative = edge_energy > guard
        pg = point_gain / base if informative else 0.0
        eg = edge_gain / base if informative else 0.0
        cg = conditional_gain / edge_energy if conditional_informative else 0.0
        ps, px, py = self.point_params[pi]
        ew, et, eo = self.edge_params[ei]
        cs, cx, cy = self.point_params[ci]
        return dict(informative=bool(informative), background_residual_energy=base,
            point_gain_fraction=pg, edge_gain_fraction=eg, point_minus_edge_fraction=pg - eg,
            point_amplitude_dn=sign * pa if informative else 0.0,
            point_sigma_px=ps, point_offset_xy=[px, py], edge_width_px=ew,
            edge_orientation_rad=et, edge_offset_px=eo, edge_amplitude_dn=ea if informative else 0.0,
            residual_rms_dn=math.sqrt(base / 625), point_absolute_gain=pg * base,
            edge_absolute_gain=eg * base, edge_residual_energy=edge_energy,
            conditional_informative=bool(conditional_informative), point_gain_after_edge_fraction=cg,
            point_after_edge_absolute_gain=cg * edge_energy,
            point_after_edge_amplitude_dn=sign * ca if conditional_informative else 0.0,
            point_after_edge_sigma_px=cs, point_after_edge_offset_xy=[cx, cy])


def verify_patches(selection):
    records = read(CONTEXT / 'observations.json')
    require(len(records) == 358 and len({r['key'] for r in records}) == 358, 'Observation record count/identity mismatch')
    expected = {o['key']: o for o in selection['observations']}
    require(set(expected) == {r['key'] for r in records}, 'Patch observation keys differ from selection')
    ordered_keys = [o['key'] for o in selection['observations']]
    require([r['key'] for r in records] == ordered_keys, 'Unexpected extraction observation order')
    diagnostic = DirectLeastSquares()
    audited = []
    maximum_errors = defaultdict(float)
    with np.load(CONTEXT / 'native_patches.npz', allow_pickle=False) as archive:
        names = [f'patch_{i:04d}' for i in range(358)]
        require(len(archive.files) == len(set(archive.files)) == 358 and set(archive.files) == set(names),
                'Patch archive key inventory mismatch')
        for index, record in enumerate(records):
            selected = expected[record['key']]
            require(set(record) == set(selected) | {
                'source_sha256', 'integer_center_xy', 'fractional_center_offset_xy', 'crop_xywh',
                'patch_array_key', 'patch_sha256', 'patch_min_dn', 'patch_max_dn', 'saturated_low_pixels',
                'saturated_high_pixels', 'features', 'zero_margin_ablation_passed'}, 'Unexpected observation fields')
            exact({k: record[k] for k in selected}, selected, 'observation source/journal metadata')
            name = names[index]
            require(record['patch_array_key'] == name and record['source_sha256'] == SOURCE_HASHES[record['clip']],
                    'Patch array/source identity mismatch')
            patch = archive[name]
            require(patch.dtype == np.dtype('uint8') and patch.shape == (25, 25), 'Exact uint8 25x25 patch required')
            require(hashlib.sha256(patch.tobytes(order='C')).hexdigest() == record['patch_sha256'], 'Patch byte SHA mismatch')
            xy = record['measurement_source_xy']
            center = [math.floor(x + 0.5) for x in xy]
            crop = [center[0] - 12, center[1] - 12, 25, 25]
            require(0 <= crop[0] <= 4784 - 25 and 0 <= crop[1] <= 3190 - 25, 'Truncated patch bounds')
            exact(record['integer_center_xy'], center, 'native center rounding')
            exact(record['crop_xywh'], crop, 'source patch origin')
            exact(record['fractional_center_offset_xy'], [xy[i] - center[i] for i in (0, 1)], 'fractional center')
            require(record['patch_min_dn'] == int(patch.min()) and record['patch_max_dn'] == int(patch.max())
                    and record['saturated_low_pixels'] == int(np.count_nonzero(patch == 0))
                    and record['saturated_high_pixels'] == int(np.count_nonzero(patch == 255)), 'Pixel diagnostics mismatch')
            features = diagnostic.measure(patch, record['polarity'])
            close_tree(record['features'], features, record['key'] + ' independent 6/7/8-column LS')
            for parameter in ('point_sigma_px', 'point_offset_xy', 'edge_width_px', 'edge_orientation_rad',
                              'edge_offset_px', 'point_after_edge_sigma_px', 'point_after_edge_offset_xy'):
                exact(record['features'][parameter], features[parameter], 'Exact best-template parameter: ' + parameter)
            for key, value in features.items():
                if type(value) is float:
                    maximum_errors[key] = max(maximum_errors[key], abs(record['features'][key] - value))
            passed = bool(features['informative'] and features['point_minus_edge_fraction'] > 0)
            require(type(record['zero_margin_ablation_passed']) is bool and record['zero_margin_ablation_passed'] == passed,
                    'Zero-margin decision mismatch')
            audited.append(dict(**{k: v for k, v in record.items() if k not in ('features', 'zero_margin_ablation_passed')},
                                features=features, zero_margin_ablation_passed=passed))
    return audited, dict(maximum_errors)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return dict(count=0)
    return dict(count=int(len(values)), minimum=float(np.min(values)), p05=float(np.quantile(values, 0.05)),
                median=float(np.median(values)), p95=float(np.quantile(values, 0.95)), maximum=float(np.max(values)))


def annotate_group(group, by_key):
    require(len(group['keys']) == len(set(group['keys'])) and set(group['keys']) <= by_key.keys(),
            'Duplicate or missing gated observation key')
    passing = [key for key in group['keys'] if by_key[key]['zero_margin_ablation_passed']]
    return dict(**group, passing_keys=passing, baseline_hit=bool(group['keys']),
                diagnostic_hit=bool(passing), new_miss=bool(group['keys']) and not passing)


def rebuild_summary(selection, observations, frozen_summary):
    by_key = {o['key']: o for o in observations}
    groups, memberships = [], defaultdict(set)
    for group in selection['groups']:
        groups.append(annotate_group(group, by_key))
        memberships[group['kind'] + '/' + group['window']].update(group['keys'])
    controls = []
    for index, control in enumerate(selection['controls']):
        selected = [o for o in observations if index in o['provisional_control_indices']]
        controls.append(dict(**control, selected_measured_states=len(selected),
                             passing_measured_states=sum(o['zero_margin_ablation_passed'] for o in selected)))
        memberships[f'control/{index}/{control["label"]}'].update(o['key'] for o in selected)
    retention = {}
    for kind in ('dense', 'pilot', 'anchor'):
        group = [g for g in groups if g['kind'] == kind]
        retention[kind] = dict(samples=len(group), baseline_hits=sum(g['baseline_hit'] for g in group),
            diagnostic_hits=sum(g['diagnostic_hit'] for g in group),
            newly_missed_samples=[dict(clip=g['clip'], window=g['window'], frame=g['frame']) for g in group if g['new_miss']],
            baseline_ambiguous_samples=sum(len(g['keys']) > 1 for g in group),
            no_longer_preserves_all_baseline_ids=sum(bool(g['keys']) and set(g['keys']) != set(g['passing_keys']) for g in group))
    distributions = {}
    for name, keys in sorted(memberships.items()):
        features = [by_key[key]['features'] for key in keys]
        distributions[name] = {feature: quantiles([f[feature] for f in features]) for feature in FEATURE_NAMES}
        distributions[name]['availability'] = dict(observations=len(keys), informative=sum(f['informative'] for f in features),
            conditional_informative=sum(f['conditional_informative'] for f in features),
            distributions_include_uninformative_zero_coded_gains=True)
    capture_records = {}
    for clip in COUNTS:
        maximum = max(o['frame'] for o in selection['observations'] if o['clip'] == clip)
        capture_records[clip] = dict(backend='FFMPEG', fps=10.0, width=4784, height=3190,
                                    decoded_through_frame=maximum, reported_frame_count=COUNTS[clip])
    summary = dict(reference_retention=retention, provisional_controls=controls, groups=groups,
        feature_distributions=distributions, selected_observations=len(observations),
        uninformative_patches=sum(not o['features']['informative'] for o in observations),
        promoted=False, detector_rerun=False, thresholds_tuned=False, airborne_precision=None,
        airborne_recall=None, false_alarms_per_minute=None, limitations=LIMITATIONS,
        completed=True, freeze_sha256=CONTEXT_FREEZE,
        outputs_sha256={name: digest(CONTEXT / name) for name in ('native_patches.npz', 'observations.json')},
        capture_records=capture_records)
    close_tree(frozen_summary, summary, 'complete independent summary')
    require(sum(c['selected_measured_states'] for c in controls) == 70, 'Control denominator changed')
    return summary


def audit():
    integrity = Integrity()
    auditor_hash = integrity.bind(Path(__file__))
    test_hash = integrity.bind(ROOT / 'tests/unit/test_audit_accuracy_v36_context.py')
    frozen, shadow, summary = verify_inputs(integrity)
    selection = rebuild_selection(shadow)
    observations, errors = verify_patches(selection)
    computed = rebuild_summary(selection, observations, summary)
    integrity.recheck()
    return dict(schema='seaqr.accuracy-v36-context-independent-audit.v1', verified=True,
        experiment=str(CONTEXT), context_freeze_sha256=CONTEXT_FREEZE, auditor_sha256=auditor_hash,
        auditor_test_sha256=test_hash, checked_file_count=len(integrity.files), checked_files_sha256=integrity.files,
        inputs_rehashed_after_analysis=True, frozen_context_tests_passed=39,
        independent_selection_recomputed=True, patches_verified=358, patch_dtype='uint8', patch_shape=[25, 25],
        feature_module_imported=False, computation='Independent direct 6-column quadratic, 7-column point/edge, and 8-column edge+point least-squares fits',
        coefficient_scaling_verified=True, numeric_tolerance=dict(relative=RTOL, absolute=ATOL),
        maximum_absolute_feature_differences=errors, reference_retention=computed['reference_retention'],
        controls=dict(selected_measured_states=70, passing_measured_states=sum(c['passing_measured_states'] for c in computed['provisional_controls']),
                      authoritative_negative_labels=False), zero_margin_any_alternative_decisions_match=True,
        complete_summary_verified=True, video_to_patch_extraction_independently_validated=False,
        extraction_boundary='No original AVI was reopened. Saved patch bytes, hashes, journal identities, frame/crop metadata and source-SHA declarations were checked; decoder output/indexing is not independently re-decoded.',
        video_files_read=False, raw16_accessed=False, remote_state_accessed=False, detector_rerun=False,
        defaults_changed=False, classifier_promoted=False, airborne_accuracy_established=False,
        scope='Selected development observations only, not whole-camera false alarms, independent generalization, or closed-loop improvement.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError('Fresh audit receipt required')
    result = audit()
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'patches_verified', 'reference_retention', 'controls',
        'maximum_absolute_feature_differences', 'video_to_patch_extraction_independently_validated')}, indent=2))


if __name__ == '__main__':
    main()
