"""Hash-bound metadata-only historical regression. Never opens camera media.

Candidate manifest schema: seaqr.feature-selection.reference-inputs.v1. Its
``clips`` map contains exactly 0029/0126/0055/0082, each with source_sha256,
frames and artifacts (journal, launch, report, execution_receipt). Each artifact
has an absolute regular JSON/JSONL path and sha256. The caller must supply the
manifest's expected SHA256. No detector, source decoder or old replay is run.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target'
JOURNALS = BASE / 'visible_validation_v34_20260923/audit_20260924/evidence/run'
COUNTS = {'0029': 687, '0126': 674, '0055': 689, '0082': 691}
SOURCES = {
    '0029': '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359',
    '0126': 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344',
    '0055': 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f',
    '0082': '465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117',
}
BASELINE_HASHES = {
    '0029': ('e97064888d5901c98ced2be81bf7982f6f21fdd571e6a5197e59aaee4b4fbcf8',
             '9c35be262ae58fe710ac4020a2f3b0f708c180717c20825d398f4aed1bb47b56',
             '4d44d725898acaa5601770f05671efd5592b4fb10f0d758f2cc703f2c89e3f14'),
    '0126': ('411a32306589a49c2b8fa9ba2e77efbdcf6fc5259158017229985613d76415eb',
             '653c760112f01ebc074e23258104b5a3976aec8114abb26865cf211009c6fe5b',
             '4dbc7cd9136701f9f0a6951e1365e9ac1d8c1759b3b1983e76dbb978b869d32f'),
    '0055': ('104bd9f1fdeb37b955ee2dce458e851f67bdd5215e9a35bd4480a929dd871657',
             'b0a99f1cd6b3f05d0703c7d05446ce918d32767e2cc48331049c95c5e68581dc',
             '08c7eaf283cfb7e3361b95ed2aa9eb919f1988e46751b06c6fe0679e23f860d8'),
    '0082': ('5e7bb45cfe6c389a0c2ae8c334f12d48896f24289f5d46b4794e942fde23af97',
             'd79b634cc2204f5f600116001c02555df3bf739e5f14498ac410dede5d56aa43',
             '846474305e0c595d1a91f63b3632d65a5c79f2b1f5660608f87668625c3d6517'),
}
REFERENCE_FILES = (
    (BASE / 'accuracy_v55_20260926/trace_01/trace.json',
     'd74980ab3b5918a7385b989d993618a7e2a274b3e47fb4923505f774b0eefdf8'),
    (BASE / 'accuracy_v40_20260925/coverage_workload_01/visible_reference_evidence.json',
     'b310121f6354c6b8d63bc5277babb16c6c10d3dd67e27dcd1dc1f25d94a6d0e8'),
)
PANELS = {'dense': 285, 'pilot': 28, 'anchor': 24, 'compact_light': 8, 'grid': 11}
BASELINE_QUALIFIED = {'dense': 284, 'pilot': 28, 'anchor': 24, 'compact_light': 6, 'grid': 10}
STAGES = ('actual_measurement', 'qualified_measurement')


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), 'Nonfinite JSON number')
    return value


def decode(text):
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(text, object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=invalid)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'Literal absolute regular metadata path required')
    require(path.suffix in ('.json', '.jsonl', '.py'), 'Only metadata/code may be read')
    return path


def sha(path):
    digest = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(bindings, path, expected):
    path = regular(path)
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid SHA256')
    require(sha(path) == expected, 'Changed bound artifact: ' + str(path))
    require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting binding')
    bindings[str(path)] = expected
    return path


def read(path):
    path = regular(path)
    require(path.suffix == '.json' and path.stat().st_size <= 64 * 1024 * 1024,
            'Bounded JSON metadata required')
    return decode(path.read_text())


def point(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(v) in (float, int) and math.isfinite(v) for v in value),
            'Finite actual source coordinate required')
    return value


def key(sample):
    return tuple(sample[k] for k in ('panel', 'clip_id', 'window_id', 'frame_index'))


def saved_stage(value):
    if isinstance(value, list):
        require(len(value) >= 3, 'Malformed saved reference stage')
        return dict(hit=value[0], assigned_id=value[1], all_gated_ids=value[2])
    return {k: value[k] for k in ('hit', 'assigned_id', 'all_gated_ids')}


def references(bindings):
    trace, grid = [read(bind(bindings, path, expected)) for path, expected in REFERENCE_FILES]
    result = []
    for record in trace['reference_summary']['records']:
        old = record['original']
        item = {k: old[k] for k in ('panel', 'clip_id', 'window_id', 'frame_index',
                                   'source_xy', 'position_uncertainty_px', 'polarity')}
        stages = old['stages']
        name = 'baseline_qualified' if 'baseline_qualified' in stages else 'strict_qualified_measurement'
        item['saved'] = dict(actual_measurement=saved_stage(stages['actual_measurement']),
                             qualified_measurement=saved_stage(stages[name]))
        require(item['saved']['qualified_measurement']['assigned_id']
                == record['original_strict_assigned_identity'], 'Changed saved assignment')
        result.append(item)
    by_key = {key(s): s for s in result}
    require(len(by_key) == len(result), 'Duplicate historical sample')
    for old in grid:
        item = dict(panel='grid', **{k: old[k] for k in ('clip_id', 'window_id', 'frame_index',
                        'source_xy', 'position_uncertainty_px', 'polarity')})
        item['saved'] = dict(actual_measurement=saved_stage(old['stages']['actual_measurement']),
                            qualified_measurement=saved_stage(old['stages']['strict_qualified_measurement']))
        if key(item) in by_key:
            require(item == by_key[key(item)], 'Changed overlapping historical sample')
        else:
            result.append(item)
            by_key[key(item)] = item
    require(Counter(s['panel'] for s in result) == PANELS, 'Historical denominator changed')
    for sample in result:
        require(sample['clip_id'] in COUNTS and type(sample['frame_index']) is int
                and 0 <= sample['frame_index'] < COUNTS[sample['clip_id']], 'Invalid reference frame')
        point(sample['source_xy'])
        require(sample['polarity'] in ('bright', 'dark')
                and type(sample['position_uncertainty_px']) in (float, int)
                and 0 < sample['position_uncertainty_px'] <= 8, 'Invalid reference polarity/gate')
    return sorted(result, key=key)


def assign(samples, observations):
    """Historical maximum-cardinality, distance-ordered gated assignment."""
    neighbors = [sorted((j for j, o in enumerate(observations)
                         if o['polarity'] == s['polarity']
                         and math.dist(o['xy'], s['source_xy']) <= s['position_uncertainty_px'] + 2),
                        key=lambda j: (math.dist(observations[j]['xy'], s['source_xy']), j))
                 for s in samples]
    owner = {}

    def augment(i, seen):
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(samples)):
        augment(i, set())
    return {i: j for j, i in owner.items()}, neighbors


def observations(row):
    actual, qualified, seen = [], [], set()
    for track in row['tracks']:
        require(type(track['segment']) is int and track['segment'] >= 0
                and track['segment'] == row['segment'], 'Track segment mismatch')
        name = track['track_id']
        require(isinstance(name, str) and re.fullmatch('(bright|dark):[^/]+', name), 'Invalid track ID')
        identity = str(track['segment']) + '/' + name
        require(identity not in seen, 'Duplicate track identity')
        seen.add(identity)
        require(type(track['measured']) is bool and type(track['qualified_moving']) is bool,
                'Boolean measurement/qualification required')
        if not track['measured']:
            require(track['measurement_source_xy'] is None, 'Coast cannot carry actual measurement')
            continue
        item = dict(id=identity, xy=point(track['measurement_source_xy']), polarity=name.split(':')[0])
        actual.append(item)
        if track['qualified_moving']:
            qualified.append(item)
    return dict(actual_measurement=actual, qualified_measurement=qualified)


def score_frame(row, samples):
    groups = defaultdict(list)
    for sample in samples:
        require(sample['frame_index'] == row['frame_index'], 'Reference/frame mismatch')
        groups[(sample['panel'], sample['window_id'])].append(sample)
    values = observations(row)
    # Preserve all denominators. An unavailable frame cannot produce a hit.
    if row['coverage']['detection_ready'] is not True:
        values = {stage: [] for stage in STAGES}
    scores = []
    for group in groups.values():
        records = [dict(sample=s, detection_ready=row['coverage']['detection_ready'], stages={}) for s in group]
        for stage in STAGES:
            matches, neighbors = assign(group, values[stage])
            for i, record in enumerate(records):
                assigned = values[stage][matches[i]]['id'] if i in matches else None
                ids = [values[stage][j]['id'] for j in neighbors[i]]
                record['stages'][stage] = dict(hit=assigned is not None, assigned_id=assigned,
                    all_gated_ids=ids, multiple_gated_alternatives=len(ids) > 1,
                    shared_gated_observation=any(sum(j in edge for edge in neighbors) > 1 for j in neighbors[i]))
        scores.extend(records)
    return scores


def validate_row(row, index):
    require(type(row.get('frame_index')) is int and row['frame_index'] == index
            and type(row.get('timestamp_ns')) is int and row['timestamp_ns'] == index * 100000000,
            'Missing, reordered or duplicate source frame')
    require(type(row.get('segment')) is int and row['segment'] >= 0, 'Invalid segment')
    cov = row['coverage']
    require(cov['full_shape_hw'] == [3190, 4784] and cov['configured_crop'] is None
            and cov['native_pixel_sampling'] is True and type(cov['detection_ready']) is bool,
            'Full native uncropped coverage required')
    if cov['detection_ready']:
        require(cov['warmup'] is False and cov['searchable_pixels'] > 0
                and not row['motion'].get('reset') and not row['motion'].get('pva_failure'),
                'Ready frame contradicts failure/warmup/coverage')


def score_journal(path, clip, samples, baseline=False):
    wanted = defaultdict(list)
    for sample in samples:
        if sample['clip_id'] == clip:
            wanted[sample['frame_index']].append(sample)
    result, count, ready = [], 0, 0
    with regular(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank journal row')
            row = decode(line)
            validate_row(row, count)
            # Validate identity/actual-coordinate semantics for every frame, not just references.
            observations(row)
            count += 1
            ready += row['coverage']['detection_ready']
            result.extend(score_frame(row, wanted.get(row['frame_index'], [])))
    require(count == COUNTS[clip], 'Incomplete/full-clip frame count mismatch')
    require(len(result) == sum(map(len, wanted.values())), 'Missing historical reference frames')
    if baseline:
        for record in result:
            for stage in STAGES:
                actual, saved = record['stages'][stage], record['sample']['saved'][stage]
                require(actual['hit'] == saved['hit'] and actual['assigned_id'] == saved['assigned_id']
                        and set(actual['all_gated_ids']) == set(saved['all_gated_ids']),
                        'Frozen baseline reference assignment/gated alternatives changed')
    return dict(frames=count, ready_frames=ready, samples=len(result), records=sorted(result, key=lambda r: key(r['sample'])))


def validate_run(clip, launch, report):
    require(launch.get('source_sha256') == report.get('source_sha256') == SOURCES[clip], 'Source identity mismatch')
    require(launch.get('fps') == 10 and report.get('frames') == COUNTS[clip]
            and report.get('completed') is True and report.get('full_clip') is True,
            'Completed full nominal10fps clip required')
    require(launch.get('configuration', {}).get('input_bit_depth') == 8, '8-bit input required')
    require(report.get('configuration') == launch.get('configuration'), 'Report/launch configuration mismatch')


def validate_candidate(clip, entry, baseline_launch, bindings):
    require(entry.get('frames') == COUNTS[clip] and entry.get('source_sha256') == SOURCES[clip],
            'Candidate source/count outside allowlist')
    artifacts = entry.get('artifacts', {})
    require(set(artifacts) == {'journal', 'report', 'launch', 'execution_receipt'}, 'Exact candidate artifact roles required')
    paths = {role: bind(bindings, spec['path'], spec['sha256']) for role, spec in artifacts.items()}
    require(len(set(paths.values())) == 4 and paths['journal'].name == 'frames.jsonl'
            and all(paths[k].suffix == '.json' for k in ('launch', 'report', 'execution_receipt')),
            'Distinct JSON artifacts and JSONL journal required')
    launch, report, receipt = [read(paths[k]) for k in ('launch', 'report', 'execution_receipt')]
    validate_run(clip, launch, report)
    require(receipt.get('passed') is True and receipt.get('error') is None
            and receipt.get('clip') == clip and receipt.get('source', {}).get('sha256') == SOURCES[clip]
            and receipt.get('processed_frames') == receipt.get('decoded_frames_verified') == COUNTS[clip],
            'Successful matching full candidate receipt required')
    for role in ('journal', 'report', 'launch'):
        require(receipt.get(role + '_sha256') == artifacts[role]['sha256'], 'Candidate receipt artifact binding mismatch')
    for field in ('detector_configuration_changed', 'tracker_configuration_changed', 'global_motion_gates_changed',
                  'annotations_supplied_to_detector', 'raw16_accessed', 'sealed_holdouts_accessed', 'production_promotion'):
        require(receipt.get(field) is False, 'Candidate must declare unchanged/protected scope: ' + field)
    require(receipt.get('feature_algorithm_changed') is True, 'Feature-change provenance required')
    for field in ('configuration', 'config_sha256', 'motion_config_sha256', 'package_sha256'):
        require(field in launch and launch[field] == baseline_launch[field], 'Frozen launch detector/package changed: ' + field)
    return paths['journal']


def compare(before, after):
    old = {key(r['sample']): r for r in before}
    new = {key(r['sample']): r for r in after}
    require(len(old) == len(before) and len(new) == len(after) and old.keys() == new.keys(),
            'Changed/duplicate reference denominator')
    result, panels = [], {}
    for k, a in old.items():
        b = new[k]
        require(a['sample'] == b['sample'], 'Reference values changed')
        changes = {stage: dict(newly_lost=a['stages'][stage]['hit'] and not b['stages'][stage]['hit'],
                              newly_recovered=not a['stages'][stage]['hit'] and b['stages'][stage]['hit'])
                   for stage in STAGES}
        result.append(dict(reference=a['sample'], baseline=a['stages'], candidate=b['stages'],
                           candidate_detection_ready=b['detection_ready'], changes=changes))
    for panel in sorted({r['reference']['panel'] for r in result}):
        group = [r for r in result if r['reference']['panel'] == panel]
        panels[panel] = dict(samples=len(group), stages={stage: dict(
            baseline_hits=sum(r['baseline'][stage]['hit'] for r in group),
            candidate_hits=sum(r['candidate'][stage]['hit'] for r in group),
            newly_lost=[list(key(r['reference'])) for r in group if r['changes'][stage]['newly_lost']],
            newly_recovered=[list(key(r['reference'])) for r in group if r['changes'][stage]['newly_recovered']],
            ambiguous_candidate_samples=sum(r['candidate'][stage]['multiple_gated_alternatives']
                or r['candidate'][stage]['shared_gated_observation'] for r in group)) for stage in STAGES})
    return dict(panels=panels, records=result,
                no_new_qualified_losses=not any(r['changes']['qualified_measurement']['newly_lost'] for r in result),
                no_new_actual_losses=not any(r['changes']['actual_measurement']['newly_lost'] for r in result))


def coherence(records):
    groups = defaultdict(list)
    for record in records:
        s = record['sample']
        groups[(s['panel'], s['clip_id'], s['window_id'])].append(record)
    result = []
    for group, rows in sorted(groups.items()):
        stages = {}
        for stage in STAGES:
            support = Counter(identity for row in rows for identity in row['stages'][stage]['all_gated_ids'])
            intersection = sorted(set.intersection(*(set(r['stages'][stage]['all_gated_ids']) for r in rows)))
            stages[stage] = dict(any_identity_samples=sum(r['stages'][stage]['hit'] for r in rows),
                best_coherent_identity_samples=max(support.values(), default=0),
                complete_coherent_identities=intersection,
                ambiguous_samples=sum(r['stages'][stage]['multiple_gated_alternatives']
                    or r['stages'][stage]['shared_gated_observation'] for r in rows))
        result.append(dict(panel=group[0], clip=group[1], window=group[2], samples=len(rows), stages=stages))
    return dict(groups=result, physical_identity_or_continuity_established=False,
                interpretation='Per-window identity support only; unknown intervening visibility and ambiguous gates are not physical track continuity.')


def compare_coherence(baseline, candidate):
    def group_key(group):
        return group['panel'], group['clip'], group['window']
    old = {group_key(g): g for g in baseline['groups']}
    new = {group_key(g): g for g in candidate['groups']}
    require(old.keys() == new.keys(), 'Coherence reference groups changed')
    changes = []
    for k, a in old.items():
        b = new[k]
        require(a['samples'] == b['samples'], 'Coherence denominator changed')
        for stage in STAGES:
            before, after = a['stages'][stage], b['stages'][stage]
            changes.append(dict(panel=k[0], clip=k[1], window=k[2], stage=stage,
                baseline_best_coherent_samples=before['best_coherent_identity_samples'],
                candidate_best_coherent_samples=after['best_coherent_identity_samples'],
                coherent_support_decreased=after['best_coherent_identity_samples'] < before['best_coherent_identity_samples'],
                lost_complete_coherent_support=bool(before['complete_coherent_identities'])
                    and not bool(after['complete_coherent_identities']),
                candidate_ambiguous_samples=after['ambiguous_samples']))
    return dict(changes=changes,
        no_new_coherent_support_losses=not any(c['coherent_support_decreased']
            or c['lost_complete_coherent_support'] for c in changes),
        literal_baseline_identity_equality_required=False,
        physical_identity_claim=False)


def run(output, candidate_manifest=None, manifest_sha=None):
    output = Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and not output.is_symlink() and output.suffix == '.json', 'Fresh absolute JSON output required')
    bindings, baseline, candidate, frames = {}, [], [], {}
    own_path = Path(__file__).resolve()
    bindings[str(own_path)] = sha(own_path)
    samples = references(bindings)
    manifest = None
    if candidate_manifest is not None:
        manifest = read(bind(bindings, candidate_manifest, manifest_sha))
        require(manifest.get('schema') == 'seaqr.feature-selection.reference-inputs.v1'
                and set(manifest.get('clips', {})) == set(COUNTS), 'Exact four-clip candidate manifest required')
    for clip in COUNTS:
        paths = [JOURNALS / ('full_repeat0_' + clip) / name for name in ('frames.jsonl', 'launch.json', 'report.json')]
        for path, digest in zip(paths, BASELINE_HASHES[clip]):
            bind(bindings, path, digest)
        launch, report = read(paths[1]), read(paths[2])
        validate_run(clip, launch, report)
        old = score_journal(paths[0], clip, samples, baseline=True)
        baseline.extend(old['records'])
        frames[clip] = dict(baseline={k: old[k] for k in ('frames', 'ready_frames', 'samples')})
        if manifest is not None:
            path = validate_candidate(clip, manifest['clips'][clip], launch, bindings)
            new = score_journal(path, clip, samples)
            candidate.extend(new['records'])
            frames[clip]['candidate'] = {k: new[k] for k in ('frames', 'ready_frames', 'samples')}
    if manifest is None:
        candidate = baseline
    comparison = compare(baseline, candidate)
    require({k: v['samples'] for k, v in comparison['panels'].items()} == PANELS, 'Changed panel denominator')
    require({k: v['stages']['qualified_measurement']['baseline_hits'] for k, v in comparison['panels'].items()}
            == BASELINE_QUALIFIED, 'Frozen baseline qualified totals changed')
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Input changed during scoring')
    old_coherence, new_coherence = coherence(baseline), coherence(candidate)
    result = dict(schema='seaqr.feature-selection.historical-score.v1', completed=True,
        baseline_self_check_passed=True, candidate_evaluated=manifest is not None,
        inputs_sha256=bindings, frames=frames, **comparison,
        baseline_coherence=old_coherence, candidate_coherence=new_coherence,
        coherence_comparison=compare_coherence(old_coherence, new_coherence),
        panels_overlap=True, samples_are_not_independent_encounters=True,
        independent_airborne_accuracy_established=False, authoritative_false_positive_rate_available=False,
        media_accessed=False, holdout_media_accessed=False, production_promoted=False,
        interpretation='Visible-feature development regression, not airborne recall. Original baseline misses remain in every denominator; no prediction counts as an actual measurement. No candidate-label input or literal baseline track-ID requirement.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline-self-check', action='store_true')
    group.add_argument('--candidate-manifest', type=Path)
    parser.add_argument('--candidate-manifest-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.candidate_manifest) != bool(args.candidate_manifest_sha256):
        parser.error('Candidate manifest requires its expected SHA256, and vice versa')
    result = run(args.output, args.candidate_manifest, args.candidate_manifest_sha256)
    print(json.dumps({k: result[k] for k in ('completed', 'baseline_self_check_passed',
        'candidate_evaluated', 'no_new_actual_losses', 'no_new_qualified_losses')}, indent=2))
