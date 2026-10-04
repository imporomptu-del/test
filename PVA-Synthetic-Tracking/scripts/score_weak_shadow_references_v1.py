"""Frozen historical-reference postprocessing, never detector/tracker input.

Export references before inspecting the new experiment. Then score only three
complete, passed per-clip audits. No media, NPZ arrays, native libraries, detector
or tracker are imported. Weak observations are reported separately, never as
ordinary measurements or proof of physical identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = Path(__file__).resolve().with_name('score_feature_selection_references.py')
HISTORICAL_SHA = 'b8698675b8296c66a4923e625219a035a5c3082a0b9b711543830a7072febd20'
CLIPS = ('0029', '0126', '0055')
REFERENCE_SCHEMA = 'seaqr.weak-shadow.historical-references.v1'
SCORE_SCHEMA = 'seaqr.weak-shadow.historical-score.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def historical():
    """Import only the immutable, metadata-only historical scorer."""
    import hashlib
    require(HISTORICAL.is_file() and not HISTORICAL.is_symlink(), 'Missing historical scorer')
    before = HISTORICAL.read_bytes()
    require(hashlib.sha256(before).hexdigest() == HISTORICAL_SHA, 'Historical scorer changed')
    spec = importlib.util.spec_from_file_location('_weak_shadow_historical_reference', HISTORICAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.sha(HISTORICAL) == HISTORICAL_SHA, 'Historical scorer changed during import')
    return module


def own_bindings(old):
    test = Path(__file__).resolve().with_name('test_score_weak_shadow_references_v1.py')
    if not test.is_file():
        test = ROOT/'tests/unit/test_score_weak_shadow_references_v1.py'
    files = (Path(__file__).resolve(), test)
    return {str(path): old.sha(path) for path in files} | {str(HISTORICAL): HISTORICAL_SHA}


def unchanged(old, bindings):
    for path, digest in bindings.items():
        require(old.sha(path) == digest, 'Changed metadata/code input: '+path)


def write_fresh(old, output, value):
    output = Path(output)
    require(output.is_absolute() and output.resolve() == output and output.suffix == '.json'
            and not output.exists() and not output.is_symlink(), 'Fresh absolute JSON output required')
    require(output.parent.is_dir(), 'Output parent must already exist')
    with output.open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return value


def sample_counts(samples):
    return dict(samples=len(samples), by_clip=dict(sorted(Counter(s['clip_id'] for s in samples).items())),
                by_panel=dict(sorted(Counter(s['panel'] for s in samples).items())))


def export_references(output):
    old = historical()
    bindings = own_bindings(old)
    all_samples = old.references(bindings)
    samples = [s for s in all_samples if s['clip_id'] in CLIPS]
    excluded = [s for s in all_samples if s['clip_id'] not in CLIPS]
    require({s['clip_id'] for s in excluded} <= {'0082'}, 'Unexpected excluded cohort')
    excluded_counts = sample_counts(excluded)
    excluded_counts['by_clip'] = {'0082':len(excluded)}
    baseline = []
    for clip in CLIPS:
        paths = [old.JOURNALS/('full_repeat0_'+clip)/name for name in ('frames.jsonl','launch.json','report.json')]
        for path, digest in zip(paths, old.BASELINE_HASHES[clip]):
            old.bind(bindings, path, digest)
        old.validate_run(clip, old.read(paths[1]), old.read(paths[2]))
        baseline.extend(old.score_journal(paths[0], clip, samples, baseline=True)['records'])
    require(len(baseline) == len(samples), 'Historical reference denominator changed')
    value = dict(schema=REFERENCE_SCHEMA, clips=list(CLIPS), counts=sample_counts(samples),
        original_all_four_clip_counts=sample_counts(all_samples), excluded_counts=excluded_counts,
        samples=samples, provenance_sha256=bindings, baseline_saved_assignments_verified=True,
        baseline_by_panel={panel:{stage:sum(r['stages'][stage]['hit'] for r in baseline if r['sample']['panel']==panel)
            for stage in old.STAGES} for panel in sorted({s['panel'] for s in samples})},
        frames={clip:old.COUNTS[clip] for clip in CLIPS}, sources={clip:old.SOURCES[clip] for clip in CLIPS},
        gate='Historical source-coordinate uncertainty + 2 pixels; original per-panel/window maximum-cardinality assignment.',
        panels_overlap=True, independent_encounters=False, references_supplied_to_tracker=False,
        new_weak_outcomes_read=False, source_media_accessed=False,
        timing='Postprocessing developed after experiment launch. Progress-only frame counts, baseline workload counts and stage pass/fail were available; no weak decisions, shadow outcomes or reference scores were inspected before reference export.')
    unchanged(old, bindings)
    return write_fresh(old, output, value)


def load_references(old, path, digest, bindings):
    doc = old.read(old.bind(bindings, path, digest))
    require(doc.get('schema') == REFERENCE_SCHEMA and doc.get('clips') == list(CLIPS)
            and doc.get('baseline_saved_assignments_verified') is True
            and doc.get('references_supplied_to_tracker') is False
            and doc.get('new_weak_outcomes_read') is False, 'Invalid pre-outcome reference export')
    require(doc.get('frames') == {c:old.COUNTS[c] for c in CLIPS}
            and doc.get('sources') == {c:old.SOURCES[c] for c in CLIPS}, 'Reference source/count scope changed')
    provenance = doc.get('provenance_sha256')
    require(type(provenance) is dict and any(Path(p).name == HISTORICAL.name and d == HISTORICAL_SHA
            for p,d in provenance.items()), 'Historical provenance missing')
    import re
    require(all(type(p) is str and Path(p).is_absolute() and type(d) is str and re.fullmatch('[0-9a-f]{64}',d)
                for p,d in provenance.items()), 'Invalid historical provenance hashes')
    # The caller pins the already self-checked export. Do not reopen big
    # historical journals or require their original local paths on the Jetson.
    kept = doc.get('samples')
    require(type(kept) is list and kept and len({old.key(s) for s in kept}) == len(kept), 'Duplicate/missing exported samples')
    for s in kept:
        require(s['clip_id'] in CLIPS and s['polarity'] in ('bright','dark') and type(s['frame_index']) is int
                and 0 <= s['frame_index'] < old.COUNTS[s['clip_id']], 'Invalid exported sample scope')
        old.point(s['source_xy'])
        require(type(s['position_uncertainty_px']) in (int,float) and 0 < s['position_uncertainty_px'] <= 8,
                'Invalid historical uncertainty')
        require(set(s['saved']) == set(old.STAGES), 'Missing historical stages')
        for value in s['saved'].values():
            require(type(value['hit']) is bool and type(value['all_gated_ids']) is list
                    and (value['assigned_id'] is not None) is value['hit']
                    and (value['assigned_id'] is None or value['assigned_id'] in value['all_gated_ids']),
                    'Invalid saved historical assignment')
    excluded = doc.get('excluded_counts', {})
    total = doc.get('original_all_four_clip_counts', {})
    require(doc.get('counts') == sample_counts(kept) and set(excluded.get('by_clip',{})) == {'0082'}
            and excluded.get('samples') == excluded['by_clip']['0082']
            and total.get('samples') == sum(old.PANELS.values()) == len(kept)+excluded['samples']
            and total.get('by_panel') == old.PANELS
            and total.get('by_clip') == {c:n for c,n in dict(doc['counts']['by_clip'],**excluded['by_clip']).items() if n}
            and {p:doc['counts']['by_panel'].get(p,0)+excluded['by_panel'].get(p,0) for p in old.PANELS} == old.PANELS,
            'Exported reference denominators changed')
    return doc


def relative_metadata(old, directory, relative, digest, bindings):
    part = Path(relative)
    require(type(relative) is str and not part.is_absolute() and '..' not in part.parts
            and str(part) == relative and part.suffix in ('.json','.jsonl'), 'Unsafe metadata artifact path')
    return old.bind(bindings, directory/part, digest)


def audited_inputs(old, directory, clip, freeze_sha, plan_sha, bindings):
    base = directory/clip
    path = base/'independent_audit.json'
    audit = old.read(old.bind(bindings, path, old.sha(path)))
    require(audit.get('schema') == 'seaqr.weak-continuation-shadow.audit.v1' and audit.get('passed') is True
            and audit.get('clip') == clip and audit.get('frames') == old.COUNTS[clip]
            and audit.get('source_sha256') == old.SOURCES[clip], 'Complete passed clip audit required')
    for key, expected in (('freeze_sha256',freeze_sha),('plan_sha256',plan_sha),
        ('baseline_journal_non_timing_exact',True),('baseline_output_state_learning_digests_exact',True),
        ('native_state_guards_unchanged',True),('production_changed',False),('weak_learning_enabled',False)):
        require(type(audit.get(key)) is type(expected) and audit[key] == expected, 'Audit isolation/provenance differs: '+key)
    hashes = audit.get('files_sha256', {})
    needed = ('clean/frames.jsonl','shadow/frames.jsonl','shadow/shadow_trace.jsonl',
              'clean.shadow.json','shadow.shadow.json','clean.v29.json','shadow.v29.json')
    require(type(hashes) is dict and all(name in hashes for name in needed), 'Audit lacks scorer input bindings')
    paths = {name:relative_metadata(old, base, name, hashes[name], bindings) for name in needed}
    # Every metadata artifact is rehashed; NPZ arrays are deliberately not read.
    for name, digest in hashes.items():
        if Path(name).suffix in ('.json','.jsonl'):
            relative_metadata(old, base, name, digest, bindings)
    for arm in ('clean','shadow'):
        receipt = old.read(paths[arm+'.shadow.json'])
        require(receipt.get('schema') == 'seaqr.weak-continuation-shadow.run.v1' and receipt.get('passed') is True
                and receipt.get('error') is None and receipt.get('clip') == clip and receipt.get('arm') == arm
                and receipt.get('processed_frames') == receipt.get('expected_frames') == old.COUNTS[clip]
                and receipt.get('freeze_sha256') == freeze_sha and receipt.get('plan_sha256') == plan_sha
                and receipt.get('source_sha256') == old.SOURCES[clip]
                and receipt.get('production_changed') is False and receipt.get('weak_learning_enabled') is False,
                'Run receipt identity/completeness differs')
        if arm == 'shadow':
            require(receipt.get('trace_sha256') == hashes['shadow/shadow_trace.jsonl'], 'Trace receipt binding differs')
    return paths


def weak_observations(old, row, baseline):
    """Applied weak measurement point, not posterior mean or ordinary detection."""
    observations, seen, inverse = [], set(), None
    for record in row['records']:
        note = record.get('weak_evidence', {})
        require(type(note.get('applied')) is bool, 'Missing weak-evidence classification')
        if not note['applied']:
            continue
        identity = str(record['segment'])+'/'+record['track_id']
        require(identity not in seen and note.get('identity') == identity and record['segment'] == row['segment'],
                'Invalid/duplicate weak identity')
        seen.add(identity)
        require(record['measured'] is False and record['measurement_source_xy'] is None
                and note.get('status') == 'weak_kinematic_correction'
                and note.get('is_ordinary_measurement') is False and note.get('physical_identity_verified') is False,
                'Weak evidence mislabeled ordinary/physical truth')
        if inverse is None:
            matrix = np.asarray(baseline['source_to_reference'], dtype=float)
            require(matrix.shape == (3,3) and np.isfinite(matrix).all(), 'Invalid current source/reference transform')
            inverse = np.linalg.inv(matrix)
        point = old.point(note['measurement_reference_xy'])
        homogeneous = inverse @ np.array([*point,1.])
        require(np.isfinite(homogeneous).all() and homogeneous[2] != 0, 'Unprojectable weak source point')
        source = (homogeneous[:2]/homogeneous[2]).tolist()
        observations.append(dict(id=identity, xy=old.point(source), polarity=record['track_id'].split(':')[0]))
    return observations


def assigned(old, samples, observations):
    matches, neighbors = old.assign(samples, observations)
    result = []
    for i in range(len(samples)):
        ids = [observations[j]['id'] for j in neighbors[i]]
        result.append(dict(hit=i in matches, assigned_id=observations[matches[i]]['id'] if i in matches else None,
            all_gated_ids=ids, multiple_gated_alternatives=len(ids)>1,
            shared_gated_observation=any(sum(j in edge for edge in neighbors)>1 for j in neighbors[i])))
    return result


def weak_score_frame(old, row, baseline, samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample['panel'],sample['window_id'])].append(sample)
    observations = weak_observations(old,row,baseline)
    if baseline['coverage']['detection_ready'] is not True:
        require(not observations, 'Weak update on unavailable detector frame')
    return {old.key(s):value for group in groups.values() for s,value in zip(group,assigned(old,group,observations))}


def score_clip(old, paths, clip, samples):
    wanted = defaultdict(list)
    for sample in samples:
        if sample['clip_id'] == clip:
            wanted[sample['frame_index']].append(sample)
    baseline_scores, shadow_scores, weak_scores, coordinates = [], [], {}, []
    with paths['clean/frames.jsonl'].open() as stream, paths['shadow/shadow_trace.jsonl'].open() as tracing:
        count = 0
        for index, lines in enumerate(itertools.zip_longest(stream,tracing)):
            require(all(type(line) is str and line.strip() for line in lines), 'Missing/blank/unequal journal extent')
            baseline, trace = [old.decode(line) for line in lines]
            old.validate_row(baseline,index)
            require(trace.get('frame_index') == index and trace.get('timestamp_ns') == baseline['timestamp_ns']
                    and trace.get('segment') == baseline['segment'] and type(trace.get('records')) is list,
                    'Shadow trace frame/segment differs')
            shadow = dict(baseline,tracks=trace['records'])
            baseline_observations,shadow_observations = old.observations(baseline),old.observations(shadow)
            def multiset(observed):
                return Counter((item['polarity'],*item['xy']) for item in observed['qualified_measurement'])
            a,b = multiset(baseline_observations),multiset(shadow_observations)
            coordinates.append(dict(frame=index,equal=a==b,baseline_count=sum(a.values()),shadow_count=sum(b.values()),
                missing_baseline_coordinate_instances=sum((a-b).values()),additional_shadow_coordinate_instances=sum((b-a).values())))
            group = wanted.get(index,[])
            before, after = old.score_frame(baseline,group), old.score_frame(shadow,group)
            for record in before:
                for stage in old.STAGES:
                    observed,saved = record['stages'][stage],record['sample']['saved'][stage]
                    require(observed['hit'] == saved['hit'] and observed['assigned_id'] == saved['assigned_id']
                            and set(observed['all_gated_ids']) == set(saved['all_gated_ids']),
                            'Baseline historical assignment/gated alternatives changed')
            baseline_scores.extend(before);shadow_scores.extend(after)
            weak_scores.update(weak_score_frame(old,trace,baseline,group))
            count += 1
    require(count == old.COUNTS[clip] and len(baseline_scores) == sum(map(len,wanted.values())), 'Incomplete clip/reference count')
    return baseline_scores,shadow_scores,weak_scores,dict(frames=count,all_frames_equal=all(r['equal'] for r in coordinates),
        differing_frame_count=sum(not r['equal'] for r in coordinates),frame_comparisons=coordinates,
        interpretation='Exact qualified ordinary-measurement source coordinate + polarity multisets, ignoring IDs. No distance tolerance or physical identity inference.')


def score(references, references_sha, directory, freeze_sha, plan_sha, output):
    old = historical()
    bindings = own_bindings(old)
    import re
    require(all(type(d) is str and re.fullmatch('[0-9a-f]{64}',d) for d in (freeze_sha,plan_sha)), 'Caller freeze/plan hashes required')
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), 'Canonical three-clip directory required')
    doc = load_references(old,Path(references),references_sha,bindings)
    before,after,weak,coordinates = [],[],{},{}
    for clip in CLIPS:
        paths = audited_inputs(old,directory,clip,freeze_sha,plan_sha,bindings)
        a,b,w,diagnostic = score_clip(old,paths,clip,doc['samples'])
        before.extend(a);after.extend(b);weak.update(w)
        coordinates[clip]=diagnostic
    comparison = old.compare(before,after)
    require(len(comparison['records']) == len(doc['samples']) == len(weak), 'Reference denominator lost')
    for record in comparison['records']:
        record['shadow'] = record.pop('candidate')
        record['shadow_detection_ready'] = record.pop('candidate_detection_ready')
        record['annotation_gated_weak_evidence'] = weak[old.key(record['reference'])]
    for panel in comparison['panels'].values():
        for value in panel['stages'].values():
            value['shadow_hits'] = value.pop('candidate_hits')
            value['ambiguous_shadow_samples'] = value.pop('ambiguous_candidate_samples')
    old_coherence,new_coherence = old.coherence(before),old.coherence(after)
    result = dict(schema=SCORE_SCHEMA, passed_integrity=True, baseline_saved_assignments_verified=True,
        freeze_sha256=freeze_sha,plan_sha256=plan_sha,references_sha256=references_sha,
        inputs_sha256=bindings,counts=doc['counts'],excluded_counts=doc['excluded_counts'],
        **comparison, baseline_coherence=old_coherence,shadow_coherence=new_coherence,
        coherence_comparison=old.compare_coherence(old_coherence,new_coherence),
        qualified_strong_coordinate_multisets=coordinates,
        differing_qualified_strong_coordinate_frames=sum(v['differing_frame_count'] for v in coordinates.values()),
        annotation_gated_weak_evidence={panel:dict(samples=sum(s['panel']==panel for s in doc['samples']),
            gated_weak_samples=sum(r['annotation_gated_weak_evidence']['hit'] for r in comparison['records'] if r['reference']['panel']==panel),
            ambiguous_samples=sum(r['annotation_gated_weak_evidence']['multiple_gated_alternatives'] or
                r['annotation_gated_weak_evidence']['shared_gated_observation'] for r in comparison['records'] if r['reference']['panel']==panel))
            for panel in sorted({s['panel'] for s in doc['samples']})},
        source_media_accessed=False,native_capture_arrays_read=False,references_supplied_to_tracker=False,
        production_promoted=False,independent_airborne_accuracy_established=False,authoritative_false_positive_rate_available=False,
        panels_overlap=True,samples_are_not_independent_encounters=True,
        interpretation='Historical exposed-development reference regression only. Every original miss and ambiguity remains. '
            'Weak annotation-gated observations never count as ordinary measurements, detections, physical identity or correction truth. '
            'No literal shadow/baseline ID equality is required.')
    unchanged(old,bindings)
    return write_fresh(old,output,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    export = sub.add_parser('export');export.add_argument('--output',type=Path,required=True)
    scoring = sub.add_parser('score')
    for name in ('references','directory','output'):
        scoring.add_argument('--'+name,type=Path,required=True)
    for name in ('references-sha256','freeze-sha256','plan-sha256'):
        scoring.add_argument('--'+name,required=True)
    args = parser.parse_args()
    result = export_references(args.output) if args.action=='export' else score(args.references,
        args.references_sha256,args.directory,args.freeze_sha256,args.plan_sha256,args.output)
    print(json.dumps(dict(schema=result['schema'],counts=result['counts']),sort_keys=True))
