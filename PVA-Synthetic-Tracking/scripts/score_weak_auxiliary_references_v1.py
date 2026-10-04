"""Historical reference regression and separately labelled auxiliary workload.

No media, detector, tracker, native libraries or capture arrays are opened.
Scores are conditional on exposed, previously selected cached rectangles.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_SHA = '57eca00bee6356bd98e553f68ac937cb41603d34d37dd9ae1a99ad560bf4377f'
HELPER_SHA = 'd4e9559a31f8b397df1f0c5e14e96c77ceba331b1350c69c56a255e05a87bb23'
CLIPS = ('0029', '0126', '0055')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def helper():
    path = HERE/'score_weak_shadow_references_v1.py'
    require(not path.is_symlink() and sha(path) == HELPER_SHA, 'Historical wrapper changed')
    spec = importlib.util.spec_from_file_location('_aux_historical_wrapper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha(path) == HELPER_SHA, 'Historical wrapper changed during import')
    return module


def separate_observations(records):
    """Raw current weak points and auxiliary predictions NEVER become detections."""
    observations = {'current_weak': [], 'auxiliary_estimate': [], 'prediction_from_weak': []}
    seen = set()
    for record in records:
        identity = record['identity']
        require(identity not in seen and record['record_type'] == 'auxiliary_gap_support', 'Invalid auxiliary identity')
        seen.add(identity)
        require(record['ordinary_measurement'] is False and record['qualified_detection'] is False
                and record['physical_identity_verified'] is False, 'Auxiliary promoted to detection/identity')
        weak = record['current_weak_observation']
        require(type(weak) is bool and record['prediction_from_weak'] is (not weak)
                and record['evidence_type'] == ('current_weak_observation' if weak else 'prediction_from_weak'),
                'Auxiliary observation/prediction classification differs')
        def item(point):
            require(len(point) == 2 and all(type(v) in (int,float) and math.isfinite(v) for v in point), 'Invalid auxiliary point')
            return dict(id=identity, xy=point, polarity=record['primary_track_id'].split(':')[0])
        observations['auxiliary_estimate'].append(item(record['source_xy']))
        if weak:
            require(record['origin_frame_index'] == record['frame_index'], 'Current weak observation has old origin')
            observations['current_weak'].append(item(record['origin_measurement_source_xy']))
        else:
            require(record['origin_frame_index'] < record['frame_index']
                    and record['current_weak_measurement_reference_xy'] is None, 'Prediction carries a new measurement')
            observations['prediction_from_weak'].append(item(record['source_xy']))
    return observations


def score_rows(old, wrapper, clean, primary, auxiliary, clip, samples):
    wanted = defaultdict(list)
    for sample in samples:
        if sample['clip_id'] == clip:
            wanted[sample['frame_index']].append(sample)
    before, after, reference_rows = [], [], []
    workload, statuses, call_statuses, dropped = Counter(), Counter(), Counter(), Counter()
    weak_events, known_diagnostics = [], []
    count = 0
    for frame, rows in enumerate(itertools.zip_longest(clean, primary, auxiliary)):
        require(all(row is not None for row in rows), 'Unequal journal lengths')
        base, main, aux = rows
        old.validate_row(base, frame)
        for row in (main, aux):
            require(row['frame_index'] == frame and row['timestamp_ns'] == base['timestamp_ns']
                    and row['segment'] == base['segment'], 'Misaligned replay row')
        updated = dict(base, tracks=main['records'])
        require(old.observations(base) == old.observations(updated), 'Primary actual measurements changed')
        group = wanted.get(frame, [])
        original, current = old.score_frame(base, group), old.score_frame(updated, group)
        for record in original:
            for stage in old.STAGES:
                observed, saved = record['stages'][stage], record['sample']['saved'][stage]
                require(observed['hit'] == saved['hit'] and observed['assigned_id'] == saved['assigned_id']
                        and set(observed['all_gated_ids']) == set(saved['all_gated_ids']), 'Historical assignment changed')
        before.extend(original); after.extend(current)
        observations = separate_observations(aux['auxiliary_records'])
        require(base['coverage']['detection_ready'] is True or not observations['auxiliary_estimate'],
                'Auxiliary support on unavailable detector frame')
        for stage, values in observations.items():
            workload[stage+'_records'] += len(values)
            workload[stage+'_frames'] += bool(values)
        workload['frames'] += 1
        workload['scheduled_frames'] += aux['capture_scheduled']
        metrics = aux['auxiliary_metrics']
        for key in ('prior_count','prior_strong_eligible_count','prepared_query_count','actual_provider_calls'):
            workload[key] += metrics[key]
        statuses.update(d['status'] for d in metrics['decisions'])
        dropped.update(d['reason'] for d in metrics['dropped_auxiliary'])
        call_statuses.update(d['status'] for d in aux['capture_calls'])
        for event in aux['auxiliary_records']:
            if event['current_weak_observation']:
                weak_events.append(dict(clip=clip, frame=frame, identity=event['identity'],
                    weak_source_xy=event['origin_measurement_source_xy'], auxiliary_source_xy=event['source_xy'],
                    strong_anchor_timestamp_ns=event['strong_anchor_timestamp_ns'], physical_identity_verified=False))
        panels = defaultdict(list)
        for sample in group:
            panels[(sample['panel'], sample['window_id'])].append(sample)
        primary_by_key = {old.key(r['sample']):r for r in current}
        for panel in panels.values():
            matches = {stage:wrapper.assigned(old, panel, obs) for stage,obs in observations.items()}
            for i,sample in enumerate(panel):
                reference_rows.append(dict(reference=sample, primary=primary_by_key[old.key(sample)]['stages'],
                    auxiliary={stage:rows[i] for stage,rows in matches.items()},
                    capture_scheduled=aux['capture_scheduled'],
                    available_capture_queries=sum(c['status']=='geometrically_complete_capture_supplied' for c in aux['capture_calls']),
                    unavailable_capture_queries=sum(c['status']!='geometrically_complete_capture_supplied' for c in aux['capture_calls'])))
        # Previously inspected nuisance mechanisms and one pre-existing marked gap;
        # these diagnostics do not change acceptance, query geometry or labels.
        if clip == '0126' and frame in (20,34,216):
            known_diagnostics.append(dict(frame=frame, context={20:'Previously reviewed cloud-texture region',
                34:'Previously reviewed ground/building region',216:'Existing marked reference gap'}[frame],
                context_applies_to_prior_inspected_ROI_not_entire_frame=True,
                auxiliary_records=aux['auxiliary_records'], capture_calls=aux['capture_calls'],
                decisions=metrics['decisions']))
        count += 1
    require(count == old.COUNTS[clip] and len(reference_rows) == sum(map(len,wanted.values())), 'Incomplete replay/references')
    return before, after, reference_rows, dict(counts=dict(workload), decision_statuses=dict(statuses),
        capture_call_statuses=dict(call_statuses), drop_reasons=dict(dropped), weak_events=weak_events,
        bounded_known_frame_diagnostics=known_diagnostics)


def score(directory, references, freeze, freeze_sha256, output):
    wrapper = helper(); old = wrapper.historical(); bindings = {}
    directory, freeze = Path(directory), Path(freeze)
    require(freeze.is_absolute() and freeze.resolve() == freeze and sha(freeze) == freeze_sha256, 'Frozen analysis manifest required')
    frozen = old.read(old.bind(bindings, freeze, freeze_sha256))
    require(frozen['schema'] == 'seaqr.weak-auxiliary-replay.freeze.v1' and frozen['pre_run'] is True, 'Analysis must be frozen before outcomes')
    for name,digest in frozen['files_sha256'].items():
        old.bind(bindings,freeze.parent/name,digest)
    require(sha(__file__) == frozen['files_sha256'][Path(__file__).name], 'Unfrozen scorer')
    doc = wrapper.load_references(old,Path(references),REFERENCE_SHA,bindings)
    before,after,records,workloads = [],[],[],{}
    for clip in CLIPS:
        base=directory/clip
        audit_path=base/'independent_audit.json'
        audit=old.read(old.bind(bindings,audit_path,sha(audit_path)))
        require(audit.get('schema')=='seaqr.weak-auxiliary-replay.audit.v1' and audit.get('passed') is True
                and audit.get('clip')==clip and audit.get('frames')==old.COUNTS[clip]
                and audit.get('freeze_sha256')==freeze_sha256 and audit.get('source_sha256')==old.SOURCES[clip],
                'Complete passed isolated replay audit required')
        receipt_path=base/'receipt.json'
        expected=audit['input_files_sha256']
        receipt=old.read(old.bind(bindings,receipt_path,expected[str(receipt_path)]))
        require(receipt['passed'] is True and receipt['error'] is None and receipt['processed_frames']==old.COUNTS[clip], 'Incomplete replay receipt')
        paths=[]
        for name in ('clean','primary','auxiliary'):
            path=base/(name+'.jsonl.gz'); digest=receipt['artifacts_sha256'][path.name]
            require(expected[str(path)]==digest,'Audit does not bind scoring journal')
            paths.append(old.bind(bindings,path,digest))
        with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b,gzip.open(paths[2],'rt') as c:
            results=score_rows(old,wrapper,(old.decode(x) for x in a),(old.decode(x) for x in b),
                (old.decode(x) for x in c),clip,doc['samples'])
        a,b,r,w=results; before.extend(a);after.extend(b);records.extend(r);workloads[clip]=w
    comparison=old.compare(before,after)
    require(len(records)==len(doc['samples'])==356,'Changed historical denominator')
    primary_hits={stage:sum(r['primary'][stage]['hit'] for r in records) for stage in old.STAGES}
    require(primary_hits==dict(actual_measurement=355,qualified_measurement=352),'Frozen baseline reference totals changed')
    result=dict(schema='seaqr.weak-auxiliary-references.v1',passed_integrity=True,freeze_sha256=freeze_sha256,
        references_sha256=REFERENCE_SHA,inputs_sha256=bindings,counts=doc['counts'],primary_hits=primary_hits,
        primary_comparison=comparison,baseline_coherence=old.coherence(before),primary_coherence=old.coherence(after),
        auxiliary_reference_hits={stage:sum(r['auxiliary'][stage]['hit'] for r in records)
            for stage in ('current_weak','auxiliary_estimate','prediction_from_weak')},
        reference_records=records,workload_by_clip=workloads,production_promoted=False,
        independent_airborne_accuracy_established=False,authoritative_false_positive_rate_available=False,
        source_media_accessed=False,detector_rerun=False,panels_overlap=True,
        limitations=['Historical exposed-development samples are not independent encounters.',
            'Weak measurements and propagated/posterior coordinates are separate; none count as ordinary detections.',
            'Archived capture availability is selected by a previous shadow, not exhaustive or blind.',
            'All new auxiliary output is unlabeled review workload, not an authoritative false-positive count.',
            'Missing capture coverage is unavailable evidence, not evidence that no object exists.'])
    wrapper.unchanged(old,bindings)
    return wrapper.write_fresh(old,Path(output),result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('directory','references','freeze','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    result=score(args.directory,args.references,args.freeze,args.freeze_sha256,args.output)
    print(json.dumps(dict(primary_hits=result['primary_hits'],auxiliary_reference_hits=result['auxiliary_reference_hits'])))
