"""Frozen causal qualification ablations over audited development journals.

No source-video decode, detector rerun, feedback change, or physical-class claim.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from accuracy_v36_policy import ARMS,CausalQualification,PolicyConfig
from score_phase20_accuracy import score_rows,validate_labels
from tiny_target.visible_regression import load_reference

AUDIT=ROOT/'results/tiny_target/visible_validation_v34_20260923/audit_20260924'
COUNTS={'0029':687,'0126':674,'0055':689,'0082':691}
REFERENCES={
 'dense_labels':ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914/annotations.json',
 'dense_packet':ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json',
 'pilot_labels':ROOT/'results/tiny_target/phase20/accuracy_baseline_v1_20260914/annotations.json',
 'pilot_packet':ROOT/'results/tiny_target/phase20/accuracy_baseline_v1_20260914/source_review/packet.json',
 'anchors_0029':ROOT/'results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json',
 'anchors_0126':ROOT/'results/tiny_target/phase19/chunk0126_avi_20260913/visual_review/visual_annotations.json',
 'controls':ROOT/'results/tiny_target/phase20/clutter_review_20260913/background_controls.json',
}
SOURCES=('scripts/accuracy_v36_policy.py','scripts/evaluate_accuracy_v36.py',
 'scripts/score_phase20_accuracy.py','tiny_target/visible_regression.py',
 'tests/unit/test_accuracy_v36_policy.py','docs/accuracy_v36_plan.md')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def verified_inputs():
    summary=read(AUDIT/'summary_verified_01.json')
    manifest=read(AUDIT/'evidence/export_manifest_v34_01.json')
    if not (summary['verified'] and summary['completed'] and summary['completed_trials']==16
            and summary['export_manifest_sha256']==sha(AUDIT/'evidence/export_manifest_v34_01.json')):
        raise ValueError('Completed audited v34 required')
    inputs={}
    for c,count in COUNTS.items():
        path=AUDIT/'evidence/run'/('full_repeat0_'+c)
        files={}
        for suffix in ('/launch.json','/report.json','/frames.jsonl','.v34.json','.v29.json'):
            relative='run/full_repeat0_'+c+suffix
            p=AUDIT/'evidence'/relative; digest=sha(p)
            if digest!=manifest['files'][relative]:raise ValueError('Changed v34 evidence '+relative)
            files[str(p)]=digest
        launch=read(path/'launch.json');report=read(path/'report.json')
        if not (report['completed'] and report['full_clip'] and report['frames']==count
                and report['source_sha256']==launch['source_sha256'] and launch['fps']==10
                and launch['configuration']['motion_quality_window_hits']==8
                and launch['configuration']['motion_quality_minimum_hits']==5
                and launch['configuration']['minimum_moving_excursion_px']==12):
            raise ValueError('Invalid source/completion/policy scope')
        inputs[c]=dict(path=str(path),files_sha256=files,source_sha256=launch['source_sha256'],frames=count)
    for name in ('dense','pilot'):
        labels=read(REFERENCES[name+'_labels']);packet=read(REFERENCES[name+'_packet'])
        validate_labels(labels,packet)
        for s in packet['plan']['sources']:
            if s['clip_id'] in inputs and s['sha256']!=inputs[s['clip_id']]['source_sha256']:
                raise ValueError('Reference source mismatch')
    for c in ('0029','0126'):
        digest,_=load_reference(REFERENCES['anchors_'+c])
        if digest!=inputs[c]['source_sha256']:raise ValueError('Anchor source mismatch')
    if read(REFERENCES['controls'])['source_sha256']!=inputs['0126']['source_sha256']:
        raise ValueError('Control source mismatch')
    return inputs


def inside(xy,crop):
    x,y,w,h=crop
    return x<=xy[0]<x+w and y<=xy[1]<y+h


def regression_samples(evaluation):
    return {(w['window_id'],e['frame_index']):e for w in evaluation['positive_windows'] for e in w['evidence']}


def compare_samples(baseline,candidate):
    a,b=regression_samples(baseline),regression_samples(candidate)
    if set(a)!=set(b):raise ValueError('Changed reference denominator')
    lost=[]; changed=[]
    for key,old in a.items():
        new=b[key]
        if old['qualified_measured_hit'] and not new['qualified_measured_hit']:
            lost.append(dict(window=key[0],frame=key[1]))
        if old['qualified_measured_hit'] and new['qualified_measured_hit'] and old['assigned_track_id']!=new['assigned_track_id']:
            changed.append(dict(window=key[0],frame=key[1],before=old['assigned_track_id'],after=new['assigned_track_id']))
    return dict(no_new_misses=not lost,lost_visible_samples=lost,changed_assignments=changed,
        baseline_hits=sum(e['qualified_measured_hit'] for e in a.values()),
        candidate_hits=sum(e['qualified_measured_hit'] for e in b.values()),samples=len(a))


def analyze_clip(c,spec,output):
    references={name:(read(REFERENCES[name+'_labels']),read(REFERENCES[name+'_packet'])) for name in ('dense','pilot')}
    required_frames=set()
    for labels,packet in references.values():
        windows={w['id']:w for w in packet['windows']}
        for entry in labels['positive_windows']:
            if windows[entry['window_id']]['clip_id']==c:
                required_frames.update(x['frame_index'] for x in entry['visible_samples'])
    anchors=[]
    if c in ('0029','0126'):
        _,events=load_reference(REFERENCES['anchors_'+c])
        anchors=[dict(event_id=e['event_id'],polarity=e['polarity'],**a) for e in events for a in e['anchors'] if a['required']]
        required_frames.update(a['frame_index'] for a in anchors)
    controls=read(REFERENCES['controls'])['controls'] if c=='0126' else []
    stats={arm:dict(measured=0,predicted=0,ids=set(),per_frame=[],controls=[dict(measured=0,predicted=0,ids=set()) for _ in controls]) for arm in ARMS}
    sparse_rows={arm:[] for arm in ARMS};anchor_evidence={arm:[] for arm in ARMS}
    filter_=CausalQualification();frame_count=0
    with (Path(spec['path'])/'frames.jsonl').open() as source,(output/(c+'_decisions.jsonl')).open('x') as log:
        for line in source:
            row=json.loads(line)
            if row['frame_index']!=frame_count or row['timestamp_ns']!=frame_count*100000000:
                raise ValueError('Non-contiguous source journal')
            decisions=filter_.update(row);frame_count+=1
            relevant=[t for t in row['tracks'] if t['qualified_moving']]
            compact=dict(frame_index=row['frame_index'],timestamp_ns=row['timestamp_ns'],segment=row['segment'],
                tracks=[dict(track_id=t['track_id'],measured=t['measured'],source_xy=t['source_xy'],
                    measurement_source_xy=t['measurement_source_xy'],**decisions[t['segment'],t['track_id']]) for t in relevant])
            log.write(json.dumps(compact,allow_nan=False)+'\n')
            for arm in ARMS:
                accepted=[t for t in relevant if decisions[t['segment'],t['track_id']]['decisions'][arm]]
                record=stats[arm]; measured=sum(t['measured'] for t in accepted)
                record['measured']+=measured;record['predicted']+=len(accepted)-measured
                record['per_frame'].append(len(accepted));record['ids'].update((t['segment'],t['track_id']) for t in accepted)
                for j,control in enumerate(controls):
                    if control['frames_inclusive'][0]<=row['frame_index']<=control['frames_inclusive'][1]:
                        local=[t for t in accepted if inside(t['measurement_source_xy'] if t['measured'] else t['source_xy'],control['crop_xywh'])]
                        m=sum(t['measured'] for t in local);record['controls'][j]['measured']+=m
                        record['controls'][j]['predicted']+=len(local)-m
                        record['controls'][j]['ids'].update((t['segment'],t['track_id']) for t in local)
                if row['frame_index'] in required_frames:
                    sparse_rows[arm].append(dict(frame_index=row['frame_index'],segment=row['segment'],coverage=row['coverage'],
                        candidates=[dict(source_xy=p['source_xy'],polarity=p['polarity']) for p in row['candidates']],
                        tracks=[{k:t[k] for k in ('track_id','segment','measured','qualified_moving','measurement_source_xy')} for t in accepted]))
                for anchor in (a for a in anchors if a['frame_index']==row['frame_index']):
                    ids=[str(t['segment'])+'/'+t['track_id'] for t in accepted if t['measured']
                        and t['track_id'].split(':')[0]==anchor['polarity']
                        and math.dist(t['measurement_source_xy'],anchor['xy'])<=anchor['uncertainty_px']+2]
                    anchor_evidence[arm].append(dict(event_id=anchor['event_id'],frame=anchor['frame_index'],ids=ids))
    if frame_count!=spec['frames']:raise ValueError('Missing source frames')
    scores={arm:{name:score_rows(sparse_rows[arm],labels,packet,c,10) for name,(labels,packet) in references.items()} for arm in ARMS}
    result={}
    for arm in ARMS:
        data=stats[arm]
        controls_out=[dict(frames_inclusive=control['frames_inclusive'],crop_xywh=control['crop_xywh'],label=control['label'],
            measured_states=counts['measured'],predicted_states=counts['predicted'],distinct_segment_track_ids=len(counts['ids']))
            for control,counts in zip(controls,data['controls'])]
        comparisons={name:compare_samples(scores['baseline'][name],scores[arm][name]) for name in references}
        lost_anchors=[dict(event_id=a['event_id'],frame=a['frame']) for a,b in zip(anchor_evidence['baseline'],anchor_evidence[arm]) if a['ids'] and not b['ids']]
        anchor_intersections={}
        for event in {x['event_id'] for x in anchor_evidence[arm]}:
            sets=[set(x['ids']) for x in anchor_evidence[arm] if x['event_id']==event]
            anchor_intersections[event]=sorted(set.intersection(*sets))
        result[arm]=dict(qualified_measured_states=data['measured'],qualified_predicted_states=data['predicted'],
            distinct_segment_track_ids=len(data['ids']),qualified_states_per_frame=(data['measured']+data['predicted'])/frame_count,
            maximum_qualified_states_in_frame=max(data['per_frame']),provisional_controls=controls_out,
            provisional_control_measured_states=sum(x['measured_states'] for x in controls_out),
            provisional_control_predicted_states=sum(x['predicted_states'] for x in controls_out),
            references=scores[arm],retention=comparisons,required_anchor_evidence=anchor_evidence[arm],
            lost_required_anchors=lost_anchors,common_id_across_required_anchors=anchor_intersections,
            known_reference_retention_passed=(all(x['no_new_misses'] and not x['changed_assignments'] for x in comparisons.values())
                and not lost_anchors and all(anchor_intersections.values())))
    return dict(clip=c,frames=frame_count,source_sha256=spec['source_sha256'],arms=result,
        decisions_sha256=sha(output/(c+'_decisions.jsonl')),airborne_precision=None,airborne_recall=None,
        false_alarms_per_minute=None,detector_rerun=False,feedback_changed=False)


def run(output):
    if output.exists():raise FileExistsError('Fresh experiment output required')
    inputs=verified_inputs()
    output.mkdir(parents=True)
    with (output/'unit.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/unit','-p','test_accuracy_v36_policy.py','-v'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    frozen=dict(pre_run=True,created_ns=time.time_ns(),policy=asdict(PolicyConfig()),arms=list(ARMS),
        source_sha256={s:sha(ROOT/s) for s in SOURCES},reference_sha256={k:sha(p) for k,p in REFERENCES.items()},
        references={k:str(p) for k,p in REFERENCES.items()},inputs=inputs,unit_log_sha256=sha(output/'unit.log'),
        v34_audit_sha256=sha(AUDIT/'summary_verified_01.json'),media_decoded=False,raw16_accessed=False,
        defaults_changed=False,detector_feedback_unchanged=True)
    write(output/'freeze.json',frozen)
    (output/'implementation').mkdir()
    for s in SOURCES:
        target=output/'implementation'/s;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/s,target)
    results={}
    for c,spec in inputs.items():
        print('Analyzing '+c+' (frozen journal only)',flush=True)
        results[c]=analyze_clip(c,spec,output);write(output/(c+'_results.json'),results[c])
    gates={}
    for arm in ARMS[1:]:
        retained=all(x['arms'][arm]['known_reference_retention_passed'] for x in results.values())
        before=results['0126']['arms']['baseline']['provisional_control_measured_states']
        after=results['0126']['arms'][arm]['provisional_control_measured_states']
        gates[arm]=dict(known_reference_retention_passed=retained,control_measured_before=before,control_measured_after=after,
            eligible_for_further_study=retained and after<before,promoted=False,
            reason='known_reference_regression' if not retained else 'further_source_review_required')
    if any(sha(ROOT/s)!=h for s,h in frozen['source_sha256'].items()) or any(sha(REFERENCES[k])!=h for k,h in frozen['reference_sha256'].items()):
        raise ValueError('Sources/references changed during experiment')
    summary=dict(completed=True,freeze_sha256=sha(output/'freeze.json'),clips={c:{'frames':x['frames'],
        'arms':{a:{k:v for k,v in y.items() if k not in ('references','required_anchor_evidence')} for a,y in x['arms'].items()}} for c,x in results.items()},
        gates=gates,defaults_changed=False,media_decoded=False,raw16_accessed=False,airborne_accuracy_established=False,
        caveat='Shadow output eligibility, not a closed-loop detector rerun. Counts are unlabeled workload or selected provisional nuisance cases, not false-positive rates.')
    write(output/'summary.json',summary);print(json.dumps(gates,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
