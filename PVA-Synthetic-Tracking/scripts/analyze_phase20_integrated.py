"""Audit completed full-development executions; never turn predictions into hits."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256
from run_phase20_maturity import evaluate
from score_phase20_accuracy import score_rows
from audit_phase20_encounter_results import inspect_run


def verify_run(path,frozen,config_name,expected_frames,full_clip):
    launch=json.loads((path/'launch.json').read_text());report=json.loads((path/'report.json').read_text())
    if not report['completed'] or report['frames']!=expected_frames or report['full_clip']!=full_clip:
        raise ValueError('Incomplete or wrong-scope execution: '+str(path))
    if launch['max_frames']!=(None if full_clip else expected_frames) or abs(launch['fps']-10)>1e-9:
        raise ValueError('Frame-limit/cadence changed')
    if [launch['source_probe']['height'],launch['source_probe']['width']]!=frozen['native_shape_hw']:
        raise ValueError('Source resolution changed')
    if launch['annotations_supplied_to_detector'] or launch['config_sha256']!=frozen['files_sha256'][config_name]:
        raise ValueError('Configuration/label boundary changed')
    if launch['motion_config_sha256']!=frozen['files_sha256']['configs/evaluation/phase20_motion_v8.json']:
        raise ValueError('Motion reference changed')
    package={n.removeprefix('tiny_target/'):v for n,v in frozen['files_sha256'].items() if n.startswith('tiny_target/')}
    if launch['package_sha256']!=package:raise ValueError('Runtime differs from freeze')
    for name,h in package.items():
        if sha256(path/'implementation'/name)!=h:raise ValueError('Snapshot changed')
    conformance=launch['exact_cuda_stabilization']['conformance']
    if not conformance['exact'] or conformance['warp_cases']!=32:raise ValueError('Cubic startup gate missing')
    if config_name=='resident_config.json' and conformance['gaussian_cases']!=33:raise ValueError('Gaussian startup gate missing')
    counts=dict(actual_pva_pairs=0,pva_failures=0,motion_resets=0,ready_frames=0)
    previous=-1
    with (path/'frames.jsonl').open() as journal:
        for row in map(json.loads,journal):
            if row['frame_index']!=previous+1:raise ValueError('Journal not contiguous')
            previous=row['frame_index'];motion=row['motion']
            if row['timestamp_ns']!=previous*100_000_000 or row['coverage']['full_shape_hw']!=frozen['native_shape_hw'] or row['coverage']['configured_crop'] is not None or row['coverage']['native_pixel_sampling'] is not True:
                raise ValueError('Frame cadence or native full-frame coverage changed')
            counts['pva_failures']+=bool(motion['pva_failure']);counts['motion_resets']+=bool(motion['reset'])
            counts['ready_frames']+=not row['coverage']['warmup']
            if previous and not motion['pva_failure']:
                b=motion['motion_backends']
                if b.get('cpu_fallback') is not False or any(b.get(k)!='PVA' for k in ('gaussian_pyramid','harris','optical_flow_pyrlk')):
                    raise ValueError('Actual PVA provenance missing')
                counts['actual_pva_pairs']+=1
    if previous+1!=expected_frames:raise ValueError('Short journal')
    return launch,report,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    frozen=json.loads((a.root/'freeze.json').read_text())
    if set(frozen['sources'])!={'0126','0029','0055','0082'}:raise ValueError('Unexpected validation cohort')
    status=json.loads((a.root/'status.json').read_text())
    if status['running'] or status['error'] or len(status['completed'])!=6:raise ValueError('Batch not successfully complete')
    if [r['name'] for r in status['completed']]!=['host_prefix','resident_prefix','full_0126','full_0029','full_0055','full_0082']:
        raise ValueError('Completed job scope/order changed')
    for name in ('host_config.json','resident_config.json'):
        if sha256(a.root/name)!=frozen['files_sha256'][name]:raise ValueError('Local configuration changed')
    lib_hash=sha256(a.root/'libseaqr_integrated.so')
    if lib_hash!=status['library_sha256']:raise ValueError('Library differs from execution')
    reference=ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914'
    packet=json.loads((reference/'scoring_packet.json').read_text())
    pilot=ROOT/'results/tiny_target/phase20/accuracy_baseline_v1_20260914'
    pf=json.loads((pilot/'scoring_freeze.json').read_text())
    for name,key in [('annotations.json','labels_sha256'),('source_review/packet.json','packet_sha256')]:
        if sha256(pilot/name)!=pf[key]:raise ValueError('Pilot changed')
    comparisons=[]
    for name,config in [('host_prefix','host_config.json'),('resident_prefix','resident_config.json')]:
        launch,report,counts=verify_run(a.root/name,frozen,config,230,False)
        if launch['source_sha256']!=frozen['sources']['0126']['sha256'] or launch['exact_cuda_stabilization']['library_sha256']!=lib_hash:
            raise ValueError('Prefix source/library changed')
        comparison=json.loads((a.root/name/'comparison.json').read_text())
        if not comparison['exact'] or comparison['output_sha256']!=sha256(a.root/name/'frames.jsonl'):
            raise ValueError('Prefix comparison missing or changed')
        if comparison['reference_sha256']!=frozen['reference_artifacts_sha256']['frames.jsonl']:
            raise ValueError('Prefix reference changed')
        comparisons.append(dict(run=name,**comparison))
    runs=[];gate=True
    for cid in ('0126','0029','0055','0082'):
        path=a.root/('full_'+cid);source=frozen['sources'][cid]
        launch,report,counts=verify_run(path,frozen,'resident_config.json',source['frames'],True)
        if launch['source_sha256']!=source['sha256'] or report['source_sha256']!=source['sha256']:
            raise ValueError('Source changed')
        if launch['exact_cuda_stabilization']['library_sha256']!=lib_hash:raise ValueError('Wrong runtime library')
        scored=evaluate(path,cid,reference)
        with (path/'frames.jsonl').open() as f:
            pilot_score=score_rows(map(json.loads,f),json.loads((pilot/'annotations.json').read_text()),
                                  json.loads((pilot/'source_review/packet.json').read_text()),cid,launch['fps'])
        windows=scored['evaluation']['positive_windows']
        if sum(w['visible_samples'] for w in windows)!={'0126':139,'0029':146,'0055':0,'0082':0}[cid]:
            raise ValueError('Dense visible-reference denominator changed')
        for w in windows:
            permitted={216} if cid=='0126' else set()
            gate &= set(w['missed_visible_frames'])<=permitted and w['ambiguity_frames']==0 and len(w['observed_track_ids'])==1
        gate &= all(w['qualified_measured_hits']==w['visible_samples'] and w['ambiguity_frames']==0 for w in pilot_score['positive_windows'])
        light=next((w for w in packet['windows'] if w['clip_id']==cid and w['id'].endswith('_lights')),None)
        reviewed=None if light is None else dict(first=light['first'],last=light['last'],fixed_crop=light['crop_xywh'])
        diagnostic=inspect_run(path,windows,reviewed)
        historical=None
        if light:
            prior=ROOT/'results/tiny_target/phase20/appearance_v7_20260914'/('cpu_'+cid)
            historical=inspect_run(prior,[],reviewed)
        runs.append(dict(clip_id=cid,frames=report['frames'],fps=report['processed_fps'],elapsed_seconds=report['elapsed_seconds'],
            timings_ms=report['timings_ms'],availability=report['availability'],execution_counts=counts,
            coverage_loss_counts=report['counts'],qualified_proposal_workload=report['qualified_track_count'],
            scored=scored,pilot=pilot_score,diagnostic=diagnostic,previous_cpu_motion_light_workload=historical,
            historical_comparison_caveat='CPU translation vs PVA changes camera-motion estimation; this is not an isolated speed comparison.' if light else None))
    repeated=[]
    with (a.root/'resident_prefix/frames.jsonl').open() as first,(a.root/'full_0126/frames.jsonl').open() as second:
        for i,(left,right) in enumerate(zip(first,second)):
            left,right=json.loads(left),json.loads(right)
            for key in ('frame_index','timestamp_ns','segment','source_to_reference','candidates','tracks','tracking_metrics'):
                if left[key]!=right[key]:repeated.append([i,key])
            lc=dict(left['coverage']);rc=dict(right['coverage']);lc.pop('detection_ms');rc.pop('detection_ms')
            if lc!=rc:repeated.append([i,'coverage'])
    execution_healthy=all(r['execution_counts']['pva_failures']==0 and r['execution_counts']['motion_resets']==0 and r['execution_counts']['ready_frames']==r['frames']-8 for r in runs)
    result=dict(schema='seaqr.exact-cuda-full-development.v1',completed=True,prefix_comparisons=comparisons,runs=runs,
        full_clip_frames=sum(r['frames'] for r in runs),known_reference_nonregression_gate=bool(gate),
        execution_healthy=execution_healthy,repeat_prefix_exact=not repeated,repeat_prefix_differences=repeated[:20],
        reviewed_accuracy_improvement_claim=False,generalization_proven=False,production_ready=False,
        airborne_precision=None,airborne_recall=None,full_frame_false_alarm_rate=None,
        frozen_accuracy_policy=True,holdouts_accessed=False,labels_used_only_after_processing=True,
        limitation='Four previously examined development clips, no verified airborne/empty cohort; count proposals as unlabeled workload.',
        analyzer_sha256=sha256(__file__),freeze_sha256=sha256(a.root/'freeze.json'),library_sha256=lib_hash)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(full_clip_frames=result['full_clip_frames'],known_reference_nonregression_gate=bool(gate),
          runs=[dict(clip_id=r['clip_id'],fps=r['fps'],workload=r['qualified_proposal_workload'],
              matches=[dict(window=w['window_id'],hits=w['qualified_measured_hits'],visible=w['visible_samples'],misses=w['missed_visible_frames'])
                       for w in r['scored']['evaluation']['positive_windows']],
              light_pairs=(r['diagnostic']['light_field_review'] or {}).get('measured_qualified_response_frames')) for r in runs]),indent=2))

if __name__=='__main__':main()
