"""Report-only verification of complete v8 evidence; never opens media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from batch_raw16_speed_v8 import schedule
from run_raw16_speed_v8 import read
from run_raw16_background_v7 import normalized_report
from compare_raw16_v8_audits import compare_audits
from summarize_raw16_cpu_v6 import compare_source_motion, difference, digest
from profile_raw16_efficiency import sha, write_json
from validate_raw16_full_frame import assess

SCORE_FIELDS=frozenset({'normalized_score_snr','raw_sum_score','ranking_score','local_clutter_center_snr',
    'local_clutter_scale_snr','neighbor_max_score_snr','contrast_snr','peak_to_neighbor_ratio',
    'maximum_raw_score_snr','maximum_selection_score','median_raw_score_snr','median_selection_score'})


def without_scores(value):
    if isinstance(value,dict):return {k:without_scores(v) for k,v in value.items() if k not in SCORE_FIELDS}
    if isinstance(value,list):return [without_scores(v) for v in value]
    return value


def compare_decisions(left,right):
    a,b=read(left/'candidate_decisions.json'),read(right/'candidate_decisions.json')
    if len(a)!=6 or len(b)!=6: raise ValueError('Incomplete candidate decisions')
    ra,rb=[read(p/'report.json') for p in (left,right)]
    ta,tb=[r['screening']['synthetic_tracking'] for r in (ra,rb)]
    result=dict(all_candidate_identities_exact=a==b,
                candidate_first_difference=difference(a,b),
                total_candidates_left=sum(len(r['decisions']) for r in a),
                total_candidates_right=sum(len(r['decisions']) for r in b),
                global_fit_identities_exact=read(left/'global_fit_identities.json')==read(right/'global_fit_identities.json'),
                availability_exact=ra['screening']['availability']==rb['screening']['availability'],
                **compare_source_motion(left,right))
    for name in ('track_pool','shortlist'):
        x,y=without_scores(ta[name]),without_scores(tb[name])
        result[name+'_decisions_exact']=x==y
        result[name+'_first_difference']=difference(x,y)
        result[name+'_count_left']=len(x);result[name+'_count_right']=len(y)
    # Set comparisons expose a mere ordering/index change versus a changed candidate.
    def physical(rows):
        return [set((tuple(c['xy']),tuple(c['velocity']),c['support']) for c in r['decisions']) for r in rows]
    sa,sb=physical(a),physical(b)
    result['per_window_physical_candidate_changes']=[dict(left_only=len(x-y),right_only=len(y-x)) for x,y in zip(sa,sb)]
    result['decision_gate_passed']=all(result[k] for k in ('all_candidate_identities_exact','global_fit_identities_exact',
        'availability_exact','source_frames_exact','motion_points_exact','track_pool_decisions_exact','shortlist_decisions_exact'))
    return result


def validated_schedule(root,stage):
    expected=schedule(stage)
    plan=read(root/(stage+'_plan.json'))
    if plan['schedule']!=expected:raise ValueError('Predeclared schedule changed')
    rows=[json.loads(line) for line in (root/(stage+'_journal.jsonl')).read_text().splitlines()]
    if len(rows)!=len(expected):raise ValueError('Incomplete schedule; partial summaries forbidden')
    for a,b in zip(expected,rows):
        if any(b.get(k)!=v for k,v in a.items()) or b['returncode']!=0:
            raise ValueError('Trial order/exit mismatch')
    return expected,plan


def control_diagnosis(root):
    trace=read(root/'trace_0040_reference/control_trace.json')
    report=read(root/'trace_0040_reference/report.json')
    targets={t['target_id']:t for t in report['injection']['specification']['targets']}
    result={}
    for target in targets:
        frames=[r for r in trace['frames'] if r['target_id']==target]
        windows=[r for r in trace['windows'] if r['target_id']==target]
        if ([r['frame_index'] for r in frames]!=list(range(8,64))
                or [r['frames'] for r in windows]!=[list(range(start,start+16)) for start in range(4,45,8)]):
            raise ValueError('Incomplete control trace')
        result[target]=dict(active_frames=len(frames),
            center_filter_valid_frames=[r['frame_index'] for r in frames if r['center_filter_valid']],
            center_above_saturation_cutoff_before_injection=[r['frame_index'] for r in frames
                if r['original_center_dn']>=r['saturation_limit_dn']],
            patch_fully_saturated_before_injection=[r['frame_index'] for r in frames if r['saturated_patch_before']==49],
            frames_with_any_saturated_patch_pixel_before_injection=sum(r['saturated_patch_before']>0 for r in frames),
            frames_with_warp_invalid_patch=sum(r['warp_valid_patch']<49 for r in frames),
            original_center_dn_min=min(r['original_center_dn'] for r in frames),
            original_center_dn_max=max(r['original_center_dn'] for r in frames),
            true_velocity_windows=windows,
            frozen_detector_truth_probes=report['injection']['synthetic_tracking_truth_probes'][target])
    return dict(targets=result,
        track_pool_evaluation=report['injection']['synthetic_track_pool_evaluation'],
        interpretation='Read original/injected values, support, and early observable-window behavior separately. '
            'A later saturated trajectory does not by itself explain every earlier miss. '
            'Diagnostic true-velocity ROI reads unchanged responses and does not steer actual detection.')


def summarize(args):
    schedules={stage:validated_schedule(args.results,stage) for stage in ('checks','timing')}
    component=read(args.component)
    if (component['passed'] is not True or component['real_media_read'] is not False
            or component['native_enabled'] is not True or len(component['motion'])!=64
            or not all(r['exact'] for r in component['motion'])
            or len(component['filter']['cases'])!=30 or not component['filter']['numerical_screen_passed']
            or len(component['moving'])!=18 or len(component['stabilization'])!=18):
        raise ValueError('Incomplete or failed generated evidence')
    trials={};identities=set()
    for stage,(rows,plan) in schedules.items():
        for row in rows:
            path=args.results/row['name']
            report=read(path/'report.json');saved=read(path/'checks.json')
            if assess(report)!=saved['checks']:raise ValueError('Saved integrity assessment differs')
            if not saved['checks']['processing_integrity_passed'] or not saved['checks']['detection_availability_passed']:
                raise ValueError('Processing/search support failed')
            exp=read(path/'experiment.json')
            if exp['error'] is not None or exp['mode']!=row['mode'] or exp['wrapper_sha256']!=plan['wrapper_sha256']:
                raise ValueError('Trial metadata mismatch')
            if (exp['provenance']['component_sha256']!=sha(args.component)
                    or exp['provenance']['package_sha256']!=component['package_sha256']
                    or exp['provenance']['point_library_sha256']!=component['point_library_sha256']):
                raise ValueError('Mixed generated/runtime evidence')
            if not read(path/'comparison.json')['diagnostic_run_passed']:
                raise ValueError('Failed diagnostic trial')
            identities.add(digest(exp['provenance']))
            if len(read(path/'global_fit_identities.json'))!=63:raise ValueError('Incomplete fits')
            if exp['cache'] is not None and exp['cache']!={'hits':62,'misses':64}:
                raise ValueError('Unexpected sequential cache use')
            trials[row['name']]=dict(wall_s=saved['elapsed_wall_s'],fps=64/saved['elapsed_wall_s'],
                source_frames=sha(path/'source_frames.json'),report_sha256=sha(path/'report.json'),
                experiment_sha256=sha(path/'experiment.json'),mode=row['mode'],
                diagnostic_pass=read(path/'comparison.json')['diagnostic_run_passed'])
    if len(identities)!=1:raise ValueError('Mixed runtime/provenance')
    audits={};decisions={};repeat_parity={};timings={}
    for clip in ('0040','0029'):
        cpu=args.results/f'audit_{clip}_cpu';gpu=args.results/f'shadow_{clip}_combined'
        audits[clip]=compare_audits(args.v7_results/f'audit_{clip}_gpu.audit.jsonl',cpu.with_suffix('.audit.jsonl'))
        if not audits[clip]['passed']:raise ValueError('Exact CPU full-array audit failed')
        comparisons=read(gpu/'experiment.json')['filter_comparisons']
        if len(comparisons)!=63 or not all(r['numerical_screen_passed'] for r in comparisons):raise ValueError('Numeric screen failed')
        decisions[clip]=compare_decisions(cpu,gpu)
        decisions[clip]['maximum_response_error']=max(r['max_abs_error'] for r in comparisons)
        decisions[clip]['response_threshold4_changes']=sum(r['threshold4_changes'] for r in comparisons)
        timings[clip]={}
        for mode in ('reference','cpu','combined'):
            attempts=[args.results/f'timed_{clip}_{n}_{mode}' for n in (1,2)]
            anchor=gpu if mode=='combined' else cpu
            parity=[normalized_report(read(p/'report.json'))==normalized_report(read(anchor/'report.json')) for p in attempts]
            repeat_parity[f'{clip}_{mode}']=parity
            if not all(parity):raise ValueError('Timed output differs from corresponding audited/shadow arm')
            for p in attempts:
                if not compare_decisions(anchor,p)['decision_gate_passed']:raise ValueError('Timed stage decisions changed')
            values=[read(p/'checks.json')['elapsed_wall_s'] for p in attempts]
            median=statistics.median(values)
            timings[clip][mode]=dict(attempts_s=values,median_s=median,fps=64/median)
        r,c,g=[timings[clip][m]['median_s'] for m in ('reference','cpu','combined')]
        timings[clip]['speedup']=dict(exact_cpu_vs_v7=r/c,experimental_combined_vs_v7=r/g,experimental_filter_vs_cpu=c/g,
                                     combined_wall_reduction_fraction=1-g/r)
    decisions['injected_0040']=compare_decisions(args.results/'trace_0040_reference',args.results/'trace_0040_combined')
    profiles={mode:read(args.results/f'profile_0040_{mode}/stage_profile.json') for mode in ('cpu','combined')}
    for mode,value in profiles.items():
        if value['error'] is not None or value['timing']['accounting_error_ns']!=0:raise ValueError('Profile accounting failed')
    generated_gate=read(args.results.parent/'motion_generated/gate.json')
    if (not generated_gate['passed'] or len(generated_gate['rows'])!=3
            or generated_gate['package_sha256']!=component['package_sha256']):
        raise ValueError('Incomplete generated PVA evidence')
    for row in generated_gate['rows']:
        path=args.results.parent/'motion_generated'/f'controls_{row["seed"]}.json'
        controls=read(path)
        if (sha(path)!=row['sha256'] or not row['exact'] or row['returncode']!=0
                or not controls['passed'] or len(controls['cases'])!=16
                or not all(r['passed'] for r in controls['cases'])):
            raise ValueError('Failed or modified PVA evidence')
    return dict(schema_version='seaqr.raw16-speed-v8-summary.v1',trials=trials,exact_cpu_audits=audits,
        experimental_decision_comparisons=decisions,timing_repeat_parity=repeat_parity,timings=timings,
        profiles=profiles,control_diagnosis=control_diagnosis(args.results),
        generated=dict(component_sha256=sha(args.component),motion_cases=len(component['motion']),
            pva_generated_cases=48,pva_gate_sha256=sha(args.results.parent/'motion_generated/gate.json'),
            filter_cases=len(component['filter']['cases']),moving_cases=len(component['moving']),
            strict_filter_gate=component['strict_gpu_filter_gate_passed'],
            generated_threshold4_changes=sum(r['threshold4_changes'] for r in component['filter']['cases']),
            stabilized_affine_exact=all(r['affine_image_exact'] and r['affine_mask_exact'] for r in component['stabilization']),
            stabilized_gpu_exact=all(r['gpu_image_exact'] for r in component['stabilization'])),
        score_fields_excluded_only_from_decision_identity=sorted(SCORE_FIELDS),
        gates=dict(exact_cpu_motion_development_passed=True,
            experimental_filter_decisions_passed=all(r['decision_gate_passed'] for r in decisions.values()),
            strict_gpu_point_filter_bit_exact=component['strict_gpu_filter_gate_passed'],stabilization_replacement_accepted=False,
            real_airborne_accuracy_validated=False,default_changed=False,production_approved=False),
        warning='Whole-pipeline bounded instrumented timing on reused unlabeled development prefixes. '
                'GPU filtering is explicitly non-bit-exact; numerical screening is not production approval. '
                'Rejected stabilization alternatives were never enabled on media. No holdout was accessed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','v7-results','component','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=summarize(args);write_json(args.output,result)
    print(json.dumps(dict(gates=result['gates'],timings=result['timings']),indent=2))
