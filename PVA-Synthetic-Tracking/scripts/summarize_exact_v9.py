"""Report-only verification of complete exact-v9 evidence. No media access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from batch_exact_v9 import schedule
from run_exact_v9 import read
from exact_v9_common import compare_audits,FFT_WORKERS
from profile_raw16_efficiency import sha,write_json,compact
from summarize_raw16_cpu_v6 import digest,difference,compare_source_motion
from run_raw16_background_v7 import normalized_report
from validate_raw16_full_frame import assess

V8_SUMMARY_SHA='4222f1fe6b1e6a2670ef89a1527025254cb9002c5bbbc86b0d2f652fc8c63251'

def validated_schedule(root,stage):
    expected=schedule(stage);plan=read(root/(stage+'_plan.json'))
    journal=[json.loads(line) for line in (root/(stage+'_journal.jsonl')).read_text().splitlines()]
    if plan['schedule']!=expected or len(journal)!=len(expected):raise ValueError('Incomplete or changed schedule; no partial summary')
    for row,event in zip(expected,journal):
        if event['returncode']!=0 or any(event.get(k)!=v for k,v in row.items()):raise ValueError('Failed or reordered trial')
    return expected,plan

def decisions(left,right):
    result=compare_source_motion(left,right)
    for filename in ('candidate_decisions.json','global_fit_identities.json'):
        a,b=read(left/filename),read(right/filename)
        if len(a)!=(6 if filename.startswith('candidate') else 63):raise ValueError('Incomplete identity records')
        result[filename]=dict(exact=a==b,first_difference=difference(a,b))
    result['exact_semantics']=normalized_report(read(left/'report.json'))==normalized_report(read(right/'report.json'))
    result['passed']=(result['source_frames_exact'] and result['motion_points_exact'] and result['exact_semantics']
        and result['candidate_decisions.json']['exact'] and result['global_fit_identities.json']['exact'])
    return result

def summarize(args):
    if sha(args.v8_summary)!=V8_SUMMARY_SHA:raise ValueError('v8 baseline summary changed')
    old=read(args.v8_summary);component=read(args.component)
    if (component['passed'] is not True or component['quick'] or not component['native'] or component['real_media_read']
        or not component['fft_exact'] or not component['warp_exact'] or not component['moving_exact']
        or len(component['fft'])!=40 or len(component['warp'])!=350 or len(component['phases'])!=1024
        or len(component['moving'])!=18 or len(component['thresholds'])!=33):raise ValueError('Generated gate incomplete')
    if not all(set(r['workers'])=={'1','2','4'} for r in component['fft']):raise ValueError('Missing FFT worker variant')
    if not all(v['exact'] and v['raw_exact'] and v['threshold4_changes']==0 for r in component['fft'] for v in r['workers'].values()):
        raise ValueError('FFT byte gate failed')
    if not all(r['image']['exact'] and r['mask']['exact'] for r in component['warp']+component['phases']):raise ValueError('Warp byte gate failed')
    if not all(r['exact'] for r in component['thresholds']):raise ValueError('Threshold gate failed')
    if not all(r['candidate_decisions_exact'] and r['maximum_response_error']==0 for r in component['moving']):raise ValueError('Moving sequence changed')
    motion_dir=args.results.parent/'motion_generated';gate=read(motion_dir/'gate.json')
    if (gate['passed'] is not True or gate['package_sha256']!=component['package_sha256'] or len(gate['rows'])!=3
        or {r['seed'] for r in gate['rows']}!={75316,129827,85723}):raise ValueError('PVA gate mismatch')
    for row in gate['rows']:
        path=motion_dir/f'controls_{row["seed"]}.json';control=read(path)
        if sha(path)!=row['sha256'] or not row['exact'] or row['returncode'] or not control['passed'] or len(control['cases'])!=16:
            raise ValueError('PVA evidence changed')
    trials={};provenance=set()
    for stage in ('checks','timing'):
        rows,plan=validated_schedule(args.results,stage)
        for row in rows:
            path=args.results/row['name'];report=read(path/'report.json');checks=read(path/'checks.json');exp=read(path/'experiment.json')
            if assess(report)!=checks['checks'] or not checks['checks']['processing_integrity_passed'] or not checks['checks']['detection_availability_passed']:
                raise ValueError('Integrity/availability changed')
            if (exp['error'] is not None or exp['wrapper_sha256']!=plan['wrapper_sha256']
                or exp['adapter_sha256']!=plan['adapter_sha256'] or exp['mode']!=row['mode']
                or exp['cache']!={'hits':62,'misses':64}):raise ValueError('Runtime/adapter/cache evidence changed')
            p=exp['provenance'];provenance.add(digest(p))
            if (p['component_sha256']!=sha(args.component) or p['package_sha256']!=component['package_sha256']
                or p['warp_library_sha256']!=component['library_sha256'] or p['pva_gate_sha256']!=sha(motion_dir/'gate.json')):
                raise ValueError('Mixed generated/runtime provenance')
            if not read(path/'comparison.json')['exact_gate_passed']:raise ValueError('Native gate failed')
            ex=exp['execution']
            if ex['experimental_v8_gpu_point_filter_used'] or ex['fft_workers']!=FFT_WORKERS:raise ValueError('Unexpected filter execution')
            if row['mode']=='exact':
                if ex['filter_calls']!=63 or ex['warp_calls']!=63:raise ValueError('Incomplete exact execution')
                if row.get('audit') and (len(ex['filter_bit_checks'])!=63 or len(ex['warp_bit_checks'])!=63 or not all(ex['filter_bit_checks']+ex['warp_bit_checks'])):
                    raise ValueError('Missing native byte comparisons')
            elif ex['filter_calls'] or ex['warp_calls']:raise ValueError('Reference arm used experimental execution')
            trials[row['name']]=dict(mode=row['mode'],wall_s=checks['elapsed_wall_s'],fps=64/checks['elapsed_wall_s'],
                report_sha256=sha(path/'report.json'),experiment_sha256=sha(path/'experiment.json'),
                peak_process_rss_kib=checks['peak_process_rss_kib'],synthetic_controls_passed=checks['checks'].get('synthetic_controls_passed'))
    if len(provenance)!=1:raise ValueError('Mixed trial provenance')
    audits={};comparisons={};timings={}
    for clip in ('0040','0029'):
        previous=args.v8_results/f'audit_{clip}_cpu';current=args.results/f'audit_{clip}_exact'
        if sha(previous/'report.json')!=old['trials'][f'audit_{clip}_cpu']['report_sha256']:raise ValueError('v8 report changed')
        audits[clip]=compare_audits(previous.with_suffix('.audit.jsonl'),current.with_suffix('.audit.jsonl'))
        if audits[clip]['raw_left_sha256']!=old['exact_cpu_audits'][clip]['raw_right_sha256'] or not audits[clip]['passed']:
            raise ValueError('Intermediate audit failed')
        comparisons[f'v8_{clip}']=decisions(previous,current)
        if not comparisons[f'v8_{clip}']['passed']:raise ValueError('v8 decisions changed')
        timings[clip]={}
        for mode in ('reference','exact'):
            samples=[]
            for repeat in (1,2):
                name=f'timed_{clip}_{repeat}_{mode}';path=args.results/name
                comparisons[name]=decisions(current,path)
                if not comparisons[name]['passed']:raise ValueError('Timed output differs from audited output')
                samples.append(trials[name]['wall_s'])
            median=statistics.median(samples);timings[clip][mode]=dict(samples_s=samples,median_s=median,fps=64/median)
        a,b=timings[clip]['reference']['median_s'],timings[clip]['exact']['median_s']
        timings[clip]['speedup']=a/b;timings[clip]['wall_reduction_fraction']=1-b/a
    previous=args.v8_results/'trace_0040_reference';current=args.results/'injected_0040_exact'
    if sha(previous/'report.json')!=old['trials']['trace_0040_reference']['report_sha256']:raise ValueError('v8 injected report changed')
    comparisons['injected']=decisions(previous,current)
    if not comparisons['injected']['passed']:raise ValueError('Injected output changed')
    injection=read(current/'report.json')['injection']['synthetic_track_pool_evaluation']
    profiles={mode:read(args.results/f'profile_0040_{mode}/stage_profile.json') for mode in ('reference','exact')}
    for mode,profile in profiles.items():
        if profile['error'] is not None or profile['timing']['accounting_error_ns']!=0:raise ValueError('Profile accounting failed')
        if not decisions(args.results/'audit_0040_exact',args.results/f'profile_0040_{mode}')['passed']:raise ValueError('Profile changed output')
    return dict(schema_version='seaqr.raw16-exact-v9-summary.v1',trials=trials,timings=timings,profiles=profiles,
        comparisons=comparisons,audits=audits,injected_controls=injection,
        generated=dict(component_sha256=sha(args.component),fft_fixtures=40,fft_worker_variants=3,
            warp_fixtures=350,warp_phases=1024,moving_sequences=18,threshold_cases=33,pva_cases=48),
        gates=dict(exact_development_passed=True,full_candidate_order_and_scores_preserved=True,
            default_changed=False,production_approved=False,real_airborne_accuracy_validated=False,
            all_injected_controls_recovered=not injection['missed_target_ids'],
            rejected_v8_direct_gpu_filter_promoted=False),
        warning='Native prefix timings, 64 frames each, two reversed rounds. Filter uses four CPU FFT workers, not direct GPU convolution. '
            'Warp is limited to the pinned Jetson build and exact translations. All scores/decisions match tested references; '
            'this is not sustained real-time qualification or a real-airborne recall/FAR result. No holdout accessed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','v8-results','v8-summary','component','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();summary=summarize(args);write_json(args.output,summary)
    print(json.dumps(dict(gates=summary['gates'],timings=summary['timings']),indent=2))
