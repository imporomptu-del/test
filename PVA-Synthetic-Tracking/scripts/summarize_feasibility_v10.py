"""Fail-closed, report-only feasibility summary; never reads camera media."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
V9_SHA='f7c24ff923afcbe0a9c5b2cbbbb33dce689c566743be789dc2eee2e522f438f4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def stats(values):
    require(bool(values),'Empty timing series')
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0 for x in values),'Invalid timing')
    return dict(n=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))


def exact_keys(rows,fields,expected):
    actual=[tuple(r[k] for k in fields) for r in rows]
    require(len(actual)==len(expected) and len(set(actual))==len(actual)
        and set(actual)==set(expected),'Incomplete or duplicate quality coverage')


def validate_timing(record,modes,ends):
    expected=[dict(scene=scene,repeat=repeat,mode=mode)
        for scene in ('dense','holes') for repeat in range(4)
        for mode in (modes if repeat%2==0 else modes[::-1])]
    require(record['passed'] is True and record['pipeline_benchmark'] is False,'Timing report did not pass as component-only')
    require(record['schedule']==expected,'Changed timing schedule')
    require([{k:t[k] for k in ('scene','repeat','mode')} for t in record['trials']]==expected,'Incomplete/reordered timing trials')
    require(record['shape']==[3190,4784] and record['window_frames']==16
        and record['stride_frames']==8 and record['velocity_count']==48,'Changed benchmark geometry/search')
    for t in record['trials']:
        require(len(t['samples'])==3,'Incomplete timing cycles')
        stats([t['initialization_and_warmup_s']])
        for index,s in enumerate(t['samples']):
            require(s['outputs_exact'] is True and s['advance_frames']==8,'Failed timed output verification')
            stats([s['host_s'],s['kernel_ms'],s['append_host_s'],s['download_host_s']])
            require(s['host_s']>0 and s['kernel_ms']>0,'Zero service time')
            if ends is not None:
                require(s['last_frame']==ends[index],'Unexpected timed window')
                require([f['index'] for f in s['frames']]==list(range(ends[index]-7,ends[index]+1)),'Missing RAW frames')
                stats([f['host_s'] for f in s['frames']]);stats([f['frontend_compute_ms'] for f in s['frames']])
            else:require(s['cycle']==index,'Unexpected tracking cycle')
    groups={}
    for scene,mode in itertools.product(('dense','holes'),modes):
        samples=[s for t in record['trials'] if (t['scene'],t['mode'])==(scene,mode) for s in t['samples']]
        group={k:stats([s[k] for s in samples]) for k in ('host_s','kernel_ms','append_host_s','download_host_s')}
        group['remaining_800ms_budget_s']=.8-group['host_s']['median']
        group['all_measured_cycles_within_800ms']=all(s['host_s']<=.8 for s in samples)
        if ends is not None:
            group['frame_host_s']=stats([f['host_s'] for s in samples for f in s['frames']])
            group['frame_frontend_compute_ms']=stats([f['frontend_compute_ms'] for s in samples for f in s['frames']])
            group['tracking_host_s']=stats([s['tracking_host_s'] for s in samples])
        groups[scene+'/'+mode]=group
    return groups


def summarize(evidence,reference):
    hashes={}
    def read(relative):
        p=evidence/relative;hashes[relative]=sha(p)
        return json.loads(p.read_text())
    from snapshot_feasibility_v10 import source_hashes
    provenance=read('results/final_provenance_02.json')
    require(provenance['real_media_read'] is False and provenance['source_sha256']==source_hashes(),
        'Local and Jetson experiment sources differ')
    plan=sha(ROOT/'docs/raw16_feasibility_v10_plan.md')
    for directory,binary,sourcekey,builder in (
        ('resident_v10_01','libresident_tracking_v10.so','sources_sha256','build_resident_v10.py'),
        ('raw_probe_v10_01','libraw_probe_v10.so','source_sha256','build_raw_probe_v10.py')):
        build=read(f'build/{directory}/build.json')
        require(build['library_sha256']==sha(evidence/f'build/{directory}/{binary}'),'Build binary hash changed')
        require(build['builder_sha256']==sha(ROOT/'scripts'/builder),'Builder changed')
        require(all(sha(ROOT/p)==h for p,h in build[sourcekey].items()),'CUDA source changed')
    qualities=[]
    for file,library in (('ring_quality_01.json','resident_v10_01/libresident_tracking_v10.so'),
                         ('combined_ring_quality_01.json','raw_probe_v10_01/libraw_probe_v10.so')):
        q=read('results/'+file);qualities.append(q)
        require(q['passed'] is True and q['real_media_read'] is False,'Ring gate failed')
        require(q['library_sha256']==sha(evidence/'build'/library),'Ring gate library mismatch')
        require(q['checker_sha256']==sha(ROOT/'scripts/check_resident_v10.py')
            and q['wrapper_sha256']==sha(ROOT/'scripts/resident_tracking_v10.py') and q['plan_sha256']==plan,'Ring provenance mismatch')
        exact_keys(q['rows'],('scene','flux','polarity','last_frame'),list(itertools.product(
            ('straight','turn','acceleration','short_visibility','holes','clutter'),(0,2,4,8,16),('bright','dark'),(15,23,31,39))))
        require(all(r['arrays_exact'] is True and r['ordered_decisions_exact'] is True for r in q['rows']),'Ring decision mismatch')
        require(len(q['guards'])==6 and all(g['passed'] is True for g in q['guards']),'Ring guard failure')
        actual=[(tuple(g['shape']),g['last_frame']) for g in q['geometry']]
        expected=list(itertools.product(((1,1),(3,7),(17,65),(193,257)),(15,16,23,31,32)))
        require(len(actual)==20 and set(actual)==set(expected) and all(g['exact'] is True for g in q['geometry']),'Geometry checks incomplete')
        require(q['positive_control_observed'] is True,'Positive control absent')
    q=read('results/raw_quality_01.json')
    require(q['implementation_integrity_passed'] is True and q['real_media_read'] is False,'RAW integrity failed')
    require(q['library_sha256']==qualities[1]['library_sha256']
        and q['ring_gate_sha256']==hashes['results/combined_ring_quality_01.json'],'RAW composition provenance mismatch')
    require(q['script_sha256']==sha(ROOT/'scripts/check_raw_probe_v10.py')
        and q['wrapper_sha256']==sha(ROOT/'scripts/resident_frontend_v10.py') and q['plan_sha256']==plan,'RAW harness changed')
    exact_keys(q['frames'],('scene','flux','index'),list(itertools.product(('dense','holes'),(0,32,64,128,256,512),range(1,44))))
    exact_keys(q['windows'],('scene','flux','last_frame'),list(itertools.product(('dense','holes'),(0,32,64,128,256,512),(19,27,35,43))))
    exact_keys(q['native'],('scene','index'),list(itertools.product(('dense','holes'),range(1,8))))
    require(all(r['prepoint_exact'] is True and r['numerical_screen_passed'] is True for r in q['frames']+q['native'])
        and all(r['handoff_exact'] is True for r in q['windows']),'RAW per-frame composition failed')
    strict=all(r['bit_exact'] for r in q['frames']+q['native'])
    losses=sum(r['near_truth']['reference'] and not r['near_truth']['direct'] for r in q['windows'] if r['flux']>0)
    gains=sum(r['near_truth']['direct'] and not r['near_truth']['reference'] for r in q['windows'] if r['flux']>0)
    positive=any(r['near_truth']['reference'] for r in q['windows'] if r['scene']=='dense' and r['flux']==512)
    require(q['strict_cpu_fft_equivalent']==strict and q['generated_target_losses']==losses
        and q['generated_target_gains']==gains and q['positive_control_observed']==positive
        and q['generated_sensitivity_gate_passed']==(positive and losses==0),'Incorrect sensitivity summary')
    guard=read('results/raw_guards_01.json')
    require(guard['passed'] is True and len(guard['checks'])==6 and all(guard['checks'].values())
        and guard['real_media_read'] is False and guard['library_sha256']==q['library_sha256']
        and guard['script_sha256']==sha(ROOT/'scripts/check_raw_probe_guards_v10.py'),'RAW state guards failed')
    a=read('results/ring_timing_01.json');b=read('results/raw_timing_01.json')
    for record,file,checker,library in (
        (a,'results/ring_quality_01.json','benchmark_resident_v10.py',qualities[0]['library_sha256']),
        (b,'results/raw_quality_01.json','benchmark_raw_probe_v10.py',q['library_sha256'])):
        require(record['gate_sha256']==hashes[file] and record['script_sha256']==sha(ROOT/'scripts'/checker)
            and record['library_sha256']==library and record['plan_sha256']==plan
            and record['real_media_read'] is False,'Timing provenance mismatch')
    require(b['strict_cpu_fft_equivalent']==strict and b['generated_sensitivity_gate_passed']==(positive and losses==0),'Timing concealed quality gate')
    stage_a=validate_timing(a,('stateless','resident_host','resident_device'),None)
    stage_b=validate_timing(b,('resident_device','resident_host'),(27,35,43))
    require(sha(reference/'summary.json')==V9_SHA,'Frozen v9 summary mismatch')
    v9=json.loads((reference/'summary.json').read_text());baseline={}
    for clip in ('0029','0040'):
        p=reference/f'source_frames_{clip}.json';frames=json.loads(p.read_text())
        require(len(frames)==64 and [f['frame_index'] for f in frames]==list(range(64)),'Incomplete archived cadence evidence')
        stamps=[f['source_timestamp_ns'] for f in frames]
        require(all(x<y for x,y in zip(stamps,stamps[1:])),'Nonmonotonic archived timestamps')
        span=(stamps[-1]-stamps[0])/1e9
        baseline[clip]=dict(exact_v9=v9['timings'][clip]['exact'],sidecar_prefix_span_s=span,
            sidecar_prefix_intervals_per_second=63/span,source_frames_sha256=sha(p))
    controls=[]
    for scene,flux in itertools.product(('dense','holes'),(0,32,64,128,256,512)):
        rows=[r for r in q['windows'] if (r['scene'],r['flux'])==(scene,flux)]
        controls.append(dict(scene=scene,flux=flux,windows=len(rows),
            near_truth_reference=sum(r['near_truth']['reference'] for r in rows),
            near_truth_direct=sum(r['near_truth']['direct'] for r in rows),
            note='Flux zero is a null coincidence check, not a target'))
    n=3190*4784
    return dict(schema='seaqr.raw16-feasibility-v10-summary.v1',report_verified=True,
        source_sha256=sha(__file__),plan_sha256=plan,artifact_sha256=hashes,
        hardware=dict(model=provenance['hardware_model'],platform=provenance['platform'],
            cuda=provenance['nvcc'],l4t=provenance['l4t']),
        target=dict(fps=10,provisional=True,frame_budget_s=.1,advance_budget_s=.8,
            minimum_three_window_acquisition_span_s=3.1,
            first_three_complete_windows_with_four_warmup_frames_s=3.5,
            confirmed_live_camera_fps=None,latency_requirement_supplied=False),
        gates=dict(exact_resident_tracking_passed=True,raw_prepoint_and_handoff_passed=True,
            strict_raw_cpu_fft_equivalence_passed=strict,generated_sensitivity_passed=positive and losses==0,
            all_diagnostic_core_cycles_within_10fps_budget=all(g['all_measured_cycles_within_800ms'] for k,g in stage_b.items() if k.endswith('/resident_device')),
            complete_pipeline_10fps_demonstrated=False,real_airborne_accuracy_validated=False,
            production_approved=False,defaults_changed=False),
        stage_a=stage_a,stage_b=stage_b,
        quality=dict(ring_windows_per_build=240,ring_geometry_cases_per_build=20,
            raw_small_prefilter_frames=516,raw_native_prefilter_frames=14,raw_windows=48,
            raw_non_bit_exact_filter_frames=sum(not r['bit_exact'] for r in q['frames']+q['native']),
            raw_ordered_candidate_mismatch_windows=sum(not r['ordered_decisions_exact'] for r in q['windows']),
            generated_target_losses=losses,generated_target_gains=gains,controls=controls),
        memory=dict(native_pixels=n,raw_uint16_frame_bytes=n*2,
            mirrored_ring_response_and_mask_bytes=n*32*5,output_maps_bytes=n*9,
            stage_a_host_response_and_mask_upload_bytes_per_advance=n*8*5,
            stage_b_raw_and_source_mask_upload_bytes_per_advance=n*8*3,
            stage_b_device_ring_copy_payload_bytes_per_advance=n*8*5*2,
            note='Payload bytes, not total DRAM bus traffic; front-end buffers and scratch are additional.'),
        frozen_v9_summary_sha256=V9_SHA,baseline=baseline,
        v9_profile_groups=v9['profiles']['exact']['timing']['groups'],
        exclusions=b['exclusions'],
        decision='No-go for 10 FPS deployment of this prototype. Continue architecture research only; '
            'neither whole-pipeline feasibility nor a hardware impossibility is established.',
        warning='Generated diagnostic core timings are not end-to-end FPS, real-airborne recall, or evidence that a faster GPU scales linearly.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.evidence,a.reference)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(result['gates'],indent=2))
