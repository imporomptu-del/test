"""Verify downloaded generated-only v11 evidence against local source."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from build_resident_v11 import transformed,TRACK_SHA,RING_SHA

ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='8a817884e89a8d158f20f694440dc6e17457a412a4c632470acc139c325a4668'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def numbers(values):
    require(len(values)==12 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in values),'Incomplete/invalid timing samples')
    return dict(n=12,minimum=min(values),median=statistics.median(values),maximum=max(values))


def coverage(rows,keys,expected):
    actual=[tuple(r[k] for k in keys) for r in rows]
    require(len(actual)==len(expected) and set(actual)==set(expected),'Missing/duplicate coverage')


def summarize(evidence,remote_root):
    hashes={}
    def read(name):
        path=evidence/name;hashes[name]=sha(path);return json.loads(path.read_text())
    build=read('build/candidate_01/build.json')
    candidate_sha=sha(evidence/'build/candidate_01/libresident_tracking_v11.so')
    require(build['returncode']==0 and build['library_sha256']==candidate_sha,'Candidate build failed or changed')
    require(build['builder_sha256']==sha(ROOT/'scripts/build_resident_v11.py')
        and build['plan_sha256']==sha(ROOT/'docs/raw16_execution_v11_plan.md'),'Builder/plan mismatch')
    require(build['reference_tracker_source_sha256']==TRACK_SHA
        and build['reference_ring_source_sha256']==RING_SHA,'Reference source mismatch')
    generated=evidence/'build/candidate_01/generated.cu'
    require(build['generated_source_sha256']==sha(generated),'Generated kernel changed')
    local=transformed().splitlines(keepends=True);remote=generated.read_text().splitlines(keepends=True)
    require(remote[0]=='#include "'+str(remote_root)+'/tiny_target/detection/cuda/synthetic_tracking.cu"\n'
        and remote[1:]==local[1:],'Undeclared kernel transformation')
    require(sha(evidence/'build/reference/libresident_tracking_v10.so')==BASE_SHA,'Resident reference changed')
    quality=read('results/ring_quality_01.json');edges=read('results/stencil_edges_01.json')
    require(quality['passed'] is True and quality['real_media_read'] is False
        and quality['library_sha256']==candidate_sha,'Main generated equality failed')
    require(quality['checker_sha256']==sha(ROOT/'scripts/check_resident_v10.py')
        and quality['wrapper_sha256']==sha(ROOT/'scripts/resident_tracking_v10.py'),'Quality source mismatch')
    coverage(quality['rows'],('scene','flux','polarity','last_frame'),list(itertools.product(
        ('straight','turn','acceleration','short_visibility','holes','clutter'),(0,2,4,8,16),('bright','dark'),(15,23,31,39))))
    require(all(r['arrays_exact'] and r['ordered_decisions_exact'] for r in quality['rows']),'Changed tracking decisions')
    require(len(quality['geometry'])==20 and all(r['exact'] for r in quality['geometry'])
        and len(quality['guards'])==6 and all(r['passed'] for r in quality['guards']),'Geometry/state guard failed')
    require(edges['passed'] is True and edges['real_media_read'] is False and edges['library_sha256']==candidate_sha
        and edges['checker_sha256']==sha(ROOT/'scripts/check_stencil_v11.py'),'Edge-check provenance failed')
    normalized=[dict(r,shape=tuple(r['shape'])) for r in edges['rows']]
    coverage(normalized,('grid','shape','polarity','scene'),list(itertools.product(
        ('normal','cutoffs','velocity_ties'),((7,13),(65,97)),('bright','dark'),('full','empty','sparse','border','flat_ties'))))
    require(all(r['exact'] for r in edges['rows']),'Interpolation edge equality failed')
    timing=read('results/stencil_timing_01.json')
    require(timing['passed'] is True and timing['real_media_read'] is False and timing['pipeline_benchmark'] is False
        and timing['library_sha256']==dict(reference=BASE_SHA,candidate=candidate_sha),'Timing provenance failed')
    require(timing['gate_sha256']==hashes['results/ring_quality_01.json']
        and timing['edges_sha256']==hashes['results/stencil_edges_01.json']
        and timing['script_sha256']==sha(ROOT/'scripts/benchmark_stencil_v11.py'),'Timing prerequisites changed')
    expected=[dict(scene=s,repeat=r,mode=m) for s in ('dense','holes') for r in range(4)
        for m in (('reference','candidate') if r%2==0 else ('candidate','reference'))]
    require(timing['schedule']==expected and [{k:t[k] for k in ('scene','repeat','mode')} for t in timing['trials']]==expected,'Incomplete tracking timing schedule')
    tracking={}
    for t in timing['trials']:
        require([s['cycle'] for s in t['samples']]==[0,1,2] and all(s['outputs_exact'] for s in t['samples']),'Failed/missing timed output')
    for scene in ('dense','holes'):
        rows={}
        for mode in ('reference','candidate'):
            samples=[s for t in timing['trials'] if (t['scene'],t['mode'])==(scene,mode) for s in t['samples']]
            rows[mode]={k:numbers([s[k] for s in samples]) for k in ('host_s','kernel_ms','append_host_s')}
        rows['host_time_reduction_fraction']=1-rows['candidate']['host_s']['median']/rows['reference']['host_s']['median']
        rows['kernel_time_reduction_fraction']=1-rows['candidate']['kernel_ms']['median']/rows['reference']['kernel_ms']['median']
        tracking[scene]=rows
    motion=read('results/motion_01.json')
    require(motion['passed'] is True and motion['real_media_read'] is False and motion['pipeline_benchmark'] is False,'Motion controls failed')
    for key,file in (('script_sha256','scripts/check_motion_lut_v11.py'),('adapter_sha256','scripts/motion_lut_v11.py'),
        ('motion_source_sha256','tiny_target/motion/pva_pyrlk.py'),('config_sha256','configs/evaluation/raw16_motion_v6.json')):
        require(motion[key]==sha(ROOT/file),'Motion source/config mismatch')
    coverage(motion['pixel_checks'],('depth','mask'),list(itertools.product((9,10,12,14,16),('all','none','pattern'))))
    require(all(r['exact'] for r in motion['pixel_checks']),'Feature pixel mismatch')
    cases=('dim_stationary','dim_subpixel','dim_translation','dim_larger_shift','high_dynamic_range',
        'bright_translation','gain_and_offset_change','sensor_fixed_pattern','moving_foreground_patch',
        'unobservable_flat','independent_noise','unsupported_rotation')
    coverage(motion['motion_checks'],('seed','case'),list(itertools.product((75316,75317,75318,75319),cases)))
    require(all(r['exact'] and r['inputs_unchanged'] and r['expected_decisions'] for r in motion['motion_checks']),
        'Motion identity/expected decision failure')
    expected_motion=[dict(scene=s,repeat=r,mode=m) for s in ('range','dim_texture') for r in range(4)
        for m in (('reference','candidate') if r%2==0 else ('candidate','reference'))]
    require([{k:t[k] for k in ('scene','repeat','mode')} for t in motion['timings']]==expected_motion,'Incomplete conversion schedule')
    conversion={}
    for t in motion['timings']:
        require([s['cycle'] for s in t['samples']]==[0,1,2] and all(s['exact'] for s in t['samples']),'Native conversion mismatch')
    for scene in ('range','dim_texture'):
        conversion[scene]={mode:numbers([s['host_s'] for t in motion['timings'] if (t['scene'],t['mode'])==(scene,mode) for s in t['samples']]) for mode in ('reference','candidate')}
        conversion[scene]['time_reduction_fraction']=1-conversion[scene]['candidate']['median']/conversion[scene]['reference']['median']
    return dict(schema='seaqr.execution-v11-summary.v1',verified=True,artifact_sha256=hashes,
        verifier_sha256=sha(__file__),tracking=tracking,motion_conversion=conversion,
        quality=dict(tracking_windows=240,geometry_cases=20,state_guards=6,interpolation_cases=60,
            motion_pixel_cases=15,motion_estimator_cases=48,tracking_timed_windows=48,conversion_timed_frames=48),
        gates=dict(exact_generated_tracking=True,exact_generated_motion=True,
            tracking_faster_in_both_scenes=all(g['host_time_reduction_fraction']>0 for g in tracking.values()),
            conversion_faster_in_both_scenes=all(g['time_reduction_fraction']>0 for g in conversion.values()),
            real_video_regression=False,full_pipeline_rebenchmarked=False,production_approved=False,defaults_changed=False),
        exclusions=timing['exclusions'],
        warning='No new full-pipeline FPS or real-airborne accuracy measurement. Conversion timing excludes the rest of motion estimation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','remote-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.evidence,a.remote_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(result['gates'],indent=2))
