"""Strictly gated 64-frame RAW0029/0040 exact-execution experiment."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_raw16_cpu_v6 as v6
from raw16_speed_v8_common import package_identity,cpu_motion_execution,estimate_identity,dense
from run_raw16_speed_v8 import read,candidate_decision,V7_ARCHIVE_SHA
from run_raw16_background_v7 import CONFIG,CONFIG_SHA,LIBRARY_SHA,INJECTED_HASHES,normalized_report,Audit
from profile_raw16_v6 import RUNTIME_SHA,BASELINE_HASHES,Spans
from profile_raw16_efficiency import sha,source_paths,compact,write_json
from summarize_raw16_cpu_v6 import digest,difference,compare_source_motion
from exact_v9_common import ExactExecution,compare_audits,SYNTHETIC_LIBRARY_SHA
from tiny_target.warp_translation_cuda import DEFAULT_LIBRARY,OPENCV_BUILD_SHA256,TABLE_SHA256
from tiny_target.raw_background_cuda import DEFAULT_LIBRARY as BACKGROUND_LIBRARY

V8_ARCHIVE_SHA='0b2a5224cfb83669c099b52dee79573f6240d9ae9fc0095e1c2cccef573d9175'

def verify(args):
    source_paths(args.clip)
    if args.injected and args.clip!='0040':raise ValueError('Frozen controls limited to 0040')
    if args.audit and (args.mode!='exact' or args.profile):raise ValueError('Audit exact arm separately from profiling')
    expected={}
    for path,identity in ((args.archive,RUNTIME_SHA),(args.v7_archive,V7_ARCHIVE_SHA),(args.v8_archive,V8_ARCHIVE_SHA)):
        if sha(path)!=identity:raise ValueError('Archived source identity mismatch')
        with tarfile.open(path) as archive:
            for item in archive:
                if not item.isfile():continue
                name=Path(item.name)
                if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe source member')
                expected[str(name)]=hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    for name,identity in expected.items():
        if sha(ROOT/name)!=identity:raise ValueError('Frozen source changed: '+name)
    package=package_identity()
    inventory={n for n in expected if n.startswith('tiny_target/') and n.endswith('.py')}
    if set(package)!=inventory|{'tiny_target/point_filter_fft_exact.py','tiny_target/warp_translation_cuda.py'}:
        raise ValueError('Unexpected runtime inventory')
    component=read(args.component)
    if (component.get('passed') is not True or component['real_media_read'] is not False
        or component['native'] is not True or component['quick'] is not False
        or not component['fft_exact'] or not component['warp_exact'] or not component['moving_exact']
        or len(component['fft'])!=40 or len(component['warp'])!=350 or len(component['phases'])!=1024
        or len(component['moving'])!=18 or len(component['thresholds'])!=33):
        raise ValueError('Complete generated exactness gates are required')
    if component['package_sha256']!=package:raise ValueError('Runtime changed since generated checks')
    for key,path in (
        ('checker_sha256',ROOT/'scripts/check_exact_v9.py'),('plan_sha256',ROOT/'docs/raw16_exact_v9_plan.md'),
        ('library_sha256',DEFAULT_LIBRARY),('cuda_source_sha256',ROOT/'tiny_target/detection/cuda/warp_translation_v9.cu')):
        if component[key]!=sha(path):raise ValueError('Generated artifact changed: '+key)
    if component['opencv_build_sha256']!=OPENCV_BUILD_SHA256 or component['table_sha256']!=TABLE_SHA256:
        raise ValueError('Unexpected interpolation/reference build')
    if sha(BACKGROUND_LIBRARY)!=LIBRARY_SHA or sha(ROOT/'build/cuda/libtiny_target_cuda.so')!=SYNTHETIC_LIBRARY_SHA:
        raise ValueError('Frozen accelerator libraries changed')
    gate=read(args.motion_controls/'gate.json')
    if gate['passed'] is not True or gate['package_sha256']!=package or len(gate['rows'])!=3:
        raise ValueError('Generated PVA control gate changed')
    for row in gate['rows']:
        path=args.motion_controls/f'controls_{row["seed"]}.json'
        current=read(path);old=read(args.evidence/f'status_controls_seed{row["seed"]}.json')
        if sha(path)!=row['sha256'] or not current['passed'] or len(current['cases'])!=16 or compact(current['cases'])!=compact(old['cases']):
            raise ValueError('Generated PVA case identity changed')
    baseline=args.evidence/f'full_frame_{args.clip}{"_injected" if args.injected else ""}_v6'
    for name,identity in (INJECTED_HASHES if args.injected else BASELINE_HASHES[args.clip]).items():
        if sha(baseline/name)!=identity:raise ValueError('Frozen reference evidence changed')
    return baseline,dict(package_sha256=package,component_sha256=sha(args.component),
        warp_library_sha256=sha(DEFAULT_LIBRARY),background_library_sha256=LIBRARY_SHA,
        synthetic_library_sha256=SYNTHETIC_LIBRARY_SHA,pva_gate_sha256=sha(args.motion_controls/'gate.json'))

def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    baseline,provenance=verify(args)
    execution=ExactExecution(shadow=args.audit);instances=[];candidates=[];fits=[];motion=[]
    spans=Spans() if args.profile else None
    original_init=dense.DensePointScreener.__init__;original_extract=dense.CandidateExtractor.extract
    def remember(screener,*a,**kw):original_init(screener,*a,**kw);instances.append(screener)
    def extract(extractor,window,**kw):
        out=original_extract(extractor,window,**kw)
        candidates.append(dict(frames=list(window.frame_indices),decisions=[candidate_decision(c) for c in out.candidates]));return out
    error=None;cache=None;status=None
    try:
        with ExitStack() as context:
            context.enter_context(patch.object(dense.DensePointScreener,'__init__',remember))
            context.enter_context(patch.object(dense.CandidateExtractor,'extract',extract))
            cache=context.enter_context(cpu_motion_execution())
            original_fit=dense.fit_global_motion
            def fit(pairs,config):
                out=original_fit(pairs,config);fits.append(estimate_identity(out));return out
            context.enter_context(patch.object(dense,'fit_global_motion',fit))
            if args.mode=='exact':execution.install(context)
            if args.audit:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                handle=context.enter_context(args.output.with_suffix('.audit.jsonl').open('x'))
                Audit(handle).install(context)
            context.enter_context(v6.profile_motion(motion))
            context.enter_context(patch.object(v6.full,'CONFIG',CONFIG))
            context.enter_context(patch.object(v6.full,'MOTION',v6.CONFIG))
            context.enter_context(patch.object(v6.full,'FROZEN_HASHES',{
                **v6.full.FROZEN_HASHES,CONFIG:CONFIG_SHA,v6.CONFIG:v6.CONFIG_SHA,BACKGROUND_LIBRARY:LIBRARY_SHA}))
            if spans:spans.install(context)
            call=argparse.Namespace(clip=args.clip,injected=args.injected,output=args.output)
            status=v6.full.run(call) if spans is None else spans.call('validation_run','validation_and_orchestration',v6.full.run,call)
    except BaseException as exc:error=repr(exc);raise
    finally:
        for screener in instances:screener.close()
        execution.close()
        if args.output.is_dir():
            write_json(args.output/'motion_profile.json',motion)
            write_json(args.output/'candidate_decisions.json',compact(candidates))
            write_json(args.output/'global_fit_identities.json',fits)
            write_json(args.output/'experiment.json',dict(schema_version='seaqr.raw16-exact-v9-run.v1',
                mode=args.mode,audit=args.audit,profile=args.profile,injected=args.injected,error=error,
                provenance=provenance,wrapper_sha256=sha(__file__),adapter_sha256=sha(ROOT/'scripts/exact_v9_common.py'),
                execution=execution.record(),cache=None if cache is None else dict(hits=cache.hits,misses=cache.misses),
                default_changed=False,production_approved=False))
            if spans:write_json(args.output/'stage_profile.json',dict(timing=spans.summary(),error=error))
    observed=normalized_report(read(args.output/'report.json'));reference=normalized_report(read(baseline/'report.json'))
    result=dict(exact_semantics=observed==reference,first_difference=difference(reference,observed),
        reference_semantic_sha256=digest(reference),candidate_semantic_sha256=digest(observed),
        **compare_source_motion(baseline,args.output))
    checks=read(args.output/'checks.json')['checks']
    passed=(status==0 and result['exact_semantics'] and result['source_frames_exact'] and result['motion_points_exact']
        and checks['processing_integrity_passed'] and checks['detection_availability_passed'])
    if args.mode=='exact':passed &= execution.filter_calls==63 and execution.warp_calls==63
    if args.audit:
        passed &= len(execution.filter_checks)==63 and len(execution.warp_checks)==63 and all(execution.filter_checks+execution.warp_checks)
        if not args.injected:
            result['array_audit']=compare_audits(args.v8_results/f'audit_{args.clip}_cpu.audit.jsonl',args.output.with_suffix('.audit.jsonl'))
            passed &= result['array_audit']['passed']
    write_json(args.output/'comparison.json',dict(exact_gate_passed=bool(passed),comparison=result,
        all_controls_recovered=checks.get('synthetic_controls_passed'),real_airborne_accuracy_validated=False,production_approved=False))
    print(json.dumps(dict(exact_gate_passed=bool(passed),mode=args.mode,comparison=result)),flush=True)
    return 0 if passed else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',choices=('0029','0040'),required=True)
    p.add_argument('--mode',choices=('reference','exact'),required=True)
    for name in ('archive','v7-archive','v8-archive','component','motion-controls','evidence','v8-results','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('injected','audit','profile'):p.add_argument('--'+name,action='store_true')
    raise SystemExit(run(p.parse_args()))
