"""Bounded v8 experiments. Explicit modes; a numerical pass is NOT exactness."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from raw16_speed_v8_common import (reference_ast_unchanged, package_identity, cpu_motion_execution,
    numeric_comparison, cpu_filter, estimate_identity, dense, gm)
from profile_raw16_efficiency import source_paths, sha, compact, write_json
from profile_raw16_v6 import RUNTIME_SHA, BASELINE_HASHES, Spans
from run_raw16_background_v7 import CONFIG, CONFIG_SHA, LIBRARY_SHA, INJECTED_HASHES, normalized_report
from summarize_raw16_cpu_v6 import compare_source_motion, difference, digest
import run_raw16_cpu_v6 as v6
from tiny_target.raw_background_cuda import RawBackgroundCuda, DEFAULT_LIBRARY as BACKGROUND_LIBRARY
from tiny_target.point_filter_cuda import PointFilterCuda, DEFAULT_LIBRARY as POINT_LIBRARY
from tiny_target.evaluation import SyntheticInjector

V7_ARCHIVE_SHA = '6f68819689517cffc802d46ee1128dd678aa41556bee6bad34833daa660b87f4'


def read(path): return json.loads(Path(path).read_text())


def verify(args):
    # Reject unknown source strings before stat(), decoding or opening a sidecar.
    source_paths(args.clip)
    if args.injected and args.clip != '0040': raise ValueError('Controls limited to 0040')
    if args.trace and not args.injected: raise ValueError('Trace requires unchanged controls')
    if args.profile and (args.shadow or args.trace or args.audit): raise ValueError('Separate profiling from heavy diagnostics')
    expected = {}
    for archive, identity in ((args.archive,RUNTIME_SHA),(args.v7_archive,V7_ARCHIVE_SHA)):
        if sha(archive) != identity: raise ValueError('Frozen archive identity changed')
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile(): continue
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive member')
                content = tar.extractfile(member).read()
                expected[str(path)] = content
    for name, content in expected.items():
        if name == 'tiny_target/motion/global_motion.py':
            reference_ast_unchanged(content.decode(),(ROOT/name).read_text())
        elif sha(ROOT/name) != hashlib.sha256(content).hexdigest():
            raise ValueError(f'Undeclared reference change: {name}')
    package = package_identity()
    inventory = {name for name in expected if name.startswith('tiny_target/') and name.endswith('.py')}
    if set(package) != inventory | {'tiny_target/point_filter_cuda.py'}:
        raise ValueError('Unexpected runtime inventory')
    comp = read(args.component)
    if not (comp['passed'] is True and comp['real_media_read'] is False and comp['native_enabled'] is True
            and len(comp['motion']) == 64 and all(r['exact'] for r in comp['motion'])
            and len(comp['filter']['cases']) == 30 and comp['filter']['numerical_screen_passed']
            and len(comp['moving']) == 18 and len(comp['stabilization']) == 18):
        raise ValueError('Incomplete generated gates')
    if comp['package_sha256'] != package: raise ValueError('Runtime changed after generated tests')
    for key, path in (
        ('script_sha256',ROOT/'scripts/check_raw16_speed_v8.py'),
        ('helper_sha256',ROOT/'scripts/raw16_speed_v8_common.py'),
        ('plan_sha256',ROOT/'docs/raw16_speed_v8_plan.md'),
        ('cuda_source_sha256',ROOT/'tiny_target/detection/cuda/point_filter_v8.cu'),
        ('point_library_sha256',POINT_LIBRARY),
    ):
        if comp[key] != sha(path): raise ValueError(f'Generated identity changed: {key}')
    if sha(BACKGROUND_LIBRARY) != LIBRARY_SHA: raise ValueError('Frozen GPU background binary changed')
    # Reuse only the identical generated PVA controls; cache/fitting were active.
    if not args.skip_motion_controls:
        for seed in (75316,129827,85723):
            current=read(args.motion_controls/f'controls_{seed}.json')
            previous=read(args.evidence/f'status_controls_seed{seed}.json')
            if (not current['passed'] or len(current['cases'])!=16 or current['runtime_sha256']!=package
                    or compact(current['cases'])!=compact(previous['cases'])):
                raise ValueError('Generated PVA controls changed')
    if args.skip_motion_controls:
        raise ValueError('Motion controls cannot be bypassed for media')
    hashes = INJECTED_HASHES if args.injected else BASELINE_HASHES[args.clip]
    baseline = args.evidence/f'full_frame_{args.clip}{"_injected" if args.injected else ""}_v6'
    for name, identity in hashes.items():
        if sha(baseline/name) != identity: raise ValueError('Archived media reference changed')
    return baseline, dict(package_sha256=package,component_sha256=sha(args.component),
                         point_library_sha256=sha(POINT_LIBRARY),background_library_sha256=LIBRARY_SHA)


class FilterExperiment:
    def __init__(self, shadow):
        self.shadow=shadow; self.devices={}; self.comparisons=[]

    def events(self, screener, current):
        if current.bit_depth != 16: raise ValueError('RAW16 only')
        cfg=screener.config; screener._last_synthetic_frame=None
        image=np.asarray(current.image,dtype=np.float32)
        valid=((current.image>cfg.dark_floor_dn)&(current.image<float(65535)*cfg.saturation_fraction))
        if current.valid_mask is not None: valid &= current.valid_mask
        if screener._background_cuda is None:
            screener._background_cuda=RawBackgroundCuda(cfg,current.shape)
        product=screener._background_cuda.step(image,valid)
        screener._background_frame_count=screener._background_cuda.count
        if product is None:
            screener._record_availability(current,valid,None,False)
            return []
        white,mask,ready=product
        device=self.devices.get(id(screener))
        if device is None:
            device=PointFilterCuda(current.shape,screener._point_kernel,screener._point_kernel_l2)
            self.devices[id(screener)]=device
        response=device(white)
        if self.shadow:
            reference=cpu_filter(white,screener._point_kernel,screener._point_kernel_l2)
            comparison=numeric_comparison(reference,response,white)
            self.comparisons.append(dict(frame_index=current.frame_index,**comparison))
            if not comparison['numerical_screen_passed']: raise ValueError('RAW response bound exceeded')
        if ready: screener._frames_screened+=1
        screener._valid_fraction_per_frame.append(float(np.mean(mask)))
        screener._record_availability(current,valid,mask,ready)
        screener._last_synthetic_frame=dense._DenseMatchedFrame(response=response,valid_mask=mask,
            timestamp_ns=current.timestamp_ns,frame_index=current.frame_index,
            segment_index=(screener._segment_index or 0),detection_ready=ready)
        return []

    def close(self):
        for device in self.devices.values(): device.close()


def candidate_decision(candidate):
    return dict(index=candidate.candidate_index,xy=[candidate.x_px,candidate.y_px],
        velocity_index=candidate.velocity_index,velocity=list(candidate.velocity_xy_px_s),
        support=candidate.supporting_frame_count,support_weight=candidate.support_weight,
        validity=candidate.to_dict()['validity'],quota=candidate.quota_cell_row_col)


class ControlTrace:
    def __init__(self):
        self.targets=();self.pending=[];self.frames=[];self.windows=[]

    def install(self, context):
        original_inject=SyntheticInjector.inject
        original_events=dense.DensePointScreener._events_for_frame
        original_integrate=dense.CudaShiftAndStack.integrate
        def inject(injector,frame):
            result=original_inject(injector,frame);self.targets=injector.spec.targets;self.pending=[]
            limit=65535*.995
            for target in self.targets:
                if not target.active(frame.frame_index): continue
                xy=target.position_at(frame.timestamp_ns);x,y=map(round,xy)
                ys,xs=slice(y-3,y+4),slice(x-3,x+4)
                original=frame.image[ys,xs];after=result.image[ys,xs]
                self.pending.append(dict(target_id=target.target_id,frame_index=frame.frame_index,xy=list(xy),
                    original_center_dn=float(frame.image[y,x]),injected_center_dn=float(result.image[y,x]),
                    original_patch_min_dn=float(original.min()),original_patch_max_dn=float(original.max()),
                    saturated_patch_before=int(np.count_nonzero(original>=limit)),
                    saturated_patch_after=int(np.count_nonzero(after>=limit)),
                    warp_valid_patch=49 if frame.valid_mask is None else int(np.count_nonzero(frame.valid_mask[ys,xs])),
                    saturation_limit_dn=limit))
            return result
        def events(screener,frame):
            result=original_events(screener,frame);matched=screener._last_synthetic_frame
            for row in self.pending:
                x,y=map(round,row['xy']);entry=dict(row,filter_state=screener._availability[-1]['filter_state'])
                if matched is not None:
                    entry.update(center_filter_valid=bool(matched.valid_mask[y,x]),
                        filter_valid_7x7=int(np.count_nonzero(matched.valid_mask[y-3:y+4,x-3:x+4])),
                        center_response=float(matched.response[y,x]))
                self.frames.append(entry)
            return result
        def integrate(backend,frames):
            result=original_integrate(backend,frames)
            for target in self.targets:
                xy=target.position_at(result.reference_timestamp_ns);x,y=map(round,xy)
                # ROI includes every sample needed by any frozen velocity at +/-3px.
                # Pixel offsets, times, masks and response values are unchanged.
                x0,y0=x-32,y-32;xs,ys=slice(x0,x+33),slice(y0,y+33)
                if x0<0 or y0<0 or x+33>result.score.shape[1] or y+33>result.score.shape[0]:
                    raise ValueError('Diagnostic ROI exceeds image')
                cropped=[replace(f,response=np.ascontiguousarray(f.response[ys,xs]),
                                  valid_mask=np.ascontiguousarray(f.valid_mask[ys,xs])) for f in frames]
                diagnostic=dense.CudaShiftAndStack(backend.config,base_path=ROOT)
                diagnostic.velocity_grid=np.ascontiguousarray([target.velocity_xy_px_s],dtype=np.float32)
                true=original_integrate(diagnostic,cropped)
                region=np.s_[29:36,29:36];valid=true.valid_mask[region]
                peak=None
                if valid.any():
                    loc=np.unravel_index(np.argmax(np.where(valid,true.score[region],-np.inf)),valid.shape)
                    py,px=loc[0]+29,loc[1]+29
                    peak=dict(xy=[x0+int(px),y0+int(py)],score=float(true.score[py,px]),
                              support=int(true.valid_support_count[py,px]))
                self.windows.append(dict(target_id=target.target_id,frames=list(result.frame_indices),
                    all_target_active=all(target.active(f.frame_index) for f in frames),
                    active_frames=sum(target.active(f.frame_index) for f in frames),
                    truth_xy=list(xy),center_valid=bool(result.valid_mask[y,x]),
                    center_selected_velocity=(result.velocity_grid_xy_px_s[int(result.velocity_index[y,x])].tolist()
                        if result.valid_mask[y,x] else None),
                    true_velocity_center_valid=bool(true.valid_mask[32,32]),
                    true_velocity_local_peak=peak,
                    true_velocity_center_support=int(true.valid_support_count[32,32])))
            return result
        context.enter_context(patch.object(SyntheticInjector,'inject',inject))
        context.enter_context(patch.object(dense.DensePointScreener,'_events_for_frame',events))
        context.enter_context(patch.object(dense.CudaShiftAndStack,'integrate',integrate))


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    baseline,provenance=verify(args)
    gpu=args.mode in ('filter','combined');cpu=args.mode in ('cpu','combined')
    if args.shadow and not gpu: raise ValueError('Shadow mode requires GPU filter')
    experiment=FilterExperiment(args.shadow);instances=[];motion_rows=[];candidates=[];fit_rows=[]
    trace=ControlTrace() if args.trace else None
    spans=Spans() if args.profile else None
    original_init=dense.DensePointScreener.__init__
    original_extract=dense.CandidateExtractor.extract
    def remember(screener,*a,**kw):
        original_init(screener,*a,**kw);instances.append(screener)
    def extract(extractor,window,**kw):
        result=original_extract(extractor,window,**kw)
        candidates.append(dict(frames=list(window.frame_indices),decisions=[candidate_decision(c) for c in result.candidates]))
        return result
    error=None;status=None;cache=None
    try:
        with ExitStack() as context:
            context.enter_context(patch.object(dense.DensePointScreener,'__init__',remember))
            context.enter_context(patch.object(dense.CandidateExtractor,'extract',extract))
            if cpu: cache=context.enter_context(cpu_motion_execution())
            original_fit=dense.fit_global_motion
            def fit(pairs,cfg):
                result=original_fit(pairs,cfg)
                fit_rows.append(estimate_identity(result))
                return result
            context.enter_context(patch.object(dense,'fit_global_motion',fit))
            if gpu:
                def events(screener,current): return experiment.events(screener,current)
                context.enter_context(patch.object(dense.DensePointScreener,'_events_for_frame_cuda',events))
            if trace: trace.install(context)
            if args.audit:
                from run_raw16_background_v7 import Audit
                if gpu: raise ValueError('Use shadow numeric audit for non-exact GPU filter')
                args.output.parent.mkdir(parents=True,exist_ok=True)
                events=context.enter_context(args.output.with_suffix('.audit.jsonl').open('x'))
                Audit(events).install(context)
            context.enter_context(v6.profile_motion(motion_rows))
            context.enter_context(patch.object(v6.full,'CONFIG',CONFIG))
            context.enter_context(patch.object(v6.full,'MOTION',v6.CONFIG))
            context.enter_context(patch.object(v6.full,'FROZEN_HASHES',{
                **v6.full.FROZEN_HASHES,CONFIG:CONFIG_SHA,v6.CONFIG:v6.CONFIG_SHA,BACKGROUND_LIBRARY:LIBRARY_SHA}))
            if spans:
                spans.install(context)
                spans.wrap(context,RawBackgroundCuda,'step',label='cuda_temporal_support_host')
                spans.wrap(context,PointFilterCuda,'__call__',label='cuda_point_filter_host')
            call=argparse.Namespace(clip=args.clip,injected=args.injected,output=args.output)
            status=v6.full.run(call) if spans is None else spans.call('validation_run','validation_and_orchestration',v6.full.run,call)
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        for screener in instances: screener.close()
        experiment.close()
        if args.output.is_dir():
            write_json(args.output/'motion_profile.json',motion_rows)
            write_json(args.output/'candidate_decisions.json',compact(candidates))
            write_json(args.output/'global_fit_identities.json',fit_rows)
            write_json(args.output/'experiment.json',dict(schema_version='seaqr.raw16-speed-v8-run.v1',
                mode=args.mode,shadow=args.shadow,audit=args.audit,profile=args.profile,trace=args.trace,
                error=error,provenance=provenance,wrapper_sha256=sha(__file__),
                filter_comparisons=experiment.comparisons,
                cache=None if cache is None else dict(hits=cache.hits,misses=cache.misses),
                default_changed=False,production_approved=False,
                warning='Experimental GPU point filter is NOT bit-exact. CPU motion remains exact-gated. '
                        'Frozen detector settings, CPU cubic stabilization and VPI cache isolation retained.'))
            if trace: write_json(args.output/'control_trace.json',dict(frames=trace.frames,windows=trace.windows))
            if spans and spans.nodes: write_json(args.output/'stage_profile.json',dict(timing=spans.summary(),error=error))
    reference=normalized_report(read(baseline/'report.json'))
    observed=normalized_report(read(args.output/'report.json'))
    compare=dict(exact_semantics=reference==observed,first_difference=difference(reference,observed),
                 reference_semantic_sha256=digest(reference),candidate_semantic_sha256=digest(observed),
                 **compare_source_motion(baseline,args.output))
    checks=read(args.output/'checks.json')['checks']
    diagnostic_pass=(status==0 and compare['source_frames_exact'] and compare['motion_points_exact']
                     and checks['processing_integrity_passed'] and checks['detection_availability_passed'])
    if not gpu: diagnostic_pass &= compare['exact_semantics']
    if args.shadow: diagnostic_pass &= len(experiment.comparisons)==63 and all(r['numerical_screen_passed'] for r in experiment.comparisons)
    write_json(args.output/'comparison.json',dict(diagnostic_run_passed=bool(diagnostic_pass),
        exact_gate_passed=bool(diagnostic_pass and compare['exact_semantics']),comparison=compare,
        real_airborne_accuracy_validated=False,production_approved=False))
    print(json.dumps(dict(diagnostic_run_passed=bool(diagnostic_pass),mode=args.mode,comparison=compare)),flush=True)
    return 0 if diagnostic_pass else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip',choices=('0029','0040'),required=True)
    parser.add_argument('--mode',choices=('reference','cpu','filter','combined'),required=True)
    for name in ('archive','v7-archive','component','evidence','motion-controls','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('shadow','audit','profile','trace','injected','skip-motion-controls'):
        parser.add_argument('--'+name,action='store_true')
    raise SystemExit(run(parser.parse_args()))
