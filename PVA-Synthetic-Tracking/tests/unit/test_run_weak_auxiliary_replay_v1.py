"""Generated-only runner guards; no historical journals, media or native GPU."""
from copy import deepcopy
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT=Path(__file__).resolve().with_name('run_weak_auxiliary_replay_v1.py')
if not SCRIPT.is_file():SCRIPT=Path(__file__).resolve().parents[2]/'scripts/run_weak_auxiliary_replay_v1.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('auxiliary_replay_test_module',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import replay_weak_divergence_v1 as h
import replay_tracking_v27 as d
import combined_v29_state as st
from accuracy_v56_capture import tile_rectangle,validate_rectangle,FLOAT_FIELDS,FLAG_FIELDS
from tiny_target.visible_baseline import VisibleConfig,VisibleTracks
from tiny_target.tracking.kalman import KalmanTrackManager

MODULES={'replay_weak_divergence_v1':h,'replay_tracking_v27':d,'combined_v29_state':st}


class FakeAuxiliary:
    def __init__(self):self.saved=None;self.done=0
    def snapshot(self):return dict(done=self.done)
    def prepare(self,frame,stamp,segment,forecasts):
        self.saved=deepcopy(forecasts)
        if forecasts:forecasts[0]['predicted_mean'][0]=999999
        return deepcopy(self.saved)
    def step(self,records,proposals,matrix,shape,provider):
        if records:records[0]['reference_xy'][0]=999999
        if proposals:proposals[0]['x']=999999
        matrix[:]=999999
        self.done+=1
        return [],{'calls':self.done}


class DummyProvider:
    enabled=False
    calls=[]
    def __call__(self,forecast):return None


def generated():
    cfg=VisibleConfig(confirmation_hits=2,minimum_moving_excursion_px=1,
        tracking_peak_nms_radius_px=2,motion_quality_enabled=False)
    old=VisibleTracks(cfg,10);rows=[]
    for frame in range(7):
        proposals=[] if frame==5 else [dict(x=float(80+frame*3),y=80.,polarity='bright',score=10.,response_dn=10.)]
        stamp=frame*100000000;learning=old.learning_centers(stamp,0)
        records,metrics=old.update(deepcopy(proposals),frame,stamp,0,np.eye(3),(192,192))
        b=dict(frame_index=frame,timestamp_ns=stamp,segment=0,candidates=proposals,
            source_to_reference=np.eye(3).tolist(),coverage=dict(full_shape_hw=[192,192]),tracks=deepcopy(records),tracking_metrics=deepcopy(metrics))
        t=dict(frame_index=frame,timestamp_ns=stamp,segment=0,strong_proposals=deepcopy(proposals),prior_forecasts=[{'obsolete':'NEVER USE'}],records=[{'weak_evidence':'NEVER USE'}])
        ref=dict(frame=frame,output=d.digest([records,metrics]),state=d.digest(st.state_of(old)),learning=d.digest(learning))
        rows.append((b,t,ref))
    return cfg,rows


class ReplayGuards(unittest.TestCase):
    def test_real_seven_frame_stream_passes_independent_primary_and_auxiliary_auditors(self):
        import audit_weak_auxiliary_replay_v1 as audit
        from weak_continuation_auxiliary_v1 import AuxiliaryGapSupport
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=AuxiliaryGapSupport(cfg,10)
        primary_audit=audit.PrimaryAudit(asdict(cfg));aux_audit=audit.AuxiliaryAudit(asdict(cfg))
        class Unavailable:
            enabled=False
            def __init__(self):self.calls=[]
            def __call__(self,forecast):
                self.calls.append(dict(query_identity=forecast['identity'],forecast=deepcopy(forecast),
                    status='outside_frozen_window',descriptor=None))
                return None
        def capture_check(call,prior,priors,shape):
            self.assertEqual(call['forecast'],prior)
            self.assertEqual(call['status'],'outside_frozen_window');return None
        for b,t,ref in rows:
            p,a=m.replay_frame(tracker,aux,b,t,ref,Unavailable(),MODULES)
            # Audit the actual serialized interface, not in-memory custom types.
            p,a,b,ref=(json.loads(json.dumps(value)) for value in (p,a,b,ref))
            primary_audit.step(p,a,b,ref);aux_audit.step(a,b,capture_check)
        self.assertEqual(primary_audit.frames,7)
        self.assertEqual(aux_audit.counts['current_weak_observation_count'],0)
        self.assertGreater(aux_audit.counts['actual_provider_calls'],0)

    def test_real_auxiliary_core_generated_full_sequence_missing_evidence(self):
        from weak_continuation_auxiliary_v1 import AuxiliaryGapSupport
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=AuxiliaryGapSupport(cfg,10)
        for b,t,ref in rows:
            p,a=m.replay_frame(tracker,aux,b,t,ref,DummyProvider(),MODULES)
            self.assertEqual(p['primary_state_sha256_before_aux'],ref['state'])
            self.assertEqual(a['auxiliary_records'],[])
            self.assertFalse(any(r['status']=='invalid_capture_or_weak_covariance' for r in a['auxiliary_metrics']['decisions']))
    def test_complete_generated_primary_parity_and_detached_hostile_mutation(self):
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=FakeAuxiliary()
        for b,t,ref in rows:
            p,a=m.replay_frame(tracker,aux,b,t,ref,DummyProvider(),MODULES)
            self.assertEqual(p['records'],b['tracks'])
            self.assertEqual(p['primary_state_before_aux'],p['primary_state_after_aux'])
            self.assertEqual(p['primary_prepare_state_sha256_before'],p['primary_prepare_state_sha256_after'])
            self.assertEqual(p['primary_output_sha256_before_aux'],ref['output'])
            self.assertEqual(p['actual_learning_sha256'],ref['learning'])
            self.assertEqual(p['primary_output_normalized_before_aux'],d.normalized([b['tracks'],b['tracking_metrics']]))
            self.assertEqual(p['primary_output_normalized_before_aux'],p['primary_output_normalized_after_aux'])
            self.assertNotIn('obsolete',str(a['primary_forecasts']))
            if a['primary_forecasts']:self.assertNotEqual(a['primary_forecasts'][0]['predicted_mean'][0],999999)

    def test_forecast_is_causal_and_detached(self):
        cfg,rows=generated();tracker=VisibleTracks(cfg,10)
        b,t,_=rows[0];tracker.update(b['candidates'],0,0,0,np.eye(3),(192,192))
        before=d.digest(st.state_of(tracker));priors=m.primary_forecasts(tracker,1,100000000,0)
        self.assertEqual(len(priors),1);self.assertEqual(priors[0]['reference_xy'],[80.,80.])
        priors[0]['predicted_covariance'][0][0]=9000
        self.assertEqual(d.digest(st.state_of(tracker)),before)
        self.assertEqual(m.primary_forecasts(tracker,1,100000000,1),[])

    def test_normalized_output_snapshots_are_taken_on_opposite_sides_of_aux_step(self):
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=FakeAuxiliary();b,t,ref=rows[0]
        phase=['before'];calls=[];old=aux.step
        def step(*args):
            result=old(*args);phase[0]='after';return result
        aux.step=step
        def normalizer(value):
            if isinstance(value,list) and len(value)==2 and isinstance(value[0],list) and isinstance(value[1],dict):
                calls.append(phase[0])
            return d.normalized(value)
        modules=dict(MODULES,replay_tracking_v27=SimpleNamespace(normalized=normalizer,digest=d.digest))
        p,_=m.replay_frame(tracker,aux,b,t,ref,DummyProvider(),modules)
        self.assertEqual(calls,['before','after'])
        self.assertEqual(p['primary_output_normalized_before_aux'],p['primary_output_normalized_after_aux'])
        self.assertIsNot(p['primary_output_normalized_before_aux'],p['primary_output_normalized_after_aux'])
        p['primary_output_normalized_before_aux'][1].clear()
        self.assertTrue(p['primary_output_normalized_after_aux'][1])

    def test_original_private_state_digest_is_hard_gate(self):
        cfg,rows=generated();b,t,ref=rows[0];ref['state']='0'*64
        with self.assertRaisesRegex(ValueError,'Original output/private-state'):
            m.replay_frame(VisibleTracks(cfg,10),FakeAuxiliary(),b,t,ref,DummyProvider(),MODULES)

    def test_original_learning_digest_is_hard_gate(self):
        cfg,rows=generated();b,t,ref=rows[0];ref['learning']='0'*64
        with self.assertRaisesRegex(ValueError,'Original preupdate learning'):
            m.replay_frame(VisibleTracks(cfg,10),FakeAuxiliary(),b,t,ref,DummyProvider(),MODULES)

    def test_prepare_actual_primary_mutation_caught(self):
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=FakeAuxiliary();b,t,ref=rows[0]
        old=aux.prepare
        def evil(*args):
            result=old(*args);tracker.previous_timestamp_ns=77;return result
        aux.prepare=evil
        with self.assertRaisesRegex(ValueError,'prepare mutated primary'):
            m.replay_frame(tracker,aux,b,t,ref,DummyProvider(),MODULES)

    def test_step_actual_primary_mutation_caught(self):
        cfg,rows=generated();tracker=VisibleTracks(cfg,10);aux=FakeAuxiliary();b,t,ref=rows[0]
        old=aux.step
        def evil(*args):
            result=old(*args);tracker.previous_timestamp_ns=77;return result
        aux.step=evil
        with self.assertRaisesRegex(ValueError,'step/provider changed primary'):
            m.replay_frame(tracker,aux,b,t,ref,DummyProvider(),MODULES)

    def test_strong_stream_difference_caught(self):
        cfg,rows=generated();b,t,ref=rows[0];t['strong_proposals'][0]['x']+=1
        with self.assertRaises(h.ParityError):m.replay_frame(VisibleTracks(cfg,10),FakeAuxiliary(),b,t,ref,DummyProvider(),MODULES)

    def test_time_alignment_caught(self):
        cfg,rows=generated();b,t,ref=rows[0];t['timestamp_ns']=1
        with self.assertRaisesRegex(ValueError,'coordinate sequence'):
            m.replay_frame(VisibleTracks(cfg,10),FakeAuxiliary(),b,t,ref,DummyProvider(),MODULES)


class CaptureGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();(self.root/'shadow/captures').mkdir(parents=True)
        self.evidence=m.BoundFiles(self.root);self.pins={};self.descriptors=[]
    def tearDown(self):self.tmp.cleanup()
    def capture(self,name,center=(128.,128.),value=1):
        rect=tile_rectangle((512,512),128,list(center),45);b=rect['capture_bounds_exclusive_xyxy'];shape=(b[3]-b[1],b[2]-b[0])
        meta=dict(frame=5,segment=0,prelearning=True,baseline_learning_unchanged=True,read_only_capture=True,
            production_selection_unchanged=True,rectangle=rect,float_fields=list(FLOAT_FIELDS),flag_fields=list(FLAG_FIELDS))
        npz=self.root/'shadow/captures'/f'{name}.npz';jsonpath=npz.with_suffix('.json')
        np.savez_compressed(npz,values=np.full(shape+(len(FLOAT_FIELDS),),value,np.float32),flags=np.zeros(shape+(len(FLAG_FIELDS),),np.uint8))
        jsonpath.write_text(json.dumps(meta))
        desc=dict(identity='0/bright:999',path=f'captures/{name}.npz',sha256=m.sha(npz),metadata_path=f'captures/{name}.json',metadata_sha256=m.sha(jsonpath))
        self.pins['shadow/'+desc['path']]=desc['sha256'];self.pins['shadow/'+desc['metadata_path']]=desc['metadata_sha256']
        self.descriptors.append(desc);return desc
    def provider(self,enabled=True):
        return m.CaptureProvider(self.evidence,self.pins,dict(frame_index=5,segment=0,capture_files=self.descriptors),[512,512],enabled,validate_rectangle)
    def query(self,center=(128.,128.)):
        return dict(frame_index=5,segment=0,identity='0/bright:1',reference_xy=list(center))
    def test_identity_ignored_and_first_path_selected_without_value_search(self):
        self.capture('z',value=2);self.capture('a',value=0);p=self.provider();value=p(self.query())
        self.assertEqual(p.calls[0]['descriptor']['path'],'captures/a.npz')
        self.assertTrue(np.all(value['values']==0));self.assertFalse(value['values'].flags.writeable)
    def test_missing_geometric_coverage_is_explicit(self):
        self.capture('a');p=self.provider();self.assertIsNone(p(self.query((400.,400.))))
        self.assertEqual(p.calls[0]['status'],'no_geometrically_complete_cached_capture')
        self.assertNotIn('shadow/captures/a.npz',self.evidence.hashes)
    def test_unscheduled_does_not_open_npz(self):
        self.capture('a');p=self.provider(False);self.assertIsNone(p(self.query()))
        self.assertEqual(p.calls[0]['status'],'outside_frozen_window')
        self.assertNotIn('shadow/captures/a.npz',self.evidence.hashes)
    def test_hash_mismatch_fails_before_array_use(self):
        desc=self.capture('a');p=self.provider();(self.root/'shadow'/desc['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Bound input changed'):p(self.query())
    def test_shared_capture_same_frame_allowed_but_each_query_logged(self):
        self.capture('a');self.descriptors.append(dict(self.descriptors[0],identity='0/dark:5'));p=self.provider()
        p(self.query());p(dict(self.query(),identity='0/dark:3'))
        self.assertEqual(len(p.inventory),1);self.assertEqual(len(p.calls),2)
    def test_geometry_native_edges_are_not_clamped_to_fake_full_gate(self):
        rect=tile_rectangle((512,512),128,[5.,5.],45)
        self.assertFalse(m.geometrically_covers(rect,self.query((5.,5.)),[512,512]))
    def test_query_wrong_frame_refused(self):
        self.capture('a');p=self.provider()
        with self.assertRaisesRegex(ValueError,'Wrong current capture query'):p(dict(self.query(),frame_index=6))


class MetadataGuards(unittest.TestCase):
    def test_strict_json_and_unsafe_paths(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
            with self.assertRaises(ValueError):m.loads(text)
        with tempfile.TemporaryDirectory() as td:
            b=m.BoundFiles(Path(td).resolve())
            for name in ('../outside','/absolute','x/../y'):
                with self.assertRaises(ValueError):b.path(name,'0'*64)
    def test_original_scope_rejected_before_reads(self):
        with self.assertRaisesRegex(ValueError,'Only original'):
            m.original_inputs('/tmp/other','0126','0'*64,{})
    def test_bad_freeze_hash_before_any_import(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td).resolve()/'freeze.json';path.write_text('{}')
            with patch.object(m.importlib,'import_module',side_effect=AssertionError('must not import')):
                with self.assertRaisesRegex(ValueError,'Bound input changed'):m.load_freeze(path,'0'*64)


class FullRunLifecycle(unittest.TestCase):
    def execute(self, root, corrupt=False):
        cfg,rows=generated();original=root/'original';original.mkdir();e=original/'0126';e.mkdir()
        (e/'clean').mkdir();(e/'shadow').mkdir();bundle=root/'bundle';bundle.mkdir()
        lines=[(json.dumps(b)+'\n').encode() for b,_,_ in rows]
        (e/'clean/frames.jsonl').write_bytes(b''.join(lines))
        (e/'shadow/shadow_trace.jsonl').write_text(''.join(json.dumps(dict(t,capture_scheduled=False,capture_files=[]))+'\n' for _,t,_ in rows))
        pins={n:m.sha(e/n) for n in ('clean/frames.jsonl','shadow/shadow_trace.jsonl')}
        refs=[r for _,_,r in rows]
        if corrupt:refs[3]['state']='0'*64
        frozen=dict(geometry=dict(path='unused',sha256='0'*64),batch_library=dict(path='unused',sha256='0'*64))
        modules=dict(MODULES,accuracy_v56_capture=SimpleNamespace(validate_rectangle=validate_rectangle),
            weak_continuation_auxiliary_v1=SimpleNamespace(AuxiliaryGapSupport=lambda *args:FakeAuxiliary()))
        @contextmanager
        def native(*args):
            yield KalmanTrackManager.update,SimpleNamespace(geometry=SimpleNamespace(fallbacks=[]),innovation_fallbacks=[]),SimpleNamespace(calls=0,fallbacks=[])
        destination=root/'result'
        with ExitStack() as stack:
            stack.enter_context(patch.object(m,'load_freeze',return_value=(frozen,m.BoundFiles(bundle))))
            stack.enter_context(patch.object(m,'original_inputs',return_value=(m.BoundFiles(original),m.BoundFiles(e),
                dict(files_sha256=pins),dict(configuration=asdict(cfg),fps=10),dict(baseline_digests=refs))))
            stack.enter_context(patch.object(m,'load_helpers',return_value=modules))
            stack.enter_context(patch.object(h,'verify_runtime',return_value={}))
            stack.enter_context(patch.object(h,'native_backend',native))
            stack.enter_context(patch.dict(m.SOURCES,{'0126':(len(rows),'0'*64)}))
            if corrupt:
                with self.assertRaisesRegex(ValueError,'Original output/private-state'):
                    m.run(m.ORIGINAL_ROOT,'0126','0'*64,bundle/'freeze.json','0'*64,destination)
            else:
                receipt=m.run(m.ORIGINAL_ROOT,'0126','0'*64,bundle/'freeze.json','0'*64,destination)
                self.assertTrue(receipt['passed']);self.assertEqual(receipt['processed_frames'],7)
                self.assertEqual(gzip.decompress((destination/'clean.jsonl.gz').read_bytes()),b''.join(lines))
                for name, info in receipt['artifact_uncompressed'].items():
                    raw=gzip.decompress((destination/name).read_bytes())
                    self.assertEqual(len(raw),info['bytes']);self.assertEqual(m.hashlib.sha256(raw).hexdigest(),info['sha256'])
                with self.assertRaisesRegex(ValueError,'Fresh isolated'):
                    m.run(m.ORIGINAL_ROOT,'0126','0'*64,bundle/'freeze.json','0'*64,destination)
        return destination
    def test_full_generated_run_raw_copy_complete_receipt_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as td:self.execute(Path(td).resolve())
    def test_original_digest_failure_retains_partial_receipt_without_retry(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.execute(Path(td).resolve(),corrupt=True)
            receipt=json.loads((p/'receipt.json').read_text())
            self.assertFalse(receipt['passed']);self.assertEqual(receipt['processed_frames'],3)
            self.assertEqual(receipt['error']['type'],'ValueError')
            self.assertEqual(len(gzip.decompress((p/'primary.jsonl.gz').read_bytes()).splitlines()),3)


if __name__=='__main__':unittest.main()
