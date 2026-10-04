"""Generated artifact tests: no media, native libraries or producer imports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'scripts' if (REPO/'scripts').is_dir() else HERE.parent))
import audit_weak_continuation_shadow_v1 as audit

A, B, C = 'a'*64, 'b'*64, 'c'*64
COUNT = 6
CAPTURE_FRAME = 4


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def read(path):
    return json.loads(path.read_text())


def fixture(root):
    native = dict.fromkeys(audit.NATIVE_STATE_KEYS, A)
    baseline = [dict(frame_index=i, timestamp_ns=i*100000000, segment=0,
        candidates=[], tracks=[], source_to_reference=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        motion=dict(reset=False, pva_timings_ms={'x':1.}, warp_timings_ms={'x':1.},
                    motion_fit=dict(timing_ms={'x':1.})), coverage=dict(detection_ms=1.), timings_ms=dict(total=1.))
        for i in range(COUNT)]
    traced = [dict(frame_index=i, timestamp_ns=i*100000000, segment=0, capture_scheduled=(i==CAPTURE_FRAME),
        prior_forecasts=[], strong_proposals=[], records=[],
        metrics={}, capture_files=[], native_state_before=None, native_state_after=None) for i in range(COUNT)]
    mean=None;p=None;last_strong=None;previous=None;used=False
    identity='0/bright:1'
    for i,row in enumerate(traced):
        ts=i*100000000
        if mean is not None:
            f=np.eye(4);f[0,2]=f[1,3]=.1
            load=np.array([[.005,0],[0,.005],[.1,0],[0,.1]])
            mean=f@mean;p=f@p@f.T+3600*load@load.T;p=(p+p.T)/2
            forecast=dict(identity=identity,polarity='bright',track_id=1,segment=0,frame_index=i,timestamp_ns=ts,
                reference_xy=mean[:2].tolist(),predicted_mean=mean.tolist(),predicted_covariance=p.tolist(),
                innovation_covariance_2x2=(p[:2,:2]+np.eye(2)*4).tolist(),strong_measurement_covariance=(np.eye(2)*4).tolist(),
                prior_qualified=previous['qualified_moving'],prior_confirmation_timestamp_ns=previous['confirmation_timestamp_ns'],
                prior_associated_update_count=previous['hits'],prior_independent_confirmation_hits=previous['independent_hits'],
                prior_last_strong_timestamp_ns=last_strong,strong_age_seconds=(ts-last_strong)/1e9,
                weak_budget_available=not used,query_eligible=previous['qualified_moving'] and not used)
            row['prior_forecasts']=[forecast]
        note=dict(identity=identity,applied=False,is_ordinary_measurement=False,physical_identity_verified=False)
        if i<4:
            point=np.array([100.+4*i,100.]);last_strong=ts
            proposal=dict(x=point[0],y=point[1],score=5.,response_dn=2.5,polarity='bright')
            row['strong_proposals']=[proposal];baseline[i]['candidates']=[dict(proposal,source_xy=point.tolist())]
            if i==0:mean=np.array([*point,0.,0.]);p=np.diag([4.,4.,22500.,22500.])
            else:
                h=np.eye(4)[:2];r=np.eye(2)*4;k=np.linalg.solve((h@p@h.T+r).T,(p@h.T).T).T
                mean=mean+k@(point-h@mean);a=np.eye(4)-k@h;p=a@p@a.T+k@r@k.T;p=(p+p.T)/2
            note['status']='strong_measurement_priority'
        elif i==CAPTURE_FRAME:
            point=np.rint(mean[:2]);before_mean=mean.copy();before_p=p.copy()
            delta=point-mean[:2];d2=float(delta@np.linalg.inv(p[:2,:2]+np.eye(2)*4)@delta)
            peak=dict(descriptive_rank=1,reference_xy=point.tolist(),polarity='bright',score=3.,
                signed_centered_temporal_dn=1.5,temporal_threshold_dn=2.,
                original_temporal_threshold_pass=False,evidence_partition='weak_temporal',competing_prior_identity_gates=[],
                identity_assignment=None,acceptance_decision=None)
            observation=dict(schema='seaqr.weak-continuation-information.v1',focal_identity=identity,
                position_gate_px=45.,mahalanobis_squared_gate=25.,observed_peaks=[peak],observed_peak_count=1,
                weak_peak_count=1,original_threshold_peak_count=0,coverage_known=True,coverage_unknown_reasons=[])
            h=np.eye(4)[:2];r=np.eye(2)*8;k=np.linalg.solve((h@p@h.T+r).T,(p@h.T).T).T
            mean=mean+k@(point-h@mean);a=np.eye(4)-k@h;p=a@p@a.T+k@r@k.T;p=(p+p.T)/2
            note.update(status='weak_kinematic_correction',applied=True,observations=observation,
                measurement_reference_xy=point.tolist(),score=3.,strong_anchor_timestamp_ns=last_strong,
                mean_before=before_mean.tolist(),covariance_before=before_p.tolist(),mean_after=mean.tolist(),covariance_after=p.tolist())
            used=True
        else:note['status']='weak_budget_used_for_strong_gap'
        record=dict(track_id='bright:1',segment=0,measured=i<4,lifecycle='tentative' if i<3 else 'confirmed' if i==3 else 'coasted',
            qualified_moving=i>=3,source_xy=mean[:2].tolist(),reference_xy=mean[:2].tolist(),velocity_reference_xy_px_s=mean[2:].tolist(),
            measurement_source_xy=[100.+4*i,100.] if i<4 else None,measurement_score=5. if i<4 else None,
            hits=min(i+1,4),independent_hits=min(i+1,4),confirmation_timestamp_ns=300000000 if i>=3 else None,
            excursion_px=float(min(i,3)*4),motion_quality=None,learning_shape_reference_xy=None,weak_evidence=note)
        row['records']=[record];previous=record
        row['metrics']={'weak_continuation':dict(schema='seaqr.weak-continuation-shadow.v1',frame_index=i,
            prior_track_count=len(row['prior_forecasts']),prior_query_eligible_count=sum(p['query_eligible'] for p in row['prior_forecasts']),
            applied_count=int(note['applied']),decisions=[note],ordinary_measured_is_strong_only=True,
            detector_learning_feedback=False,strong_associations_recomputed_by_shadow=True,frozen_detector_stream=True)}
    cap = root/'shadow/captures/one.npz'
    cap.parent.mkdir(parents=True)
    cap.write_bytes(b'opaque generated NPZ; must not be decoded')
    metadata = dict(frame=CAPTURE_FRAME, segment=0, prelearning=True,
                    read_only_capture=True, production_selection_unchanged=True)
    write(cap.with_suffix('.json'), metadata)
    traced[CAPTURE_FRAME].update(native_state_before=native, native_state_after=deepcopy(native),
        capture_files=[dict(identity='0/bright:1',path='captures/one.npz',sha256=audit.file_sha(cap),
                           metadata_path='captures/one.json',metadata_sha256=audit.file_sha(cap.with_suffix('.json')))])
    lines(root/'shadow/shadow_trace.jsonl', traced)
    runtime=dict(before=dict(numpy='frozen',opencv_threads=12),after=dict(numpy='frozen',opencv_threads=2))
    for arm in ('clean','shadow'):
        arm_rows = deepcopy(baseline)
        if arm == 'shadow':
            for row in arm_rows:
                row['timings_ms']['total'] = 99.
                row['motion']['pva_timings_ms'] = {'other':77.}
                row['motion']['warp_timings_ms'] = {'other':66.}
                row['motion']['motion_fit']['timing_ms'] = {'other':55.}
                row['coverage']['detection_ms'] = 44.
        lines(root/arm/'frames.jsonl', arm_rows)
        write(root/(arm+'.shadow.json'),dict(schema=audit.RUN_SCHEMA,passed=True,error=None,arm=arm,
            clip='0029',processed_frames=COUNT,expected_frames=COUNT,production_changed=False,
            weak_learning_enabled=False,freeze_sha256=A,plan_sha256=B,source_sha256=C,
            trace_sha256=audit.file_sha(root/'shadow/shadow_trace.jsonl') if arm=='shadow' else None,
            baseline_digests=[dict(frame=i,output=A,state=B,learning=C) for i in range(COUNT)],
            runtime=deepcopy(runtime),capture_count=1 if arm=='shadow' else 0))
        launch=dict(source='/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0029.avi',source_sha256=C,
            config_sha256=audit.CONFIG_SHA,motion_config_sha256=audit.MOTION_SHA,expected_frames=COUNT,max_frames=None,
            fps=10.,annotations_supplied_to_detector=False,package_sha256={'x':A},code_sha256={'y':B},
            configuration={'tracking_peak_nms_radius_px':0.})
        write(root/arm/'launch.json',launch)
        write(root/arm/'report.json',dict(completed=True,full_clip=True,frames=COUNT,source_sha256=C,
            configuration=launch['configuration'],frame_decode=dict(decoded_frames=COUNT,consumed_frames=COUNT,
                dropped_frames=0,worker_joined=True,capture_released=True)))
        write(root/(arm+'.execution.json'),{'passed':True})
        write(root/(arm+'.v29.json'),dict(schema='seaqr.visible-combined-v29.v1',passed=True,error=None,clip='0029',
            arm='combined',frames=None,processed_frames=COUNT,execution_policy='serial_reference',config_sha256=audit.CONFIG_SHA,
            freeze_sha256='60b79d450672d131b517e9ed5a33fdeb40a6a2a9b29c6c0f584a9a8e93cc0dbc',
            **dict.fromkeys(('raw16_accessed','defaults_changed','production_approved','new_accuracy_validated',
                            'staged_v24_enabled','native_motion_v25_enabled'),False),
            comparison=dict(exact=True,frames=COUNT,reference_journal_sha256=audit.ARCHIVE_SHA['0029'],
                            journal_sha256=audit.file_sha(root/arm/'frames.jsonl'),execution_sha256=audit.file_sha(root/(arm+'.execution.json')))))
    return baseline,traced


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.baseline,self.trace = fixture(self.root)
        self.mapping = patch.dict(audit.SOURCES, {'0029':(COUNT,C)})
        self.mapping.start()
        self.windows = patch.dict(audit.WINDOWS, {'0029':((CAPTURE_FRAME,CAPTURE_FRAME),)})
        self.windows.start()

    def tearDown(self):
        self.mapping.stop(); self.windows.stop(); self.temp.cleanup()

    def run_audit(self):
        return audit.audit_run(self.root,'0029',A,B)

    def rebind_baseline(self,arm='shadow'):
        path=self.root/(arm+'.v29.json');value=read(path)
        value['comparison']['journal_sha256']=audit.file_sha(self.root/arm/'frames.jsonl');write(path,value)

    def save_trace(self):
        lines(self.root/'shadow/shadow_trace.jsonl',self.trace)
        p=self.root/'shadow.shadow.json';r=read(p)
        r['trace_sha256']=audit.file_sha(self.root/'shadow/shadow_trace.jsonl');write(p,r)

    def receipt_change(self, key, value, arm='shadow'):
        p=self.root/(arm+'.shadow.json');r=read(p);r[key]=value;write(p,r)

    def test_pass_full_scope_exact_baseline_timing_only(self):
        result=self.run_audit()
        self.assertTrue(result['passed']);self.assertEqual(result['frames'],COUNT)
        self.assertEqual(result['capture_files_verified'],1)
        self.assertFalse(result['native_arrays_interpreted'])

    def test_unknown_timing_named_field_is_not_excluded(self):
        rows=deepcopy(self.baseline);rows[1]['tracks']=[{'timing_ms':1}]
        lines(self.root/'shadow/frames.jsonl',rows)
        self.rebind_baseline()
        with self.assertRaisesRegex(ValueError,'non-timing'):self.run_audit()

    def test_boolean_integer_drift_is_not_equal(self):
        rows=deepcopy(self.baseline);rows[1]['motion']['reset']=0
        lines(self.root/'shadow/frames.jsonl',rows)
        self.rebind_baseline()
        with self.assertRaisesRegex(ValueError,'non-timing'):self.run_audit()

    def test_all_five_only_allowed_paths_are_excluded(self):
        raw={'motion':{'motion_fit':{'timing_ms':1},'warp_timings_ms':2,'pva_timings_ms':3},
             'coverage':{'detection_ms':4},'timings_ms':5,'extra':{'timing_ms':6}}
        self.assertEqual(audit.without_timing(raw),{'motion':{'motion_fit':{}},'coverage':{},'extra':{'timing_ms':6}})

    def test_blank_truncated_or_extra_rows_fail(self):
        path=self.root/'shadow/frames.jsonl';original=path.read_text()
        for changed in (original+'\n',original+original.splitlines()[0]+'\n','\n'.join(original.splitlines()[:-1])+'\n'):
            path.write_text(changed)
            with self.assertRaises(ValueError):self.run_audit()
        path.write_text(original)

    def test_frame_and_timestamp_contiguity(self):
        for key,value in (('frame_index',0),('timestamp_ns',1),('segment',False)):
            rows=deepcopy(self.baseline);rows[1][key]=value
            lines(self.root/'shadow/frames.jsonl',rows)
            with self.assertRaises(ValueError):self.run_audit()

    def test_private_state_learning_and_output_all_exact(self):
        p=self.root/'shadow.shadow.json';original=read(p)
        for key in ('state','learning','output'):
            value=deepcopy(original);value['baseline_digests'][1][key]='d'*64;write(p,value)
            with self.assertRaisesRegex(ValueError,'digests'):self.run_audit()
        write(p,original)

    def test_wrong_scope_or_failed_receipt_rejected(self):
        p=self.root/'shadow.shadow.json';original=read(p)
        for key,value in (('passed',False),('error','boom'),('arm','clean'),('clip','0126'),
                          ('processed_frames',2),('expected_frames',4),('production_changed',True),
                          ('weak_learning_enabled',True),('freeze_sha256',C),('plan_sha256',C),('source_sha256',B)):
            bad=deepcopy(original);bad[key]=value;write(p,bad)
            with self.assertRaises(ValueError,msg=key):self.run_audit()
        write(p,original)

    def test_native_state_change_rejected(self):
        self.trace[CAPTURE_FRAME]['native_state_after']['variance']=B;self.save_trace()
        with self.assertRaisesRegex(ValueError,'native detector state'):self.run_audit()

    def test_native_state_missing_or_wrong_keys_rejected(self):
        self.trace[CAPTURE_FRAME]['native_state_after']=None;self.save_trace()
        with self.assertRaisesRegex(ValueError,'native state'):self.run_audit()

    def test_unscheduled_capture_rejected(self):
        self.trace[CAPTURE_FRAME]['capture_scheduled']=False;self.save_trace()
        with self.assertRaisesRegex(ValueError,'window|schedule'):self.run_audit()

    def test_duplicate_or_foreign_capture_identity(self):
        self.trace[CAPTURE_FRAME]['capture_files'][0]['identity']='0/dark:2';self.save_trace()
        with self.assertRaisesRegex(ValueError,'captured forecast'):self.run_audit()

    def test_capture_binary_tamper_rejected(self):
        (self.root/'shadow/captures/one.npz').write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError,'Changed bound'):self.run_audit()

    def test_metadata_wrong_frame_or_not_passive(self):
        path=self.root/'shadow/captures/one.json';original=read(path)
        for key,value in (('frame',2),('segment',1),('prelearning',False),
                          ('read_only_capture',False),('production_selection_unchanged',False)):
            bad=deepcopy(original);bad[key]=value;write(path,bad)
            self.trace[CAPTURE_FRAME]['capture_files'][0]['metadata_sha256']=audit.file_sha(path);self.save_trace()
            with self.assertRaises(ValueError,msg=key):self.run_audit()

    def test_capture_traversal_and_symlink_refused(self):
        self.trace[CAPTURE_FRAME]['capture_files'][0]['path']='captures/../../clean/frames.jsonl';self.save_trace()
        with self.assertRaises(ValueError):self.run_audit()
        self.trace[CAPTURE_FRAME]['capture_files'][0]['path']='captures/link.npz'
        (self.root/'shadow/captures/link.npz').symlink_to(self.root/'shadow/captures/one.npz');self.save_trace()
        with self.assertRaisesRegex(ValueError,'Symlink'):self.run_audit()

    def test_capture_count_mismatch(self):
        self.receipt_change('capture_count',2)
        with self.assertRaisesRegex(ValueError,'count mismatch'):self.run_audit()

    def test_mutation_detected_on_final_rehash(self):
        ev=audit.Evidence(self.root);ev.path('clean/frames.jsonl')
        (self.root/'clean/frames.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):ev.unchanged()

    def test_duplicate_json_nonfinite_overflow_refused(self):
        for value in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e999}'):
            with self.assertRaises(ValueError):audit.loads(value)

    def test_runtime_changes_rejected(self):
        p=self.root/'shadow.shadow.json';value=read(p);value['runtime']['after']['numpy']='changed';write(p,value)
        with self.assertRaisesRegex(ValueError,'runtime'):self.run_audit()

    def test_caller_freeze_required(self):
        with self.assertRaises(ValueError):audit.audit_run(self.root,'0029','',B)
        with self.assertRaises(ValueError):audit.audit_run(self.root,'holdout',A,B)

    def test_unmatched_tentative_remains_tentative(self):
        checking=audit.ShadowStateAudit(0)
        checking.step(self.trace[0],self.baseline[0])
        row=deepcopy(self.trace[1]);old=deepcopy(self.trace[0]['records'][0]);forecast=row['prior_forecasts'][0]
        note=dict(identity='0/bright:1',applied=False,is_ordinary_measurement=False,
                  physical_identity_verified=False,status='not_prior_strong_qualified')
        old.update(measured=False,measurement_source_xy=None,measurement_score=None,
                   reference_xy=forecast['predicted_mean'][:2],velocity_reference_xy_px_s=forecast['predicted_mean'][2:],
                   weak_evidence=note,lifecycle='tentative')
        row.update(records=[old],strong_proposals=[])
        row['metrics']['weak_continuation']['decisions']=[note]
        baseline=deepcopy(self.baseline[1]);baseline['candidates']=[]
        checking.step(row,baseline)
        self.assertEqual(checking.records,2)

    def test_invalid_capture_status_is_integrity_failure(self):
        row=self.trace[CAPTURE_FRAME]
        note=row['records'][0]['weak_evidence'];note.update(status='invalid_capture_or_weak_covariance',applied=False)
        row['metrics']['weak_continuation']['decisions']=[deepcopy(note)];self.save_trace()
        with self.assertRaisesRegex(ValueError,'integrity failure'):self.run_audit()

    def test_weak_cannot_change_hits_measurement_or_learning(self):
        original=deepcopy(self.trace)
        for key,value in (('hits',5),('independent_hits',5),('confirmation_timestamp_ns',400000000),
                          ('measurement_source_xy',[1.,2.]),('learning_shape_reference_xy',[[1,2]])):
            self.trace=deepcopy(original);self.trace[CAPTURE_FRAME]['records'][0][key]=value;self.save_trace()
            with self.assertRaises(ValueError,msg=key):self.run_audit()

    def test_weak_covariance_and_forecast_math_checked(self):
        original=deepcopy(self.trace)
        self.trace[CAPTURE_FRAME]['records'][0]['weak_evidence']['covariance_after'][0][0]+=.01
        self.trace[CAPTURE_FRAME]['metrics']['weak_continuation']['decisions']=deepcopy([self.trace[CAPTURE_FRAME]['records'][0]['weak_evidence']])
        self.save_trace()
        with self.assertRaisesRegex(ValueError,'Joseph covariance'):self.run_audit()
        self.trace=deepcopy(original);self.trace[2]['prior_forecasts'][0]['predicted_mean'][0]+=.01;self.save_trace()
        with self.assertRaisesRegex(ValueError,'forecast mean'):self.run_audit()

    def test_archive_comparison_required(self):
        p=self.root/'shadow.v29.json';value=read(p);value['comparison']['exact']=False;write(p,value)
        with self.assertRaisesRegex(ValueError,'archive comparison'):self.run_audit()

    def test_partition_uses_native_dn_predicate_not_rounded_score(self):
        row=deepcopy(self.trace[CAPTURE_FRAME]);prior=row['prior_forecasts'][0]
        obs=row['records'][0]['weak_evidence']['observations']
        obs['observed_peaks'][0]['score']=4.0
        # Division rounding alone does not turn a subthreshold numerator into a strong peak.
        audit.ShadowStateAudit(0).check_observations(obs,prior,{prior['identity']:prior})


if __name__=='__main__':
    unittest.main()
