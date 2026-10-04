"""Generated metadata only; no archived outcomes, media, or tracker imports."""
import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name('audit_weak_auxiliary_replay_v1.py')
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2]/'scripts/audit_weak_auxiliary_replay_v1.py'
spec = importlib.util.spec_from_file_location('auxiliary_independent_audit', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NumericalTests(unittest.TestCase):
    def test_joseph_independent_closed_form(self):
        mean = np.array([2., 3., .5, -.25])
        p = np.diag([4., 4., 9., 9.])
        point = np.array([8., -3.])
        actual_mean, actual_p = m.correct(mean,p,point,np.eye(2)*4)
        np.testing.assert_allclose(actual_mean, [4., 1., .5, -.25], rtol=0,atol=1e-14)
        np.testing.assert_allclose(actual_p, np.diag([8/3,8/3,9,9]),rtol=0,atol=1e-14)

    def test_prediction_uses_owned_velocity_and_process_noise(self):
        mean = np.array([2.,3.,4.,-5.]); p=np.eye(4)
        actual_mean, actual_p=m.predict(mean,p,.1)
        np.testing.assert_allclose(actual_mean,[2.4,2.5,4.,-5.],rtol=0,atol=1e-14)
        self.assertAlmostEqual(actual_p[0,0],1.1)
        self.assertAlmostEqual(actual_p[0,2],1.9)
        self.assertAlmostEqual(actual_p[2,2],37.)
        np.testing.assert_array_equal(p,np.eye(4))

    def test_covariance_and_boolean_numeric_rejected(self):
        for value in ([[1,0],[1,1]], [[1,0],[0,-1]], [[True,0],[0,1]]):
            with self.assertRaises(ValueError):m.covariance(value,2,'test')
        with self.assertRaises(ValueError):m.vector([True,0.],2,'query')
        with self.assertRaises(ValueError):m.predict(np.zeros(4),np.eye(4),-.1)

    def test_tolerance_only_numerical_not_primary_exact(self):
        self.assertLess(m.close([1.],[1.+5e-8],'aux'),m.ATOL)
        with self.assertRaises(ValueError):m.close([1.],[1.+2e-7],'aux')
        with self.assertRaises(ValueError):m.exact([1.],[1.+1e-12],'primary')
        with self.assertRaises(ValueError):m.exact(True,1,'typed flag')

    def test_primary_gate_full_bounds_not_identity(self):
        self.assertTrue(m.full_disk_in_bounds([50.,50.],[5,5,96,96]))
        self.assertFalse(m.full_disk_in_bounds([50.01,50.],[5,5,96,96]))
        self.assertFalse(m.full_disk_in_bounds([50.,50.],[6,5,96,96]))


class SerializedStateTests(unittest.TestCase):
    def test_independent_tagged_decode_and_digest(self):
        a=np.array([1.,2.],np.float64)
        value=['dict', [['a',['array',a.dtype.str,[2],a.tobytes().hex()]],
            ['b',['dataclass','Generated',[['number',['float',(1.5).hex()]],['ids',['set',[2,3]]]]]]]]
        self.assertEqual(m.decoded_normalized(value),dict(a=[1.,2.],b=dict(number=1.5,ids=[2,3])))
        expected=hashlib.sha256(json.dumps(value,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(m.normalized_sha(value),expected)

    def test_invalid_or_ambiguous_tagged_state_rejected(self):
        bad=[['float','inf'], ['float','nonsense'], ['dict',[['x',1],['x',2]]], ['array','O',[1],'00'*8],
            ['array','<f8',[2],'00'*8], ['array','<f8',[True],'00'*8], ['unknown',[]]]
        for value in bad:
            with self.subTest(value=value),self.assertRaises((ValueError,TypeError)):
                m.decoded_normalized(value)

    def test_old_private_state_nonfinite_sentinel_is_opaque_not_a_numeric_measurement(self):
        tag=['dataclass','_RunningMoments',[['count',0],['total',['float',0.0.hex()]],
            ['total_squared',['float',0.0.hex()]],['minimum',['float','inf']],['maximum',['float','-inf']]]]
        sentinel=m.decoded_normalized(tag)['minimum']
        self.assertEqual(sentinel,{'opaque_empty_running_minimum':'inf'})
        with self.assertRaises(ValueError):m.vector([sentinel,0],2,'measurement')
        tag[2][0][1]=1
        with self.assertRaises(ValueError):m.decoded_normalized(tag)

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}','{"a":1e999}','{"a":NaN}'):
            with self.assertRaises(ValueError):m.loads(text)


def generated_sequence(missing=False):
    """Actual new producer/core over an entirely generated strong stream."""
    sys.path.insert(0,str(SCRIPT.parent))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run_weak_auxiliary_replay_v1 as runner
    import replay_weak_divergence_v1 as historical
    import replay_tracking_v27 as digest
    import combined_v29_state as state
    from tiny_target.visible_baseline import VisibleConfig,VisibleTracks
    from weak_continuation_auxiliary_v1 import AuxiliaryGapSupport
    from weak_continuation_information_v1 import enumerate_peaks
    from test_weak_continuation_shadow_v1 import capture,point
    cfg=VisibleConfig(confirmation_hits=2,minimum_moving_excursion_px=1,motion_quality_enabled=False)
    original=VisibleTracks(cfg,10);rows=[]
    for frame in range(13):
        proposals=[point({0:80,1:82,4:88}[frame])] if frame in (0,1,4) else []
        ts=frame*100000000;learning=original.learning_centers(ts,0)
        records,metrics=original.update(deepcopy(proposals),frame,ts,0,np.eye(3),(192,192))
        b=dict(frame_index=frame,timestamp_ns=ts,segment=0,candidates=proposals,
            source_to_reference=np.eye(3).tolist(),coverage=dict(full_shape_hw=[192,192]),
            tracks=deepcopy(records),tracking_metrics=deepcopy(metrics))
        t=dict(frame_index=frame,timestamp_ns=ts,segment=0,strong_proposals=deepcopy(proposals))
        ref=dict(frame=frame,output=digest.digest([records,metrics]),state=digest.digest(state.state_of(original)),
            learning=digest.digest(learning))
        rows.append((b,t,ref))
    class Provider:
        enabled=True
        def __init__(self):self.calls=[]
        def __call__(self,prior):
            self.calls.append(dict(query_identity=prior['identity'],forecast=deepcopy(prior)))
            return None if missing else capture(prior)
    def check(call,prior,priors,shape):
        m.exact(call['forecast'],prior,'generated capture query')
        m.require(call['query_identity']==prior['identity'],'generated capture owner')
        if missing:return None
        cap=capture(prior)
        return enumerate_peaks(cap['values'],cap['flags'],cap['metadata'],prior,priors)
    primary=VisibleTracks(cfg,10);aux=AuxiliaryGapSupport(cfg,10);result=[]
    modules={'replay_weak_divergence_v1':historical,'replay_tracking_v27':digest,'combined_v29_state':state}
    for b,t,ref in rows:
        p,a=runner.replay_frame(primary,aux,b,t,ref,Provider(),modules)
        result.append((p,a,b,ref))
    return asdict(cfg),result,check


class GeneratedCausalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.rows,check=generated_sequence()
        cls.check=staticmethod(check)

    def verify(self,rows=None):
        primary=m.PrimaryAudit(self.config);aux=m.AuxiliaryAudit(self.config)
        for p,a,b,ref in self.rows if rows is None else rows:
            primary.step(p,a,b,ref);aux.step(a,b,self.check)
        return primary,aux

    def test_current_producer_schema_prediction_reset_and_deletion(self):
        primary,aux=self.verify()
        self.assertEqual(primary.frames,13)
        self.assertEqual(aux.counts['current_weak_observation_count'],2)
        self.assertGreater(aux.counts['prediction_from_weak_count'],0)
        self.assertEqual(aux.previous['states'],{})
        self.assertLessEqual(aux.max_error,m.ATOL)

    def test_primary_output_hit_or_digest_tamper_rejected(self):
        for target in ('records','digest','state'):
            rows=deepcopy(self.rows);p,a,b,ref=rows[2]
            if target=='records':p['records'][0]['hits']+=1
            elif target=='digest':p['actual_learning_sha256']='0'*64
            else:p['primary_state_sha256_after_aux']='0'*64
            with self.subTest(target=target),self.assertRaises(ValueError):self.verify(rows)

    def test_auxiliary_cannot_seed_primary_query_or_omit_competitor(self):
        for target in ('center','omitted'):
            rows=deepcopy(self.rows);a=rows[3][1]
            if target=='center':a['primary_forecasts'][0]['predicted_mean'][0]+=1
            else:a['primary_forecasts']=[]
            with self.subTest(target=target),self.assertRaises(ValueError):self.verify(rows)

    def test_wrong_covariance_or_origin_or_measurement_credit_rejected(self):
        for key in ('covariance_after','origin_frame_index','ordinary_measurement'):
            rows=deepcopy(self.rows);record=rows[2][1]['auxiliary_records'][0]
            if key=='covariance_after':record[key][0][0]+=.01
            elif key=='origin_frame_index':record[key]+=1
            else:record[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify(rows)

    def test_second_weak_query_in_same_gap_rejected(self):
        rows=deepcopy(self.rows);a=rows[3][1]
        a['auxiliary_queries']=deepcopy(a['primary_forecasts'])
        with self.assertRaisesRegex(ValueError,'query set'):self.verify(rows)

    def test_counts_and_used_budget_are_reconstructed(self):
        for target in ('count','budget'):
            rows=deepcopy(self.rows);a=rows[2][1]
            if target=='count':a['auxiliary_metrics']['current_weak_observation_count']+=1
            else:a['aux_state_after']['used_strong_gaps']={}
            with self.subTest(target=target),self.assertRaises(ValueError):self.verify(rows)

    def test_missing_capture_keeps_explicit_unknown_and_zero_corrections(self):
        config,rows,check=generated_sequence(missing=True)
        primary=m.PrimaryAudit(config);aux=m.AuxiliaryAudit(config)
        for p,a,b,ref in rows:
            primary.step(p,a,b,ref);aux.step(a,b,check)
        self.assertEqual(aux.counts['current_weak_observation_count'],0)
        self.assertGreater(aux.statuses['missing_capture'],0)


class FileGuards(unittest.TestCase):
    def test_source_media_refused_and_changed_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();path=root/'metadata.json';path.write_text('{"x":1}')
            b=m.Bindings();b.read(path,m.sha(path));path.write_text('{"x":2}')
            with self.assertRaisesRegex(ValueError,'Changed bound'):b.unchanged()
            media=root/'source.avi';media.write_bytes(b'not-media')
            with self.assertRaisesRegex(ValueError,'metadata/array/code'):b.bind(media,m.sha(media))

    def test_symlink_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'a.json';p.write_text('{}');(root/'b.json').symlink_to(p)
            for value in ('../a.json','/a.json','x/../a.json','b.json'):
                with self.subTest(value=value),self.assertRaises(ValueError):m.scoped_path(root,value)

    def test_changed_freeze_rejected_before_enumerator_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();freeze=root/'freeze.json';freeze.write_text('{}')
            with patch.object(m.importlib.util,'spec_from_file_location',side_effect=AssertionError('must not import')):
                with self.assertRaisesRegex(ValueError,'Changed bound'):m.load_contract(root,freeze,'0'*64)


if __name__=='__main__':unittest.main()
