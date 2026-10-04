import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/run_weak_continuation_information_v1.py'
spec = importlib.util.spec_from_file_location('weak_runner', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = []
    for i in range(3):
        t = dict(segment=0, track_id='bright:1', reference_xy=[10.,20.], velocity_reference_xy_px_s=[0.,0.],
            measured=i!=1, measurement_source_xy=[10.,20.] if i!=1 else None, hits=1 if i<2 else 2,
            qualified_moving=True, lifecycle='confirmed' if i!=1 else 'coasted', independent_hits=4,
            confirmation_timestamp_ns=0)
        rows.append(dict(frame_index=i, timestamp_ns=i*100000000, segment=0,
            motion=dict(reset=False, accepted=True), source_to_reference=np.eye(3).tolist(), tracks=[t],
            tracking_metrics={'bright':{'association_audit': [] if i!=2 else [dict(track_id=1,
                gaussian_twice_nll_without_constant=2*math.log(908.9))]}}))
    return rows


def evaluate(rows):
    with tempfile.TemporaryDirectory() as d, patch.object(m, 'FRAMES', (1,2)), patch.object(m, 'FOCAL', '0/bright:1'):
        p=Path(d)/'journal.jsonl'
        p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        return m.derive_forecasts(p)


class ForecastTests(unittest.TestCase):
    def test_prediction_and_coast_covariance(self):
        f,d=evaluate(fixture())
        self.assertEqual(len(f),2)
        for r,variance in zip(f,(233.09,908.9)):
            t=r['forecasts'][0]
            self.assertEqual(t['reference_xy'],[10.,20.])
            np.testing.assert_allclose(t['innovation_covariance_2x2'],np.eye(2)*variance,rtol=0,atol=1e-12)
        self.assertEqual(d['posterior_states_checked'],2)
        self.assertFalse(f[1]['forecasts'][0]['previous_measured'])

    def test_current_measurement_does_not_change_current_forecast(self):
        original=fixture(); modified=copy.deepcopy(original)
        p=m.covariance_predict(m.covariance_predict(np.diag([4.,4.,22500.,22500.]),.1),.1)
        gain,_=m.covariance_correct(p)
        mean=np.array([10.,20.,0.,0.])+gain@np.array([1.,0.])
        t=modified[2]['tracks'][0]
        t['reference_xy']=mean[:2].tolist();t['velocity_reference_xy_px_s']=mean[2:].tolist();t['measurement_source_xy']=[11.,20.]
        modified[2]['tracking_metrics']['bright']['association_audit'][0]['gaussian_twice_nll_without_constant']+=1/908.9
        self.assertEqual(evaluate(original)[0],evaluate(modified)[0])

    def test_incomplete_birth_history_rejected(self):
        rows=fixture();rows[0]['tracks'][0]['hits']=50
        with self.assertRaisesRegex(ValueError,'incomplete birth'):
            evaluate(rows)

    def test_inconsistent_posterior_rejected(self):
        rows=fixture();rows[1]['tracks'][0]['reference_xy'][0]+=1
        with self.assertRaisesRegex(ValueError,'posterior mismatch'):
            evaluate(rows)

    def test_inconsistent_likelihood_rejected(self):
        rows=fixture();rows[2]['tracking_metrics']['bright']['association_audit'][0]['gaussian_twice_nll_without_constant']+=1
        with self.assertRaisesRegex(ValueError,'Gaussian cost'):
            evaluate(rows)

    def test_segment_change_without_reset_rejected(self):
        rows=fixture();rows[1]['segment']=1
        with self.assertRaisesRegex(ValueError,'segment without reset'):
            evaluate(rows)

    def test_capture_reset_is_not_a_valid_continuation(self):
        rows=fixture();rows[1]['motion']['reset']=True
        with self.assertRaisesRegex(ValueError,'geometry not accepted'):
            evaluate(rows)

    def test_duplicate_track_rejected(self):
        rows=fixture();rows[0]['tracks'].append(copy.deepcopy(rows[0]['tracks'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate track'):
            evaluate(rows)

    def test_missing_past_qualification_not_promoted_from_current(self):
        rows=fixture();rows[0]['tracks'][0]['qualified_moving']=False
        with self.assertRaisesRegex(ValueError,'past qualification'):
            evaluate(rows)

    def test_covariance_is_symmetric_positive_and_not_mutated(self):
        p=np.diag([4.,4.,22500.,22500.]);before=p.copy()
        predicted=m.covariance_predict(p,.1);gain,corrected=m.covariance_correct(predicted)
        np.testing.assert_array_equal(p,before)
        np.testing.assert_array_equal(corrected,corrected.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(corrected)>0))
        self.assertAlmostEqual(gain[0,0],229.09/233.09)

    def test_bad_dt(self):
        for dt in (-.1,0,1.1):
            with self.assertRaises(ValueError):m.transition(dt)

    def test_existing_output_never_opens_inputs(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m,'build_plan') as builder:
            with self.assertRaisesRegex(ValueError,'fresh output'):
                m.run(Path(d)/'absent.json',Path(d))
            builder.assert_not_called()

    def test_changed_plan_refuses_output(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m,'build_plan',return_value={'x':2}):
            root=Path(d).resolve()
            p=root/'plan.json';m.write(p,{'x':1})
            with self.assertRaisesRegex(ValueError,'frozen plan'):
                m.run(p,root/'run')
            self.assertFalse((root/'run').exists())

    def test_plan_mutation_during_read_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();p=root/'plan.json';m.write(p,{'x':1})
            def changed(path):
                path.write_text('{"x":2}')
                return {'x':1}
            with patch.object(m,'read',side_effect=changed), patch.object(m,'build_plan') as builder:
                with self.assertRaisesRegex(ValueError,'changed while reading'):
                    m.run(p,root/'run')
                builder.assert_not_called()
            self.assertFalse((root/'run').exists())

    def test_strict_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
                p.write_text(value)
                with self.assertRaises(ValueError):m.read(p)
            with self.assertRaises(FileExistsError):m.write(p,{})


if __name__=='__main__':unittest.main()
