import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from combined_v29_protocol import arm_flags, schedule, full_schedule, validate_scope, speed_gate
from combined_v29_state import StateAudit, state_of
from replay_tracking_v27 import digest


def receipts():
    rows = {}
    for spec in schedule():
        if spec['state_audit']:
            continue
        ratio = dict(v20=1., v26=1.25, v28=1.1, combined=1.5)[spec['arm']]
        duration = 256./ratio
        rows[spec['name']] = dict(**spec, passed=True, error=None, processed_frames=128,
            wall_s=128/ratio, fps=ratio, consumer_frame_ms=[duration]*128,
            execution=dict(frames=[dict(request_ns=1, ready_ns=1000001,
                consumer_complete_ns=int(duration*1e6)+2000001) for _ in range(128)]))
    return rows


class CombinedProtocolTests(unittest.TestCase):
    def test_four_arms_are_explicit(self):
        self.assertEqual([arm_flags(a) for a in ('v20', 'v26', 'v28', 'combined')],
                         [(False, False), (True, False), (False, True), (True, True)])
        with self.assertRaises(ValueError): arm_flags('automatic')

    def test_schedule_and_allowed_media_scope(self):
        self.assertEqual(len(schedule()), 26)
        self.assertEqual(len(full_schedule()), 4)
        self.assertEqual(len({s['name'] for s in schedule()+full_schedule()}), 30)
        for spec in schedule()+full_schedule():
            validate_scope(spec['clip'], spec['arm'], spec['frames'], spec['state_audit'])
        for values in (('0001','combined',128,False), ('0029','v20',None,False),
                       ('0126','v20',128,True), ('0126','combined',128.,False),
                       ('0126','combined',True,False), ('0126','combined',128,1)):
            with self.assertRaises(ValueError): validate_scope(*values)

    def test_nominal_gate_and_no_partial_sample(self):
        rows = receipts()
        self.assertTrue(speed_gate(rows)['passed'])
        rows.pop(next(iter(rows)))
        with self.assertRaises(ValueError): speed_gate(rows)

    def test_timing_validation(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            rows = receipts(); rows['0126_repeat0_v20']['wall_s'] = value
            with self.assertRaises(ValueError): speed_gate(rows)
        for key, value in (('state_audit', True), ('processed_frames', 127), ('fps', 123.)):
            rows = receipts(); rows['0126_repeat0_v20'][key] = value
            with self.assertRaises(ValueError): speed_gate(rows)

    def test_threshold_and_each_pair_required(self):
        rows = receipts()
        for r in rows.values():
            if r['arm'] == 'combined': r.update(fps=1.19, wall_s=128/1.19)
        self.assertFalse(speed_gate(rows)['passed'])
        rows = receipts(); rows['0126_repeat0_combined'].update(fps=.99, wall_s=128/.99)
        self.assertFalse(speed_gate(rows)['passed'])

    def test_latency_and_single_component_regression(self):
        rows = receipts()
        for i in range(3): rows[f'0126_repeat{i}_combined']['consumer_frame_ms'] = [300.]*128
        self.assertFalse(speed_gate(rows)['passed'])
        rows = receipts()
        for i in range(3): rows[f'0126_repeat{i}_v26'].update(fps=1.6, wall_s=128/1.6)
        self.assertFalse(speed_gate(rows)['passed'])

    def test_smoke_state_and_actual_learning_input(self):
        from tiny_target.visible_baseline import VisibleTracks, VisiblePointDetector
        tracker = SimpleNamespace(managers={'x':SimpleNamespace(a=1)}, extents={}, qualified=set(),
            summary={}, ever_qualified=set(), previous_records=[], previous_timestamp_ns=0, quality={})
        learning = [[1., 2.]]
        outputs = ([dict(track_id='x')], {'count':1})
        expected = [dict(frame=0, output=digest(list(outputs)), state=digest(state_of(tracker)), learning=digest(learning))]
        audit = StateAudit(expected)
        with ExitStack() as stack:
            stack.enter_context(patch.object(VisiblePointDetector, 'update', lambda *a: ([], {})))
            stack.enter_context(patch.object(VisibleTracks, 'update', lambda *a: outputs))
            audit.install(stack)
            VisiblePointDetector.update(None, None, None, 0, learning)
            result = VisibleTracks.update(tracker, [], 0, 0, 0, None, (1,1))
            self.assertEqual(result, outputs)
            audit.finish(1)
        with self.assertRaises(AssertionError): StateAudit(expected).finish(1)

    def test_missing_latency_rejected(self):
        rows = receipts(); rows['0126_repeat0_combined']['consumer_frame_ms'].pop()
        with self.assertRaises(ValueError): speed_gate(rows)


if __name__ == '__main__':
    unittest.main()
