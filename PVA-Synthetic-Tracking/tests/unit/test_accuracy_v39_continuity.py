"""Synthetic only: grace is bounded metadata, never a confirmed measurement."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v39_continuity import CausalMeasuredContinuity, ContinuityConfig


def track(qualified=True, measured=True, tid="bright:1", segment=0, ready=True):
    return dict(track_id=tid, segment=segment, qualified_moving=qualified,
        measured=measured, measurement_source_xy=[20., 30.] if measured else None,
        confirmation_timestamp_ns=0, excursion_px=20.,
        motion_quality=dict(ready=ready, passed=qualified and ready,
            quadratic_fit_rmse_px=(1. if qualified else 4.) if ready else None,
            maximum_rmse_px=3.))


def row(frame, tracks=None, timestamp=None, segment=0, reset=False):
    return dict(frame_index=frame, timestamp_ns=frame*100_000_000 if timestamp is None else timestamp,
                segment=segment, motion=dict(reset=reset), tracks=[] if tracks is None else tracks)


class ContinuityTests(unittest.TestCase):
    def test_two_measured_failures_then_expiry_without_self_refresh(self):
        p=CausalMeasuredContinuity()
        p.update(row(0,[track()]))
        for f in (1,2):
            d=p.update(row(f,[track(False)]))[(0,"bright:1")]
            self.assertTrue(d["renderable"])
            self.assertTrue(d["added_degraded_measurement"])
            self.assertFalse(d["confirmed_output"])
            self.assertEqual(d["anchor_frame"],0)
        d=p.update(row(3,[track(False)]))[(0,"bright:1")]
        self.assertFalse(d["renderable"])
        self.assertEqual(d["reason"],"grace_budget_expired")
        self.assertFalse(p.update(row(4,[track(False)]))[(0,"bright:1")]["renderable"])

    def test_time_budget_is_independent_of_frame_budget(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        self.assertFalse(p.update(row(1,[track(False)],timestamp=201_000_000))[(0,"bright:1")]["renderable"])

    def test_frame_budget_is_independent_of_elapsed_time(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        for f in (1,2):self.assertTrue(p.update(row(f,[track(False)],timestamp=f*10))[(0,"bright:1")]["renderable"])
        self.assertFalse(p.update(row(3,[track(False)],timestamp=30))[(0,"bright:1")]["renderable"])

    def test_no_grace_before_actual_qualification(self):
        p=CausalMeasuredContinuity()
        self.assertFalse(p.update(row(0,[track(False)]))[(0,"bright:1")]["renderable"])

    def test_coast_breaks_lineage_even_if_baseline_coast_is_qualified(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        d=p.update(row(1,[track(True,False)]))[(0,"bright:1")]
        self.assertTrue(d["renderable"]);self.assertFalse(d["measured"])
        self.assertEqual(d["status"],"baseline_qualified_prediction")
        self.assertIsNone(d["anchor_frame"])
        self.assertFalse(p.update(row(2,[track(False)]))[(0,"bright:1")]["renderable"])

    def test_new_good_measurement_reacquires_after_long_failure(self):
        p=CausalMeasuredContinuity()
        for f in range(7):p.update(row(f,[track(f in (0,6))]))
        d=p.update(row(7,[track(False)]))[(0,"bright:1")]
        self.assertTrue(d["added_degraded_measurement"]);self.assertEqual(d["anchor_frame"],6)

    def test_missing_id_breaks_lineage(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]));p.update(row(1))
        self.assertFalse(p.update(row(2,[track(False)]))[(0,"bright:1")]["renderable"])

    def test_reset_and_segment_change_break_lineage(self):
        for segment,reset in ((0,True),(1,False)):
            with self.subTest(segment=segment):
                p=CausalMeasuredContinuity();p.update(row(0,[track()]))
                d=p.update(row(1,[track(False,segment=segment)],segment=segment,reset=reset))[(segment,"bright:1")]
                self.assertFalse(d["renderable"])

    def test_unready_or_failed_other_conditions_are_not_graced(self):
        for name,value in (("confirmation_timestamp_ns",None),("excursion_px",11.),("excursion_px",float("nan"))):
            with self.subTest(name=name,value=value):
                p=CausalMeasuredContinuity();p.update(row(0,[track()]));t=track(False);t[name]=value
                self.assertFalse(p.update(row(1,[t]))[(0,"bright:1")]["renderable"])
                self.assertFalse(p.update(row(2,[track(False)]))[(0,"bright:1")]["renderable"])
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        self.assertFalse(p.update(row(1,[track(False,ready=False)]))[(0,"bright:1")]["renderable"])

    def test_different_identity_cannot_borrow_good_evidence(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        self.assertFalse(p.update(row(1,[track(False,tid="bright:2")]))[(0,"bright:2")]["renderable"])

    def test_all_baseline_states_preserved_and_input_unchanged(self):
        p=CausalMeasuredContinuity()
        original=row(0,[track(),track(True,False,tid="dark:2")]);before=copy.deepcopy(original)
        d=p.update(original)
        self.assertEqual(original,before)
        self.assertTrue(all(v["renderable"] and v["confirmed_output"] for v in d.values()))
        self.assertTrue(all(v["physical_class"]=="unknown" and not v["airborne_confirmed"] for v in d.values()))

    def test_no_prediction_is_added_by_grace(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        self.assertFalse(p.update(row(1,[track(False,False)]))[(0,"bright:1")]["renderable"])

    def test_bad_frame_does_not_mutate_state(self):
        p=CausalMeasuredContinuity();p.update(row(0,[track()]))
        for bad in (row(2,[track(False)]),row(1,[track(False)],timestamp=0),row(1,[track(False),track(False)])):
            with self.assertRaises(ValueError):p.update(bad)
        self.assertTrue(p.update(row(1,[track(False)]))[(0,"bright:1")]["added_degraded_measurement"])

    def test_bad_measurement_and_inconsistent_quality_rejected(self):
        for invalid in ([float("nan"),3.],[True,3.],None):
            p=CausalMeasuredContinuity();t=track();t["measurement_source_xy"]=invalid
            with self.assertRaises(ValueError):p.update(row(0,[t]))
        for mutation in (dict(passed=False),dict(quadratic_fit_rmse_px=None),dict(maximum_rmse_px=True),dict(ready=False)):
            p=CausalMeasuredContinuity();t=track();t["motion_quality"].update(mutation)
            with self.assertRaises(ValueError):p.update(row(0,[t]))

    def test_configuration_validation(self):
        for kwargs in (dict(maximum_gap_frames=True),dict(maximum_gap_frames=0),dict(maximum_gap_ns=0),dict(minimum_excursion_px=float("inf"))):
            with self.assertRaises(ValueError):ContinuityConfig(**kwargs)

    def test_interleaved_track_order_does_not_change_decisions(self):
        a,b=CausalMeasuredContinuity(),CausalMeasuredContinuity()
        for f in range(5):
            tracks=[track(f%3==0,tid="bright:1"),track(f%2==0,tid="dark:2")]
            self.assertEqual(a.update(row(f,tracks)),b.update(row(f,list(reversed(tracks)))))

    def test_alternating_failures_require_actual_new_good_evidence(self):
        p=CausalMeasuredContinuity()
        for f in range(20):
            d=p.update(row(f,[track(f%2==0)]))[(0,"bright:1")]
            self.assertEqual(d["anchor_frame"],f if f%2==0 else f-1)
            self.assertEqual(d["confirmed_output"],f%2==0)
            self.assertEqual(d["added_degraded_measurement"],f%2!=0)


if __name__=="__main__":unittest.main()
