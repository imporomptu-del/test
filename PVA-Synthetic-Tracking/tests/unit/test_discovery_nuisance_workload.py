import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_discovery_nuisance_workload.py"
spec = importlib.util.spec_from_file_location("nuisance_workload", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(index, ready=True):
    metric = dict(active_track_count=0, birth_count=0, deleted_track_count=0,
                  dropped_birth_count_at_active_track_cap=0, associated_candidate_count=0,
                  lifecycle_counts=dict(tentative=0, confirmed=0, coasted=0), max_active_tracks=256,
                  birth_admission={})
    return dict(frame_index=index, timestamp_ns=index*100_000_000, candidates=[], tracks=[],
                coverage=dict(full_shape_hw=[3190, 4784], detection_ready=ready,
                              unavailable_reason=None if ready else "warmup", dropped_at_tile_cap=0,
                              dropped_at_frame_cap=0),
                tracking_metrics=dict(bright=metric.copy(), dark=metric.copy()))


class AuditTests(unittest.TestCase):
    def test_quarter_boundaries_and_outside_are_descriptive(self):
        actual = audit.y_quarters([(0, 0), (1, 797.5), (1, 1595), (1, 2392.5), (1, 3190), (-1, 4)])
        self.assertEqual(actual, {"0": 1, "1": 1, "2": 1, "3": 1, "outside_image": 2})

    def test_fixed_windows_readiness_and_measured_prediction_separation(self):
        rows = [row(i, i != 50) for i in range(673)]
        t = dict(segment=0, track_id="dark:1", hits=5, qualified_moving=True,
                 measured=True, source_xy=[2, 2500], measurement_source_xy=[2, 2500])
        rows[51]["tracks"] = [t]
        rows[52]["tracks"] = [dict(t, measured=False)]
        rows[51]["candidates"] = [dict(x=2, y=2500, source_xy=[2, 2500], polarity="dark", score=4, noise_sigma_dn=.5)]
        rows[51]["tracking_metrics"]["dark"].update(active_track_count=256, birth_count=1,
                                                      dropped_birth_count_at_active_track_cap=3)
        rows[51]["tracking_metrics"]["dark"]["lifecycle_counts"] = dict(tentative=250, confirmed=6, coasted=0)
        actual = audit.analyze(rows)
        burst = actual["windows"]["burst_50_105"]
        self.assertEqual((burst["frames"], burst["ready_frames"]), (56, 55))
        self.assertEqual((burst["qualified_measured_records"], burst["qualified_predicted_records"]), (1, 1))
        self.assertEqual(burst["candidate_y_quarters"]["3"], 1)
        self.assertEqual(burst["dropped_births"], 3)
        self.assertEqual(burst["per_polarity_cap_saturated_frame_polarity_count"], 1)
        self.assertEqual(actual["windows"]["following_106_161"]["frames"], 56)

    def test_history_is_measurements_only_and_segment_specific(self):
        rows = [row(i) for i in range(673)]
        track = dict(segment=0, track_id="bright:1", hits=5, qualified_moving=True,
                     measured=True, source_xy=[20, 30], measurement_source_xy=[20, 30])
        for i in (50, 60):
            rows[i]["tracks"] = [track.copy()]
        rows[55]["tracks"] = [dict(track, measured=False)]
        rows[70]["tracks"] = [dict(track, segment=1)]
        window = audit.analyze(rows)["windows"]["burst_50_105"]
        self.assertEqual(window["qualified_measured_last_eight_measurements_span_frames_max"], 10)
        self.assertEqual(window["qualified_measured_last_eight_measurements_span_frames_median"], 0)
        self.assertEqual(window["qualified_identities"], 2)

    def test_strict_loader_hash_and_continuity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()/"frames.jsonl"
            rows = [row(i) for i in range(673)]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            self.assertEqual(len(audit.load_journal(path, audit.digest(path))), 673)
            expected = audit.digest(path)
            with patch.object(audit, "digest", side_effect=[expected, "0"*64]):
                with self.assertRaisesRegex(ValueError, "changed while"):
                    audit.load_journal(path, expected)
            with self.assertRaisesRegex(ValueError, "hash"):
                audit.load_journal(path, "0"*64)
            rows[10]["frame_index"] = 9
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "continuity"):
                audit.load_journal(path, audit.digest(path))

    def test_native_raw_measurements_not_reference_or_filtered_coordinates(self):
        rows = [row(i) for i in range(673)]
        rows[70]["candidates"] = [dict(x=20, y=50, source_xy=[20, 2500],
                                        polarity="bright", score=4, noise_sigma_dn=.5)]
        rows[70]["tracks"] = [dict(segment=0, track_id="bright:1", hits=5, qualified_moving=True,
                                   measured=True, source_xy=[20, 30], measurement_source_xy=[20, 1600])]
        result = audit.analyze(rows)["windows"]["burst_50_105"]
        self.assertEqual(result["candidate_y_quarters"], {"0": 0, "1": 0, "2": 0, "3": 1, "outside_image": 0})
        self.assertEqual(result["qualified_measured_y_quarters"], {"0": 0, "1": 0, "2": 1, "3": 0, "outside_image": 0})

    def test_json_rejects_duplicate_and_nonfinite(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                audit.decode(text)

    def test_save_serializes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()/"out.json"
            actual = audit.analyze([row(i) for i in range(673)])
            audit.save_new(path, actual)
            self.assertEqual(json.loads(path.read_text()), actual)
            with self.assertRaisesRegex(ValueError, "new absolute"):
                audit.save_new(path, actual)


if __name__ == "__main__":
    unittest.main()
