import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("nuisance_origin_packet_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/nuisance_origin_packet_v1.py"
spec = importlib.util.spec_from_file_location("nuisance_origin_packet", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def candidate(x=300, y=300, score=6.0, polarity="bright", peak=None):
    return dict(x=x, y=y, source_xy=[x, y], score=score, polarity=polarity,
                response_dn=3.0, noise_sigma_dn=.5,
                shape={"peak_reference_xy": peak or [x, y]})


def fixture():
    rows = [dict(frame_index=i, segment=0, source_to_reference=np.eye(3).tolist(), candidates=[])
            for i in range(m.FRAMES)]
    for f in m.BURST_ANCHORS:
        rows[f]["candidates"] = [candidate(), candidate(600, 600, polarity="dark")]
    for f in m.TARGET_ANCHORS:
        rows[f]["candidates"] = [candidate(700, 700, polarity="dark")]
    refs = [dict(frame_index=f, measurement_source_xy=[700, 700]) for f in m.TARGET_ANCHORS]
    return rows, refs


def plan_from(rows, refs):
    episodes, groups = m.select_episodes(rows, refs)
    return dict(schema=m.SCHEMA, clip="0240", patch_size=17, source_shape_hw=[m.HEIGHT, m.WIDTH],
                source_frames=673, input_sha256=dict(journal=m.JOURNAL_SHA, source=m.SOURCE_SHA,
                manifest=m.MANIFEST_SHA), episodes=episodes, frame_points=groups)


class PacketTests(unittest.TestCase):
    def test_inventory_and_fixed_centers(self):
        p = m.validate_plan(plan_from(*fixture()))
        self.assertEqual(len(p["episodes"]), 8)
        self.assertEqual(len(p["frame_points"]), 25)
        self.assertEqual(sum(map(len, p["frame_points"].values())), 80)
        e = p["episodes"][0]
        self.assertEqual(e["frames"], [58, 59, 60, 61, 62])
        self.assertEqual(e["comparator_reference_xy"], [348, 300])

    def test_score_ties_y_x_then_index(self):
        rows, refs = fixture()
        rows[60]["candidates"] += [candidate(100, 200), candidate(90, 200), candidate(90, 200)]
        e = m.select_episodes(rows, refs)[0][0]
        self.assertEqual(e["candidate_index"], 3)
        self.assertEqual(e["sample_reference_xy"], [90, 200])

    def test_peak_not_centroid(self):
        rows, refs = fixture()
        rows[60]["candidates"][0] = candidate(310, 320, peak=[300, 300])
        e, g = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["candidate_centroid_reference_xy"], [310, 320])
        self.assertEqual(g["60"][0]["reference_xy"], [300, 300])

    def test_unshaped_peak_uses_xy(self):
        c = candidate()
        c["shape"] = None
        self.assertEqual(m.peak(c), [300, 300])

    def test_transform_maps_reference_back_to_native(self):
        row = dict(source_to_reference=[[2, 0, 10], [0, 2, -6], [0, 0, 1]])
        np.testing.assert_equal(m.source_xy(row, [30, 14]), [10, 10])
        row["source_to_reference"] = [[1, 0, 2], [0, 1, 3], [0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "singular"):
            m.source_xy(row, [10, 10])

    def test_comparator_checks_all_five_frames_and_inclusive_radius(self):
        rows, refs = fixture()
        rows[58]["candidates"] = [candidate(365, 300)]  # Exactly17px from +48 control.
        e, _ = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["comparator_reference_xy"], [252, 300])

    def test_explicit_missing_comparator(self):
        rows, refs = fixture()
        rows[59]["candidates"] = [candidate(300 + dx, 300 + dy) for dx, dy in m.OFFSETS]
        e, g = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["comparator_status"], "no_candidate_free_offset_with_full_support")
        self.assertFalse(any(p["point_id"] == "burst_060_bright/comparator" for p in g["60"]))

    def test_topscore_edge_not_silently_replaced(self):
        rows, refs = fixture()
        rows[60]["candidates"].append(candidate(3, 3, score=100))
        e, _ = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["sample_status"], "insufficient_patch_support")
        self.assertEqual(e[0]["candidate_score"], 100)

    def test_source_edge_checked_across_history(self):
        rows, refs = fixture()
        rows[58]["source_to_reference"][0][2] = 295
        e, _ = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["sample_status"], "insufficient_patch_support")

    def test_target_nearest_dark_inclusive_radius(self):
        rows, refs = fixture()
        rows[435]["candidates"] = [candidate(700, 700, polarity="bright"),
            candidate(708, 700, score=7, polarity="dark"), candidate(709, 700, score=100, polarity="dark")]
        e, _ = m.select_episodes(rows, refs)
        self.assertEqual(e[6]["candidate_index"], 1)
        self.assertEqual(e[6]["sample_reference_xy"], [708, 700])

    def test_target_nearest_beats_score(self):
        rows, refs = fixture()
        rows[435]["candidates"] += [candidate(701, 700, score=100, polarity="dark")]
        self.assertEqual(m.select_episodes(rows, refs)[0][6]["candidate_index"], 0)

    def test_missing_candidate_explicit(self):
        rows, refs = fixture()
        rows[60]["candidates"] = []
        e, _ = m.select_episodes(rows, refs)
        self.assertEqual(e[0]["sample_status"], "missing_candidate")
        self.assertEqual(e[1]["sample_status"], "missing_candidate")

    def test_cross_reset_rejected(self):
        rows, refs = fixture()
        rows[61]["segment"] = 1
        with self.assertRaisesRegex(ValueError, "reset"):
            m.select_episodes(rows, refs)

    def test_validate_rejects_added_frames_or_points(self):
        p = plan_from(*fixture())
        p["frame_points"]["400"] = []
        with self.assertRaisesRegex(ValueError, "frame inventory"):
            m.validate_plan(p)
        p = plan_from(*fixture())
        p["frame_points"]["60"][0]["reference_xy"] = [8, 8]
        with self.assertRaisesRegex(ValueError, "point inventory"):
            m.validate_plan(p)

    def test_validate_hash_scope(self):
        p = plan_from(*fixture())
        p["clip"] = "0170"
        with self.assertRaises(ValueError):
            m.validate_plan(p)
        p = plan_from(*fixture())
        p["input_sha256"]["source"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hashes"):
            m.validate_plan(p)

    def test_no_absence_truth_or_offset_relabeling(self):
        p = plan_from(*fixture())
        p["episodes"][0]["label"] = "negative"
        with self.assertRaisesRegex(ValueError, "labels"):
            m.validate_plan(p)
        p = plan_from(*fixture())
        p["episodes"][0]["comparator_reference_xy"] = [600, 300]
        with self.assertRaisesRegex(ValueError, "offset"):
            m.validate_plan(p)

    def test_renderer_refuses_changed_plan_before_decode(self):
        p = plan_from(*fixture())
        p["input_paths"] = dict(journal=str(m.JOURNAL_PATH), manifest=str(m.MANIFEST_PATH), source=str(m.SOURCE_PATH))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            m.write_json(path, p)
            with patch.object(m, "build_plan", return_value=dict(p, changed=True)), patch("cv2.VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "deterministic"):
                    m.render(path, Path(temp) / "output")
                decoder.assert_not_called()

    def test_renderer_refuses_existing_destination_before_decode(self):
        p = plan_from(*fixture())
        p["input_paths"] = dict(journal=str(m.JOURNAL_PATH), manifest=str(m.MANIFEST_PATH), source=str(m.SOURCE_PATH))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            m.write_json(path, p)
            with patch.object(m, "build_plan", return_value=p), patch("cv2.VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    m.render(path, Path(temp))
                decoder.assert_not_called()

    def test_native_nearest_preserves_pixels(self):
        a = np.broadcast_to(np.arange(m.WIDTH, dtype=np.uint16).astype(np.uint8), (m.HEIGHT, m.WIDTH))
        crop, xy = m.native_crop(a, [300.5, 300.49])
        self.assertEqual(xy, [301, 300])
        np.testing.assert_equal(crop, a[292:309, 293:310])
        with self.assertRaises(ValueError):
            m.native_crop(a, [1, 1])

    def test_native_context_exact_and_edges_not_clamped(self):
        a = np.broadcast_to(np.arange(m.WIDTH, dtype=np.uint16).astype(np.uint8), (m.HEIGHT, m.WIDTH))
        crop, xy = m.native_crop(a, [300.5, 300.49], 129)
        self.assertEqual(xy, [301, 300])
        self.assertEqual(crop.shape, (129, 129))
        np.testing.assert_equal(crop, a[236:365, 237:366])
        self.assertTrue(m.full_patch([300, m.HEIGHT - 36]))
        self.assertFalse(m.full_patch([300, m.HEIGHT - 36], 129))
        with self.assertRaisesRegex(ValueError, "no clamping"):
            m.native_crop(a, [300, m.HEIGHT - 36], 129)
        for size in (0, 16, 131, True):
            with self.assertRaises(ValueError):
                m.native_crop(a, [300, 300], size)

    def test_no_overwrites_or_linked_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            m.write_json(path, {"test": 1})
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                m.write_json(path, {"test": 2})
            self.assertEqual(json.loads(path.read_text()), {"test": 1})
            linked = Path(temp) / "linked.json"
            linked.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "unlinked"):
                m.verify_file(linked, m.sha(path))

    def test_bad_hash_rejected_before_metadata_read_or_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "wrong.json"
            path.write_text("{}")
            with patch.object(m, "select_episodes") as selected, patch.object(m, "JOURNAL_PATH", path):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    m.build_plan(journal_path=path)
                selected.assert_not_called()

    def test_unapproved_input_path_rejected_before_any_file_read(self):
        with patch.object(m, "verify_file") as verify:
            with self.assertRaisesRegex(ValueError, "exact approved"):
                m.build_plan(source_path=Path("/unapproved/chunk_0240.avi"))
            verify.assert_not_called()

    def test_strict_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                m.read_json(text)


if __name__ == "__main__":
    unittest.main()
