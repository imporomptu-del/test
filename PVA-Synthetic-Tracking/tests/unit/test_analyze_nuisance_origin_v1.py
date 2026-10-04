import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("analyze_nuisance_origin_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_nuisance_origin_v1.py"
spec = importlib.util.spec_from_file_location("origin_analysis", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def desc(a):
    a = np.asarray(a)
    raw = a.tobytes()
    return dict(dtype=a.dtype.str, shape=list(a.shape), data_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())


def fixture():
    config = dict(tile_size=256, input_bit_depth=8, pixel_noise_model="background_residual", background_alpha=.05,
        pixel_noise_alpha=.1, pixel_noise_clip_sigma=4, noise_floor_dn=.5, learning_protection_mode="variance_only")
    params = dict(background_alpha=float(np.float32(.05)), pixel_noise_alpha=float(np.float32(.1)),
        pixel_noise_clip_sigma_squared=16.0, noise_floor_squared=.25, variance_only=True)
    episodes, groups = [], {}
    requests = [(f, p, "burst") for f in (60, 75, 90) for p in ("bright", "dark")]
    requests += [(f, "dark", "known_target") for f in (435, 450)]
    for index, (anchor, polarity, kind) in enumerate(requests):
        name = f"{kind}_{anchor:03d}_{polarity}"
        xy = [100 + index * 90, 200]
        episode = dict(episode_id=name, anchor_frame=anchor, frames=list(range(anchor - 2, anchor + 3)),
            polarity=polarity, sample_status="available", comparator_status="available", label="uncertain",
            sample_reference_xy=xy, candidate_centroid_reference_xy=[xy[0] + 2, xy[1] + 1],
            candidate_score=4.0, candidate_response_dn=2.0 if polarity == "bright" else -2.0, candidate_noise_sigma_dn=.5)
        episodes.append(episode)
        for f in episode["frames"]:
            for role, location in (("sample", xy), ("comparator", [xy[0] + 48, xy[1]])):
                groups.setdefault(str(f), []).append(dict(point_id=name + "/" + role, episode_id=name,
                    role=role, reference_xy=location, source_xy=[float(v) for v in location]))
    plan = dict(clip="0240", patch_size=17, source_shape_hw=[3190, 4784], frame_points=groups, episodes=episodes)
    stats = np.tile(np.array([0, .5], np.float32), (247, 1))
    records = []
    for f in m.CAPTURE_FRAMES:
        points, raw = [], []
        for frozen in groups[str(f)]:
            e = next(e for e in episodes if e["episode_id"] == frozen["episode_id"])
            x, y = frozen["reference_xy"]
            ti = (y // 256) * 19 + x // 256
            spatial = np.full((17, 17), 3 if e["polarity"] == "bright" else -1, np.float32)
            background, variance = np.ones((17, 17), np.float32), np.full((17, 17), .25, np.float32)
            support, eligible, learn = [np.ones((17, 17), np.bool_) for _ in range(3)]
            if frozen["role"] == "sample":
                learn[8, 8] = False
            t, ba, va = m.update_state(spatial, background, variance, support, learn, params)
            matches = []
            if frozen["role"] == "sample":
                matches = [dict(x=x, y=y, polarity=e["polarity"], cell_index=2 * ti + (e["polarity"] == "dark"), rank=0,
                    score=4.0, response_dn=float(t[8, 8]), noise_sigma_dn=.5)]
                raw.extend(matches)
            a = dict(warped_image=np.full((17, 17), 40, np.float32), warped_blur=np.full((17, 17), 40, np.float32),
                spatial=spatial, temporal_pre_learning=t, background_pre_finish=background, variance_pre_finish=variance,
                background_post_finish=ba, variance_post_finish=va, support_pre_finish=support,
                eligible_pre_finish=eligible, support_post_finish=support, learning_mask_post_finish=learn)
            center = dict(spatial_dn=float(spatial[8, 8]), temporal_dn=float(t[8, 8]), background_dn=1.0,
                variance_dn_squared=.25, tile_center_dn=0.0, tile_sigma_float32_dn=.5, pixel_sigma_dn=.5,
                effective_sigma_dn=.5, tile_sigma_float64_dn=.5, tile_floor_or_variance_comparison="equal",
                background_after_dn=float(ba[8, 8]), variance_after_dn_squared=float(va[8, 8]),
                supported=True, eligible=True, learn_variance=bool(learn[8, 8]), variance_protected=not bool(learn[8, 8]),
                raw_peak_matches=matches, exact_peak_reconstruction_count=len(matches))
            points.append(dict(frozen, tile_index=ti, **{k: desc(v) for k, v in a.items()}, center_values=center))
        records.append(dict(frame_index=f, timestamp_ns=f * 100000000, shape_hw=[3190, 4784], finish_parameters=params,
            points=points, pre_tile_statistics=desc(stats), pre_tile_sigmas_float64=desc(np.full(247, .5, np.float64)), raw_peaks=raw))
    capture = dict(schema=m.CAPTURE_SCHEMA, processed_frames=673, native_finish_calls=673,
        capture_frames=m.CAPTURE_FRAMES, front_instances=1, original_update_results_returned_unchanged=True,
        diagnostic_device_state_set=False, busy_flag_overridden=False, point_snapshots=80, records=records)
    return plan, capture, config


def stored_fixture(root):
    """Entirely generated metadata and arrays; no camera media or runtime code."""
    root = Path(root)
    directory, bundle = root / "copy", root / "bundle"
    directory.mkdir()
    bundle.mkdir()
    (directory / "run").mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    plan, capture, config = fixture()
    plan.update(schema="seaqr.nuisance-origin.packet.v1",
        input_sha256=dict(source=m.SOURCE_SHA, journal=m.JOURNAL_SHA, manifest=m.MANIFEST_SHA))
    for name in m.MEMBERS:
        (bundle / name).write_text("# generated fixture, not executed\n")
    write(bundle / "packet_plan.json", plan)
    freeze = dict(schema=m.RUN_SCHEMA + ".freeze", files={name: m.sha(bundle / name) for name in m.MEMBERS})
    write(bundle / "freeze.json", freeze)
    (directory / "freeze.json").write_bytes((bundle / "freeze.json").read_bytes())
    digest = m.sha(bundle / "freeze.json")
    workspace = "/tmp/seaqr_nuisance_origin_20261001_ABC123"
    source = dict(path="/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0240.avi", sha256=m.SOURCE_SHA,
        frames=673, width=4784, height=3190, fps=10, codec="mjpeg", pixel_format="yuvj420p")
    common = dict(schema=m.RUN_SCHEMA, passed=True, error=None, freeze_sha256=digest, workspace=workspace,
        source=source, source_clip="0240", full_frames=673, clocks_changed=False,
        clock_policy_before={"clock": 1}, clock_policy_after={"clock": 1}, inputs={"bound": "fixture"},
        identities={"identity": "fixture"}, **{key: False for key in ("algorithm_changed", "production_changed",
        "raw16_accessed", "sealed_holdouts_accessed", "source_class_inferred", "performance_benchmark")})
    write(directory / "preflight.json", dict(common, preflight=True))
    write(directory / "capture.json", capture)
    write(directory / "run/report.json", {"frames": 673})
    write(directory / "run/launch.json", dict(configuration=config, source=source["path"], source_sha256=m.SOURCE_SHA, expected_frames=673, fps=10))
    (directory / "run/frames.jsonl").write_text("".join(json.dumps(dict(frame_index=i, timestamp_ns=i * 100000000)) + "\n" for i in range(673)))
    parity = dict(schema="seaqr.feature-residual-trace.v1.parity", passed=True, rows_compared=673,
        expected_frames=673, original_journal_sha256=m.JOURNAL_SHA, diagnostic_journal_sha256=m.sha(directory / "run/frames.jsonl"),
        mismatch_frames=[], numeric_tolerance=False, excluded_paths=m.TIMING_PATHS)
    write(directory / "parity.json", parity)
    files = dict(capture="capture.json", parity="parity.json", journal="run/frames.jsonl", report="run/report.json",
                 launch="run/launch.json", preflight="preflight.json")
    result = dict(common, preflight=False, non_timing_journal_parity=True, processed_frames=673, decoded_frames_verified=673,
        **{key + "_sha256": m.sha(directory / name) for key, name in files.items()})
    write(directory / "result.json", result)
    artifact_map = {workspace + "/result.json": m.sha(directory / "result.json")}
    artifact_map.update({workspace + "/" + name: m.sha(directory / name) for name in files.values()})
    write(directory / "batch_status.json", dict(schema=m.RUN_SCHEMA + ".batch", complete=True, passed=True, error=None,
        current=None, freeze_sha256=digest, workers=1, automatic_retries=0, production_changed=False,
        phases=[dict(name=name, returncode=0) for name in ("tests", "preflight", "run")], child_artifacts_sha256=artifact_map))
    return directory, bundle, digest


class OriginAnalysisTests(unittest.TestCase):
    def test_generated_full_capture_and_anchor_bits(self):
        result, arrays = m.analyze_capture(*fixture())
        self.assertEqual(len(result["exact_anchor_reconstructions"]), 8)
        self.assertEqual(len(arrays), 80)
        self.assertEqual(result["exact_background_variance_updates_verified"], 23120)
        self.assertEqual(result["episodes"][0]["roles"]["sample"]["center_variance_protected_frames"], [58, 59, 60, 61, 62])
        self.assertFalse(result["precision_or_recall_estimated"])

    def test_peak_coordinates_not_centroid(self):
        plan, capture, config = fixture()
        result, _ = m.analyze_capture(plan, capture, config)
        self.assertEqual(result["exact_anchor_reconstructions"][0]["reference_xy"], plan["episodes"][0]["sample_reference_xy"])
        self.assertNotEqual(result["exact_anchor_reconstructions"][0]["reference_xy"], plan["episodes"][0]["candidate_centroid_reference_xy"])

    def test_descriptor_hash_corruption(self):
        d = desc(np.zeros((17, 17), np.float32))
        d["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bytes/hash"):
            m.array(d, "<f4", (17, 17))

    def test_descriptor_dtype_shape_and_boolean_bytes(self):
        d = desc(np.zeros((17, 17), np.float64))
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            m.array(d, "<f4", (17, 17))
        d = desc(np.zeros((17, 16), np.float32))
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            m.array(d, "<f4", (17, 17))
        d = desc(np.array([2], np.uint8))
        d["dtype"] = "|b1"
        with self.assertRaisesRegex(ValueError, "boolean"):
            m.array(d, "|b1", (1,))

    def test_reject_nonfinite_descriptor(self):
        d = desc(np.array([np.nan], np.float32))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.array(d, "<f4", (1,))

    def test_eligibility_is_not_learning_mask(self):
        plan, capture, config = fixture()
        p = capture["records"][0]["points"][0]
        p["learning_mask_post_finish"] = copy.deepcopy(p["eligible_pre_finish"])
        with self.assertRaisesRegex(ValueError, "variance learning"):
            m.analyze_capture(plan, capture, config)

    def test_phase_center_mislabel_rejected(self):
        plan, capture, config = fixture()
        capture["records"][0]["points"][0]["center_values"]["learn_variance"] = True
        with self.assertRaisesRegex(ValueError, "phase semantics"):
            m.analyze_capture(plan, capture, config)

    def test_state_corruption_one_ulp_fails(self):
        plan, capture, config = fixture()
        p = capture["records"][0]["points"][0]
        a = m.array(p["background_post_finish"], "<f4", (17, 17)).copy()
        a[0, 0] = np.nextafter(a[0, 0], np.float32(np.inf))
        p["background_post_finish"] = desc(a)
        with self.assertRaisesRegex(ValueError, "background learning"):
            m.analyze_capture(plan, capture, config)

    def test_temporal_corruption_fails(self):
        plan, capture, config = fixture()
        capture["records"][0]["points"][0]["temporal_pre_learning"] = desc(np.zeros((17, 17), np.float32))
        with self.assertRaisesRegex(ValueError, "temporal differs"):
            m.analyze_capture(plan, capture, config)

    def test_anchor_mismatch_fails(self):
        plan, capture, config = fixture()
        plan["episodes"][0]["candidate_score"] = 5.0
        with self.assertRaisesRegex(ValueError, "archived baseline"):
            m.analyze_capture(plan, capture, config)

    def test_raw_peak_noise_mismatch_fails(self):
        plan, capture, config = fixture()
        capture["records"][0]["raw_peaks"][0]["noise_sigma_dn"] = 1.0
        with self.assertRaisesRegex(ValueError, "noise_sigma_dn"):
            m.analyze_capture(plan, capture, config)

    def test_missing_frame_fails(self):
        plan, capture, config = fixture()
        capture["records"].pop()
        with self.assertRaisesRegex(ValueError, "capture frame"):
            m.analyze_capture(plan, capture, config)

    def test_exact_learning_update_and_clipping(self):
        spatial = np.array([[100, -10], [3, 4]], np.float32)
        bg, var = np.ones((2, 2), np.float32), np.full((2, 2), .25, np.float32)
        support = np.array([[True, True], [False, True]])
        learn = np.array([[True, False], [False, True]])
        params = dict(background_alpha=.5, pixel_noise_alpha=.5, pixel_noise_clip_sigma_squared=4., noise_floor_squared=.25, variance_only=True)
        t, b, v = m.update_state(spatial, bg, var, support, learn, params)
        np.testing.assert_array_equal(t, [[99, -11], [2, 3]])
        np.testing.assert_array_equal(b, [[50.5, -4.5], [1, 2.5]])
        np.testing.assert_array_equal(v, [[.625, .25], [.25, .625]])
        params["variance_only"] = False
        _, b, _ = m.update_state(spatial, bg, var, support, learn, params)
        np.testing.assert_array_equal(b, [[50.5, 1], [1, 2.5]])

    def test_learning_outside_support_rejected(self):
        a = np.ones((1, 1), np.float32)
        with self.assertRaisesRegex(ValueError, "outside support"):
            m.update_state(a, a, a, np.array([[False]]), np.array([[True]]), {})

    def test_invalid_parity_rejected(self):
        result = dict(passed=True, error=None, preflight=False, non_timing_journal_parity=True, processed_frames=673,
            decoded_frames_verified=673, full_frames=673, journal_sha256="a" * 64)
        parity = dict(schema="seaqr.feature-residual-trace.v1.parity", passed=True, rows_compared=673,
            expected_frames=673, original_journal_sha256=m.JOURNAL_SHA, diagnostic_journal_sha256="a" * 64,
            mismatch_frames=[], numeric_tolerance=False, excluded_paths=m.TIMING_PATHS)
        m.verify_parity(result, parity)
        for key, value in (("rows_compared", 672), ("mismatch_frames", [60]), ("numeric_tolerance", True),
                           ("excluded_paths", []), ("original_journal_sha256", "0" * 64)):
            with self.assertRaisesRegex(ValueError, "parity attestation"):
                m.verify_parity(result, dict(parity, **{key: value}))

    def test_fixed_shared_display_limits(self):
        np.testing.assert_array_equal(m.display_pixels(np.array([-20, -8, 0, 8, 20]), (-8, 8)), [0, 0, 128, 255, 255])
        v = m.statistics([-9, 0, 9], (-8, 8))
        self.assertEqual((v["display_below"], v["display_above"]), (1, 1))

    def test_hash_bindings_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            path.write_text("{}")
            bindings = {}
            m.bound(path, m.sha(path), bindings)
            m.recheck(bindings)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "changed"):
                m.recheck(bindings)

    def test_no_overwrite_before_reads(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "load_verified") as load:
            with patch("sys.argv", ["analysis", "--directory", temp, "--bundle", temp,
                        "--freeze-sha256", "0" * 64, "--output", temp]):
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    m.main()
            load.assert_not_called()

    def test_completed_loader_roundtrip_generated_only(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, bundle, digest = stored_fixture(temp)
            plan, capture, config, bindings = m.load_verified(directory, bundle, digest)
            self.assertEqual(len(m.analyze_capture(plan, capture, config)[0]["exact_anchor_reconstructions"]), 8)
            self.assertIn(str((directory / "batch_status.json").resolve()), bindings)
            self.assertIn(str((directory / "freeze.json").resolve()), bindings)

    def test_loader_requires_complete_batch_and_bound_map(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, bundle, digest = stored_fixture(temp)
            path = directory / "batch_status.json"
            original = json.loads(path.read_text())
            for key, value in (("complete", False), ("child_artifacts_sha256", {}), ("phases", [])):
                path.write_text(json.dumps(dict(original, **{key: value})))
                with self.assertRaises(ValueError):
                    m.load_verified(directory, bundle, digest)

    def test_loader_detects_capture_and_copied_freeze_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, bundle, digest = stored_fixture(temp)
            path = directory / "capture.json"
            original = path.read_text()
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.load_verified(directory, bundle, digest)
            path.write_text(original)
            (directory / "freeze.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.load_verified(directory, bundle, digest)

    def test_loader_checks_preflight_workspace_even_after_rebinding_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, bundle, digest = stored_fixture(temp)
            pre = json.loads((directory / "preflight.json").read_text())
            pre["workspace"] = "/tmp/unrelated"
            (directory / "preflight.json").write_text(json.dumps(pre))
            result = json.loads((directory / "result.json").read_text())
            result["preflight_sha256"] = m.sha(directory / "preflight.json")
            (directory / "result.json").write_text(json.dumps(result))
            batch = json.loads((directory / "batch_status.json").read_text())
            for name in ("result.json", "preflight.json"):
                batch["child_artifacts_sha256"][result["workspace"] + "/" + name] = m.sha(directory / name)
            (directory / "batch_status.json").write_text(json.dumps(batch))
            with self.assertRaisesRegex(ValueError, "matching preflight"):
                m.load_verified(directory, bundle, digest)

    def test_strict_json_applies_to_journals(self):
        for line in ('{"frame_index":0,"frame_index":1}', '{"timestamp_ns":NaN}', '{"v":1e999}'):
            with self.assertRaises(ValueError):
                m.decode(line)


if __name__ == "__main__":
    unittest.main()
