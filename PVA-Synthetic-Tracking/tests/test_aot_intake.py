"""Generated-only AOT intake checks: no publisher data or network access.

Fixtures exercise the adapter and acquisition guards, not detector accuracy.
Every URL read is replaced by a mock, including a default fail-closed guard.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "intake_aot_pilot.py"
SPEC = importlib.util.spec_from_file_location("seaqr_generated_aot_intake", SCRIPT)
intake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(intake)

FLIGHT = "0123456789abcdef0123456789abcdef"
OTHER_FLIGHT = "fedcba9876543210fedcba9876543210"
STAMP = 1550844897919368155


def generated_sample(count=320, positive=30, flight=FLIGHT):
    entities = []
    for frame in range(count):
        stamp = STAMP + frame * 100_000_000
        entity = {
            "time": stamp,
            "blob": {"frame": frame},
            "flight_id": flight,
            "img_name": f"{stamp}{flight}.png",
        }
        if frame < positive:
            entity.update({
                "id": "Airplane1", "bb": [1703.2, 939.2, 6.0, 6.0],
                "labels": {"is_above_horizon": 1},
            })
            entity["blob"]["range_distance_m"] = 650.0
        entities.append(entity)
    return {
        "metadata": {
            "data_path": f"train/{flight}/", "fps": 10.0,
            "number_of_frames": count,
            "resolution": {"width": 2448, "height": 2048},
        },
        "entities": entities,
    }


def generated_selection(count=300):
    sample = generated_sample(count=count)
    return {
        "part": "part1", "flight_id": FLIGHT,
        "frames": intake.frames_for(FLIGHT, sample),
        "metadata_prefix_sha256": "0" * 64,
        "max_source_png_bytes": intake.IMAGE_CAP,
        "bb_convention": "left,top,width,height",
        "pilot_type": "development convenience sample",
        "physical_class_source": "publisher airborne annotations, not SEAQR predictions",
    }


def generated_listing(selection, size=10, truncated=False, prefix=None, token=None):
    prefix = prefix or f"part1/Images/{FLIGHT}/"
    root = ET.Element("ListBucketResult", xmlns=intake.NS["s"])
    ET.SubElement(root, "Prefix").text = prefix
    ET.SubElement(root, "IsTruncated").text = str(truncated).lower()
    if token is not None:
        ET.SubElement(root, "NextContinuationToken").text = token
    for row in selection["frames"]:
        item = ET.SubElement(root, "Contents")
        ET.SubElement(item, "Key").text = prefix + row["img_name"]
        ET.SubElement(item, "Size").text = str(size)
        ET.SubElement(item, "ETag").text = '"' + "0" * 32 + '"'
    return ET.tostring(root)


class FakeResponse:
    def __init__(self, data=b"abc", *, url=None, status=200, headers=None):
        self.data = data
        self.url = url or intake.BASE + "part1/ImageSets/generated.json"
        self.status = status
        self.headers = {"Content-Length": str(len(data)), **(headers or {})}
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_sizes.append(size)
        return self.data[:size]


class GeneratedIntakeTests(unittest.TestCase):
    def setUp(self):
        no_network = mock.patch.object(
            intake.urllib.request, "urlopen",
            side_effect=AssertionError("tests must never access the network"),
        )
        self.network_guard = no_network.start()
        self.addCleanup(no_network.stop)
        directory = tempfile.TemporaryDirectory(prefix="seaqr_generated_aot_intake_")
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name)

    def test_large_timestamps_remain_exact_and_multiobjects_are_grouped(self):
        sample = generated_sample(count=3, positive=2)
        extra = copy.deepcopy(sample["entities"][0])
        extra.update(id="Bird2", bb=[100.5, 200.5, 16.0, 16.0])
        extra["blob"].pop("range_distance_m")
        sample["entities"].insert(1, extra)
        original = copy.deepcopy(sample)
        rows = intake.frames_for(FLIGHT, sample)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["timestamp_ns"], "1550844897919368155")
        self.assertEqual(rows[0]["img_name"], f"{STAMP}{FLIGHT}.png")
        self.assertEqual([row["airborne_label_count"] for row in rows], [2, 1, 0])
        self.assertEqual(rows[0]["entities"][0]["bb"], [1703.2, 939.2, 6.0, 6.0])
        self.assertEqual(sample, original)

    def test_complete_samples_preserve_exact_integer_timestamps(self):
        sample = generated_sample(count=2, positive=1)
        raw = json.dumps({"metadata": {"version": "generated"}, "samples": {FLIGHT: sample}}).encode()
        meta, samples = intake.parse_complete_samples(raw)
        self.assertEqual(meta, {"version": "generated"})
        self.assertEqual(samples, [(FLIGHT, sample)])
        self.assertIs(type(samples[0][1]["entities"][0]["time"]), int)
        self.assertEqual(samples[0][1]["entities"][0]["time"], STAMP)

    def test_truncated_final_sample_never_contributes_partial_entities(self):
        first = generated_sample(count=2, positive=1)
        second = generated_sample(count=2, positive=2, flight=OTHER_FLIGHT)
        start = '{"metadata": {}, "samples": {' + json.dumps(FLIGHT) + ":" + json.dumps(first) + ","
        ending = json.dumps(OTHER_FLIGHT) + ":" + json.dumps(second) + "}}"
        cuts = [2, len(json.dumps(OTHER_FLIGHT)), ending.index('"entities"') + 20, len(ending) - 5]
        for cut in cuts:
            with self.subTest(cut=cut):
                _, samples = intake.parse_complete_samples((start + ending[:cut]).encode())
                self.assertEqual(samples, [(FLIGHT, first)])

    def test_duplicate_sequence_keys_are_rejected(self):
        sample = json.dumps(generated_sample(count=1, positive=1))
        raw = ('{"metadata": {}, "samples": {' + json.dumps(FLIGHT) + ":" + sample
               + "," + json.dumps(FLIGHT) + ":" + sample + "}}")
        with self.assertRaises(ValueError):
            intake.parse_complete_samples(raw.encode())

    def test_no_complete_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            intake.parse_complete_samples(b'{"metadata": {}, "samples": {"0123')

    def test_missing_frame_disagrees_with_published_census(self):
        sample = generated_sample(count=3, positive=2)
        del sample["entities"][1]
        with self.assertRaises(ValueError):
            intake.frames_for(FLIGHT, sample)

    def test_noninteger_frames_and_timestamps_are_rejected(self):
        for field, value in [("frame", True), ("frame", 0.0), ("time", float(STAMP)), ("time", True)]:
            sample = generated_sample(count=2, positive=1)
            entity = sample["entities"][0]
            (entity["blob"] if field == "frame" else entity)[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                intake.frames_for(FLIGHT, sample)

    def test_nonincreasing_timestamps_are_rejected(self):
        for timestamp in [STAMP, STAMP - 1]:
            sample = generated_sample(count=2, positive=1)
            sample["entities"][1]["time"] = timestamp
            sample["entities"][1]["img_name"] = f"{timestamp}{FLIGHT}.png"
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                intake.frames_for(FLIGHT, sample)

    def test_inconsistent_source_identity_and_mixed_empty_record_are_rejected(self):
        cases = []
        sample = generated_sample(count=2, positive=1)
        sample["entities"][0]["img_name"] = "wrong.png"
        cases.append(sample)
        sample = generated_sample(count=2, positive=1)
        sample["entities"][0]["flight_id"] = OTHER_FLIGHT
        cases.append(sample)
        sample = generated_sample(count=2, positive=1)
        empty = copy.deepcopy(sample["entities"][0])
        del empty["bb"], empty["id"]
        sample["entities"].append(empty)
        cases.append(sample)
        for index, sample in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                intake.frames_for(FLIGHT, sample)

    def test_invalid_boxes_and_incomplete_labels_are_rejected(self):
        for box in [[0, 0, 0, 6], [0, 0, -1, 6], [0, 0, 6], [float("nan"), 0, 6, 6],
                    [0, 0, float("inf"), 6], [True, 0, 6, 6]]:
            sample = generated_sample(count=2, positive=1)
            sample["entities"][0]["bb"] = box
            with self.subTest(box=box), self.assertRaises((ValueError, TypeError)):
                intake.frames_for(FLIGHT, sample)
        for missing in ["bb", "id"]:
            sample = generated_sample(count=2, positive=1)
            del sample["entities"][0][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                intake.frames_for(FLIGHT, sample)

    def test_bad_object_ids_and_horizon_values_are_rejected(self):
        for bad_id in ["", "   ", None, 7]:
            sample = generated_sample(count=2, positive=1)
            sample["entities"][0]["id"] = bad_id
            with self.subTest(id=bad_id), self.assertRaises((ValueError, TypeError)):
                intake.frames_for(FLIGHT, sample)
        for bad_horizon in [2, True, "1"]:
            sample = generated_sample(count=2, positive=1)
            sample["entities"][0]["labels"]["is_above_horizon"] = bad_horizon
            with self.subTest(horizon=bad_horizon), self.assertRaises((ValueError, TypeError)):
                intake.frames_for(FLIGHT, sample)

    def test_duplicate_object_id_in_one_frame_is_rejected(self):
        sample = generated_sample(count=2, positive=1)
        sample["entities"].append(copy.deepcopy(sample["entities"][0]))
        with self.assertRaises(ValueError):
            intake.frames_for(FLIGHT, sample)

    def test_selection_thresholds_apply_only_to_first_300_frames(self):
        for positives, qualifies in [(29, False), (30, True), (280, True), (281, False)]:
            sample = generated_sample(count=320, positive=positives)
            candidates, census = intake.choose_candidates([(FLIGHT, sample)])
            with self.subTest(positive=positives):
                self.assertEqual(bool(candidates), qualifies)
                self.assertEqual(census[0]["qualifies"], qualifies)
                self.assertEqual(census[0]["prefix_labeled_frames"], positives)
                self.assertEqual(census[0]["prefix_empty_frames"], 300 - positives)
                if qualifies:
                    self.assertEqual(len(candidates[0][2]), 300)
                    self.assertEqual(candidates[0][2][-1]["source_frame"], 299)

    def test_selection_retains_publisher_order_not_quality_ranking(self):
        samples = [(OTHER_FLIGHT, generated_sample(flight=OTHER_FLIGHT)),
                   (FLIGHT, generated_sample(positive=200))]
        candidates, _ = intake.choose_candidates(samples)
        self.assertEqual([candidate[0] for candidate in candidates], [OTHER_FLIGHT, FLIGHT])

    def test_short_or_noncontiguous_prefix_never_qualifies(self):
        short = generated_sample(count=299)
        gap = generated_sample()
        for entity in gap["entities"][150:]:
            entity["blob"]["frame"] += 1
        for sample in [short, gap]:
            candidates, census = intake.choose_candidates([(FLIGHT, sample)])
            self.assertEqual(candidates, [])
            self.assertFalse(census[0]["qualifies"])

    def test_fetch_uses_unsigned_bounded_conditional_request(self):
        url = intake.BASE + "part1/ImageSets/generated.json"
        response = FakeResponse(url=url, status=206, headers={
            "Content-Range": f"bytes 0-2/{intake.META_BYTES}", "ETag": '"generated"',
        })
        with mock.patch.object(intake.urllib.request, "urlopen", return_value=response) as request:
            data, _ = intake.fetch(url, 3, byte_range=(0, 2), etag='"generated"')
        self.assertEqual(data, b"abc")
        self.assertEqual(response.read_sizes, [4])
        headers = {key.lower(): value for key, value in request.call_args.args[0].header_items()}
        self.assertEqual(headers["range"], "bytes=0-2")
        self.assertEqual(headers["if-match"], '"generated"')
        self.assertNotIn("authorization", headers)
        self.assertNotIn("x-amz-request-payer", headers)

    def test_fetch_rejects_foreign_source_without_network_access(self):
        with self.assertRaises(ValueError):
            intake.fetch("https://example.invalid/generated", 10)
        self.network_guard.assert_not_called()

    def test_chunked_listing_is_allowed_only_with_bounded_body(self):
        url = intake.BASE + "?list-type=2&prefix=part1%2FImages%2Fgenerated%2F&max-keys=1000"
        response = FakeResponse(url=url)
        del response.headers["Content-Length"]
        with mock.patch.object(intake.urllib.request, "urlopen", return_value=response):
            self.assertEqual(intake.fetch(url, 3)[0], b"abc")
        self.assertEqual(response.read_sizes, [4])
        oversized = FakeResponse(data=b"abcd", url=url)
        del oversized.headers["Content-Length"]
        with mock.patch.object(intake.urllib.request, "urlopen", return_value=oversized):
            with self.assertRaises(ValueError):
                intake.fetch(url, 3)

    def test_missing_object_length_is_not_treated_as_chunked_listing(self):
        response = FakeResponse()
        del response.headers["Content-Length"]
        with mock.patch.object(intake.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(ValueError):
                intake.fetch(response.url, 3)
        self.assertEqual(response.read_sizes, [])

    def test_fetch_rejects_redirect_status_size_etag_and_range_mismatch(self):
        url = intake.BASE + "part1/ImageSets/generated.json"
        cases = [
            (FakeResponse(url="https://example.invalid/redirect"), {}),
            (FakeResponse(status=206), {}),
            (FakeResponse(headers={"Content-Length": "4"}), {}),
            (FakeResponse(headers={"Content-Length": "-1"}), {}),
            (FakeResponse(headers={"Content-Length": "2"}), {}),
            (FakeResponse(headers={"ETag": '"wrong"'}), {"etag": '"expected"'}),
            (FakeResponse(status=206, headers={"Content-Range": "bytes 1-3/999"}), {"byte_range": (0, 2)}),
        ]
        for response, kwargs in cases:
            with self.subTest(headers=response.headers, kwargs=kwargs):
                with mock.patch.object(intake.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        intake.fetch(url, 3, **kwargs)

    def test_save_is_idempotent_but_never_overwrites_different_bytes(self):
        path = self.out / "nested" / "generated.bin"
        intake.save(path, b"original")
        intake.save(path, b"original")
        with self.assertRaises(ValueError):
            intake.save(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")

    def test_save_json_rejects_nonfinite_values_before_creating_file(self):
        path = self.out / "generated.json"
        with self.assertRaises(ValueError):
            intake.save_json(path, {"invalid": float("nan")})
        self.assertFalse(path.exists())

    def test_metadata_preserves_raw_nan_and_normalizes_only_derived_optional_range(self):
        sample = generated_sample()
        sample["entities"][0]["blob"]["range_distance_m"] = float("nan")
        sample["entities"][-1]["blob"]["range_distance_m"] = float("nan")
        raw = json.dumps({"metadata": {"version": "generated"}, "samples": {FLIGHT: sample}}).encode()
        with mock.patch.object(intake, "RANGE_BYTES", len(raw)):
            with mock.patch.object(intake, "fetch", return_value=(raw, {})) as fetch:
                with mock.patch("builtins.print"):
                    intake.metadata_phase(self.out)
                    intake.metadata_phase(self.out)
                self.assertEqual(fetch.call_count, 1)
        self.assertEqual((self.out / "metadata/groundtruth.prefix8MiB.bin").read_bytes(), raw)
        derived = json.loads((self.out / "metadata/selected_sequence_annotations.json").read_text())
        self.assertIsNone(derived["entities"][0]["blob"]["range_distance_m"])
        self.assertIsNone(derived["entities"][-1]["blob"]["range_distance_m"])
        self.assertEqual(derived["entities"][1]["blob"]["range_distance_m"], 650.0)
        self.assertEqual(derived["seaqr_derived_data_notice"]["normalized_nan_ranges"], 2)
        self.assertEqual(derived["seaqr_derived_data_notice"]["unmodified_source_prefix_sha256"], intake.sha(raw))
        preselection = json.loads((self.out / "preselection.json").read_text())
        self.assertEqual(len(preselection["frames"]), 300)
        self.assertIsNone(preselection["frames"][0]["entities"][0]["blob"]["range_distance_m"])

    def test_cached_metadata_hash_mismatch_is_rejected_without_fetch(self):
        intake.save(self.out / "metadata/groundtruth.prefix8MiB.bin", b"generated")
        intake.save_json(self.out / "metadata/source_receipt.json", {"sha256": "0" * 64})
        with mock.patch.object(intake, "RANGE_BYTES", 9):
            with mock.patch.object(intake, "fetch") as fetch:
                with self.assertRaises(ValueError):
                    intake.metadata_phase(self.out)
                fetch.assert_not_called()

    def write_inventory_fixture(self, directory=None, *, count=300, full_count=None):
        directory = directory or self.out
        selection = generated_selection(count)
        intake.save_json(directory / "preselection.json", selection)
        sample = generated_sample(count=full_count if full_count is not None else count)
        intake.save_json(directory / "metadata/selected_sequence_annotations.json", sample)
        return selection

    def test_inventory_freezes_exact_object_byte_sum(self):
        self.write_inventory_fixture(full_count=320)
        complete_inventory = generated_selection(320)
        with mock.patch.object(intake, "fetch", return_value=(generated_listing(complete_inventory), {})) as fetch:
            with mock.patch("builtins.print"):
                intake.inventory_phase(self.out)
        frozen = json.loads((self.out / "frozen_image_manifest.json").read_text())
        self.assertEqual(frozen["source_png_bytes"], 3000)
        self.assertEqual(frozen["preselection_sha256"], intake.sha((self.out / "preselection.json").read_bytes()))
        self.assertEqual(len(frozen["frames"]), 300)
        self.assertEqual(frozen["listed_sequence_objects"], 320)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn("list-type=2", fetch.call_args.args[0])
        self.assertIn("max-keys=1000", fetch.call_args.args[0])
        self.assertEqual(fetch.call_args.args[1], 1024 * 1024)

    def test_inventory_rejects_byte_and_frame_caps_without_manifest(self):
        for count, size in [(300, intake.IMAGE_CAP // 300 + 1), (299, 10)]:
            with self.subTest(count=count, size=size):
                directory = self.out / f"case_{count}"
                selection = self.write_inventory_fixture(directory, count=count)
                with mock.patch.object(intake, "fetch", return_value=(generated_listing(selection, size=size), {})):
                    with self.assertRaises(ValueError):
                        intake.inventory_phase(directory)
                self.assertFalse((directory / "frozen_image_manifest.json").exists())

    def test_inventory_wrong_prefix_and_missing_pagination_token_fail_closed(self):
        selection = generated_selection()
        for name, data in [("prefix", generated_listing(selection, prefix="part1/Images/wrong/")),
                           ("token", generated_listing(selection, truncated=True))]:
            directory = self.out / name
            self.write_inventory_fixture(directory)
            with self.subTest(name=name), mock.patch.object(intake, "fetch", return_value=(data, {})):
                with self.assertRaises(ValueError):
                    intake.inventory_phase(directory)
            self.assertFalse((directory / "frozen_image_manifest.json").exists())

    def test_inventory_missing_complete_annotations_fails_closed(self):
        selection = generated_selection()
        intake.save_json(self.out / "preselection.json", selection)
        with mock.patch.object(intake, "fetch", return_value=(generated_listing(selection), {})) as fetch:
            with self.assertRaises((FileNotFoundError, ValueError)):
                intake.inventory_phase(self.out)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn("list-type=2", fetch.call_args.args[0])
        self.assertFalse((self.out / "frozen_image_manifest.json").exists())
        self.assertFalse((self.out / "source_png").exists())

    def test_inventory_requires_exact_full_annotation_name_census(self):
        # Disagreements are beyond the 300 selected frames: checking only the
        # pilot subset would incorrectly allow every one of these listings.
        missing = generated_selection(319)
        extra = generated_selection(321)
        substituted = generated_selection(320)
        substituted["frames"][-1]["img_name"] = f"{STAMP + 999_000_000_000}{FLIGHT}.png"
        for name, listed in [("missing", missing), ("extra", extra), ("substituted", substituted)]:
            directory = self.out / name
            self.write_inventory_fixture(directory, full_count=320)
            with self.subTest(name=name):
                with mock.patch.object(intake, "fetch", return_value=(generated_listing(listed), {})):
                    with self.assertRaisesRegex(ValueError, "annotation census"):
                        intake.inventory_phase(directory)
                self.assertFalse((directory / "frozen_image_manifest.json").exists())
                self.assertFalse((directory / "source_png").exists())

    def write_download_fixture(self, mutate=None):
        selection = generated_selection()
        intake.save_json(self.out / "preselection.json", selection)
        manifest = copy.deepcopy(selection)
        for row in manifest["frames"]:
            row["source_object"] = {
                "key": f"part1/Images/{FLIGHT}/{row['img_name']}",
                "bytes": 10, "etag": '"' + hashlib.md5(b"generated!").hexdigest() + '"',
            }
        manifest.update({
            "source_png_bytes": 3000, "listed_sequence_objects": 300,
            "preselection_sha256": intake.sha((self.out / "preselection.json").read_bytes()),
        })
        if mutate:
            mutate(manifest)
        intake.save_json(self.out / "frozen_image_manifest.json", manifest)
        return manifest

    def assert_download_rejected_before_image_access(self):
        fake_cv2 = types.SimpleNamespace()
        with mock.patch.dict("sys.modules", {"cv2": fake_cv2, "numpy": types.SimpleNamespace()}):
            with mock.patch.object(intake, "fetch", side_effect=AssertionError("must validate before image fetch")) as fetch:
                with self.assertRaises(ValueError):
                    intake.download_phase(self.out)
                fetch.assert_not_called()
        self.assertFalse((self.out / "source_png").exists())

    def test_download_rejects_changed_preselection_before_access(self):
        self.write_download_fixture(lambda manifest: manifest.update(preselection_sha256="f" * 64))
        self.assert_download_rejected_before_image_access()

    def test_download_rejects_manifest_annotation_substitution_before_access(self):
        def mutate(manifest):
            manifest["frames"][-1]["entities"][0]["blob"]["frame"] = 998
        self.write_download_fixture(mutate)
        self.assert_download_rejected_before_image_access()

    def test_download_rejects_manifest_frame_reordering_before_access(self):
        def mutate(manifest):
            manifest["frames"][0], manifest["frames"][1] = manifest["frames"][1], manifest["frames"][0]
        self.write_download_fixture(mutate)
        self.assert_download_rejected_before_image_access()

    def test_download_rejects_late_bad_path_before_any_image_access(self):
        def mutate(manifest):
            manifest["frames"][-1]["source_object"]["key"] = "part1/Images/other/generated.png"
        self.write_download_fixture(mutate)
        self.assert_download_rejected_before_image_access()

    def test_download_rejects_nonpositive_size_before_any_image_access(self):
        def mutate(manifest):
            manifest["frames"][0]["source_object"]["bytes"] = intake.IMAGE_CAP + 1
            manifest["frames"][-1]["source_object"]["bytes"] = -intake.IMAGE_CAP
            manifest["source_png_bytes"] = sum(row["source_object"]["bytes"] for row in manifest["frames"])
        self.write_download_fixture(mutate)
        self.assert_download_rejected_before_image_access()

    def test_download_rejects_bad_late_etag_before_any_image_access(self):
        def mutate(manifest):
            manifest["frames"][-1]["source_object"]["etag"] = '"multipart-2"'
        self.write_download_fixture(mutate)
        self.assert_download_rejected_before_image_access()


if __name__ == "__main__":
    unittest.main()
