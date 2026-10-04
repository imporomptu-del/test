"""Independent, offline V56 full-replay parity audit; never opens source media.

The five timing-key exclusions deliberately match the inspected frozen
compare_phase20_exact_runs.without_timing contract.  This module imports neither
that utility nor the detector, runner, bridge or its diagnostic interpretation.
Capture NPZs are hashed as opaque evidence, not loaded as image/array input.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path


FRAME_COUNT = 674
CAPTURE_FRAMES = tuple(range(213, 219))
TIMING_KEYS = frozenset(("timings_ms", "warp_timings_ms", "pva_timings_ms", "timing_ms", "detection_ms"))
SOURCE = "/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi"
SOURCE_SHA = "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344"
CONFIG_SHA = "7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f"
MOTION_CONFIG_SHA = "fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1"
V29_FREEZE_SHA = "60b79d450672d131b517e9ed5a33fdeb40a6a2a9b29c6c0f584a9a8e93cc0dbc"
LIBRARY_SHA = "fd689b653175eb9baf3e84259ccccd431ba3ec8aa6c6fa4378a596eb03f8a027"
GEOMETRY_SHA = "1d81a0369a78ca462fce16e9655e9c81811d4544e071c013682f5aaf58821782"
BATCH_SHA = "bdabc75a633da7cb72c0565a3d2b87dcb6662b17b12ba96ccd7a7c586bebb644"
TRANSFORMED_SHA = "571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325"
BLAS_SHA = "37bbe4b1cbc29e5c68a3df2d3cfcc5aac97ee66c856f40fabedea760cc3b452a"
# Canonical JSON hashes computed from the already-exported V34 full-repeat0
# launch/V29/execution metadata, not from either new arm's assertions.
PINNED_FIELDS = {
    "configuration": "a38e916272323c2b4a7240bbeda36738c2d0c7021c9bc066993fdbeb75bbb6f5",
    "package_sha256": "80b6969c62e5774caf7ac3d9bec8743b896bff2daef7260ba7adb35b55220b24",
    "code_sha256": "60ad5927e9437d53b63de573a9ae8dc2136a9a6c6644ebb597de689a8a62a1f8",
    "frame_decode": "2a45cacef2ea3dc0204d3886bfdda37f3c1f43f84cdb1c435b550e3cb17fc991",
    "source_probe": "2383989adfc78de6d2c3d64ea514d5ec2a7278f6409fd1f8f51c9f4217d5f63a",
    "exact_cuda_stabilization": "07ade0099c6b4d71d63f5b9355cab0522b58cd4b18e5e4f605906a7c8cc01e84",
    "external_accelerators": "3575b91eef22a389af53721781252036eb78432dfef0ac930947ff42869df58b",
    "source_sha256": "2854313fc4c9a2d0cc590589e1cf36ff62fcad2feac3933b065c1c554ba80c91",
    "runtime_sha256": "0253282d1fba26f2b97542dbca3b95ea08ed447748c2d3fadd8a00841ffffc2a",
}
MOTION_PINS = {
    "wrapper_sha256": "d8828cfc1734fe33ad1aefd6256818badf0093f25690ecde7c923939c152c492",
    "adapter_sha256": "038e45d83c46909958fcbdf5b94d791f7a69733ef765851c7d0fdf77c19a48f8",
    "method_sha256": "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_ok(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON number: " + value)


def loads(value):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_nonfinite)


def encoded(value):
    # Unlike Python equality, this distinguishes True/1 and 1/1.0.
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def exact(left, right, description):
    require(encoded(left) == encoded(right), "Changed " + description)


def pinned(value, name):
    require(hashlib.sha256(encoded(value).encode()).hexdigest() == PINNED_FIELDS[name], "Changed archived " + name)


def without_timing(value):
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key not in TIMING_KEYS}
    return value


class Evidence:
    """Literal in-run paths only; no following paths declared by a receipt."""
    def __init__(self, directory):
        self.root = Path(directory)
        require(self.root.is_dir() and not self.root.is_symlink(), "Regular run directory required")
        self.hashes = {}

    def path(self, relative):
        part = Path(relative)
        require(not part.is_absolute() and ".." not in part.parts, "Unsafe evidence path")
        target = self.root
        for name in part.parts:
            target = target / name
            require(not target.is_symlink(), "Symlink evidence refused")
        require(target.is_file(), "Missing evidence: " + relative)
        actual = file_sha(target)
        if relative in self.hashes:
            require(actual == self.hashes[relative], "Evidence changed during audit: " + relative)
        self.hashes[relative] = actual
        return target

    def read(self, relative):
        result = loads(self.path(relative).read_text())
        require(type(result) is dict, "JSON object required: " + relative)
        return result

    def unchanged(self):
        for relative, expected in list(self.hashes.items()):
            require(file_sha(self.path(relative)) == expected, "Evidence changed during audit")


def _hash_map(value, description):
    require(type(value) is dict and bool(value), "Missing " + description)
    require(all(type(key) is str and digest_ok(item) for key, item in value.items()), "Invalid " + description)


def _freeze(evidence):
    frozen = evidence.read("freeze.json")
    require(frozen["schema"] == "seaqr.accuracy-v56-freeze.v1" and frozen["pre_run"] is True, "Missing pre-run freeze")
    _hash_map(frozen["files_sha256"], "frozen files")
    for relative, expected in frozen["files_sha256"].items():
        require(Path(relative).suffix in (".py", ".cu", ".md", ".json", ".so"), "Non-code path in freeze")
        require(not relative.startswith(("clean/", "probe/", "captures/")) and relative != "freeze.json", "Run output included in source freeze")
        evidence.path(relative)
        require(evidence.hashes[relative] == expected, "Changed frozen source " + relative)
    for key in ("plan_sha256", "runner_sha256", "auditor_sha256"):
        require(digest_ok(frozen[key]) and frozen[key] in frozen["files_sha256"].values(), "Unbound frozen " + key)
    require(frozen["auditor_sha256"] == file_sha(__file__), "Changed executing auditor")
    _hash_map(frozen["native_sources_sha256"], "native source identities")
    require(frozen["v29_freeze_sha256"] == V29_FREEZE_SHA and frozen["original_library_sha256"] == LIBRARY_SHA, "Changed frozen legacy identities")
    for key, relative in (("bridge_sha256", "build/capture.so"), ("build_sha256", "build/build.json"), ("generated_cuda_smoke_sha256", "generated_cuda_smoke.log")):
        evidence.path(relative)
        require(frozen[key] == evidence.hashes[relative], "Changed frozen " + key)
    build = evidence.read("build/build.json")
    require(build["passed"] is True and type(build["returncode"]) is int and build["returncode"] == 0, "Failed bridge build")
    exact(build["native_sources_sha256"], frozen["native_sources_sha256"], "bridge native layout hashes")
    for name, expected in frozen["native_sources_sha256"].items():
        require(Path(name).name == name and name.endswith(".cu"), "Invalid native source basename")
        evidence.path("build/" + name)
        require(evidence.hashes["build/" + name] == expected, "Changed compiled native layout source")
    require(type(frozen["runtime_reference"]) is dict, "Missing frozen runtime")
    return frozen


def _runtime(value):
    require(type(value) is dict and set(value) == {"before", "after"}, "Missing before/after runtime")
    before, after = value["before"], value["after"]
    keys = ("blas", "affinity", "numpy", "opencv", "thread_environment", "clock_ticks")
    for key in keys:
        exact(before[key], after[key], "numerical runtime " + key)
    exact(before["affinity"], list(range(12)), "CPU affinity")
    exact(before["thread_environment"], dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")), "thread environment")
    require(before["numpy"] == "1.26.1" and before["opencv"] == "4.10.0", "Changed numerical versions")
    require(type(before["clock_ticks"]) is int and before["clock_ticks"] == 100, "Changed clock ticks")
    require(type(before["blas"]) is list and len(before["blas"]) == 1, "Changed BLAS population")
    blas = before["blas"][0]
    require(type(blas["threads"]) is int and blas["threads"] == 12 and blas["sha256"] == BLAS_SHA, "Changed BLAS implementation/policy")
    require(type(after["opencv_threads"]) is int and after["opencv_threads"] == 2, "Changed OpenCV policy")


def _sidecar(record, arm, count, capture_frames):
    require(record["schema"] == "seaqr.accuracy-v56-replay.v1", "Unknown V56 schema")
    require(record["passed"] is True and record["error"] is None, "Failed V56 arm")
    require(record["arm"] == arm and type(record["processed_frames"]) is int and record["processed_frames"] == count, "Wrong V56 arm/extent")
    expected = list(capture_frames) if arm == "probe" else []
    exact(record["capture_frames"], expected, "capture scope")
    identities = record["identities"]
    require(type(identities) is dict and set(identities) == {"runner_sha256", "bridge_sha256", "plan_sha256", "freeze_sha256"}, "Invalid V56 identity schema")
    _hash_map(identities, "V56 identities")
    _hash_map(record["frozen_code"], "frozen code identities")
    rows = record["private_state_digests"]
    require(type(rows) is list and len(rows) == count, "Incomplete private state sequence")
    for index, row in enumerate(rows):
        require(type(row) is dict and set(row) == {"frame", "output", "state", "learning"}, "Invalid private state row")
        require(type(row["frame"]) is int and row["frame"] == index, "Private state order changed")
        require(all(digest_ok(row[key]) for key in ("output", "state", "learning")), "Invalid state digest")
    _runtime(record["runtime"])


def _launch(record, count):
    require(record["source"] == SOURCE and record["source_sha256"] == SOURCE_SHA, "Changed source identity")
    require(record["config_sha256"] == CONFIG_SHA and record["motion_config_sha256"] == MOTION_CONFIG_SHA, "Changed configuration identity")
    require(type(record["fps"]) in (int, float) and record["fps"] == 10, "Changed nominal file cadence")
    require(type(record["expected_frames"]) is int and record["expected_frames"] == count and record["max_frames"] is None, "Not a complete replay")
    require(record["annotations_supplied_to_detector"] is False, "Annotations supplied to detector")
    _hash_map(record["package_sha256"], "package hashes")
    _hash_map(record["code_sha256"], "code hashes")
    require(type(record["configuration"]) is dict and bool(record["configuration"]), "Missing configuration")
    require(record["exact_cuda_stabilization"]["library_sha256"] == LIBRARY_SHA, "Changed CUDA binary")
    require(record["external_accelerators"]["median"]["library_sha256"] == LIBRARY_SHA, "Changed median binary")
    for key in ("configuration", "package_sha256", "code_sha256", "frame_decode", "source_probe", "exact_cuda_stabilization", "external_accelerators"):
        pinned(record[key], key)


def _report(record, launch, count):
    require(record["completed"] is True and record["full_clip"] is True, "Incomplete report")
    require(type(record["frames"]) is int and record["frames"] == count, "Wrong report count")
    require(record["source_sha256"] == SOURCE_SHA and record["faint_target_synthetic_branch_enabled"] is False, "Changed report scope")
    exact(record["configuration"], launch["configuration"], "report configuration")
    for key in ("counts", "qualified_tracks", "qualified_track_count", "availability", "detection_status"):
        require(key in record, "Missing report semantic field " + key)
    decode = record["frame_decode"]
    exact(decode["contract"], launch["frame_decode"], "decoder contract")
    for key in ("decoded_frames", "consumed_frames"):
        require(type(decode[key]) is int and decode[key] == count, "Incomplete " + key)
    require(decode["worker_joined"] is True and decode["capture_released"] is True, "Decoder not closed")
    require(type(decode["dropped_frames"]) is int and decode["dropped_frames"] == 0, "Dropped frames")
    require(type(decode["read_calls"]) is int and decode["read_calls"] in (count, count + 1), "Changed decoder reads")
    require(type(decode["maximum_observed_frames_ahead"]) is int and decode["maximum_observed_frames_ahead"] == 1, "Changed prefetch bound")
    for key in ("elapsed_seconds", "processed_fps"):
        require(type(record[key]) in (int, float) and math.isfinite(record[key]) and record[key] > 0, "Invalid report timing")


def _v29(record, count, journal_sha, execution_sha):
    require(record["schema"] == "seaqr.visible-combined-v29.v1", "Unknown V29 schema")
    require(record["passed"] is True and record["error"] is None, "Inherited V29 gate failed")
    require(record["clip"] == "0126" and record["arm"] == "combined" and record["frames"] is None and record["state_audit"] is False, "Changed V29 scope")
    require(type(record["processed_frames"]) is int and record["processed_frames"] == count, "Wrong inherited count")
    for key, value in (("freeze_sha256", V29_FREEZE_SHA), ("config_sha256", CONFIG_SHA), ("library_sha256", LIBRARY_SHA), ("geometry_library_sha256", GEOMETRY_SHA), ("batch_library_sha256", BATCH_SHA), ("tracking_transformed_sha256", TRANSFORMED_SHA)):
        require(record[key] == value, "Changed inherited " + key)
    require(record["execution_policy"] == "serial_reference" and record["gpu_front"] is True and record["tracking_stage"] is True, "Changed execution policy")
    for key in ("raw16_accessed", "defaults_changed", "production_approved", "new_accuracy_validated", "staged_v24_enabled", "native_motion_v25_enabled"):
        require(record[key] is False, "Unexpected V29 scope flag " + key)
    comparison = record["comparison"]
    require(comparison["exact"] is True and type(comparison["frames"]) is int and comparison["frames"] == count, "Inherited exact gate missing")
    require(comparison["journal_sha256"] == journal_sha and comparison["execution_sha256"] == execution_sha, "Inherited artifact hash mismatch")
    require(digest_ok(comparison["reference_journal_sha256"]), "Missing historical reference binding")
    _hash_map(record["source_sha256"], "V29 source identities")
    pinned(record["source_sha256"], "source_sha256")


def _execution(record, count):
    require(record["passed"] is True and record["error"] is None and record["closed"] is True, "Motion execution failed/unclosed")
    require(record["branch"] == "visible" and record["clip"] == "0126" and record["mode"] == "reuse" and record["frames"] is None and record["injected"] is False, "Changed motion execution scope")
    require(type(record["processed_frames"]) is int and record["processed_frames"] == count, "Motion frame count")
    require(type(record["reuse_hits"]) is int and record["reuse_hits"] == count - 2 and type(record["reuse_misses"]) is int and record["reuse_misses"] == 1, "Motion reuse lifecycle changed")
    for key in ("wrapper_sha256", "adapter_sha256", "method_sha256"):
        require(record[key] == MOTION_PINS[key], "Changed motion provenance " + key)
    _hash_map(record["runtime_sha256"], "motion runtime")
    pinned(record["runtime_sha256"], "runtime_sha256")
    require(type(record["motion"]) is list and len(record["motion"]) == count - 1, "Incomplete motion rows")
    for index, row in enumerate(record["motion"], 1):
        require(type(row["frame"]) is int and row["frame"] == index and "identity" in row and "error" not in row, "Invalid motion row")


def _captures(evidence, record, frames):
    entries = record["captures"]
    require(type(entries) is list and len(entries) == len(frames), "Incomplete capture manifest")
    metadata_rows = {}
    for frame, entry in zip(frames, entries):
        require(type(entry) is dict and set(entry) == {"frame", "npz_path", "metadata_path", "npz_sha256", "metadata_sha256"}, "Invalid capture manifest row")
        require(type(entry["frame"]) is int and entry["frame"] == frame, "Capture order/scope changed")
        for suffix, label in (("npz", "npz"), ("json", "metadata")):
            relative = f"captures/frame_{frame:06d}.{suffix}"
            require(entry[label + "_path"] == relative, "Capture path outside frozen scope")
            evidence.path(relative)
            require(entry[label + "_sha256"] == evidence.hashes[relative], "Capture hash mismatch")
        metadata = evidence.read(entry["metadata_path"])
        require(type(metadata["frame"]) is int and metadata["frame"] == frame, "Wrong capture metadata frame")
        require(metadata["prelearning"] is True and metadata["full_exposed_state_unchanged"] is True, "Capture not certified prelearning/read-only")
        require(metadata["original_library_sha256"] == LIBRARY_SHA, "Capture native identity changed")
        _hash_map(metadata["native_state_before"], "capture pre-state")
        exact(metadata["native_state_before"], metadata["native_state_after"], "native state across capture")
        metadata_rows[frame] = metadata
    return metadata_rows


def _audit_pair(directory, *, expected_count, capture_frames):
    """Private generated-fixture seam; production audit_run always uses674/213..218."""
    evidence = Evidence(directory)
    frozen = _freeze(evidence)
    arms = {}
    for arm in ("clean", "probe"):
        side = evidence.read(arm + ".v56.json")
        _sidecar(side, arm, expected_count, capture_frames)
        exact(side["frozen_code"], frozen["files_sha256"], "sidecar frozen source map")
        exact(side["identities"], dict(runner_sha256=frozen["runner_sha256"], bridge_sha256=frozen["bridge_sha256"], plan_sha256=frozen["plan_sha256"], freeze_sha256=evidence.hashes["freeze.json"]), "sidecar freeze identities")
        for key in ("blas", "affinity", "numpy", "opencv", "thread_environment", "clock_ticks"):
            exact(side["runtime"]["before"][key], frozen["runtime_reference"][key], "frozen runtime " + key)
        launch = evidence.read(arm + "/launch.json")
        _launch(launch, expected_count)
        report = evidence.read(arm + "/report.json")
        _report(report, launch, expected_count)
        execution = evidence.read(arm + ".execution.json")
        _execution(execution, expected_count)
        journal = evidence.path(arm + "/frames.jsonl")
        legacy = evidence.read(arm + ".v29.json")
        _v29(legacy, expected_count, evidence.hashes[arm + "/frames.jsonl"], evidence.hashes[arm + ".execution.json"])
        captures = _captures(evidence, side, capture_frames if arm == "probe" else ())
        arms[arm] = dict(side=side, launch=launch, report=report, execution=execution, legacy=legacy, journal=journal, captures=captures)
    left, right = arms["clean"], arms["probe"]
    exact(left["launch"], right["launch"], "complete launch provenance")
    for key in ("identities", "frozen_code", "private_state_digests", "runtime"):
        exact(left["side"][key], right["side"][key], "sidecar " + key)
    for key in ("source_sha256", "freeze_sha256"):
        exact(left["legacy"][key], right["legacy"][key], "inherited " + key)
    exact(left["legacy"]["comparison"]["reference_journal_sha256"], right["legacy"]["comparison"]["reference_journal_sha256"], "historical reference binding")
    for key in ("wrapper_sha256", "adapter_sha256", "method_sha256", "runtime_sha256"):
        exact(left["execution"][key], right["execution"][key], "motion " + key)
    exact([[r["frame"], r["identity"]] for r in left["execution"]["motion"]], [[r["frame"], r["identity"]] for r in right["execution"]["motion"]], "motion identities")
    report_ignored = {"elapsed_seconds", "processed_fps", "timings_ms"}
    exact({k: v for k, v in left["report"].items() if k not in report_ignored}, {k: v for k, v in right["report"].items() if k not in report_ignored}, "report non-timing fields")
    count = 0
    with left["journal"].open() as a, right["journal"].open() as b:
        for index, pair in enumerate(zip_longest(a, b)):
            require(None not in pair, "Unequal journal lengths")
            rows = [loads(line) for line in pair]
            for row in rows:
                require(type(row) is dict and type(row["frame_index"]) is int and row["frame_index"] == index, "Noncontiguous journal")
                require(all(key in row for key in ("timestamp_ns", "segment", "source_to_reference", "motion", "coverage", "candidates", "tracks", "tracking_metrics", "timings_ms")), "Incomplete journal schema")
            exact(without_timing(rows[0]), without_timing(rows[1]), "journal semantics at frame " + str(index))
            if index in right["captures"]:
                metadata = right["captures"][index]
                for key in ("tracks", "tracking_metrics", "source_to_reference", "segment"):
                    exact(metadata[key], rows[1][key], "capture/journal " + key)
                candidates = [{k: v for k, v in candidate.items() if k != "source_xy"} for candidate in rows[1]["candidates"]]
                exact(metadata["post_shape"], candidates, "capture/journal candidate lineage")
            count += 1
    require(count == expected_count, "Incomplete journal extent")
    evidence.unchanged()
    return dict(schema="seaqr.accuracy-v56-independent-audit.v1", passed=True,
                audited_at_utc=datetime.now(timezone.utc).isoformat(), frames=count,
                private_state_rows=count, capture_frames=list(capture_frames),
                ignored_journal_timing_keys=sorted(TIMING_KEYS),
                exact_full_journal_semantics=True, exact_private_state_and_learning=True,
                exact_motion_identities=True, exact_launch_provenance=True,
                inherited_v29_gates_passed=True, capture_content_interpreted=False,
                source_media_accessed=False, accuracy_improvement_claim=False,
                files_sha256=evidence.hashes, auditor_sha256=file_sha(__file__))


def audit_run(directory):
    return _audit_pair(directory, expected_count=FRAME_COUNT, capture_frames=CAPTURE_FRAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Existing audit output")
    result = audit_run(args.run)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("V56 independent audit passed:674 full-frame comparisons and6capture artifacts.")


if __name__ == "__main__":
    main()
