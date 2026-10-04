#!/usr/bin/env python3
"""Baseline-only saved-proposal tracker parity; no pixels, policy changes or replay of detection.

Uses frozen v20/v28 Python adapters and unchanged strict-FP C++ sources rebuilt
for this host. Original observable output parity is mandatory with zero numeric
tolerance. Two independent local trackers additionally check state determinism;
the original run did not archive internal state or learning-center hashes.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930/evidence"
OUTPUT_ROOT = ROOT.parent / "outputs/seaqr_tracker_capacity_20261001"
CONFIG = ROOT / "results/tiny_target/visible_front_v26_20260920/evidence/candidate_config.json"
CONFIG_SHA = "7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f"
METHOD_SHA = "571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325"
CLIPS = ("0170", "0240")
FILES = {
    "0170": {
        "run/frames.jsonl": "56687056477c79f5cb9aa8c338c260438966c706fcd393ff29afa5536029eca5",
        "run/launch.json": "b4bfe0263c7f6aa216722b70bd909d211c95e26e0e2fc87f75c82bf0e3577a7b",
        "execution_receipt.json": "7e19411fb7622b02fa4900ec20e8d7ce44208c10da5c5cc6f431187601208a53",
    },
    "0240": {
        "run/frames.jsonl": "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92",
        "run/launch.json": "c84c212c1d40200dc783924366a2bec422a053f34bd23ede3951e5bb9b6af869",
        "execution_receipt.json": "a0062c34858f827860a106b75e4a29ad3c0a5e7e09cea3e12fb492ac3766ed28",
    },
}
SOURCE_SHA = {
    "0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
    "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585",
}
ADAPTERS = {
    "tracking_geometry_v20": "2021152f598082b88cabac762f12a7bb886def17a85b85cec57d1c0b31ad4e52",
    "tracking_batch_v27": "6ff824041e254bb7ada457263708e5210cb503f62d5691ee3af012564c7a4793",
    "tracking_stage_v28": "92b4e5a7b2be8556de434f1a216530d896e428e6c62e4879789ec7a9cc360975",
}
CPP = {
    "tracking_geometry_v20.cpp": "4bfb91cfa7023f0239205bbacfcd9f2b48792bdccdc61daaf05770a2832dc416",
    "tracking_batch_v27.cpp": "8e6be11aaf88d6355c8b88136cabd5a53fdb6f42c613124cae296f0b43192a3a",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            out.update(block)
    return out.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), "regular absolute input required: " + str(path))
    return path


def bind(path, expected, hashes):
    path = regular(path)
    require(sha(path) == expected, "input hash differs: " + str(path))
    hashes[str(path)] = expected


def decode(text):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON")
        return value
    return json.loads(text, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    return decode(regular(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_inputs():
    hashes, launches, receipts = {}, {}, {}
    bind(CONFIG, CONFIG_SHA, hashes)
    config = read(CONFIG)
    for name, expected in ADAPTERS.items():
        bind(ROOT / "scripts" / (name + ".py"), expected, hashes)
    for name, expected in CPP.items():
        bind(ROOT / "scripts" / name, expected, hashes)
    for clip in CLIPS:
        for relative, expected in FILES[clip].items():
            bind(EVIDENCE / clip / relative, expected, hashes)
        launch = read(EVIDENCE / clip / "run/launch.json")
        receipt = read(EVIDENCE / clip / "execution_receipt.json")
        require(receipt["passed"] is True and receipt["non_timing_journal_parity_passed"] is True
                and receipt["processed_frames"] == 673 and receipt["tracker_configuration_changed"] is False,
                "incomplete or changed original execution")
        require(receipt["journal_sha256"] == FILES[clip]["run/frames.jsonl"]
                and receipt["launch_sha256"] == FILES[clip]["run/launch.json"]
                and receipt["tracking_transformed_sha256"] == METHOD_SHA, "receipt identity differs")
        require(launch["config_sha256"] == CONFIG_SHA and launch["configuration"] == config
                and launch["fps"] == 10 and launch["expected_frames"] == 673
                and launch["max_frames"] is None and launch["source_sha256"] == SOURCE_SHA[clip],
                "frozen source/configuration/count differs")
        for name, expected in ADAPTERS.items():
            require(receipt["adapters"][name]["sha256"] == expected, "frozen adapter differs")
        for relative, expected in launch["package_sha256"].items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                    "unsafe package path")
            bind(ROOT / "tiny_target" / relative, expected, hashes)
        launches[clip], receipts[clip] = launch, receipt
    require(launches["0170"]["package_sha256"] == launches["0240"]["package_sha256"], "package differs across clips")
    return hashes, launches, receipts


def verify_unchanged(hashes):
    for path, expected in hashes.items():
        require(sha(regular(path)) == expected, "input changed during check: " + path)


def output_guard(output):
    output = Path(output)
    require(output.is_absolute() and output.parent == OUTPUT_ROOT
            and re.fullmatch(r"baseline_[A-Za-z0-9_-]+", output.name), "outside bounded output scope")
    require(OUTPUT_ROOT.parent.resolve() == OUTPUT_ROOT.parent and not OUTPUT_ROOT.is_symlink(), "linked output parent")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    require(not output.exists() and not output.is_symlink(), "refusing existing output")
    output.mkdir()
    return output


def build_native(output):
    native = output / "native"
    native.mkdir()
    source = native / "source"
    source.mkdir()
    for name, expected in CPP.items():
        original = ROOT / "scripts" / name
        require(sha(regular(original)) == expected, "C++ source differs before copy")
        shutil.copyfile(original, source / name)
        require(sha(source / name) == expected, "C++ source copy differs")
    compiler = shutil.which("c++")
    require(compiler is not None, "local C++ compiler unavailable")
    version = subprocess.run([compiler, "--version"], text=True, capture_output=True, timeout=30)
    require(version.returncode == 0, "compiler version query failed")
    records, libraries = [], {}
    for label, filename in (("geometry", "tracking_geometry_v20.cpp"), ("batch", "tracking_batch_v27.cpp")):
        library = native / ("libtracking_" + label + ".so")
        command = [compiler, "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC",
                   str(source / filename), "-o", str(library)]
        process = subprocess.run(command, text=True, capture_output=True, timeout=60)
        record = dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr,
                      library_sha256=sha(library) if process.returncode == 0 else None)
        records.append(record)
        write_new(native / (label + "_build.json"), record)
        require(process.returncode == 0, "local native build failed")
        libraries[label] = library
    manifest = dict(compiler=compiler, compiler_version=version.stdout, source_sha256=CPP,
                    builds=records, original_jetson_binary_reused=False,
                    note="Unchanged frozen C++ sources rebuilt for local OS/architecture; runtime numerical equality remains unproven until archive parity.")
    write_new(native / "build_manifest.json", manifest)
    return libraries, manifest


def runtime_info():
    import numpy as np
    import cv2
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), numpy=np.__version__, opencv=cv2.__version__,
                blas=pools, thread_environment={k: os.environ.get(k) for k in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")})


def normalized(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), value.tobytes().hex()]
    if isinstance(value, np.generic):
        return normalized(value.item())
    if is_dataclass(value):
        return ["dataclass", type(value).__name__, [(f.name, normalized(getattr(value, f.name))) for f in fields(value)]]
    if isinstance(value, dict):
        return ["dict", [(normalized(k), normalized(v)) for k, v in sorted(value.items(), key=lambda item: repr(item[0]))]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [normalized(v) for v in value]]
    if isinstance(value, set):
        return ["set", sorted(normalized(v) for v in value)]
    if isinstance(value, deque):
        return ["deque", value.maxlen, [normalized(v) for v in value]]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) in (int, str, bool) or value is None:
        return value
    raise TypeError("Unrecognized state type " + str(type(value)))


def value_sha(value):
    return hashlib.sha256(json.dumps(normalized(value), separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshot(tracker):
    return dict(managers={p: vars(manager) for p, manager in tracker.managers.items()},
                extents=tracker.extents, qualified=tracker.qualified, summary=tracker.summary,
                ever_qualified=tracker.ever_qualified, previous_records=tracker.previous_records,
                previous_timestamp_ns=tracker.previous_timestamp_ns,
                quality={key: vars(value) for key, value in tracker.quality.items()})


def json_value(value):
    # Match the original journal's JSON representation before strict comparison.
    return decode(json.dumps(value, allow_nan=False))


def first_difference(expected, actual, path=()):
    if type(expected) is not type(actual):
        return dict(path=list(path), kind="type", expected_type=type(expected).__name__, actual_type=type(actual).__name__,
                    expected=expected, actual=actual)
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return dict(path=list(path), kind="keys", missing=sorted(expected.keys()-actual.keys()), extra=sorted(actual.keys()-expected.keys()))
        for key in sorted(expected):
            found = first_difference(expected[key], actual[key], path+(key,))
            if found is not None:
                return found
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return dict(path=list(path), kind="length", expected=len(expected), actual=len(actual))
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = first_difference(left, right, path+(index,))
            if found is not None:
                return found
    elif type(expected) is float:
        if expected.hex() != actual.hex():
            return dict(path=list(path), kind="float_bits", expected=expected, actual=actual,
                        expected_hex=expected.hex(), actual_hex=actual.hex(), absolute_difference=abs(expected-actual))
    elif expected != actual:
        return dict(path=list(path), kind="value", expected=expected, actual=actual)
    return None


def make_adapter(libraries):
    from tiny_target.tracking.kalman import KalmanTrackManager
    from tracking_geometry_v20 import GeometryV20
    from tracking_stage_v28 import TrackingStageV28
    geometry = GeometryV20(libraries["geometry"])
    optimized = TrackingStageV28(libraries["batch"])
    method = optimized.adapt(geometry.adapter(KalmanTrackManager.update))
    require(optimized.transformed_sha256 == METHOD_SHA, "compiled Python tracking source differs")
    return method, geometry, optimized


def replay_rows(rows, tracker_pair, methods, manager_class, derived_tracker, state_stream, expected_frames=673):
    import numpy as np
    summary = dict(attempted_frames=0, exact_archive_frames=0, dual_state_exact_frames=0,
                   derived_learning_exact_frames=0, passed=False, first_difference=None)
    for index, row in enumerate(rows):
        require(index < expected_frames and type(row["frame_index"]) is int and row["frame_index"] == index
                and row["timestamp_ns"] == index*100_000_000, "frame/timestamp discontinuity")
        require(row["coverage"]["full_shape_hw"] == [3190, 4784], "native shape differs")
        summary["attempted_frames"] += 1
        outputs, centers, states = [], [], []
        for tracker, method in zip(tracker_pair, methods):
            centers.append(tracker.learning_centers(row["timestamp_ns"], row["segment"]))
            with patch.object(manager_class, "update", method):
                out = tracker.update(row["candidates"], index, row["timestamp_ns"], row["segment"],
                                     np.array(row["source_to_reference"]), row["coverage"]["full_shape_hw"])
            outputs.append(json_value(dict(tracks=out[0], tracking_metrics=out[1])))
            states.append(value_sha(snapshot(tracker)))
        derived_centers = derived_tracker.learning_centers(row["timestamp_ns"], row["segment"])
        learned = [value_sha(c) for c in centers]
        derived_hash = value_sha(derived_centers)
        archived = dict(tracks=row["tracks"], tracking_metrics=row["tracking_metrics"])
        differences = [
            ("original_archive_observables", first_difference(archived, outputs[0])),
            ("dual_replay_observables", first_difference(outputs[0], outputs[1])),
            ("dual_replay_internal_state", None if states[0] == states[1] else dict(expected=states[0], actual=states[1])),
            ("dual_replay_learning_centers", None if learned[0] == learned[1] else dict(expected=learned[0], actual=learned[1])),
            ("archive_derived_learning_centers", None if learned[0] == derived_hash else dict(expected=derived_hash, actual=learned[0])),
        ]
        known_count = row["coverage"].get("learning_protection", {}).get("causal_measured_track_centers")
        if known_count is not None:
            differences.append(("archive_learning_center_count", first_difference(known_count, len(centers[0]))))
        failure = next(((label, diff) for label, diff in differences if diff is not None), None)
        record = dict(frame_index=index, detection_ready=row["coverage"]["detection_ready"],
                      coverage_warmup=row["coverage"].get("warmup"),
                      output_sha256=[value_sha(o) for o in outputs],
                      archive_output_sha256=value_sha(archived), internal_state_sha256=states,
                      learning_centers_sha256=learned, archive_derived_learning_sha256=derived_hash,
                      differences={label: diff for label, diff in differences if diff is not None})
        state_stream.write(json.dumps(record, allow_nan=False)+"\n")
        state_stream.flush()
        if failure:
            summary["first_difference"] = dict(frame_index=index, category=failure[0], detail=failure[1],
                detection_ready=row["coverage"]["detection_ready"], warmup=row["coverage"].get("warmup"),
                candidate_records=len(row["candidates"]), compared_during_cold_start_or_warmup=not row["coverage"]["detection_ready"])
            return summary
        summary["exact_archive_frames"] += 1
        summary["dual_state_exact_frames"] += 1
        summary["derived_learning_exact_frames"] += 1
        derived_tracker.previous_records = row["tracks"]
        derived_tracker.previous_timestamp_ns = row["timestamp_ns"]
        if index and index % 50 == 0:
            print(json.dumps(dict(exact_archive_frames=index+1)), flush=True)
    require(summary["attempted_frames"] == expected_frames, "incomplete journal")
    summary["passed"] = True
    return summary


def run_clip(clip, launch, libraries, output):
    from tiny_target.visible_baseline import VisibleConfig, VisibleTracks
    from tiny_target.tracking.kalman import KalmanTrackManager
    cfg = VisibleConfig(**launch["configuration"])
    trackers = [VisibleTracks(cfg, launch["fps"]) for _ in range(2)]
    derived = VisibleTracks(cfg, launch["fps"])
    adapters = [make_adapter(libraries) for _ in range(2)]
    journal = EVIDENCE / clip / "run/frames.jsonl"
    require(sha(journal) == FILES[clip]["run/frames.jsonl"], "journal changed before replay")
    state_path = output / (clip + "_state_hashes.jsonl")
    with journal.open() as stream, state_path.open("x") as state_stream:
        result = replay_rows((decode(line) for line in stream), trackers, [a[0] for a in adapters],
                             KalmanTrackManager, derived, state_stream)
    require(sha(journal) == FILES[clip]["run/frames.jsonl"], "journal changed during replay")
    result.update(state_hashes_file=str(state_path), state_hashes_sha256=sha(state_path), adapter_runs=[])
    for _, geometry, optimized in adapters:
        stats = dict(geometry_calls=geometry.calls, geometry_fallbacks=geometry.fallbacks,
                     batch_calls=optimized.geometry.calls, batch_fallbacks=optimized.geometry.fallbacks,
                     batch_track_rows=optimized.geometry.track_rows, innovation_tracks=optimized.innovation_tracks,
                     innovation_fallbacks=optimized.innovation_fallbacks)
        result["adapter_runs"].append(stats)
        require(geometry.calls == geometry.fallbacks == optimized.geometry.fallbacks == optimized.innovation_fallbacks == 0,
                "unexpected native fallback")
    return result


def run(output):
    output = output_guard(output)
    result = dict(schema="seaqr.tracker-capacity-baseline-shadow.v1", passed=False, candidate_implemented=False,
                  algorithm_changed=False, source_media_opened=False, raw16_or_holdouts_accessed=False,
                  detector_replayed=False, production_promotion=False, no_tolerance_relaxation=True,
                  original_archive_internal_state_parity_claimed=False,
                  note="Exact archive tracks/metrics are required. Local dual-state determinism and archive-derived learning centers are additional checks, not hashes of archived internal state. Frozen detections make any future changed-policy replay a shadow, not a causal full-pipeline accuracy or FPS test.",
                  script_sha256=sha(Path(__file__).resolve()), clips={}, error=None, stopped_at_first_difference=False)
    hashes = {}
    try:
        hashes, launches, receipts = validate_inputs()
        result["inputs_sha256"] = hashes
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "scripts"))
        for name in ADAPTERS:
            module = importlib.import_module(name)
            require(Path(module.__file__).resolve() == ROOT / "scripts" / (name+".py"), "import shadowing")
        result["runtime_local"] = runtime_info()
        result["runtime_original"] = receipts["0170"]["runtime_before"]
        libraries, manifest = build_native(output)
        result["local_build"] = manifest
        result["tracking_transformed_sha256"] = METHOD_SHA
        for clip in CLIPS:
            result["clips"][clip] = run_clip(clip, launches[clip], libraries, output)
            if not result["clips"][clip]["passed"]:
                result["stopped_at_first_difference"] = True
                result["not_attempted_clips"] = list(CLIPS[CLIPS.index(clip)+1:])
                break
        result["passed"] = len(result["clips"]) == 2 and all(v["passed"] for v in result["clips"].values())
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if hashes:
            try:
                verify_unchanged(hashes)
                result["inputs_unchanged_after_check"] = True
            except Exception as exc:
                result["inputs_unchanged_after_check"] = False
                result["passed"] = False
                result["postcheck_error"] = repr(exc)
        write_new(output / "baseline_result.json", result)
    print(json.dumps(dict(passed=result["passed"], result=str(output / "baseline_result.json"),
                         sha256=sha(output / "baseline_result.json"), error=result["error"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
