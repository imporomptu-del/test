"""Bounded historical tracker intervention replay, not new weak selection.

Reads only hash-bound chunk0126 metadata frames0..40. Uses the frozen V28
tracker adapter and original Joseph weak update; never opens source media or
native image captures. A parity failure is fatal, not an alternate explanation.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_SHA = "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344"
ATOL = 1e-7
CODE = {
    "tracking_stage_v28.py": "92b4e5a7b2be8556de434f1a216530d896e428e6c62e4879789ec7a9cc360975",
    "tracking_geometry_v20.py": "2021152f598082b88cabac762f12a7bb886def17a85b85cec57d1c0b31ad4e52",
    "tracking_batch_v27.py": "6ff824041e254bb7ada457263708e5210cb503f62d5691ee3af012564c7a4793",
    "weak_continuation_shadow_v1.py": "4d15015eee7a15adf177d25cd947c5f59b829a01c6a2cb960d83a3aa721e8d09",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), "Nonfinite JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


class ParityError(ValueError):
    def __init__(self, path, actual, expected):
        self.detail = dict(path=path, actual=actual, expected=expected)
        super().__init__("Replay parity mismatch at " + path)


def compare(actual, expected, path="value", numeric=False):
    """Exact structure/discrete fields; optional absolute-only float allowance."""
    if type(actual) is not type(expected):
        # JSON has no tuple; internal instrumentation explicitly normalizes first.
        raise ParityError(path, actual, expected)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ParityError(path + ".keys", sorted(actual), sorted(expected))
        for key in sorted(actual):
            compare(actual[key], expected[key], path + "." + key, numeric)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ParityError(path + ".length", len(actual), len(expected))
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{i}]", numeric)
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or (abs(actual-expected) > ATOL if numeric else actual != expected):
            raise ParityError(path, actual, expected)
    elif actual != expected:
        raise ParityError(path, actual, expected)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def compare_records(actual, expected, path):
    require(len(actual) == len(expected), path + ": record count differs")
    for index, (a, b) in enumerate(zip(actual, expected)):
        require(set(a) == set(b), path + ": record keys differ")
        for key in a:
            compare(a[key], b[key], f"{path}[{index}].{key}",
                    numeric=key in ("source_xy", "reference_xy", "velocity_reference_xy_px_s"))


def bound_path(root, name, digest):
    part = Path(name)
    require(not part.is_absolute() and ".." not in part.parts and str(part) == name, "Unsafe input name")
    result = root
    for p in part.parts:
        result /= p
        require(not result.is_symlink(), "Symlink input refused")
    require(result.is_file() and sha(result) == digest, "Changed bound input " + name)
    return result


def read_inputs(directory, receipt_sha256):
    root = Path(directory)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), "Canonical prefix directory required")
    receipt_path = bound_path(root, "receipt.json", receipt_sha256)
    receipt = loads(receipt_path.read_text())
    require(receipt.get("schema") == "seaqr.weak-shadow.divergence-prefix.v1" and receipt.get("passed") is True
            and receipt.get("error") is None and receipt.get("clip") == "0126"
            and receipt.get("source_sha256") == SOURCE_SHA and receipt.get("frame_start") == 0
            and receipt.get("frame_end_inclusive") == 40 and receipt.get("frames") == 41
            and receipt.get("original_full_frames") == 674 and receipt.get("rows_filtered") is False
            and receipt.get("raw_prefix_lines_preserved") is True, "Wrong bounded extraction receipt")
    artifacts = receipt["artifacts"]
    paths = {name: bound_path(root, name, info["sha256"]) for name, info in artifacts.items()}
    audit = loads(paths["original_audit.json"].read_text())
    require(audit.get("schema") == "seaqr.weak-continuation-shadow.audit.v1" and audit.get("passed") is True
            and audit.get("clip") == "0126" and audit.get("frames") == 674 and audit.get("source_sha256") == SOURCE_SHA
            and audit.get("baseline_journal_non_timing_exact") is True
            and audit.get("baseline_output_state_learning_digests_exact") is True
            and audit.get("native_state_guards_unchanged") is True and audit.get("production_changed") is False
            and audit.get("weak_learning_enabled") is False, "Passed full parent audit required")
    for key, name in (("audit_sha256", "original_audit.json"), ("freeze_sha256", "freeze.json"),
                      ("plan_sha256", "plan.json")):
        require(receipt[key] == artifacts[name]["sha256"], "Parent binding differs")
    reference = loads(paths["reference_0126.json"].read_text())
    require(reference["source_sha256"] == SOURCE_SHA and reference["fps"] == 10
            and reference["expected_frames"] == 674 and reference["config_sha256"] == receipt["configuration_sha256"],
            "Reference launch differs")
    journals = []
    for name in ("clean_prefix.jsonl.gz", "shadow_prefix.jsonl.gz"):
        info = artifacts[name]
        require(type(info["raw_bytes"]) is int and 0 < info["raw_bytes"] <= 67108864
                and info["frames"] == 41, "Bounded raw prefix required")
        with gzip.open(paths[name], "rb") as stream:
            raw = stream.read(info["raw_bytes"] + 1)
        require(len(raw) == info["raw_bytes"] and hashlib.sha256(raw).hexdigest() == info["raw_prefix_sha256"],
                "Raw prefix binding differs")
        rows = [loads(line) for line in raw.splitlines()]
        require(len(rows) == 41, "Exactly 41 prefix rows required")
        journals.append(rows)
    for name, info in artifacts.items():
        bound_path(root, name, info["sha256"])
    bound_path(root, "receipt.json", receipt_sha256)
    return receipt, reference, journals[0], journals[1]


def verify_runtime(reference):
    import tiny_target
    root = Path(tiny_target.__file__).resolve().parent
    for name, digest in reference["package_sha256"].items():
        bound_path(root, name, digest)
    modules = {}
    for name, digest in CODE.items():
        spec = importlib.util.find_spec(name[:-3])
        require(spec is not None and spec.origin is not None, "Missing frozen adapter " + name)
        path = Path(spec.origin)
        require(path.is_absolute() and not path.is_symlink() and sha(path) == digest, "Changed frozen adapter " + name)
        modules[name] = dict(path=str(path), sha256=digest)
    return dict(package=reference["package_sha256"], adapters=modules)


def manager_state(manager, timestamp):
    result = []
    for tid, track in sorted(manager._tracks.items()):
        mean, covariance = manager._predicted_state(track, timestamp)
        names = ("birth_timestamp_ns", "state_timestamp_ns", "last_measurement_timestamp_ns", "age_windows",
                 "associated_update_count", "independent_confirmation_hits", "missed_windows", "lifecycle_state",
                 "confirmation_timestamp_ns", "last_raw_response", "last_candidate_index", "last_credited_frame_indices")
        result.append(dict(track_id=tid, mean=track.mean.tolist(), covariance=track.covariance.tolist(),
            predicted_mean=mean.tolist(), predicted_covariance=covariance.tolist(),
            **{name: plain(getattr(track, name)) for name in names}))
    return result


def visible_state(tracker):
    return dict(extents=deepcopy(tracker.extents), qualified=sorted(tracker.qualified),
                quality={k: dict(history=plain(list(v.history)), latest=deepcopy(v.latest)) for k, v in tracker.quality.items()})


@contextmanager
def native_backend(geometry, geometry_sha256, batch_library, batch_sha256):
    from tracking_geometry_v20 import GeometryV20
    from tracking_stage_v28 import TrackingStageV28
    from tiny_target.tracking.kalman import KalmanTrackManager
    require(sha(geometry) == geometry_sha256 and sha(batch_library) == batch_sha256, "Native library pin differs")
    scalar = GeometryV20(geometry)
    optimized = TrackingStageV28(batch_library)
    method = optimized.adapt(scalar.adapter(KalmanTrackManager.update))
    yield method, optimized, scalar
    require(sha(geometry) == geometry_sha256 and sha(batch_library) == batch_sha256, "Native library changed during replay")
    require(not scalar.fallbacks and not optimized.geometry.fallbacks and not optimized.innovation_fallbacks,
            "Unexpected native fallback")


def replay_rows(clean, shadow, reference, method, diagnostics=None):
    """Low-level generated-test API; production callers must first bind inputs/runtime."""
    from tiny_target.visible_baseline import VisibleConfig, VisibleTracks, map_point
    from tiny_target.visible_quality import suppress_nearby
    from tiny_target.tracking.kalman import KalmanTrackManager
    from weak_continuation_shadow_v1 import joseph_weak_update
    cfg = VisibleConfig(**reference["configuration"])
    trackers = {arm: VisibleTracks(cfg, reference["fps"]) for arm in ("baseline", "shadow")}
    diagnostics = [] if diagnostics is None else diagnostics
    require(len(clean) == len(shadow) and bool(clean), "Paired prefix required")
    active = None
    captured = {}

    def update(manager, batch, *, include_quality_evidence=True):
        require(active is not None, "Unexpected tracking dispatch")
        polarity = next(p for p, m in trackers[active].managers.items() if m is manager)
        info = dict(prior=manager_state(manager, batch.reference_timestamp_ns), next_track_id_before=manager._next_track_id)
        result = method(manager, batch, include_quality_evidence=include_quality_evidence)
        info.update(associations=plain(result.associations), born_track_ids=plain(result.born_track_ids),
            deleted_tracks=plain(result.deleted_tracks), reset_reason=result.reset_reason,
            next_track_id_after=manager._next_track_id, post_strong=manager_state(manager, batch.reference_timestamp_ns))
        captured[polarity] = info
        return result

    with patch.object(KalmanTrackManager, "update", update):
        for frame, (b, s) in enumerate(zip(clean, shadow)):
            for row in (b, s):
                require(row["frame_index"] == frame and row["timestamp_ns"] == frame * 100000000,
                        "Contiguous 10Hz prefix from zero required")
            require(b["segment"] == s["segment"], "Segment mismatch")
            proposals = deepcopy(s["strong_proposals"])
            compare([{k: v for k, v in p.items() if k != "source_xy"} for p in b["candidates"]],
                    [{k: v for k, v in p.items() if k != "source_xy"} for p in proposals], f"frame{frame}.proposal_stream")
            matrix = np.asarray(b["source_to_reference"], np.float64)
            shape = tuple(b["coverage"]["full_shape_hw"])
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "Invalid coordinate transform")
            kept, suppressed = suppress_nearby(proposals, cfg.tracking_peak_nms_radius_px)
            ordinals = {id(p): i for i, p in enumerate(proposals)}
            mapping = {p: [dict(candidate_index=i, original_proposal_index=ordinals[id(item)],
                x=item["x"], y=item["y"], score=item["score"], response_dn=item["response_dn"])
                for i, item in enumerate(q for q in kept if q["polarity"] == p)] for p in ("bright", "dark")}
            out = dict(frame_index=frame, timestamp_ns=b["timestamp_ns"], segment=b["segment"],
                       candidate_order=mapping, resolution_nms=suppressed, arms={})
            for active in ("baseline", "shadow"):
                tracker = trackers[active]
                before_visible = visible_state(tracker)
                captured = {}
                records, metrics = tracker.update(deepcopy(proposals), frame, b["timestamp_ns"], b["segment"], matrix, shape)
                weak_events = []
                if active == "shadow":
                    actual = {r["track_id"]: r for r in records}
                    for archived in s["records"]:
                        note = archived["weak_evidence"]
                        if not note["applied"]:
                            continue
                        tid = archived["track_id"]
                        require(tid in actual and not actual[tid]["measured"], "Logged weak owner absent or strongly matched")
                        polarity, raw_id = tid.split(":")
                        manager = tracker.managers[polarity]
                        track = manager._tracks[int(raw_id)]
                        require(note["identity"] == f"{b['segment']}/{tid}" and
                                track.last_measurement_timestamp_ns == note["strong_anchor_timestamp_ns"], "Logged weak anchor differs")
                        compare(track.mean.tolist(), note["mean_before"], f"frame{frame}.{tid}.weak_mean_before", True)
                        compare(track.covariance.tolist(), note["covariance_before"], f"frame{frame}.{tid}.weak_cov_before", True)
                        mean, covariance = joseph_weak_update(track.mean, track.covariance,
                            note["measurement_reference_xy"], manager._measurement_covariance())
                        compare(mean.tolist(), note["mean_after"], f"frame{frame}.{tid}.weak_mean_after", True)
                        compare(covariance.tolist(), note["covariance_after"], f"frame{frame}.{tid}.weak_cov_after", True)
                        track.mean, track.covariance = mean, covariance
                        actual[tid].update(reference_xy=mean[:2].tolist(), source_xy=map_point(np.linalg.inv(matrix), *mean[:2]),
                                           velocity_reference_xy_px_s=mean[2:].tolist())
                        weak_events.append(deepcopy(note))
                    expected = [{k: v for k, v in r.items() if k != "weak_evidence"} for r in s["records"]]
                    expected_metrics = {k: v for k, v in s["metrics"].items() if k != "weak_continuation"}
                else:
                    expected, expected_metrics = b["tracks"], b["tracking_metrics"]
                compare_records(records, expected, f"frame{frame}.{active}.records")
                compare(metrics, expected_metrics, f"frame{frame}.{active}.metrics", numeric=True)
                out["arms"][active] = dict(managers=captured, visible_before=before_visible,
                    visible_after=visible_state(tracker), weak_events=weak_events, records=records,
                    metrics=metrics, parity_passed=True)
            diagnostics.append(out)
    return diagnostics


def run(directory, receipt_sha256, geometry, geometry_sha256, batch_library, batch_sha256, output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh replay output required")
    report = dict(schema="seaqr.weak-divergence-replay.v1", passed=False, error=None, frames=[],
        script_sha256=sha(__file__), input_receipt_sha256=receipt_sha256,
        media_accessed=False, weak_selection_reevaluated=False, production_changed=False,
        replay_kind="Historical logged interventions over frozen strong proposals, not a new counterfactual policy",
        parity_policy=dict(posterior_pose_absolute_tolerance=ATOL, metrics_float_absolute_tolerance=ATOL,
            relative_tolerance=0, measurements_ids_quality_and_discrete_fields="exact"))
    try:
        receipt, reference, clean, shadow = read_inputs(directory, receipt_sha256)
        report["runtime_hashes"] = verify_runtime(reference)
        with native_backend(geometry, geometry_sha256, batch_library, batch_sha256) as (method, optimized, scalar):
            replay_rows(clean, shadow, reference, method, report["frames"])
            report["adapter"] = dict(transformed_sha256=optimized.transformed_sha256,
                geometry_library_sha256=geometry_sha256, batch_library_sha256=batch_sha256,
                geometry_calls=optimized.geometry.calls, scalar_geometry_calls=scalar.calls,
                innovation_batches=optimized.innovation_batches)
        read_inputs(directory, receipt_sha256)
        verify_runtime(reference)
        report["passed"] = True
    except BaseException as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc),
                               detail=exc.detail if isinstance(exc, ParityError) else None)
        raise
    finally:
        report["completed_frames"] = len(report["frames"])
        opener = gzip.open if output.suffix == ".gz" else open
        with opener(output, "xt") as stream:
            json.dump(report, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "geometry", "batch-library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "geometry-sha256", "batch-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.directory, args.receipt_sha256, args.geometry, args.geometry_sha256,
        args.batch_library, args.batch_sha256, args.output)
