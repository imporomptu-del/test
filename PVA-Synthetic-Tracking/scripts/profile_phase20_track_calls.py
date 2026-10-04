"""Frozen-journal call-count comparison, not a detector replay or FPS claim."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, VisibleTracks, sha256
from verify_phase20_host_efficiency import load_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists")
    old_k = load_frozen(args.reference, "tiny_target.tracking._calls_oracle", "tiny_target/tracking/kalman.py")
    old_v = load_frozen(args.reference, "tiny_target._calls_oracle", "tiny_target/visible_baseline.py")
    old_v.KalmanTrackManager = old_k.KalmanTrackManager
    old_v.KalmanTrackingConfig = old_k.KalmanTrackingConfig
    cfg = json.loads((args.reference / "config.json").read_text())
    frozen = json.loads((args.reference / "freeze.json").read_text())
    if sha256(args.reference / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen policy changed")
    journal = args.reference / "pva_0126/frames.jsonl"
    managers = [old_v.VisibleTracks(old_v.VisibleConfig(**cfg), 10), VisibleTracks(VisibleConfig(**cfg), 10)]
    profiles = [cProfile.Profile(), cProfile.Profile()]
    pairs = 0
    with journal.open() as handle:
        for i in range(96):
            row = json.loads(next(handle))
            if row["frame_index"] != i:
                raise ValueError("Noncontiguous workload")
            results = []
            for manager, profile in zip(managers, profiles):
                if i >= 72:
                    profile.enable()
                results.append(manager.update(row["candidates"], i, row["timestamp_ns"], row["segment"],
                    np.asarray(row["source_to_reference"]), row["coverage"]["full_shape_hw"]))
                profile.disable()
            if results[0] != results[1]:
                raise AssertionError("Execution shortcut changed track output")
            pairs += 1
    details = {}
    for name, profile in zip(("before", "after"), profiles):
        details[name] = [dict(file=k[0], line=k[1], function=k[2], calls=v[1], self_ms=v[2]*1000,
            cumulative_ms=v[3]*1000) for k, v in pstats.Stats(profile).stats.items()]
        details[name].sort(key=lambda v: -v["self_ms"])
    result = dict(passed=True, exact_track_output_pairs=pairs, profiled_frames_inclusive=[72, 95],
        profiles=details, journal_sha256=sha256(journal), script_sha256=sha256(__file__),
        limitation="Host call-count evidence with cProfile overhead; saved detections are only a fixed execution workload. Not fresh detector accuracy or Jetson FPS.")
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(passed=True, exact_track_output_pairs=pairs, calls={
        name: {key: sum(f["calls"] for f in rows if f["function"] == key)
               for key in ("inv", "slogdet", "solve", "_quality_evidence", "summary")}
        for name, rows in details.items()}), indent=2))


if __name__ == "__main__":
    main()
