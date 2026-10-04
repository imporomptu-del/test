"""Bounded steady-state call profile after a speed trial, never benchmark FPS."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector, VisibleTracks, PvaMotion, map_point, sha256
from compare_phase20_exact_runs import without_timing, shape_accelerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "motion-config", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite profiling evidence")
    if str(args.source) != "/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi" or sha256(args.source) != "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344":
        raise ValueError("Only the authorized development126 source is allowed")
    cfg = VisibleConfig(**json.loads(args.config.read_text()))
    if cfg.motion_backend != "pva" or cfg.stabilization_execution != "cuda_cubic_resident":
        raise ValueError("Expected exact integrated PVA/GPU path")
    launch = json.loads((args.reference / "launch.json").read_text())
    native = shape_accelerator(launch)
    if native is not None and sha256(cfg.native_shape_library) != native['library_sha256']:
        raise ValueError('Profiling native shape library differs from reference')
    if launch["config_sha256"] != sha256(args.config):
        raise ValueError("Profiling configuration differs from reference")
    for name, digest in launch["package_sha256"].items():
        if sha256(ROOT / "tiny_target" / name) != digest:
            raise ValueError("Profiling implementation differs from reference")
    with (args.reference / "frames.jsonl").open() as handle:
        reference = [json.loads(next(handle)) for _ in range(96)]
    cv2.setNumThreads(cfg.opencv_threads)
    profile = cProfile.Profile()
    cap = cv2.VideoCapture(str(args.source))
    detector = VisiblePointDetector(cfg)
    tracks = VisibleTracks(cfg, 10)
    motion = PvaMotion(args.motion_config, cfg.stabilization_execution, cfg.cuda_median_library)
    record = dict(passed=False, profiled_frames_inclusive=[72, 95], exact_prefix_frames=0,
        source_sha256=sha256(args.source), script_sha256=sha256(__file__),
        config_sha256=sha256(args.config), library_sha256=sha256(cfg.cuda_median_library),
        reference_sha256=sha256(args.reference / "frames.jsonl"),
        native_shape_accelerator=native,
        warning="cProfile adds overhead; call durations include native work/waits. Not throughput, GPU utilization or kernel occupancy.")
    timings, differences = [], []
    try:
        for i in range(96):
            if i >= 72:
                profile.enable()
            start = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("Decode failed")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ts = i*100_000_000
            image, valid, matrix, segment, metadata = motion.update(gray, i, ts)
            proposals, coverage = detector.update(image, valid, segment, tracks.learning_centers(ts, segment))
            records, metrics = tracks.update(proposals, i, ts, segment, matrix, image.shape)
            inverse = np.linalg.inv(matrix)
            for p in proposals:
                p["source_xy"] = map_point(inverse, p["x"], p["y"])
            duration = (time.perf_counter() - start)*1000
            profile.disable()
            if i >= 72:
                timings.append(duration)
            actual = dict(candidates=proposals, tracks=records, tracking_metrics=metrics,
                source_to_reference=matrix.tolist(), motion=metadata)
            for key, value in actual.items():
                if without_timing(reference[i][key]) != without_timing(value):
                    differences.append([i, key])
            if metadata["pva_failure"] or metadata["reset"]:
                raise ValueError("PVA failure/reset")
            record["exact_prefix_frames"] += 1
            if (i+1) % 24 == 0:
                print(json.dumps(dict(frames=i+1)), flush=True)
        stats = pstats.Stats(profile)
        functions = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1],
            self_ms=v[2]*1000, cumulative_ms=v[3]*1000) for k, v in stats.stats.items()]
        record.update(passed=not differences, first_differences=differences[:20], difference_count=len(differences),
            python_profile=sorted(functions, key=lambda row: -row["self_ms"]), instrumented_frame_ms=timings)
        if differences:
            raise AssertionError(differences[:10])
    except Exception as exc:
        record["error"] = repr(exc)
        raise
    finally:
        profile.disable()
        cap.release()
        detector.close()
        motion.close()
        with args.output.open("x") as handle:
            json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k not in ("python_profile", "instrumented_frame_ms")}, indent=2))


if __name__ == "__main__":
    main()
