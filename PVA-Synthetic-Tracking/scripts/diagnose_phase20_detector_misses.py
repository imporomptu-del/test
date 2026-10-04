"""Replay frozen CPU motion/detection with read-only, post-score gate diagnostics.

The diagnostic callback is injected in-memory before state learning. Labels only
select reported scalars; they cannot alter proposals, updates or thresholds.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import textwrap

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tiny_target.visible_baseline as baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    launch = json.loads((a.parent / "launch.json").read_text())
    if launch["configuration"]["motion_backend"] != "cpu_translation":
        raise ValueError("CPU exact replay only")
    source = Path(launch["source"])
    # Explicitly scoped, never discover or inspect other media.
    if source.name != "chunk_0126.avi":
        raise ValueError("Only the requested 126 diagnostic")
    local = ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0126.avi"
    if baseline.sha256(local) != launch["source_sha256"]:
        raise ValueError("Source identity mismatch")
    expected = launch["code_sha256"]["visible_baseline.py"]
    if baseline.sha256(ROOT / "tiny_target/visible_baseline.py") != expected:
        raise ValueError("Run before changing the detector implementation")
    labels = json.loads(a.labels.read_text())
    wanted = {
        s["frame_index"]: s
        for e in labels["positive_windows"]
        if e["window_id"] == "126"
        for s in e["visible_samples"]
    }
    cfg = baseline.VisibleConfig(**launch["configuration"])
    cv2.setNumThreads(cfg.opencv_threads)
    motion = baseline.CpuTranslation(cfg)
    # Same trusted implementation, one observer inserted; no output alteration.
    code = textwrap.dedent(inspect.getsource(baseline.VisiblePointDetector.update))
    marker = "    self.background[support] += cfg.background_alpha * temporal[support]"
    assert code.count(marker) == 1
    code = code.replace(marker, "    observer(locals())\n" + marker)
    diagnostics = []
    frame_index = -1
    matrix = None

    def observer(state):
        if frame_index not in wanted:
            return
        s = wanted[frame_index]
        rx, ry = baseline.map_point(matrix, *s["xy"])
        x, y = round(rx), round(ry)
        size = cfg.tile_size
        tx, ty = (x // size) * size, (y // size) * size
        temporal = state["temporal"]
        spatial = state["spatial"]
        sample = temporal[
            ty : ty + size : cfg.noise_sample_stride,
            tx : tx + size : cfg.noise_sample_stride,
        ]
        support = state["support"][
            ty : ty + size : cfg.noise_sample_stride,
            tx : tx + size : cfg.noise_sample_stride,
        ]
        sample = sample[support]
        center = float(np.median(sample)) if sample.size else 0.0
        sigma = max(
            cfg.noise_floor_dn,
            1.4826 * float(np.median(np.abs(sample - center))) if sample.size else 0.0,
        )
        points = []
        for yy in range(y - 5, y + 6):
            for xx in range(x - 5, x + 6):
                ps = max(sigma, float(np.sqrt(state["self"].variance[yy, xx])))
                points.append(
                    dict(
                        x=xx,
                        y=yy,
                        spatial_dn=float(spatial[yy, xx]),
                        temporal_dn=float(temporal[yy, xx]),
                        tile_sigma=sigma,
                        pixel_sigma=ps,
                        temporal_snr=float((temporal[yy, xx] - center) / ps),
                        spatial_snr=float(spatial[yy, xx] / ps),
                        peak=bool(state["peaks"][yy, xx]),
                        eligible=bool(state["eligible"][yy, xx]),
                    )
                )
        best = max(
            points,
            key=lambda v: min(
                v["temporal_snr"] / cfg.temporal_threshold_sigma,
                v["spatial_snr"] / cfg.spatial_threshold_sigma,
            )
            if v["peak"]
            else -1e20,
        )
        diagnostics.append(
            dict(frame_index=frame_index, reference_xy=s["xy"], best_peak=best)
        )

    namespace = dict(vars(baseline), observer=observer)
    exec(compile(code, "<read-only-gate-observer>", "exec"), namespace)
    detector = baseline.VisiblePointDetector(cfg)
    detector.update = namespace["update"].__get__(detector)
    cap = cv2.VideoCapture(str(local))
    equal = 0
    deviations = []
    try:
        with (a.parent / "frames.jsonl").open() as f:
            for row in map(json.loads, f):
                frame_index = row["frame_index"]
                if frame_index > 218:
                    break
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError("Short source")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                image, valid, matrix, segment, _ = motion.update(
                    gray, frame_index, row["timestamp_ns"]
                )
                proposals, coverage = detector.update(image, valid, segment)
                inverse = np.linalg.inv(matrix)
                for q in proposals:
                    q["source_xy"] = baseline.map_point(inverse, q["x"], q["y"])
                if proposals == row["candidates"]:
                    equal += 1
                else:
                    deviations.append(frame_index)
                if frame_index % 25 == 0:
                    print(
                        json.dumps(
                            dict(frame=frame_index, exact_candidate_frames=equal)
                        ),
                        flush=True,
                    )
    finally:
        cap.release()
    with a.output.open("x") as f:
        json.dump(
            dict(
                source_sha256=launch["source_sha256"],
                baseline_sha256=expected,
                labels_sha256=baseline.sha256(a.labels),
                observer_sha256=baseline.sha256(__file__),
                exact_candidate_frames=equal,
                differing_candidate_frames=deviations,
                diagnostics=diagnostics,
                diagnostic_labels_not_detector_inputs=True,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
