"""Paired detector-only parity/timing check; not end-to-end or airborne validation.

Uses one explicit already-authorized source. Identical unwarped grayscale inputs
are fed to old and current implementations; state and candidate equality are
required. Historical state is rebuilt equally, not claimed to match a full run.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector, sha256


def legacy_module(path):
    name = "tiny_target._phase20_prechange_visible"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(a, b, image, valid, segment):
    results = []
    times = []
    for model in (a, b):
        before = time.perf_counter()
        r = model.update(image, valid, segment)
        times.append((time.perf_counter() - before) * 1000)
        results.append(r)
    if results[0][0] != results[1][0]:
        raise AssertionError("Candidate differences")
    cover = []
    for _, v in results:
        v = dict(v)
        v.pop("detection_ms")
        cover.append(v)
    if cover[0] != cover[1]:
        raise AssertionError("Coverage differences")
    for name in ("background", "variance", "previous_valid"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.source.name != "chunk_0126.avi"
        or sha256(a.source)
        != "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344"
    ):
        raise ValueError("Only the explicitly authorized source126")
    old = legacy_module(a.legacy)
    cfg = VisibleConfig(
        **json.loads((ROOT / "configs/evaluation/phase20_visible_v7.json").read_text())
    )
    cv2.setNumThreads(cfg.opencv_threads)
    # Adversarial coverage: both models, odd-sized tile seams, invalid pixels,
    # segment resets, warmup, no pixel noise, and both spatial background modes.
    checks = 0
    rng = np.random.default_rng(908)
    for noise in (False, True):
        for model in ("frame_difference", "background_residual"):
            for spatial in ("box13", "median5"):
                params = dict(
                    pixel_noise_enabled=noise,
                    pixel_noise_model=model,
                    spatial_background=spatial,
                    warmup_frames=2,
                    tile_size=32,
                )
                x = old.VisiblePointDetector(old.VisibleConfig(**params))
                y = VisiblePointDetector(VisibleConfig(**params))
                for i in range(24):
                    im = rng.normal(30, 2, (67, 99)).astype(np.float32)
                    im[30, 20 + i] = 150
                    valid = np.ones(im.shape, bool)
                    valid[10:20, 10:20] = i % 3 != 0
                    compare(x, y, im, valid, i // 12)
                    checks += 1
    x = old.VisiblePointDetector(
        old.VisibleConfig(
            **json.loads(
                (ROOT / "configs/evaluation/phase20_visible_v7.json").read_text()
            )
        )
    )
    y = VisiblePointDetector(cfg)
    cap = cv2.VideoCapture(str(a.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 70)
    old_ms = []
    new_ms = []
    frames = 0
    try:
        for i in range(32):
            ok, bgr = cap.read()
            if (
                not ok
                or bgr.shape[:2] != (3190, 4784)
                or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != 71 + i
            ):
                raise ValueError("Source frame mismatch")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            valid = np.ones(gray.shape, bool)
            pair = (
                compare(x, y, gray, valid, 0)
                if i % 2 == 0
                else list(reversed(compare(y, x, gray, valid, 0)))
            )
            if i >= 8:
                old_ms.append(pair[0])
                new_ms.append(pair[1])
            frames += 1
            if i % 8 == 7:
                print(json.dumps(dict(paired_native_frames=frames)), flush=True)
    finally:
        cap.release()

    def summary(v):
        return dict(
            median=float(np.median(v)),
            mean=float(np.mean(v)),
            p95=float(np.percentile(v, 95)),
            samples=v,
        )

    result = dict(
        exact_candidates_coverage_and_used_state=True,
        adversarial_frame_pairs=checks,
        native_source_frame_pairs=frames,
        source_frames_inclusive=[70, 101],
        timed_frame_pairs=24,
        source_sha256=sha256(a.source),
        old_code_sha256=sha256(a.legacy),
        new_code_sha256=sha256(ROOT / "tiny_target/visible_baseline.py"),
        benchmark_sha256=sha256(__file__),
        platform=platform.platform(),
        python=platform.python_version(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        opencv_threads=cv2.getNumThreads(),
        old_detection_ms=summary(old_ms),
        new_detection_ms=summary(new_ms),
        median_speedup=float(np.median(old_ms) / np.median(new_ms)),
        persistent_previous_spatial_bytes_saved=3190 * 4784 * 4,
        timing_scope="Detector-only, unwarped source frames, paired alternating order; no PVA, decode, motion, tracking or steady-state end-to-end timing",
        learning_protection_enabled=False,
        thresholds_changed=False,
        real_time_claim=False,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("old_detection_ms", "new_detection_ms")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
