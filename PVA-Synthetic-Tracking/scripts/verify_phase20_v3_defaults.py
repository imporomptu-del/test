"""Check opt-in changes leave default numerical behavior unchanged."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_kalman_tracking import config, batch, candidate
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector
from score_phase20_accuracy import digest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    previous = (
        ROOT
        / "results/tiny_target/phase20/encounter_accuracy_v2_20260914/learning_guard_experiment/cpu_0029"
    )
    launch = json.loads((previous / "launch.json").read_text())
    for name in ("tracking/kalman.py", "visible_baseline.py"):
        if digest(previous / "implementation" / name) != launch["package_sha256"][name]:
            raise ValueError("Previous implementation changed")
    old_k = load(
        "tiny_target.tracking._v3_parity",
        previous / "implementation/tracking/kalman.py",
    )
    old_d = load(
        "tiny_target._v3_parity", previous / "implementation/visible_baseline.py"
    )
    rng = np.random.default_rng(203)
    tracking_frames = 0
    for model in ("position_only", "position_velocity"):
        for cost in ("mahalanobis", "gaussian_nll"):
            cfg = config(measurement_model=model, association_cost=cost)
            old_cfg = asdict(cfg)
            old_cfg.pop("association_assignment")
            old_cfg.pop("association_prior", None)
            old_cfg.pop("association_appearance", None)
            before, after = (
                old_k.KalmanTrackManager(old_k.KalmanTrackingConfig(**old_cfg)),
                KalmanTrackManager(cfg),
            )
            for f in range(80):
                cs = tuple(
                    candidate(
                        i, 20 + f + i * 12 + int(rng.integers(-1, 2)), 30 + i * 6, 10, 0
                    )
                    for i in range(3)
                    if rng.random() > 0.2
                )
                b = batch(f * 100_000_000, (f,), cs, segment=f // 40)
                x, y = before.update(b).to_dict(), after.update(b).to_dict()
                x.pop("timings_ms")
                y.pop("timings_ms")
                y["metrics"].pop("association_assignment")
                if x != y:
                    raise ValueError("Default tracker output changed")
                tracking_frames += 1
    detector_frames = 0
    for model in ("frame_difference", "background_residual"):
        for background in ("box13", "median5"):
            cfg = dict(
                pixel_noise_enabled=True,
                pixel_noise_model=model,
                spatial_background=background,
            )
            before = old_d.VisiblePointDetector(old_d.VisibleConfig(**cfg))
            after = VisiblePointDetector(
                VisibleConfig(**cfg, learning_protection_mode="variance_only")
            )
            for f in range(30):
                frame = rng.normal(30, 2, (67, 99)).astype(np.float32)
                frame[30, 20 + f] += 50
                valid = np.ones(frame.shape, bool)
                if f % 4 == 0:
                    valid[:, :5] = False
                x, xc = before.update(frame, valid, f // 15)
                y, yc = after.update(frame, valid, f // 15, [(20 + f, 30)])
                xc.pop("detection_ms")
                yc.pop("detection_ms")
                if x != y or xc != yc:
                    raise ValueError("Disabled learning protection changed detections")
                for name in (
                    "background",
                    "variance",
                    "previous_valid",
                    "previous_spatial",
                ):
                    np.testing.assert_array_equal(
                        getattr(before, name), getattr(after, name)
                    )
                detector_frames += 1
    phase19 = {
        "configs/evaluation/phase19_dense_screen_v1.json": "e9eb5d86e64beb8bcaf3ffb77967120e1745b16838eff9722aa49657e940a8ed",
        "tiny_target/dense_screen.py": "86a811c08c80ee2089f9ead363e51150392dd804191b5798240a05dfa0ed7628",
        "configs/tiny_target_phase12_cfar_test.yaml": "473b19b76f9a25035bf7b5d7f02b899144f3e3cd8369706712a012df350de4fe",
    }
    for name, sha in phase19.items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen Phase19 file changed")
    result = dict(
        default_tracking_frame_pairs_identical=tracking_frames,
        disabled_protection_detector_frame_pairs_identical=detector_frames,
        timing_excluded=True,
        added_assignment_policy_metadata_excluded=True,
        frozen_phase19_files_intact=True,
        experimental_policies_default_off=True,
        current_code_sha256={
            name: digest(ROOT / "tiny_target" / name)
            for name in ("visible_baseline.py", "tracking/kalman.py")
        },
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
