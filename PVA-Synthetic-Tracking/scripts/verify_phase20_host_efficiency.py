"""Exact execution-only oracle against the frozen pre-optimization runtime."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_kalman_tracking import config, batch, candidate
from test_visible_learning import cfg as visible_config
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisiblePointDetector, VisibleTracks, sha256


def load_frozen(parent, name, relative):
    frozen = json.loads((parent / "freeze.json").read_text())
    with tarfile.open(parent / "runtime.tar.gz") as archive:
        code = archive.extractfile(relative).read()
    if hashlib.sha256(code).hexdigest() != frozen["files_sha256"][relative]:
        raise ValueError("Frozen reference changed")
    module = types.ModuleType(name)
    module.__file__ = str(parent / "runtime.tar.gz" / relative)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite verification")
    old_k = load_frozen(args.reference, "tiny_target.tracking._host_oracle", "tiny_target/tracking/kalman.py")
    old_l = load_frozen(args.reference, "tiny_target._learning_oracle", "tiny_target/visible_learning.py")
    old_v = load_frozen(args.reference, "tiny_target._visible_oracle", "tiny_target/visible_baseline.py")
    old_s = load_frozen(args.reference, "tiny_target._shape_oracle", "tiny_target/visible_shapes.py")
    old_v.KalmanTrackManager = old_k.KalmanTrackManager
    old_v.KalmanTrackingConfig = old_k.KalmanTrackingConfig
    old_v.shape_learning_mask = old_l.shape_learning_mask
    old_v.consolidate_half_height = old_s.consolidate_half_height
    rng = np.random.default_rng(482601)
    policies, frames = [], 0
    for model in ("position_only", "position_velocity"):
        for cost in ("mahalanobis", "gaussian_nll"):
            modes = ("none", "log_response", "log_response_coast") if (
                model == "position_only" and cost == "gaussian_nll") else ("none",)
            for mode in modes:
                for assignment in ("greedy", "global_min_cost"):
                    cfg = config(measurement_model=model, association_cost=cost,
                        association_appearance=mode, association_assignment=assignment,
                        association_prior="hit_maturity" if cost == "gaussian_nll" else "none",
                        maximum_position_residual_px=45., maximum_velocity_residual_px_s=100.,
                        max_active_tracks=80, max_missed_windows=7, birth_policy="spatial_fair")
                    before = old_k.KalmanTrackManager(old_k.KalmanTrackingConfig(**asdict(cfg)))
                    after = KalmanTrackManager(cfg)
                    for f in range(80):
                        cs = tuple(candidate(i, 20 + f*2 + (i % 6)*13 + int(rng.integers(-2, 3)),
                            20 + (i // 6)*16 + int(rng.integers(-2, 3)), 20, 0,
                            score=float(rng.uniform(2, 50))) for i in range(36) if rng.random() > .2)
                        b = batch(f*100_000_000 + (f % 3)*137, (f,), cs, segment=f//40)
                        left, right = before.update(b).to_dict(), after.update(b).to_dict()
                        left.pop("timings_ms"); right.pop("timings_ms")
                        if left != right:
                            raise AssertionError((model, cost, mode, assignment, f))
                        for tid in before._tracks:
                            a, z = before._tracks[tid], after._tracks[tid]
                            np.testing.assert_array_equal(a.mean, z.mean)
                            np.testing.assert_array_equal(a.covariance, z.covariance)
                        frames += 1
                    policies.append(dict(model=model, cost=cost, appearance=mode, assignment=assignment))
    closed_loop_frames = 0
    for appearance in ("log_response", "log_response_coast"):
        cfg = replace(visible_config(), tracking_association_appearance=appearance)
        old_values = asdict(cfg)
        for field in ('native_shape_library', 'native_shape_library_sha256'):
            if field not in old_v.VisibleConfig.__dataclass_fields__:
                if old_values[field] is not None:
                    raise ValueError('CPU oracle cannot enable a native shape library')
                del old_values[field]
        old_cfg = old_v.VisibleConfig(**old_values)
        before, after = old_v.VisiblePointDetector(old_cfg), VisiblePointDetector(cfg)
        bt, at = old_v.VisibleTracks(old_cfg, 10), VisibleTracks(cfg, 10)
        for f in range(120):
            shape = (111, 173) if f < 60 else (113, 175)
            image = rng.normal(30, 1, shape).astype(np.float32)
            image[40, 15 + f % 100] += 120
            if f % 5:
                image[46, 150 - f % 100] += 70
            image[80, 120] += 70 if f % 3 else 15
            valid = rng.random(shape) > .0001
            ts, segment = f*100_000_000, f//60
            left_centers, right_centers = bt.learning_centers(ts, segment), at.learning_centers(ts, segment)
            if left_centers != right_centers:
                raise AssertionError("Causal learning feedback differs")
            bp, bc = before.update(image, valid, segment, left_centers)
            ap, ac = after.update(image, valid, segment, right_centers)
            bc.pop("detection_ms"); ac.pop("detection_ms")
            if (bp, bc) != (ap, ac):
                raise AssertionError("Closed-loop detector output differs")
            for state in ("background", "variance", "previous_spatial", "previous_valid"):
                np.testing.assert_array_equal(getattr(before, state), getattr(after, state))
            if bt.update(copy.deepcopy(bp), f, ts, segment, np.eye(3), shape) != at.update(
                    copy.deepcopy(ap), f, ts, segment, np.eye(3), shape):
                raise AssertionError("Visible journal output differs")
            closed_loop_frames += 1
    result = dict(passed=True, exact_generic_track_frame_pairs=frames,
        exact_closed_loop_detector_track_feedback_state_pairs=closed_loop_frames,
        policies=policies, timing_only_excluded=True, approximate_arithmetic=False,
        reference_freeze_sha256=sha256(args.reference / "freeze.json"),
        script_sha256=sha256(__file__), implementation_sha256={name: sha256(ROOT / name) for name in (
            "tiny_target/tracking/kalman.py", "tiny_target/visible_baseline.py", "tiny_target/visible_learning.py",
            "tiny_target/visible_resident.py", "tiny_target/tracking/quadratic.py",
            "tiny_target/visible_shapes.py", "tiny_target/visible_noise.py", "tiny_target/visible_shapes_native.py")},
        native_shapes_enabled=False,
        scope="Synthetic reference-CPU numerical equivalence, not native shape validation, real-video accuracy or throughput.")
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "policies"}, indent=2))


if __name__ == "__main__":
    main()
