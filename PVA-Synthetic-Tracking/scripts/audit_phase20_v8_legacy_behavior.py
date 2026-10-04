"""Compare default global-motion behavior against the actual frozen V7 module."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.motion import (
    GlobalMotionConfig,
    MotionCorrespondences,
    fit_global_motion,
)
from tiny_target.visible_baseline import sha256

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    snapshot = (
        ROOT
        / "results/tiny_target/phase20/v7_frozen_evaluation_20260913/snapshot/tiny_target/motion/global_motion.py"
    )
    assert (
        sha256(snapshot)
        == "2c59587733b0b7b9f928c293ca83f963227fb5da5344d74146df9015148b3f22"
    )
    name = "tiny_target.motion._frozen_v7_audit"
    spec = importlib.util.spec_from_file_location(name, snapshot)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    spec.loader.exec_module(old)
    failures = []
    count = 0
    accepted = 0
    rng = np.random.default_rng(813)
    for model in ("translation", "similarity"):
        for case in range(30):
            previous = rng.uniform([20, 20], [980, 780], (80, 2))
            if case % 5 == 0:
                previous *= 0.1  # Clustered support must retain old rejection.
            shift = rng.normal(0, 10, 2)
            angle = 0.01 if model == "similarity" else 0
            matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            current = previous @ matrix.T + shift + rng.normal(0, 0.02, previous.shape)
            bad = (case % 5) * 10
            current[:bad] = rng.uniform([0, 0], [1000, 800], (bad, 2))
            matches = MotionCorrespondences(
                previous,
                current,
                np.ones(80),
                np.zeros(80),
                0,
                1,
                0,
                100000000,
                (1000, 800),
                (500, 400),
                dict(usable_for_transform=case % 7 != 0),
                {},
                {},
            )
            config = dict(model=model, ransac_iterations=100)
            before = old.fit_global_motion(matches, old.GlobalMotionConfig(**config))
            after = fit_global_motion(matches, GlobalMotionConfig(**config))
            x, y = before.to_dict(), after.to_dict()
            x.pop("timing_ms")
            y.pop("timing_ms")
            if x != y:
                failures.append(dict(model=model, case=case))
            accepted += before.accepted
            count += 1
    result = dict(
        cases=count,
        accepted_by_old_policy=accepted,
        rejected_by_old_policy=count - accepted,
        exact_default_estimate_match=not failures,
        failures=failures,
        old_module_sha256=sha256(snapshot),
        new_module_sha256=sha256(ROOT / "tiny_target/motion/global_motion.py"),
        scope="Seeded correspondence-level backward-compatibility audit, not camera ground truth or hardware validation",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)
