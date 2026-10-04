"""Frozen detection-component oracle and paired synthetic host timings; no media."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unit"))
from verify_phase20_host_efficiency import load_frozen
from test_detection_batching import scalar_noise_reference
from tiny_target.visible_baseline import sha256
from tiny_target.visible_shapes import consolidate_half_height
from tiny_target.visible_resident import sample_layout, SparseSpatial
from tiny_target.visible_noise import tile_noise_statistics


def paired_timing(before, after, repeats=16):
    before(); after()
    times = dict(before=[], after=[])
    for i in range(repeats):
        pairs = (("before", before), ("after", after))
        for name, function in pairs if i % 2 == 0 else pairs[::-1]:
            start = time.perf_counter()
            function()
            times[name].append(1000 * (time.perf_counter() - start))
    return dict(samples_ms=times, median_ms={k: float(np.median(v)) for k, v in times.items()},
        repeat_count=repeats, order="alternating, one warmup each", pipeline_speed_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite verification")
    old_s = load_frozen(args.reference, "tiny_target._detection_shape_oracle", "tiny_target/visible_shapes.py")
    old_r = load_frozen(args.reference, "tiny_target._detection_sparse_oracle", "tiny_target/visible_resident.py")
    rng = np.random.default_rng(614052)
    cv2.setNumThreads(2)
    record = dict(passed=False, media_accessed=False, shape_cases=0,
        reference_freeze_sha256=sha256(args.reference / "freeze.json"),
        script_sha256=sha256(__file__), numpy_version=np.__version__, opencv_version=cv2.__version__,
        implementation_sha256={p: sha256(ROOT / p) for p in (
            "tiny_target/visible_shapes.py", "tiny_target/visible_resident.py", "tiny_target/visible_noise.py")})
    for case in range(260):
        image = rng.normal(size=(71, 103)).astype(np.float32)
        image[30, 16:33] = 10; image[40, 33:42] = -8
        image[20:26, 60:65] = 12
        if case % 3 == 0:
            image[21:25, 61:64] = 0  # Hollow shapes cannot invent centers.
        image *= 10. ** ((case % 13) - 6)
        mask = rng.random(image.shape) > .003
        seeds = rng.integers([0, 0], [103, 71], (90, 2))
        seeds = np.concatenate((seeds, [[x, 30] for x in range(16, 33, 2)],
            [[x, 40] for x in range(33, 42, 2)], [[62, 20], [16, 30], [16, 30]]))
        rng.shuffle(seeds)
        proposals = [dict(x=int(x), y=int(y), polarity="bright" if image[y, x] >= 0 else "dark",
            score=float(abs(image[y, x])), response_dn=float(image[y, x]), noise_sigma_dn=1.) for x, y in seeds]
        radius = 1 + case % 8
        if old_s.consolidate_half_height(proposals, image, mask, radius, include_support=bool(case % 2)) != \
                consolidate_half_height(proposals, image, mask, radius, include_support=bool(case % 2)):
            raise AssertionError(("shape", case))
        record["shape_cases"] += 1

    shape = (3190, 4784)
    layout, ids = sample_layout(shape, 256, 4)
    samples = rng.normal(0, 3, len(ids)).astype(np.float32)
    support = np.ones(shape, bool)
    support[:6] = False; support[-6:] = False; support[:, :6] = False; support[:, -6:] = False
    functions = (lambda: scalar_noise_reference(samples, support, layout, 4, .5),
        lambda: tile_noise_statistics(samples, support, layout, 4, .5))
    left, right = [f() for f in functions]
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1] == right[1]
    record["noise_native_layout"] = paired_timing(*functions)

    seeds = np.asarray([[24 + (i % 64) * 48, 24 + (i // 64) * 48] for i in range(512)], np.int32)
    dy, dx = np.mgrid[-8:9, -8:9]
    blob = 20 * np.exp(-(dx * dx + dy * dy) / 5)
    patches = np.asarray([(-1 if i % 2 else 1) * blob + rng.normal(0, .3, (17, 17))
        for i in range(len(seeds))], np.float32)
    proposals = [dict(x=int(x), y=int(y), polarity="dark" if i % 2 else "bright",
        score=20., response_dn=20., noise_sigma_dn=1.) for i, (x, y) in enumerate(seeds)]
    before = old_r.SparseSpatial(shape, seeds, patches)
    after = SparseSpatial(shape, seeds, patches)
    np.testing.assert_array_equal(before.keys, after.keys)
    np.testing.assert_array_equal(before.values, after.values)
    assert old_s.consolidate_half_height(proposals, before, support, include_support=True) == \
        consolidate_half_height(proposals, after, support, include_support=True)
    record["sparse_native_shape"] = paired_timing(
        lambda: old_r.SparseSpatial(shape, seeds, patches), lambda: SparseSpatial(shape, seeds, patches))
    record["shape_512_native_coordinates"] = paired_timing(
        lambda: old_s.consolidate_half_height(proposals, before, support, include_support=True),
        lambda: consolidate_half_height(proposals, after, support, include_support=True))
    record["passed"] = True
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k not in (
        "noise_native_layout", "sparse_native_shape", "shape_512_native_coordinates")}, indent=2))
    for key in ("noise_native_layout", "sparse_native_shape", "shape_512_native_coordinates"):
        print(json.dumps({key: record[key]["median_ms"]}))


if __name__ == "__main__":
    main()
