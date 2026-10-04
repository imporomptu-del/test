"""Frozen shape oracle plus paired synthetic timings; never reads any media."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests/unit'))
from tiny_target.visible_baseline import sha256
from tiny_target.visible_shapes_native import NativeShapes
from verify_phase20_host_efficiency import load_frozen
from verify_phase20_detection_batch import paired_timing
from test_native_shapes import random_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('reference', 'library', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite verification evidence')
    old_s = load_frozen(args.reference, 'tiny_target._native_shape_oracle', 'tiny_target/visible_shapes.py')
    old_r = load_frozen(args.reference, 'tiny_target._native_sparse_oracle', 'tiny_target/visible_resident.py')
    native = NativeShapes(args.library, sha256(args.library))
    rng = np.random.default_rng(619913)
    cv2.setNumThreads(2)
    record = dict(passed=False, media_accessed=False, shape_cases=0,
        reference_freeze_sha256=sha256(args.reference/'freeze.json'), library_sha256=sha256(args.library),
        script_sha256=sha256(__file__), numpy_version=np.__version__, opencv_version=cv2.__version__,
        implementation_sha256={p: sha256(ROOT/p) for p in ('tiny_target/visible_shapes_native.py',
            'tiny_target/visible_resident.py', 'scripts/phase20_native_shapes.cpp')})
    for case in range(600):
        proposals, shape, seeds, patches, mask = random_case(rng, case)
        if case % 2:
            patches += rng.normal(0, .1, patches.shape).astype(np.float32)
        expected = old_s.consolidate_half_height(proposals, old_r.SparseSpatial(shape, seeds, patches), mask,
            include_support=bool(case % 3))
        actual = native.consolidate(proposals, shape, seeds, patches, mask, include_support=bool(case % 3))
        if actual != expected:
            raise AssertionError(('shape', case))
        record['shape_cases'] += 1
    shape = (3190, 4784)
    seeds = np.asarray([[24+(i % 64)*48, 24+(i//64)*48] for i in range(512)], np.int32)
    dy, dx = np.mgrid[-8:9, -8:9]
    blob = 20*np.exp(-(dx*dx+dy*dy)/5)
    patches = np.asarray([(-1 if i%2 else 1)*blob+rng.normal(0, .3, (17, 17)) for i in range(512)], np.float32)
    proposals = [dict(x=int(x), y=int(y), polarity='dark' if i%2 else 'bright', score=20.,
        response_dn=20., noise_sigma_dn=1.) for i, (x, y) in enumerate(seeds)]
    mask = np.ones(shape, bool)
    def before():
        return old_s.consolidate_half_height(proposals, old_r.SparseSpatial(shape, seeds, patches), mask, include_support=True)
    def after():
        return native.consolidate(proposals, shape, seeds, patches, mask, include_support=True)
    if before() != after():
        raise AssertionError('Native-coordinate cap case differs')
    record['native_coordinates_512_with_setup'] = paired_timing(before, after)
    record['passed'] = True
    with args.output.open('x') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'native_coordinates_512_with_setup'}, indent=2))
    print(json.dumps(record['native_coordinates_512_with_setup']['median_ms']))


if __name__ == '__main__':
    main()
