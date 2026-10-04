"""Generated-only equivalence and full-call timings; no camera source imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
sys.path[:0] = [str(HERE), str(RUNTIME if RUNTIME.exists() else HERE.parent)]
from learning_mask_v17 import LearningMaskV17, reference, REFERENCE_SHA
from build_learning_mask_v17 import sha


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check(output, library):
    if output.exists():
        raise FileExistsError(output)
    candidate = LearningMaskV17(library)
    record = dict(passed=False, error=None, real_media_read=False, cases=[], invalid=[], timings=[],
        library_sha256=sha(library), reference_sha256=REFERENCE_SHA,
        source_sha256={name: sha(HERE/name) for name in ('learning_mask_v17.py', 'learning_mask_v17.cpp',
            'build_learning_mask_v17.py', 'check_learning_mask_v17.py', 'visible_speed_v17_candidate.md')})
    def compare(support, regions, margin, label):
        before = digest(support), json.dumps(regions, sort_keys=True)
        calls = candidate.calls
        a, b = reference(support, regions, margin), candidate(support, regions, margin)
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        unchanged = before == (digest(support), json.dumps(regions, sort_keys=True))
        record['cases'].append(dict(label=label, shape=list(support.shape), margin=margin, exact=same,
                                   inputs_unchanged=unchanged, native=candidate.calls>calls, digest=digest(b)))
        if not same or not unchanged:
            raise AssertionError('Generated mask mismatch '+str(label))
    try:
        for seed in (127, 819):
            rng = np.random.default_rng(seed)
            for h, w in ((1, 1), (17, 23), (65, 81), (256, 384)):
                raw = rng.integers(0, 256, (h, w*2), dtype=np.uint8)
                support = raw.view(np.bool_)[:, ::2]  # noncanonical, noncontiguous bool
                for margin in (.5, 1, 2, 3.5, 8, 16):
                    for count in (0, 1, 16):
                        regions = []
                        for j in range(count):
                            xy = rng.integers(-16, max(h, w)+16, (min(1024, 8+64*j), 2)).astype(float)
                            xy += rng.choice([-.5, 0, .5], xy.shape)
                            xy[:4] = [[0, 0], [w-1, h-1], [1e100, -1e100], [.5, .5]]
                            regions.append(dict(support_reference_xy=xy.tolist()))
                        compare(support, regions, margin, [seed, h, w, margin, count])
        rng = np.random.default_rng(73291)
        h, w = 3190, 4784
        regions = []
        yy, xx = np.mgrid[:8, :8]
        for j in range(256):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            regions.append(dict(support_reference_xy=np.column_stack((xx.ravel()+cx-3.5, yy.ravel()+cy-3.5)).tolist()))
        for kind in ('none', 'holes', 'all'):
            support = np.zeros((h, w), bool) if kind == 'none' else np.ones((h, w), bool)
            if kind == 'holes':
                support[::7, ::11] = False
            for margin in (2, 16):
                compare(support, regions, margin, ['native', kind, margin])
        for invalid in ([], [[float('nan'), 2]], [[2, 3]]*1025):
            errors = []
            for fn in (reference, candidate):
                try:
                    fn(support, [dict(support_reference_xy=invalid)], 2)
                except Exception as exc:
                    errors.append(type(exc).__name__)
            if errors != ['ValueError', 'ValueError']:
                raise AssertionError('Invalid footprint behavior changed')
            record['invalid'].append(errors)
        for repeat in range(4):
            for mode in (('reference', 'candidate') if repeat%2==0 else ('candidate', 'reference')):
                fn = reference if mode == 'reference' else candidate
                started = time.perf_counter()
                result = fn(support, regions, 2)
                elapsed = 1000*(time.perf_counter()-started)
                record['timings'].append(dict(repeat=repeat, mode=mode, full_call_ms=elapsed, digest=digest(result)))
        if len({r['digest'] for r in record['timings']}) != 1:
            raise AssertionError('Timed result mismatch')
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['native_calls'] = candidate.calls
        with output.open('x') as f:
            json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in record.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    a = p.parse_args()
    check(a.output, a.library)
