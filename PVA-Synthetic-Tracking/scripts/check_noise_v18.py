"""Generated-only exactness and cost gate: no media or dataset manifests."""
import argparse
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from profile_visible_v17 import sha, write
from noise_v18 import NoiseV18, reference, REFERENCE_SHA
from tiny_target.visible_resident import sample_layout

HERE = Path(__file__).resolve().parent


def compare(candidate, samples, support, layout, stride, floor):
    before = (samples.tobytes(), support.tobytes())
    n, f = candidate.calls, candidate.fallbacks
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = reference(samples, support, layout, stride, floor)
        actual = candidate(samples, support, layout, stride, floor)
    if (actual[0].tobytes() != expected[0].tobytes()
            or np.float64(actual[1]).tobytes() != np.float64(expected[1]).tobytes()):
        raise AssertionError('Noise bytes changed')
    if before != (samples.tobytes(), support.tobytes()):
        raise AssertionError('Noise input mutated')
    return dict(exact=True, inputs_unchanged=True, native=candidate.calls>n,
                fallback=candidate.fallbacks>f)


def cases(candidate, include_large=True):
    rows = []
    rng = np.random.default_rng(180917)
    for i in range(240):
        shape = (int(rng.integers(1, 180)), int(rng.integers(1, 220)))
        tile, stride = (8, 17, 32, 64)[i%4], (1, 2, 3, 4, 7)[i%5]
        layout, ids = sample_layout(shape, tile, stride)
        values = rng.normal(size=len(ids)).astype(np.float32) * np.float32(10.**(i%19-9))
        support = rng.random(shape) > (i%7)/6
        if i%9 == 0:
            values[:] = rng.choice([0., 3., -3.], len(ids))
        if i%11 == 0:
            values = np.repeat(values, 2)[::2]
            support = np.repeat(support, 2, axis=1)[:, ::2]
        rows.append(dict(name=f'random_{i}', **compare(candidate, values, support, layout, stride, .5)))
    # Fixed geometry with changing support/data catches incorrect mask/result caching.
    for shape in ((33, 51), (51, 33), (33, 51)):
        for stride in (1, 4, 1):
            layout, ids = sample_layout(shape, 17, stride)
            for mask_case in range(3):
                support = rng.random(shape) > mask_case/2
                values = rng.normal(size=len(ids)).astype(np.float32)
                rows.append(dict(name=f'cache_{len(rows)}', **compare(candidate, values, support, layout, stride, 2.25)))
    # Even median, repeated ranks, cancellation and adversarial float mantissas.
    for n in range(1, 66):
        layout, _ = sample_layout((1, n), 128, 1)
        values = rng.uniform(-100, 100, n).astype(np.float32)
        if n%3 == 0:
            values[:] = np.nextafter(np.float32(2), np.float32(3))
        rows.append(dict(name=f'ranks_{n}', **compare(candidate, values, np.ones((1,n),bool), layout, 1, 0.)))
    layout, ids = sample_layout((8,16), 8, 1)
    for special, raw in ((np.nan,None), (np.inf,None), (-np.inf,None), (-0.,None),
                         (1e-40,0x000116c2), (-1e-40,0x800116c2), (1e30,None), (-1e30,None)):
        values = np.ones(len(ids), np.float32)
        if raw is None:
            values[0] = special
        else:
            # Avoid the host's double-to-float conversion flushing the fixture
            # itself to zero before the candidate ever receives it.
            values.view(np.uint32)[0] = raw
        row = compare(candidate, values, np.ones((8,16),bool), layout, 1, .5)
        if not row['fallback']:
            raise AssertionError('Special numeric fallback not exercised: '+repr(special))
        rows.append(dict(name=f'special_{len(rows)}', **row))
    for dtype in (np.float32, np.float64):
        for floor in (-1., -0., np.nan, np.inf, .5):
            rows.append(dict(name=f'generic_{len(rows)}', **compare(candidate,
                np.ones(len(ids),dtype), np.ones((8,16),bool), layout, 1, floor)))
    if include_large:
        shape = (3190, 4784)
        layout, ids = sample_layout(shape, 256, 4)
        for i in range(4):
            values = rng.normal(0, 3, len(ids)).astype(np.float32)
            support = np.ones(shape, bool)
            if i == 1:
                support[:6] = False; support[-6:] = False
                support[:,:6] = False; support[:,-6:] = False
            elif i == 2:
                support = (rng.integers(0, 256, shape, dtype=np.uint8)).view(np.bool_)
            elif i == 3:
                support[:] = False
            rows.append(dict(name=f'native_geometry_{i}', **compare(candidate, values, support, layout, 4, .5)))
    return rows


def run(library, output):
    candidate = NoiseV18(library)
    record = dict(passed=False, error=None, real_media_read=False,
                  reference_sha256=REFERENCE_SHA, library_sha256=sha(library),
                  source_sha256={name:sha(HERE/name) for name in
                      ('noise_v18.py','noise_v18.cpp','build_noise_v18.py','check_noise_v18.py')},
                  numpy_version=np.__version__, cases=[], timings=[])
    plan = HERE/'visible_speed_v18_plan.md'
    if not plan.exists():
        plan = HERE.parent/'docs/visible_speed_v18_plan.md'
    record['plan_sha256'] = sha(plan)
    try:
        record['cases'] = cases(candidate)
        layout, ids = sample_layout((3190,4784),256,4)
        values = np.random.default_rng(891).normal(0,3,len(ids)).astype(np.float32)
        support = np.ones((3190,4784),bool)
        expected = reference(values,support,layout,4,.5)
        for repeat in range(4):
            for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference')):
                function = reference if mode=='reference' else candidate
                start = time.perf_counter()
                actual = function(values,support,layout,4,.5)
                elapsed = 1000*(time.perf_counter()-start)
                if actual[0].tobytes()!=expected[0].tobytes() or actual[1]!=expected[1]:
                    raise AssertionError('Timing changed output')
                record['timings'].append(dict(repeat=repeat,mode=mode,ms=elapsed))
        record.update(passed=True, native_calls=candidate.calls, fallback_calls=candidate.fallbacks,
                      geometry_builds=candidate.geometry_builds)
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        write(output, record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.library,a.output)
