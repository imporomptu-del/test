"""Generated CPU parity/microbenchmark against archived, hash-locked v5 source."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
import sys
import tarfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import sha, write_json, compact
from tiny_target.motion import geometry
from tiny_target.motion.pva_pyrlk import PvaMotionConfig, _feature_eligibility
from tiny_target.types import Frame, TimestampSource

BASELINE = ROOT/'results/tiny_target/raw16_motion_v5_20260915/verified_runtime.tgz'
BASELINE_SHA = '6b3ea3419ca60514c95bcf31e1eaf7fa706e83b19111ec0f59a95f3ef11ef286'


def frozen_oracle():
    if sha(BASELINE) != BASELINE_SHA:
        raise ValueError('Archived v5 reference changed')
    with tarfile.open(BASELINE) as archive:
        modules = []
        for name in ('geometry', 'pva_pyrlk'):
            module_name = 'tiny_target.motion._cpu_v5_oracle_' + name
            if module_name in sys.modules:
                modules.append(sys.modules[module_name])
                continue
            module = types.ModuleType(module_name)
            module.__package__ = 'tiny_target.motion'
            sys.modules[module_name] = module
            source = archive.extractfile('tiny_target/motion/' + name + '.py').read()
            exec(compile(source, str(BASELINE) + ':' + name, 'exec'), module.__dict__)
            modules.append(module)
    old_geometry, old_motion = modules
    for name in ('lift_points_to_full_resolution', 'select_spatially_distributed', 'grid_coverage'):
        setattr(old_motion, name, getattr(old_geometry, name))
    return old_motion, old_geometry


def fixtures():
    rng = np.random.default_rng(693152)
    for name, shape, bits, count, radius, masked in (
        ('native_raw16', (3190, 4784), 16, 60000, 2, False),
        ('native_raw16_masked', (3190, 4784), 16, 60000, 2, True),
        ('odd_u8', (721, 1279), 8, 12000, 1, True),
        ('odd_raw12_radius8', (193, 317), 12, 5000, 8, True),
        ('large_radius_fallback', (39, 57), 16, 600, 12, False),
    ):
        image = rng.integers(0, 1 << bits, shape, dtype=np.uint16)
        if bits == 8:
            image = image.astype(np.uint8)
        mask = rng.random(shape) > .1 if masked else None
        source = Frame(image=image, bit_depth=bits, valid_mask=mask, timestamp_ns=0,
                       frame_index=0, source_id=name, timestamp_source=TimestampSource.MANIFEST)
        size = (max(1, shape[1]//2), max(1, shape[0]//2))
        points = rng.uniform([-1, -1], [size[0]+1, size[1]+1], (count, 2)).astype(np.float32)
        points[:5] = [[np.nan, 0], [0, np.inf], [-np.inf, 0], [0, 0], [size[0], size[1]]]
        scores = rng.integers(0, 1000, count).astype(np.float32)
        cfg = PvaMotionConfig(feature_border_px=0, saturated_neighborhood_radius_px=radius,
            exclusion_regions_xyxy=((13., 17., 31., 47.),))
        yield name, source, size, points, scores, cfg


def measure(source, size, points, scores, cfg, eligibility, selection, coverage, policy=None):
    started = time.perf_counter()
    eligible, reasons = eligibility(points, source, size, cfg)
    eligible_done = time.perf_counter()
    options = {} if policy is None else dict(execution=policy)
    selected = selection(points, scores, size, grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols,
        max_features=cfg.max_features, max_per_cell=cfg.max_features_per_cell,
        eligible_mask=eligible, **options)
    selected_done = time.perf_counter()
    grid = coverage(points, size, grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols, **options)
    finished = time.perf_counter()
    return compact(dict(eligible=eligible, reasons=reasons, selected=selected, coverage=grid)), dict(
        eligibility_ms=(eligible_done-started)*1000,
        selection_ms=(selected_done-eligible_done)*1000,
        coverage_ms=(finished-selected_done)*1000, total_ms=(finished-started)*1000)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    old_motion, old_geometry = frozen_oracle()
    rows = []
    for name, source, size, points, scores, cfg in fixtures():
        pixel_hash = source.pixel_sha256()
        observed, timings = {}, {}
        for repeat in range(3):
            order = ('frozen_v5', 'current_reference', 'batched')
            if repeat % 2:
                order = tuple(reversed(order))
            for mode in order:
                if mode == 'frozen_v5':
                    result, timing = measure(source, size, points, scores, cfg,
                        old_motion._feature_eligibility, old_geometry.select_spatially_distributed,
                        old_geometry.grid_coverage)
                else:
                    policy = 'reference' if mode == 'current_reference' else 'batched_exact_v1'
                    result, timing = measure(source, size, points, scores,
                        replace(cfg, feature_cpu_policy=policy), _feature_eligibility,
                        geometry.select_spatially_distributed, geometry.grid_coverage, policy)
                if mode in observed and result != observed[mode]:
                    raise ValueError('Repeated CPU output changed')
                observed[mode] = result
                timings.setdefault(mode, []).append(timing)
        exact = observed['frozen_v5'] == observed['current_reference'] == observed['batched']
        if pixel_hash != source.pixel_sha256():
            raise ValueError('Source mutation')
        medians = {mode: {key: statistics.median(t[key] for t in attempts)
                         for key in attempts[0]} for mode, attempts in timings.items()}
        rows.append(dict(name=name, shape=source.shape, bit_depth=source.bit_depth,
            points=len(points), exact=exact, outputs=observed, median_ms=medians,
            attempts=timings, source_pixel_sha256=pixel_hash))
        print(name, 'exact', exact, 'median total ms',
              {k:round(v['total_ms'], 3) for k,v in medians.items()}, flush=True)
    passed = all(r['exact'] for r in rows)
    write_json(output, dict(schema_version='seaqr.motion-cpu-parity.v1', passed=passed,
        cases=rows, baseline_sha256=BASELINE_SHA, script_sha256=sha(__file__),
        runtime_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target/motion').glob('*.py'))},
        python=platform.python_version(), numpy=np.__version__,
        warning='Generated CPU microbenchmark, not end-to-end throughput or target accuracy.'))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
