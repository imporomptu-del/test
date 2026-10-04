"""Generated-input selected-region CUDA cost/parity probe, not pipeline FPS."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from resident_tracking_v10 import ResidentTracker
from streaming_schedule_v14 import ScheduleConfig, required_halo

LIBRARY = Path('/tmp/seaqr_execution_v11_j10kKI/build/candidate_01/libresident_tracking_v11.so')
LIBRARY_SHA = 'f3b2912fea55e86e264f1b284f103fc055dcb70590bd9716cb7b164a717a2b4f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(arrays):
    return [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays]


def tile_set(n, config):
    # Explicit corners, seams, interior and partial bottom/right cores, then
    # spatially distributed choices. Fixed before measurements, not data driven.
    ids = list(dict.fromkeys([0, 18, 19, 123, 124, 228, 245, 246]+
                            np.linspace(0, config.tiles-1, 32, dtype=int).tolist()))
    return ids[:n]


def compare_core(expected, actual, core, padded):
    x0, y0, x1, y1 = core
    px0, py0, _, _ = padded
    reference = [a[y0:y1, x0:x1] for a in expected]
    observed = [a[y0-py0:y1-py0, x0-px0:x1-px0] for a in actual]
    fields = ('score', 'velocity', 'support', 'valid')
    counts = {field: int(np.count_nonzero(a != b)) for field, a, b in zip(fields, reference, observed)}
    return dict(exact=all(v == 0 for v in counts.values()), mismatched_pixels=counts,
                max_abs_score_difference=float(np.max(np.abs(reference[0]-observed[0]))),
                reference_sha256=digest(reference), observed_sha256=digest(observed))


def cold_region(frames, masks, ids, grid, polarity, rectangle):
    x0, y0, x1, y1 = rectangle
    start = time.perf_counter()
    device = ResidentTracker((y1-y0, x1-x0), grid, LIBRARY)
    try:
        if polarity != 'bright':
            device.reset(polarity=polarity)
        for i, source in enumerate(ids):
            device.push(frames[source, y0:y1, x0:x1], masks[source, y0:y1, x0:x1],
                        i, i*100000000, polarity=polarity)
        kernel_ms = device.run()
        result = device.download()
    finally:
        device.close()
    return result, 1000*(time.perf_counter()-start), kernel_ms


def run(output):
    if output.exists():
        raise FileExistsError(output)
    if sha(LIBRARY) != LIBRARY_SHA:
        raise ValueError('Existing frozen library changed')
    cfg = ScheduleConfig()
    if required_halo([i*100000000 for i in range(16)]) > cfg.halo:
        raise ValueError('Insufficient halo')
    grid = np.asarray([(x, y) for y in range(-3, 4) for x in range(-3, 4)
                       if (x, y) != (0, 0)], np.float32)
    report = dict(schema='seaqr.selective-gpu-v14.v1', configuration=asdict(cfg),
                  library_sha256=LIBRARY_SHA, real_media_read=False,
                  pipeline_benchmark=False, production_approved=False,
                  source_sha256={p: sha(Path(__file__).parent/p) for p in
                                 ('benchmark_selective_v14.py', 'resident_tracking_v10.py', 'streaming_schedule_v14.py')},
                  trials=[], passed=False, error=None,
                  exclusions=['source/camera decode', 'motion/warp', 'background/filter',
                              'full-frame history generation and maintenance', 'candidate extraction/association',
                              'correctness comparison/hash after timing'],
                  timing_boundary='ROI includes cold allocation, full 16-frame CPU response/mask packing and upload, '
                                  'kernel, download, close. Dense is persistent eight-frame advance plus full output download.',
                  warning='Component-only generated response-space test. No sensor precision/recall evidence.')
    rng = np.random.default_rng(161414)
    frames = rng.standard_normal((16, cfg.height, cfg.width), dtype=np.float32)
    masks = np.ones(frames.shape, dtype=np.bool_)
    masks[:, 1022:1027, 1275:1300] = False
    for i in range(16):
        masks[i, 3100+i:3104+i, 4620:4660] = False
        for x, y in ((128, 128), (255, 1024), (3190, 1600), (4700, 3120)):
            frames[i, y, x+round(.2*i)] += np.float32(2)
            frames[i, y+20, x+round(.1*i)] -= np.float32(2)
    report['source_history_bytes'] = frames.nbytes+masks.nbytes
    try:
        for polarity in ('bright', 'dark'):
            dense = ResidentTracker((cfg.height, cfg.width), grid, LIBRARY)
            expected = {}
            try:
                if polarity != 'bright':
                    dense.reset(polarity=polarity)
                for i in range(16):
                    dense.push(frames[i], masks[i], i, i*100000000, polarity=polarity)
                dense.run()
                expected[0] = dense.download()
                for i in range(16, 24):
                    dense.push(frames[i % 16], masks[i % 16], i, i*100000000, polarity=polarity)
                dense.run()
                expected[8] = dense.download()
                end = 23
                for repeat in range(2):
                    modes = ('dense', 'roi8', 'roi32') if repeat == 0 else ('roi32', 'roi8', 'dense')
                    for mode in modes:
                        samples = []
                        for cycle in range(3):
                            if mode == 'dense':
                                start = time.perf_counter()
                                for i in range(end+1, end+9):
                                    dense.push(frames[i % 16], masks[i % 16], i, i*100000000, polarity=polarity)
                                end += 8
                                kernel = dense.run()
                                arrays = dense.download()
                                elapsed = 1000*(time.perf_counter()-start)
                                phase = (end-15) % 16
                                same = digest(arrays) == digest(expected[phase])
                                sample = dict(cycle=cycle, elapsed_ms=elapsed, kernel_ms=kernel, exact=same,
                                              phase=phase, output_sha256=digest(arrays))
                            else:
                                phase = 8 if cycle % 2 == 0 else 0
                                ids = [(i+phase) % 16 for i in range(16)]
                                parts = []
                                for tile in tile_set(int(mode[3:]), cfg):
                                    core, padded = cfg.rect(tile), cfg.rect(tile, True)
                                    arrays, elapsed, kernel = cold_region(frames, masks, ids, grid, polarity, padded)
                                    comparison = compare_core(expected[phase], arrays, core, padded)
                                    parts.append(dict(tile=tile, core=core, padded=padded, elapsed_ms=elapsed,
                                                      kernel_ms=kernel, comparison=comparison))
                                sample = dict(cycle=cycle, phase=phase, elapsed_ms=sum(p['elapsed_ms'] for p in parts),
                                              kernel_ms=sum(p['kernel_ms'] for p in parts),
                                              exact=all(p['comparison']['exact'] for p in parts), parts=parts)
                            samples.append(sample)
                        report['trials'].append(dict(polarity=polarity, repeat=repeat, mode=mode, samples=samples))
                        print(json.dumps(dict(polarity=polarity, repeat=repeat, mode=mode,
                                              elapsed_ms=[s['elapsed_ms'] for s in samples],
                                              exact=all(s['exact'] for s in samples))), flush=True)
            finally:
                dense.close()
        report['passed'] = len(report['trials']) == 12 and all(s['exact'] for t in report['trials'] for s in t['samples'])
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        with output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
