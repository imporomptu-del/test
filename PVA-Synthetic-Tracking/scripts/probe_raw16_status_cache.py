"""Generated-only diagnostic of VPI constructor/status reuse. Not a runtime fix."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from check_raw16_motion_controls import cases, evaluate_mode, frame
from profile_raw16_efficiency import write_json, sha
from tiny_target.config import load_config
from tiny_target.motion import PvaPyrLkMotionEstimator, GlobalMotionConfig


def flags(array):
    if array is None:
        return None
    with array.rlock_cpu() as data:
        a = np.array(data, copy=True)
    return dict(id=array.id, shape=list(a.shape), nonzero=int(np.count_nonzero(a)))


class VpiProxy:
    def __init__(self, vpi, log, host_wrap=False):
        self.vpi, self.log, self.host_wrap = vpi, log, host_wrap

    def __getattr__(self, name):
        return getattr(self.vpi, name)

    def OpticalFlowPyrLK(self, *args, **kwargs):
        initial = kwargs.get('kptstatus')
        owner = None
        if self.host_wrap and initial is not None:
            with initial.rlock_cpu() as data:
                owner = np.array(data, dtype=np.uint8, copy=True).reshape(-1)
            initial = self.vpi.asarray(owner)
            kwargs['kptstatus'] = initial
        row = dict(initial=flags(initial))
        flow = self.vpi.OpticalFlowPyrLK(*args, **kwargs)
        row['flow_id'] = flow.id
        self.log.append(row)

        def call(*args, **kwargs):
            _keep_host_owner_alive = owner
            result = flow(*args, **kwargs)
            kwargs['stream'].sync()
            row['initial_after'] = flags(initial)
            row['returned_status'] = flags(result[1])
            return result
        return call


def run(output):
    if output.exists():
        raise FileExistsError(output)
    cfg = load_config(ROOT/'configs/evaluation/raw16_motion_v5.json').raw
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    clean = next(c for c in cases(75407) if c[0] == 'dim_subpixel')
    unrelated = next(c for c in cases(75413) if c[0] == 'independent_noise')
    modes = {}
    for clear, host_wrap in ((False, False), (True, False), (False, True)):
        log = []
        estimator = PvaPyrLkMotionEstimator(cfg['motion'])
        estimator._vpi = VpiProxy(estimator._vpi, log, host_wrap)
        observations = []
        for name, a, b, truth in (clean, unrelated, clean):
            if clear:
                estimator._vpi.clear_cache()
            result = evaluate_mode(estimator, fitter, frame(a, 0, name), frame(b, 1, name), truth)
            observations.append(result)
            print('clear', clear, 'host_wrap', host_wrap, name, result.get('correspondences', {}).get('accepted_count'), flush=True)
        exact = observations[0]['correspondences']['correspondences'] == observations[2]['correspondences']['correspondences']
        modes[f'clear={clear},host_wrap={host_wrap}'] = dict(observations=observations, flow_calls=log, identical_clean_replay=exact)
        print('exact clean replay', exact, flush=True)
    write_json(output, dict(modes=modes, script_sha256=sha(__file__),
        warning='Generated-only cache ablation with extra synchronization. Not a performance result or a promoted workaround.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
