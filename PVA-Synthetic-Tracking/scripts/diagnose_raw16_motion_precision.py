"""Generated-only diagnostic: identical native-U16 features, PVA vs CUDA LK.

This is not a production fallback. It isolates the optical-flow stage without
changing image inputs, fit gates or the PVA Harris/pyramid stages.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import vpi
from check_raw16_motion_controls import MOTION, MOTION_SHA256, cases, frame, evaluate_mode
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig


class CudaFlowOnly:
    def __getattr__(self, name):
        return getattr(vpi, name)

    def OpticalFlowPyrLK(self, *args, **kwargs):
        kwargs['backend'] = vpi.Backend.CUDA
        return vpi.OpticalFlowPyrLK(*args, **kwargs)


def main(output):
    if output.exists() or sha(MOTION) != MOTION_SHA256:
        raise ValueError('Existing output or modified frozen configuration')
    cv2.setNumThreads(2)
    cfg = load_config(MOTION).raw
    base = replace(PvaMotionConfig.from_mapping(cfg['motion']), feature_intensity_mapping='raw_linear_u16_v1')
    estimator = PvaPyrLkMotionEstimator(base)
    estimator._vpi = CudaFlowOnly()
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    results = []
    for name, previous, current, truth in cases():
        if name not in {'dim_translation', 'bright_translation', 'gain_and_offset_change', 'sensor_fixed_pattern'}:
            continue
        a, b = frame(previous, 0, name), frame(current, 1, name)
        result = evaluate_mode(estimator, fitter, a, b, truth)
        if 'correspondences' in result:
            result['correspondences']['backends']['optical_flow_pyrlk'] = 'CUDA'
        results.append(dict(name=name, expected_translation=truth, input_sha256=[a.pixel_sha256(), b.pixel_sha256()], result=result))
        print(name, result['accepted'], result.get('translation_error_px'), flush=True)
    write_json(output, dict(schema_version='seaqr.raw16-motion-flow-diagnostic.v1',
        diagnostic_only=True, flow_backend='CUDA', results=results,
        config_sha256=sha(MOTION), script_sha256=sha(__file__),
        estimator_sha256=sha(ROOT/'tiny_target/motion/pva_pyrlk.py')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
