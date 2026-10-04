"""Observe why the frozen injected controls lose support; never retune them."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

import validate_raw16_full_frame as validation
from profile_raw16_efficiency import sha, write_json
from tiny_target.dense_screen import DensePointScreener
from tiny_target.evaluation import SyntheticInjector


def run(output):
    observations = []
    original_inject = SyntheticInjector.inject
    original_events = DensePointScreener._events_for_frame
    pending = {}

    def inject(injector, frame):
        result = original_inject(injector, frame)
        pending.clear()
        limit = ((1 << frame.bit_depth) - 1) * .995
        for target in injector.spec.targets:
            if not target.active(frame.frame_index):
                continue
            tx, ty = target.position_at(frame.timestamp_ns)
            x, y = round(tx), round(ty)
            ys, xs = slice(y - 3, y + 4), slice(x - 3, x + 4)
            before = frame.image[ys, xs]
            after = result.image[ys, xs]
            pending[target.target_id] = dict(frame_index=frame.frame_index, target_id=target.target_id,
                xy=[tx, ty], original_center_dn=float(frame.image[y, x]),
                injected_center_dn=float(result.image[y, x]),
                original_patch_min_max_dn=[float(before.min()), float(before.max())],
                saturated_patch_pixels_before=int(np.count_nonzero(before >= limit)),
                saturated_patch_pixels_after=int(np.count_nonzero(after >= limit)),
                warp_valid_patch_pixels=int(np.count_nonzero(frame.valid_mask[ys, xs])) if frame.valid_mask is not None else 49,
                sensor_saturation_threshold_dn=limit)
        return result

    def events(screener, frame):
        result = original_events(screener, frame)
        matched = screener._last_synthetic_frame
        for item in pending.values():
            x, y = (round(v) for v in item['xy'])
            ys, xs = slice(y - 3, y + 4), slice(x - 3, x + 4)
            entry = dict(item, filter_state=screener._availability[-1]['filter_state'])
            if matched is not None:
                entry.update(center_filter_valid=bool(matched.valid_mask[y, x]),
                             filter_valid_patch_pixels=int(np.count_nonzero(matched.valid_mask[ys, xs])),
                             point_response_at_center=float(matched.response[y, x]),
                             center_background_sigma_dn=float(np.sqrt(screener._background_variance[y, x])))
            observations.append(entry)
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(SyntheticInjector, 'inject', inject))
        stack.enter_context(patch.object(DensePointScreener, '_events_for_frame', events))
        code = validation.run(argparse.Namespace(clip='0040', injected=True, output=output))
    write_json(output / 'control_support_trace.json', dict(script_sha256=sha(__file__),
        observations=observations,
        warning='Diagnostic repeat of the same frozen control configuration; no source, target or detection-policy changes.'))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
