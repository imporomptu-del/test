"""Fail closed on any non-timing journal change in an execution-only trial."""
from dataclasses import asdict
from itertools import zip_longest
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256
from tiny_target.visible_decode import decode_contract

TIMING_KEYS = {"timings_ms", "warp_timings_ms", "pva_timings_ms", "timing_ms", "detection_ms"}
EXECUTION_FIELDS = {'cuda_median_library', 'native_shape_library', 'native_shape_library_sha256', 'frame_decode_execution'}


def validate_decode(launch, report):
    execution = VisibleConfig(**launch['configuration']).frame_decode_execution
    contract = launch.get('frame_decode')
    if contract is None and execution == 'sequential' and 'frame_decode' not in report:
        return None  # Archived, synchronous implementation predates this field.
    if contract != decode_contract(execution):
        raise ValueError('Missing or changed decode provenance')
    stats = report.get('frame_decode', {})
    count = report['frames']
    if (stats.get('contract') != contract or stats.get('decoded_frames') != count
            or stats.get('consumed_frames') != count or stats.get('worker_joined') is not True
            or stats.get('capture_released') is not True or stats.get('dropped_frames') != 0
            or stats.get('read_calls') not in {count, count + 1}
            or stats.get('maximum_observed_frames_ahead') != int(execution == 'prefetch_one' and count > 0)):
        raise ValueError('Decode did not drain and close without frame loss')
    return stats


def transition_for_pair(left, right, frozen_transition):
    """An inherited GPU transition applies to accuracy, not same-binary timing."""
    transition = (None if left['exact_cuda_stabilization']['library_sha256'] ==
        right['exact_cuda_stabilization']['library_sha256'] else frozen_transition)
    return validate_gpu_transition(left, right, transition)


def shape_accelerator(launch):
    cfg = VisibleConfig(**launch['configuration'])
    provenance = launch.get('external_accelerators', {}).get('shape')
    if cfg.native_shape_library is None:
        if provenance is not None:
            raise ValueError('Unexpected native shape provenance')
        return None
    if (not isinstance(provenance, dict) or provenance.get('abi') != 1
            or provenance.get('library_path') != cfg.native_shape_library
            or provenance.get('library_sha256') != cfg.native_shape_library_sha256
            or provenance.get('backend') != 'native_cpu_bookkeeping'
            or provenance.get('fallback') is not False
            or provenance.get('centroid_reductions') != 'NumPy float64 reference order'
            or provenance.get('radius_px') != 8):
        raise ValueError('Missing or changed native shape provenance')
    return provenance


def without_timing(value):
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items()
                if k not in TIMING_KEYS}
    return value


def validate_gpu_transition(left, right, transition):
    before=left['exact_cuda_stabilization']['library_sha256']
    after=right['exact_cuda_stabilization']['library_sha256']
    if before==after and transition is None:
        return None
    if not isinstance(transition,dict):
        raise ValueError('Compiled GPU library changed')
    expected={'schema','before_library_sha256','after_library_sha256','candidate_build_sha256'}
    if (set(transition)!=expected or transition['schema']!='seaqr.exact-gpu-transition.v1'
            or transition['before_library_sha256']!=before or transition['after_library_sha256']!=after
            or before==after):
        raise ValueError('GPU transition does not match explicit frozen binary pair')
    for key in expected-{'schema'}:
        value=transition[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('GPU transition requires complete SHA256 provenance')
    return transition


def compare(before, after, output, *, gpu_transition=None):
    left = json.loads((before / "launch.json").read_text())
    right = json.loads((after / "launch.json").read_text())
    a, b = [json.loads((p / "report.json").read_text()) for p in (before, after)]
    if not all(r["completed"] and r["full_clip"] for r in (a, b)) or a["frames"] != b["frames"]:
        raise ValueError("Expected equal complete full clips")
    if left["source_sha256"] != right["source_sha256"] or left["fps"] != right["fps"]:
        raise ValueError("Source or cadence changed")
    configs = [asdict(VisibleConfig(**r["configuration"])) for r in (left, right)]
    changes = {k: [configs[0][k], configs[1][k]] for k in configs[0] if configs[0][k] != configs[1][k]}
    if set(changes) - EXECUTION_FIELDS:
        raise ValueError("Execution-only comparison changed algorithm policy")
    native_libraries = dict(before=shape_accelerator(left), after=shape_accelerator(right))
    decoding = dict(before=validate_decode(left, a), after=validate_decode(right, b))
    gpu_transition=validate_gpu_transition(left,right,gpu_transition)
    differences, count = [], 0
    with (before / "frames.jsonl").open() as first, (after / "frames.jsonl").open() as second:
        for i, pair in enumerate(zip_longest(first, second)):
            if None in pair:
                raise ValueError("Journal length changed")
            rows = [json.loads(line) for line in pair]
            if any(r["frame_index"] != i for r in rows):
                raise ValueError("Journal is not contiguous")
            reduced = [without_timing(r) for r in rows]
            for key in set(reduced[0]) | set(reduced[1]):
                if key not in reduced[0] or key not in reduced[1] or reduced[0][key] != reduced[1][key]:
                    differences.append([i, key])
            count += 1
    if count != a["frames"]:
        raise ValueError("Truncated journal")
    result = dict(exact=not differences, frames=count, difference_count=len(differences),
        first_differences=differences[:20], fields="All journal fields; only explicitly named timing fields excluded",
        ignored_timing_keys=sorted(TIMING_KEYS),
        before_fps=a["processed_fps"], after_fps=b["processed_fps"],
        speedup=b["processed_fps"] / a["processed_fps"],
        timings_ms=dict(before=a["timings_ms"], after=b["timings_ms"]),
        configuration_changes=changes, native_shape_accelerators=native_libraries,
        frame_decode=decoding,
        explicit_gpu_transition=gpu_transition,
        reference_sha256=sha256(before / "frames.jsonl"),
        output_sha256=sha256(after / "frames.jsonl"), comparator_sha256=sha256(__file__),
        timing_caveat="Sequential matching full clips, not repeated thermal/load-controlled distributions.")
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    if differences:
        raise AssertionError("Non-timing output changed: " + str(differences[:10]))
    return result
