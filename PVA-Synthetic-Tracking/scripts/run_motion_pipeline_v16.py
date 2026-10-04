"""Isolated full RAW pipeline regression; frozen dependencies remain read-only."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
FRONT = Path('/tmp/seaqr_motion_front_v15_boL2EI')
ARCHIVE = Path('/tmp/seaqr_video_v13_IyK7eQ/results')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def validate_scope(clip, mode, injected):
    if clip not in ('0029', '0040') or mode not in ('reference', 'candidate'):
        raise ValueError('Outside frozen experiment scope')
    if injected and clip != '0040':
        raise ValueError('Only unchanged 0040 control is allowed')


def run(args):
    validate_scope(args.clip, args.mode, args.injected)
    if args.output.exists() or args.output.with_suffix('.execution.json').exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(FRONT), str(RUNTIME), str(RUNTIME/'scripts')]
    from run_motion_front_real_v15 import verify_gate
    gate_path = FRONT/'generated_03.json'
    verify_gate(gate_path)
    from motion_front_v15 import DirectMotionV15
    from motion_front_pixels_v15 import logical_identity
    from motion_reuse_v12 import ReuseMotionV12
    from profile_raw16_efficiency import compact
    from tiny_target import motion, dense_screen
    import run_exact_v9 as raw
    # Imported by absolute local harness path; no dependency module is edited.
    sys.path.insert(0, str(HERE))
    from summarize_motion_front_v15 import validate_execution_fields

    rows, instances = [], []
    base = ReuseMotionV12 if args.mode == 'reference' else DirectMotionV15

    class Recorded(base):
        def __init__(self, config):
            super().__init__(config)
            instances.append(self)

        def estimate(self, previous, current):
            started = time.perf_counter()
            result = super().estimate(previous, current)
            elapsed = time.perf_counter()-started
            identity = compact(result)
            validate_execution_fields(identity, args.mode)
            rows.append(dict(frame=current.frame_index, identity=identity, estimator_s=elapsed))
            return result

    normalize = raw.normalized_report
    record = dict(schema='seaqr.motion-pipeline-v16-run.v1', clip=args.clip,
                  mode=args.mode, injected=args.injected, profile=args.profile,
                  passed=False, error=None, closed=False, defaults_changed=False,
                  production_approved=False, wrapper_sha256=sha(__file__),
                  gate_sha256=sha(gate_path), plan_sha256=sha(HERE/'motion_pipeline_v16_plan.md'))
    try:
        with ExitStack() as stack:
            for module in (motion, dense_screen, raw.v6.sequence):
                stack.enter_context(patch.object(module, 'PvaPyrLkMotionEstimator', Recorded))
            # Only v15's two documented execution metadata fields are removed;
            # all original exact-v9 gates and numerical comparisons still run.
            stack.enter_context(patch.object(raw, 'normalized_report', lambda report: logical_identity(normalize(report))))
            call = argparse.Namespace(clip=args.clip, mode='exact', output=args.output,
                archive=RUNTIME/'verified_runtime.tgz', v7_archive=RUNTIME/'final_sources.tgz',
                v8_archive=RUNTIME/'v8_sources.tgz', component=RUNTIME/'results/generated/full_01.json',
                motion_controls=RUNTIME/'results/motion_generated',
                evidence=Path('/tmp/seaqr_raw16_cpu_v6_rW7HL8/results/final'),
                v8_results=Path('/tmp/seaqr_raw16_speed_v8_boh3Kh/results/final'),
                audit=False, profile=args.profile, injected=args.injected)
            if raw.run(call) != 0:
                raise AssertionError('Original exact pipeline gates failed')
        archived = (RUNTIME/'results/final/injected_0040_exact' if args.injected
                    else ARCHIVE/f'raw_{args.clip}_repeat0_reuse')
        for name in ('source_frames.json', 'candidate_decisions.json', 'global_fit_identities.json'):
            if read(args.output/name) != read(archived/name):
                raise AssertionError('Archived full output changed: '+name)
        if not args.injected:
            old = read(archived.with_suffix('.execution.json'))['motion']
            if [logical_identity(r['identity']) for r in rows] != [logical_identity(r['identity']) for r in old]:
                raise AssertionError('Complete archived motion identity changed')
        if len(rows) != 63 or len(instances) != 1 or instances[0].hits != 62 or instances[0].misses != 1:
            raise AssertionError('Incomplete estimator/reuse lifecycle')
        checks = read(args.output/'checks.json')
        record.update(passed=True, wall_s=checks['elapsed_wall_s'],
                      fps=64/checks['elapsed_wall_s'], checks=checks['checks'])
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        for instance in instances:
            instance.close()
        record.update(closed=all(i.closed for i in instances), motion=rows,
                      reuse_hits=sum(i.hits for i in instances), reuse_misses=sum(i.misses for i in instances))
        write(args.output.with_suffix('.execution.json'), record)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--injected', action='store_true')
    parser.add_argument('--profile', action='store_true')
    raise SystemExit(run(parser.parse_args()))
