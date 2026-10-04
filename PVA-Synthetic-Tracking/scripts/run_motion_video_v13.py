"""Bounded real-input runner; frozen runtimes are read-only dependencies."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
VISIBLE = Path('/tmp/seaqr_phase20_decode_v10_Iz9RSF')
RAW_RESULTS = RUNTIME / 'results/final'
VISIBLE_IDS = ('0029', '0126', '0055', '0082')
RAW_IDS = ('0029', '0040')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def validate_request(branch, clip, frames, injected):
    if branch not in ('visible', 'raw'):
        raise ValueError('Unknown branch')
    if clip not in (VISIBLE_IDS if branch == 'visible' else RAW_IDS):
        raise ValueError('Outside frozen development scope')
    if branch == 'raw' and frames != 64:
        raise ValueError('RAW scope is exactly 64 frames')
    if branch == 'visible' and frames not in (None, 96, 128):
        raise ValueError('Unknown visible extent')
    if injected and (branch != 'raw' or clip != '0040'):
        raise ValueError('Only unchanged RAW0040 controls')


def rss():
    return {row.split(':')[0]: row.split(':')[1].strip()
            for row in Path('/proc/self/status').read_text().splitlines()
            if row.startswith(('VmRSS:', 'VmHWM:'))}


def run(args):
    # Authorize scope before importing a source reader or accessing any media.
    validate_request(args.branch, args.clip, args.frames, args.injected)
    if args.output.exists() or args.output.with_suffix('.execution.json').exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(RUNTIME), str(RUNTIME/'scripts')]
    from motion_reuse_v12 import ReuseMotionV12, generated_method
    from profile_raw16_efficiency import compact
    from tiny_target import motion, dense_screen
    from tiny_target.motion import pva_pyrlk as pva
    import run_exact_v9 as raw
    from tiny_target import visible_baseline

    if sha(Path(pva.__file__)) != '7b96fef337350500835a64be3f43e32651c52ffb18a637a1b35ed0d4c627949f':
        raise ValueError('Unknown runtime')
    adapter = Path(sys.modules['motion_reuse_v12'].__file__)
    rows, instances = [], []
    base = pva.PvaPyrLkMotionEstimator if args.mode == 'reference' else ReuseMotionV12

    class Recorded(base):
        def __init__(self, config):
            super().__init__(config)
            instances.append(self)

        def estimate(self, previous, current):
            started = time.perf_counter()
            try:
                result = super().estimate(previous, current)
            except BaseException as exc:
                rows.append(dict(frame=current.frame_index, error=repr(exc)))
                raise
            elapsed = time.perf_counter()-started
            rows.append(dict(frame=current.frame_index, identity=compact(result),
                             estimator_s=elapsed,
                             rss=rss() if current.frame_index % 32 == 0 else None))
            return result

    record = dict(branch=args.branch, clip=args.clip, mode=args.mode, frames=args.frames,
                  injected=args.injected, passed=False, error=None, closed=False,
                  wrapper_sha256=sha(__file__), adapter_sha256=sha(adapter),
                  method_sha256=hashlib.sha256(generated_method().encode()).hexdigest(),
                  runtime_sha256={str(p.relative_to(RUNTIME)): sha(p)
                                  for p in sorted((RUNTIME/'tiny_target').rglob('*.py'))
                                  if not p.name.startswith('._')},
                  defaults_changed=False, production_approved=False)
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            for module in (motion, dense_screen, raw.v6.sequence):
                stack.enter_context(patch.object(module, 'PvaPyrLkMotionEstimator', Recorded))
            if args.branch == 'visible':
                frozen = read(VISIBLE/'freeze.json')
                source = frozen['sources'][args.clip]
                expected_path = '/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_'+args.clip+'.avi'
                if source['path'] != expected_path or Path(expected_path).is_symlink():
                    raise ValueError('Source redirected')
                if sha(expected_path) != source['sha256']:
                    raise ValueError('Source identity changed')
                cfg = VISIBLE/'config.json'
                if sha(cfg) != frozen['config_sha256']:
                    raise ValueError('Visible configuration changed')
                motion_cfg = RUNTIME/'configs/evaluation/phase20_motion_v8.json'
                if sha(motion_cfg) != read(VISIBLE/('pva_'+args.clip)/'launch.json')['motion_config_sha256']:
                    raise ValueError('Visible motion configuration changed')
                report = visible_baseline.run(Path(expected_path), cfg, args.output, motion_cfg, args.frames)
                record['pipeline_fps'] = report['processed_fps']
                record['processed_frames'] = report['frames']
                record['passed'] = report['completed']
            else:
                call = argparse.Namespace(clip=args.clip, mode='exact', output=args.output,
                    archive=RUNTIME/'verified_runtime.tgz', v7_archive=RUNTIME/'final_sources.tgz',
                    v8_archive=RUNTIME/'v8_sources.tgz', component=RUNTIME/'results/generated/full_01.json',
                    motion_controls=RUNTIME/'results/motion_generated',
                    evidence=Path('/tmp/seaqr_raw16_cpu_v6_rW7HL8/results/final'),
                    v8_results=Path('/tmp/seaqr_raw16_speed_v8_boh3Kh/results/final'),
                    audit=False, profile=False, injected=args.injected)
                record['passed'] = raw.run(call) == 0
                record['processed_frames'] = 64
                record['pipeline_fps'] = 64/read(args.output/'checks.json')['elapsed_wall_s']
            if not record['passed']:
                raise AssertionError('Original pipeline gate failed')
            if len(rows) != record['processed_frames']-1:
                raise AssertionError('Missing motion calls')
            if args.mode == 'reuse' and sum(instance.hits for instance in instances) < len(rows)-1:
                raise AssertionError('Expected adjacent-frame reuse not exercised')
    except BaseException as exc:
        record['passed'] = False
        record['error'] = repr(exc)
        raise
    finally:
        try:
            for instance in instances:
                if args.mode == 'reuse':
                    instance.close()
                else:
                    instance._stream.sync()
            record['closed'] = True
        finally:
            record.update(process_wall_s=time.perf_counter()-started,
                          reuse_hits=sum(getattr(i, 'hits', 0) for i in instances),
                          reuse_misses=sum(getattr(i, 'misses', 0) for i in instances),
                          instances=len(instances), rss_final=rss(), motion=rows)
            write(args.output.with_suffix('.execution.json'), record)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('visible', 'raw'), required=True)
    parser.add_argument('--mode', choices=('reference', 'reuse'), required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--injected', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
