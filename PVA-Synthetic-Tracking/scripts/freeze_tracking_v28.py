"""Write a fresh pre-run hash manifest for journal-only v28 execution."""
import argparse
from pathlib import Path
import time

from profile_visible_v17 import read, sha, write

SOURCES = ('tracking_stage_v28.py', 'check_tracking_stage_v28.py', 'freeze_tracking_v28.py',
           'tracking_batch_v27.py', 'tracking_geometry_v20.py', 'replay_tracking_v27.py',
           'profile_visible_v17.py', 'tracking_v28_plan.md', 'test_tracking_stage_v28.py',
           'build_tracking_batch_v27.py', 'build_tracking_geometry_v20.py',
           'tracking_batch_v27.cpp', 'tracking_geometry_v20.cpp')


def freeze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    here = Path(__file__).resolve().parent
    files = {str(here / name): sha(here / name) for name in SOURCES}
    files.update({str(args.library): sha(args.library), str(args.geometry): sha(args.geometry)})
    import tiny_target
    package = Path(tiny_target.__file__).parent
    for clip in ('0126', '0082'):
        parent = args.parents / (clip + '_repeat0_reference')
        launch = read(parent / 'launch.json')
        report = read(parent / 'report.json')
        baseline_path = args.baselines / f'profile_{clip}_01.json'
        baseline = read(baseline_path)
        if (Path(launch['source']).stem != 'chunk_' + clip or launch['max_frames'] != 128
                or not report['completed'] or report['full_clip'] or report['frames'] != 128
                or not baseline['passed'] or baseline['error'] is not None):
            raise ValueError('Incomplete or out-of-scope development fixture')
        for name in ('launch.json', 'report.json', 'frames.jsonl'):
            path = parent / name
            actual = sha(path)
            field = {'launch.json': 'launch_sha256', 'report.json': 'report_sha256',
                     'frames.jsonl': 'journal_sha256'}[name]
            if actual != baseline['replay'][field]:
                raise ValueError('Fixture changed from checked v27 baseline')
            files[str(path)] = actual
        files[str(baseline_path)] = sha(baseline_path)
        for name, expected in launch['package_sha256'].items():
            path = package / name
            if sha(path) != expected:
                raise ValueError('Frozen runtime changed: ' + name)
            files[str(path)] = expected
    write(args.output, dict(pre_run=True, created_ns=time.time_ns(), media_read=False,
        defaults_changed=False, frames_per_prefix=128, clips=['0126', '0082'],
        schedule=['v20', 'v27', 'v28', 'v28', 'v27', 'v20'], files=files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'geometry', 'parents', 'baselines', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    freeze(parser.parse_args())
