"""Post-run supplemental tests and compact, hash-manifested evidence export."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile

from profile_visible_v17 import read, sha, write
from freeze_tracking_v28 import SOURCES

EXTRA = ('test_tracking_stage_v28_failures.py', 'test_kalman_tracking.py',
         'finalize_tracking_v28.py')


def finish(archive):
    here = Path(__file__).resolve().parent
    if archive.exists() or (here / 'post_run_01.json').exists() or (here / 'supplemental_unit_01.json').exists():
        raise FileExistsError('Existing completed artifact')
    run = read(here / 'replays_01.json')
    if not run['passed'] or run['error'] is not None or len(run['replays']) != 12 or len(run['profiles']) != 2:
        raise ValueError('Replay experiment incomplete')
    freeze = read(here / 'freeze_01.json')
    for path, expected in freeze['files'].items():
        if sha(path) != expected:
            raise ValueError('Dependency changed: ' + path)
    with (here / 'supplemental_unit_01.log').open('x') as log:
        done = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_tracking_stage_v28_failures'],
                              cwd=here, stdout=log, stderr=subprocess.STDOUT)
    write(here / 'supplemental_unit_01.json', dict(passed=done.returncode == 0,
        returncode=done.returncode, media_read=False, post_timing=True,
        source_sha256={n: sha(here / n) for n in SOURCES + EXTRA},
        log_sha256=sha(here / 'supplemental_unit_01.log')))
    if done.returncode:
        raise RuntimeError('Supplemental tests failed')
    names = SOURCES + EXTRA + ('freeze_01.json', 'replays_01.json', 'unit_01.log',
        'run_01.log', 'supplemental_unit_01.json', 'supplemental_unit_01.log')
    for name in names:
        path = here / name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Expected regular file: ' + name)
    write(here / 'post_run_01.json', dict(post_run=True, media_read=False,
        defaults_changed=False, files={n: sha(here / n) for n in names}))
    with tarfile.open(archive, 'x:gz') as bundle:
        for name in names + ('post_run_01.json',):
            bundle.add(here / name, arcname=name, recursive=False)
    print('Evidence archive', archive, sha(archive), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    finish(parser.parse_args().archive)
