"""Allowlisted post-run v34 evidence export; no input media or runtime copies."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

SOURCES = ('visible_validation_v34.py', 'run_visible_validation_v34.py',
    'test_visible_validation_v34.py', 'test_run_visible_validation_v34.py',
    'visible_validation_v34_plan.md', 'start_visible_validation_v34_tmux.sh',
    'v33_audit.json', 'audit_visible_clocks_v33.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def files_for(schedule):
    files = list(SOURCES) + ['freeze.json', 'unit.log', 'dependency_preflight.json',
        'export_visible_validation_v34.py']
    files += ['run/' + n for n in ('batch.json', 'status.json', 'original_policy.json',
        'run_identity.json', 'controller_restoration.json', 'watchdog_restoration.json',
        'transition_preflight.json', 'telemetry.jsonl')]
    files += [f'run/transitions/preflight_{i}_{mode}.json'
        for i in range(3) for mode in ('fixed', 'auto')]
    for spec in schedule:
        n = spec['name']
        if not n or Path(n).name != n or n in ('.', '..'):
            raise ValueError('Unsafe trial name')
        files += ['run/' + n + suffix for suffix in ('.log', '.v29.json', '.v34.json',
            '.execution.json', '/launch.json', '/report.json', '/frames.jsonl')]
        files += ['run/transitions/' + n + '.json']
        if spec['traced']:
            files += ['run/' + n + suffix for suffix in ('.trace30.json', '.sqlite', '.nsys-rep')]
    if len(files) != len(set(files)):
        raise ValueError('Duplicate evidence path')
    return files


def export(directory, archive):
    directory = directory.resolve()
    manifest = directory / 'export_manifest_v34_01.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError('Fresh export required')
    read = lambda p: json.loads(p.read_text())
    batch, status = read(directory/'run/batch.json'), read(directory/'run/status.json')
    if (not batch['completed'] or batch['error'] is not None or status['running']
            or len(batch['rows']) != 16 or status['completed'] != 16):
        raise ValueError('Completed stopped v34 required')
    files = files_for(batch['schedule'])
    for name in files:
        p = directory/name
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(directory):
            raise ValueError('Expected regular evidence file ' + name)
    data = dict(post_run=True, media_included=False,
        files={name: sha(directory/name) for name in files})
    with manifest.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
    with tarfile.open(archive, 'x:gz') as bundle:
        for name in files + [manifest.name]:
            bundle.add(directory/name, arcname=name, recursive=False)
    print(json.dumps(dict(archive=str(archive), sha256=sha(archive),
        files=len(files)+1, bytes=archive.stat().st_size)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--archive', required=True, type=Path)
    a = p.parse_args()
    export(a.directory, a.archive)
