"""Explicit allowlist export of completed v33 evidence; no input media."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

SOURCES = ('visible_clocks_v33.py', 'test_visible_clocks_v33.py',
    'visible_clocks_v33_plan.md', 'start_visible_clocks_v33_tmux.sh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_for(schedule):
    files = list(SOURCES)+['freeze.json', 'unit.log', 'dependency_preflight.json',
        'export_visible_clocks_v33.py']
    files += ['run/'+n for n in ('batch.json','status.json','original_policy.json',
        'run_identity.json','controller_restoration.json','watchdog_restoration.json',
        'transition_preflight.json','telemetry.jsonl')]
    files += [f'run/transitions/preflight_{i}_{mode}.json' for i in range(3) for mode in ('fixed','auto')]
    for s in schedule:
        n=s['name']
        if not n or Path(n).name != n or n in ('.','..'):
            raise ValueError('Unsafe trial name')
        files += ['run/'+n+suffix for suffix in ('.log','.v29.json','.v31.json',
            '.execution.json','/launch.json','/report.json','/frames.jsonl')]
        files += ['run/transitions/'+n+'.json']
    if len(files)!=len(set(files)):
        raise ValueError('Duplicate evidence path')
    return files


def export(directory, archive):
    directory=directory.resolve()
    manifest=directory/'export_manifest_v33_01.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError('Fresh export required')
    read=lambda p: json.loads(p.read_text())
    b=read(directory/'run/batch.json'); status=read(directory/'run/status.json')
    if not b['completed'] or b['error'] is not None or status['running'] or len(b['rows'])!=50:
        raise ValueError('Completed stopped v33 required')
    files=files_for(b['schedule'])
    for n in files:
        p=directory/n
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(directory):
            raise ValueError('Expected regular evidence file '+n)
    data=dict(post_run=True, media_included=False, files={n:sha(directory/n) for n in files})
    with manifest.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)
    with tarfile.open(archive,'x:gz') as bundle:
        for n in files+[manifest.name]:
            bundle.add(directory/n,arcname=n,recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(files)+1,
                         bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True); p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args(); export(a.directory,a.archive)
