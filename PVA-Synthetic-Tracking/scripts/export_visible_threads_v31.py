"""Export complete v31 evidence without source media or redundant runtime copies."""
import argparse
from pathlib import Path
import tarfile
from profile_visible_interaction_v30 import read, write, sha


def export(directory, archive):
    if archive.exists() or (directory/'export_manifest.json').exists():
        raise FileExistsError('Fresh evidence export required')
    batch = read(directory/'batch.json')
    if not batch['passed'] or batch['error'] is not None or read(directory/'status.json')['running'] or len(batch['rows']) != len(batch['schedule']):
        raise ValueError('Completed stopped experiment required')
    frozen = read(directory/'freeze.json')
    files = list(frozen['sources'])+['freeze.json','unit.log','generated.json','generated.log',
        'batch.json','batch.log','status.json','export_visible_threads_v31.py']
    for spec in batch['schedule']:
        n=spec['name']
        files += [n+suffix for suffix in ('.log','.v29.json','.v31.json','.execution.json','/launch.json','/report.json','/frames.jsonl')]
        if spec['traced']:files += [n+'.trace30.json',n+'.sqlite']
    if len(files)!=len(set(files)):raise ValueError('Duplicate artifact')
    for name in files:
        path=directory/name
        if not path.is_file() or path.is_symlink() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Expected regular relative evidence file '+name)
    write(directory/'export_manifest.json',dict(post_run=True,media_included=False,files={n:sha(directory/n) for n in files}))
    with tarfile.open(archive,'x:gz') as bundle:
        for name in files+['export_manifest.json']:bundle.add(directory/name,arcname=name,recursive=False)
    print(sha(archive),archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True,type=Path); p.add_argument('--archive',required=True,type=Path)
    a=p.parse_args();export(a.directory,a.archive)
