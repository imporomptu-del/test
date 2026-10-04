"""Export completed v29 receipts/journals; never include source videos."""
import argparse
from pathlib import Path
import tarfile

from profile_visible_v17 import read, sha, write
from run_visible_combined_v29 import SOURCES


def export(directory, archive):
    if archive.exists() or (directory/'export_manifest.json').exists():
        raise FileExistsError('Fresh evidence export required')
    batch = read(directory/'batch.json')
    status = read(directory/'status.json')
    if (not batch['passed'] or batch['error'] is not None or status['running']
            or len(batch['rows']) != len(batch['schedule'])):
        raise ValueError('Experiment must finish before export')
    files = list(SOURCES) + ['freeze.json', 'unit_gate.json', 'unit_gate.log', 'batch.json',
        'batch.log', 'status.json', 'export_visible_combined_v29.py']
    for spec, row in zip(batch['schedule'], batch['rows']):
        name = spec['name']
        if Path(name).name != name or row['name'] != name or row['returncode'] != 0:
            raise ValueError('Invalid trial artifact scope')
        files += [name+'.v29.json', name+'.execution.json', name+'.log']
        files += [name+'/'+f for f in ('launch.json', 'report.json', 'frames.jsonl')]
    if len(files) != len(set(files)):
        raise ValueError('Duplicate artifact path')
    for name in files:
        path = directory/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Expected regular evidence file: '+name)
    write(directory/'export_manifest.json', dict(post_run=True, includes_media=False,
        files={n: sha(directory/n) for n in files},
        note='Only source, protocol, receipts and output journals. Runtime snapshots are unchanged and archived in dependencies.'))
    with tarfile.open(archive, 'x:gz') as bundle:
        for name in files+['export_manifest.json']:
            bundle.add(directory/name, arcname=name, recursive=False)
    print('Evidence archive', archive, sha(archive), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    export(args.directory, args.archive)
