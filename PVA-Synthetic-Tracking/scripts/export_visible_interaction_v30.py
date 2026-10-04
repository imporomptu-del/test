"""Export a completed diagnostic batch and preserve the failed telemetry attempt."""
import argparse
from pathlib import Path
import tarfile
from profile_visible_interaction_v30 import read, write, sha


def export(directory, archive):
    if archive.exists() or (directory/'export_manifest.json').exists():
        raise FileExistsError('Fresh export required')
    batch = read(directory/'batch.json')
    if not batch['passed'] or batch['error'] is not None or len(batch['rows']) != 12 or read(directory/'status.json')['running']:
        raise ValueError('Complete stopped diagnostic batch required')
    freeze = read(directory/'freeze.json')
    files = {n: directory/n for n in list(freeze['sources']) + ['freeze.json', 'unit.log',
        'telemetry_probe.json', 'batch.json', 'batch.log', 'status.json', 'export_visible_interaction_v30.py']}
    for spec in batch['schedule']:
        name = spec['name']
        suffixes = ['.log', '.v29.json', '.execution.json', '/launch.json', '/report.json', '/frames.jsonl']
        if spec['mode'] == 'trace':
            suffixes += ['.sqlite', '.trace30.json', '.tegrastats.log']
        for ending in suffixes:
            files[name+ending] = directory/(name+ending)
    failed = Path('/tmp/seaqr_visible_interaction_v30_qsP2aT')
    old = read(failed/'freeze.json')
    for name in list(old['sources']) + ['freeze.json', 'unit.log', 'batch.json', 'batch.log', 'status.json',
        'trace_0_0082_v26.log', 'trace_0_0082_v26.v29.json', 'trace_0_0082_v26.trace30.json',
        'trace_0_0082_v26/launch.json', 'trace_0_0082_v26/report.json', 'trace_0_0082_v26/frames.jsonl']:
        files['failed_attempt/'+name] = failed/name
    for name, path in files.items():
        if not path.is_file() or path.is_symlink() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid export member '+name)
    write(directory/'export_manifest.json', dict(post_run=True, media_included=False,
        files={n: sha(p) for n, p in files.items()},
        note='SQLite has complete diagnostic event tables. Original .nsys-rep and failed initial SQLite stay on Jetson.'))
    files['export_manifest.json'] = directory/'export_manifest.json'
    with tarfile.open(archive, 'x:gz') as bundle:
        for name, path in files.items():
            bundle.add(path, arcname=name, recursive=False)
    print(sha(archive), archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    export(args.directory, args.archive)
