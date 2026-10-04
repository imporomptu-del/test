"""Export experiment reports and harness only; never package media."""
import argparse
import hashlib
from pathlib import Path
import tarfile


def collect(root, output):
    root = root.resolve()
    if output.exists():
        raise FileExistsError(output)
    selected = [p for p in root.iterdir() if p.suffix in ('.py', '.md')]
    selected += [p for p in (root/'results').rglob('*')
                 if p.suffix in ('.json', '.jsonl', '.log') and 'implementation' not in p.parts]
    if not selected:
        raise ValueError('No evidence')
    with tarfile.open(output, 'x:gz') as archive:
        for path in sorted(selected):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError('Unsafe evidence path')
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    digest = hashlib.sha256()
    with output.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    print(digest.hexdigest(), output.name, output.stat().st_size, len(selected), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
