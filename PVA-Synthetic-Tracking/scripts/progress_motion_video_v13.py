"""Read only a bounded tail of experiment journals, never camera media."""
import json
from pathlib import Path

root = Path('/tmp/seaqr_video_v13_IyK7eQ/results')
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or directory.with_suffix('.execution.json').exists():
        continue
    journal = directory/'frames.jsonl'
    if not journal.exists():
        log = directory.with_suffix('.log')
        progress = None
        if log.exists():
            with log.open('rb') as handle:
                size = handle.seek(0, 2)
                handle.seek(max(0, size-8192))
                data = handle.read()
            for line in reversed(data.splitlines()):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict) and 'frames' in row:
                    progress = row['frames']
                    break
        print(json.dumps(dict(run=directory.name, completed_frames=progress)))
        continue
    with journal.open('rb') as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size-2*1024*1024))
        data = handle.read()
    rows = data.splitlines()
    if size > len(data):
        rows = rows[1:]
    for line in reversed(rows):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        print(json.dumps(dict(run=directory.name, completed_frames=row['frame_index']+1)))
        break
