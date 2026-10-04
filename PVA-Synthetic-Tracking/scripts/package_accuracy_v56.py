"""Create a new, bounded diagnostic bundle; no media or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    **{n: 'scripts/'+n for n in ('run_accuracy_v56_diagnostic.py', 'batch_accuracy_v56.py', 'accuracy_v56_capture.py',
        'accuracy_v56_capture.cu', 'audit_accuracy_v56_replay.py')},
    **{n: 'tests/unit/'+n for n in ('test_accuracy_v56_capture.py', 'test_accuracy_v56_replay.py')},
    'plan.md': 'docs/accuracy_v56_plan.md',
    'probes.json': 'configs/evaluation/accuracy_v56_diagnostic_probes.json',
}


def package(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, relative in SOURCES.items():
        shutil.copyfile(ROOT/relative, output/name)
    old = ROOT/'results/tiny_target/visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_0126.v34.json'
    runtime = json.loads(old.read_text())['runtime_before']
    with (output/'runtime_reference.json').open('x') as stream:
        json.dump(runtime, stream, indent=2, allow_nan=False)
    with (output/'transfer_manifest.json').open('x') as stream:
        json.dump(dict(schema='seaqr.accuracy-v56-transfer.v1', files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()
            and p.name != 'transfer_manifest.json'}, runtime_source=str(old.relative_to(ROOT)),
            runtime_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
            packager_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
