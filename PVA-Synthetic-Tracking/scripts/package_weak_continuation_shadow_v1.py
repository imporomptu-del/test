"""Package only code, plans, tests and pinned metadata; never camera media."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('run_weak_continuation_shadow_v1.py', 'weak_continuation_shadow_v1.py',
           'weak_continuation_information_v1.py', 'accuracy_v56_capture.py',
           'audit_weak_continuation_shadow_v1.py', 'package_weak_continuation_shadow_v1.py',
           'summarize_weak_continuation_shadow_v1.py')
TESTS = ('test_weak_continuation_shadow_v1.py', 'test_weak_continuation_information_v1.py',
         'test_run_weak_continuation_shadow_v1.py', 'test_audit_weak_continuation_shadow_v1.py',
         'test_weak_shadow_lifecycle_v1.py', 'test_summarize_weak_continuation_shadow_v1.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(output):
    plan_path = ROOT/'configs/evaluation/weak_continuation_shadow_v1.json'
    plan = json.loads(plan_path.read_text())
    files = {name:ROOT/'scripts'/name for name in SCRIPTS}
    files.update({name:ROOT/'tests/unit'/name for name in TESTS})
    files['plan.json'] = plan_path
    base = ROOT/'results/tiny_target/visible_validation_v34_20260923/audit_20260924/evidence/run'
    for clip, spec in plan['clips'].items():
        path = base/f'full_repeat0_{clip}/launch.json'
        if sha(path) != spec['launch_sha256']:
            raise ValueError('Changed reference launch '+clip)
        files[f'reference_{clip}.json'] = path
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing regular bundle input '+str(path))
    identities = {name:sha(path) for name,path in files.items()}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name,path in files.items():
        shutil.copyfile(path, output/name)
        if sha(output/name) != identities[name] or sha(path) != identities[name]:
            raise ValueError('Input changed while packaging')
    with (output/'transfer_manifest.json').open('x') as stream:
        json.dump(dict(schema='seaqr.weak-continuation-shadow.transfer.v1', files_sha256=identities,
                       contains_media=False, source_root=str(ROOT)), stream, indent=2)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(package(parser.parse_args().output))
