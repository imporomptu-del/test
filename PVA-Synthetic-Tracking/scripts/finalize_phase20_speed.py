"""Verify and archive the accuracy/speed experiment without production claims."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256
from run_phase20_maturity import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accuracy',type=Path,required=True);p.add_argument('--speed',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.accuracy/'audit.json').read_text())
    comparison=json.loads((a.speed/'pipeline/comparison.json').read_text())
    if not comparison['exact_candidates_tracks_and_coverage']:raise ValueError('Acceleration changed output')
    if any(r['reviewed_regression_gate_pass'] is False for r in audit['runs']):raise ValueError('Reference regression')
    if sorted(Path(r['run']).name for r in audit['runs'])!=['cpu_0029','cpu_0055','cpu_0082','cpu_0126','pva_0126']:
        raise ValueError('Missing accuracy run')
    library=a.speed/'pipeline/libseaqr_median.so'
    if sha256(library)!=comparison['external_accelerators']['median']['library_sha256']:
        raise ValueError('Compiled library differs')
    frozen=json.loads((a.speed/'pipeline/freeze.json').read_text())
    for path,h in frozen['files_sha256'].items():
        if sha256(ROOT/path)!=h:raise ValueError('Accelerated source changed')
    artifacts=[library,a.accuracy/'audit.json',a.accuracy/'freeze.json',a.accuracy/'policy.md',
        a.accuracy/'visual_review_notes.md',a.accuracy/'pva_prefix_anchors.json',
        a.speed/'pipeline/comparison.json',a.speed/'pipeline/freeze.json',a.speed/'pipeline/pva_config.json',
        a.speed/'cuda_median_benchmark.json',a.speed/'state_update_benchmark.json',a.speed/'default_parity_final.json']
    for r in audit['runs']:
        for n,h in r['artifacts_sha256'].items():
            path=Path(r['run'])/n
            if sha256(path)!=h:raise ValueError('Audited accuracy artifact changed')
            artifacts.append(path)
    for n,h in comparison['artifacts_sha256'].items():
        if sha256(n)!=h:raise ValueError('Compared pipeline artifact changed')
        artifacts.append(Path(n))
    for cid in ['0055','0082']:
        artifacts.extend(sorted((a.accuracy/('review_'+cid)).glob('*')))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/unit'],cwd=ROOT,capture_output=True,text=True)
    match=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not match:raise RuntimeError(tests.stdout+tests.stderr)
    copies=a.speed/'verification_tools';copies.mkdir(exist_ok=False)
    paths=[ROOT/'scripts'/n for n in ['finalize_phase20_speed.py','compare_phase20_acceleration.py',
        'prepare_phase20_accelerated.py','benchmark_phase20_cuda_median.py','benchmark_phase20_state_update.py',
        'phase20_cuda_median.cu','run_phase20_frozen_extra.py','audit_phase20_v5.py','score_phase20_accuracy.py',
        'run_phase20_maturity.py','review_phase20_clutter.py','verify_phase20_v3_defaults.py']]
    paths.extend(sorted((ROOT/'tests/unit').glob('*.py')))
    for path in paths:
        dest=copies/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    record=dict(tests_passed=int(match.group(1)),test_stderr=tests.stderr,
        all_frozen_target_checks_pass=True,remaining_126_missed_frames=[216],
        appearance_accuracy_default_promoted=False,background_response_tradeoff_unresolved=True,
        exact_accelerated_frames=comparison['frames'],speedup=comparison['speedup'],
        before_fps=comparison['before_fps'],after_fps=comparison['after_fps'],
        total_fresh_video_frames=sum(r['frames'] for r in audit['runs'])+comparison['frames'],
        real_time=False,full_clip_accelerated_validation=False,airborne_accuracy_verified=False,
        production_promoted=False,sealed_holdout_accessed=False,
        artifacts_sha256={str(p.resolve()):sha256(p) for p in artifacts},
        verification_tools_sha256={str(p.relative_to(copies)):sha256(p) for p in copies.rglob('*') if p.is_file()})
    write(a.speed/'verification.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['test_stderr','artifacts_sha256','verification_tools_sha256']},indent=2))


if __name__=='__main__':main()
