"""Archive the exact-output resident CUDA trial; no default or accuracy promotion."""
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
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--profiled-cpu',action='store_true',help='Verify CPU execution-only follow-up with profiling/warp evidence')
    a=p.parse_args();d=a.root;pipeline=d/'pipeline'
    c=json.loads((pipeline/'comparison.json').read_text())
    if not c['exact_candidates_tracks_and_coverage'] or c['frames']!=230 or c['actual_pva_pairs']!=229:
        raise ValueError('Exact 230-frame actual-PVA comparison required')
    if a.profiled_cpu:
        profile=json.loads((d/'profile.json').read_text())
        warp=json.loads((d/'cuda_warp_check.json').read_text())
        if not profile['passed'] or profile['exact_frames']!=96 or profile['difference_count']:
            raise ValueError('Missing profiling parity')
        if warp['all_exact'] or warp['promoted']:raise ValueError('Unexpected warp trial state')
    else:
        correctness=json.loads((d/'correctness_v2.json').read_text())
        if not correctness['passed'] or correctness['synthetic_pairs']!=72 or correctness['native_pairs']!=24:
            raise ValueError('Missing detector/state parity')
    anchors=json.loads((d/'pva_prefix_anchors.json').read_text())
    if any(not r['all_required_anchors_same_measured_id'] for r in anchors['sparse_anchor_results']):
        raise ValueError('Anchor regression')
    if anchors['counts'].get('pva_errors',0) or anchors['counts'].get('motion_resets',0):
        raise ValueError('PVA errors or resets')
    dense=c['scored']['evaluation']['positive_windows']
    if len(dense)!=1 or dense[0]['qualified_measured_hits']!=138 or dense[0]['missed_visible_frames']!=[216]:
        raise ValueError('Dense reference regression')
    pilot=c['pilot_score']['positive_windows']
    if len(pilot)!=1 or pilot[0]['visible_samples']!=12 or pilot[0]['qualified_measured_hits']!=12:
        raise ValueError('Pilot regression')
    defaults=json.loads((d/'default_parity.json').read_text())
    if not defaults['experimental_policies_default_off'] or not defaults['frozen_phase19_files_intact']:
        raise ValueError('Defaults/frozen files changed')
    if defaults['default_tracking_frame_pairs_identical']!=320 or defaults['disabled_protection_detector_frame_pairs_identical']!=120:
        raise ValueError('Incomplete default parity')
    frozen=json.loads((pipeline/'freeze.json').read_text())
    for path,h in frozen['files_sha256'].items():
        if sha256(ROOT/path)!=h:raise ValueError('Source changed since freeze: '+path)
    if sha256(pipeline/'runtime.tar.gz')!=frozen['archive_sha256']:raise ValueError('Archive changed')
    library=pipeline/'libseaqr_resident.so'
    if sha256(library)!=c['external_accelerators']['median']['library_sha256']:
        raise ValueError('Library mismatch')
    for path,h in c['artifacts_sha256'].items():
        if sha256(path)!=h:raise ValueError('Compared artifact changed')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/unit'],cwd=ROOT,capture_output=True,text=True)
    match=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not match:raise ValueError(tests.stdout+tests.stderr)
    copies=d/'verification_tools';copies.mkdir(exist_ok=False)
    paths=[ROOT/'scripts'/name for name in (
        'finalize_phase20_resident.py','prepare_phase20_accelerated.py','compare_phase20_acceleration.py',
        'check_phase20_resident.py','summarize_phase20_pva_prefix.py','run_phase20_maturity.py',
        'score_phase20_accuracy.py','verify_phase20_v3_defaults.py','phase20_cuda_resident.cu','phase20_cuda_median.cu')]
    paths+=list(sorted((ROOT/'tests/unit').glob('*.py')))
    if a.profiled_cpu:
        paths += [ROOT/'scripts'/n for n in ('phase20_cuda_profile.cu','profile_phase20_resident.py','check_phase20_cuda_warp.py')]
    for path in paths:
        target=copies/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    artifacts=[d/n for n in ('default_parity.json','pva_prefix_anchors.json')]
    artifacts += [d/n for n in (('profile.json','cuda_warp_check.json') if a.profiled_cpu else ('correctness_v1.json','correctness_v2.json'))]
    artifacts += [pipeline/n for n in ('comparison.json','freeze.json','pva_config.json','runtime.tar.gz','libseaqr_resident.so')]
    record=dict(tests_passed=int(match.group(1)),test_stderr=tests.stderr,
        exact_accelerated_frames=230,actual_pva_pairs=229,
        before_fps=c['before_fps'],after_fps=c['after_fps'],speedup=c['speedup'],
        remaining_126_missed_frames=[216],background_response_tradeoff_unresolved=True,
        full_clip_validation=False,real_time=False,airborne_accuracy_verified=False,
        production_promoted=False,sealed_holdout_accessed=False,
        artifacts_sha256={str(p.resolve()):sha256(p) for p in artifacts},
        verification_tools_sha256={str(p.relative_to(copies)):sha256(p) for p in copies.rglob('*') if p.is_file()})
    write(d/'verification.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('test_stderr','artifacts_sha256','verification_tools_sha256')},indent=2))


if __name__=='__main__':main()
