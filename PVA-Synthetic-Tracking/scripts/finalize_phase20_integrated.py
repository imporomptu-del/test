"""Local tests and artifact-integrity closeout for the exact-CUDA experiment."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    target=a.root/'verification.json'
    if target.exists():raise ValueError('Verification already exists')
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/unit'))
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        tests=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    frozen=json.loads((a.root/'pipeline/freeze.json').read_text())
    execution=[n for n in frozen['files_sha256'] if n.startswith('tiny_target/') or n.endswith('.cu')]
    if any(sha256(ROOT/n)!=frozen['files_sha256'][n] for n in execution):raise ValueError('Execution code changed since freeze')
    with tarfile.open(a.root/'pipeline/runtime.tar.gz') as tar:
        for name,h in frozen['files_sha256'].items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=h:raise ValueError('Frozen archive differs')
    warp=json.loads((a.root/'conformance/synthetic_extended_v2.json').read_text())['results'][0]
    gaussian=json.loads((a.root/'conformance/gaussian_v3.json').read_text())[0]
    native=json.loads((a.root/'conformance/integrated_check_v1.json').read_text())
    if warp['exact_cases']!=2508 or any(c['pixels_differ'] or c['mask_differ'] for c in warp['cases']):raise ValueError('Warp conformance failed')
    if len(gaussian['cases'])!=137 or any(c['different'] for c in gaussian['cases']):raise ValueError('Gaussian conformance failed')
    if not native['passed'] or (native['synthetic_pairs'],native['native_pairs'])!=(72,24):raise ValueError('Closed-loop conformance failed')
    default=json.loads((a.root/'default_parity.json').read_text())
    if (default['default_tracking_frame_pairs_identical'],default['disabled_protection_detector_frame_pairs_identical'])!=(320,120):
        raise ValueError('Default behavior changed')
    accuracy=json.loads((a.root/'full_development_audit.json').read_text())
    control=json.loads((a.root/'pipeline/control_comparison.json').read_text())
    control_freeze=json.loads((a.root/'pipeline/control_freeze.json').read_text())
    if control['frames']!=frozen['sources']['0029']['frames']:
        raise ValueError('Full chunk29 control missing')
    for name,key in [('control_0029','reference_journal_sha256'),('full_0029','integrated_journal_sha256')]:
        if sha256(a.root/'pipeline'/name/'frames.jsonl')!=control[key]:
            raise ValueError('Control comparison journal changed')
    if (sha256(a.root/'pipeline/control_freeze.json')!=control['control_freeze_sha256']
            or sha256(a.root/'pipeline/control_config.json')!=control_freeze['config_sha256']
            or control_freeze['parent_freeze_sha256']!=sha256(a.root/'pipeline/freeze.json')
            or control_freeze['library_sha256']!=accuracy['library_sha256']):
        raise ValueError('Control provenance changed')
    lifecycle_path=a.root/'conformance/cuda_memory_check.json'
    lifecycle=json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else None
    if lifecycle is not None and (not lifecycle['passed'] or lifecycle['frames']!=24
            or lifecycle['library_sha256']!=accuracy['library_sha256']):
        raise ValueError('Final-library lifecycle check failed')
    execution_passed=(tests.wasSuccessful() and accuracy['execution_healthy']
                      and accuracy['repeat_prefix_exact'] and control['exact'])
    checks_passed=execution_passed and accuracy['known_reference_nonregression_gate']
    result=dict(unit_tests_run=tests.testsRun,unit_tests_passed=tests.wasSuccessful(),failures=len(tests.failures),errors=len(tests.errors),
        execution_equivalence_checks_passed=bool(execution_passed),
        functional_and_reference_checks_passed=bool(checks_passed),
        full_chunk29_reference_control_exact=control['exact'],
        full_chunk29_reference_control_frames=control['frames'],
        control_comparison_sha256=sha256(a.root/'pipeline/control_comparison.json'),
        exact_warp_cases=2508,exact_gaussian_cases=137,synthetic_stateful_pairs=72,native_stateful_pairs=24,
        executed_package_and_cuda_match_current=True,frozen_archive_verified=True,
        default_tracking_pairs=320,default_detector_pairs=120,
        full_clip_frames=accuracy['full_clip_frames'],known_reference_nonregression_gate=accuracy['known_reference_nonregression_gate'],
        execution_healthy=accuracy['execution_healthy'],repeat_prefix_exact=accuracy['repeat_prefix_exact'],
        gpu_lifecycle_check=lifecycle,
        memory_safety_proven=False,
        compute_sanitizer_run=False,
        compute_sanitizer_limitation='Compute Sanitizer was unavailable on the Jetson; no installation or system changes were made.',
        sanitizer_log_sha256=sha256(a.root/'conformance/sanitizer.log') if (a.root/'conformance/sanitizer.log').exists() else None,
        generalization_proven=False,production_ready=False,script_sha256=sha256(__file__),
        verification_tools_archive_sha256=sha256(a.root/'verification_tools.tar.gz'),
        main_result_archive_sha256=sha256(a.root/'pipeline/result_bundle.tar.gz'),
        control_result_archive_sha256=sha256(a.root/'pipeline/control_bundle.tar.gz'),
        runtime_archive_sha256=sha256(a.root/'pipeline/runtime.tar.gz'),audit_sha256=sha256(a.root/'full_development_audit.json'),
        test_files_sha256={str(n.relative_to(ROOT)):sha256(n) for n in sorted((ROOT/'tests/unit').glob('test_*.py'))})
    with target.open('x') as f:json.dump(result,f,indent=2)
    with (a.root/'unit_tests.log').open('x') as f:f.write(stream.getvalue())
    print(json.dumps({k:v for k,v in result.items() if k!='test_files_sha256'},indent=2))
    if not checks_passed:raise SystemExit(1)

if __name__=='__main__':main()
