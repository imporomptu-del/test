"""Locally verify downloaded native/component/probe evidence, without media."""
import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256
from compare_phase20_exact_runs import without_timing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','timing-reference','kernel-freeze','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite diagnostic verification')
    frozen=json.loads((args.root/'freeze.json').read_text())
    speed=json.loads((args.root.parent/'verified/summary.json').read_text())
    if not speed['passed_execution_equivalence'] or speed['freeze_sha256'] != sha256(args.root/'freeze.json'):
        raise ValueError('Full-video verification must pass first')
    component=json.loads((args.root/'component_preflight.json').read_text())
    if (component['passed'] is not True or component['media_accessed'] is not False
            or component['shape_cases'] != 600
            or component['library_sha256'] != frozen['native_shape_build']['library_sha256']
            or component['reference_freeze_sha256'] != sha256(args.timing_reference/'freeze.json')
            or component['script_sha256'] != frozen['files_sha256']['scripts/verify_phase20_native_shapes.py']):
        raise ValueError('Component oracle provenance changed')
    for name,digest in component['implementation_sha256'].items():
        if digest != frozen['files_sha256'][name]:
            raise ValueError('Component runtime changed')
    reference=args.root/'pva_0126'
    host=json.loads((args.root/'host_profile.json').read_text())
    if (host['passed'] is not True or host['exact_prefix_frames'] != 96
            or host['reference_sha256'] != sha256(reference/'frames.jsonl')
            or host['config_sha256'] != frozen['config_sha256']
            or host['library_sha256'] != frozen['compiled_library_sha256']
            or host['script_sha256'] != frozen['files_sha256']['scripts/profile_phase20_host_speed.py']):
        raise ValueError('Host profile provenance changed')
    probe_root=args.root/'transfer_probe'
    probe=json.loads((probe_root/'measurement/transfer_profile.json').read_text())
    build_path=probe_root/'libseaqr_transfer_probe.so.build.json'
    build=json.loads(build_path.read_text())
    if (probe['passed'] is not True or probe['exact_prefix_frames'] != 96
            or probe['profiled_frame_count'] != 24 or probe['profiled_frames_inclusive'] != [72,95]
            or probe['reference_journal_sha256'] != sha256(reference/'frames.jsonl')
            or probe['reference_cuda_library_sha256'] != frozen['compiled_library_sha256']
            or probe['probe_build_sha256'] != sha256(build_path) or probe['probe_build'] != build
            or build['library_sha256'] != sha256(probe_root/'libseaqr_transfer_probe.so')
            or build['kernel_freeze_sha256'] != sha256(args.kernel_freeze)
            or probe['script_sha256'] != build['sources_sha256']['probe_phase20_cuda_transfers.py']
            or probe['full_clip_fps_claim'] is not False):
        raise ValueError('Transfer probe provenance changed')
    kernels=json.loads(args.kernel_freeze.read_text())['files_sha256']
    archive=args.root.parent/'transfer_probe_sources.tar.gz'
    import hashlib
    with tarfile.open(archive) as tar:
        for name,digest in build['sources_sha256'].items():
            if hashlib.sha256(tar.extractfile('scripts/'+name).read()).hexdigest() != digest:
                raise ValueError('Probe source archive changed')
            if name != 'phase20_cuda_transfer_probe.cu' and name.endswith('.cu') and kernels['scripts/'+name] != digest:
                raise ValueError('Probe changed original kernels')
    run=probe_root/'measurement/run'
    before=json.loads((reference/'launch.json').read_text())
    after=json.loads((run/'launch.json').read_text())
    configs=[asdict(VisibleConfig(**r['configuration'])) for r in (before,after)]
    changes={k for k in configs[0] if configs[0][k] != configs[1][k]}
    if (changes != {'cuda_median_library'} or before['package_sha256'] != after['package_sha256']
            or before['source_sha256'] != after['source_sha256'] or before['fps'] != after['fps']
            or after['exact_cuda_stabilization']['library_sha256'] != build['library_sha256']
            or before['external_accelerators']['shape'] != after['external_accelerators']['shape']):
        raise ValueError('Probe changed algorithm/source/native policy')
    for name,digest in after['package_sha256'].items():
        if sha256(run/'implementation'/name) != digest:
            raise ValueError('Probe implementation snapshot changed')
    with (reference/'frames.jsonl').open() as f:
        rows_before=[json.loads(row) for row in islice(f,96)]
    with (run/'frames.jsonl').open() as f:
        rows_after=[json.loads(row) for row in f]
    if len(rows_before) != 96 or len(rows_after) != 96:
        raise ValueError('Truncated diagnostic prefix')
    for i,(a,b) in enumerate(zip(rows_before,rows_after)):
        if a['frame_index'] != i or b['frame_index'] != i or without_timing(a) != without_timing(b):
            raise ValueError('Diagnostic non-timing output changed: '+str(i))
    result=dict(passed=True, media_accessed=False, component_cases=600, exact_probe_prefix_frames=96,
        profiled_frames=24, full_clip_fps_claim=False, script_sha256=sha256(__file__),
        speed_verification_sha256=sha256(args.root.parent/'verified/summary.json'),
        source_archive_sha256=sha256(archive), probe_journal_sha256=sha256(run/'frames.jsonl'),
        probe_result_sha256=sha256(probe_root/'measurement/transfer_profile.json'),
        caveat=probe['caveat'])
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
