"""Freeze accuracy-preserving execution changes against a passed accuracy trial."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accuracy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-workspace',required=True)
    p.add_argument('--resident-reference',type=Path,help='Passed prior CUDA-median experiment directory; enables resident core')
    p.add_argument('--execution-reference',type=Path,help='Passed resident run; execution-only CPU indexing/cache changes')
    a=p.parse_args()
    if a.resident_reference and a.execution_reference:raise ValueError('Choose one reference mode')
    for name in ('cpu_audit.json','pva_audit.json'):
        if not all(r['reviewed_regression_gate_pass'] for r in json.loads((a.accuracy/name).read_text())['runs']):
            raise ValueError('Accuracy trial did not pass')
    cfg=asdict(VisibleConfig(**json.loads((a.accuracy/'pva_config.json').read_text())))
    cfg.update(state_update_backend='inplace',spatial_filter_backend='cuda_median5',
        cuda_median_library=a.remote_workspace+'/libseaqr_median.so')
    reference=a.accuracy/'pva_0126'
    if a.resident_reference:
        comparison=json.loads((a.resident_reference/'comparison.json').read_text())
        if not comparison['exact_candidates_tracks_and_coverage']:raise ValueError('Reference parity failed')
        reference=a.resident_reference/'pva_0126'
        previous=json.loads((reference/'launch.json').read_text())['configuration']
        if previous['state_update_backend']!='inplace' or previous['spatial_filter_backend']!='cuda_median5':
            raise ValueError('Expected prior in-place/CUDA-median reference')
        cfg=asdict(VisibleConfig(**previous))
        cfg.update(state_update_backend='cuda_resident',cuda_median_library=a.remote_workspace+'/libseaqr_resident.so')
    if a.execution_reference:
        comparison=json.loads((a.execution_reference/'comparison.json').read_text())
        if not comparison['exact_candidates_tracks_and_coverage']:raise ValueError('Reference parity failed')
        reference=a.execution_reference/'pva_0126'
        cfg=asdict(VisibleConfig(**json.loads((reference/'launch.json').read_text())['configuration']))
        if cfg['state_update_backend']!='cuda_resident' or cfg['spatial_filter_backend']!='cuda_median5':
            raise ValueError('Expected resident reference')
        cfg['cuda_median_library']=a.remote_workspace+'/libseaqr_resident.so'
    cfg=asdict(VisibleConfig(**cfg))
    a.output.mkdir(parents=True,exist_ok=False)
    cp=a.output/'pva_config.json';cp.write_text(json.dumps(cfg,indent=2))
    paths=list(sorted((ROOT/'tiny_target').rglob('*.py')))+[
        ROOT/'configs/evaluation/phase20_motion_v8.json',ROOT/'scripts/phase20_cuda_median.cu']
    if a.resident_reference or a.execution_reference:paths.append(ROOT/'scripts/phase20_cuda_resident.cu')
    with tarfile.open(a.output/'runtime.tar.gz','w:gz') as tar:
        for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)))
        tar.add(cp,arcname='pva_config.json')
    (a.output/'freeze.json').write_text(json.dumps(dict(
        files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        config_sha256=sha256(cp),archive_sha256=sha256(a.output/'runtime.tar.gz'),
        reference_run=str(reference.resolve()),
        reference_artifacts_sha256={n:sha256(reference/n) for n in ['launch.json','report.json','frames.jsonl']},
        experiment='resident_cpu_indexing' if a.execution_reference else ('cuda_resident' if a.resident_reference else 'cuda_median_inplace'),
        allowed_clip_ids=['0126'],max_frames=230,policy='Execution-only changes; exact outputs required; no accuracy-policy changes'),indent=2))


if __name__=='__main__':main()
