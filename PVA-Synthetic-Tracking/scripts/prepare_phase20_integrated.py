"""Freeze exact CUDA warp/integration and a bounded full-development validation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-workspace',required=True);a=p.parse_args()
    reference=ROOT/'results/tiny_target/phase20/efficiency_v3_20260914/pipeline/pva_0126'
    comparison=json.loads((reference.parent/'comparison.json').read_text())
    if not comparison['exact_candidates_tracks_and_coverage']:raise ValueError('Reference gate failed')
    cfg=asdict(VisibleConfig(**json.loads((reference/'launch.json').read_text())['configuration']))
    cfg['cuda_median_library']=a.remote_workspace+'/libseaqr_integrated.so'
    a.output.mkdir(parents=True,exist_ok=False)
    config_paths=[]
    for name,execution in [('host','cuda_cubic_host'),('resident','cuda_cubic_resident')]:
        value=asdict(VisibleConfig(**dict(cfg,stabilization_execution=execution)))
        path=a.output/(name+'_config.json');path.write_text(json.dumps(value,indent=2));config_paths.append(path)
    packet=json.loads((ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json').read_text())
    sources={}
    for cid in ('0126','0029','0055','0082'):
        source=next(s for s in packet['plan']['sources'] if s['clip_id']==cid)
        sources[cid]=dict(path='/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_'+cid+'.avi',
                          sha256=source['sha256'],frames=source['frames'])
    paths=sorted((ROOT/'tiny_target').rglob('*.py'))+[ROOT/'configs/evaluation/phase20_motion_v8.json']
    paths += [ROOT/'scripts'/name for name in ('phase20_cuda_integrated.cu','phase20_cuda_warp_exact.cu',
        'phase20_cuda_resident.cu','phase20_cuda_median.cu','run_phase20_integrated_batch.py',
        'check_phase20_integrated.py','check_phase20_exact_warp.py','check_phase20_exact_gaussian.py')]
    freeze=dict(files_sha256={str(path.relative_to(ROOT)):sha256(path) for path in paths},sources=sources,
        remote_reference_run='/tmp/seaqr_phase20_indexed_wYo7ID/pva_0126',local_reference_run=str(reference),
        reference_artifacts_sha256={n:sha256(reference/n) for n in ('launch.json','frames.jsonl','report.json')},
        policy='Execution changes only. Exact230-frame reference parity gates both host and resident modes before four full development clips. No labels, tuning, holdout access or overwrite.',
        native_shape_hw=[3190,4784],source_bit_depth=8,container_fps=10,
        cuda_build_command='/usr/local/cuda/bin/nvcc -O3 --fmad=false -arch=sm_87 -Xcompiler -fPIC -shared scripts/phase20_cuda_integrated.cu -o libseaqr_integrated.so')
    freeze['files_sha256'].update({path.name:sha256(path) for path in config_paths})
    frozen=a.output/'freeze.json';frozen.write_text(json.dumps(freeze,indent=2))
    with tarfile.open(a.output/'runtime.tar.gz','w:gz') as tar:
        for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)))
        for path in config_paths+[frozen]:tar.add(path,arcname=path.name)

if __name__=='__main__':main()
