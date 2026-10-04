"""Independently verify downloaded kernel builds, profiles and repeated prefixes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256
from build_phase20_kernel_probe import SITES,instrument
from build_phase20_peak_gate import transform
from repeat_phase20_kernel_speed import check_prefix
from compare_phase20_exact_runs import shape_accelerator,validate_gpu_transition


def require(condition,message):
    if not condition:raise ValueError(message)


def verify_build(root,name):
    path=root/(name+'.so');record_path=path.with_suffix('.so.build.json')
    record=json.loads(record_path.read_text())
    require(sha256(path)==record['library_sha256'],'Changed library: '+name)
    for source,expected in record['sources_sha256'].items():
        require(sha256(root/'scripts'/source)==expected,'Changed build source: '+source)
    for source,expected in record['generated_sha256'].items():
        require(sha256(root/(name+'_sources')/source)==expected,'Changed generated source: '+source)
    command=record['command']
    require(command[:5]==['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_87','-Xptxas=-v'],
        'Unexpected CUDA flags')
    require(len(command)==11 and command[5:8]==['-Xcompiler','-fPIC','-shared']
        and command[9]=='-o','Unexpected CUDA build command')
    return record


def verify_profile(root,name,library_name,reference):
    build=verify_build(root,library_name)
    profile=json.loads((root/name/'kernel_profile.json').read_text())
    require(profile['passed'] and profile['profiled_frame_count']==24 and profile['exact_prefix_frames']==96,
        'Incomplete kernel profile')
    require(profile['profiled_frames_inclusive']==[72,95],'Different profiling interval')
    require(profile['probe_build']==build and profile['probe_build_sha256']==sha256(root/(library_name+'.so.build.json')),
        'Profile build mismatch')
    require(profile['reference_journal_sha256']==sha256(reference/'frames.jsonl'),'Changed profile reference')
    check_prefix(reference,root/name/'run',96)
    launch=json.loads((root/name/'run/launch.json').read_text())
    old=json.loads((reference/'launch.json').read_text())
    for key in ('source_sha256','fps','motion_config_sha256','package_sha256'):
        require(launch[key]==old[key],'Profile launch changed: '+key)
    configs=[asdict(VisibleConfig(**l['configuration'])) for l in (launch,old)]
    require(all(configs[0][k]==configs[1][k] for k in configs[0] if k!='cuda_median_library'),
        'Profile changed algorithm/native settings')
    require(launch['exact_cuda_stabilization']['library_sha256']==build['library_sha256'],'Profile used wrong GPU binary')
    shape_accelerator(launch)
    for source,expected in launch['package_sha256'].items():
        require(sha256(root/name/'run/implementation'/source)==expected,'Profile implementation changed')
    rows={r['kernel']:r for r in profile['kernels']}
    required={'warp_u8','gaussian_horizontal','gaussian_vertical','median5',
        'residual_prepare','gather_samples','select_peaks','finish_state'}
    require(required<=rows.keys() and all(rows[k]['calls']==24 for k in required),'Missing profile launches')
    for name,row in rows.items():
        values=np.asarray(row['samples_ms'],float)
        require(len(values)==row['calls'] and np.isfinite(values).all() and np.all(values>=0),'Invalid kernel samples')
        require(abs(float(values.sum())/24-row['ms_per_profiled_frame'])<1e-9,'Kernel summary mismatch')
    return {name:row['ms_per_profiled_frame'] for name,row in rows.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();r=a.root
    if a.output.exists():raise ValueError('Never overwrite evidence verification')
    require(sha256(r/'kernel_freeze.json')=='eefe558fcfdf3679cd9f35362efe93e6c4bb5a64089b2d69aff2a8663b3a1173',
        'Original kernel freeze changed')
    kernel_freeze=json.loads((r/'kernel_freeze.json').read_text())
    for name in SITES:
        require(sha256(r/'scripts'/name)==kernel_freeze['files_sha256']['scripts/'+name],
            'Original frozen kernel source changed')
    frozen=json.loads((r/'pipeline/freeze.json').read_text())
    with tarfile.open(r/'pipeline/runtime.tar.gz') as archive:
        for name,expected in frozen['files_sha256'].items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,'Runtime archive changed')
    builds={name:verify_build(r,name) for name in
        ('libseaqr_peak_gate','libseaqr_test_before','libseaqr_test_after')}
    for name,build in builds.items():
        candidate=name!='libseaqr_test_before'
        require(build['diagnostic_only']==(name!='libseaqr_peak_gate'),'Diagnostic/benchmark binary confusion')
        for source in SITES:
            original=(r/'scripts'/source).read_text()
            expected=transform(original) if candidate and source=='phase20_cuda_resident.cu' else original
            require((r/(name+'_sources')/source).read_text()==expected,'More than the allowed predicate reorder changed')
    require(set(builds['libseaqr_peak_gate']['generated_sha256'])==set(SITES)
        and Path(builds['libseaqr_peak_gate']['command'][8]).name=='phase20_cuda_integrated.cu',
        'Benchmark source includes diagnostic entry points')
    require(sha256(r/'pipeline/libseaqr_integrated.so')==builds['libseaqr_peak_gate']['library_sha256'],
        'Pipeline used a different candidate')
    conformance=json.loads((r/'peak_gate_conformance.json').read_text())
    require(conformance['passed'] and conformance['synthetic_pairs']==672
        and conformance['scalar_oracle_pairs']==420 and conformance['native_size_synthetic_pairs']==3,
        'Conformance incomplete')
    require(conformance['builds']==[builds['libseaqr_test_before'],builds['libseaqr_test_after']],
        'Conformance build mismatch')
    require(conformance['script_sha256']==sha256(r/'scripts/verify_phase20_peak_gate.py'),
        'Conformance verifier changed')
    profiles={
        'before':verify_profile(r,'baseline_kernel_profile','libseaqr_kernel_probe',a.reference/'pva_0126'),
        'after':verify_profile(r,'candidate_kernel_profile','libseaqr_candidate_probe',r/'pipeline/pva_0126')}
    for lib,parent in (('libseaqr_kernel_probe',r/'scripts'),
            ('libseaqr_candidate_probe',r/'libseaqr_peak_gate_sources')):
        for source in SITES:
            require((r/(lib+'_sources')/source).read_text()==instrument(source,(parent/source).read_text()),
                'Probe changed kernel code')
    require(json.loads((r/'libseaqr_candidate_probe.so.build.json').read_text())['candidate_library_sha256']
        ==builds['libseaqr_peak_gate']['library_sha256'],'Wrong profiled candidate')
    repeat_root=r/'pipeline/repeats'
    repeated=json.loads((repeat_root/'repeat_summary.json').read_text())
    require(repeated['passed'] and len(repeated['comparisons'])==6,'Repeats incomplete')
    repeat_rows=[]
    for cid in ('0126','0082'):
        for pair in range(3):
            item=dict(clip_id=cid,pair=pair)
            for label in ('before','after'):
                path=repeat_root/f'{cid}_pair{pair}_{label}'
                reference=a.reference/('pva_'+cid)
                check_prefix(reference,path,128)
                launch=json.loads((path/'launch.json').read_text())
                old=json.loads((reference/'launch.json').read_text())
                validate_gpu_transition(old,launch,None if label=='before' else frozen['gpu_transition'])
                shape_accelerator(launch)
                require(launch['source_sha256']==old['source_sha256'] and launch['fps']==old['fps'],'Repeat source/cadence changed')
                base=a.reference if label=='before' else r/'pipeline'
                require(launch['config_sha256']==sha256(base/'config.json')
                    and launch['configuration']==json.loads((base/'config.json').read_text()),'Repeat policy changed')
                require(launch['motion_config_sha256']==old['motion_config_sha256'],'Repeat motion settings changed')
                require(launch['package_sha256']=={k.removeprefix('tiny_target/'):v
                    for k,v in frozen['files_sha256'].items() if k.startswith('tiny_target/')},
                    'Repeat runtime changed')
                report=json.loads((path/'report.json').read_text())
                require(report['completed'] and report['frames']==128 and not report['full_clip'],'Wrong repeat extent')
                item[label+'_fps']=report['processed_fps']
                recorded=next(row for row in repeated['comparisons'] if row['clip_id']==cid and row['pair']==pair)[label]
                require(recorded['fps']==report['processed_fps'] and recorded['journal_sha256']==sha256(path/'frames.jsonl'),
                    'Repeat summary changed')
                for source,expected in launch['package_sha256'].items():
                    require(sha256(path/'implementation'/source)==expected,'Repeat implementation changed')
            item['speedup']=item['after_fps']/item['before_fps'];repeat_rows.append(item)
    result=dict(passed=True,profiles_ms_per_frame=profiles,repeated_prefix_comparisons=repeat_rows,
        exact_repeated_frame_comparisons=1536,exact_profile_frame_comparisons=192,
        source_transform_verified=True,benchmark_has_no_test_or_profiling_ABI=True,
        unit_test_log_sha256=sha256(r/'unit_tests_full.log'),
        script_sha256=sha256(__file__),production_ready=False,
        caveat='Verifies supplied build provenance, source transformation and all non-timing journals. No claim of exhaustive CUDA memory checking, airborne accuracy or real-time throughput.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
