"""Strict per-frame comparison of real PVA executions, plus measured throughput."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256
from run_phase20_maturity import evaluate,write
from score_phase20_accuracy import score_rows,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/'freeze.json').read_text())
    before=Path(frozen['reference_run']);after=a.root/'pva_0126'
    for n,h in frozen['reference_artifacts_sha256'].items():
        if sha256(before/n)!=h:raise ValueError('Reference changed')
    launches=[json.loads((d/'launch.json').read_text()) for d in (before,after)]
    reports=[json.loads((d/'report.json').read_text()) for d in (before,after)]
    configs=[asdict(VisibleConfig(**l['configuration'])) for l in launches]
    differences={k:[configs[0][k],configs[1][k]] for k in configs[0] if configs[0][k]!=configs[1][k]}
    resident=frozen.get('experiment')=='cuda_resident'
    execution=frozen.get('experiment')=='resident_cpu_indexing'
    expected={'cuda_median_library'} if execution else ({'state_update_backend','cuda_median_library'} if resident else {'state_update_backend','spatial_filter_backend','cuda_median_library'})
    if set(differences)!=expected:
        raise ValueError('Unexpected configuration changes')
    if resident and (configs[0]['state_update_backend']!='inplace' or configs[0]['spatial_filter_backend']!='cuda_median5'):
        raise ValueError('Wrong resident reference backends')
    if execution and (configs[0]['state_update_backend']!='cuda_resident' or configs[0]['spatial_filter_backend']!='cuda_median5'):
        raise ValueError('Wrong execution reference backends')
    if configs[1]['state_update_backend']!=('cuda_resident' if resident or execution else 'inplace') or configs[1]['spatial_filter_backend']!='cuda_median5':
        raise ValueError('Wrong accelerated backends')
    for l,r in zip(launches,reports):
        if not r['completed'] or r['frames']!=230 or l['max_frames']!=230 or r['full_clip']:
            raise ValueError('Completed matching230-frame prefixes required')
        if l['source_sha256']!=launches[0]['source_sha256']:raise ValueError('Source differs')
    if sha256(a.root/'pva_config.json')!=frozen['config_sha256'] or launches[1]['config_sha256']!=frozen['config_sha256']:
        raise ValueError('Config hash mismatch')
    for path,h in launches[1]['package_sha256'].items():
        if frozen['files_sha256']['tiny_target/'+path]!=h or sha256(after/'implementation'/path)!=h:
            raise ValueError('Accelerated runtime snapshot differs from freeze')
    if not launches[1].get('external_accelerators',{}).get('median',{}).get('backend')=='CUDA':
        raise ValueError('Missing explicit CUDA execution metadata')
    bad=[];frames=0;pva_pairs=0
    with (before/'frames.jsonl').open() as x,(after/'frames.jsonl').open() as y:
        for left,right in itertools.zip_longest(x,y):
            if left is None or right is None:raise ValueError('Unequal journals')
            left,right=json.loads(left),json.loads(right)
            if left['frame_index']!=frames or right['frame_index']!=frames:raise ValueError('Noncontiguous frames')
            for k in ('frame_index','timestamp_ns','segment','source_to_reference','candidates','tracks','tracking_metrics'):
                if left[k]!=right[k]:bad.append(dict(frame_index=frames,field=k))
            lc,rc=dict(left['coverage']),dict(right['coverage']);lc.pop('detection_ms');rc.pop('detection_ms')
            if lc!=rc:bad.append(dict(frame_index=frames,field='coverage'))
            if frames:
                backends=right['motion']['motion_backends']
                if backends.get('cpu_fallback') is not False or any(backends.get(k)!='PVA' for k in ('gaussian_pyramid','harris','optical_flow_pyrlk')):
                    raise ValueError('PVA not actually used')
                pva_pairs+=1
            frames+=1
    ref=ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914'
    evaluated=evaluate(after,'0126',ref)
    pilot=ROOT/'results/tiny_target/phase20/accuracy_baseline_v1_20260914'
    pf=json.loads((pilot/'scoring_freeze.json').read_text())
    if digest(pilot/'annotations.json')!=pf['labels_sha256'] or digest(pilot/'source_review/packet.json')!=pf['packet_sha256']:
        raise ValueError('Pilot changed')
    with (after/'frames.jsonl').open() as f:
        pilot_score=score_rows(map(json.loads,f),json.loads((pilot/'annotations.json').read_text()),
            json.loads((pilot/'source_review/packet.json').read_text()),'0126',launches[1]['fps'])
    result=dict(frames=frames,actual_pva_pairs=pva_pairs,configuration_changes=differences,
        exact_candidates_tracks_and_coverage=not bad,difference_count=len(bad),first_differences=bad[:20],
        before_fps=reports[0]['processed_fps'],after_fps=reports[1]['processed_fps'],
        speedup=reports[1]['processed_fps']/reports[0]['processed_fps'],
        timings_ms={'before':reports[0]['timings_ms'],'after':reports[1]['timings_ms']},
        scored=evaluated,pilot_score=pilot_score,full_clip_validation=False,real_time=False,
        external_accelerators=launches[1]['external_accelerators'],
        caveat='Single serial prefix comparison, not sustained real-time/thermal validation; kernel copies included.',
        artifacts_sha256={str(d/n):sha256(d/n) for d in (before,after) for n in ('launch.json','frames.jsonl','report.json')})
    write(a.root/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('frames','actual_pva_pairs','exact_candidates_tracks_and_coverage','difference_count','before_fps','after_fps','speedup')}))
    if bad:raise SystemExit(1)


if __name__=='__main__':main()
