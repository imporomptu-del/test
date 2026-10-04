#!/usr/bin/env python3
"""Compact descriptive accounting for the frozen image-preservation experiment."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ARMS=('global_translation','local_translation','local_affine')
PATHS={
    'current_peak_ratio':('metrics','current_target_delta_comparison','ratios','peak_abs_dn'),
    'current_l1_ratio':('metrics','current_target_delta_comparison','ratios','l1_dn'),
    'current_l2_ratio':('metrics','current_target_delta_comparison','ratios','l2_dn'),
    'current_centroid_error_px':('metrics','current_target_delta_comparison','centroid_error_px'),
    'residual_template_gain':('metrics','delta','template_gain'),
    'residual_l1_ratio':('metrics','delta_comparison','ratios','l1_dn'),
    'residual_centroid_error_px':('metrics','delta_comparison','centroid_error_px'),
    'oracle_error_rms_dn':('metrics','oracle_error','rms_dn'),
    'isolated_response_over_clean_rms':('metrics','visibility','isolated_response_over_clean_rms'),
    'isolated_response_over_abs_clean_template':('metrics','visibility','isolated_response_over_abs_clean_template'),
    'clean_rms_dn':('metrics','visibility','clean_rms_dn'),
    'valid_fraction':('valid_fraction',),
    'oracle_absolute_mass_coverage':('oracle_absolute_mass_coverage',),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(record,path):
    for key in path:
        if record is None: return None
        record=record.get(key)
    return record


def stats(values):
    valid=[float(v) for v in values if v is not None]
    if not all(np.isfinite(valid)): raise ValueError('nonfinite summary value')
    return dict(total=len(values),count=len(valid),null_count=len(values)-len(valid),
        minimum=min(valid) if valid else None,median=float(np.median(valid)) if valid else None,
        p90=float(np.quantile(valid,.9)) if valid else None,maximum=max(valid) if valid else None)


def arm_summary(records):
    eligible=[r for r in records if r['full_support_eligible']]
    reasons=Counter(reason for r in records for reason in r['support_reasons'])
    return dict(total=len(records),full_support_eligible=len(eligible),
        roi_fully_supported=sum(r['full_roi_support'] for r in records),
        unavailable=sum(r['unavailable'] for r in records),partial=sum(r['partial'] for r in records),
        source_psf_partial=sum(r['source_psf_partial'] for r in records),
        support_reasons_nonexclusive=dict(reasons),
        metrics_on_all_available={name:stats([get(r,path) for r in records]) for name,path in PATHS.items()},
        metrics_on_full_support={name:stats([get(r,path) for r in eligible]) for name,path in PATHS.items()})


def summarize_probes(probes):
    result=dict(total=len(probes),arms={a:arm_summary([p['arms'][a] for p in probes]) for a in ARMS},pairwise={})
    for left,right in itertools.combinations(ARMS,2):
        common=[p for p in probes if all(p['arms'][a]['full_support_eligible'] for a in (left,right))]
        result['pairwise'][f'{left}__{right}']=dict(total=len(probes),common_full_support=len(common),
            missing_count=len(probes)-len(common),query_ids=[p['id'] for p in common],
            arms={a:{name:stats([get(p['arms'][a],path) for p in common]) for name,path in PATHS.items()} for a in (left,right)})
    result['all_three_full_support_count']=sum(all(p['arms'][a]['full_support_eligible'] for a in ARMS) for p in probes)
    return result


def boundary_sequences(rows):
    result=[]
    for row in rows:
        for site in range(4):
            for sign in (-16.,16.):
                probes=[p for p in row['probes'] if p['group']=='boundary' and p['site_id']==site and p['amplitude']==sign]
                assert [p['sweep_step'] for p in probes]==[-1.,-.5,0.,.5,1.]
                for arm in ARMS:
                    flags=[p['arms'][arm]['full_support_eligible'] for p in probes]
                    centroids=[get(p['arms'][arm],('metrics','current_target_delta','centroid_xy')) if full else None
                               for p,full in zip(probes,flags)]
                    steps=[None if a is None or b is None else (np.asarray(b)-a).tolist() for a,b in zip(centroids,centroids[1:])]
                    result.append(dict(previous_index=row['previous_index'],site_id=site,amplitude=sign,arm=arm,
                        full_support=flags,current_centroids_xy=centroids,adjacent_centroid_steps_xy=steps,
                        adjacent_step_length_px=[None if v is None else float(np.linalg.norm(v)) for v in steps],
                        adjacent_eligibility_changes=sum(a!=b for a,b in zip(flags,flags[1:])),
                        interpretation='Spatial sweep under one fixed field, not video tracking; gaps not bridged.'))
    return result


def generated_summary(generated):
    output=dict(counts=generated['counts'],groups=[])
    for group in ('trajectories','support_dropouts','static_repeats'):
        for camera in ('identity','translation','affine'):
            for sigma in (.6,1.2):
                trajectories=[t for t in generated[group] if t['camera']==camera and t['sigma_px']==sigma]
                if not trajectories: continue
                frames=[f for t in trajectories for f in t['frames']]
                pairs=[p for t in trajectories for p in t['adjacent']]
                output['groups'].append(dict(group=group,camera=camera,sigma=sigma,trajectories=len(trajectories),
                    frames=len(frames),missing_frames=sum(not f['available'] for f in frames),
                    adjacent_pairs=len(pairs),missing_adjacent_pairs=sum(not p['available'] for p in pairs),
                    centroid_error_px=stats([f['current_target_increment']['centroid_error_to_continuous_target_px'] for f in frames]),
                    centroid_oracle_error_px=stats([f['current_target_increment']['centroid_error_px'] for f in frames]),
                    peak_ratio=stats([None if not f['available'] or
                        abs(f['current_target_increment']['continuous_oracle']['polarity_peak_dn'])<=1e-12 else
                        (f['current_target_increment']['observed']['polarity_peak_dn']/f['current_target_increment']['continuous_oracle']['polarity_peak_dn']) for f in frames]),
                    current_template_gain=stats([f['current_target_increment']['core_observed']['template_gain'] for f in frames]),
                    adjacent_centroid_step_error_px=stats([p['centroid_step_error_px'] for p in pairs]),
                    adjacent_residual_gain=stats([p['isolated_target_residual']['core_observed']['template_gain'] for p in pairs])))
    return output


def summarize(result):
    if not result['passed'] or result['hashes_before']!=result['hashes_after']:
        raise ValueError('incomplete or changed provenance')
    rows=result['rows']
    assert [r['previous_index'] for r in rows]==[0,42,85,127,170,212,255,298]
    probes=[p for r in rows for p in r['probes']]
    assert dict(Counter(p['group'] for p in probes))==dict(main=3072,boundary=320,border=128)
    assert len({p['id'] for p in probes})==3520 and all(set(p['arms'])==set(ARMS) for p in probes)
    return dict(schema='seaqr.aot.image-preservation-summary.v1',completed_counts=result['completed_counts'],
        per_pair=[dict(previous_index=r['previous_index'],dense=r['dense'],all_three_common=r['all_three_common'],
            pairwise_common=r['pairwise_common'],probes=summarize_probes(r['probes'])) for r in rows],
        groups={g:summarize_probes([p for p in probes if p['group']==g]) for g in ('main','boundary','border')},
        main_by_sigma={str(s):summarize_probes([p for p in probes if p['group']=='main' and p['sigma']==s]) for s in (.6,1.2)},
        main_by_sign={str(a):summarize_probes([p for p in probes if p['group']=='main' and p['amplitude']==a]) for a in (-16.,16.)},
        boundary_sequences=boundary_sequences(rows),generated=generated_summary(result['generated']),
        interpretation='Descriptive fixed-probe workload; model masks/oracles differ. No detector recall, calibrated SNR, end-to-end estimator safety or production acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before=sha(args.result)
    value=summarize(json.loads(args.result.read_text()))
    value.update(result_sha256=before,script_sha256=sha(__file__))
    assert sha(args.result)==before
    with args.output.open('x') as out:
        json.dump(value,out,allow_nan=False,indent=2);out.write('\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output))))


if __name__=='__main__':main()
