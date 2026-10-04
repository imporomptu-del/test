"""Report-only provenance/state/output audit of the bounded tracking experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from profile_visible_v17 import read,sha,write
from verify_visible_overlap_v23 import ROOT,local_source,verify_dependencies
from verify_visible_v17 import require,distribution
from check_tracking_geometry_v20 import primitive_cases
from check_tracking_batch_v27 import FILES
from replay_tracking_v27 import digest
from build_tracking_batch_v27 import GEOMETRY_SHA

V26=ROOT/'results/tiny_target/visible_front_v26_20260920/evidence'
V20=ROOT/'results/tiny_target/visible_speed_v20_20260918/evidence'


def summarize(rows):
    expected=[('chunk_'+c,i,m) for c in ('0126','0082') for i in range(2)
              for m in (('reference','candidate') if i==0 else ('candidate','reference'))]
    require([(r['clip'],r['repeat'],r['mode']) for r in rows]==expected,'Changed replay timing schedule')
    result={}
    for c in ('0126','0082'):
        selected={m:[r for r in rows if r['clip']=='chunk_'+c and r['mode']==m] for m in ('reference','candidate')}
        for arm in selected.values():
            for r in arm:
                require(r['frames']==128 and not r['profiled'] and len(r['tracking_ms'])==128,'Invalid replay timing scope')
                require(all(type(v) in (int,float) and v>0 for v in r['tracking_ms']),'Invalid tracking duration')
        stats={m:distribution([v for r in arm for v in r['tracking_ms']]) for m,arm in selected.items()}
        paired=[sum(selected['reference'][i]['tracking_ms'])/sum(selected['candidate'][i]['tracking_ms']) for i in range(2)]
        result[c]=dict(tracking_ms=stats,speedup=stats['reference']['mean_ms']/stats['candidate']['mean_ms'],
            paired_speedups=paired,saved_ms_per_frame=stats['reference']['mean_ms']-stats['candidate']['mean_ms'])
    return result


def verify(evidence):
    manifest=read(evidence/'evidence_manifest.json')
    require(manifest['post_run_manifest'] and not manifest['media_decoded'] and not manifest['production_changed'],
        'Changed experiment scope')
    for n,d in manifest['files_sha256'].items():
        require(not Path(n).is_absolute() and '..' not in Path(n).parts,'Unsafe evidence path')
        require(sha(evidence/n)==d,'Changed manifested artifact '+n)
    required=set(FILES)|{'profile_0126_01.json','profile_0082_01.json','checked_replays_01.json',
        'tracking_geometry_v20.cpp','tracking_geometry_v20.py','check_tracking_geometry_v20.py',
        'build_tracking_geometry_v20.py','profile_visible_v17.py','finalize_tracking_v27.py',
        'build_01/build.json','build_01/libtracking_batch_v27.so','build_01/source/tracking_batch_v27.cpp',
        'build_01/source/tracking_geometry_v20.cpp','unit_gate.json','unit_gate.log'}
    require(set(manifest['files_sha256'])==required,'Incomplete evidence manifest')
    unit=read(evidence/'unit_gate.json')
    require(unit['passed'] and unit['returncode']==0 and unit['log_sha256']==sha(evidence/'unit_gate.log')
        and unit['test_sha256']==sha(evidence/'test_tracking_batch_v27.py'),'Unit gate changed/failed')
    dependency=verify_dependencies();g=read(evidence/'checked_replays_01.json');b=read(evidence/'build_01/build.json')
    require(g['passed'] and g['error'] is None and not g['media_read'],'Incomplete or failed replay gate')
    require(set(g['source_sha256'])==set(FILES),'Incomplete source set')
    for n,d in g['source_sha256'].items():require(d==sha(evidence/n)==sha(local_source(n)),'Changed source '+n)
    for n in ('tracking_geometry_v20.py','tracking_geometry_v20.cpp','check_tracking_geometry_v20.py',
              'build_tracking_geometry_v20.py','profile_visible_v17.py'):
        require(sha(evidence/n)==sha(V20/n)==sha(ROOT/'scripts'/n),'Changed v20 dependency '+n)
    require(b['passed'] and b['returncode']==0 and g['build_sha256']==sha(evidence/'build_01/build.json')
        and b['library_sha256']==g['library_sha256']==sha(evidence/'build_01/libtracking_batch_v27.so')
        and g['geometry_library_sha256']==dependency['library_sha256']
        and b['builder_sha256']==g['source_sha256']['build_tracking_batch_v27.py'],'Changed native build')
    expected_sources={'tracking_geometry_v20.cpp':GEOMETRY_SHA,'tracking_batch_v27.cpp':g['source_sha256']['tracking_batch_v27.cpp']}
    require(b['source_sha256']==expected_sources and all(sha(evidence/'build_01/source'/n)==d for n,d in expected_sources.items()),
        'Changed compiled geometry arithmetic')
    require('-fno-fast-math' in b['command'] and '-ffp-contract=off' in b['command'] and '-ffast-math' not in b['command'],
            'Unsafe numerical build')
    supplement_dir=evidence.parent/'numerical_supplement_v2'
    supplement=read(supplement_dir/'explicit_subnormal_02.json')
    from check_tracking_subnormal_v27 import cases as binary_cases
    require(supplement['passed'] and not supplement['media_read'] and supplement['library_sha256']==g['library_sha256']
        and supplement['original_library_sha256']==dependency['library_sha256'],'Invalid supplemental binary64 gate')
    require([c['name'] for c in supplement['cases']]==[c[0] for c in binary_cases()]
        and len(supplement['cases'])==16 and all(c['exact'] for c in supplement['cases']),'Incomplete explicit bit-pattern cases')
    require(set(supplement['source_sha256'])=={'check_tracking_subnormal_v27.py','tracking_batch_v27.py',
        'tracking_geometry_v20.py','check_tracking_geometry_v20.py'},'Incomplete numerical checker source set')
    for row,(_,_,_,_,bits,sign) in zip(supplement['cases'],binary_cases()):
        require(row['input_bits_hex']==f'{bits:016x}' and row['mean_bits_hex']==f'{sign:016x}',
                'Changed explicit subnormal input bits')
    for name,d in supplement['source_sha256'].items():
        path=supplement_dir/name if name=='check_tracking_subnormal_v27.py' else evidence/name
        require(d==sha(path)==sha(ROOT/'scripts'/name),'Changed binary64 checker dependency')
    smallest=supplement['arithmetic_fixture_nextafter_bits_hex']
    require(smallest=='0000000000000001','Original arithmetic fixture lost subnormal bits')
    display=supplement['arithmetic_fixture_display']
    require(display in ('0.0','5e-324'),'Unrecognized fixture display')
    expected_fixtures=[dict(index=i,name=f'special_d{d}_{display}',input_bits_hex=smallest) for i,d in ((64,2),(132,4))]
    require(supplement['original_subnormal_fixtures']==expected_fixtures,'Original subnormal fixture bits/names changed')
    names=[c[0] for c in primitive_cases()]
    # Names are presentation, not numerical input identity. The integer-bit
    # probe above verifies both original fixtures; the supplemental cases
    # independently check explicit positive/negative subnormal bit patterns.
    for item in expected_fixtures:names[item['index']]=item['name']
    names += [f'capacity_{t}_{n}_{d}' for t,n,d in
        ((0,0,2),(0,5,4),(1,0,2),(2,17,4),(256,512,2),(512,512,4),(513,1,2),(1,1025,2))]
    generated=g['generated']
    require([r['name'] for r in generated['cases']]==names and len(names)==144
        and all(r['exact'] for r in generated['cases']) and {r['native'] for r in generated['cases']}=={True,False},
        'Incomplete primitive geometry gate')
    require(generated['replays']==read(V20/'generated_01.json')['replays'],'Generated tracker outputs differ from archived v20')
    from tracking_batch_v27 import START,BATCH_START,BATCH_NEW
    from tracking_geometry_v20 import OLD
    import inspect,textwrap,hashlib
    from tiny_target.tracking.kalman import KalmanTrackManager
    source=textwrap.dedent(inspect.getsource(KalmanTrackManager.update).replace(START,BATCH_START).replace(OLD,BATCH_NEW))
    require(generated['transformed_sha256']==hashlib.sha256(source.encode()).hexdigest(),'Changed adapted tracker')
    profiles={}
    for c in ('0126','0082'):
        p=read(evidence/f'profile_{c}_01.json');r=p['replay'];parent=V26/(c+'_repeat0_reference')
        require(p['passed'] and p['error'] is None and not p['media_read']
            and p['script_sha256']==g['source_sha256']['replay_tracking_v27.py']
            and p['plan_sha256']==g['source_sha256']['tracking_v27_plan.md'],'Changed baseline profile')
        require(r['profiled'] and r['exact'] and r['frames']==128 and r['clip']=='chunk_'+c
            and r['geometry_library_sha256']==dependency['library_sha256'],'Wrong baseline profile scope')
        require(r['journal_sha256']==sha(parent/'frames.jsonl') and r['launch_sha256']==sha(parent/'launch.json')
            and r['report_sha256']==sha(parent/'report.json'),'Replay parent changed')
        with (parent/'frames.jsonl').open() as f:original=[json.loads(line) for line in f]
        require(len(original)==128 and len(r['digests'])==128,'Missing replay digest')
        for i,(row,d) in enumerate(zip(original,r['digests'])):
            require(row['frame_index']==d['frame']==i and digest([row['tracks'],row['tracking_metrics']])==d['output'],
                    'Independent recorded-output digest mismatch')
        populations=r['populations'];track_rows=sum(v['tracks'] for frame in populations for v in frame.values())
        require(len(populations)==128 and r['geometry_calls']==track_rows and r['geometry_fallbacks']==0,'Missing reference geometry calls')
        profiles[c]=p
    timing=summarize(g['replays'])
    for r in g['replays']:
        p=profiles[r['clip'].removeprefix('chunk_')]['replay']
        for key in ('digests','populations','journal_sha256','launch_sha256','report_sha256','geometry_library_sha256'):
            require(r[key]==p[key],'Replay changed '+key)
        require(r['exact'] and r['geometry_fallbacks']==0 and r['batch_fallbacks']==0,'Inexact/unexpected fallback replay')
        if r['mode']=='candidate':
            require(r['geometry_calls']==0 and r['batch_track_rows']==p['geometry_calls']
                and r['batch_calls']==sum(v['tracks']>0 for frame in p['populations'] for v in frame.values()),'Incomplete native batch path')
        else:require(r['geometry_calls']==p['geometry_calls'] and r['batch_calls']==r['batch_track_rows']==0,'Reference changed')
    return dict(verified=True,completed=True,scope='tracking-only saved development replays',timing=timing,
        completed_replays=8,verified_frame_instances=1024,unique_development_frames=256,
        profiles={c:p['replay']['profile'] for c,p in profiles.items()},
        exact_private_state_and_learning_digest_matches=True,full_pipeline_tested=False,media_decoded=False,
        defaults_changed=False,raw16_paused=True,production_approved=False,new_accuracy_validated=False,
        generated_cases=144,generated_tracker_scenarios=12,
        supplemental_binary64_cases=16,supplement=supplement,
        supplement_sha256=sha(supplement_dir/'explicit_subnormal_02.json'),
        source_sha256=g['source_sha256'],library_sha256=g['library_sha256'],gate_sha256=sha(evidence/'checked_replays_01.json'),
        evidence_manifest_sha256=sha(evidence/'evidence_manifest.json'),unit_gate_sha256=sha(evidence/'unit_gate.json'),
        verifier_sha256=sha(__file__),
        note='Unprofiled tracking spans exclude JSON parsing, state serialization and comparisons. '
             'They are not new pipeline FPS, sensor-to-alert latency or independent accuracy examples.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.evidence);write(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('profiles','source_sha256')},indent=2))
