"""Frozen, local, four-journal check of observation output (no inference/media)."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from replay_visible_output import replay, sha

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT/'results/tiny_target/visible_validation_v34_20260923/audit_20260924'
SOURCE_CHECK = PROJECT/'outputs/seaqr_source_check_20260927'
SOURCE_126 = PROJECT/'outputs/seaqr_8bit_source_scan_20260926/encounter_50_58'
CLIPS = ('0029', '0055', '0082', '0126')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def source_reference_check(output, clip, path):
    reference = read(path)
    samples = {r['frame']: r for r in reference['samples']}
    results = []
    with (output/clip/'channels.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); frame = row['frame_index']
            if frame not in samples:
                continue
            sample = samples[frame]
            scored = sample['visibility'] == 'visible'
            gates = None
            if scored:
                gates = {str(g): {
                    channel: [dict(identity=t['identity'], source_xy=t['source_xy'],
                                   distance_px=math.dist(t['source_xy'], sample['source_xy']))
                              for t in row[channel]
                              if t['track_id'].startswith('bright:') and t['measured']
                              and math.dist(t['source_xy'], sample['source_xy']) <= g]
                    for channel in ('observation_alerts','track_context')}
                    for g in (3,5,8)}
            results.append(dict(frame=frame,visibility=sample['visibility'],scored=scored,gates=gates))
    assert [r['frame'] for r in results] == list(samples)
    visible = [r for r in results if r['scored']]
    return dict(reference_sha256=sha(path), visible_frames=len(visible),
                uncertain_frames=len(results)-len(visible),
                measured_hit_frames={str(g): [r['frame'] for r in visible if r['gates'][str(g)]['observation_alerts']]
                                     for g in (3,5,8)},rows=results,
                interpretation='Frozen source-visible localization only; not airborne-class accuracy')


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh result folder required')
    manifest_path=AUDIT/'evidence/export_manifest_v34_01.json'
    if sha(manifest_path)!='6c9272f2563368fae7c5378d0c97b225fa5f45fc7c2268badd55798c27a951a5':
        raise ValueError('Historical manifest changed')
    manifest=read(manifest_path)
    old=read(SOURCE_CHECK/'freeze.json')
    if sha(SOURCE_CHECK/'freeze.json')!='aa99071c9551f7617fae8f3c6d50e3400169ca623075d0e13d00ce0c0fcd26f7':
        raise ValueError('Source-first freeze changed')
    bindings={}
    def bind(name,path,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Changed input '+name)
        bindings[name]=dict(path=str(Path(path).resolve()),sha256=actual)
    for name in ('reference','source_review0082','source_packet0029','source_packet0082'):
        entry=old['inputs'][name];bind(name,entry['path'],entry['sha256'])
    bind('prior_freeze',SOURCE_CHECK/'freeze.json')
    bind('reference0126',SOURCE_126/'reference.json','5ebf9d9b9060e7f97858c33c7bb5491802218df131c14e8176bedd9210ab6a64')
    bind('prior_score0029',SOURCE_CHECK/'primary_score.json','62dad3380516f37222c14451b95832c9ab2b9e68f2a64dc16aab769ed136fc31')
    bind('manifest',manifest_path)
    bind('plan',ROOT/'docs/observation_output_v1_plan.md')
    for name,path in [('policy',ROOT/'tiny_target/visible_output.py'),
            ('replay',ROOT/'scripts/replay_visible_output.py'),('runner',Path(__file__)),
            ('independent_audit',ROOT/'scripts/audit_visible_output.py'),
            ('generated_tests',ROOT/'tests/unit/test_visible_output.py'),
            ('audit_tests',ROOT/'tests/unit/test_audit_visible_output.py'),
            ('replay_tests',ROOT/'tests/unit/test_replay_visible_output.py'),
            ('renderer',ROOT/'scripts/render_observation_output.py')]:
        bind(name,path)
    trials={}
    for clip in CLIPS:
        for key,file in [('journal','frames.jsonl'),('launch','launch.json'),('report','report.json')]:
            relative=f'run/full_repeat0_{clip}/{file}'
            bind(f'{key}_{clip}',AUDIT/'evidence'/relative,manifest['files'][relative])
        report=read(bindings[f'report_{clip}']['path']);launch=read(bindings[f'launch_{clip}']['path'])
        assert report['completed'] and report['full_clip'] and launch['fps']==10
        assert launch['source_sha256']==report['source_sha256']
        trials[clip]=dict(frames=report['frames'],source_sha256=report['source_sha256'])
    unchanged={str(p.resolve()):sha(p) for p in (ROOT/'tiny_target').rglob('*.py') if p.name!='visible_output.py'}
    unchanged[str((ROOT/'scripts/render_visible_v34_demo.py').resolve())]=sha(ROOT/'scripts/render_visible_v34_demo.py')
    output.mkdir(parents=True)
    write(output/'freeze.json',dict(schema='seaqr.observation-output-freeze.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),inputs=bindings,trials=trials,
        unchanged_production_sha256=unchanged,labels_changed=False,
        policy='Exactly qualified AND measured observations; preserve all qualified context, including predictions'))
    summaries={}
    for clip in CLIPS:
        entry=bindings[f'journal_{clip}']
        summaries[clip]=replay(entry['path'],entry['sha256'],trials[clip]['frames'],f'chunk{clip}',output/clip)
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_visible_output.py'),
            '--journal',entry['path'],'--channels',str(output/clip/'channels.jsonl'),
            '--summary',str(output/clip/'summary.json'),'--output',str(output/clip/'independent_audit.json')],check=True)
    reference_checks={clip:source_reference_check(output,clip,path) for clip,path in
        [('0029',SOURCE_CHECK/'reference.json'),('0126',SOURCE_126/'reference.json')]}
    prior=read(SOURCE_CHECK/'primary_score.json')['scopes']['moving0029']['summary']['gates']
    for g in ('3','5','8'):
        assert reference_checks['0029']['measured_hit_frames'][g]==prior[g]['bright']['qualified_measured']['matched_frames']
        assert reference_checks['0126']['measured_hit_frames'][g]==list(range(500,580))
    write(output/'source_reference_checks.json',reference_checks)
    for name,entry in bindings.items():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed frozen input '+name)
    for path,expected in unchanged.items():
        if sha(path)!=expected:raise ValueError('Unexpected production change '+path)
    result=dict(schema='seaqr.observation-output-check.v1',completed=True,
        frames=sum(r['frames'] for r in summaries.values()),clips=summaries,
        all_frozen_inputs_unchanged=True,existing_production_code_unchanged=True,
        all_frozen_qualified_measurement_reference_hits_preserved=True,
        freeze_sha256=sha(output/'freeze.json'),new_inference=False,remote_deployment=False,
        false_positive_improvement=None,airborne_accuracy=None,
        interpretation='Separate observation and prediction channels; display semantics, not classifier improvement')
    write(output/'summary.json',result)
    print(json.dumps(dict(frames=result['frames'],clips={k:v['counts'] for k,v in summaries.items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
