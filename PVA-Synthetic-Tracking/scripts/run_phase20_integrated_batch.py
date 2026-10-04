"""One worker: strict prefix parity gates, then four frozen development clips.

No label input, automatic tuning, retry, overwrite, or holdout discovery.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256


def compare_prefix(before,after):
    left_launch=json.loads((before/'launch.json').read_text());right_launch=json.loads((after/'launch.json').read_text())
    left_report=json.loads((before/'report.json').read_text());right_report=json.loads((after/'report.json').read_text())
    if left_launch['source_sha256']!=right_launch['source_sha256'] or right_report['frames']!=230:
        raise ValueError('Prefix source/count mismatch')
    from tiny_target.visible_baseline import VisibleConfig
    from dataclasses import asdict
    configs=[asdict(VisibleConfig(**v['configuration'])) for v in (left_launch,right_launch)]
    changes={k:[configs[0][k],configs[1][k]] for k in configs[0] if configs[0][k]!=configs[1][k]}
    if set(changes)!={'cuda_median_library','stabilization_execution'}:raise ValueError('Unexpected policy change')
    with (before/'frames.jsonl').open() as first,(after/'frames.jsonl').open() as second:
        left=[json.loads(line) for line in first]
        right=[json.loads(line) for line in second]
    if len(left)!=230 or len(right)!=230:raise ValueError('Prefix is incomplete')
    differences=[]
    for i,(a,b) in enumerate(zip(left,right)):
        for key in ('frame_index','timestamp_ns','segment','source_to_reference','candidates','tracks','tracking_metrics'):
            if a[key]!=b[key]:differences.append([i,key])
        ac=dict(a['coverage']);bc=dict(b['coverage']);ac.pop('detection_ms');bc.pop('detection_ms')
        if ac!=bc:differences.append([i,'coverage'])
        if b['motion']['pva_failure'] or b['motion']['reset']:raise ValueError('Unexpected PVA failure/reset')
        if i:
            backends=b['motion']['motion_backends']
            if backends.get('cpu_fallback') is not False or any(backends.get(k)!='PVA' for k in ('gaussian_pyramid','harris','optical_flow_pyrlk')):
                raise ValueError('Actual PVA execution required')
    result=dict(exact=not differences,difference_count=len(differences),first_differences=differences[:20],
                frames=230,actual_pva_pairs=229,configuration_changes=changes,
                before_fps=left_report['processed_fps'],after_fps=right_report['processed_fps'],
                speedup=right_report['processed_fps']/left_report['processed_fps'],
                timings_ms=dict(before=left_report['timings_ms'],after=right_report['timings_ms']),
                reference_sha256=sha256(before/'frames.jsonl'),output_sha256=sha256(after/'frames.jsonl'))
    with (after/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
    if differences:raise AssertionError('Prefix parity failed; full-clip batch will not run')
    return result


def main():
    frozen=json.loads((ROOT/'freeze.json').read_text())
    if set(frozen['sources'])!={'0126','0029','0055','0082'}:raise ValueError('Unexpected source scope')
    for path,expected in frozen['files_sha256'].items():
        if sha256(ROOT/path)!=expected:raise ValueError('Frozen runtime changed: '+path)
    before=Path(frozen['remote_reference_run'])
    for name,h in frozen['reference_artifacts_sha256'].items():
        if sha256(before/name)!=h:raise ValueError('Frozen reference changed')
    lock=(ROOT/'batch.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(running=True,completed=[],current=None,started_unix=time.time(),error=None,
                library_sha256=sha256(ROOT/'libseaqr_integrated.so'))
    def save():
        pending=ROOT/'status.pending.json';pending.write_text(json.dumps(status,indent=2));pending.replace(ROOT/'status.json')
    jobs=[('host_prefix','0126',230,'host_config.json'),('resident_prefix','0126',230,'resident_config.json')]
    jobs += [('full_'+cid,cid,None,'resident_config.json') for cid in ('0126','0029','0055','0082')]
    try:
        for name,cid,limit,config in jobs:
            source=frozen['sources'][cid]
            # Access exactly the pre-authorized source, never enumerate the media directory.
            if sha256(source['path'])!=source['sha256']:raise ValueError('Changed source: '+cid)
            if (ROOT/name).exists():raise ValueError('Existing output is never overwritten: '+name)
            status.update(current=name,current_started_unix=time.time());save()
            cmd=[sys.executable,'-m','tiny_target.visible_baseline','--source',source['path'],'--config',str(ROOT/config),
                 '--motion-config',str(ROOT/'configs/evaluation/phase20_motion_v8.json'),'--output',str(ROOT/name)]
            if limit is not None:cmd+=['--max-frames',str(limit)]
            print(json.dumps({'start':name}),flush=True)
            with (ROOT/(name+'.log')).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=10800)
            report=json.loads((ROOT/name/'report.json').read_text())
            if not report['completed'] or report['frames']!=(limit or source['frames']):raise ValueError('Incomplete result')
            if limit is not None:compare_prefix(before,ROOT/name)
            status['completed'].append(dict(name=name,frames=report['frames'],fps=report['processed_fps'],
                qualified_proposal_workload=report['qualified_track_count'],elapsed_seconds=report['elapsed_seconds']))
            save();print(json.dumps(status['completed'][-1]),flush=True)
    except BaseException as exc:
        status['error']=repr(exc);raise
    finally:
        status.update(running=False,finished_unix=time.time());save();lock.close()

if __name__=='__main__':main()
