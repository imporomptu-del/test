"""Alternating short end-to-end checks after the full-clip candidate gate."""
import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256
from compare_phase20_exact_runs import without_timing,shape_accelerator,transition_for_pair,validate_decode,EXECUTION_FIELDS


def check_prefix(reference,output,count):
    with (reference/'frames.jsonl').open() as f:
        before=[json.loads(r) for r in islice(f,count)]
    with (output/'frames.jsonl').open() as f:
        after=[json.loads(r) for r in f]
    if len(before)!=count or len(after)!=count:
        raise ValueError('Truncated repeat prefix')
    for i,(a,b) in enumerate(zip(before,after)):
        if a['frame_index']!=i or b['frame_index']!=i or without_timing(a)!=without_timing(b):
            raise AssertionError('Repeat changed non-timing output at '+str(i))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite repeated measurements')
    frozen=json.loads((a.candidate/'freeze.json').read_text())
    status=json.loads((a.candidate/'status.json').read_text())
    if status['running'] or status['error'] or len(status['completed'])!=4:
        raise ValueError('Full candidate gate must pass first')
    reference_freeze=json.loads((a.reference/'freeze.json').read_text())
    if frozen['sources']!=reference_freeze['sources'] or set(frozen['sources'])!={'0029','0126','0055','0082'}:
        raise ValueError('Unexpected source scope')
    for name,expected in frozen['files_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Frozen runtime changed: '+name)
    for path in (a.reference,a.candidate):
        f=json.loads((path/'freeze.json').read_text())
        if sha256(path/'config.json')!=f['config_sha256'] or sha256(path/'libseaqr_integrated.so')!=f['compiled_library_sha256']:
            raise ValueError('Repeat configuration/library changed')
    configs=[asdict(VisibleConfig(**json.loads((path/'config.json').read_text()))) for path in (a.reference,a.candidate)]
    if any(configs[0][key]!=configs[1][key] for key in configs[0] if key not in EXECUTION_FIELDS):
        raise ValueError('Repeat changed policy')
    a.output.mkdir()
    record=dict(passed=False,comparisons=[],prefix_frames=128,alternating_pairs_per_clip=3,
        clip_ids=['0126','0082'],full_clip_fps_claim=False,labels_used_during_processing=False,
        no_frame_skipping=True,no_power_clock_or_service_changes=True,
        reference_freeze_sha256=sha256(a.reference/'freeze.json'),candidate_freeze_sha256=sha256(a.candidate/'freeze.json'),
        script_sha256=sha256(__file__),
        caveat='Three alternating before/after pairs on two fixed 128-frame development prefixes. Includes decode and journaling, excludes hashing/startup. Natural dynamic clocks and load, not locked-clock or repeated full-video distributions.')
    telemetry=None
    try:
        with (a.output/'tegrastats.log').open('x') as log:
            telemetry=subprocess.Popen(['/usr/bin/tegrastats','--interval','1000'],stdout=log,stderr=subprocess.STDOUT)
        for cid in record['clip_ids']:
            source=frozen['sources'][cid]
            if source['path']!='/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_'+cid+'.avi' or sha256(source['path'])!=source['sha256']:
                raise ValueError('Repeat source changed')
            for pair in range(3):
                values={}
                order=('before','after') if pair%2==0 else ('after','before')
                for label in order:
                    base=a.reference if label=='before' else a.candidate
                    output=a.output/f'{cid}_pair{pair}_{label}'
                    command=[sys.executable,'-m','tiny_target.visible_baseline','--source',source['path'],
                        '--config',str(base/'config.json'),'--motion-config',str(ROOT/'configs/evaluation/phase20_motion_v8.json'),
                        '--output',str(output),'--max-frames','128']
                    print(json.dumps(dict(clip=cid,pair=pair,implementation=label,started_unix=time.time())),flush=True)
                    with output.with_suffix('.log').open('x') as log:
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                    reference=a.reference/('pva_'+cid)
                    check_prefix(reference,output,128)
                    launch=json.loads((output/'launch.json').read_text())
                    expected_launch=json.loads((reference/'launch.json').read_text())
                    shape_accelerator(launch)
                    if launch['source_sha256']!=source['sha256'] or launch['fps']!=expected_launch['fps']:
                        raise ValueError('Repeat provenance changed')
                    transition_for_pair(expected_launch,launch,None if label=='before' else frozen.get('gpu_transition'))
                    report=json.loads((output/'report.json').read_text())
                    decode_stats = validate_decode(launch, report)
                    if not report['completed'] or report['frames']!=128 or report['full_clip']:
                        raise ValueError('Unexpected repeat extent')
                    values[label]=dict(fps=report['processed_fps'],timings_ms=report['timings_ms'],
                        frame_decode=decode_stats,
                        timing_note='Overlapping worker durations are not summed as wall latency.',
                        journal_sha256=sha256(output/'frames.jsonl'),exact_non_timing_prefix=True)
                record['comparisons'].append(dict(clip_id=cid,pair=pair,order=list(order),**values,
                    speedup=values['after']['fps']/values['before']['fps']))
        record['summary']={cid:dict(
            before_fps_median=float(np.median([r['before']['fps'] for r in record['comparisons'] if r['clip_id']==cid])),
            after_fps_median=float(np.median([r['after']['fps'] for r in record['comparisons'] if r['clip_id']==cid])),
            paired_speedups=[r['speedup'] for r in record['comparisons'] if r['clip_id']==cid]) for cid in record['clip_ids']}
        record['passed']=True
    except Exception as exc:
        record['error']=repr(exc)
        raise
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait(timeout=10)
        with (a.output/'repeat_summary.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
