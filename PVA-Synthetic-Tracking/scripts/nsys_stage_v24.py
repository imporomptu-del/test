"""Read-only common-clock v24 stage placement and correlated CUDA analysis."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch
import nsys_overlap_v23 as shared
from nsys_overlap_v23 import sha, _require as require
from nsys_timeline_v22 import clip, intersection, length
from run_visible_stage_v24 import validate_snapshot

EXTRA=('cpu_prepare','warp_wait','warp_gpu')


def validate_receipt(path):
    path=Path(path);r=json.loads(path.read_text())
    require(path.name.endswith('.v24.json') and r['schema']=='seaqr.visible-stage-v24.v1'
        and r['passed'] and r['error'] is None and r['traced'] and r['mode']=='staged'
        and r['frames']==r['processed_frames']==128 and r['clip'] in ('0126','0082'),
        'Not a clean bounded v24 stage trace')
    require(all(r[k] is False for k in ('gpu_changed','algorithm_changed','raw16_accessed','defaults_changed')),
            'Frozen scope changed')
    require(r['nvtx_push_pop_counts']==[768,768],'Incomplete stage NVTX receipt')
    validate_snapshot(r['execution'],128)
    stem=path.name[:-len('.v24.json')]
    launch_path=path.parent/stem/'launch.json';old_path=path.parent/(stem+'.v20.json')
    launch=json.loads(launch_path.read_text());old=json.loads(old_path.read_text())
    require(launch['configuration']['input_bit_depth']==8
        and launch['source'].endswith('/chunk_'+r['clip']+'.avi')
        and launch['external_accelerators']['median']['library_sha256']==shared.CUDA_SHA256
        and launch['exact_cuda_stabilization']['library_sha256']==shared.CUDA_SHA256,
        'Changed source/CUDA identity')
    require(sha(old_path)==r['baseline_receipt_sha256'] and old['passed'] and old['error'] is None
            and old['processed_frames']==128,'Failed/changed baseline chain')
    return r,dict(receipt_sha256=sha(path),launch_sha256=sha(launch_path),baseline_receipt_sha256=sha(old_path),
        original_cuda_library_sha256=shared.CUDA_SHA256,
        evidence_scope='Actual v24 receipt and launch checked. Full journal/motion verification is separate.')


def validate_extra(rows,frames,tids):
    result=defaultdict(dict)
    for r in rows:
        prefix,index,stage=r['text'].split('|');i=int(index)
        require(prefix=='seaqr24' and 0<=i<128 and stage in EXTRA,'Unexpected v24 annotation')
        require(stage not in result[i] and r['end'] is not None and r['start']<r['end'],
                'Duplicate/incomplete/reversed v24 annotation')
        require(r['globalTid']==tids['motion_worker'] and r.get('endGlobalTid') in (0,None,r['globalTid']),
                'Stage migrated/cross-thread ending')
        result[i][stage]=r
    require(set(result)==set(range(128)) and all(set(v)==set(EXTRA) for v in result.values()),
            'Missing v24 stage coverage')
    for i,parts in result.items():
        cpu,wait,warp=(parts[k] for k in EXTRA);motion=frames[i]['motion_worker']
        order=[motion['start'],cpu['start'],cpu['end'],wait['start'],wait['end'],warp['start'],warp['end'],motion['end']]
        require(order==sorted(order),'Stage nesting/order differs from policy')
        if i:require(warp['start']>=frames[i-1]['detector']['end'],'GPU warp released before preceding detector')
    return dict(result)


def attribute_warp(data,runtime,extra,names,pid):
    correlations=defaultdict(list)
    for api in runtime:
        if api['globalTid'] is not None and shared._pid_for_tid(api['globalTid'])==pid:
            correlations[api['correlationId']].append(api)
    result={i:[] for i in range(128)};counts={i:Counter() for i in range(128)}
    for kind,events in data.items():
        for event in events:
            matches=set()
            for api in correlations[event.get('correlationId')]:
                for i,parts in extra.items():
                    r=parts['warp_gpu']
                    if api['globalTid']==r['globalTid'] and r['start']<=api['start']<=api['end']<=r['end']:
                        matches.add(i)
            require(len(matches)<=1,'Ambiguous future-warp CUDA launch correlation')
            if matches:
                i,=matches;result[i].append((event['start'],event['end']))
                if kind=='KERNEL':counts[i][names[event['shortName']]]+=1
    # seaqr_warp_gaussian launches gaussian5 twice: horizontal then vertical.
    # Both passes execute inside CudaCubicTranslation.__call__(device=True).
    require(all(counts[i]==Counter(warp_cubic=1,gaussian5=2) for i in range(128)),
            'Full-resolution warp/Gaussian device coverage differs')
    return result,counts


def analyze(sqlite_path,receipt_path):
    # Reuse the unchanged, tested common-clock CUDA/scheduler implementation.
    # Only its receipt boundary is adapted; the real v24 receipt is returned
    # unchanged. No file or fabricated v23 receipt is created.
    with patch.object(shared,'validate_receipt',validate_receipt):
        base=shared.analyze(sqlite_path,receipt_path)
    db=sqlite3.connect(Path(sqlite_path).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names=dict(db.execute('SELECT id,value FROM StringIds'));ranges=[];additional=[]
        for r in shared._rows(db,'NVTX_EVENTS',tables):
            label=r['text'] if r['text'] is not None else names.get(r.get('textId'),'')
            if label.startswith(('seaqr23|','seaqr24|')):
                r['text']=label;(ranges if label.startswith('seaqr23|') else additional).append(r)
        frames,tids=shared.validate_ranges(ranges);extra=validate_extra(additional,frames,tids)
        pid=base['process']['globalPid']
        data={k:[r for r in shared._rows(db,'CUPTI_ACTIVITY_KIND_'+k,tables) if r['globalPid']==pid]
              for k in ('KERNEL','MEMCPY','MEMSET')}
        runtime=shared._rows(db,'CUPTI_ACTIVITY_KIND_RUNTIME',tables)
        warp,counts=attribute_warp(data,runtime,extra,names,pid)
        windows={}
        for first in (0,32):
            start,end=frames[first]['detector']['start'],frames[127]['detector']['start']
            scale=(127-first)*1e6;metrics={}
            for stage in EXTRA:
                spans=clip([(v[stage]['start'],v[stage]['end']) for v in extra.values()],start,end)
                metrics[stage+'_host_union_ms_per_interval']=length(spans)/scale
            for prior in ('detector','tracking'):
                for stage in ('cpu_prepare','warp_gpu'):
                    host=device=0
                    for i in range(1,128):
                        main=clip([(frames[i-1][prior]['start'],frames[i-1][prior]['end'])],start,end)
                        worker=clip([(extra[i][stage]['start'],extra[i][stage]['end'])],start,end)
                        host+=length(intersection(worker,main))
                        if stage=='warp_gpu':device+=length(intersection(clip(warp[i],start,end),main))
                    metrics[stage+'_host_with_previous_'+prior+'_ms_per_interval']=host/scale
                    if stage=='warp_gpu':metrics['warp_cuda_with_previous_'+prior+'_ms_per_interval']=device/scale
            metrics['correlated_warp_cuda_union_ms_per_interval']=length(clip(
                [span for spans in warp.values() for span in spans],start,end))/scale
            windows[str(first)+'_126']=dict(start_ns=start,end_ns=end,intervals=127-first,**metrics)
        return dict(schema='seaqr.nsys-stage-v24.v1',passed=True,clip=base['clip'],
            stage_policy_verified=True,stage_range_count=len(ranges)+len(additional),
            full_resolution_kernel_coverage={str(i):dict(c) for i,c in counts.items()},
            stage_windows=windows,common_clock_cuda_and_scheduling=base,
            sqlite_sha256=sha(sqlite_path),analyzer_sha256=sha(__file__),
            common_analyzer_sha256=sha(Path(shared.__file__)),performance_comparison=False,
            limitations=base['limitations']+[
                'Full-resolution CUDA attribution requires runtime correlation and the launching worker '
                'inside warp_gpu. Copy/memset events without such correlation are not assigned.',
                'Host cpu_prepare includes existing small VPI CUDA operations, not pure CPU execution.',
                'Host perf_counter timestamps are never mixed with the Nsight trace clock.'])
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.sqlite,a.receipt)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False)
