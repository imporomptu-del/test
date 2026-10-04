"""Post-run source/reference-coordinate diagnostics in preselected light ROIs.

Image-fixed responses may be structures, sensor artifacts or tracked objects;
this observer diagnoses coordinate behavior, not physical class or false alarms.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256,map_point


def extent(points):
    return math.hypot(max(p[0] for p in points)-min(p[0] for p in points),
                      max(p[1] for p in points)-min(p[1] for p in points)) if points else 0.


def inspect(path,cid):
    if cid not in {'0055','0082'}:raise ValueError('Only the preselected development light fields')
    reference=ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914'
    packet=json.loads((reference/'scoring_packet.json').read_text())
    freeze=json.loads((reference/'scoring_freeze.json').read_text())
    if sha256(reference/'scoring_packet.json')!=freeze['packet_sha256']:raise ValueError('Review scope changed')
    window=next(w for w in packet['windows'] if w['id']==cid+'_lights')
    source=next(s for s in packet['plan']['sources'] if s['clip_id']==cid)
    launch=json.loads((path/'launch.json').read_text());report=json.loads((path/'report.json').read_text())
    if launch['source_sha256']!=source['sha256'] or not report['completed'] or not report['full_clip']:
        raise ValueError('Wrong or incomplete source run')
    x,y,w,h=window['crop_xywh'];histories=defaultdict(list);frames=0
    for row in map(json.loads,(path/'frames.jsonl').open()):
        if not window['first']<=row['frame_index']<=window['last']:continue
        frames+=1;m=row['source_to_reference']
        for t in row['tracks']:
            if not (t['measured'] and t['qualified_moving']):continue
            sx,sy=t['measurement_source_xy']
            if not (x<=sx<x+w and y<=sy<y+h):continue
            histories[(t['segment'],t['track_id'])].append(dict(frame=row['frame_index'],source_xy=[sx,sy],
                reference_xy=map_point(m,sx,sy),translation_xy=[m[0][2],m[1][2]],
                fitted_speed_px_s=math.hypot(*t['velocity_reference_xy_px_s']),historical_excursion_px=t['excursion_px']))
    if frames!=window['last']-window['first']+1:raise ValueError('Incomplete review interval')
    entries=[]
    for (segment,tid),rows in histories.items():
        raw=extent([r['source_xy'] for r in rows]);stabilized=extent([r['reference_xy'] for r in rows])
        entries.append(dict(segment=segment,track_id=tid,measured_qualified_frame_pairs=len(rows),
            first_frame=rows[0]['frame'],last_frame=rows[-1]['frame'],source_extent_px=raw,
            reference_extent_px=stabilized,transform_translation_extent_px=extent([r['translation_xy'] for r in rows]),
            last_fitted_speed_px_s=rows[-1]['fitted_speed_px_s'],last_historical_excursion_px=rows[-1]['historical_excursion_px'],
            source_extent_le_2px_with_reference_extent_ge_12px=raw<=2 and stabilized>=12,
            near_image_fixed_with_historical_motion=raw<=2 and rows[-1]['historical_excursion_px']>=12 and rows[-1]['fitted_speed_px_s']<1,
            evidence=rows))
    entries.sort(key=lambda e:(-e['measured_qualified_frame_pairs'],e['segment'],e['track_id']))
    return dict(clip_id=cid,run=str(path.resolve()),backend=launch['configuration']['motion_backend'],window=window,
        journal_sha256=sha256(path/'frames.jsonl'),measured_qualified_frame_pairs=sum(len(v) for v in histories.values()),
        transform_dominated_response_pairs=sum(e['measured_qualified_frame_pairs'] for e in entries if e['source_extent_le_2px_with_reference_extent_ge_12px']),
        near_image_fixed_historical_motion_pairs=sum(e['measured_qualified_frame_pairs'] for e in entries if e['near_image_fixed_with_historical_motion']),
        diagnostic_rule='Describe <=2px source extent but >=12px reference extent; not an object rejection rule.',
        physical_class='unresolved',false_positive_count=None,entries=entries)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--clip',choices=['0055','0082'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();value=inspect(a.run,a.clip);value['observer_sha256']=sha256(__file__)
    with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k!='entries'},indent=2))

if __name__=='__main__':main()
