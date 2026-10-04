"""Bounded three-panel review of output channels using existing source crops."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from replay_visible_output import sha

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'outputs/seaqr_source_check_20260927'
GREEN,AMBER=(75,235,105),(0,190,255)


def load(path):return json.loads(Path(path).read_text())


def selected(records,roi):
    x,y,w,h=roi
    return [r for r in records if x<=r['source_xy'][0]<x+w and y<=r['source_xy'][1]<y+h]


def annotate(panel,records,roi,scale):
    x,y,_,_=roi
    for r in records:
        px=round((r['source_xy'][0]-x)*scale);py=round((r['source_xy'][1]-y)*scale)
        color=GREEN if r['measured'] else AMBER
        if r['measured']:
            cv2.circle(panel,(px,py),10,color,1,cv2.LINE_AA)
            label=r['track_id']+' M'
        else:
            # Dashed square cannot be confused with the measured circular marker.
            for dx in (-10,10):
                cv2.line(panel,(px+dx,py-10),(px+dx,py-4),color,1)
                cv2.line(panel,(px+dx,py+4),(px+dx,py+10),color,1)
            for dy in (-10,10):
                cv2.line(panel,(px-10,py+dy),(px-4,py+dy),color,1)
                cv2.line(panel,(px+4,py+dy),(px+10,py+dy),color,1)
            age=r['last_measurement_age_ns']
            label=r['track_id']+(f' P +{age/1e9:.1f}s' if age is not None else ' P age?')
        tw=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0]
        tx=max(0,min(px+13,panel.shape[1]-tw-2));ty=max(12,min(py-12,panel.shape[0]-2))
        cv2.putText(panel,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(panel,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)


def render(run,output):
    run,output=Path(run).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh video directory required')
    summary=load(run/'summary.json');freeze=load(run/'freeze.json')
    assert summary['completed'] and summary['freeze_sha256']==sha(run/'freeze.json')
    for name in ('source_packet0029','source_packet0082'):
        entry=freeze['inputs'][name];assert sha(entry['path'])==entry['sha256']
    specs={'0029':([0,59],[1550,2600,700,350],1),
           '0082':([300,379],[3150,2880,240,200],3)}
    output.mkdir(parents=True)
    validation={}
    for clip,(interval,roi,scale) in specs.items():
        lo,hi=interval;x,y,w,h=roi;pw,ph=w*scale,h*scale
        gap=16;ww=3*pw+2*gap;hh=ph+190
        channels_path=run/clip/'channels.jsonl'
        assert sha(channels_path)==summary['clips'][clip]['channels_sha256']
        rows={}
        with channels_path.open() as f:
            for line in f:
                row=json.loads(line)
                if lo<=row['frame_index']<=hi:rows[row['frame_index']]=row
        assert sorted(rows)==list(range(lo,hi+1))
        packet=load(freeze['inputs'][f'source_packet{clip}']['path'])
        if clip=='0029':
            inputs={r['source_frame']:(SOURCE/'moving0029/dense'/f"original_f{r['source_frame']:04d}.png",r['crop_file_sha256']) for r in packet['frames']}
        else:
            inputs={r['frame_index']:(SOURCE/'clutter0082'/f"f{r['frame_index']:04d}_native.png",r['png_sha256']) for r in packet['frames']}
        target=output/f'chunk{clip}_source_alerts_track_context.mp4'
        process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo',
            '-pix_fmt','bgr24','-s',f'{ww}x{hh}','-framerate','10','-i','pipe:0','-an',
            '-c:v','libx264','-threads','2','-crf','8','-preset','veryfast','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        rendered=[]
        try:
            for frame,row in rows.items():
                path,expected=inputs[frame];assert sha(path)==expected
                pixels=cv2.imread(str(path));assert pixels.shape[:2]==(h,w)
                original=cv2.resize(pixels,(pw,ph),interpolation=cv2.INTER_NEAREST)
                fresh,context=original.copy(),original.copy()
                alerts=selected(row['observation_alerts'],roi);tracks=selected(row['track_context'],roi)
                assert all(t['measured'] for t in alerts)
                annotate(fresh,alerts,roi,scale);annotate(context,tracks,roi,scale)
                canvas=np.full((hh,ww,3),23,np.uint8)
                def text(message,xx,yy,color=(235,235,235),size=.65):
                    cv2.putText(canvas,message,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
                for i,panel in enumerate((original,fresh,context)):
                    origin=i*(pw+gap);canvas[90:90+ph,origin:origin+pw]=panel
                assert np.array_equal(canvas[90:90+ph,:pw],original)
                text('ORIGINAL / no overlays',12,27)
                text('OBSERVATION ALERTS / measured now',pw+gap+12,27,GREEN)
                text('TRACK CONTEXT / gaps retained',2*(pw+gap)+12,27)
                text(f'chunk{clip} | frame {frame} | source time {frame/10:.1f}s | {scale}x nearest-neighbor | unchanged saved tracker',12,63)
                predictions=sum(not t['measured'] for t in tracks)
                text(f'In crop: {len(alerts)} current qualified observations; {predictions} retained predictions. No prediction is an observation alert.',12,ph+119)
                text('Green circle M = actual measurement. Dashed amber P = prediction; +time is age since last measurement.',12,ph+146)
                text('Physical class unknown. No airborne/false-positive claim. 10fps AVI playback; original lossless crops retained.',12,ph+175,size=.59)
                process.stdin.write(canvas.tobytes())
                rendered.append(dict(frame=frame,alerts=alerts,context=tracks))
            process.stdin.close();assert process.wait(timeout=60)==0
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
        cap=cv2.VideoCapture(str(target));count=0;errors=[]
        qa={25,26,59} if clip=='0029' else {344,349,353}
        while True:
            ok,encoded=cap.read()
            if not ok:break
            frame=lo+count;assert encoded.shape==(hh,ww,3)
            pixels=cv2.imread(str(inputs[frame][0]));expected=cv2.resize(pixels,(pw,ph),interpolation=cv2.INTER_NEAREST)
            diff=np.abs(encoded[90:90+ph,:pw].astype(float)-expected.astype(float))
            errors.append(dict(frame=frame,mean_dn=float(diff.mean()),max_dn=float(diff.max())))
            if frame in qa:assert cv2.imwrite(str(output/f'qa_{clip}_f{frame:04d}.png'),encoded)
            count+=1
        cap.release();assert count==hi-lo+1
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(target)]))
        meta=probe['streams'][0]
        assert int(meta['nb_frames'])==count and meta['avg_frame_rate']=='10/1'
        validation[clip]=dict(path=str(target),sha256=sha(target),ffprobe=probe,decoded_frames=count,
            channel_sha256=sha(channels_path),roi_xywh=roi,source_scale=scale,
            all_left_pixels_exact_before_lossy_encoding=True,encoded_left_errors=errors,rendered=rendered)
    with (output/'validation.json').open('x') as f:json.dump(dict(videos=validation,
        run_summary_sha256=sha(run/'summary.json'),renderer_sha256=sha(__file__),
        encoding='H264 CRF8 viewing copies; no contrast enhancement; source PNGs retained'),f,indent=2)
    print(json.dumps({k:v['path'] for k,v in validation.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.run,args.output)
