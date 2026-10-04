"""Local presentation export from verified v34 journals; never reruns detection."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT/'results/tiny_target/visible_validation_v34_20260923/audit_20260924'
SOURCES = {
    '0126': PROJECT/'outputs/jetson_review_clips_20260913/chunk_0126.avi',
    '0029': PROJECT/'outputs/jetson_review_clips_20260913/chunk_0029.avi',
    '0055': PROJECT/'outputs/v7_frozen_evaluation_20260913/sources/chunk_0055.avi',
    '0082': PROJECT/'outputs/v7_frozen_evaluation_20260913/sources/chunk_0082.avi',
}
# Only the already-reviewed presentation windows; not detector ROIs or labels.
ZOOMS = {
    '0126': [('01_chunk126_09s-23s_native_comparison', 90, 229, (2550, 2450, 850, 650), 150)],
    '0029': [('02_chunk029_04s-10s_native_comparison', 40, 100, (1200, 2200, 650, 650), 70),
             ('03_chunk029_22s-39s_native_comparison', 220, 390, (2550, 2250, 1100, 850), 280)],
}
GREEN, AMBER, WHITE = (85, 235, 120), (0, 190, 255), (238, 241, 245)
HEADER, FOOTER, GAP = 144, 72, 24


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def selected(row, crop):
    x, y, w, h = crop
    result = []
    for track in row['tracks']:
        if not track['qualified_moving']:
            continue
        xy = track['measurement_source_xy'] if track['measured'] else track['source_xy']
        if xy is None or len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError('Invalid saved track coordinate')
        if x <= xy[0] < x+w and y <= xy[1] < y+h:
            result.append((track, xy))
    return result


def text(canvas, message, x, y, scale=.63, color=WHITE, max_width=None):
    width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    if max_width:
        scale *= min(1.0, max_width/max(width, 1))
    cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA)


def annotate(panel, records, crop, native=False):
    x, y, w, h = crop
    sx, sy = panel.shape[1]/w, panel.shape[0]/h
    for track, xy in records:
        px, py = round((xy[0]-x)*sx), round((xy[1]-y)*sy)
        px, py = min(panel.shape[1]-1, px), min(panel.shape[0]-1, py)
        radius = 12 if native else 7
        color = GREEN if track['measured'] else AMBER
        if track['measured']:
            cv2.circle(panel, (px, py), radius, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(panel, (px-radius, py-radius), (px+radius, py+radius), color, 1, cv2.LINE_AA)
        # Source pixels at marker centers remain unobscured. IDs are offset.
        label = track['track_id']
        scale = .40 if native else .32
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        tx = max(0, min(px+radius+3, panel.shape[1]-tw-1))
        ty = max(12, min(py-8, panel.shape[0]-2))
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        text(panel, label, tx, ty, scale, color)


def canvas_for(frame, row, clip, fps, processing_fps, crop=None):
    native = crop is not None
    if crop is None:
        h, w = frame.shape[:2]
        crop = (0, 0, w, h)
        width, height = 1600, round(h*1600/w)
        left = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        records = selected(row, crop)
        annotate(left, records, crop)
        panels, output_width = [left], width
    else:
        x, y, w, h = crop
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError('Crop outside frame')
        left = frame[y:y+h, x:x+w].copy()
        right = left.copy()
        records = selected(row, crop)
        annotate(right, records, crop, native=True)
        panels, height, output_width = [left, right], h, 2*w+GAP
    output_height = HEADER+height+FOOTER
    output_height += output_height % 2
    out = np.full((output_height, output_width, 3), (27, 24, 20), dtype=np.uint8)
    offset = 0
    for panel in panels:
        out[HEADER:HEADER+height, offset:offset+panel.shape[1]] = panel
        offset += panel.shape[1]+GAP
    title = 'NATIVE-PIXEL COMPARISON' if native else 'FULL-FRAME OVERVIEW (downscaled for display)'
    index = row['frame_index']
    text(out, f'SEAQR / v34   |   chunk {clip}   |   {title}', 18, 28, .74, max_width=output_width-36)
    text(out, f'Source {index/fps:05.1f}s  /  frame {index}   |   Playback {fps:g} FPS (source time)   |   Jetson processing average {processing_fps:.2f} FPS',
        18, 57, .61, max_width=output_width-36)
    text(out, 'Green circle = measured this frame', 18, 85, .59, GREEN)
    text(out, 'Amber square = predicted only / coasted', 485, 85, .59, AMBER, max_width=output_width-503)
    if native:
        text(out, 'SOURCE / no overlays / 1:1 pixels', 18, 125, .59)
        text(out, 'SAVED ALGORITHM OUTPUT / 1:1 pixels', crop[2]+GAP+18, 125, .59)
    else:
        text(out, 'All qualified track states shown; candidates/unqualified tracks are not drawn.', 18, 120, .61)
    measured = sum(t['measured'] for t, _ in records)
    text(out, f'In this view: {measured} measured, {len(records)-measured} predicted. Motion: {row["motion"].get("status", "initialization")}.',
        18, HEADER+height+25, .59, max_width=output_width-36)
    note = ('Post-hoc review crop; all qualified states inside shown. Not an accuracy sample.' if native
        else 'Display downsampling can hide tiny objects. Use native close-ups and original AVI files.')
    text(out, note+' Markers are not verified airborne labels.', 18, HEADER+height+53, .53, max_width=output_width-36)
    return out


class Encoder:
    def __init__(self, path, shape, fps):
        self.path, self.count = path, 0
        self.shape = shape
        self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{shape[1]}x{shape[0]}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-threads', '2', '-preset', 'fast', '-crf', '12', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def send(self, frame):
        if frame.shape != self.shape:
            raise ValueError('Changed output dimensions')
        self.process.stdin.write(frame.tobytes())
        self.count += 1

    def finish(self, expected):
        self.process.stdin.close()
        if self.process.wait() != 0 or self.count != expected:
            raise RuntimeError('Incomplete encode')
        result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration',
            '-of', 'json', str(self.path)]))['streams'][0]
        if int(result['nb_read_frames']) != expected:
            raise RuntimeError('Encoded frame count mismatch')
        return result

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


def run(output):
    summary = read(AUDIT/'summary_verified_01.json')
    manifest = read(AUDIT/'evidence/export_manifest_v34_01.json')
    if not summary['verified'] or not summary['completed'] or summary['completed_trials'] != 16:
        raise ValueError('Completed verified v34 required')
    if sha(AUDIT/'evidence/export_manifest_v34_01.json') != summary['export_manifest_sha256']:
        raise ValueError('Changed audit manifest')
    if output.exists():
        raise FileExistsError('Fresh presentation folder required')
    # Validate all four local source files and selected evidence BEFORE media decode.
    launches = {}
    for clip, source in SOURCES.items():
        trial = 'full_repeat0_'+clip
        path = AUDIT/'evidence/run'/trial
        for suffix in ('/launch.json', '/report.json', '/frames.jsonl', '.v29.json', '.v34.json'):
            relative = 'run/'+trial+suffix
            if sha(AUDIT/'evidence'/relative) != manifest['files'][relative]:
                raise ValueError('Changed audited evidence '+relative)
        launch, report = read(path/'launch.json'), read(path/'report.json')
        if sha(source) != launch['source_sha256'] or not report['full_clip'] or not report['completed']:
            raise ValueError('Source/reference mismatch '+clip)
        if report['frames'] != summary['clean_full'][clip]['frames_per_run'] or launch['fps'] != 10:
            raise ValueError('Unexpected media scope')
        launches[clip] = launch
    output.mkdir(parents=True)
    for name in ('videos', 'originals', 'results', 'evidence', 'stills', 'tools'):
        (output/name).mkdir()
    shutil.copy2(__file__, output/'tools/render_visible_v34_demo.py')
    for source_name, dest in (('README.md', 'results/verified_benchmark_report.md'),
            ('summary_verified_01.json', 'results/summary_verified_01.json'),
            ('audit_01.log', 'results/audit_01.log'), ('audit_unit_01.log', 'results/audit_unit_01.log'),
            ('live_readback_01.json', 'results/live_readback_01.json')):
        shutil.copy2(AUDIT/source_name, output/dest)
    shutil.copy2(AUDIT/'evidence/freeze.json', output/'results/experiment_freeze.json')
    result = dict(schema='seaqr.v34.presentation.v1', complete=False,
        benchmark_summary_sha256=sha(AUDIT/'summary_verified_01.json'), renderer_sha256=sha(__file__),
        posthoc=True, detector_rerun=False, all_tracks_in_view=True,
        overlay_filter='qualified_moving only; all such states within view; no manual track selection',
        playback='10 FPS nominal source time, NOT Jetson processing time',
        encoding='H264 CRF12 yuv420p viewing copies; no contrast enhancement',
        synthetic_branch_enabled=False, new_airborne_accuracy_claim=False, clips={}, videos=[])
    for clip, source in SOURCES.items():
        launch = launches[clip]
        trial = 'full_repeat0_'+clip
        path = AUDIT/'evidence/run'/trial
        fps = launch['fps']
        count = summary['clean_full'][clip]['frames_per_run']
        processing_fps = summary['clean_full'][clip]['pooled_fps']
        shutil.copy2(source, output/'originals'/source.name)
        ep = output/'evidence'/trial
        ep.mkdir()
        for name in ('launch.json', 'report.json'):
            shutil.copy2(path/name, ep/name)
        # Losslessly compressed full journal retained so the displayed subset is auditable.
        with (path/'frames.jsonl').open('rb') as inp, gzip.open(ep/'frames.jsonl.gz', 'xb', compresslevel=1) as dst:
            shutil.copyfileobj(inp, dst)
        cap = cv2.VideoCapture(str(source))
        if (not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count
                or cap.get(cv2.CAP_PROP_FPS) != fps):
            raise ValueError('Unexpected video probe '+clip)
        encoders, descriptions = {}, {}
        try:
            with (path/'frames.jsonl').open() as journal:
                for index in range(count):
                    row = json.loads(next(journal))
                    if row['frame_index'] != index or row['timestamp_ns'] != round(index/fps*1e9):
                        raise ValueError('Frame/timestamp misalignment')
                    ok, frame = cap.read()
                    if not ok or frame.shape[:2] != (launch['source_probe']['height'], launch['source_probe']['width']):
                        raise ValueError('Missing/wrong source frame')
                    full_name = f'chunk{clip}_full_overview'
                    jobs = [(full_name, 0, count-1, None, min(150, count-1))]
                    jobs += ZOOMS.get(clip, [])
                    for name, first, last, crop, poster_index in jobs:
                        if not first <= index <= last:
                            continue
                        canvas = canvas_for(frame, row, clip, fps, processing_fps, crop)
                        if name not in encoders:
                            encoders[name] = Encoder(output/'videos'/(name+'.mp4'), canvas.shape, fps)
                            descriptions[name] = dict(file='videos/'+name+'.mp4', clip=clip, trial=trial,
                                first_frame=first, last_frame_inclusive=last, frame_count=last-first+1,
                                crop_xywh=crop, native_source_pixels=crop is not None,
                                other_regions_hidden=crop is not None, source_sha256=launch['source_sha256'],
                                journal_sha256=manifest['files']['run/'+trial+'/frames.jsonl'],
                                full_clip_processing_pooled_fps=processing_fps)
                        encoders[name].send(canvas)
                        if index == poster_index:
                            if not cv2.imwrite(str(output/'stills'/(name+'.png')), canvas):
                                raise RuntimeError('Poster write failed')
                        # Keep a documented coasted frame visible in QA, not silently replaced.
                        if clip == '0126' and crop is not None and index == 216:
                            if not cv2.imwrite(str(output/'stills/chunk0126_frame216_prediction.png'), canvas):
                                raise RuntimeError('Coast poster failed')
                    if index % 100 == 0:
                        print(f'Rendering chunk{clip}: {index+1}/{count}', flush=True)
                if journal.readline():
                    raise ValueError('Extra journal frames')
            ok, _ = cap.read()
            if ok:
                raise ValueError('Extra source frames')
            for name, encoder in encoders.items():
                desc = descriptions[name]
                desc['ffprobe'] = encoder.finish(desc['frame_count'])
                result['videos'].append(desc)
        finally:
            cap.release()
            for encoder in encoders.values():
                encoder.stop()
        result['clips'][clip] = dict(source='originals/'+source.name, sha256=sha(source),
            native_width=launch['source_probe']['width'], native_height=launch['source_probe']['height'],
            frames=count, nominal_fps=fps, pooled_processing_fps=processing_fps,
            evidence='evidence/'+trial)
        print('Finished chunk'+clip, flush=True)
    result['complete'] = True
    result['files'] = {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'media_manifest.json', result)
    print(json.dumps(dict(complete=True, output=str(output), videos=len(result['videos']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
