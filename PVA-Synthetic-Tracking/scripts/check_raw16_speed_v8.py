"""Predeclared media-free filter, motion and stabilization experiments."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from raw16_speed_v8_common import (filter_parameters, cpu_filter, numeric_comparison,
    frozen_motion_module, estimate_identity, package_identity, gm, dense)
from profile_raw16_efficiency import sha, write_json
from tiny_target.point_filter_cuda import PointFilterCuda, DEFAULT_LIBRARY
from tiny_target.motion import MotionCorrespondences
from tiny_target.detection import integrated_gaussian_kernel


def direct_double(image, kernel, norm):
    padded = np.pad(image.astype(np.float64), 4)
    answer = np.zeros(image.shape, np.float64)
    for y in range(9):
        for x in range(9):
            answer += padded[y:y+image.shape[0], x:x+image.shape[1]] * float(kernel[y, x])
    # CPU in-place float32 division casts the Python normalizer to float32.
    return answer / float(np.float32(norm))


def filter_checks(native):
    kernel, norm = filter_parameters()
    rng = np.random.default_rng(816532)
    rows = []
    shapes = [(1, 1), (3, 7), (8, 9), (67, 100), (129, 193)]
    if native:
        shapes += [(3190, 4784)]
    for shape in shapes:
        device = PointFilterCuda(shape, kernel, norm)
        try:
            for kind in ('zero', 'constant', 'noise', 'impulses', 'mask_hole'):
                image = rng.normal(0, 3, shape).astype(np.float32)
                if kind == 'zero': image[:] = 0
                if kind == 'constant': image[:] = 2048
                if kind == 'impulses':
                    image[:] = 0
                    image[0, 0] = 4096; image[-1, -1] = -4096
                    image[shape[0]//2, shape[1]//2] = 0.001
                if kind == 'mask_hole':
                    image[shape[0]//3:shape[0]*2//3, shape[1]//3:shape[1]*2//3] = 0
                before = sha_array(image)
                timings = {'cpu': [], 'gpu': []}
                for repeat in range(3):
                    order = ('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')
                    for mode in order:
                        start = time.perf_counter()
                        result = cpu_filter(image, kernel, norm) if mode == 'cpu' else device(image)
                        timings[mode].append(time.perf_counter()-start)
                        if mode == 'cpu': a = result
                        else: b = result
                row = dict(shape=list(shape), kind=kind, **numeric_comparison(a, b, image),
                           timings_s=timings, median_s={k:statistics.median(v) for k,v in timings.items()})
                if image.size < 100000:
                    truth = direct_double(image, kernel, norm)
                    row['cpu_oracle_max_error'] = float(np.max(np.abs(a-truth)))
                    row['gpu_oracle_max_error'] = float(np.max(np.abs(b-truth)))
                if before != sha_array(image): raise AssertionError('Filter input mutated')
                rows.append(row)
                print('filter', shape, kind, row['max_abs_error'], row['numerical_screen_passed'], flush=True)
        finally:
            device.close()
    # Deliberately target the discontinuous threshold, including adjacent floats.
    shape = (67, 100); image = np.zeros(shape, np.float32)
    unit = np.zeros(shape, np.float32); unit[33, 50] = 1
    unit_response = cpu_filter(unit, kernel, norm)[33, 50]
    center = np.float32(4 / unit_response)
    device = PointFilterCuda(shape, kernel, norm)
    try:
        values = [center]
        lo = hi = center
        for _ in range(16):
            lo = np.nextafter(lo, np.float32(-np.inf)); hi = np.nextafter(hi, np.float32(np.inf))
            values.extend([lo, hi])
        threshold_rows = []
        for amplitude in values:
            image[:] = 0; image[33, 50] = amplitude
            a, b = cpu_filter(image, kernel, norm), device(image)
            threshold_rows.append(dict(amplitude=float(amplitude), cpu=float(a[33,50]), gpu=float(b[33,50]),
                                      cpu_accept=bool(a[33,50]>=4), gpu_accept=bool(b[33,50]>=4)))
    finally:
        device.close()
    return dict(cases=rows, threshold_cases=threshold_rows,
                numerical_screen_passed=all(r['numerical_screen_passed'] for r in rows),
                bit_exact=all(r['bit_exact'] for r in rows),
                threshold_decisions_exact=all(r['cpu_accept']==r['gpu_accept'] for r in threshold_rows))


def sha_array(value):
    import hashlib
    return hashlib.sha256(value.tobytes()).hexdigest()


def moving_checks():
    cfg, _ = dense.load_dense_screen_config(ROOT/'configs/evaluation/raw16_background_v7.json')
    screener = dense.DensePointScreener(cfg)
    backend = screener._synthetic_window.tracker
    extractor = screener._synthetic_extractor
    kernel, norm = filter_parameters()
    shape = (192, 256)
    device = PointFilterCuda(shape, kernel, norm)
    rows = []
    try:
        for speed in ((2., 1.), (-2., -1.), (0., 0.)):
            for flux in (0., 1., 2., 4., 8., 16.):
                rng = np.random.default_rng(516278)
                frames = {'cpu': [], 'gpu': []}
                worst = 0.
                for index in range(16):
                    image = rng.normal(0, 1, shape).astype(np.float32)
                    mask = np.ones(shape, bool)
                    mask[:4] = False; mask[-4:] = False; mask[:,:4] = False; mask[:,-4:] = False
                    mask[80:88, 50:58] = False
                    image[~mask] = 0
                    tx, ty = 128.25 + speed[0]*index*.32, 96.25 + speed[1]*index*.32
                    x, y = round(tx), round(ty)
                    psf = integrated_gaussian_kernel(.8, 3, tx-x, ty-y)
                    image[y-3:y+4, x-3:x+4] += np.float32(flux)*psf
                    a, b = cpu_filter(image, kernel, norm), device(image)
                    comparison = numeric_comparison(a, b, image)
                    if not comparison['numerical_screen_passed']: raise AssertionError('Moving-filter bound exceeded')
                    worst = max(worst, comparison['max_abs_error'])
                    for mode, response in (('cpu',a), ('gpu',b)):
                        frames[mode].append(dense._DenseMatchedFrame(response=response, valid_mask=mask,
                            timestamp_ns=index*320000000, frame_index=index, segment_index=0, detection_ready=True))
                outputs = {}
                for mode in ('cpu', 'gpu'):
                    window = backend.integrate(frames[mode])
                    ranking = extractor.ranking_surface(window)
                    batch = extractor.extract(window, ranking_surface=ranking)
                    identities = [[c.x_px, c.y_px, list(c.velocity_xy_px_s), c.candidate_index] for c in batch.candidates]
                    tx, ty = 128.25+speed[0]*2.4, 96.25+speed[1]*2.4
                    nearby = [r for r in identities if np.hypot(r[0]-tx,r[1]-ty)<=3]
                    outputs[mode] = dict(candidate_identities=identities, target_candidates=nearby,
                        valid_sha256=sha_array(window.valid_mask), support_sha256=sha_array(window.valid_support_count),
                        score_at_truth=float(window.score[round(ty),round(tx)]))
                same = all(outputs['cpu'][k] == outputs['gpu'][k] for k in
                           ('candidate_identities','target_candidates','valid_sha256','support_sha256'))
                rows.append(dict(velocity=list(speed), flux=flux, candidate_decisions_exact=same,
                                 maximum_response_error=worst, outputs=outputs))
                print('moving', speed, flux, same, flush=True)
    finally:
        device.close(); screener.close()
    return rows


def motion_checks(archive):
    oracle = frozen_motion_module(archive)
    rng = np.random.default_rng(861427)
    rows = []
    cfg_value = json.loads((ROOT/'configs/evaluation/raw16_motion_v6.json').read_text())['global_motion']
    cfg, old_cfg = gm.GlobalMotionConfig.from_mapping(cfg_value), oracle.GlobalMotionConfig.from_mapping(cfg_value)
    for n in (0, 1, 29, 30, 63, 100, 500, 1000):
        for kind in ('clean', 'noise', 'outliers', 'tie', 'threshold', 'collinear', 'sparse', 'unrelated'):
            p = rng.uniform([0,0],[4783,3189],(n,2)).astype(np.float32)
            q = p + np.array([.21875,-.65625],np.float32)
            metrics = {'usable_for_transform':True}
            if kind == 'noise': q += rng.normal(0,.2,q.shape).astype(np.float32)
            if kind == 'outliers': q[:n//3] = rng.uniform([0,0],[4783,3189],(n//3,2))
            if kind == 'tie': q[:n//2] = p[:n//2] + np.array([8.,3.],np.float32)
            if kind == 'threshold': q[::2,0] += np.nextafter(np.float32(1.),np.float32(0.))
            if kind == 'collinear': p[:,1]=100; q[:,1]=99.34375
            if kind == 'sparse':
                p[:,1] *= .05; q = p + np.array([.21875,-.65625],np.float32)
                metrics = {'usable_for_transform':False,'quality_rejection_reasons':['low_grid_coverage']}
            if kind == 'unrelated': q = rng.uniform([0,0],[4783,3189],(n,2)).astype(np.float32)
            pairs = MotionCorrespondences(previous_points=p, current_points=q,
                harris_scores=np.ones(n,np.float32),forward_backward_error_px=np.zeros(n,np.float32),
                previous_frame_index=7,current_frame_index=8,previous_timestamp_ns=0,current_timestamp_ns=320000000,
                full_image_size=(4784,3190),motion_image_size=(2392,1595),metrics=metrics,timings_ms={},backends={})
            measured = {}; identities = {}
            for mode in ('frozen','reference','batched'):
                start = time.perf_counter()
                result = (oracle.fit_global_motion(pairs,old_cfg) if mode=='frozen' else
                          gm.fit_global_motion(pairs,cfg,execution='reference' if mode=='reference' else 'translation_batched_exact_v1'))
                measured[mode] = time.perf_counter()-start
                identities[mode] = estimate_identity(result)
            exact = identities['frozen']==identities['reference']==identities['batched']
            rows.append(dict(points=n,kind=kind,exact=exact,timing_s=measured,identities=identities))
            if not exact: raise AssertionError(f'RANSAC parity failed: {n} {kind}')
    return rows


def stabilization_checks(native):
    rng = np.random.default_rng(861428)
    shapes = [(37, 59), (193, 257)] + ([(3190,4784)] if native else [])
    shifts = [(0.,0.),(2.,-3.),(.21875,-.65625),(.015625,-.015625),
              (float(np.nextafter(.015625,0.)),float(np.nextafter(-.015625,0.))), (120.,-120.)]
    rows = []
    for shape in shapes:
        image = rng.integers(0,65536,shape,dtype=np.uint16).astype(np.float32)
        mask = (rng.random(shape)>.1).astype(np.uint8)
        for tx,ty in shifts:
            matrix = np.eye(3); matrix[:2,2] = [tx,ty]
            size = (shape[1],shape[0])
            start = time.perf_counter()
            reference = cv2.warpPerspective(image,matrix,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
            reference_mask = cv2.warpPerspective(mask,matrix,size,flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            cpu_s = time.perf_counter()-start
            start = time.perf_counter()
            affine = cv2.warpAffine(image,matrix[:2],size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
            affine_mask = cv2.warpAffine(mask,matrix[:2],size,flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            affine_s = time.perf_counter()-start
            gpu = cv2.cuda_GpuMat()
            start = time.perf_counter(); gpu.upload(image)
            out = cv2.cuda.warpPerspective(gpu,matrix,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT).download()
            gpu_s = time.perf_counter()-start
            row = dict(shape=list(shape),translation=[tx,ty],
                affine_image_exact=reference.tobytes()==affine.tobytes(),affine_mask_exact=reference_mask.tobytes()==affine_mask.tobytes(),
                gpu_image_exact=reference.tobytes()==out.tobytes(),
                affine_max_error_dn=float(np.max(np.abs(reference-affine))),gpu_max_error_dn=float(np.max(np.abs(reference-out))),
                cpu_image_and_mask_s=cpu_s,affine_image_and_mask_s=affine_s,gpu_image_upload_warp_download_s=gpu_s)
            rows.append(row); print('stabilization',row,flush=True)
    return rows


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    cv2.setNumThreads(2)
    record = dict(schema_version='seaqr.raw16-speed-v8-generated.v1',real_media_read=False,
        package_sha256=package_identity(),script_sha256=sha(__file__),
        helper_sha256=sha(ROOT/'scripts/raw16_speed_v8_common.py'),
        point_library_sha256=sha(DEFAULT_LIBRARY),cuda_source_sha256=sha(ROOT/'tiny_target/detection/cuda/point_filter_v8.cu'),
        plan_sha256=sha(ROOT/'docs/raw16_speed_v8_plan.md'),native_enabled=args.native,
        passed=False,production_approved=False)
    try:
        record['motion'] = motion_checks(args.archive)
        record['filter'] = filter_checks(args.native)
        record['moving'] = moving_checks()
        record['stabilization'] = stabilization_checks(args.native)
        record['passed'] = (record['filter']['numerical_screen_passed'] and all(r['exact'] for r in record['motion']))
        record['strict_gpu_filter_gate_passed'] = record['filter']['bit_exact']
        record['stabilization_promoted'] = False
    finally:
        write_json(args.output,record)
    print(json.dumps(dict(passed=record['passed'],strict_gpu_filter_gate_passed=record['strict_gpu_filter_gate_passed'])),flush=True)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native',action='store_true')
    raise SystemExit(run(parser.parse_args()))
