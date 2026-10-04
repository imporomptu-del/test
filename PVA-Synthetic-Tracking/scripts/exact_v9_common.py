"""Scoped research execution and strict cross-workspace audit comparison."""
from __future__ import annotations
from contextlib import contextmanager,ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np

from raw16_speed_v8_common import dense
from profile_raw16_efficiency import compact,sha
from summarize_raw16_cpu_v6 import digest,difference
from compare_raw16_v8_audits import COUNTS,SYNTHETIC_LIBRARY_SHA
from tiny_target.point_filter_fft_exact import PointFilterFftExact
from tiny_target.warp_translation_cuda import WarpTranslationCuda

FFT_WORKERS=4
LIBRARY_PATHS=frozenset({
    '/tmp/seaqr_raw16_speed_v8_boh3Kh/build/cuda/libtiny_target_cuda.so',
    '/tmp/seaqr_exact_v9_rS2LFx/build/cuda/libtiny_target_cuda.so',
})

def exact_array(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def normalize_audit(rows):
    def walk(value):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                if key=='library_path':
                    if item not in LIBRARY_PATHS:raise ValueError('Unknown synthetic library path')
                    result[key]='<verified-sha256:'+SYNTHETIC_LIBRARY_SHA+'>'
                else:result[key]=walk(item)
            return result
        if isinstance(value,list):return [walk(item) for item in value]
        return value
    result=walk(rows)
    for row in result:
        if row['stage']=='full_resolution_warp':
            backend=row['value']['backend']
            if backend not in ('opencv_cpu','cuda_translation_reference_v9'):
                raise ValueError('Unknown stabilization execution')
            row['value']['backend']='<verified-reference-cubic-execution>'
    return result

def compare_audits(left,right):
    raw=[[json.loads(line) for line in Path(p).read_text().splitlines()] for p in (left,right)]
    counts=[{s:sum(r['stage']==s for r in rows) for s in {r['stage'] for r in rows}} for rows in raw]
    a,b=map(normalize_audit,raw)
    complete=counts[0]==counts[1]==COUNTS
    return dict(passed=complete and a==b,complete=complete,first_difference=difference(a,b),
        event_count=len(a),normalized_left_sha256=digest(a),normalized_right_sha256=digest(b),
        raw_left_sha256=sha(left),raw_right_sha256=sha(right),
        normalization='Only two SHA-verified library paths and the explicit CPU/reference-CUDA warp execution label. No array, score or numerical state changes.')

class ExactExecution:
    def __init__(self,shadow=False):
        self.shadow=shadow;self.filters={};self.warps={};self.current=None
        self.filter_calls=0;self.warp_calls=0;self.filter_checks=[];self.warp_checks=[]

    def install(self,stack):
        cv=dense._load_cv2();original_filter=cv.filter2D
        original_events=dense.DensePointScreener._events_for_frame_cuda
        original_init=dense.FullResolutionStabilizer.__init__
        original_warp=dense.FullResolutionStabilizer._warp_cpu
        def initialized(stabilizer,*a,**kw):
            original_init(stabilizer,*a,**kw)
            cfg=stabilizer.config
            if stabilizer.backend!='opencv_cpu' or cfg.interpolation!='cubic' or cfg.border_value!=0:
                raise ValueError('Exact adapter requires frozen CPU cubic/zero-border reference')
            stabilizer.backend='cuda_translation_reference_v9'
        def events(screener,frame):
            if self.current is not None:raise RuntimeError('Nested background scope is unsupported')
            self.current=screener
            try:return original_events(screener,frame)
            finally:self.current=None
        def filtered(image,ddepth,kernel,*a,**kw):
            if self.current is None:return original_filter(image,ddepth,kernel,*a,**kw)
            if a or kw!={'borderType':cv.BORDER_CONSTANT} or ddepth!=cv.CV_32F or not exact_array(kernel,self.current._point_kernel):
                raise ValueError('Unexpected reference point-filter call')
            key=id(self.current)
            if key not in self.filters:
                self.filters[key]=PointFilterFftExact(image.shape,kernel,self.current._point_kernel_l2,workers=FFT_WORKERS)
            out=self.filters[key].correlate(image);self.filter_calls+=1
            if self.shadow:
                ref=original_filter(image,ddepth,kernel,*a,**kw)
                same=exact_array(ref,out);self.filter_checks.append(same)
                if not same:raise AssertionError('Native FFT response not bit-exact')
            return out
        def warped(stabilizer,image,mask,matrix):
            key=id(stabilizer)
            if key not in self.warps:self.warps[key]=WarpTranslationCuda(image.shape)
            start=time.perf_counter_ns();out,valid=self.warps[key](image,mask,matrix)
            elapsed=(time.perf_counter_ns()-start)/1e6;self.warp_calls+=1
            if self.shadow:
                ref,rm,_=original_warp(stabilizer,image,mask,matrix)
                same=exact_array(ref,out) and exact_array(rm,valid);self.warp_checks.append(same)
                if not same:raise AssertionError('Native warp image/mask not bit-exact')
            return out,valid,{'cuda_reference_warp_and_transfers':elapsed}
        stack.enter_context(patch.object(dense.FullResolutionStabilizer,'__init__',initialized))
        stack.enter_context(patch.object(dense.FullResolutionStabilizer,'_warp_cpu',warped))
        stack.enter_context(patch.object(dense.DensePointScreener,'_events_for_frame_cuda',events))
        stack.enter_context(patch.object(cv,'filter2D',filtered))

    def close(self):
        for dev in self.filters.values():dev.close()
        for dev in self.warps.values():dev.close()

    def record(self):
        return dict(filter_calls=self.filter_calls,warp_calls=self.warp_calls,
            filter_bit_checks=self.filter_checks,warp_bit_checks=self.warp_checks,
            fft_workers=FFT_WORKERS,filter_backend='cpu_cached_reference_fp64_fft',
            warp_backend='cuda_translation_reference_v9',experimental_v8_gpu_point_filter_used=False)
