"""GPU float median prototype: transfers included, CPU oracle, no pipeline promotion."""
import argparse
import cProfile
import ctypes
import io
import json
from pathlib import Path
import pstats
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector,sha256


class CudaMedian:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(Path(path).resolve()))
        self.lib.seaqr_create.restype=ctypes.c_void_p
        self.lib.seaqr_destroy.argtypes=[ctypes.c_void_p]
        self.lib.seaqr_median.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
        self.lib.seaqr_median.restype=ctypes.c_int
        self.handle=self.lib.seaqr_create()
        if not self.handle:raise RuntimeError('CUDA workspace allocation failed')

    def __call__(self,im,k):
        if im.ndim!=2 or im.dtype!=np.float32 or k!=5 or min(im.shape)<1 or not np.isfinite(im).all():
            raise ValueError('Only finite nonempty float32 2D median5 is supported')
        im=np.ascontiguousarray(im);out=np.empty_like(im)
        code=self.lib.seaqr_median(self.handle,im.ctypes.data,out.ctypes.data,*im.shape)
        if code:raise RuntimeError(f'CUDA median failed: {code}; no fallback')
        return out

    def close(self):
        if self.handle:self.lib.seaqr_destroy(self.handle);self.handle=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    if a.source.name!='chunk_0126.avi' or sha256(a.source)!='c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344':
        raise ValueError('Explicit development source126 only')
    cfg=VisibleConfig(**json.loads(a.config.read_text()))
    if cfg.learning_protection_enabled or cfg.spatial_background!='median5':
        raise ValueError('Surviving no-protection median5 configuration required')
    cv2.setNumThreads(cfg.opencv_threads)
    gpu=CudaMedian(a.library);cpu=cv2.medianBlur
    rng=np.random.default_rng(20260914)
    cases=0
    try:
        for h,w in [(1,1),(1,19),(17,1),(3,7),(33,67),(127,129)]:
            for pattern in ('random','flat','impulse','ramp'):
                x=rng.normal(25,30,(h,w)).astype(np.float32)
                if pattern=='flat':x[:]=13.5
                if pattern=='impulse':x[:]=0;x[h//2,w//2]=255
                if pattern=='ramp':x=np.arange(h*w,dtype=np.float32).reshape(h,w)%257
                expected=cpu(x,5);actual=gpu(x,5)
                if not np.array_equal(expected,actual):raise AssertionError(f'Median mismatch {h,w,pattern}')
                cases+=1
        old,new=VisiblePointDetector(cfg),VisiblePointDetector(cfg)
        cap=cv2.VideoCapture(str(a.source));cap.set(cv2.CAP_PROP_POS_FRAMES,70)
        timings={'cpu_detector_ms':[],'gpu_median_detector_ms':[],'cpu_median_ms':[],'gpu_median_with_transfers_ms':[]}
        prof=cProfile.Profile()
        for i in range(20):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Decode failed')
            x=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            valid=np.ones(x.shape,bool)
            # Separate isolated median timings, both directions, first 4 warm up.
            medians={}
            for name,fn in ([('cpu',cpu),('gpu',gpu)] if i%2==0 else [('gpu',gpu),('cpu',cpu)]):
                start=time.perf_counter();out=fn(x,5);medians[name]=(out,1000*(time.perf_counter()-start))
            if not np.array_equal(medians['cpu'][0],medians['gpu'][0]):raise AssertionError('Native median differs')
            results={};elapsed={}
            for name,det,fn in ([('cpu',old,cpu),('gpu',new,gpu)] if i%2==0 else [('gpu',new,gpu),('cpu',old,cpu)]):
                cv2.medianBlur=fn
                start=time.perf_counter()
                results[name]=det.update(x,valid,0)
                elapsed[name]=1000*(time.perf_counter()-start)
            cv2.medianBlur=cpu
            for r in results.values():r[1].pop('detection_ms')
            if results['cpu']!=results['gpu']:raise AssertionError('Candidates or coverage changed')
            for field in ('background','variance','previous_valid'):
                if not np.array_equal(getattr(old,field),getattr(new,field)):raise AssertionError('Detector state changed')
            if i>=4:
                timings['cpu_detector_ms'].append(elapsed['cpu'])
                timings['gpu_median_detector_ms'].append(elapsed['gpu'])
                timings['cpu_median_ms'].append(medians['cpu'][1])
                timings['gpu_median_with_transfers_ms'].append(medians['gpu'][1])
            print(json.dumps({'paired_native_frames':i+1}),flush=True)
        cap.release()
        # Profiling is a separate, untimed extra frame; paired timings above
        # apply the same instrumentation to both implementations.
        prof.enable();old.update(x,valid,0);prof.disable()
        stream=io.StringIO();pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
        record=dict(exact_median_cases=cases,exact_native_detector_pairs=20,
            exact_candidates_coverage_state=True,source_frames_inclusive=[70,89],
            thresholds_changed=False,quantization=False,gpu_transfers_included=True,
            profiling_separate_from_timing=True,profile_frames=1,benchmark_not_end_to_end=True,production_promoted=False,
            timings={k:dict(median=float(np.median(v)),samples=v) for k,v in timings.items()},
            profile=stream.getvalue(),source_sha256=sha256(a.source),config_sha256=sha256(a.config),
            library_sha256=sha256(a.library),script_sha256=sha256(__file__))
        a.output.write_text(json.dumps(record,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k not in ('profile','timings')}))
        print(stream.getvalue())
    finally:
        cv2.medianBlur=cpu;gpu.close()


if __name__=='__main__':main()
