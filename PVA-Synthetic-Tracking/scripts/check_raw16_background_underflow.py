"""Generated-only diagnostic of the initial GPU prototype's underflow behavior."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_raw16_efficiency import compact,sha,write_json
from tiny_target.dense_screen import DensePointScreener,load_dense_screen_config
from tiny_target.raw_background_cuda import RawBackgroundCuda
from tiny_target.types import Frame,TimestampSource


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    cfg,_=load_dense_screen_config(ROOT/'configs/evaluation/raw16_full_frame_v2.json')
    cfg=replace(cfg,synthetic_tracking_enabled=False)
    cpu=DensePointScreener(cfg);gpu=RawBackgroundCuda(cfg,(16,24),args.library)
    image=np.full((16,24),1000,np.float32);valid=np.ones(image.shape,bool)
    try:
        for index in range(2):
            frame=Frame(image=image,valid_mask=valid,timestamp_ns=index*100000001,
                frame_index=index,source_id='generated-underflow',bit_depth=16,
                timestamp_source=TimestampSource.MANIFEST)
            cpu._events_for_frame(frame);gpu.step(image,valid)
            if index==0:
                cpu._background_variance.view(np.uint32).flat[:4]=np.array([0x00800000,0x00800001,0x007fffff,1],np.uint32)
                gpu.set_state_for_test(cpu._background_location,cpu._background_variance,cpu._background_support)
        a,b=cpu._background_variance,gpu.debug_state()['variance']
        result=dict(exact=a.tobytes()==b.tobytes(),library_sha256=sha(args.library),
            source='generated constants and bit-patterns only; no media',
            cpu_variance=compact(a),gpu_variance=compact(b),
            cpu_first_bits=a.view(np.uint32).flat[:4].tolist(),gpu_first_bits=b.view(np.uint32).flat[:4].tolist())
        write_json(args.output,result);print(result,flush=True)
    finally:gpu.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
