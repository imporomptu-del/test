from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import cv2
import numpy as np

from tiny_target.dense_screen import DensePointScreener,DenseScreenConfig,DenseScreenError
from tiny_target.raw_background_cuda import RawBackgroundCuda,RawBackgroundCudaError,checked_shape,parameters
from tiny_target.types import Frame,TimestampSource


def frame(image,index=0):
    return Frame(image=image,timestamp_ns=index*100000001,frame_index=index,
                 source_id='generated-gpu-routing-test',bit_depth=16,
                 timestamp_source=TimestampSource.MANIFEST)


class ReferenceGpuStandIn:
    """Exercise host integration separately from real device arithmetic checks."""
    def __init__(self,config,shape):
        self.cpu=DensePointScreener(replace(config,background_execution='masked_ufunc'))
        self.count=0;self.closed=False;self.resets=0
    def reset(self):
        self.count=0;self.resets+=1
        self.cpu._background_location=None;self.cpu._background_variance=None
        self.cpu._background_support=None;self.cpu._background_frame_count=0
    def close(self):self.closed=True
    def step(self,image,input_valid):
        if self.closed:raise RawBackgroundCudaError('closed')
        current=replace(frame(image,self.count),valid_mask=input_valid)
        original=cv2.filter2D;captured=[]
        def capture(a,*args,**kwargs):
            captured.append(a.copy());return original(a,*args,**kwargs)
        with patch.object(cv2,'filter2D',capture):self.cpu._events_for_frame(current)
        self.count+=1
        result=self.cpu._last_synthetic_frame
        return None if result is None else (captured[0],result.valid_mask,result.detection_ready)


class RawBackgroundCudaTests(unittest.TestCase):
    def config(self):
        return DenseScreenConfig(background_execution='masked_ufunc',
            per_frame_event_screen_enabled=False,crop_width=32,crop_height=24)

    def test_default_and_opt_in_limits(self):
        self.assertEqual(DenseScreenConfig().background_execution,'indexed_reference')
        with self.assertRaises(ValueError):replace(self.config(),background_execution='cuda_temporal_exact_v1',per_frame_event_screen_enabled=True)
        with self.assertRaises(ValueError):replace(self.config(),background_execution='cuda_temporal_exact_v1',spatial_background_radius_px=5)
        replace(self.config(),background_execution='cuda_temporal_exact_v1')

    def test_parameter_rounding_keeps_complements_separate(self):
        cfg=replace(self.config(),background_update_rate=.7,background_outlier_update_rate=.13)
        params,warmup,required=parameters(cfg)
        self.assertEqual(params[2],np.float32(1.-.7))
        self.assertEqual(params[4],np.float32(1.-.13))
        self.assertEqual((warmup,required),(4,65))
        with self.assertRaises(ValueError):parameters(replace(cfg,noise_sigma_floor_dn=1e30))

    def test_shape_bounds_and_types(self):
        self.assertEqual(checked_shape((3190,4784)),(3190,4784))
        for shape in ((0,10),(True,10),(2,),(10.,20),(4000,9000)):
            with self.subTest(shape=shape),self.assertRaises(ValueError):checked_shape(shape)

    def test_missing_library_fails_without_fallback(self):
        with self.assertRaises(FileNotFoundError):
            RawBackgroundCuda(self.config(),(24,32),Path('/this-gpu-library-does-not-exist.so'))

    def test_wrong_cpu_float_mode_is_rejected_before_allocation(self):
        lib=MagicMock();lib.seaqr_raw_background_abi.return_value=2
        with tempfile.NamedTemporaryFile() as artifact:
            with patch('tiny_target.raw_background_cuda.C.CDLL',return_value=lib), \
                 patch('tiny_target.raw_background_cuda.cpu_flush_to_zero_enabled',return_value=False):
                with self.assertRaisesRegex(RawBackgroundCudaError,'flush-to-zero'):
                    RawBackgroundCuda(self.config(),(24,32),Path(artifact.name))
        lib.seaqr_raw_background_create.assert_not_called()

    def test_old_abi_is_rejected_before_allocation(self):
        lib=MagicMock();lib.seaqr_raw_background_abi.return_value=1
        with tempfile.NamedTemporaryFile() as artifact:
            with patch('tiny_target.raw_background_cuda.C.CDLL',return_value=lib):
                with self.assertRaisesRegex(RawBackgroundCudaError,'ABI'):
                    RawBackgroundCuda(self.config(),(24,32),Path(artifact.name))
        lib.seaqr_raw_background_create.assert_not_called()

    def test_bad_input_never_reaches_cuda(self):
        gpu=RawBackgroundCuda.__new__(RawBackgroundCuda)
        gpu.handle=1;gpu.failed=False;gpu.shape=(2,3);gpu.lib=MagicMock()
        try:
            for image,mask in (
                (np.ones((2,3),np.uint16),np.ones((2,3),bool)),
                (np.full((2,3),np.nan,np.float32),np.ones((2,3),bool)),
                (np.ones((2,3),np.float32),np.ones((2,3),np.uint8)),
                (np.ones((2,3),np.float32),np.ones((3,2),bool)),
            ):
                with self.assertRaises(ValueError):gpu.step(image,mask)
            gpu.lib.seaqr_raw_background_step.assert_not_called()
        finally:gpu.close()

    def test_cuda_failure_poisons_state_without_advancing_age(self):
        gpu=RawBackgroundCuda.__new__(RawBackgroundCuda)
        gpu.handle=1;gpu.failed=False;gpu.shape=(2,3);gpu.count=0;gpu.warmup=4
        gpu.lib=MagicMock();gpu.lib.seaqr_raw_background_step.return_value=2
        gpu.lib.seaqr_raw_background_error.return_value=b'injected CUDA failure'
        with self.assertRaisesRegex(RawBackgroundCudaError,'no fallback'):
            gpu.step(np.ones((2,3),np.float32),np.ones((2,3),bool))
        self.assertTrue(gpu.failed);self.assertEqual(gpu.count,0)
        with self.assertRaises(RawBackgroundCudaError):gpu.reset()
        gpu.close();gpu.close()
        gpu.lib.seaqr_raw_background_destroy.assert_called_once_with(1)

    def test_closed_or_failed_device_cannot_continue(self):
        gpu=RawBackgroundCuda.__new__(RawBackgroundCuda)
        gpu.handle=None;gpu.failed=False
        with self.assertRaises(RawBackgroundCudaError):gpu.reset()
        gpu.handle=1;gpu.failed=True
        with self.assertRaises(RawBackgroundCudaError):gpu._live()
        gpu.handle=None

    def test_integration_preserves_reference_frames_resets_and_availability(self):
        cfg=self.config();cpu=DensePointScreener(cfg)
        gpu=DensePointScreener(replace(cfg,background_execution='cuda_temporal_exact_v1'))
        rng=np.random.default_rng(41)
        with patch('tiny_target.raw_background_cuda.RawBackgroundCuda',ReferenceGpuStandIn):
            for index in range(24):
                image=rng.integers(1000,1200,(24,32)).astype(np.float32)
                image.flat[:3]=[0,65535,32769]
                current=replace(frame(image,index),valid_mask=rng.random(image.shape)>.05)
                for screen in (cpu,gpu):screen.process(current,segment_index=index//12)
                a,b=cpu._last_synthetic_frame,gpu._last_synthetic_frame
                self.assertEqual(a is None,b is None)
                if a is not None:
                    self.assertEqual(a.response.tobytes(),b.response.tobytes())
                    self.assertEqual(a.valid_mask.tobytes(),b.valid_mask.tobytes())
                    self.assertEqual(a.detection_ready,b.detection_ready)
                    self.assertEqual(a.frame_index,b.frame_index)
                self.assertEqual(cpu._background_frame_count,gpu._background_frame_count)
            self.assertEqual(cpu.finalize(),gpu.finalize())
            self.assertEqual(gpu._background_cuda.resets,1)
            gpu.close();self.assertTrue(gpu._background_cuda.closed)

    def test_non_raw_frame_rejected_before_device_access(self):
        gpu=DensePointScreener(replace(self.config(),background_execution='cuda_temporal_exact_v1'))
        with patch('tiny_target.raw_background_cuda.RawBackgroundCuda') as constructor:
            with self.assertRaises(DenseScreenError):
                gpu.process(replace(frame(np.ones((24,32),np.uint16)),bit_depth=8))
            constructor.assert_not_called()

    def test_cpu_mode_never_loads_gpu(self):
        cpu=DensePointScreener(self.config())
        with patch('tiny_target.raw_background_cuda.RawBackgroundCuda') as constructor:
            cpu.process(frame(np.ones((24,32),np.uint16)*1000));cpu.close()
            constructor.assert_not_called()


if __name__=='__main__':unittest.main()
