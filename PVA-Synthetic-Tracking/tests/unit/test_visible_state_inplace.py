from dataclasses import replace
import unittest
import numpy as np
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector


class InplaceStateTests(unittest.TestCase):
    def test_exact_candidates_coverage_and_state_all_modes(self):
        rng=np.random.default_rng(666)
        for noise in (False,True):
            for model in ('background_residual','frame_difference'):
                for bg in ('median5','box13'):
                    for protection in ('background_and_variance','variance_only'):
                        cfg=VisibleConfig(pixel_noise_enabled=noise,pixel_noise_model=model,
                            spatial_background=bg,warmup_frames=2,tile_size=32,
                            learning_exclusion_radius_px=3,learning_protection_mode=protection)
                        a,b=VisiblePointDetector(cfg),VisiblePointDetector(replace(cfg,state_update_backend='inplace'))
                        for i in range(16):
                            image=rng.normal(20,5,(67,99)).astype(np.float32)
                            image[30,20+i]=140
                            mask=rng.random(image.shape)>.005
                            centers=[(20+i,30)] if i%3 else []
                            x,y=a.update(image,mask,i//8,centers),b.update(image,mask,i//8,centers)
                            x[1].pop('detection_ms');y[1].pop('detection_ms')
                            self.assertEqual(x,y)
                            for name in ('background','variance','previous_spatial','previous_valid'):
                                self.assertTrue(np.array_equal(getattr(a,name,None),getattr(b,name,None)),name)

    def test_reset_resizes_scratch(self):
        d=VisiblePointDetector(VisibleConfig(state_update_backend='inplace'))
        for segment,shape in enumerate([(31,43),(51,65)]):
            d.update(np.zeros(shape,np.float32),np.ones(shape,bool),segment)
            self.assertEqual(d._state_scratch.shape,shape)

    def test_invalid_and_default(self):
        self.assertEqual(VisibleConfig().state_update_backend,'indexed')
        with self.assertRaises(ValueError):VisibleConfig(state_update_backend='gpu_magic')

    def test_cuda_is_explicit_and_missing_library_fails(self):
        self.assertEqual(VisibleConfig().spatial_filter_backend,'cpu')
        with self.assertRaises(ValueError):VisibleConfig(spatial_filter_backend='cuda_median5')
        with self.assertRaises(ValueError):VisibleConfig(cuda_median_library='/missing/lib.so')
        cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',cuda_median_library='/missing/lib.so')
        with self.assertRaises(FileNotFoundError):VisiblePointDetector(cfg)

    def test_observed_shape_protection_parity(self):
        cfg=VisibleConfig(spatial_background='median5',shape_measurement_mode='mutual_half_height_r8',
            pixel_noise_enabled=True,pixel_noise_model='background_residual',warmup_frames=2,
            learning_protection_mode='variance_only',learning_protection_geometry='observed_shape')
        a,b=VisiblePointDetector(cfg),VisiblePointDetector(replace(cfg,state_update_backend='inplace'))
        rng=np.random.default_rng(37)
        for i in range(20):
            im=rng.normal(20,2,(67,99)).astype(np.float32);im[30,20+i]=140
            args=(im,np.ones(im.shape,bool),i//10,[dict(support_reference_xy=[[20+i,30]])])
            x,y=a.update(*args),b.update(*args)
            x[1].pop('detection_ms');y[1].pop('detection_ms');self.assertEqual(x,y)
            self.assertTrue(np.array_equal(a.background,b.background))
            self.assertTrue(np.array_equal(a.variance,b.variance))
