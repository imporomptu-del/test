from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector
from tiny_target.visible_resident import sample_layout,SparseSpatial,PEAK_DTYPE
from tiny_target.visible_shapes import consolidate_half_height


class ResidentHostTests(unittest.TestCase):
    def test_samples_match_original_tile_slices(self):
        for shape in [(1,1),(67,99),(100,257)]:
            a=np.arange(np.prod(shape)).reshape(shape)
            layout,indices=sample_layout(shape,32,3)
            for ys,xs,offset,count in layout:
                self.assertTrue(np.array_equal(a.ravel()[indices[offset:offset+count]],a[ys,xs][::3,::3].ravel()))

    def test_sparse_shape_matches_dense_with_overlap(self):
        a=np.zeros((67,99),np.float32);a[30:33,30:33]=20;a[31,31]=30;a[31,34]=10
        ps=[dict(x=x,y=31,polarity='bright',score=10.,response_dn=20.,noise_sigma_dn=2.) for x in (31,34,55)]
        seeds=np.array([[p['x'],p['y']] for p in ps]);patches=np.stack([a[y-8:y+9,x-8:x+9] for x,y in seeds])
        s=SparseSpatial(a.shape,seeds,patches);mask=np.ones(a.shape,bool)
        self.assertEqual(consolidate_half_height(ps,a,mask,include_support=True),
                         consolidate_half_height(ps,s,mask,include_support=True))
        with self.assertRaises(ValueError):s[np.array([0]),np.array([0])]
        with self.assertRaises(ValueError):s[0:2,0:2]

    def test_empty_sparse_and_abi(self):
        s=SparseSpatial((10,10),[],[])
        self.assertEqual(consolidate_half_height([],s,np.ones((10,10),bool))[0],[])
        self.assertEqual(PEAK_DTYPE.itemsize,20)

    def test_resident_config_fails_for_unsupported_modes(self):
        cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',
            cuda_median_library='/explicit/library.so',state_update_backend='cuda_resident',
            pixel_noise_enabled=True,pixel_noise_model='background_residual')
        for change in [dict(pixel_noise_enabled=False),dict(tile_size=512),dict(max_candidates_per_frame=513),
                       dict(max_candidates_per_tile_polarity=17),dict(pixel_noise_model='frame_difference')]:
            with self.assertRaises(ValueError):replace(cfg,**change)

    def test_missing_resident_library_fails_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',
                cuda_median_library=str(Path(directory)/'missing.so'),state_update_backend='cuda_resident',
                pixel_noise_enabled=True,pixel_noise_model='background_residual')
            with self.assertRaises(FileNotFoundError):VisiblePointDetector(cfg)

    def test_native_shapes_are_explicit_hashed_and_off_by_default(self):
        self.assertIsNone(VisibleConfig().native_shape_library)
        self.assertIsNone(VisibleConfig().native_shape_library_sha256)
        cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',
            cuda_median_library='/explicit/gpu.so',state_update_backend='cuda_resident',
            pixel_noise_enabled=True,pixel_noise_model='background_residual',
            shape_measurement_mode='mutual_half_height_r8',
            native_shape_library='/explicit/native.so',native_shape_library_sha256='a'*64)
        for changes in [dict(native_shape_library=''), dict(native_shape_library=None),
                        dict(native_shape_library_sha256=None),dict(native_shape_library_sha256='z'*64),
                        dict(state_update_backend='indexed'),dict(shape_measurement_mode='none')]:
            with self.assertRaises(ValueError):replace(cfg,**changes)
