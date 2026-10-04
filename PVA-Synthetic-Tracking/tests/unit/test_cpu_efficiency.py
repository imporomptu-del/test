"""Exact execution parity for cached planning and indexed shape searches."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from tiny_target.visible_shapes import consolidate_half_height
from tiny_target.tracking.kalman import _mahalanobis_einsum_path


class CpuEfficiencyTests(unittest.TestCase):
    def test_cached_path_is_exact_for_both_measurement_dimensions(self):
        rng=np.random.default_rng(62)
        for n in (0,1,2,3,17,128,512):
            for dim in (2,4):
                r=rng.normal(size=(n,dim));a=rng.normal(size=(dim,dim));a=a@a.T
                before=np.einsum('ni,ij,nj->n',r,a,r,optimize=True)
                after=np.einsum('ni,ij,nj->n',r,a,r,optimize=_mahalanobis_einsum_path(n,dim))
                np.testing.assert_array_equal(before,after)

    def test_path_cache_is_reused_and_bounded(self):
        _mahalanobis_einsum_path.cache_clear()
        first=_mahalanobis_einsum_path(10,2)
        self.assertIs(first,_mahalanobis_einsum_path(10,2))
        self.assertEqual(_mahalanobis_einsum_path.cache_info().hits,1)
        for n in range(100):_mahalanobis_einsum_path(n,2)
        self.assertLessEqual(_mahalanobis_einsum_path.cache_info().currsize,64)

    def test_indexed_shapes_match_frozen_exhaustive_search(self):
        root=Path(__file__).resolve().parents[2]
        path=root/'results/tiny_target/phase20/efficiency_v2_20260914/pipeline/pva_0126/implementation/visible_shapes.py'
        spec=importlib.util.spec_from_file_location('frozen_exhaustive_shape',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rng=np.random.default_rng(623)
        for case in range(120):
            im=rng.normal(size=(45,61)).astype(np.float32)
            im[20,12:35]=10;im[28,18:37]=-8
            if case%7==0:im[10:22,20:33]=5
            mask=rng.random(im.shape)>.002
            coords=[(int(x),int(y)) for x,y in zip(rng.integers(0,61,55),rng.integers(0,45,55))]
            coords += [(x,20) for x in range(12,35,2)]+[(x,28) for x in range(18,37,2)]
            coords += coords[-5:]  # Duplicate seeds must retain the old semantics.
            rng.shuffle(coords)
            ps=[dict(x=x,y=y,polarity='bright' if im[y,x]>=0 else 'dark',score=float(abs(im[y,x])),
                     response_dn=float(im[y,x]),noise_sigma_dn=1.) for x,y in coords]
            radius=(case%8)+1
            self.assertEqual(old.consolidate_half_height(ps,im,mask,radius,include_support=bool(case%2)),
                             consolidate_half_height(ps,im,mask,radius,include_support=bool(case%2)))

