"""CPU-only contracts for the opt-in, fail-closed CUDA bridge."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from tiny_target.visible_warp_exact import translation_maps,cubic_reference_table,CudaWarpFrame,CudaCubicTranslation
from tiny_target.visible_baseline import VisibleConfig


class ExactWarpContracts(unittest.TestCase):
    def test_identity_maps(self):
        for shape in ((1,1),(17,31),(67,131),(3190,4784)):
            xm,ym=translation_maps(shape,np.eye(3))
            for n,m in zip(shape[::-1],(xm,ym)):
                np.testing.assert_array_equal(m[:,0],np.arange(n))
                np.testing.assert_array_equal(m[:,2],np.arange(n))
                self.assertFalse(m[:,1].any())
                self.assertEqual(m.dtype,np.int32)
                self.assertTrue(m.flags.c_contiguous)

    def test_maps_keep_existing_table_quantization(self):
        m=np.eye(3);m[:2,2]=(.125,-.625)
        xm,ym=translation_maps((17,131),m)
        np.testing.assert_array_equal(xm[:,0]+xm[:,1]/32,np.arange(131)-.125)
        np.testing.assert_array_equal(ym[:,0]+ym[:,1]/32,np.arange(17)+.625)

    def test_rejects_nontranslation_and_invalid_bounds(self):
        bad=[np.zeros((3,3)),np.eye(2),np.full((3,3),np.nan)]
        for at,value in [((0,1),1e-15),((0,0),1.00001),((2,0),.01),((0,2),32701)]:
            m=np.eye(3);m[at]=value;bad.append(m)
        for m in bad:
            with self.assertRaises(ValueError):translation_maps((17,31),m)
        for shape in ((0,1),(1,0),(32767,1),(1,32767)):
            with self.assertRaises(ValueError):translation_maps(shape,np.eye(3))

    def test_coefficient_table_identity_and_finite(self):
        table=cubic_reference_table()
        self.assertEqual(table.shape,(32,32,16))
        self.assertEqual(table.dtype,np.float32)
        self.assertTrue(np.isfinite(table).all())
        expected=np.zeros(16,np.float32);expected[5]=1
        np.testing.assert_array_equal(table[0,0],expected)

    def test_device_ticket_lifetime_and_mask_binding(self):
        mask=np.ones((17,31),bool)
        owner=SimpleNamespace(handle=123,generation=1)
        ticket=CudaWarpFrame(owner,mask);ticket.validate(mask)
        self.assertEqual(ticket.shape,mask.shape);self.assertEqual(ticket.size,mask.size)
        with self.assertRaises(ValueError):ticket.validate(mask.copy())
        owner.generation+=1
        with self.assertRaises(ValueError):ticket.validate(mask)
        owner.generation=1;owner.handle=None
        with self.assertRaises(ValueError):ticket.validate(mask)
        owner.handle=123;ticket.consumed=True
        with self.assertRaises(ValueError):ticket.validate(mask)

    def test_invalid_inputs_are_rejected_before_cuda_access(self):
        warp=CudaCubicTranslation.__new__(CudaCubicTranslation)
        mask=np.ones((17,31),np.uint8)
        for image in (np.ones((17,31),np.uint16),np.ones((17,31),np.float64),np.full((17,31),np.nan,np.float32),np.ones((17,31,3),np.uint8)):
            with self.assertRaises(ValueError):warp(image,mask,np.eye(3))
        image=np.ones((17,31),np.uint8)
        with self.assertRaises(ValueError):warp(image,mask.astype(float),np.eye(3))
        with self.assertRaises(ValueError):warp(image,mask,np.eye(3),erosion_px=True)

    def test_failed_startup_conformance_closes_and_refuses_backend(self):
        from unittest.mock import patch,Mock
        warp=CudaCubicTranslation.__new__(CudaCubicTranslation);warp.close=Mock()
        with patch.object(CudaCubicTranslation,'__call__',return_value=(np.zeros((17,31),np.float32),np.ones((17,31),np.uint8))):
            with self.assertRaisesRegex(RuntimeError,'incompatible'):warp.verify_reference()
        warp.close.assert_called_once()

    def test_cross_library_ticket_rejected_before_foreign_pointer_use(self):
        owner=SimpleNamespace(handle=123,generation=1,lib=SimpleNamespace(_name='/first.so'))
        mask=np.ones((17,31),bool);ticket=CudaWarpFrame(owner,mask)
        with self.assertRaisesRegex(ValueError,'same compiled library'):
            ticket.prepare(SimpleNamespace(_name='/second.so'),456,mask,True,.25,np.empty(1,np.float32))

    def test_opt_in_never_changes_default_execution(self):
        self.assertEqual(VisibleConfig().stabilization_execution,'reference')
        with self.assertRaises(ValueError):VisibleConfig(stabilization_execution='cuda_cubic_resident')
        cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',
            cuda_median_library='/explicit/library.so',state_update_backend='cuda_resident',
            pixel_noise_enabled=True,pixel_noise_model='background_residual',motion_backend='pva')
        for mode in ('cuda_cubic_host','cuda_cubic_resident'):
            candidate=replace(cfg,stabilization_execution=mode)
            self.assertEqual(candidate.temporal_threshold_sigma,cfg.temporal_threshold_sigma)
            for change in (dict(motion_backend='cpu_translation'),dict(state_update_backend='inplace')):
                with self.assertRaises(ValueError):replace(candidate,**change)
