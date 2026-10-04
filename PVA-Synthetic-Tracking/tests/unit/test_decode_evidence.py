"""Overlap validation must not quietly allow a detector or binary change."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tiny_target.visible_baseline import VisibleConfig

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('decode_evidence_test', ROOT/'scripts/verify_phase20_decode_evidence.py')
module = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec.loader.exec_module(module)


class DecodeEvidenceTests(unittest.TestCase):
    def test_only_explicit_decode_change_allowed(self):
        before = asdict(VisibleConfig())
        after = dict(before, frame_decode_execution='prefetch_one')
        module.check_config(before, after)
        with self.assertRaisesRegex(ValueError, 'policy'):
            module.check_config(before, dict(after, spatial_threshold_sigma=2.9))

    def test_missing_transition_is_not_an_overlap_trial(self):
        cfg = asdict(VisibleConfig())
        with self.assertRaisesRegex(ValueError, 'explicit'):
            module.check_config(cfg, cfg)

    def test_native_binary_cannot_change_with_decode(self):
        before = asdict(VisibleConfig(spatial_background='median5', spatial_filter_backend='cuda_median5',
            cuda_median_library='/gpu.so', state_update_backend='cuda_resident', pixel_noise_enabled=True,
            pixel_noise_model='background_residual', shape_measurement_mode='mutual_half_height_r8',
            native_shape_library='/native.so', native_shape_library_sha256='a'*64))
        after = dict(before, frame_decode_execution='prefetch_one', native_shape_library_sha256='b'*64)
        with self.assertRaisesRegex(ValueError, 'native binary'):
            module.check_config(before, after)
