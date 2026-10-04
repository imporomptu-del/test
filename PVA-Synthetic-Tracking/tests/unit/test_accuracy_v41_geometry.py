import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v41_geometry import source_templates


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = (np.arange(80*90).reshape(80, 90) % 200 + 20).astype(np.uint8)
        self.eye = np.eye(3)

    def call(self, **kwargs):
        args = dict(current=self.image, prior=self.image, current_origin=[0, 0], prior_origin=[0, 0],
                    current_to_reference=self.eye, prior_to_reference=self.eye,
                    current_actual_xy=[40, 40], prior_actual_xy=[40, 40])
        args.update(kwargs)
        return source_templates(**args)

    def test_identity_and_native_center(self):
        r = self.call(current_actual_xy=[40.5, 40.49], prior_actual_xy=[40.5, 40.49])
        np.testing.assert_equal(r['current'], self.image[28:53, 29:54])
        np.testing.assert_equal(r['current'], r['stationary_prior'])
        np.testing.assert_equal(r['current'], r['transported_prior'])

    def test_camera_translation_not_object_motion(self):
        hc = self.eye.copy(); hc[:2, 2] = [-3, 2]
        r = self.call(current_to_reference=hc, prior_actual_xy=[37, 42])
        np.testing.assert_equal(r['stationary_prior'], self.image[30:55, 25:50])
        np.testing.assert_equal(r['stationary_prior'], r['transported_prior'])
        self.assertEqual(r['geometry']['prior_transport_offset_xy'], [0, 0])

    def test_transform_order_and_origins(self):
        hp = self.eye.copy(); hp[:2, 2] = [100, 200]
        hc = self.eye.copy(); hc[:2, 2] = [103, 198]
        r = self.call(current_origin=[1000, 2000], prior_origin=[1000, 2000],
                      current_actual_xy=[1040, 2040], prior_actual_xy=[1043, 2038],
                      current_to_reference=hc, prior_to_reference=hp)
        np.testing.assert_equal(r['stationary_prior'], self.image[26:51, 31:56])
        np.testing.assert_equal(r['stationary_prior'], r['transported_prior'])

    def test_residual_shift_and_sensitivity_direction(self):
        r = self.call(prior_actual_xy=[36, 41], shift=(1, -1))
        np.testing.assert_equal(r['stationary_prior'], self.image[27:52, 29:54])
        np.testing.assert_equal(r['transported_prior'], self.image[28:53, 25:50])
        self.assertEqual(r['geometry']['prior_transport_offset_xy'], [-4, 1])

    def test_border_is_unknown_not_padding(self):
        r = self.call(current_actual_xy=[1, 2], prior_actual_xy=[1, 2])
        self.assertTrue(np.isnan(r['current'][:10]).all())
        self.assertTrue(np.isfinite(r['current'][10:, 11:]).all())

    def test_interpolation_uses_fractional_transform(self):
        hc = self.eye.copy(); hc[:2, 2] = [0.25, 0.5]
        r = self.call(current_to_reference=hc, prior_actual_xy=[40.25, 40.5])
        expected = (self.image[28:53, 28:53].astype(float)*.375 + self.image[28:53, 29:54]*.125
                    + self.image[29:54, 28:53]*.375 + self.image[29:54, 29:54]*.125)
        np.testing.assert_allclose(r['stationary_prior'], expected)

    def test_invalid_matrices_shifts_and_pixels(self):
        for kwargs in [dict(prior_to_reference=np.zeros((3, 3))), dict(current_to_reference=np.eye(2)),
                       dict(shift=(2, 0)), dict(current=self.image.astype(float)),
                       dict(current_actual_xy=[True, 3]), dict(prior_actual_xy=[np.nan, 0])]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): self.call(**kwargs)

    def test_no_mutation(self):
        original = self.image.copy(); transform = self.eye.copy()
        self.call(prior_actual_xy=[44, 43])
        np.testing.assert_equal(self.image, original)
        np.testing.assert_equal(self.eye, transform)

    def test_saturated_positive_weight_corner_is_not_smoothed_into_support(self):
        self.image[40, 41] = 255
        hc = self.eye.copy(); hc[0, 2] = .25
        r = self.call(current_to_reference=hc, prior_actual_xy=[40.25, 40])
        self.assertTrue(np.isnan(r['stationary_prior'][12, 12]))
        self.assertTrue(np.isfinite(r['current'][12, 12]))

    def test_zero_weight_saturated_neighbor_does_not_remove_exact_pixel(self):
        self.image[40, 41] = 255
        r = self.call()
        self.assertTrue(np.isfinite(r['stationary_prior'][12, 12]))
        self.assertTrue(np.isnan(r['stationary_prior'][12, 13]))


if __name__ == '__main__':
    unittest.main()
