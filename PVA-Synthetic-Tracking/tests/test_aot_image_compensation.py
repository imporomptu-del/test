"""Generated-only cubic image operator checks; never open experiment imagery."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    spec = importlib.util.spec_from_file_location("generated_image_compensation",
        ROOT / "scripts/aot_image_compensation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_maps(shape):
    yy, xx = np.indices(shape, dtype=np.float32)
    return xx, yy


def cubic_weight(distance):
    """Independent Keys cubic polynomial, not OpenCV's coefficient table."""
    x, a = abs(float(distance)), -.75
    if x <= 1:
        return (a+2)*x**3-(a+3)*x**2+1
    if x < 2:
        return a*x**3-5*a*x**2+8*a*x-4*a
    return 0.


def scalar_quantized_pull(image, fixed_xy, coefficients, valid):
    output = np.full(valid.shape, np.nan, np.float64)
    for y, x in zip(*np.nonzero(valid)):
        bx, by = map(int, fixed_xy[y, x])
        phase = int(coefficients[y, x])
        fx, fy = (phase % 32)/32, (phase // 32)/32
        value = 0.
        for j in range(-1, 3):
            for i in range(-1, 3):
                value += (float(image[by+j, bx+i]) * cubic_weight(fx-i) * cubic_weight(fy-j))
        output[y, x] = value
    return output


def erode_false_border(mask, radius=2):
    height, width = mask.shape
    result = np.zeros_like(mask)
    for y in range(radius, height-radius):
        for x in range(radius, width-radius):
            result[y, x] = mask[y-radius:y+radius+1, x-radius:x+radius+1].all()
    return result


class ImageCompensationSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_cubic_matches_independent_scalar_for_signed_shifts_and_affine_maps(self):
        shape = (31, 37)
        yy, xx = np.indices(shape, dtype=np.float32)
        image = np.random.default_rng(20260928).integers(0, 256, shape, dtype=np.uint8)
        cases = [(xx, yy), (xx+2, yy-1), (xx-2, yy+1),
                 (xx+.25, yy-.5), (xx-.75, yy+.375),
                 (1.01*xx+.035*yy-.4, -.02*xx+.99*yy+.6)]
        for qx, qy in cases:
            with self.subTest(qx=float(qx[5, 5]), qy=float(qy[5, 5])):
                field = self.core.field_from_maps(qx, qy)
                actual = self.core.pull(image, field)
                expected = scalar_quantized_pull(image, field["fixed_xy"], field["coefficients"], field["valid"])
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual[field["valid"]], expected[field["valid"]], rtol=0, atol=1e-4)
                np.testing.assert_array_equal(np.isnan(actual), ~field["valid"])

    def test_identity_is_exact_only_on_conservatively_valid_full_footprint(self):
        image = np.arange(31*37, dtype=np.float32).reshape(31, 37) / 5
        xx, yy = identity_maps(image.shape)
        field = self.core.field_from_maps(xx, yy)
        result = self.core.pull(image, field)
        np.testing.assert_array_equal(result[field["valid"]], image[field["valid"]])
        expected_kernel = (xx >= 1) & (xx <= 34) & (yy >= 1) & (yy <= 28)
        np.testing.assert_array_equal(field["kernel_valid"], expected_kernel)
        np.testing.assert_array_equal(field["valid"], erode_false_border(expected_kernel))
        self.assertFalse(field["valid"][0, 0])
        self.assertTrue(np.isnan(result[0, 0]))

    def test_quantized_carry_defines_footprint_not_unquantized_floor(self):
        xx, yy = identity_maps((12, 16))
        xx[5, 5], yy[5, 5] = 8.9999, 6.9999
        xx[6, 5] = 13.9999
        field = self.core.field_from_maps(xx, yy, erosion_px=0)
        np.testing.assert_array_equal(field["fixed_xy"][5, 5], [9, 7])
        self.assertEqual(int(field["coefficients"][5, 5]), 0)
        self.assertEqual(int(field["fixed_xy"][6, 5, 0]), 14)
        self.assertFalse(field["kernel_valid"][6, 5], "Quantized x=14 has an out-of-image tap at16")

    def test_kernel_support_includes_zero_weight_taps_and_does_not_wrap(self):
        xx, yy = identity_maps((12, 16))
        field = self.core.field_from_maps(xx, yy, erosion_px=0)
        self.assertFalse(field["valid"][5, 0])
        self.assertFalse(field["valid"][5, 14])
        self.assertTrue(field["valid"][5, 1])
        self.assertTrue(field["valid"][5, 13])
        image = np.zeros((12, 16), np.float32)
        image[:, -1] = 255
        result = self.core.pull(image, field)
        self.assertTrue(np.isnan(result[5, 0]))
        self.assertEqual(float(result[5, 1]), 0.)

    def test_combined_support_erodes_once_in_full_image_with_false_border(self):
        xx, yy = identity_maps((31, 37))
        model = np.ones(xx.shape, bool)
        model[12, 15] = False
        numerical = np.ones(xx.shape, bool)
        numerical[18, 25] = False
        field = self.core.field_from_maps(xx, yy, model_support=model, numerical=numerical)
        np.testing.assert_array_equal(field["valid_pre"], model & numerical & field["kernel_valid"])
        np.testing.assert_array_equal(field["valid"], erode_false_border(field["valid_pre"]))
        self.assertFalse(field["valid"][10:15, 13:18].any())
        self.assertFalse(field["valid"][16:21, 23:28].any())

    def test_roi_keeps_full_image_erosion_and_pads_outside_native_as_invalid(self):
        xx, yy = identity_maps((31, 37))
        field = self.core.field_from_maps(xx, yy)
        roi = self.core.field_roi(field, (10, 10, 17, 17))
        np.testing.assert_array_equal(roi["valid"], field["valid"][10:17, 10:17])
        self.assertTrue(roi["valid"].all(), "ROI edge is not a new erosion boundary")
        outside = self.core.field_roi(field, (-3, -2, 8, 7))
        self.assertEqual(outside["valid"].shape, (9, 11))
        self.assertFalse(outside["valid"][:2].any())
        self.assertFalse(outside["valid"][:, :3].any())
        np.testing.assert_array_equal(outside["valid"][2:, 3:], field["valid"][:7, :8])

    def test_integer_source_crop_origin_preserves_same_native_cubic_samples(self):
        image = np.random.default_rng(7).integers(0, 256, (40, 50), dtype=np.uint8)
        xx, yy = identity_maps(image.shape)
        field = self.core.field_from_maps(xx+.375, yy-.25)
        roi = self.core.field_roi(field, (15, 15, 25, 25))
        whole = self.core.pull(image, roi)
        cropped = self.core.pull(image[10:31, 10:31], roi, origin_xy=(10, 10))
        np.testing.assert_array_equal(cropped, whole)

    def test_maps_masks_and_source_images_are_unchanged_by_pull(self):
        image = np.random.default_rng(8).integers(0, 256, (24, 24), dtype=np.uint8)
        xx, yy = identity_maps(image.shape)
        image.setflags(write=False)
        xcopy, ycopy, before = xx.copy(), yy.copy(), image.copy()
        field = self.core.field_from_maps(xx, yy)
        fields_before = {key: value.copy() for key, value in field.items() if isinstance(value, np.ndarray)}
        self.core.pull(image, field)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(xx, xcopy)
        np.testing.assert_array_equal(yy, ycopy)
        for key, value in fields_before.items():
            np.testing.assert_array_equal(field[key], value)

    def test_ramp_and_impulse_match_scalar_and_cubic_ringing_is_not_clamped(self):
        xx, yy = identity_maps((25,25))
        impulse = np.zeros(xx.shape,np.float32)
        impulse[12,12] = 255
        field = self.core.field_from_maps(xx+.5,yy+.25)
        for image in (xx*3+yy*2,impulse):
            actual = self.core.pull(image,field)
            expected = scalar_quantized_pull(image,field["fixed_xy"],field["coefficients"],field["valid"])
            np.testing.assert_allclose(actual[field["valid"]],expected[field["valid"]],atol=1e-4,rtol=0)
        self.assertLess(float(np.nanmin(actual)),0.,"Cubic negative lobes must not be clamped away")


class ImageCompensationPsfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_psf_is_native_coordinate_gaussian_with_euclidean_six_sigma_truncation(self):
        shape, origin, center, sigma = (13, 15), (100, 200), (106.25, 205.5), .6
        yy, xx = np.indices(shape, dtype=np.float64)
        d2 = (xx+origin[0]-center[0])**2+(yy+origin[1]-center[1])**2
        expected = np.where(d2 <= (6*sigma)**2, 16*np.exp(-d2/(2*sigma*sigma)), 0.)
        result = self.core.psf(shape, center, sigma, 16, origin_xy=origin)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(self.core.psf(shape, center, sigma, -16, origin_xy=origin), -result, rtol=0, atol=0)
        self.assertTrue((result[d2 > (6*sigma)**2] == 0).all())
        self.assertLess(float(result.max()), 16, "A fractional center need not attain the declared peak amplitude")

    def test_u8_injection_rints_then_clips_and_records_effective_signed_increment(self):
        for base, amplitude in ((250, 16), (5, -16), (128, 16), (128, -16)):
            image = np.full((17, 19), base, np.uint8)
            original = image.copy()
            center, origin = (108.25, 208.5), (100, 200)
            result = self.core.inject_patch(image, origin, center, .6, amplitude)
            intended = self.core.psf(image.shape, center, .6, amplitude, origin_xy=origin)
            expected = np.clip(np.rint(image.astype(float)+intended), 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(result["image"], expected)
            np.testing.assert_allclose(result["intended_delta"], intended, rtol=0, atol=0)
            np.testing.assert_array_equal(result["effective_delta"], expected.astype(np.float32)-image.astype(np.float32))
            self.assertEqual(result["effective_delta"].dtype, np.float32)
            np.testing.assert_array_equal(image, original)
            if amplitude < 0:
                self.assertLess(float(result["effective_delta"].min()), 0., "Dark injection must not wrap U8 subtraction")

    def test_subpixel_rendering_uses_requested_center_not_rounded_anchor(self):
        image = np.full((21, 21), 128, np.uint8)
        first = self.core.inject_patch(image, (0, 0), (10., 10.), .6, 16)
        second = self.core.inject_patch(image, (0, 0), (10.5, 10.5), .6, 16)
        self.assertFalse(np.array_equal(first["intended_delta"], second["intended_delta"]))
        self.assertFalse(np.array_equal(first["effective_delta"], second["effective_delta"]))
        self.assertEqual(float(first["intended_delta"][10, 10]), 16.)
        self.assertLess(float(second["intended_delta"].max()), 16.)


def generated_saved_row(matrix=None):
    matrix = np.eye(3).tolist() if matrix is None else matrix
    return dict(case_id="generated", cells=[dict(cell_id=i, arms={arm:dict(training_eligible=True,
        hull=[[0,0],[79,0],[79,59],[0,59]], fit=dict(valid=True,native_matrix=copy.deepcopy(matrix)))
        for arm in ("global_translation", "local_translation", "local_affine")}) for i in range(48)])


class ImageCompensationFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_saved_forward_affine_map_is_not_inverted_and_stripes_are_identical(self):
        matrix = [[1.01,.02,.25],[-.015,.99,-.5],[0,0,1]]
        row = generated_saved_row(matrix)
        before = copy.deepcopy(row)
        first = self.core.build_field(row, "local_affine", shape=(60,80), stripe_rows=1)
        second = self.core.build_field(row, "local_affine", shape=(60,80), stripe_rows=17)
        yy, xx = np.indices((60,80),dtype=float)
        np.testing.assert_array_equal(first["qx"], (1.01*xx+.02*yy+.25).astype(np.float32))
        np.testing.assert_array_equal(first["qy"], (-.015*xx+.99*yy-.5).astype(np.float32))
        self.assertEqual(self.core.field_hashes(first), self.core.field_hashes(second))
        self.assertEqual(row,before)

    def test_each_cell_uses_own_saved_matrix_without_blending(self):
        row = generated_saved_row()
        for cell in row["cells"]:
            for fit in cell["arms"].values():
                fit["fit"]["native_matrix"][0][2] = .5*(cell["cell_id"]%8)
        field = self.core.build_field(row,"global_translation",shape=(60,80))
        self.assertEqual(float(field["qx"][25,39]),40.5)
        self.assertEqual(float(field["qx"][25,40]),42.)
        self.assertEqual(float(field["qx"][25,40]-field["qx"][25,39]-1),.5)

    def test_numerical_fit_gate_and_pointwise_hull_are_distinct_without_fallback(self):
        row = generated_saved_row()
        for cell in row["cells"]:
            cell["arms"]["local_affine"]["hull"] = [[0,0],[45.5,0],[45.5,59],[0,59]]
        row["cells"][0]["arms"]["local_affine"]["fit"] = None
        row["cells"][27]["arms"]["local_affine"]["training_eligible"] = False
        field = self.core.build_field(row,"local_affine",shape=(60,80))
        self.assertFalse(field["numerical"][:10,:10].any())
        self.assertTrue(field["numerical"][30:40,30:40].all())
        self.assertFalse(field["model_support"][30:40,30:40].any())
        self.assertTrue(field["model_support"][20,45])
        self.assertFalse(field["model_support"][20,46])
        image = np.full((60,80),128,np.uint8)
        result = self.core.pull(image,field)
        np.testing.assert_array_equal(np.isnan(result),~field["valid"])
        bad = copy.deepcopy(row)
        bad["cells"].reverse()
        with self.assertRaises(ValueError): self.core.build_field(bad,"local_affine",shape=(60,80))

    def test_source_validity_rejects_every_tap_including_zero_weights(self):
        image = np.full((25,25),128,np.uint8)
        xx,yy = identity_maps(image.shape)
        field = self.core.field_from_maps(xx,yy,erosion_px=0)
        source_valid = np.ones(image.shape,bool)
        source_valid[10,10] = False
        output = self.core.pull(image,field,source_valid=source_valid)
        expected = field["valid"].copy()
        expected[8:12,8:12] = False
        np.testing.assert_array_equal(np.isnan(output),~expected)
        self.assertEqual(float(output[12,12]),128.)
        with self.assertRaises(ValueError):
            self.core.pull(image[5:15,5:15],field,origin_xy=(5,5))


class ImageCompensationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_signed_lobes_l2_template_units_and_polarity_centroid(self):
        values = np.array([[3.,-4.],[np.nan,0.]])
        template = np.array([[1.,-1.],[np.nan,0.]])
        positive = self.core.array_metrics(values,origin_xy=(10,20),template=template)
        negative = self.core.array_metrics(values,origin_xy=(10,20),polarity=-1,template=template)
        self.assertEqual((positive["count"],positive["positive_mass_dn"],positive["negative_mass_dn"]),(3,3.,4.))
        self.assertEqual((positive["l1_dn"],positive["l2_dn"],positive["signed_mass_dn"]),(7.,5.,-1.))
        self.assertAlmostEqual(positive["rms_dn"],5/np.sqrt(3))
        self.assertAlmostEqual(positive["template_response_dn"],7/np.sqrt(2))
        self.assertEqual(positive["template_gain"],3.5)
        self.assertEqual(positive["centroid_xy"],[10.,20.])
        self.assertEqual(negative["centroid_xy"],[11.,20.])

    def test_identity_static_target_cancels_residual_without_false_erasure_or_ratios(self):
        image = np.full((96,96),128,np.uint8)
        field = self.core.field_from_maps(*identity_maps(image.shape))
        for amplitude in (-16,16):
            result = self.core.probe(image,image,field,(48.25,48.25),(48.25,48.25),.6,amplitude,return_arrays=True)
            self.assertFalse(result["unavailable"])
            self.assertEqual(result["metrics"]["delta"]["l1_dn"],0.)
            self.assertIsNone(result["metrics"]["delta"]["centroid_xy"])
            self.assertIsNone(result["metrics"]["delta"]["template_response_dn"])
            self.assertIsNone(result["oracle_absolute_mass_coverage"])
            self.assertGreater(result["metrics"]["current_target_delta"]["l1_dn"],0.)
            self.assertIsNone(result["metrics"]["visibility"]["isolated_response_over_clean_rms"])
            self.assertIsNone(result["metrics"]["visibility"]["detector_pass"])
            self.assertFalse(result["metrics"]["visibility"]["calibrated_snr"])
            np.testing.assert_array_equal(result["arrays"]["delta"],np.zeros((65,65)))

    def test_undefined_or_unsupported_zero_oracle_never_becomes_full_support_pass(self):
        image = np.full((96,96),128,np.uint8)
        xx,yy = identity_maps(image.shape)
        for numerical in (np.ones(image.shape,bool),np.zeros(image.shape,bool)):
            field = self.core.field_from_maps(xx,yy,numerical=numerical,model_support=np.zeros(image.shape,bool))
            result = self.core.probe(image,image,field,(48.25,48.25),(48.25,48.25),.6,16,return_arrays=True)
            self.assertTrue(result["unavailable"])
            self.assertFalse(result["full_support_eligible"])
            self.assertEqual(result["metrics"]["delta"]["count"],0)
            self.assertIsNone(result["metrics"]["delta"]["peak_abs_dn"])
            self.assertTrue(np.isnan(result["arrays"]["delta"]).all())

    def test_paired_residual_difference_matches_effective_injection_and_analytic_oracle(self):
        image = np.random.default_rng(19).integers(16,240,(96,96),dtype=np.uint8)
        xx,yy = identity_maps(image.shape)
        field = self.core.field_from_maps(xx+.375,yy-.25)
        p0,c0,sigma,amp = (48.25,48.25),(49.125,48.),.6,-16
        before = image.copy()
        hashes = self.core.field_hashes(field)
        result = self.core.probe(image,image,field,p0,c0,sigma,amp,return_arrays=True)
        arrays = result["arrays"]
        yy0,xx0 = np.indices((65,65),dtype=float)
        px,py = xx0+16,yy0+16
        def independent(x,y,center):
            d=(x-center[0])**2+(y-center[1])**2
            return np.where(d<=(6*sigma)**2,amp*np.exp(-d/(2*sigma**2)),0.)
        expected = independent(px+.375,py-.25,c0)-independent(px,py,p0)
        np.testing.assert_allclose(arrays["oracle"],expected,atol=1e-10,rtol=0)
        np.testing.assert_allclose(arrays["delta"],arrays["injected_residual"]-arrays["clean_residual"],atol=0,rtol=0)
        previous_injection = self.core.inject_patch(image,(0,0),p0,sigma,amp)
        current_injection = self.core.inject_patch(image,(0,0),c0,sigma,amp)
        local = self.core.field_roi(field,(16,16,81,81))
        linear = self.core.pull(current_injection["effective_delta"],local)-previous_injection["effective_delta"][16:81,16:81]
        np.testing.assert_allclose(arrays["delta"],linear,rtol=0,atol=1e-4)
        self.assertEqual(self.core.field_hashes(field),hashes)
        np.testing.assert_array_equal(image,before)

    def test_border_previous_injection_statistics_exclude_artificial_padding(self):
        image = np.full((40,40),128,np.uint8)
        field = self.core.field_from_maps(*identity_maps(image.shape))
        for amplitude in (-16,16):
            result = self.core.probe(image,image,field,(1.25,1.25),(1.75,1.25),1.2,amplitude)
            direct = self.core.inject_patch(image,(0,0),(1.25,1.25),1.2,amplitude)
            previous = result["previous_injection"]
            self.assertEqual(previous["clipped_pixel_count"],0)
            self.assertEqual(previous["effective"]["l1_dn"],direct["stats"]["effective"]["l1_dn"])
            self.assertAlmostEqual(previous["native_intended_l1_dn"],direct["stats"]["native_intended_l1_dn"],places=10)
            self.assertTrue(result["source_psf_partial"])
            self.assertTrue(result["partial"])
            self.assertFalse(result["full_support_eligible"])

    def test_target_footprint_mask_loss_is_reported_not_renormalized_as_preserved(self):
        image = np.full((96,96),128,np.uint8)
        xx,yy = identity_maps(image.shape)
        support = np.ones(image.shape,bool)
        support[:,49:] = False
        field = self.core.field_from_maps(xx,yy,model_support=support)
        result = self.core.probe(image,image,field,(48.25,48.25),(48.75,48.25),1.2,16)
        self.assertTrue(result["partial"])
        self.assertFalse(result["unavailable"])
        self.assertFalse(result["full_support_eligible"])
        self.assertLess(result["significant_oracle_valid_pixels"],result["significant_oracle_pixels"])
        self.assertLess(result["oracle_absolute_mass_coverage"],1.)
        self.assertGreater(result["valid_pixels"],0)

    def test_zero_residual_uses_nonempty_union_of_target_psfs_for_support(self):
        image = np.full((96,96),128,np.uint8)
        xx,yy = identity_maps(image.shape)
        support = np.ones(image.shape,bool)
        support[40:56,40:56] = False
        field = self.core.field_from_maps(xx,yy,model_support=support)
        result = self.core.probe(image,image,field,(48.25,48.25),(48.25,48.25),.6,16)
        self.assertFalse(result["unavailable"])
        self.assertEqual(result["significant_oracle_pixels"],0)
        self.assertGreater(result["significant_target_union_pixels"],0)
        self.assertEqual(result["significant_target_union_valid_pixels"],0)
        self.assertFalse(result["full_target_footprint_support"])
        self.assertFalse(result["full_support_eligible"])

    def test_saved_affine_continuous_oracle_uses_float64_not_remap_map_rounding(self):
        matrix = np.array([[1.001234567,.002345678,.123456789],[-.00123456,.998765432,-.234567891],[0,0,1]])
        row = generated_saved_row(matrix.tolist())
        for cell in row["cells"]:
            cell["arms"]["local_affine"]["hull"] = [[0,0],[127,0],[127,127],[0,127]]
        image = np.full((128,128),128,np.uint8)
        field = self.core.build_field(row,"local_affine",shape=image.shape)
        p0 = np.array([64.25,64.25])
        c0 = (matrix@[*p0,1])[:2]+[.5,0]
        result = self.core.probe(image,image,field,p0,c0,.6,16,return_arrays=True)
        yy,xx = np.indices((65,65),dtype=np.float64)
        px,py = xx+32,yy+32
        qx = matrix[0,0]*px+matrix[0,1]*py+matrix[0,2]
        qy = matrix[1,0]*px+matrix[1,1]*py+matrix[1,2]
        def gaussian(x,y,c):
            radius=(x-c[0])**2+(y-c[1])**2
            return np.where(radius<3.6**2,16*np.exp(-radius/(2*.6**2)),0)
        expected = gaussian(qx,qy,c0)-gaussian(px,py,p0)
        np.testing.assert_allclose(result["arrays"]["oracle"],expected,atol=1e-10,rtol=0)
        rounded = gaussian(qx.astype(np.float32),qy.astype(np.float32),c0)-gaussian(px,py,p0)
        self.assertGreater(float(np.max(abs(rounded-expected))),1e-7)

    def test_current_target_outside_or_touching_roi_remains_truncated_not_full_preservation(self):
        image=np.full((128,128),128,np.uint8)
        field=self.core.field_from_maps(*identity_maps(image.shape))
        for center in ((96.,64.25),(110.,64.25)):
            result=self.core.probe(image,image,field,(64.25,64.25),center,.6,16)
            self.assertTrue(result["full_roi_support"])
            self.assertTrue(result["oracle_roi_truncated"])
            self.assertFalse(result["full_support_eligible"])
            self.assertTrue(result["previous_oracle_present"])
        self.assertFalse(result["current_oracle_present"])

    def test_empty_current_source_context_retains_null_injection_schema(self):
        image=np.full((96,96),128,np.uint8)
        xx,yy=identity_maps(image.shape)
        field=self.core.field_from_maps(xx,yy,numerical=np.zeros(image.shape,bool))
        result=self.core.probe(image,image,field,(48.25,48.25),(1000.,1000.),.6,16)
        self.assertTrue(result["unavailable"])
        self.assertFalse(result["full_support_eligible"])
        self.assertTrue(result["current_injection"]["empty_native_context"])
        self.assertEqual(result["current_injection"]["effective"]["count"],0)
        self.assertIsNone(result["current_injection"]["effective"]["l1_dn"])
        self.assertGreater(result["current_injection"]["excluded_intended_l1_dn"],0)


if __name__ == "__main__":
    unittest.main()
