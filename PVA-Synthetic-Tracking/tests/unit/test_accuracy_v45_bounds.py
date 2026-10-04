"""Positive-box bounds: exact small-box extrema and correlated input probes."""

import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v42_localized as legacy
import accuracy_v43_bounds as old_bounds
from accuracy_v43_components import prepare_components
import accuracy_v45_bounds as bounds


def fixture(polarity="bright", fixed_light=False):
    y,x=np.indices((129,129));background=50+20*(x>=64)+8*np.sin(y/12)
    centers=[[29+4*i,64] for i in range(8)]
    def point(cx,cy,amplitude=30):
        return amplitude*np.exp(-((x-cx)**2+(y-cy)**2)/2)
    sign=1 if polarity=="bright" else -1
    history=np.stack([background+sign*point(*center) for center in centers])
    if fixed_light:
        history+=sign*point(78,48,25)
    return history,centers,[0,0],polarity


class NormalizationIntervalTests(unittest.TestCase):
    def test_two_coordinate_exact_formula(self):
        lo,hi,metadata=bounds.normalization_interval([1.,2.],[3.,4.])
        self.assertTrue(metadata["available"])
        np.testing.assert_allclose(lo,[1/np.sqrt(17),2/np.sqrt(13)],rtol=1e-14)
        np.testing.assert_allclose(hi,[3/np.sqrt(13),4/np.sqrt(17)],rtol=1e-14)

    def test_all_corner_extrema_enclosed_without_joint_attainability_claim(self):
        lower=np.array([.1,.3,1.,2.]);upper=np.array([2.,.5,4.,3.])
        lo,hi,metadata=bounds.normalization_interval(lower,upper)
        corners=np.array([np.where(bits,upper,lower) for bits in itertools.product((0,1),repeat=4)])
        normalized=corners/np.linalg.norm(corners,axis=1)[:,None]
        self.assertTrue(np.all(normalized>=lo-1e-15))
        self.assertTrue(np.all(normalized<=hi+1e-15))
        np.testing.assert_allclose(lo,normalized.min(axis=0),atol=1e-15)
        np.testing.assert_allclose(hi,normalized.max(axis=0),atol=1e-15)
        self.assertFalse(metadata["joint_attainability_claimed"])

    def test_random_and_fully_correlated_box_points_are_enclosed(self):
        rng=np.random.default_rng(4501)
        for count in (2,9,289):
            lower=rng.uniform(0,.8,count);upper=lower+rng.uniform(.01,1,count)
            lo,hi,metadata=bounds.normalization_interval(lower,upper)
            self.assertTrue(metadata["available"])
            values=[lower,upper,(lower+upper)/2]
            values.extend(lower+rng.uniform(0,1)*(upper-lower) for _ in range(20))
            values.extend(rng.uniform(lower,upper) for _ in range(20))
            for value in values:
                actual=value/np.linalg.norm(value)
                self.assertTrue(np.all(actual>=lo-1e-14))
                self.assertTrue(np.all(actual<=hi+1e-14))

    def test_one_nonzero_pixel_and_exact_zero_hann_positions(self):
        lo,hi,metadata=bounds.normalization_interval([0.,.1,0.],[0.,10.,0.])
        self.assertTrue(metadata["available"])
        np.testing.assert_array_equal(lo,[0.,1.,0.]);np.testing.assert_array_equal(hi,[0.,1.,0.])

    def test_nan_support_is_preserved_and_not_filled(self):
        lower=np.array([np.nan,.3,0.]);upper=np.array([np.nan,.8,0.])
        lo,hi,metadata=bounds.normalization_interval(lower,upper)
        self.assertTrue(metadata["available"])
        np.testing.assert_array_equal(np.isnan(lo),np.isnan(lower))
        np.testing.assert_array_equal(np.isnan(hi),np.isnan(upper))
        self.assertEqual(lo[2],0);self.assertEqual(hi[2],0)

    def test_collapsed_or_nearly_zero_lower_norm_is_unknown(self):
        for lower in ([0.,0.],[1e-7,1e-7],[1e-6,0.]):
            lo,hi,metadata=bounds.normalization_interval(lower,[1.,1.])
            self.assertIsNone(lo);self.assertIsNone(hi);self.assertFalse(metadata["available"])
            self.assertIn("lower_norm",metadata["reason"])

    def test_dominant_coordinate_avoids_subtraction_cancellation(self):
        lo,hi,metadata=bounds.normalization_interval([1e150,1.],[1e150,2.])
        self.assertTrue(metadata["available"])
        self.assertGreater(lo[1],0)
        self.assertAlmostEqual(lo[1]/1e-150,1.,places=12)
        self.assertAlmostEqual(hi[1]/1e-150,2.,places=12)

    def test_extreme_finite_norm_overflow_is_json_safe_unknown(self):
        lo,hi,metadata=bounds.normalization_interval(np.full(9,1e308),np.full(9,1e308))
        self.assertIsNone(lo);self.assertIsNone(hi);json.dumps(metadata,allow_nan=False)

    def test_malformed_boxes_rejected(self):
        cases=[([-1.],[1.]),([2.],[1.]),([np.nan],[1.]),([1.],[np.inf]),([True],[True]),([1.,2.],[3.])]
        for lower,upper in cases:
            with self.subTest(lower=lower,upper=upper),self.assertRaises(ValueError):
                bounds.normalization_interval(lower,upper)

    def test_outward_accumulation_encloses_high_precision_endpoint_formula(self):
        rng=np.random.default_rng(450017)
        with localcontext() as context:
            context.prec=100
            for count in (2,3,17,289):
                lower=10**rng.uniform(-8,8,count)
                upper=lower*(1+rng.uniform(0,1,count))
                lo,hi,metadata=bounds.normalization_interval(lower,upper)
                self.assertTrue(metadata["available"])
                dl=[Decimal.from_float(float(v)) for v in lower]
                du=[Decimal.from_float(float(v)) for v in upper]
                sum_lower=sum(v*v for v in dl);sum_upper=sum(v*v for v in du)
                for index in range(count):
                    exact_lower=dl[index]/(dl[index]**2+sum_upper-du[index]**2).sqrt()
                    exact_upper=du[index]/(du[index]**2+sum_lower-dl[index]**2).sqrt()
                    self.assertLessEqual(Decimal.from_float(float(lo[index])),exact_lower)
                    self.assertGreaterEqual(Decimal.from_float(float(hi[index])),exact_upper)


class ComponentBoxTests(unittest.TestCase):
    def test_nominal_components_unchanged_and_bounds_no_wider_than_v43(self):
        for polarity in ("bright","dark"):
            args=fixture(polarity,fixed_light=True)
            components=prepare_components(*args)
            original={name:components[name].copy() for name in ("background","moving_template","fixed_templates")}
            old=old_bounds.component_bounds(*args,components,protected=True)
            new=bounds.component_bounds(*args,components,protected=True)
            self.assertTrue(old["available"] and new["available"])
            for key in ("background_bound129","moving_template_bound129","fixed_template_bounds"):
                np.testing.assert_array_equal(np.isfinite(new[key]),np.isfinite(old[key]))
                finite=np.isfinite(old[key]);self.assertTrue(np.all(new[key][finite]<=old[key][finite]+1e-12))
            for key,value in original.items():np.testing.assert_array_equal(components[key],value)
            self.assertTrue(new["metadata"]["nominal_components_unchanged"])
            self.assertFalse(new["metadata"]["joint_interval_attainability_claimed"])
            json.dumps(new["metadata"],allow_nan=False)

    def test_used_stamp_with_collapsed_lower_norm_is_not_dropped(self):
        args=fixture();components=prepare_components(*args)
        entry=old_bounds._protected_history(components["template_history"],components["metadata"])["moving"]
        entry=dict(entry,raw_weighted_norms=np.asarray(entry["raw_weighted_norms"]).copy())
        entry["raw_weighted_norms"][0]=1e-5
        lo,hi,metadata=bounds._combined_interval(entry,components["moving_stamp"],1.)
        self.assertIsNone(lo);self.assertIsNone(hi)
        self.assertFalse(metadata["uncertain_used_stamp_omitted"])
        self.assertEqual(metadata["used_history_indices"],entry["history_indices"])

    def test_correlated_raw_history_perturbations_with_frozen_used_membership(self):
        args=fixture(fixed_light=True);history,centers,offset,polarity=args
        components=prepare_components(*args)
        bound=bounds.component_bounds(*args,components,protected=True)
        self.assertTrue(bound["available"])
        self.assertGreater(len(components["fixed_templates"]),0)
        record=old_bounds._protected_history(components["template_history"],components["metadata"])
        y,x=np.indices((129,129));rng=np.random.default_rng(4502)
        noises=[np.full(history.shape,.5),np.full(history.shape,-.5),
                np.broadcast_to(.5*((x+y)%2*2-1),history.shape),
                np.broadcast_to(.5*(np.arange(8)%2*2-1)[:,None,None],history.shape)]
        noises.extend(rng.uniform(-.5,.5,history.shape) for _ in range(8))
        for noise in noises:
            perturbed=history+noise;masked=perturbed.copy()
            for index,center in enumerate(centers):
                masked[index,np.maximum(abs(x-center[0]),abs(y-center[1]))<=8]=np.nan
            background,_=legacy._median(masked,3)
            np.testing.assert_array_less(abs(background-components["background"]),bound["background_bound129"]+1e-12)
            moving_stamps=[legacy._positive_unit_stamp(legacy._stamp(perturbed[i]-background,centers[i]))
                           for i in record["moving"]["history_indices"]]
            self.assertTrue(all(stamp is not None for stamp in moving_stamps))
            moving=legacy._place(legacy._combine_stamps(moving_stamps),64.+np.asarray(offset))
            self.assertTrue(np.all(abs(moving-components["moving_template"])<=bound["moving_template_bound129"]+1e-12))
            highpass=np.stack([legacy._box_highpass(frame) for frame in perturbed])
            for j,entry in enumerate(record["fixed"]):
                fixed_stamps=[legacy._positive_unit_stamp(legacy._stamp(highpass[i],entry["centre_xy"]))
                              for i in entry["history_indices"]]
                self.assertTrue(all(stamp is not None for stamp in fixed_stamps))
                fixed=legacy._place(legacy._combine_stamps(fixed_stamps),entry["centre_xy"])
                self.assertTrue(np.all(abs(fixed-components["fixed_templates"][j])<=bound["fixed_template_bounds"][j]+1e-12))

    def test_missing_pixels_keep_the_original_component_support(self):
        history,centers,offset,polarity=fixture();history[:,64,66]=np.nan
        components=prepare_components(history,centers,offset,polarity)
        result=bounds.component_bounds(history,centers,offset,polarity,components,protected=True)
        self.assertTrue(result["available"])
        np.testing.assert_array_equal(np.isfinite(result["moving_template_bound129"]),np.isfinite(components["moving_template"]))
        self.assertTrue(np.isnan(result["background_bound129"][64,66]))

    def test_legacy_provenance_and_subpixel_placement_remain_compatible(self):
        history,centers,_,polarity=fixture();offset=[.3,-.2]
        components=legacy.prepare_components(history,centers,offset,polarity)
        result=bounds.component_bounds(history,centers,offset,polarity,components,protected=False)
        self.assertTrue(result["available"])
        self.assertFalse(result["metadata"]["protected_fixed_components"])
        np.testing.assert_array_equal(np.isfinite(result["moving_template_bound129"]),np.isfinite(components["moving_template"]))
        self.assertTrue(np.all(result["moving_template_bound129"]>=0))


if __name__=="__main__":unittest.main()
