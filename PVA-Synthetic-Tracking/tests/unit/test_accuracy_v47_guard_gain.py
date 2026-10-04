from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v47_guard_gain import estimate_guard_gain, intersect_gain_halfspaces, outward_float_interval


class GuardGainTests(unittest.TestCase):
    def fixture(self, gain=1.25):
        y, x = np.indices((129, 129))
        background = (50+20*((x//8+y//8) % 2)).astype(float)
        plane = 10+(x-64)/8-(y-64)/16
        return dict(current129=gain*background+plane,
                    history129=np.stack([background.copy() for _ in range(8)]),
                    background129=background, background_bound129=np.full((129, 129), .5),
                    prior_centers_xy=[[64.,64.]]*8, response_bound=.5)

    def test_exact_positive_negative_and_zero_halfspaces(self):
        result = intersect_gain_halfspaces([(2, ">=", 1), (-3, ">=", -6), (0, "<=", 2)])
        self.assertTrue(result["feasible"])
        self.assertEqual(result["lower"], Fraction(1,2))
        self.assertEqual(result["upper"], Fraction(2))
        result = intersect_gain_halfspaces([(-2, "<=", -3), (4, "<=", 8)])
        self.assertEqual((result["lower"], result["upper"]), (Fraction(3,2), Fraction(2)))
        self.assertFalse(intersect_gain_halfspaces([(0, ">=", 1)])["feasible"])
        self.assertFalse(intersect_gain_halfspaces([(0, "<=", -1)])["feasible"])
        self.assertFalse(intersect_gain_halfspaces([(1, "<=", -1)])["feasible"])
        self.assertFalse(intersect_gain_halfspaces([(1, ">=", 2),(1, "<=", 1)])["feasible"])
        self.assertIsNone(intersect_gain_halfspaces([])["upper"])

    def test_binary_float_fraction_is_not_decimal_rational_assumption(self):
        result = intersect_gain_halfspaces([(1., ">=", .1), (1., "<=", .1)])
        self.assertEqual(result["lower"], Fraction.from_float(.1))
        self.assertNotEqual(result["lower"], Fraction(1,10))
        extended = np.nextafter(np.longdouble(1),np.longdouble(2))
        exact = Fraction(*extended.as_integer_ratio())
        self.assertEqual(intersect_gain_halfspaces([(1,">=",extended)])["lower"],exact)
        for bad in (math.inf, math.nan, True, "1"):
            with self.assertRaises(ValueError): intersect_gain_halfspaces([(bad, ">=", 0)])
        with self.assertRaises(ValueError): intersect_gain_halfspaces([(1, "=", 1)])

    def test_outward_float_conversion_contains_exact_rational_endpoints(self):
        for lower, upper in ((Fraction(1,10), Fraction(1,3)), (Fraction(1,10**400), Fraction(2,10**400)),
                             (Fraction(0), Fraction(0)), (Fraction(2**200), Fraction(2**200+1))):
            floats = outward_float_interval(lower, upper)
            self.assertLessEqual(Fraction.from_float(floats[0]), lower)
            self.assertGreaterEqual(Fraction.from_float(floats[1]), upper)
        self.assertIsNone(outward_float_interval(Fraction(10**400), Fraction(2*10**400)))
        with self.assertRaises(ValueError): outward_float_interval(2,1)

    def test_known_shared_gain_is_inside_nonempty_outer_interval(self):
        for gain in (0., .5, 1., 1.25, 2.):
            result = estimate_guard_gain(**self.fixture(gain))
            self.assertTrue(result["available"], result["reasons"])
            self.assertLessEqual(result["gain_interval"][0], gain)
            self.assertGreaterEqual(result["gain_interval"][1], gain)
            self.assertEqual(result["candidate_count"], 144)
            self.assertEqual(result["eligible_count"], 144)
            self.assertGreater(result["stencil_count"], 0)
            self.assertEqual(result["motion_status"], "unknown")
            self.assertEqual(result["physical_class"], "unknown")
            json.dumps(result, allow_nan=False)

    def test_exact_zero_error_gain_and_large_affine_offset(self):
        args = self.fixture()
        args["background_bound129"] = np.zeros((129,129)); args["response_bound"] = 0.
        result = estimate_guard_gain(**args)
        self.assertEqual(result["gain_interval"], [1.25,1.25])
        self.assertEqual(result["exact_gain_interval"], ["5/4","5/4"])
        y,x = np.indices((129,129))
        args["current129"] = args["current129"]+2.**40+2.**20*x-2.**18*y
        changed = estimate_guard_gain(**args)
        self.assertEqual(changed["gain_interval"], result["gain_interval"])
        self.assertEqual(changed["contrast_constraints"], result["contrast_constraints"])

    def test_contrasts_and_inequalities_independently_reconstruct(self):
        args = self.fixture(); result = estimate_guard_gain(**args)
        lower,upper = Fraction(0), None
        for stencil, contrast in zip(result["stencils"],result["contrast_constraints"]):
            points = stencil["pixels_xy"]
            weights = stencil["weights"]
            self.assertEqual(weights,[1,-2,1])
            for axis in (0,1): self.assertEqual(sum(w*p[axis] for w,p in zip(weights,points)),0)
            self.assertEqual(sum(weights),0)
            yc=sum(Fraction.from_float(float(args["current129"][y,x]))*w for w,(x,y) in zip(weights,points))
            bc=sum(Fraction.from_float(float(args["background129"][y,x]))*w for w,(x,y) in zip(weights,points))
            ey=sum(abs(w)*Fraction.from_float(args["response_bound"]) for w in weights)
            eb=sum(abs(w)*Fraction.from_float(float(args["background_bound129"][y,x])) for w,(x,y) in zip(weights,points))
            self.assertEqual(contrast,dict(response=str(yc),background=str(bc),response_error=str(ey),background_error=str(eb)))
            # Independent interval clipping: a*g<=b in both inequalities.
            for a,b in ((-(bc+eb),-(yc-ey)),(bc-eb,yc+ey)):
                if a>0:upper=b/a if upper is None else min(upper,b/a)
                elif a<0:lower=max(lower,b/a)
                else:self.assertGreaterEqual(b,0)
        self.assertEqual(result["exact_gain_interval"],[str(lower),str(upper)])

    def test_no_current_core_access_including_nan_and_inf_poison(self):
        args = self.fixture(); result = estimate_guard_gain(**args)
        for poison in (np.nan,np.inf,-np.inf,1e200):
            changed=dict(args);changed["current129"]=args["current129"].copy()
            changed["current129"][32:97,32:97]=poison
            self.assertEqual(estimate_guard_gain(**changed), result)
        changed=dict(args);changed["current129"]=args["current129"].copy()
        changed["current129"][9,9]=np.nan
        self.assertEqual(estimate_guard_gain(**changed), result)

    def test_every_prior_center_has_radius_twelve_exclusion(self):
        args=self.fixture();args["prior_centers_xy"]=[[16.,64.],[112.,64.],None,None,None,None,None,None]
        result=estimate_guard_gain(**args)
        self.assertEqual(result["missing_prior_center_indices"],list(range(2,8)))
        for x,y in result["used_points_xy"]:
            self.assertGreater(max(abs(x-16),abs(y-64)),12)
            self.assertGreater(max(abs(x-112),abs(y-64)),12)
        self.assertGreater(result["prior_rejection_counts_nonexclusive"]["prior_foreground_footprint"],0)
        self.assertTrue(result["provenance"]["missing_prior_centers_do_not_establish_guard_purity"])

    def test_prior_support_filters_do_not_use_current_values(self):
        args=self.fixture();args["history129"][3,8,8]=np.nan
        args["background129"][8,16]=np.nan;args["background_bound129"][8,24]=-1.
        args["background_bound129"][8,32]=np.inf
        result=estimate_guard_gain(**args)
        self.assertEqual(result["eligible_count"],140)
        for p in ((8,8),(16,8),(24,8),(32,8)):self.assertNotIn(list(p),result["used_points_xy"])
        args["current129"]=np.full((129,129),np.nan)
        invalid=estimate_guard_gain(**args)
        self.assertEqual(invalid["eligible_support_sha256"],result["eligible_support_sha256"])
        self.assertEqual(invalid["used_support_sha256"],result["used_support_sha256"])
        self.assertEqual(invalid["stencil_sha256"],result["stencil_sha256"])

    def test_any_used_current_nan_is_unknown_without_shrinking_support(self):
        args=self.fixture();result=estimate_guard_gain(**args)
        x,y=result["used_points_xy"][0];args["current129"][y,x]=np.nan
        invalid=estimate_guard_gain(**args)
        self.assertFalse(invalid["available"])
        self.assertEqual(invalid["reasons"],["nonfinite_current_on_fixed_used_guard_support"])
        self.assertEqual(invalid["used_support_sha256"],result["used_support_sha256"])
        self.assertEqual(invalid["stencil_count"],result["stencil_count"])
        self.assertEqual(invalid["current_nonfinite_used_point_count"],1)
        self.assertEqual(invalid["contrast_constraints"],[])

    def test_eligible_but_unused_current_point_is_never_gathered(self):
        args=self.fixture();args["history129"][:]=np.nan
        for x,y in ((8,8),(16,8),(24,8),(120,120)):
            args["history129"][:,y,x]=args["background129"][y,x]
        result=estimate_guard_gain(**args)
        self.assertEqual(result["eligible_count"],4)
        self.assertEqual(result["used_count"],3)
        self.assertEqual(result["stencil_count"],1)
        args["current129"][120,120]=np.nan
        self.assertEqual(estimate_guard_gain(**args),result)

    def test_finite_zero_and_255_are_not_reclassified_as_saturation(self):
        args=self.fixture();args["background_bound129"]*=0;args["response_bound"]=0.
        for value in (0.,255.):
            args["current129"]=np.full((129,129),value)
            result=estimate_guard_gain(**args)
            self.assertTrue(result["available"],result["reasons"])
            self.assertEqual(result["gain_interval"],[0.,0.])

    def test_no_contrasts_empty_intersection_and_unbounded_are_distinct_unknowns(self):
        args=self.fixture();args["background_bound129"][:]=np.nan
        result=estimate_guard_gain(**args)
        self.assertEqual(result["reasons"],["no_prior_selected_guard_contrasts"])
        self.assertIsNone(result["current_nonfinite_used_point_count"])
        args=self.fixture();args["background129"][:]=50.;args["current129"][:]=50.
        self.assertEqual(estimate_guard_gain(**args)["reasons"],["guard_gain_outer_interval_unbounded"])
        args=self.fixture();args["current129"]=-args["background129"]
        args["response_bound"]=0.;args["background_bound129"]*=0
        self.assertEqual(estimate_guard_gain(**args)["reasons"],["guard_gain_necessary_constraints_inconsistent"])

    def test_same_priors_can_support_different_current_gain_intervals(self):
        one,two=self.fixture(1.),self.fixture(2.)
        np.testing.assert_array_equal(one["history129"],two["history129"])
        a,b=estimate_guard_gain(**one),estimate_guard_gain(**two)
        self.assertLess(a["gain_interval"][1],b["gain_interval"][0])
        self.assertTrue(a["provenance"]["same_frame_guard_uses_current_pixels"])

    def test_simultaneous_correlated_background_and_response_errors_contain_gain(self):
        rng=np.random.default_rng(4709)
        for gain in (0.,.5,1.25,2.):
            for aligned in (-1.,1.):
                args=self.fixture(gain)
                signs=rng.choice([-1.,1.],(129,129))
                args["current129"]+=gain*.5*signs+aligned*.5*signs
                result=estimate_guard_gain(**args)
                self.assertTrue(result["available"],result["reasons"])
                self.assertLessEqual(result["gain_interval"][0],gain)
                self.assertGreaterEqual(result["gain_interval"][1],gain)
                self.assertTrue(result["provenance"]["no_independence_or_sample_count_reduction"])

    def test_bilinear_xy_is_annihilated_and_not_affine_validity_certificate(self):
        args=self.fixture();base=estimate_guard_gain(**args)
        y,x=np.indices((129,129));args["current129"]+=x*y/64
        result=estimate_guard_gain(**args)
        self.assertEqual(result["gain_interval"],base["gain_interval"])
        self.assertEqual(result["contrast_constraints"],base["contrast_constraints"])
        self.assertTrue(result["provenance"]["horizontal_vertical_contrasts_also_annihilate_bilinear_xy"])
        self.assertFalse(result["provenance"]["global_affine_fit_or_joint_model_feasibility_certified"])

    def test_sparse_grid_misses_between_grid_contamination_by_design(self):
        args=self.fixture();base=estimate_guard_gain(**args)
        args["current129"][9,9]+=1000
        self.assertEqual(estimate_guard_gain(**args),base)
        self.assertTrue(base["provenance"]["sparse_grid_can_miss_between_grid_contamination"])

    def test_inputs_unmutated_and_malformed_shapes_rejected(self):
        args=self.fixture();before={k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
        estimate_guard_gain(**args)
        for k,v in before.items():np.testing.assert_array_equal(args[k],v)
        for key,value in (("current129",np.zeros((25,25))), ("response_bound",-.1),
                          ("prior_centers_xy",[None]*7), ("prior_centers_xy",[[np.nan,1.]]*8)):
            changed=dict(args);changed[key]=value
            with self.assertRaises(ValueError):estimate_guard_gain(**changed)


if __name__=="__main__":unittest.main()
