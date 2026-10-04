"""Synthetic-only endpoint-hull and exact-centering checks; no media inputs."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v47_bounded_background as model


def fixture(amplitude=35.):
    yy,xx=np.mgrid[-1:1:9j,-1:1:9j]
    x,y=xx.ravel(),yy.ravel()
    P=np.column_stack((np.ones(len(x)),x,y))
    B=40+4*np.sin(3*x)+2*np.cos(2*y)
    m=np.exp(-((x-.1)**2+(y+.1)**2)/.04); m/=np.linalg.norm(m)
    f1=np.exp(-((x+.65)**2+(y-.6)**2)/.05); f1/=np.linalg.norm(f1)
    f2=np.exp(-((x-.7)**2+(y-.6)**2)/.06); f2/=np.linalg.norm(f2)
    F=np.column_stack((f1,f2))
    return dict(y=B+amplitude*m+F@np.array([3.,-2.])+P@np.array([20.,.4,-.3]),
        B=B,F=F,m=m,P=P,gain_interval=[.98,1.02],response_bound=.02,
        background_bound=.01,fixed_bound=1e-5,source_bound=1e-5,
        gain_provenance={"kind":"synthetic_oracle_interval","physical_validity":"not self-certified"})


def direct_numerator(y,B,F,m,P,gain):
    affine=np.linalg.qr(P,mode="reduced")[0]
    q=lambda v:v-affine@(affine.T@v)
    source,response=q(m),q(y-gain*B)
    if F.shape[1]:
        fixed=np.linalg.qr(q(F),mode="reduced")[0]
        response=response-fixed@(fixed.T@response)
    return float(source@response)


def core_record(interval,available=True):
    sign=None if not available else "positive" if interval[0]>0 else "negative" if interval[1]<0 else "unresolved"
    return dict(available=available,reasons=[] if available else ["generated_unknown"],
        numerator=None if not available else sum(interval)/2,
        error_bound=None if not available else (interval[1]-interval[0])/2,
        interval=interval,interval_excludes_zero=None if not available else sign!="unresolved",
        coefficient_sign=sign,motion_status="unknown",physical_class="unknown",
        diagnostics={"no_production_gate":True,"complete_old_record":"preserved"})


class ArithmeticTests(unittest.TestCase):
    def test_rounding_allowance_encloses_exact_float_linear_combination(self):
        y=np.array([1.,.3,1e200,1e-200]); B=np.array([.1,.7,-3e199,3e-201]); gain=1.3
        ey,ub=np.array([.1,.2,.3,0.]),np.array([.2,.1,.5,0.])
        value=model.centered_response(y,B,gain,ey,ub)
        self.assertTrue(value["available"])
        for i in range(len(y)):
            exact=Fraction(float(y[i]))-Fraction(gain)*Fraction(float(B[i]))
            arithmetic=abs(exact-Fraction(float(value["response"][i])))
            admitted=Fraction(float(ey[i]))+Fraction(gain)*Fraction(float(ub[i]))+arithmetic
            self.assertGreaterEqual(Fraction(float(value["centering_roundoff_bound"][i])),arithmetic)
            self.assertGreaterEqual(Fraction(float(value["response_bound"][i])),admitted)
            if value["response_bound"][i]>0:
                previous=np.nextafter(value["response_bound"][i],-np.inf)
                self.assertLess(Fraction(float(previous)),admitted)

    def test_cancelling_overflowing_intermediate_remains_exactly_representable(self):
        largest=np.finfo(float).max
        value=model.centered_response(np.array([largest]),np.array([largest]),2.,0.,0.)
        self.assertTrue(value["available"])
        self.assertEqual(value["response"][0],-largest)
        self.assertEqual(value["response_bound"][0],0.)

    def test_subnormal_arithmetic_error_is_not_dropped(self):
        tiny=np.nextafter(0.,1.)
        value=model.centered_response(np.array([0.]),np.array([tiny]),tiny,0.,0.)
        self.assertTrue(value["available"])
        self.assertEqual(value["response"][0],0.)
        self.assertEqual(value["response_bound"][0],tiny)
        self.assertGreaterEqual(Fraction(float(value["response_bound"][0])),Fraction(float(tiny))**2)

    def test_unrepresentable_response_or_bound_is_unknown_without_arrays(self):
        largest=np.finfo(float).max
        for B,ub,expected in ((largest,0.,"response"),(0.,largest,"bound")):
            value=model.centered_response(np.array([0.]),np.array([B]),largest,0.,ub)
            self.assertFalse(value["available"]); self.assertIn(expected,value["reason"])
            self.assertIsNone(value["response"]); self.assertIsNone(value["response_bound"])

    def test_seeded_extreme_scale_fraction_audit(self):
        rng=np.random.default_rng(470019)
        for _ in range(100):
            y=rng.uniform(-1,1,5)*10.**rng.uniform(-300,300,5)
            B=rng.uniform(-1,1,5)*10.**rng.uniform(-300,300,5)
            gain=float(10.**rng.uniform(-100,100))
            value=model.centered_response(y,B,gain,.5,.5)
            if not value["available"]:
                continue
            for i in range(5):
                exact=Fraction(float(y[i]))-Fraction(gain)*Fraction(float(B[i]))
                rounding=abs(exact-Fraction(float(value["response"][i])))
                bound=Fraction(.5)+Fraction(gain)*Fraction(.5)+rounding
                self.assertGreaterEqual(Fraction(float(value["response_bound"][i])),bound)


class EndpointTests(unittest.TestCase):
    def test_missing_or_unsupported_gain_never_accesses_input_or_core(self):
        for interval in (None,[-1.,1.],[2.,1.],[0.,np.inf],[np.nan,1.],[True,False],
                         [True,1.],[[1.],[2.,3.]],[1.],"gain"):
            with self.subTest(interval=interval), mock.patch.object(model,"source_presence") as core:
                value=model.bounded_background_presence(None,None,None,None,None,gain_interval=interval,
                    response_bound=.5,background_bound=.5,fixed_bound=None,source_bound=.1,gain_provenance=None)
                self.assertFalse(value["available"]); self.assertIsNone(value["interval"])
                self.assertEqual(value["endpoint_evaluations"],[]); core.assert_not_called()
                json.dumps(value,allow_nan=False)

    def test_missing_declared_errors_do_not_silently_become_zero(self):
        for key in ("response_bound","background_bound","fixed_bound","source_bound"):
            case=fixture(); case[key]=None
            with self.subTest(key=key), mock.patch.object(model,"source_presence") as core:
                value=model.bounded_background_presence(**case)
                self.assertFalse(value["available"]); core.assert_not_called()

    def test_endpoint_hull_retains_complete_records_and_never_selects_favorable(self):
        endpoints=[core_record([2.,4.]),core_record([-5.,-1.])]
        with mock.patch.object(model,"source_presence",side_effect=deepcopy(endpoints)) as core:
            value=model.bounded_background_presence(**fixture())
        self.assertTrue(value["available"]); self.assertEqual(value["interval"],[-5.,4.])
        self.assertEqual(value["coefficient_sign"],"unresolved")
        self.assertIsNone(value["numerator"]); self.assertIsNone(value["error_bound"])
        self.assertEqual(value["endpoint_nominal_numerators"],[3.,-3.])
        self.assertEqual([r["source_presence"] for r in value["endpoint_evaluations"]],endpoints)
        self.assertEqual(core.call_count,2)

    def test_all_fixed_columns_and_rows_pass_unchanged(self):
        case=fixture(); captured=[]
        def check(y,F,m,P,**bounds):
            captured.append((y.copy(),F.copy(),m.copy(),P.copy(),deepcopy(bounds)))
            return core_record([1.,2.])
        with mock.patch.object(model,"source_presence",side_effect=check):
            value=model.bounded_background_presence(**case)
        self.assertTrue(value["available"])
        for index,(response,F,m,P,bounds) in enumerate(captured):
            np.testing.assert_array_equal(F,case["F"])
            np.testing.assert_array_equal(m,case["m"])
            np.testing.assert_array_equal(P,case["P"])
            self.assertEqual(response.shape,case["y"].shape)
            gain=case["gain_interval"][index]
            self.assertTrue(np.all(bounds["response_bound"]>=case["response_bound"]+gain*case["background_bound"]))
            self.assertEqual(bounds["nuisance_bound"].shape,case["F"].shape)

    def test_singleton_explicitly_evaluates_and_retains_both_identical_endpoints(self):
        case=fixture(); case["gain_interval"]=[1.,1.]
        value=model.bounded_background_presence(**case)
        self.assertTrue(value["diagnostics"]["singleton_gain_interval"])
        self.assertEqual(value["diagnostics"]["endpoint_calls"],2)
        self.assertEqual(value["endpoint_evaluations"][0]["source_presence"],value["endpoint_evaluations"][1]["source_presence"])

    def test_one_or_both_failed_endpoints_leave_no_operative_hull(self):
        for endpoints in ((core_record(None,False),core_record([1.,2.])),
                          (core_record(None,False),core_record(None,False))):
            with self.subTest(endpoints=endpoints), mock.patch.object(model,"source_presence",side_effect=deepcopy(endpoints)):
                value=model.bounded_background_presence(**fixture())
            self.assertFalse(value["available"]); self.assertIsNone(value["interval"])
            self.assertIsNone(value["coefficient_sign"]); self.assertIsNone(value["endpoint_nominal_numerators"])
            self.assertEqual(len(value["endpoint_evaluations"]),2)
            self.assertEqual([r["source_presence"] for r in value["endpoint_evaluations"]],list(endpoints))

    def test_source_absence_and_wide_gain_ambiguity_do_not_become_positive(self):
        absent=fixture(0.); absent["gain_interval"]=[1.,1.]
        value=model.bounded_background_presence(**absent)
        self.assertTrue(value["available"]); self.assertEqual(value["coefficient_sign"],"unresolved")
        case=fixture(); case.update(F=np.empty((81,0)),m=case["B"].copy(),y=case["B"].copy(),
                                   gain_interval=[0.,2.],fixed_bound=None)
        value=model.bounded_background_presence(**case)
        self.assertTrue(value["available"]); self.assertEqual(value["coefficient_sign"],"unresolved")
        self.assertLess(value["interval"][0],0); self.assertGreater(value["interval"][1],0)

    def test_uncertain_fixed_overlap_and_flat_source_remain_unknown(self):
        for mode in ("overlap","flat"):
            case=fixture()
            if mode=="overlap": case.update(F=case["m"][:,None],fixed_bound=.001)
            else: case["m"]=np.ones_like(case["m"])
            value=model.bounded_background_presence(**case)
            self.assertFalse(value["available"]); self.assertIsNone(value["interval"])
            self.assertTrue(all(not e["available"] for e in value["endpoint_evaluations"]))

    def test_affine_changes_preserve_numerators_with_resolution_caveat(self):
        case=fixture(); first=model.bounded_background_presence(**case)
        changed=deepcopy(case)
        changed["y"]+=case["P"]@np.array([80.,2.,-3.])
        changed["B"]+=case["P"]@np.array([15.,-1.,.2])
        second=model.bounded_background_presence(**changed)
        self.assertTrue(first["available"] and second["available"])
        np.testing.assert_allclose(first["endpoint_nominal_numerators"],second["endpoint_nominal_numerators"],atol=1e-10,rtol=0)
        for a,b in zip(first["endpoint_evaluations"],second["endpoint_evaluations"]):
            np.testing.assert_allclose(a["source_presence"]["analytic_interval"],b["source_presence"]["analytic_interval"],atol=1e-9,rtol=0)
        self.assertFalse(first["diagnostics"]["whole_pipeline_is_ieee_certified_enclosure"])

    def test_inputs_and_provenance_are_not_mutated_or_self_certified(self):
        case=fixture(); original=deepcopy(case)
        value=model.bounded_background_presence(**case)
        for key in ("y","B","F","m","P"): np.testing.assert_array_equal(case[key],original[key])
        self.assertEqual(case["gain_provenance"],original["gain_provenance"])
        self.assertFalse(value["gain_provenance_certified"])
        self.assertEqual(value["motion_status"],"unknown"); self.assertEqual(value["physical_class"],"unknown")
        self.assertFalse(value["is_motion_or_classification_gate"])
        json.dumps(value,allow_nan=False)

    def test_malformed_support_bounds_and_core_output_fail_closed(self):
        for key,bad in (("y",np.full(81,np.nan)),("B",np.ones(80)),("fixed_bound",-.1),
                        ("source_bound",np.ones(80))):
            case=fixture(); case[key]=bad
            with self.subTest(key=key), self.assertRaises(ValueError): model.bounded_background_presence(**case)
        invalid=core_record([1.,2.]); invalid["coefficient_sign"]="negative"
        with mock.patch.object(model,"source_presence",return_value=invalid), self.assertRaisesRegex(ValueError,"disagree"):
            model.bounded_background_presence(**fixture())

    def test_600_simultaneous_correlated_perturbations_fit_inside_hull(self):
        case=fixture(); result=model.bounded_background_presence(**case)
        self.assertTrue(result["available"])
        low,high=result["interval"]; rng=np.random.default_rng(470031)
        for index in range(600):
            if index<4:
                sign=(-1.)**index
                dy=np.full(81,sign*case["response_bound"])
                db=np.full(81,(-sign if index<2 else sign)*case["background_bound"])
                dm=np.full(81,sign*case["source_bound"])
                df=np.full(case["F"].shape,sign*case["fixed_bound"])
            else:
                shared=rng.uniform(-1,1,81)
                dy=shared*case["response_bound"]; db=shared*case["background_bound"]
                dm=shared*case["source_bound"]
                df=rng.uniform(-1,1,case["F"].shape)*case["fixed_bound"]
            gain=case["gain_interval"][index%2] if index<4 else float(rng.uniform(*case["gain_interval"]))
            actual=direct_numerator(case["y"]+dy,case["B"]+db,case["F"]+df,case["m"]+dm,case["P"],gain)
            self.assertGreaterEqual(actual,low-1e-10); self.assertLessEqual(actual,high+1e-10)


if __name__ == "__main__":
    unittest.main()
