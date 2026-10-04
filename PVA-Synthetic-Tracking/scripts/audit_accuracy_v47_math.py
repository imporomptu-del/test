"""Independent synthetic endpoint/centering audit; no media or cache inputs.

The fixed matrix is declared before results. Full-space QR residualization and
full-design least squares are separate from the tested projected V45 solver.
Monte Carlo/adversarial finite cases are bug-finding, not proof, confidence,
guard-to-core transfer validation, or physical-object accuracy.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from accuracy_v47_bounded_background import bounded_background_presence, centered_response


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v47_20260926"
SEED = 470071
PERTURBATIONS_PER_CASE = 32


def matrix_manifest():
    return dict(seed=SEED, rows=[32,64], fixed_columns=[0,2], source_coefficients=[-10.,0.,10.],
                gain_intervals=[[1.1,1.1],[.7,1.4]], case_count=24,
                perturbations_per_case=PERTURBATIONS_PER_CASE, total_realizations=768,
                perturbation_modes=["shared_sign_corners", "independent_corners",
                                    "nominal_residual_adversarial_corners", "uniform_correlated"],
                gain_fractions=[0.,1.,.125,.25,.5,.75,.875],
                bounds=dict(response=.05,background=.01,fixed=.0001,source=.0001),
                numerical_comparison_atol=1e-9, no_gain_selection_from_results=True)


def build_matrix():
    cases=[]
    for n in (32,64):
        yy,xx=np.indices((n//8,8));x=2*xx.ravel()/7-1;y=2*yy.ravel()/(n//8-1)-1
        P=np.column_stack((np.ones(n),x,y))
        for nf in (0,2):
            rng=np.random.default_rng(SEED+n+nf)
            m=np.exp(-((x-.15)**2+(y+.1)**2)/.18);m/=np.linalg.norm(m)
            F=rng.normal(size=(n,nf))
            if nf:F/=np.linalg.norm(F,axis=0)
            B=30+2*np.sin(3*x)+np.cos(2*y)+3*m
            residual=.2*rng.normal(size=n)
            for amplitude in (-10.,0.,10.):
                for interval in ([1.1,1.1],[.7,1.4]):
                    case_id=f'n{n}_f{nf}_a{amplitude:g}_g{interval[0]:g}_{interval[1]:g}'
                    response=1.05*B+amplitude*m+F@np.arange(2.,2.+nf)+P@np.array([100.,.5,-.3])+residual
                    cases.append(dict(case_id=case_id,args=dict(y=response.copy(),B=B.copy(),F=F.copy(),
                        m=m.copy(),P=P.copy(),gain_interval=list(interval),response_bound=.05,
                        background_bound=.01,fixed_bound=.0001 if nf else None,source_bound=.0001,
                        gain_provenance={"kind":"fixed_synthetic_test_interval","not_guard_derived":True})))
    return cases


def direct_partial_regression(y,B,F,m,P,gain):
    """One QR of the full exact affine plus perturbed fixed nuisance design."""
    nuisance=np.column_stack((P,F))
    scaled=nuisance/np.linalg.norm(nuisance,axis=0)
    q=np.linalg.qr(scaled,mode="reduced")[0]
    z=y-gain*B
    rm=m-q@(q.T@m);rz=z-q@(q.T@z)
    numerator=float(rm@rz)
    coefficient=float(np.linalg.lstsq(np.column_stack((nuisance,m)),z,rcond=None)[0][-1])
    return numerator,coefficient,float(rm@rm)


def _input_hashes(cases):
    result={}
    for case in cases:
        for name,value in case["args"].items():
            if isinstance(value,np.ndarray):
                payload=str((value.shape,value.dtype.str)).encode()+value.tobytes()
            else:payload=json.dumps(value,sort_keys=True,allow_nan=False).encode()
            result[case["case_id"]+":"+name]=hashlib.sha256(payload).hexdigest()
    return result


def audit_matrix(cases=None, perturbations=PERTURBATIONS_PER_CASE):
    cases=build_matrix() if cases is None else cases
    before=_input_hashes(cases);issues=[];records=[];realizations=0
    for ci,case in enumerate(cases):
        a=case["args"];result=bounded_background_presence(**a)
        json.dumps(result,allow_nan=False)
        entry=dict(case_id=case["case_id"],available=result["available"],reasons=result["reasons"],
                   interval=result["interval"],realizations=0,max_hull_violation=0.,
                   max_endpoint_affine_identity_error=0.,max_coefficient_identity_error=0.)
        if not result["available"]:
            issues.append(dict(case_id=case["case_id"],reason="predeclared_regular_system_unavailable"))
            records.append(entry);continue
        if (result["motion_status"]!="unknown" or result["physical_class"]!="unknown"
                or result["is_motion_or_classification_gate"] or result["production_changed"]):
            issues.append(dict(case_id=case["case_id"],reason="physical_or_production_claim"))
        if result["numerator"] is not None or result["error_bound"] is not None:
            issues.append(dict(case_id=case["case_id"],reason="invented_single_gain_estimand"))
        endpoints=result["endpoint_evaluations"]
        expected_hull=[min(e["source_presence"]["interval"][0] for e in endpoints),
                       max(e["source_presence"]["interval"][1] for e in endpoints)]
        if result["interval"]!=expected_hull or len(endpoints)!=2:
            issues.append(dict(case_id=case["case_id"],reason="endpoint_hull_not_exact_union"))
        for index,e in enumerate(endpoints):
            if (e["gain"]!=a["gain_interval"][index] or e["source_presence"]["diagnostics"]["rows"]!=len(a["y"])
                    or e["source_presence"]["diagnostics"]["original_nuisance_columns"]!=a["F"].shape[1]
                    or e["source_presence"]["motion_status"]!="unknown" or e["source_presence"]["physical_class"]!="unknown"):
                issues.append(dict(case_id=case["case_id"],reason="endpoint_membership_or_physics_changed"))
        low,high=result["interval"];glo,ghi=a["gain_interval"]
        rng=np.random.default_rng(SEED+1000+ci)
        nominal_nuisance=np.column_stack((a["P"],a["F"]))
        q=np.linalg.qr(nominal_nuisance,mode="reduced")[0]
        adversarial=np.where(a["m"]-q@(q.T@a["m"])>=0,1.,-1.)
        for iteration in range(perturbations):
            mode=iteration%4
            shared=rng.choice([-1.,1.],len(a["y"]))
            if mode==0:
                dy=shared*.05;db=-shared*.01;dm=shared*.0001
                df=np.broadcast_to(shared[:,None],a["F"].shape)*.0001
            elif mode==1:
                dy=shared*.05;db=rng.choice([-1.,1.],len(shared))*.01
                dm=rng.choice([-1.,1.],len(shared))*.0001
                df=rng.choice([-1.,1.],a["F"].shape)*.0001
            elif mode==2:
                sign=-1. if iteration%8==2 else 1.
                dy=sign*adversarial*.05;db=-sign*adversarial*.01;dm=sign*adversarial*.0001
                df=np.broadcast_to(sign*adversarial[:,None],a["F"].shape)*.0001
            else:
                shared=rng.uniform(-1.,1.,len(shared))
                dy=shared*.05;db=shared*.01;dm=shared*.0001
                df=rng.uniform(-1.,1.,a["F"].shape)*.0001
            fraction=(0.,1.,.125,.25,.5,.75,.875)[iteration%7]
            gain=(1-fraction)*glo+fraction*ghi
            vals=(a["y"]+dy,a["B"]+db,a["F"]+df,a["m"]+dm,a["P"])
            left=direct_partial_regression(*vals,glo)[0]
            right=direct_partial_regression(*vals,ghi)[0]
            interior,coefficient,energy=direct_partial_regression(*vals,gain)
            violation=max(0.,*(low-v for v in (left,right,interior)),*(v-high for v in (left,right,interior)))
            affine_error=abs(interior-((1-fraction)*left+fraction*right))
            coefficient_error=abs(interior-coefficient*energy)
            entry["max_hull_violation"]=max(entry["max_hull_violation"],violation)
            entry["max_endpoint_affine_identity_error"]=max(entry["max_endpoint_affine_identity_error"],affine_error)
            entry["max_coefficient_identity_error"]=max(entry["max_coefficient_identity_error"],coefficient_error)
            if max(violation,affine_error,coefficient_error)>1e-9:
                issues.append(dict(case_id=case["case_id"],iteration=iteration,reason="independent_refit_discrepancy",
                                   hull_violation=violation,affine_error=affine_error,coefficient_error=coefficient_error))
            if result["coefficient_sign"]!="unresolved" and (coefficient>0)!=(result["coefficient_sign"]=="positive"):
                issues.append(dict(case_id=case["case_id"],iteration=iteration,reason="certified_sign_not_reflected_in_direct_coefficient"))
            entry["realizations"]+=1;realizations+=1
        records.append(entry)
    if before!=_input_hashes(cases):issues.append(dict(reason="audit_or_core_mutated_inputs"))
    return dict(passed=not issues,issues=issues,case_count=len(cases),realizations=realizations,
                direct_partial_regressions=3*realizations,records=records)


def audit_centering():
    tiny=np.nextafter(0.,1.);largest=np.finfo(float).max
    cases=[("intermediate_overflow_cancels",[largest],[largest],2.,0.,0.),
           ("subnormal_product",[0.],[tiny],tiny,0.,0.),
           ("half_subnormal",[0.],[tiny],.5,0.,0.),
           ("large_cancellation",[1e300],[1e300],1.,.5,.5),
           ("unrepresentable_response",[0.],[largest],largest,0.,0.),
           ("unrepresentable_bound",[0.],[0.],largest,0.,largest)]
    rng=np.random.default_rng(SEED+4000)
    for index in range(32):
        y=np.ldexp(rng.uniform(-1.,1.,4),rng.integers(-1000,1001,4))
        b=np.ldexp(rng.uniform(-1.,1.,4),rng.integers(-1000,1001,4))
        g=float(np.ldexp(rng.uniform(.5,1.),int(rng.integers(-500,501))))
        cases.append((f"wide_scale_{index}",y,b,g,.5,.5))
    issues=[];rows=0;available=0;unknown=0
    for name,y,b,g,ey,ub in cases:
        value=centered_response(np.asarray(y),np.asarray(b),g,ey,ub)
        expected=[];representable=True
        for yi,bi in zip(y,b):
            exact=Fraction(float(yi))-Fraction(g)*Fraction(float(bi))
            try:rounded=float(exact)
            except OverflowError:representable=False;break
            if not math.isfinite(rounded):representable=False;break
            arithmetic=abs(exact-Fraction(rounded))
            required=Fraction(ey)+Fraction(g)*Fraction(ub)+arithmetic
            try:bound=float(required)
            except OverflowError:representable=False;break
            if not math.isfinite(bound):representable=False;break
            if Fraction(bound)<required:bound=math.nextafter(bound,math.inf)
            if not math.isfinite(bound):representable=False;break
            expected.append((rounded,required,arithmetic,bound))
        if value["available"]!=representable:
            issues.append(dict(case_id=name,reason="centering_representability_disagrees"))
        if not value["available"]:
            unknown+=1
            if any(value[k] is not None for k in ("response","response_bound","centering_roundoff_bound")):
                issues.append(dict(case_id=name,reason="partial_arithmetic_arrays_leaked"))
            continue
        available+=1
        for index,(rounded,required,arithmetic,bound) in enumerate(expected):
            rows+=1
            got=float(value["response_bound"][index]);ar=float(value["centering_roundoff_bound"][index])
            if value["response"][index]!=rounded or got!=bound or Fraction(got)<required or Fraction(ar)<arithmetic:
                issues.append(dict(case_id=name,row=index,reason="fraction_center_or_outer_bound_mismatch"))
            if got>0 and Fraction(math.nextafter(got,-math.inf))>=required:
                issues.append(dict(case_id=name,row=index,reason="assembled_bound_not_minimal_outward_float"))
    return dict(passed=not issues,issues=issues,cases=len(cases),available_cases=available,
                unknown_cases=unknown,exact_scalar_rows_checked=rows)


def audit_structural_unknowns():
    cases=[]
    for reason in ("missing_gain","source_in_fixed_span","source_in_affine_span","uncertain_fixed_rank"):
        a=build_matrix()[3]["args"]
        if reason=="missing_gain":a["gain_interval"]=None
        elif reason=="source_in_fixed_span":a.update(F=a["m"][:,None],fixed_bound=.0001)
        elif reason=="source_in_affine_span":a["m"]=a["P"][:,0].copy()
        else:a.update(F=np.column_stack((a["m"],a["m"])),fixed_bound=1.)
        value=bounded_background_presence(**a)
        cases.append(dict(case_id=reason,available=value["available"],reasons=value["reasons"],
                          no_operative_hull=value["interval"] is None and value["coefficient_sign"] is None))
    issues=[dict(case_id=c["case_id"],reason="expected_structural_unknown_not_preserved")
            for c in cases if c["available"] or not c["no_operative_hull"]]
    return dict(passed=not issues,issues=issues,records=cases)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dependencies():
    return [Path(__file__).resolve(),ROOT/"tests/unit/test_accuracy_v47_math_audit.py",
            ROOT/"scripts/accuracy_v47_bounded_background.py",ROOT/"scripts/accuracy_v47_guard_gain.py",
            ROOT/"scripts/accuracy_v47_probe.py",ROOT/"scripts/accuracy_v45_presence.py",
            ROOT/"scripts/accuracy_v45_bounds.py",ROOT/"scripts/accuracy_v44_causal_probe.py",
            ROOT/"scripts/accuracy_v44_contrast.py",
            ROOT/"scripts/accuracy_v43_bounds.py",ROOT/"scripts/accuracy_v43_components.py",
            ROOT/"scripts/accuracy_v42_localized.py",
            ROOT/"docs/accuracy_v47_plan.md"]


def run(output, *, execute=False):
    if not execute:raise ValueError("Explicit execute required after root freeze approval")
    output=Path(output).resolve()
    if output.parent!=OUTPUT_ROOT or output.suffix!=".json":raise ValueError("Fresh JSON in dedicated V47 root required")
    freeze=output.with_name(output.stem+"_freeze.json")
    if output.exists() or freeze.exists():raise FileExistsError("Never overwrite a persisted audit or freeze")
    dependencies={str(p):_sha(p) for p in _dependencies()}
    cases=build_matrix();inputs=_input_hashes(cases)
    output.parent.mkdir(parents=True,exist_ok=True)
    with freeze.open("x") as stream:
        json.dump(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),manifest=matrix_manifest(),
                       dependency_sha256=dependencies,input_value_sha256=inputs,
                       numerical_checks_started=False,synthetic_only=True),stream,indent=2,allow_nan=False)
        stream.write("\n")
    if any(_sha(p)!=h for p,h in dependencies.items()) or inputs!=_input_hashes(cases):
        raise ValueError("Frozen audit inputs/code changed before checks")
    matrix,centering,structural=audit_matrix(cases),audit_centering(),audit_structural_unknowns()
    if any(_sha(p)!=h for p,h in dependencies.items()) or inputs!=_input_hashes(cases):
        raise ValueError("Frozen audit inputs/code changed during checks")
    issues=matrix["issues"]+centering["issues"]+structural["issues"]
    report=dict(completed=True,passed=not issues,issues=issues,created_at_utc=datetime.now(timezone.utc).isoformat(),
                freeze_path=str(freeze),freeze_sha256=_sha(freeze),audit_files_sha256=dependencies,
                matrix=matrix,centering=centering,
                structural_unknowns=structural,real_data_accessed=False,solver_or_production_changed=False,
                proof_scope="For one fixed simultaneous realization, the residualized numerator is affine in gain; endpoint hull encloses every interior gain. Exact Fraction centering checked separately.",
                limits=["Finite seeded/adversarial refits are bug-finding, not a proof of all numerical implementations.",
                        "No real images, camera-noise calibration, guard purity, gain-to-core transfer, motion/class or detection accuracy validated.",
                        "Whole V45 SVD/projection remains subject to its declared non-IEEE-certified numerical-resolution guard.",
                        "This is the bounded-gain partial-regression numerator, not unrestricted-background least squares or source amplitude."])
    with output.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--execute",action="store_true")
    arguments=parser.parse_args();result=run(arguments.output,execute=arguments.execute)
    print(json.dumps(dict(passed=result["passed"],issues=result["issues"],realizations=result["matrix"]["realizations"]),indent=2))
