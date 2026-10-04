"""Generated tests for the bounded larger-shift extension; no hardware/media."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
large = load("diagnose_aot_large_shifts")
helper = load("diagnose_aot_features")
REPLACEMENTS = [
  [
    "Predeclared 61-call natural-texture integer-shift diagnostic",
    "Predeclared 93-call larger natural-texture integer-shift diagnostic"
  ],
  [
    "seaqr.aot.natural-shifts",
    "seaqr.aot.large-shifts"
  ],
  [
    "seaqr_aot_natural_shifts_20260927",
    "seaqr_aot_large_shifts_20260928"
  ],
  [
    "diagnose_aot_natural_shifts.py",
    "diagnose_aot_large_shifts.py"
  ],
  [
    "test_aot_natural_shifts.py",
    "test_aot_large_shifts.py"
  ],
  [
    "SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))",
    "SHIFTS = ((0, 0), (1, 0), (-1, 0), (16, 0), (-16, 0), (32, 0), (-32, 0), (48, 0), (-48, 0), (32, -16), (-32, 16))"
  ],
  [
    "CONTROL_SEED, WIDTH, HEIGHT, PAIR_COUNT = 20260927, 2448, 2048, 61",
    "CONTROL_SEED, WIDTH, HEIGHT, PAIR_COUNT = 20260927, 2448, 2048, 93"
  ],
  [
    "fixed61 natural-shift inventory differs",
    "fixed93 larger-shift inventory differs"
  ],
  [
    "len(cases) == 7",
    "len(cases) == len(SHIFTS)"
  ],
  [
    "scope/integrity/61 completed calls",
    "scope/integrity/93 completed calls"
  ],
  [
    "receipt[\"completed_motion_pair_calls\"] % 7 == 0",
    "receipt[\"completed_motion_pair_calls\"] % len(SHIFTS) == 0"
  ]
]

class LargerShiftTests(unittest.TestCase):
    def test_only_predeclared_source_adaptations(self):
        old=(SCRIPTS/"diagnose_aot_natural_shifts.py").read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),"79b6d5c288352bfd0c44d7a4fa3b2181eab7277719197c2f0e95243df2b0660e")
        for before,after in REPLACEMENTS:
            self.assertGreater(old.count(before),0)
            old=old.replace(before,after)
        self.assertEqual(old,(SCRIPTS/"diagnose_aot_large_shifts.py").read_text())

    def test_computational_functions_byte_identical(self):
        a=(SCRIPTS/"diagnose_aot_natural_shifts.py").read_text()
        b=(SCRIPTS/"diagnose_aot_large_shifts.py").read_text()
        def functions(source):
            return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body
                    if isinstance(node,ast.FunctionDef)}
        fa,fb=functions(a),functions(b)
        for name in ("references","load_module","encode_raw","decode_raw","preflow_support",
                     "make_capture_class","attach_attrition","generate_cases","compress_result"):
            self.assertEqual(fa[name],fb[name],name)

    def test_exact_case_order_count_and_unchanged_arm(self):
        shifts=((0,0),(1,0),(-1,0),(16,0),(-16,0),(32,0),(-32,0),(48,0),(-48,0),(32,-16),(-32,16))
        self.assertEqual(large.SHIFTS,shifts)
        expected=[f"aot_prev{i:03d}_dx{dx:+d}_dy{dy:+d}" for i in (0,42,85,127,170,212,255,298) for dx,dy in shifts]
        expected+=["high_contrast_static","high_contrast_translated","low_contrast_static","low_contrast_translated","flat_static"]
        self.assertEqual(large.case_ids(),expected)
        self.assertEqual(large.PAIR_COUNT,len(expected))
        self.assertEqual(len(set(expected)),93)
        self.assertEqual(large.ARM,load("diagnose_aot_natural_shifts").ARM)

    def test_shift_sign_codes_and_no_wrap(self):
        original=np.arange(110*120,dtype=np.uint16).reshape(110,120).astype(np.uint8)
        original.setflags(write=False)
        for dx,dy in large.SHIFTS:
            expected=np.full_like(original,128)
            for y in range(110):
                for x in range(120):
                    if 0<=x-dx<120 and 0<=y-dy<110:
                        expected[y,x]=original[y-dy,x-dx]
            actual=helper.translate_no_wrap(original,dx,dy,128)
            np.testing.assert_array_equal(actual,expected)
            self.assertFalse(np.shares_memory(actual,original))

    def test_large_shift_preflow_support_uses_truth_and_exact_boundaries(self):
        points=np.array([[128,128],[127.9,128],[200,128],[2300,1800],[2319,1919],[129,1919],[2400,2000]],float)
        for dx,dy in large.SHIFTS:
            expected=[128<=x<2320 and 128<=y<1920 and 128<=x+dx<2320 and 128<=y+dy<1920 for x,y in points]
            result=large.preflow_support(points,(dx,dy))
            self.assertEqual(result["mask"],expected)
            self.assertEqual(result["fixed_support_count"],sum(expected))
            self.assertEqual(result["selected_indices"],np.flatnonzero(expected).tolist())

    def fixture_rows(self):
        rows=[]
        record=dict(dtype="uint8",shape=[2,2],sha256="a"*64)
        for index in large.PREVIOUS_INDICES:
            for shift in large.SHIFTS:
                rows.append(dict(case_id=large.case_id(index,shift),source_previous_index=index,
                    expected_shift_xy=list(shift),native_pixel_sha256=dict(current="b"*64),
                    capture=dict(pixels=dict(previous=dict(native=copy.deepcopy(record))),
                        proxy=dict(previous=copy.deepcopy(record),current=dict(sha256="c"*64)),
                        s16=dict(previous=copy.deepcopy(record)),
                        harris=dict(coordinates=copy.deepcopy(record),scores=copy.deepcopy(record)),
                        selection=dict(selected_indices=copy.deepcopy(record)))))
        rows.extend(dict(case_id=name) for name in large.CONTROL_NAMES)
        return rows

    def test_group_invariance_requires_eleven_rows_and_full_ninety_three(self):
        rows=self.fixture_rows()
        self.assertEqual(len(large.invariance(rows)),8)
        self.assertEqual(len(large.invariance(rows[:11],complete=False)),1)
        with self.assertRaises(ValueError):
            large.invariance(rows[:7],complete=False)
        with self.assertRaises(ValueError):
            large.invariance(rows[:-1])

    def test_changed_previous_features_fail_invariance(self):
        rows=self.fixture_rows()
        rows[4]["capture"]["harris"]["scores"]["sha256"]="f"*64
        with self.assertRaises(ValueError):
            large.invariance(rows)

    def manifest(self):
        hashes=dict(script_sha256="a"*64,tests_sha256="b"*64,plan_sha256="c"*64)
        value=dict(schema=large.PLAN_SCHEMA,input_workspace=str(large.INPUT_WORKSPACE),
            helper_sha256=large.HELPER_SHA,factor_helper_sha256=large.FACTOR_SHA,
            baseline_harness_sha256=large.HARNESS_SHA,baseline_journal_sha256=large.BASELINE_JOURNAL_SHA,
            reference_factor_result_sha256=large.REFERENCE_FACTOR_SHA,previous_indices=list(large.PREVIOUS_INDICES),
            shifts=[list(s) for s in large.SHIFTS],control_names=list(large.CONTROL_NAMES),
            control_seed=large.CONTROL_SEED,arm=large.ARM,motion_pair_calls=93,**hashes)
        return value,hashes

    def test_manifest_binds_larger_design_and_sources(self):
        value,hashes=self.manifest()
        large.validate_manifest(value,hashes)
        for key,bad in (("motion_pair_calls",61),("shifts",[[0,0]]),("script_sha256","f"*64),
                        ("schema","seaqr.aot.natural-shifts-plan.v1"),("previous_indices",[0]),
                        ("reference_factor_result_sha256","f"*64)):
            changed=copy.deepcopy(value);changed[key]=bad
            with self.assertRaises(ValueError):
                large.validate_manifest(changed,hashes)

    def test_scope_does_not_accept_prior_workspaces_or_parent_paths(self):
        p=Path("/tmp/seaqr_aot_large_shifts_20260928_A1b2C3")
        self.assertEqual(large.scope_path(p),p)
        for bad in ("/tmp","/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3",str(p)+"/../elsewhere",str(p)+"/child"):
            with self.assertRaises(ValueError):
                large.scope_path(bad)

    def test_no_overwrite_of_success_failure_or_compression(self):
        for name in large.OUTPUT_NAMES:
            with patch.object(large.os,"geteuid",return_value=1000),patch.object(Path,"is_dir",return_value=True),\
                    patch.object(Path,"is_symlink",return_value=False),patch.object(Path,"exists",lambda p:p.name==name):
                with self.assertRaises(ValueError):
                    large.workspace_guard("/tmp/seaqr_aot_large_shifts_20260928_A1b2C3")

    def test_root_never_runs(self):
        with patch.object(large.os,"geteuid",return_value=0),patch.object(Path,"is_dir",side_effect=AssertionError):
            with self.assertRaises(ValueError):
                large.workspace_guard("/tmp/seaqr_aot_large_shifts_20260928_A1b2C3")

if __name__=="__main__":
    unittest.main()
