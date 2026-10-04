"""Repeat the unchanged V47 mathematical audit in a separately frozen V48 run.

Only output-root and dependency-inventory globals differ in the isolated run
function. Numerical cases, tests, tolerances, audit functions and solver are
the original V47 objects. No process-global monkeypatch or real-data access.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

import audit_accuracy_v47_math as original

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v48_20260926"


def dependencies():
    return sorted(set(original._dependencies()+[
        Path(__file__).resolve(), ROOT/"tests/unit/test_accuracy_v48_math_audit.py",
        ROOT/"docs/accuracy_v48_plan.md"]))


def run(output, *, execute=False):
    environment = dict(original.run.__globals__)
    environment.update(OUTPUT_ROOT=OUTPUT_ROOT, _dependencies=dependencies)
    isolated = FunctionType(original.run.__code__, environment,
                           "v48_unchanged_math_audit", original.run.__defaults__,
                           original.run.__closure__)
    isolated.__kwdefaults__ = dict(original.run.__kwdefaults__ or {})
    return isolated(output, execute=execute)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    result=run(args.output,execute=args.execute)
    print(json.dumps(dict(passed=result["passed"],issues=result["issues"],
                         realizations=result["matrix"]["realizations"]),indent=2))
