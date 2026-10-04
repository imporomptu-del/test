"""Isolated saved-proposal experiment: change only eligible victim ordering.

This module does not run a replay, open media, change a tracker configuration,
patch a module/class, or claim causal detector accuracy. The caller must first
pass the frozen zero-tolerance baseline replay on the original Jetson runtime.
Saved-proposal comparison remains a shadow after the first changed causal
learning input; a causal full-pipeline replay is required before promotion.
"""
from __future__ import annotations

import hashlib
import heapq
import inspect
import marshal
from pathlib import Path
import re
import textwrap
from types import FunctionType

POLICY = "maturity_first_eligible_v1"
METHOD_SHA = "571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325"
TRACKER_SHA = "54e36d56d98eb4533661d58f921e5350c12805fcec7acc14c11258e5884c29ac"
ADAPTER_SHA = {
    "tracking_geometry_v20": "2021152f598082b88cabac762f12a7bb886def17a85b85cec57d1c0b31ad4e52",
    "tracking_batch_v27": "6ff824041e254bb7ada457263708e5210cb503f62d5691ee3af012564c7a4793",
    "tracking_stage_v28": "92b4e5a7b2be8556de434f1a216530d896e428e6c62e4879789ec7a9cc360975",
}


class MaturityFirstVictimIndex:
    """Same eligibility and dynamic cell quota; maturity first among eligible.

    The unchanged admission function supplies only unassigned, never-confirmed
    tracks with missed_windows > 0. Newborns are absent from this snapshot.
    Each take rechecks current occupancy, so a one-hit track in an ineligible
    cell cannot displace an eligible two-hit track. No maturity is protected
    absolutely: if only eligible two/three-hit tracks remain they can be taken.
    """

    def __init__(self, tracks, track_cells, replaceable):
        self.cells = {}
        for tid in replaceable:
            track = tracks[tid]
            self.cells.setdefault(track_cells[tid], []).append(
                (track.independent_confirmation_hits, -track.missed_windows, tid))
        for values in self.cells.values():
            heapq.heapify(values)

    def take(self, occupancy, minimum_exclusive):
        best = None
        for cell, values in self.cells.items():
            if occupancy[cell] <= minimum_exclusive:
                continue
            hits, negative_missed, tid = values[0]
            key = -hits, occupancy[cell], -negative_missed, -tid
            if best is None or key > best[0]:
                best = key, cell
        if best is None:
            return None
        cell = best[1]
        tid = heapq.heappop(self.cells[cell])[2]
        if not self.cells[cell]:
            del self.cells[cell]
        return tid


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(function):
    """Runtime-specific bytecode identity, not a cross-Python source identity."""
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def _module_binding(expected):
    path = Path(__file__).resolve()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
            and path.is_file() and not Path(__file__).is_symlink()
            and sha(path) == expected, "caller-bound candidate module differs")
    return dict(path=str(path), sha256=expected,
                class_name=MaturityFirstVictimIndex.__name__,
                class_source_sha256=hashlib.sha256(
                    inspect.getsource(MaturityFirstVictimIndex).encode()).hexdigest(),
                class_code_sha256={name: code_sha(getattr(MaturityFirstVictimIndex, name))
                                   for name in ("__init__", "take")})


def _verified_scope(method):
    import tracking_geometry_v20 as geometry
    import tracking_batch_v27 as batch
    import tracking_stage_v28 as stage
    from tiny_target.tracking.kalman import KalmanTrackManager

    for module in (geometry, batch, stage):
        require(sha(inspect.getfile(module)) == ADAPTER_SHA[module.__name__],
                "frozen adapter source differs")
    original = KalmanTrackManager.update
    require(sha(inspect.getfile(original)) == TRACKER_SHA,
            "original tracker must be installed before constructing candidate")
    require(isinstance(method, FunctionType), "generated baseline function required")
    scope = method.__globals__
    require(scope.get("update") is method
            and scope.get("_VictimIndex_v28") is stage.VictimIndex,
            "original v28 function and victim binding required")
    owner = getattr(scope.get("_prepare_v28"), "__self__", None)
    require(type(owner) is stage.TrackingStageV28
            and scope.get("_batch_v27") is owner.geometry
            and scope.get("_predict_all_v28") is stage.predict_all
            and scope.get("_record_values_v28") is stage.record_values,
            "original v28 helper bindings required")

    # Recompile the pinned transformation without loading/building a library or
    # mutating the caller's adapter. adapt() only binds/compiles at this point.
    reference_scope = dict(original.__globals__, _geometry_v20=owner.geometry.fallback)
    reference_source = textwrap.dedent(inspect.getsource(original).replace(geometry.OLD, geometry.NEW))
    exec(compile(reference_source, "<tracking_geometry_v20:exact>", "exec", dont_inherit=True), reference_scope)
    reference_stage = object.__new__(stage.TrackingStageV28)
    reference_stage.geometry = object.__new__(batch.BatchGeometryV27)
    reference = reference_stage.adapt(reference_scope["update"])
    require(reference_stage.transformed_sha256 == owner.transformed_sha256 == METHOD_SHA,
            "frozen generated tracking source differs")
    functions = {"update": method, "_births_v28": scope.get("_births_v28"),
                 "_record_v28": scope.get("_record_v28")}
    for name, function in functions.items():
        expected = reference if name == "update" else reference.__globals__[name]
        require(isinstance(function, FunctionType) and function.__globals__ is scope
                and function.__code__ == expected.__code__
                and function.__defaults__ == expected.__defaults__
                and function.__kwdefaults__ == expected.__kwdefaults__
                and function.__closure__ == expected.__closure__,
                "frozen generated function differs: " + name)
    return scope, functions


def _clone(function, scope):
    cloned = FunctionType(function.__code__, scope, function.__name__,
                          function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__dict__.update(function.__dict__)
    cloned.__qualname__ = function.__qualname__
    cloned.__module__ = function.__module__
    cloned.__doc__ = function.__doc__
    return cloned


def make_candidate_method(baseline_method, *, expected_module_sha256):
    """Return (isolated update, binding receipt); never mutate the baseline.

    Call after baseline.make_adapter(libraries), before patching manager.update.
    Use separate baseline and candidate adapter instances: their performance
    counters are mutable even though tracking arithmetic is unchanged. Existing
    runner input hashes, zero-tolerance baseline gates and configuration checks
    remain required; this hook does not replace them.
    """
    binding = _module_binding(expected_module_sha256)
    original_scope, functions = _verified_scope(baseline_method)
    scope = dict(original_scope)
    scope["_VictimIndex_v28"] = MaturityFirstVictimIndex
    clones = {name: _clone(function, scope) for name, function in functions.items()}
    scope.update(clones)
    scope["_admit_births"] = clones["_births_v28"]
    scope["_record"] = clones["_record_v28"]
    binding.update(
        schema="seaqr.tracker-maturity-candidate.binding.v1", policy=POLICY,
        victim_total_key=["-independent_confirmation_hits", "cell_occupancy",
                          "missed_windows", "-track_id"],
        unchanged_transformed_source_sha256=METHOD_SHA,
        frozen_tracker_sha256=TRACKER_SHA, frozen_adapter_sha256=dict(ADAPTER_SHA),
        unchanged_function_code_sha256={name: code_sha(fn) for name, fn in functions.items()},
        changed_global_bindings=["_VictimIndex_v28"],
        isolated_function_rebindings=["update", "_births_v28", "_admit_births", "_record_v28", "_record"],
        eligibility_changed=False, configuration_changed=False,
        original_scope_unchanged=True, source_media_opened=False,
        production_promotion=False, saved_proposals_shadow_only=True,
        requires_causal_rerun_after_learning_input_divergence=True)
    require(sha(binding["path"]) == expected_module_sha256, "candidate changed during binding")
    return clones["update"], binding
