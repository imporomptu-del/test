"""Versioned datasets, early synthetic injection, and evaluation metrics."""

from .core import (
    DatasetEntry,
    DatasetInventory,
    EvaluationError,
    SyntheticInjectionSpec,
    SyntheticInjector,
    SyntheticTarget,
    ThresholdAccumulator,
    latency_summary,
    load_dataset_inventory,
    load_injection_spec,
    match_candidates,
    transformed_target_truth,
)

__all__ = [
    "DatasetEntry",
    "DatasetInventory",
    "EvaluationError",
    "SyntheticInjectionSpec",
    "SyntheticInjector",
    "SyntheticTarget",
    "ThresholdAccumulator",
    "latency_summary",
    "load_dataset_inventory",
    "load_injection_spec",
    "match_candidates",
    "transformed_target_truth",
]
