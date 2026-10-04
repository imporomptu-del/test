#!/usr/bin/env python3
"""Validate SCRUM-75 per-resolution samples and build JSON/Markdown reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scrum-75.pyrlk-pva.v1"
COMBINED_SCHEMA_VERSION = "scrum-75.pyrlk-pva-combined.v1"
TICKET = "SCRUM-75"
TASK_BRANCH = "scrum-75-pva-pyrlk"
JETSON_CHECKOUT = "/home/serg/project/methods_test/skymove"
EXPECTED_RESOLUTIONS = ((1920, 1080), (3184, 2124))
METHODS = ("ofa", "pyrlk_pva")
TIMING_STAGES = (
    "prep_ms",
    "submit_host_ms",
    "accelerator_wait_ms",
    "submit_ms",
    "rlock_ms",
    "total_ms",
)


class ValidationError(RuntimeError):
    """Raised when a partial, mismatched, or misleading report is detected."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"Missing input report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"Top-level JSON must be an object: {path}")
    return value


def _stats_ms(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    xs = sorted(float(x) for x in samples)
    count = len(xs)

    def nearest_rank(fraction: float) -> float:
        return xs[max(0, math.ceil(fraction * count) - 1)]

    return {
        "count": count,
        "mean_ms": round(statistics.mean(xs), 6),
        "std_ms": round(statistics.pstdev(xs), 6) if count > 1 else 0.0,
        "min_ms": round(xs[0], 6),
        "p50_ms": round(nearest_rank(0.50), 6),
        "p95_ms": round(nearest_rank(0.95), 6),
        "max_ms": round(xs[-1], 6),
    }


def _stats_values(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    xs = sorted(float(x) for x in samples)
    count = len(xs)

    def nearest_rank(fraction: float) -> float:
        return xs[max(0, math.ceil(fraction * count) - 1)]

    return {
        "count": count,
        "mean": round(statistics.mean(xs), 6),
        "std": round(statistics.pstdev(xs), 6) if count > 1 else 0.0,
        "min": round(xs[0], 6),
        "p50": round(nearest_rank(0.50), 6),
        "p95": round(nearest_rank(0.95), 6),
        "max": round(xs[-1], 6),
    }


def _recompute_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        stage: _stats_ms([float(sample[stage]) for sample in samples])
        for stage in TIMING_STAGES
    }
    total_mean = result["total_ms"]["mean_ms"]
    result["fps"] = {
        "end_to_end": round(1000.0 / total_mean, 6) if total_mean > 0 else 0.0,
        "definition": "1000 / mean(total_ms)",
    }
    if samples and "detected_features" in samples[0]:
        result["features"] = {
            "detected": _stats_values(
                [float(sample["detected_features"]) for sample in samples]
            ),
            "tracked": _stats_values(
                [float(sample["tracked_features"]) for sample in samples]
            ),
            "lost": _stats_values(
                [float(sample["lost_features"]) for sample in samples]
            ),
            "loss_rate": _stats_values(
                [float(sample["feature_loss_rate"]) for sample in samples]
            ),
        }
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValidationError(f"{label} must be finite and non-negative: {number}")
    return number


def _validate_method(
    method_name: str,
    method: dict[str, Any],
    measured_pairs: list[dict[str, Any]],
    runs: int,
    label: str,
) -> None:
    samples = method.get("samples")
    _require(isinstance(samples, list), f"{label}.{method_name}.samples must be a list")
    _require(
        len(samples) == runs,
        f"{label}.{method_name} has {len(samples)} samples; expected {runs}",
    )
    expected_hashes = [pair["pair_sha256"] for pair in measured_pairs]
    observed_hashes: list[str] = []
    for index, sample in enumerate(samples):
        _require(
            isinstance(sample, dict),
            f"{label}.{method_name}.samples[{index}] must be an object",
        )
        observed_hashes.append(str(sample.get("pair_sha256", "")))
        stage_values = {
            stage: _finite_nonnegative(
                sample.get(stage), f"{label}.{method_name}[{index}].{stage}"
            )
            for stage in TIMING_STAGES
        }
        submit_parts = (
            stage_values["submit_host_ms"] + stage_values["accelerator_wait_ms"]
        )
        _require(
            abs(submit_parts - stage_values["submit_ms"]) <= 0.05,
            f"{label}.{method_name}[{index}] submit timing does not reconcile",
        )
        accounted = (
            stage_values["prep_ms"]
            + stage_values["submit_ms"]
            + stage_values["rlock_ms"]
        )
        _require(
            abs(accounted - stage_values["total_ms"]) <= 0.05,
            f"{label}.{method_name}[{index}] total timing does not reconcile",
        )
        if method_name == "pyrlk_pva":
            detected = int(sample.get("detected_features", -1))
            tracked = int(sample.get("tracked_features", -1))
            lost = int(sample.get("lost_features", -1))
            loss_rate = _finite_nonnegative(
                sample.get("feature_loss_rate"),
                f"{label}.{method_name}[{index}].feature_loss_rate",
            )
            _require(detected > 0, f"{label} PyrLK detected no features")
            _require(
                tracked >= 0 and lost >= 0 and tracked + lost == detected,
                f"{label} PyrLK feature counts do not reconcile",
            )
            _require(loss_rate <= 1.0, f"{label} feature loss rate exceeds one")
            _require(
                abs(loss_rate - (lost / detected)) <= 1e-8,
                f"{label} feature loss rate does not match counts",
            )
    _require(
        observed_hashes == expected_hashes,
        f"{label}.{method_name} pair hashes do not match frame_set",
    )
    recomputed = _recompute_summary(samples)
    _require(
        method.get("summary") == recomputed,
        f"{label}.{method_name} stored summary does not match raw samples",
    )
    backends = method.get("backends", {})
    _require(
        backends.get("cpu_fallback") is False,
        f"{label}.{method_name} did not prove CPU fallback is disabled",
    )


def _validate_resolution(
    data: dict[str, Any], expected: tuple[int, int], label: str,
) -> None:
    _require(data.get("schema_version") == SCHEMA_VERSION, f"{label}: wrong schema")
    _require(
        data.get("artifact_type") == "resolution_samples",
        f"{label}: wrong artifact type",
    )
    _require(data.get("ticket") == TICKET, f"{label}: wrong ticket")
    config = data.get("config", {})
    resolution = config.get("resolution", {})
    observed = (resolution.get("width"), resolution.get("height"))
    _require(observed == expected, f"{label}: resolution {observed}, expected {expected}")
    runs = config.get("runs")
    warmup = config.get("warmup")
    _require(isinstance(runs, int) and runs > 0, f"{label}: invalid runs")
    _require(isinstance(warmup, int) and warmup >= 0, f"{label}: invalid warmup")
    _require(config.get("frames") == runs + warmup + 1, f"{label}: frame count mismatch")
    _require(config.get("ofa", {}).get("gridsize") == 4, f"{label}: OFA grid is not 4")
    _require(
        config.get("pyrlk", {}).get("pyramid_levels") == 4,
        f"{label}: PyrLK levels are not 4",
    )
    _require(
        config.get("pyrlk", {}).get("harris_input_format") == "S16",
        f"{label}: PVA Harris input is not S16",
    )
    _require(
        config.get("pyrlk", {}).get("harris_min_nms_distance") == 8,
        f"{label}: PVA Harris NMS distance is not 8",
    )
    calibration = config.get("pyrlk", {}).get("harris_calibration", {})
    _require(calibration.get("passed") is True, f"{label}: Harris calibration failed")
    selected_strength = _finite_nonnegative(
        calibration.get("selected_strength"),
        f"{label}: selected Harris strength",
    )
    _require(selected_strength > 0, f"{label}: Harris strength must be positive")
    _require(
        config.get("pyrlk", {}).get("harris_strength_effective")
        == selected_strength,
        f"{label}: effective Harris strength does not match calibration",
    )
    attempts = calibration.get("attempts")
    _require(isinstance(attempts, list) and attempts, f"{label}: calibration attempts missing")
    selected_counts = attempts[-1].get("selected_feature_counts", {})
    _require(
        selected_counts.get("min", 0)
        >= calibration.get("minimum_required_features_per_pair", 1),
        f"{label}: calibrated feature minimum was not met",
    )
    selected_pair_window = calibration.get("selected_pair_window", {})
    selected_pair_count = (
        selected_pair_window.get("stop_exclusive_in_capture", 0)
        - selected_pair_window.get("start_in_capture", 0)
    )
    _require(
        selected_pair_count == runs + warmup,
        f"{label}: selected Harris pair window has the wrong size",
    )
    expected_pyramid = "PVA" if expected == (1920, 1080) else "CUDA"
    _require(
        config.get("pyrlk", {}).get("pyramid_backend") == expected_pyramid,
        f"{label}: expected {expected_pyramid} pyramid backend",
    )
    _require(
        config.get("pyrlk", {}).get("harris_backend") == "PVA"
        and config.get("pyrlk", {}).get("optical_flow_backend") == "PVA",
        f"{label}: Harris/PyrLK backend is not explicitly PVA",
    )

    frame_set = data.get("frame_set", {})
    frames = frame_set.get("frames")
    pairs = frame_set.get("pairs")
    _require(isinstance(frames, list), f"{label}: frame hashes missing")
    _require(isinstance(pairs, list), f"{label}: pair hashes missing")
    _require(len(frames) == runs + warmup + 1, f"{label}: frame hash count mismatch")
    _require(len(pairs) == runs + warmup, f"{label}: pair hash count mismatch")
    _require(frame_set.get("pixels_persisted") is False, f"{label}: raw pixels persisted")
    acquisition = data.get("acquisition", {})
    captured_frames = acquisition.get("captured_frames")
    selected_frames = acquisition.get("selected_frames")
    discarded_frames = acquisition.get("discarded_capture_frames")
    _require(
        isinstance(captured_frames, int) and captured_frames >= len(frames),
        f"{label}: invalid captured frame count",
    )
    _require(selected_frames == len(frames), f"{label}: selected frame count mismatch")
    _require(
        discarded_frames == captured_frames - selected_frames,
        f"{label}: discarded frame count mismatch",
    )
    measured_pairs = [pair for pair in pairs if pair.get("phase") == "measured"]
    _require(len(measured_pairs) == runs, f"{label}: measured pair count mismatch")

    methods = data.get("methods")
    _require(isinstance(methods, dict), f"{label}: methods missing")
    _require(set(methods) == set(METHODS), f"{label}: unexpected method set")
    for method_name in METHODS:
        _validate_method(
            method_name,
            methods[method_name],
            measured_pairs,
            runs,
            label,
        )
    pva_hashes = [
        sample["pair_sha256"] for sample in methods["pyrlk_pva"]["samples"]
    ]
    ofa_hashes = [sample["pair_sha256"] for sample in methods["ofa"]["samples"]]
    _require(pva_hashes == ofa_hashes, f"{label}: OFA/PyrLK inputs differ")
    fairness = data.get("fairness", {})
    _require(
        fairness.get("same_in_memory_frames") is True
        and fairness.get("same_measured_pair_hashes") is True
        and fairness.get("input_hashes_unchanged_after_methods") is True,
        f"{label}: fairness assertions are incomplete",
    )
    _require(
        fairness.get("feature_window_selected_before_timing") is True
        and fairness.get("selection_applied_equally_to_both_methods") is True,
        f"{label}: feature-window fairness assertions are incomplete",
    )
    validation = data.get("validation", {})
    _require(validation.get("fallback_used") is False, f"{label}: fallback detected")
    _require(
        validation.get("sample_counts_match_runs") is True,
        f"{label}: sample count validation failed",
    )
    smoke = validation.get("backend_smoke", {})
    _require(smoke.get("passed") is True, f"{label}: backend smoke failed")
    support = smoke.get("pyrlk_backend_support", {})
    _require(
        support.get("cuda_smoke_validated") is True
        and support.get("pva_smoke_validated") is True
        and support.get("cpu_used") is False,
        f"{label}: CUDA/PVA support validation is incomplete",
    )


def _comparison_rows(data_by_resolution: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resolution_key in ("1920x1080", "3184x2124"):
        data = data_by_resolution[resolution_key]
        for method_name in METHODS:
            method = data["methods"][method_name]
            summary = method["summary"]
            row: dict[str, Any] = {
                "resolution": resolution_key,
                "method": method_name,
                "algorithm_kind": method["algorithm_kind"],
                "backends": method["backends"],
                "prep_ms": summary["prep_ms"],
                "submit_host_ms": summary["submit_host_ms"],
                "accelerator_wait_ms": summary["accelerator_wait_ms"],
                "submit_ms": summary["submit_ms"],
                "rlock_ms": summary["rlock_ms"],
                "total_ms": summary["total_ms"],
                "fps": summary["fps"]["end_to_end"],
            }
            if method_name == "pyrlk_pva":
                row["features"] = summary["features"]
            rows.append(row)
    return rows


def _validate_cross_resolution(
    first: dict[str, Any], second: dict[str, Any],
) -> None:
    _require(first.get("run_id") == second.get("run_id"), "run_id mismatch")
    first_config = first["config"]
    second_config = second["config"]
    for field in ("source", "runs", "warmup"):
        _require(
            first_config.get(field) == second_config.get(field),
            f"cross-resolution config mismatch: {field}",
        )
    _require(
        first.get("git", {}).get("commit") == second.get("git", {}).get("commit"),
        "Git commit mismatch between resolutions",
    )
    _require(
        first.get("host", {}).get("hostname")
        == second.get("host", {}).get("hostname"),
        "hostname mismatch between resolutions",
    )


def _format_mean(method: dict[str, Any], stage: str) -> float:
    return float(method["summary"][stage]["mean_ms"])


def _recorded_option(report: dict[str, Any], option: str, fallback: str) -> str:
    """Return a CLI option captured in raw-sample provenance."""
    argv = report.get("argv", [])
    if not isinstance(argv, list):
        return fallback
    try:
        index = argv.index(option)
    except ValueError:
        return fallback
    if index + 1 >= len(argv):
        return fallback
    return str(argv[index + 1])


def _render_markdown(combined: dict[str, Any]) -> str:
    resolutions = combined["resolutions"]
    first = resolutions["1920x1080"]
    source = first["config"]["source"]
    runs = first["config"]["runs"]
    warmup = first["config"]["warmup"]
    requested_strength = first["config"]["pyrlk"]["harris_strength_requested"]
    if source == "live_skyeye62am":
        source_argument = "--camera"
        output_pattern = "results/ofa/pyrlk_cam_run_<UTC_TIMESTAMP>/"
        source_arguments = [
            source_argument,
            "--camera-name",
            first["acquisition"].get("requested_name", "SkyEye"),
        ]
        reproducibility_note = (
            "A rerun reproduces the method and configuration, but not the exact live "
            "camera frames. The raw sample JSON preserves the measured frame hashes."
        )
    elif source == "deterministic_synthetic":
        source_argument = "--synthetic"
        output_pattern = "results/ofa/pyrlk_synthetic_run_<UTC_TIMESTAMP>/"
        source_arguments = [
            source_argument,
            "--seed",
            _recorded_option(first, "--seed", "75"),
        ]
        reproducibility_note = (
            "The synthetic input is deterministic when the recorded seed and Git "
            "commit are used."
        )
    else:
        raise ValidationError(f"Unsupported report source for reproduction: {source}")
    matrix_command = " ".join(
        shlex.quote(str(value))
        for value in (
            "./run_pyrlk_pva_matrix.sh",
            *source_arguments,
            "--runs",
            runs,
            "--warmup",
            warmup,
            "--harris-strength",
            requested_strength,
        )
    )
    lines = [
        "# SCRUM-75 — VPI PVA PyrLK vs OFA",
        "",
        f"- Generated (UTC): `{combined['generated_at_utc']}`",
        f"- Run ID: `{combined['run_id']}`",
        f"- Source: `{source}`",
        f"- Timed pairs per resolution: **{runs}** (warmup **{warmup}**)",
        f"- Host: `{first['host']['hostname']}`",
        f"- Git commit: `{first['git']['commit']}`",
        f"- VPI: `{first['host']['vpi']}`",
        "- OFA gridsize: **4**; PyrLK pyramid levels: **4**",
        "",
        "## Backend boundaries",
        "",
        "| resolution | OFA route | PyrLK preparation | PyrLK flow |",
        "|---|---|---|---|",
        "| 1920×1080 | CUDA pyramid → VIC block-linear → OFA | "
        "CUDA U8→S16 → PVA Harris; PVA pyramid | **PVA PyrLK** |",
        "| 3184×2124 | CUDA pyramid → VIC block-linear → OFA | "
        "CUDA U8→S16 → PVA Harris; **CUDA pyramid** | **PVA PyrLK** |",
        "",
        "VPI 3.2.4 exposes PyrLK on CPU, CUDA, and PVA. Synthetic smoke tests "
        "validated CUDA and PVA on this host; CPU was neither tested nor used. "
        "The 3184×2124 pyramid uses CUDA because PVA pyramid construction is "
        "limited to 3264×2048, while Harris and PyrLK remain on PVA.",
        "",
        "## Harris calibration",
        "",
        "A single fixed Harris strength and one continuous feature-bearing frame "
        "window are selected per resolution before timed work. Calibration uses "
        "S16 input and PVA with NMS distance 8, and is excluded from method timing. "
        "Both methods consume the identical selected window.",
        "",
        "| resolution | requested | selected | selected minimum | capture window | attempts |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for resolution_key, label in (
        ("1920x1080", "1920×1080"),
        ("3184x2124", "3184×2124"),
    ):
        calibration = resolutions[resolution_key]["config"]["pyrlk"]["harris_calibration"]
        selected_counts = calibration["attempts"][-1]["selected_feature_counts"]
        acquisition = resolutions[resolution_key]["acquisition"]
        window = calibration["selected_frame_window"]
        lines.append(
            f"| {label} | {calibration['requested_strength']:g} | "
            f"**{calibration['selected_strength']:g}** | "
            f"{selected_counts['min']:.0f} | "
            f"[{window['start_in_capture']}, "
            f"{window['stop_exclusive_in_capture']}) of "
            f"{acquisition['captured_frames']} frames | "
            f"{len(calibration['attempts'])} |"
        )
    lines.extend([
        "",
        "## OFA vs PyrLK comparison",
        "",
        "| resolution | method | prep mean | submit mean | rlock mean | "
        "total mean | end-to-end fps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for resolution_key, label in (
        ("1920x1080", "1920×1080"),
        ("3184x2124", "3184×2124"),
    ):
        for method_name, method_label in (("ofa", "OFA dense"), ("pyrlk_pva", "PVA PyrLK sparse")):
            method = resolutions[resolution_key]["methods"][method_name]
            lines.append(
                f"| {label} | **{method_label}** | "
                f"{_format_mean(method, 'prep_ms'):.3f} | "
                f"{_format_mean(method, 'submit_ms'):.3f} | "
                f"{_format_mean(method, 'rlock_ms'):.3f} | "
                f"{_format_mean(method, 'total_ms'):.3f} | "
                f"{method['summary']['fps']['end_to_end']:.2f} |"
            )

    lines.extend([
        "",
        "`submit_ms` is call-start through explicit stream synchronization. "
        "It includes dispatch and scheduling overhead and is not a pure kernel timer.",
        "",
        "## Stage distributions",
        "",
        "| resolution | method | stage | mean | std | p50 | p95 | max |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for resolution_key, label in (
        ("1920x1080", "1920×1080"),
        ("3184x2124", "3184×2124"),
    ):
        for method_name, method_label in (("ofa", "OFA"), ("pyrlk_pva", "PyrLK")):
            summary = resolutions[resolution_key]["methods"][method_name]["summary"]
            for stage in ("prep_ms", "submit_ms", "rlock_ms", "total_ms"):
                stats = summary[stage]
                lines.append(
                    f"| {label} | {method_label} | `{stage}` | "
                    f"{stats['mean_ms']:.3f} | {stats['std_ms']:.3f} | "
                    f"{stats['p50_ms']:.3f} | {stats['p95_ms']:.3f} | "
                    f"{stats['max_ms']:.3f} |"
                )

    lines.extend([
        "",
        "## PyrLK feature tracking",
        "",
        "| resolution | detected mean | tracked mean | lost mean | loss rate mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for resolution_key, label in (
        ("1920x1080", "1920×1080"),
        ("3184x2124", "3184×2124"),
    ):
        features = resolutions[resolution_key]["methods"]["pyrlk_pva"]["summary"]["features"]
        lines.append(
            f"| {label} | {features['detected']['mean']:.1f} | "
            f"{features['tracked']['mean']:.1f} | {features['lost']['mean']:.1f} | "
            f"{features['loss_rate']['mean'] * 100.0:.3f}% |"
        )

    lines.extend([
        "",
        "## Timing definitions",
        "",
        "- **prep_ms:** format conversion, pyramid construction, feature detection "
        "where applicable, payload creation, then explicit synchronization.",
        "- **submit_host_ms:** latency until the Python optical-flow call returns.",
        "- **accelerator_wait_ms:** call return through stream synchronization.",
        "- **submit_ms:** `submit_host_ms + accelerator_wait_ms`.",
        "- **rlock_ms:** post-sync CPU mapping and NumPy copy only.",
        "- **total_ms:** directly measured preparation start through completed readback.",
        "- **fps:** `1000 / mean(total_ms)`.",
        "",
        "## Fairness and validation",
        "",
    ])
    for resolution_key, label in (
        ("1920x1080", "1920×1080"),
        ("3184x2124", "3184×2124"),
    ):
        data = resolutions[resolution_key]
        lines.extend([
            f"- **{label}:** one capture/generation, identical measured pair "
            f"hashes, frame-set digest `{data['frame_set']['frame_set_sha256']}`, "
            "and unchanged input hashes after both methods.",
        ])
    lines.extend([
        "- All accelerator operations name explicit backends; fallback is disabled.",
        "- Summary statistics were recomputed from raw per-pair samples before this "
        "report was emitted.",
        "- Camera acquisition/wait/crop/copy is shared and excluded from method totals.",
        "",
        "## Caveats",
        "",
        "- OFA is dense optical flow; PyrLK tracks sparse Harris features. The table "
        "compares pipeline timing and resource behavior, not accuracy equivalence.",
        "- OFA gridsize 4 and PyrLK pyramid levels 4 are unrelated parameters.",
        "- Wall-clock submit-to-sync includes dispatch, scheduling, and synchronization "
        "overhead; it is not a hardware-kernel profiler measurement.",
        "- Method order is recorded in each per-resolution JSON to expose possible "
        "thermal or DVFS ordering effects.",
        "- SCRUM-75 contains stale ASI676 wording; the verified live device and this "
        "report use the SkyEye62AM named by the repository and handoff.",
        "",
        "## Reproduction",
        "",
        f"- Jetson checkout: `{JETSON_CHECKOUT}`",
        f"- Task branch: `{TASK_BRANCH}`",
        "- Use the Git commit recorded at the top of this report.",
        "- Scripts:",
        f"  - `{JETSON_CHECKOUT}/bench_pyrlk_pva.py` — captures or generates shared "
        "frames and records raw per-pair samples.",
        f"  - `{JETSON_CHECKOUT}/run_pyrlk_pva_matrix.sh` — enforces exclusive camera "
        "ownership and runs both resolutions sequentially.",
        f"  - `{JETSON_CHECKOUT}/run_pyrlk_pva_summary.py` — validates the raw samples, "
        "recomputes statistics, and renders the JSON and Markdown summaries.",
        "",
        "Run from `agxorin1`:",
        "",
        "```bash",
        f"cd {JETSON_CHECKOUT}",
        "python3 bench_pyrlk_pva.py --check-camera-ownership",
        matrix_command,
        "```",
        "",
        f"New reports are written under `{output_pattern}`. Existing report files are "
        "never overwritten.",
        "",
        "The matrix script first creates one raw JSON sample file per resolution. It "
        "then invokes `run_pyrlk_pva_summary.py`, which validates both inputs, checks "
        "shared-frame hashes and backend evidence, recomputes all summary statistics, "
        "renders this document in `_render_markdown()`, and exclusively creates "
        "`flow_report.json` and `flow_report.md`.",
        "",
        reproducibility_note,
        "",
        "## Artifacts",
        "",
        "- `flow_report.json` — combined validated machine-readable report",
        "- `flow_report.md` — this report",
        "- `samples_1920x1080.json` — raw per-pair samples and frame hashes",
        "- `samples_3184x2124.json` — raw per-pair samples and frame hashes",
        "",
    ])
    return "\n".join(lines)


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError as exc:
        raise ValidationError(f"Refusing to overwrite existing output: {path}") from exc


def build_combined(
    input_1080: Path,
    input_full: Path,
) -> dict[str, Any]:
    first = _load_json(input_1080)
    second = _load_json(input_full)
    _validate_resolution(first, EXPECTED_RESOLUTIONS[0], str(input_1080))
    _validate_resolution(second, EXPECTED_RESOLUTIONS[1], str(input_full))
    _validate_cross_resolution(first, second)
    resolutions = {"1920x1080": first, "3184x2124": second}
    return {
        "schema_version": COMBINED_SCHEMA_VERSION,
        "artifact_type": "combined_flow_report",
        "ticket": TICKET,
        "run_id": first["run_id"],
        "generated_at_utc": _utc_now(),
        "input_artifacts": [
            {"name": input_1080.name, "sha256": _sha256_file(input_1080)},
            {"name": input_full.name, "sha256": _sha256_file(input_full)},
        ],
        "resolutions": resolutions,
        "comparison_rows": _comparison_rows(resolutions),
        "validation": {
            "schema_valid": True,
            "required_resolutions_present": True,
            "raw_samples_recomputed": True,
            "same_pair_hashes_per_resolution": True,
            "explicit_backends_only": True,
            "cpu_fallback_used": False,
            "cuda_and_pva_pyrlk_smoke_validated": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate two SCRUM-75 sample files and build flow_report.json/md",
    )
    parser.add_argument("--input-1080", type=Path, required=True)
    parser.add_argument("--input-full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs without creating report files",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")

    combined = build_combined(args.input_1080, args.input_full)
    if args.check_only:
        print(
            f"Validated {args.input_1080} and {args.input_full}: "
            f"run_id={combined['run_id']}"
        )
        return

    json_path = args.output_dir / "flow_report.json"
    markdown_path = args.output_dir / "flow_report.md"
    if json_path.exists() or markdown_path.exists():
        raise ValidationError(
            f"Refusing to overwrite report in existing output: {args.output_dir}"
        )
    encoded = json.dumps(
        combined,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    markdown = _render_markdown(combined)
    _write_exclusive(json_path, encoded)
    _write_exclusive(markdown_path, markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
