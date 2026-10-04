#!/usr/bin/env python3
"""SCRUM-75: fair OFA vs VPI PVA PyrLK benchmark on shared frame pairs.

Camera access is deliberately lazy and requires an ownership check performed by
``run_pyrlk_pva_matrix.sh``.  Synthetic mode exercises the same accelerator
paths without importing the camera SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


SKYMOVE = Path(__file__).resolve().parent
SCHEMA_VERSION = "scrum-75.pyrlk-pva.v1"
TICKET = "SCRUM-75"
VPI_REQUIRED_PREFIX = "3.2."
OFA_GRIDSIZE = 4
PYRAMID_LEVELS = 4
PYRAMID_SCALE = 0.5
PVA_PYRAMID_MAX_WIDTH = 3264
PVA_PYRAMID_MAX_HEIGHT = 2048
MIN_HARRIS_FEATURES_PER_PAIR = 8
LIVE_PAIR_RESERVE = 32
CAMERA_USB_ID = "0547:1469"
SUPPORTED_RESOLUTIONS = {(1920, 1080), (3184, 2124)}
TIMING_STAGES = (
    "prep_ms",
    "submit_host_ms",
    "accelerator_wait_ms",
    "submit_ms",
    "rlock_ms",
    "total_ms",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_numpy() -> Any:
    global np
    try:
        import numpy as numpy_module
    except Exception as exc:
        raise RuntimeError(
            "NumPy is unavailable in the active environment; no dependency "
            "installation was attempted"
        ) from exc
    np = numpy_module
    return numpy_module


def _run_readonly(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=SKYMOVE,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return subprocess.CompletedProcess(
            command,
            127 if isinstance(exc, FileNotFoundError) else 126,
            stdout="",
            stderr=f"{command[0]} is unavailable: {exc}",
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=f"{command[0]} timed out",
        )


def _git_metadata() -> dict[str, Any]:
    def value(*args: str) -> str:
        result = _run_readonly(["git", *args])
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    tracked_status = value("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": value("rev-parse", "HEAD"),
        "branch": value("branch", "--show-current"),
        "upstream": value(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}",
        ),
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
    }


def _host_metadata(vpi: Any) -> dict[str, Any]:
    try:
        import cv2

        opencv_version = cv2.__version__
    except Exception as exc:  # pragma: no cover - environment metadata only
        opencv_version = f"unavailable: {exc}"

    l4t_path = Path("/etc/nv_tegra_release")
    l4t = l4t_path.read_text(encoding="utf-8").strip() if l4t_path.exists() else ""

    def command_output(command: list[str]) -> str:
        result = _run_readonly(command)
        if result.returncode == 0:
            return result.stdout.strip()
        return f"unavailable (exit {result.returncode}): {result.stderr.strip()}"

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "opencv": opencv_version,
        "vpi": getattr(vpi, "__version__", "unknown"),
        "vpi_backend_constants": [
            name
            for name in ("PVA", "OFA", "CUDA", "VIC")
            if hasattr(vpi.Backend, name)
        ],
        "l4t": l4t,
        "nvpmodel_query": command_output(["nvpmodel", "-q"]),
        "jetson_clocks_show": command_output(["jetson_clocks", "--show"]),
    }


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
        index = max(0, math.ceil(fraction * count) - 1)
        return xs[index]

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
        index = max(0, math.ceil(fraction * count) - 1)
        return xs[index]

    return {
        "count": count,
        "mean": round(statistics.mean(xs), 6),
        "std": round(statistics.pstdev(xs), 6) if count > 1 else 0.0,
        "min": round(xs[0], 6),
        "p50": round(nearest_rank(0.50), 6),
        "p95": round(nearest_rank(0.95), 6),
        "max": round(xs[-1], 6),
    }


def _method_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        stage: _stats_ms([float(sample[stage]) for sample in samples])
        for stage in TIMING_STAGES
    }
    total_mean = summary["total_ms"]["mean_ms"]
    summary["fps"] = {
        "end_to_end": round(1000.0 / total_mean, 6) if total_mean > 0 else 0.0,
        "definition": "1000 / mean(total_ms)",
    }
    if samples and "detected_features" in samples[0]:
        summary["features"] = {
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
    return summary


def _frame_sha256(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"skymove-gray-frame-v1\0")
    digest.update(str(frame.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{frame.shape[0]}x{frame.shape[1]}".encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(np.ascontiguousarray(frame)).cast("B"))
    return digest.hexdigest()


def _pair_sha256(previous_hash: str, current_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"skymove-gray-pair-v1\0")
    digest.update(previous_hash.encode("ascii"))
    digest.update(b"\0")
    digest.update(current_hash.encode("ascii"))
    return digest.hexdigest()


def _build_frame_set(
    frames: list[np.ndarray], warmup: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hashes = [_frame_sha256(frame) for frame in frames]
    pairs: list[dict[str, Any]] = []
    for index in range(len(frames) - 1):
        pairs.append({
            "pair_index": index,
            "previous_frame_index": index,
            "current_frame_index": index + 1,
            "previous_sha256": hashes[index],
            "current_sha256": hashes[index + 1],
            "pair_sha256": _pair_sha256(hashes[index], hashes[index + 1]),
            "phase": "warmup" if index < warmup else "measured",
        })

    frame_set_digest = hashlib.sha256(
        "\n".join(hashes).encode("ascii")
    ).hexdigest()
    return ({
        "format": "GRAY8/U8",
        "dtype": str(frames[0].dtype),
        "shape": list(frames[0].shape),
        "hash_algorithm": "SHA-256 over schema tag, dtype, shape, and pixels",
        "frame_set_sha256": frame_set_digest,
        "frames": [
            {"frame_index": index, "sha256": digest}
            for index, digest in enumerate(hashes)
        ],
        "pairs": pairs,
        "pixels_persisted": False,
    }, pairs)


def _generate_synthetic_frames(
    width: int, height: int, count: int, seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    rows = (np.arange(height, dtype=np.uint16) // 16)[:, None]
    cols = (np.arange(width, dtype=np.uint16) // 16)[None, :]
    checker = (((rows + cols) & 1).astype(np.uint8) * 180) + 32
    noise = rng.integers(0, 16, size=(height, width), dtype=np.uint8)
    base = np.ascontiguousarray(checker + noise, dtype=np.uint8)
    frames = [
        np.ascontiguousarray(
            np.roll(base, shift=(2 * index, 3 * index), axis=(0, 1)),
            dtype=np.uint8,
        )
        for index in range(count)
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return frames, {
        "kind": "deterministic_checkerboard_translation",
        "seed": seed,
        "translation_per_frame_pixels": {"x": 3, "y": 2},
        "generation_ms": round(elapsed_ms, 6),
        "included_in_method_timing": False,
    }


def _camera_owner_lines() -> list[str]:
    """Return PID/command evidence for known camera owners; never kills anything."""
    current_pid = os.getpid()
    owners: dict[int, str] = {}
    process_result = _run_readonly(
        ["ps", "-eo", "pid=,ppid=,user=,comm=,args="]
    )
    if process_result.returncode != 0:
        raise RuntimeError(
            "Cannot confirm camera ownership because process inspection failed: "
            f"{process_result.stderr.strip()}"
        )
    process_rows: list[tuple[int, int, str]] = []
    parent_by_pid: dict[int, int] = {}
    for line in process_result.stdout.splitlines():
        fields = line.strip().split(maxsplit=4)
        if len(fields) < 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid = int(fields[0])
        parent_pid = int(fields[1])
        process_rows.append((pid, parent_pid, line.strip()))
        parent_by_pid[pid] = parent_pid

    excluded_pids = {current_pid}
    cursor = current_pid
    while cursor in parent_by_pid:
        parent_pid = parent_by_pid[cursor]
        if parent_pid <= 0 or parent_pid in excluded_pids:
            break
        excluded_pids.add(parent_pid)
        cursor = parent_pid

    pattern = re.compile(
        r"ToupLite|toupcam|bench_ofa_pair\.py|bench_pyrlk_pva\.py|"
        r"camera_bench\.py|gst-launch|v4l2-ctl",
        re.IGNORECASE,
    )
    for pid, _parent_pid, line in process_rows:
        if pid not in excluded_pids and pattern.search(line):
            owners[pid] = line

    lsusb = _run_readonly(["lsusb", "-d", CAMERA_USB_ID])
    if lsusb.returncode >= 126:
        raise RuntimeError(
            "Cannot confirm camera ownership because lsusb is unavailable: "
            f"{lsusb.stderr.strip()}"
        )
    match = re.search(r"Bus\s+(\d+)\s+Device\s+(\d+):", lsusb.stdout)
    if match:
        device_path = f"/dev/bus/usb/{match.group(1)}/{match.group(2)}"
        fuser = _run_readonly(["fuser", device_path])
        if fuser.returncode >= 126:
            raise RuntimeError(
                "Cannot confirm camera ownership because fuser is unavailable: "
                f"{fuser.stderr.strip()}"
            )
        for token in fuser.stdout.split():
            token = token.rstrip("c")
            if not token.isdigit():
                continue
            pid = int(token)
            if pid in excluded_pids or pid in owners:
                continue
            details = _run_readonly(
                ["ps", "-p", str(pid), "-o", "pid=,user=,comm=,args="]
            ).stdout.strip()
            owners[pid] = details or f"PID {pid} owns {device_path}"
    return [owners[pid] for pid in sorted(owners)]


def _assert_camera_free() -> None:
    owners = _camera_owner_lines()
    if owners:
        evidence = "\n".join(f"  {line}" for line in owners)
        raise RuntimeError(
            "Camera is occupied; refusing to open it. Owning process(es):\n"
            f"{evidence}"
        )


def _capture_camera_frames(
    args: argparse.Namespace, count: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    _assert_camera_free()
    sys.path.insert(0, str(SKYMOVE))
    from camera_bench import capture_grays, open_bench_camera

    session = open_bench_camera(
        args.camera_id,
        args.camera_name,
        roi_w=args.width,
        roi_h=args.height,
        resolution_index=args.resolution_index,
    )
    try:
        if (session.out_w, session.out_h) != (args.width, args.height):
            raise RuntimeError(
                "Camera returned unexpected dimensions: "
                f"{session.out_w}x{session.out_h}, expected "
                f"{args.width}x{args.height}"
            )
        frames, wait_crop_copy_ms = capture_grays(session, count)
        return frames, {
            "kind": "live_skyeye62am",
            "camera_model": "SkyEye62AM",
            "camera_index": session.cam_idx,
            "sensor_dimensions": [session.sensor_w, session.sensor_h],
            "output_dimensions": [session.out_w, session.out_h],
            "requested_name": args.camera_name,
            "resolution_index": args.resolution_index,
            "wait_crop_copy_ms": _stats_ms(wait_crop_copy_ms),
            "timing_note": (
                "Includes queue wait/frame acquisition plus crop/copy; excluded "
                "from both optical-flow method totals."
            ),
            "included_in_method_timing": False,
        }
    finally:
        session.close()


def _pyramid_backend_name(width: int, height: int) -> str:
    if width <= PVA_PYRAMID_MAX_WIDTH and height <= PVA_PYRAMID_MAX_HEIGHT:
        return "PVA"
    return "CUDA"


def _harris_strength_candidates(requested: float) -> list[float]:
    candidates = [requested]
    for candidate in (5.0, 1.0, 0.1, 0.01, 0.001, 0.0001):
        if candidate < requested and not any(
            math.isclose(candidate, existing, rel_tol=0.0, abs_tol=1e-12)
            for existing in candidates
        ):
            candidates.append(candidate)
    return candidates


def _find_feature_window(
    counts: list[int], required_pairs: int,
) -> tuple[int, int] | None:
    """Return a contiguous [start, stop) pair window meeting the feature floor."""
    start = 0
    length = 0
    for index, count in enumerate(counts):
        if count >= MIN_HARRIS_FEATURES_PER_PAIR:
            if length == 0:
                start = index
            length += 1
            if length >= required_pairs:
                return start, start + required_pairs
        else:
            length = 0
    return None


def _calibrate_harris_strength(
    vpi: Any,
    frames: list[np.ndarray],
    requested: float,
    required_pairs: int,
) -> dict[str, Any]:
    """Choose one fixed strength and continuous feature-bearing frame window."""
    stream = _make_pva_stream(vpi)
    attempts: list[dict[str, Any]] = []
    calibration_started = time.perf_counter()
    for strength in _harris_strength_candidates(requested):
        counts: list[int] = []
        attempt_started = time.perf_counter()
        for frame in frames[:-1]:
            frame_u8 = vpi.asimage(frame, vpi.Format.U8)
            frame_s16 = frame_u8.convert(
                vpi.Format.S16,
                backend=vpi.Backend.CUDA,
                stream=stream,
            )
            features, _scores = frame_s16.harriscorners(
                backend=vpi.Backend.PVA,
                gradient_size=3,
                block_size=3,
                strength=strength,
                sensitivity=0.0625,
                min_nms_distance=8,
                stream=stream,
            )
            stream.sync()
            counts.append(int(features.size))
        attempt = {
            "strength": strength,
            "feature_counts": _stats_values([float(count) for count in counts]),
            "eligible_pair_count": sum(
                count >= MIN_HARRIS_FEATURES_PER_PAIR for count in counts
            ),
            "elapsed_ms": round(
                (time.perf_counter() - attempt_started) * 1000.0,
                6,
            ),
        }
        attempts.append(attempt)
        selected_window = _find_feature_window(counts, required_pairs)
        print(
            f"Harris calibration strength={strength:g}: "
            f"min={min(counts)}, mean={statistics.mean(counts):.1f}, "
            f"max={max(counts)}, eligible="
            f"{attempt['eligible_pair_count']}/{len(counts)}",
            flush=True,
        )
        if selected_window is not None:
            selected_start, selected_stop = selected_window
            selected_counts = counts[selected_start:selected_stop]
            attempt["selected_feature_counts"] = _stats_values(
                [float(count) for count in selected_counts]
            )
            attempt["selected_pair_window"] = {
                "start_in_capture": selected_start,
                "stop_exclusive_in_capture": selected_stop,
            }
            return {
                "passed": True,
                "requested_strength": requested,
                "selected_strength": strength,
                "minimum_required_features_per_pair": MIN_HARRIS_FEATURES_PER_PAIR,
                "required_contiguous_pairs": required_pairs,
                "frames_evaluated": len(counts),
                "captured_frames_evaluated": len(frames),
                "selected_pair_window": {
                    "start_in_capture": selected_start,
                    "stop_exclusive_in_capture": selected_stop,
                },
                "selected_frame_window": {
                    "start_in_capture": selected_start,
                    "stop_exclusive_in_capture": selected_stop + 1,
                },
                "attempts": attempts,
                "elapsed_ms": round(
                    (time.perf_counter() - calibration_started) * 1000.0,
                    6,
                ),
                "included_in_method_timing": False,
                "backends": {"u8_to_s16": "CUDA", "harris": "PVA"},
                "harris_input_format": "S16",
                "min_nms_distance": 8,
            }
    raise RuntimeError(
        "PVA Harris calibration could not find a continuous "
        f"{required_pairs}-pair window yielding at least "
        f"{MIN_HARRIS_FEATURES_PER_PAIR} features per previous frame; attempts="
        f"{[(item['strength'], item['eligible_pair_count']) for item in attempts]}"
    )


def _make_pva_stream(vpi: Any) -> Any:
    return vpi.Stream(vpi.Backend.CUDA | vpi.Backend.PVA)


def _make_ofa_stream(vpi: Any) -> Any:
    return vpi.Stream(vpi.Backend.CUDA | vpi.Backend.VIC | vpi.Backend.OFA)


def _run_pyrlk_pair(
    vpi: Any,
    stream: Any,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    pyramid_backend_name: str,
    flow_backend_name: str,
    harris_strength: float,
) -> dict[str, Any]:
    pyramid_backend = getattr(vpi.Backend, pyramid_backend_name)
    flow_backend = getattr(vpi.Backend, flow_backend_name)

    total_start = time.perf_counter()
    previous_u8 = vpi.asimage(previous, vpi.Format.U8)
    current_u8 = vpi.asimage(current, vpi.Format.U8)
    previous_pyramid = previous_u8.gaussian_pyramid(
        PYRAMID_LEVELS,
        PYRAMID_SCALE,
        backend=pyramid_backend,
        stream=stream,
    )
    current_pyramid = current_u8.gaussian_pyramid(
        PYRAMID_LEVELS,
        PYRAMID_SCALE,
        backend=pyramid_backend,
        stream=stream,
    )
    previous_s16 = previous_u8.convert(
        vpi.Format.S16,
        backend=vpi.Backend.CUDA,
        stream=stream,
    )
    features, scores = previous_s16.harriscorners(
        backend=vpi.Backend.PVA,
        gradient_size=3,
        block_size=3,
        strength=harris_strength,
        sensitivity=0.0625,
        min_nms_distance=8,
        stream=stream,
    )
    stream.sync()
    detected = int(features.size)
    capacity = int(features.capacity)
    if detected <= 0:
        raise RuntimeError("PVA Harris returned zero features")
    optflow = vpi.OpticalFlowPyrLK(
        previous_pyramid,
        features,
        backend=flow_backend,
    )
    prep_end = time.perf_counter()

    submit_start = prep_end
    tracked_points, status = optflow(
        current_pyramid,
        winsize=11,
        maxiter=6,
        stream=stream,
    )
    submit_host_end = time.perf_counter()
    stream.sync()
    submit_complete = time.perf_counter()

    rlock_start = submit_complete
    with tracked_points.rlock_cpu() as tracked_data:
        tracked_copy = np.array(tracked_data, dtype=np.float32, copy=True)
    with status.rlock_cpu() as status_data:
        status_copy = np.array(status_data, dtype=np.uint8, copy=True)
    rlock_end = time.perf_counter()

    if tracked_copy.shape[0] != detected or status_copy.shape[0] != detected:
        raise RuntimeError(
            "PyrLK output size mismatch: "
            f"detected={detected}, tracked={tracked_copy.shape[0]}, "
            f"status={status_copy.shape[0]}"
        )
    lost = int(np.count_nonzero(status_copy))
    tracked = detected - lost
    prep_ms = (prep_end - total_start) * 1000.0
    submit_host_ms = (submit_host_end - submit_start) * 1000.0
    accelerator_wait_ms = (submit_complete - submit_host_end) * 1000.0
    submit_ms = (submit_complete - submit_start) * 1000.0
    rlock_ms = (rlock_end - rlock_start) * 1000.0
    total_ms = (rlock_end - total_start) * 1000.0
    accounted_ms = prep_ms + submit_ms + rlock_ms
    return {
        "prep_ms": round(prep_ms, 6),
        "submit_host_ms": round(submit_host_ms, 6),
        "accelerator_wait_ms": round(accelerator_wait_ms, 6),
        "submit_ms": round(submit_ms, 6),
        "rlock_ms": round(rlock_ms, 6),
        "total_ms": round(total_ms, 6),
        "accounted_ms": round(accounted_ms, 6),
        "unattributed_ms": round(total_ms - accounted_ms, 6),
        "detected_features": detected,
        "feature_capacity": capacity,
        "tracked_features": tracked,
        "lost_features": lost,
        "feature_loss_rate": round(lost / detected, 9),
        "output_points_shape": list(tracked_copy.shape),
        "status_zero_means_tracked": True,
        "scores_capacity": int(scores.capacity),
    }


def _run_ofa_pair(
    vpi: Any,
    stream: Any,
    previous: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    previous_y8 = vpi.asimage(previous, vpi.Format.Y8_ER)
    current_y8 = vpi.asimage(current, vpi.Format.Y8_ER)
    previous_pyramid = previous_y8.gaussian_pyramid(
        PYRAMID_LEVELS,
        PYRAMID_SCALE,
        backend=vpi.Backend.CUDA,
        stream=stream,
    )
    current_pyramid = current_y8.gaussian_pyramid(
        PYRAMID_LEVELS,
        PYRAMID_SCALE,
        backend=vpi.Backend.CUDA,
        stream=stream,
    )
    previous_ofa = previous_pyramid.convert(
        vpi.Format.Y8_ER_BL,
        backend=vpi.Backend.VIC,
        stream=stream,
    )
    current_ofa = current_pyramid.convert(
        vpi.Format.Y8_ER_BL,
        backend=vpi.Backend.VIC,
        stream=stream,
    )
    stream.sync()
    prep_end = time.perf_counter()

    submit_start = prep_end
    flow = vpi.optflow_dense(
        previous_ofa,
        current_ofa,
        quality=vpi.OptFlowQuality.LOW,
        gridsize=OFA_GRIDSIZE,
        backend=vpi.Backend.OFA,
        stream=stream,
    )
    submit_host_end = time.perf_counter()
    stream.sync()
    submit_complete = time.perf_counter()

    rlock_start = submit_complete
    with flow.rlock_cpu() as flow_data:
        flow_copy = np.array(flow_data, dtype=np.float32, copy=True)
    rlock_end = time.perf_counter()
    if flow_copy.size <= 0:
        raise RuntimeError("OFA returned an empty flow grid")

    prep_ms = (prep_end - total_start) * 1000.0
    submit_host_ms = (submit_host_end - submit_start) * 1000.0
    accelerator_wait_ms = (submit_complete - submit_host_end) * 1000.0
    submit_ms = (submit_complete - submit_start) * 1000.0
    rlock_ms = (rlock_end - rlock_start) * 1000.0
    total_ms = (rlock_end - total_start) * 1000.0
    accounted_ms = prep_ms + submit_ms + rlock_ms
    return {
        "prep_ms": round(prep_ms, 6),
        "submit_host_ms": round(submit_host_ms, 6),
        "accelerator_wait_ms": round(accelerator_wait_ms, 6),
        "submit_ms": round(submit_ms, 6),
        "rlock_ms": round(rlock_ms, 6),
        "total_ms": round(total_ms, 6),
        "accounted_ms": round(accounted_ms, 6),
        "unattributed_ms": round(total_ms - accounted_ms, 6),
        "output_flow_shape": list(flow_copy.shape),
    }


def _attach_pair_identity(
    sample: dict[str, Any],
    pair: dict[str, Any],
    warmup: int,
) -> dict[str, Any]:
    sample.update({
        "measurement_index": pair["pair_index"] - warmup,
        "pair_index": pair["pair_index"],
        "previous_frame_index": pair["previous_frame_index"],
        "current_frame_index": pair["current_frame_index"],
        "previous_sha256": pair["previous_sha256"],
        "current_sha256": pair["current_sha256"],
        "pair_sha256": pair["pair_sha256"],
    })
    return sample


def _benchmark_pyrlk(
    vpi: Any,
    frames: list[np.ndarray],
    pairs: list[dict[str, Any]],
    warmup: int,
    width: int,
    height: int,
    harris_strength: float,
) -> dict[str, Any]:
    pyramid_backend = _pyramid_backend_name(width, height)
    stream = _make_pva_stream(vpi)
    measured: list[dict[str, Any]] = []
    for pair in pairs:
        index = pair["pair_index"]
        sample = _run_pyrlk_pair(
            vpi,
            stream,
            frames[index],
            frames[index + 1],
            pyramid_backend_name=pyramid_backend,
            flow_backend_name="PVA",
            harris_strength=harris_strength,
        )
        if index >= warmup:
            measured.append(_attach_pair_identity(sample, pair, warmup))
            print(
                f"  PVA PyrLK pair {len(measured)}/{len(pairs) - warmup}: "
                f"{sample['total_ms']:.3f} ms, "
                f"tracked {sample['tracked_features']}/"
                f"{sample['detected_features']}",
                flush=True,
            )
    return {
        "method": "vpi_pva_pyrlk_sparse",
        "algorithm_kind": "sparse_feature_tracking",
        "backends": {
            "u8_to_s16": "CUDA",
            "harris": "PVA",
            "gaussian_pyramid": pyramid_backend,
            "optical_flow_pyrlk": "PVA",
            "cpu_fallback": False,
        },
        "samples": measured,
        "summary": _method_summary(measured),
    }


def _benchmark_ofa(
    vpi: Any,
    frames: list[np.ndarray],
    pairs: list[dict[str, Any]],
    warmup: int,
) -> dict[str, Any]:
    stream = _make_ofa_stream(vpi)
    measured: list[dict[str, Any]] = []
    for pair in pairs:
        index = pair["pair_index"]
        sample = _run_ofa_pair(vpi, stream, frames[index], frames[index + 1])
        if index >= warmup:
            measured.append(_attach_pair_identity(sample, pair, warmup))
            print(
                f"  OFA pair {len(measured)}/{len(pairs) - warmup}: "
                f"{sample['total_ms']:.3f} ms",
                flush=True,
            )
    return {
        "method": "vpi_ofa_dense_study08_synchronized",
        "algorithm_kind": "dense_optical_flow",
        "comparison_reference": (
            "Algorithm-equivalent to the Study-08 path in bench_ofa_pair.py, "
            "with explicit synchronization so CPU readback excludes accelerator wait."
        ),
        "backends": {
            "gaussian_pyramid": "CUDA",
            "block_linear_conversion": "VIC",
            "dense_optical_flow": "OFA",
            "cpu_fallback": False,
        },
        "samples": measured,
        "summary": _method_summary(measured),
    }


def _validate_vpi_environment() -> Any:
    try:
        import vpi
    except Exception as exc:
        raise RuntimeError(f"VPI import failed: {exc}") from exc

    version = getattr(vpi, "__version__", "unknown")
    if not version.startswith(VPI_REQUIRED_PREFIX):
        raise RuntimeError(
            f"SCRUM-75 requires installed VPI 3.2.x; observed {version}"
        )
    missing = [
        name
        for name in ("PVA", "OFA", "CUDA", "VIC")
        if not hasattr(vpi.Backend, name)
    ]
    if missing:
        raise RuntimeError(f"Required explicit VPI backends unavailable: {missing}")
    return vpi


def _smoke_test(
    vpi: Any,
    width: int,
    height: int,
    seed: int,
    harris_strength: float,
) -> dict[str, Any]:
    print(
        f"Synthetic backend smoke test at {width}x{height} before camera access...",
        flush=True,
    )
    frames, _ = _generate_synthetic_frames(width, height, 2, seed + 100000)
    pva_sample = _run_pyrlk_pair(
        vpi,
        _make_pva_stream(vpi),
        frames[0],
        frames[1],
        pyramid_backend_name=_pyramid_backend_name(width, height),
        flow_backend_name="PVA",
        harris_strength=harris_strength,
    )
    ofa_sample = _run_ofa_pair(
        vpi,
        _make_ofa_stream(vpi),
        frames[0],
        frames[1],
    )

    cuda_frames, _ = _generate_synthetic_frames(512, 512, 2, seed + 200000)
    cuda_sample = _run_pyrlk_pair(
        vpi,
        _make_pva_stream(vpi),
        cuda_frames[0],
        cuda_frames[1],
        pyramid_backend_name="CUDA",
        flow_backend_name="CUDA",
        harris_strength=harris_strength,
    )
    return {
        "passed": True,
        "requested_resolution_route": {
            "resolution": [width, height],
            "pyramid_backend": _pyramid_backend_name(width, height),
            "harris_backend": "PVA",
            "pyrlk_backend": "PVA",
            "detected_features": pva_sample["detected_features"],
            "tracked_features": pva_sample["tracked_features"],
        },
        "ofa_backend": {
            "validated": True,
            "output_flow_shape": ofa_sample["output_flow_shape"],
        },
        "pyrlk_backend_support": {
            "installed_vpi_api": ["CPU", "CUDA", "PVA"],
            "cuda_smoke_validated": True,
            "pva_smoke_validated": True,
            "cpu_tested": False,
            "cpu_used": False,
            "cuda_detected_features": cuda_sample["detected_features"],
        },
        "fallback_used": False,
    }


def _validate_frames(
    frames: list[np.ndarray], width: int, height: int, expected: int,
) -> None:
    if len(frames) != expected:
        raise RuntimeError(f"Expected {expected} frames, got {len(frames)}")
    for index, frame in enumerate(frames):
        if frame.dtype != np.uint8 or frame.shape != (height, width):
            raise RuntimeError(
                f"Frame {index} is {frame.shape}/{frame.dtype}; expected "
                f"({height}, {width})/uint8"
            )
        if not frame.flags.c_contiguous:
            frames[index] = np.ascontiguousarray(frame, dtype=np.uint8)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(encoded)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite existing output: {path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SCRUM-75 shared-frame OFA vs VPI PVA PyrLK benchmark. "
            "Camera mode never falls back to CPU."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", action="store_true", help="Use live SkyEye62AM")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic generated GRAY8 frames (no camera import)",
    )
    parser.add_argument(
        "--check-camera-ownership",
        action="store_true",
        help="Read-only ownership check; do not import VPI or open the camera",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--camera-name", default="SkyEye")
    parser.add_argument("--resolution-index", type=int, default=None)
    parser.add_argument("--harris-strength", type=float, default=20.0)
    parser.add_argument(
        "--method-order",
        choices=("pva-first", "ofa-first"),
        default="pva-first",
    )
    parser.add_argument("--pyramid-levels", type=int, default=PYRAMID_LEVELS)
    parser.add_argument("--ofa-gridsize", type=int, default=OFA_GRIDSIZE)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--camera-ownership-confirmed",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.check_camera_ownership:
        return
    if args.camera == args.synthetic:
        parser.error("select exactly one of --camera or --synthetic")
    if args.width is None or args.height is None:
        parser.error("--width and --height are required")
    if (args.width, args.height) not in SUPPORTED_RESOLUTIONS:
        parser.error(
            "supported resolutions are 1920x1080 and 3184x2124"
        )
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.harris_strength <= 0:
        parser.error("--harris-strength must be greater than zero")
    if args.pyramid_levels != PYRAMID_LEVELS:
        parser.error("SCRUM-75 requires exactly four pyramid levels")
    if args.ofa_gridsize != OFA_GRIDSIZE:
        parser.error("SCRUM-75 comparator requires OFA gridsize 4")
    if args.output_json is None:
        parser.error("--output-json is required")
    if args.output_json.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_json}")
    if args.camera and not args.camera_ownership_confirmed:
        parser.error(
            "camera mode must be launched by run_pyrlk_pva_matrix.sh after "
            "the exclusive ownership check"
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    if args.check_camera_ownership:
        owners = _camera_owner_lines()
        if owners:
            print("Camera is occupied; refusing benchmark. Owning process(es):")
            for line in owners:
                print(f"  {line}")
            raise SystemExit(2)
        print("Camera ownership check: free (camera not opened)")
        return

    _load_numpy()
    vpi = _validate_vpi_environment()
    run_id = args.run_id or (
        f"SCRUM75_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    print(
        f"SCRUM-75 {args.width}x{args.height}: source="
        f"{'camera' if args.camera else 'synthetic'}, "
        f"runs={args.runs}, warmup={args.warmup}, order={args.method_order}",
        flush=True,
    )

    smoke = _smoke_test(
        vpi,
        args.width,
        args.height,
        args.seed,
        args.harris_strength,
    )
    required_pairs = args.warmup + args.runs
    selected_frame_count = required_pairs + 1
    capture_frame_count = selected_frame_count
    if args.camera:
        capture_frame_count += LIVE_PAIR_RESERVE
    if args.camera:
        frames, acquisition = _capture_camera_frames(args, capture_frame_count)
        source_name = "live_skyeye62am"
    else:
        frames, acquisition = _generate_synthetic_frames(
            args.width, args.height, capture_frame_count,
            args.seed,
        )
        source_name = "deterministic_synthetic"
    _validate_frames(frames, args.width, args.height, capture_frame_count)
    harris_calibration = _calibrate_harris_strength(
        vpi,
        frames,
        args.harris_strength,
        required_pairs,
    )
    effective_harris_strength = float(harris_calibration["selected_strength"])
    selected_window = harris_calibration["selected_frame_window"]
    selected_start = int(selected_window["start_in_capture"])
    selected_stop = int(selected_window["stop_exclusive_in_capture"])
    captured_frame_count = len(frames)
    frames = frames[selected_start:selected_stop]
    _validate_frames(frames, args.width, args.height, selected_frame_count)
    acquisition.update({
        "captured_frames": captured_frame_count,
        "selected_frames": len(frames),
        "discarded_capture_frames": captured_frame_count - len(frames),
        "selected_frame_window": dict(selected_window),
        "selection_note": (
            "One continuous feature-bearing window was selected before method "
            "timing; OFA and PyrLK consumed the identical selected frames."
        ),
    })
    frame_set, pairs = _build_frame_set(frames, args.warmup)
    original_hashes = [entry["sha256"] for entry in frame_set["frames"]]

    methods: dict[str, Any] = {}
    order = (
        ("pyrlk_pva", "ofa")
        if args.method_order == "pva-first"
        else ("ofa", "pyrlk_pva")
    )
    for method in order:
        print(f"Running {method} on shared frame pairs...", flush=True)
        if method == "pyrlk_pva":
            methods[method] = _benchmark_pyrlk(
                vpi,
                frames,
                pairs,
                args.warmup,
                args.width,
                args.height,
                effective_harris_strength,
            )
        else:
            methods[method] = _benchmark_ofa(vpi, frames, pairs, args.warmup)

    final_hashes = [_frame_sha256(frame) for frame in frames]
    if final_hashes != original_hashes:
        raise RuntimeError("Shared input frames were modified during benchmarking")

    pva_pairs = [sample["pair_sha256"] for sample in methods["pyrlk_pva"]["samples"]]
    ofa_pairs = [sample["pair_sha256"] for sample in methods["ofa"]["samples"]]
    if pva_pairs != ofa_pairs:
        raise RuntimeError("OFA and PyrLK did not consume identical measured pairs")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "resolution_samples",
        "ticket": TICKET,
        "run_id": run_id,
        "generated_at_utc": _utc_now(),
        "argv": sys.argv,
        "git": _git_metadata(),
        "host": _host_metadata(vpi),
        "config": {
            "source": source_name,
            "resolution": {"width": args.width, "height": args.height},
            "runs": args.runs,
            "warmup": args.warmup,
            "frames": selected_frame_count,
            "method_order": list(order),
            "ofa": {
                "gridsize": OFA_GRIDSIZE,
                "quality": "LOW",
                "pyramid_levels": PYRAMID_LEVELS,
            },
            "pyrlk": {
                "pyramid_levels": PYRAMID_LEVELS,
                "pyramid_scale": PYRAMID_SCALE,
                "pyramid_backend": _pyramid_backend_name(args.width, args.height),
                "harris_input_format": "S16",
                "harris_backend": "PVA",
                "harris_gradient_size": 3,
                "harris_block_size": 3,
                "harris_strength": effective_harris_strength,
                "harris_strength_requested": args.harris_strength,
                "harris_strength_effective": effective_harris_strength,
                "harris_calibration": harris_calibration,
                "harris_sensitivity": 0.0625,
                "harris_min_nms_distance": 8,
                "winsize": 11,
                "maxiter": 6,
                "optical_flow_backend": "PVA",
            },
        },
        "timing_contract": {
            "clock": "time.perf_counter",
            "capture_excluded": True,
            "prep_ms": (
                "Format conversion, pyramid construction, feature detection where "
                "applicable, payload creation, and an explicit stream sync."
            ),
            "submit_host_ms": "Python optical-flow call return latency only.",
            "accelerator_wait_ms": (
                "Time from Python call return through explicit stream sync completion."
            ),
            "submit_ms": (
                "Optical-flow call start through explicit stream sync completion; "
                "includes dispatch/scheduling and is not pure kernel time."
            ),
            "rlock_ms": (
                "Post-sync CPU lock/map and NumPy copy; excludes accelerator wait."
            ),
            "total_ms": "Direct start-of-prep through completed CPU readback.",
            "percentiles": "nearest-rank over measured pairs",
        },
        "acquisition": acquisition,
        "frame_set": frame_set,
        "methods": methods,
        "fairness": {
            "same_in_memory_frames": True,
            "same_measured_pair_hashes": True,
            "input_hashes_unchanged_after_methods": True,
            "capture_once_per_resolution": True,
            "feature_window_selected_before_timing": True,
            "selection_applied_equally_to_both_methods": True,
            "note": (
                "A continuous frame window meeting the disclosed PVA Harris "
                "feature floor was selected before timing and then held identical "
                "for both methods. OFA is dense optical flow and PyrLK is sparse "
                "feature tracking; this is a timing/resource comparison, not an "
                "accuracy equivalence."
            ),
        },
        "validation": {
            "backend_smoke": smoke,
            "harris_calibration": harris_calibration,
            "fallback_used": False,
            "camera_opened": bool(args.camera),
            "camera_ownership_rechecked_immediately_before_open": bool(args.camera),
            "sample_counts_match_runs": all(
                len(method["samples"]) == args.runs
                for method in methods.values()
            ),
        },
    }
    if not payload["validation"]["sample_counts_match_runs"]:
        raise RuntimeError("Measured sample count does not match --runs")

    _write_json_exclusive(args.output_json, payload)
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
