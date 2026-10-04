"""Create an honest review pack from an unlabeled Phase 15 field replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_visualizer import (
    CANVAS_SIZE,
    _display_point,
    _draw_cross,
    _fit_main,
    _font,
    _local_crop,
    _tone_map,
    latest_completed_batch,
    map_reference_points_to_source,
)
from .frame_source import FfmpegVideoSource
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.unlabeled-field-review.v1"


@dataclass(frozen=True, slots=True)
class FieldReviewPolicy:
    minimum_observations: int = 5
    minimum_observation_fraction: float = 0.96
    minimum_measurement_speed_px_s: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 2
        ):
            raise ValueError("minimum_observations must be an integer >= 2")
        if (
            not math.isfinite(self.minimum_observation_fraction)
            or not 0 < self.minimum_observation_fraction <= 1
        ):
            raise ValueError("minimum_observation_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.minimum_measurement_speed_px_s)
            or self.minimum_measurement_speed_px_s < 0
        ):
            raise ValueError("minimum_measurement_speed_px_s must be nonnegative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_observations": self.minimum_observations,
            "minimum_observation_fraction": self.minimum_observation_fraction,
            "minimum_measurement_speed_px_s": self.minimum_measurement_speed_px_s,
        }


def _read_report(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read motion report {resolved}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != (
        "seaqr.tiny-target.motion.v10"
    ):
        raise ValueError("field review requires a motion.v10 report")
    return resolved, report, hashlib.sha256(raw).hexdigest()


def _threshold_batches(
    report: Mapping[str, Any], threshold_sigma: float
) -> tuple[str, Sequence[Mapping[str, Any]]]:
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("motion report has no evaluation section")
    if evaluation.get("candidate_threshold_sweep_parameter") != "cfar_threshold_sigma":
        raise ValueError("field review requires a CFAR threshold sweep")
    sweep = evaluation.get("candidate_threshold_sweep")
    if not isinstance(sweep, Mapping):
        raise ValueError("motion report has no CFAR threshold batches")
    threshold_key = f"{float(threshold_sigma):g}"
    batches = sweep.get(threshold_key)
    if not isinstance(batches, list):
        raise ValueError(f"CFAR threshold {threshold_key} is absent from the report")
    return threshold_key, batches


def track_passes_policy(
    track: Mapping[str, Any], policy: FieldReviewPolicy
) -> bool:
    quality = track.get("quality_evidence")
    if not isinstance(quality, Mapping):
        return False
    speed = quality.get("measurement_speed_px_s")
    if not isinstance(speed, Mapping):
        return False
    return bool(
        track.get("lifecycle_state") in {"confirmed", "coasted"}
        and int(quality.get("observation_count", 0)) >= policy.minimum_observations
        and float(quality.get("observation_fraction_of_age_windows", 0))
        >= policy.minimum_observation_fraction
        and float(speed.get("mean", 0)) >= policy.minimum_measurement_speed_px_s
    )


def _compact_track(
    track: Mapping[str, Any],
    *,
    ever_qualified: bool,
    qualified_at_latest: bool,
    first_qualification_timestamp_ns: int | None,
) -> dict[str, Any]:
    quality = track["quality_evidence"]
    return {
        "track_id": int(track["track_id"]),
        "review_label": "unreviewed",
        "lifecycle_state_at_latest_observation": track["lifecycle_state"],
        "birth_timestamp_ns": int(track["birth_timestamp_ns"]),
        "confirmation_timestamp_ns": track["confirmation"]["timestamp_ns"],
        "latest_measurement_timestamp_ns": int(track["last_measurement_timestamp_ns"]),
        "latest_state_timestamp_ns": int(track["state"]["timestamp_ns"]),
        "latest_state_xy_vx_vy": [float(value) for value in track["state"]["mean"]],
        "age_windows": int(track["age_windows"]),
        "associated_update_count": int(track["associated_update_count"]),
        "missed_windows": int(track["missed_windows"]),
        "observation_count": int(quality["observation_count"]),
        "observation_fraction": float(quality["observation_fraction_of_age_windows"]),
        "mean_measurement_speed_px_s": float(quality["measurement_speed_px_s"]["mean"]),
        "mean_detector_score_snr": float(quality["detector_score_snr"]["mean"]),
        "mean_selection_score": float(quality["selection_score"]["mean"]),
        "mean_peak_to_neighbor_ratio": float(quality["peak_to_neighbor_ratio"]["mean"]),
        "ever_passed_frozen_diagnostic_policy": ever_qualified,
        "passes_frozen_diagnostic_policy_at_latest_observation": qualified_at_latest,
        "first_diagnostic_qualification_timestamp_ns": first_qualification_timestamp_ns,
    }


def analyze(
    report_path: str | Path,
    *,
    threshold_sigma: float = 8.0,
    policy: FieldReviewPolicy | None = None,
) -> dict[str, Any]:
    """Summarize detections without assigning truth labels to an unlabeled scene."""

    resolved_policy = policy or FieldReviewPolicy()
    path, report, digest = _read_report(report_path)
    threshold_key, batches = _threshold_batches(report, threshold_sigma)
    latest_tracks: dict[int, Mapping[str, Any]] = {}
    first_qualified: dict[int, int] = {}
    per_window = []
    total_candidates = 0
    total_reservations = 0
    windows_at_candidate_cap = 0
    for batch in batches:
        candidate_batch = batch["candidate_batch"]
        candidates = candidate_batch["candidates"]
        tracks = batch["confirmed_or_coasted_tracks"]
        qualified_ids = []
        for track in tracks:
            track_id = int(track["track_id"])
            current = latest_tracks.get(track_id)
            if current is None or int(track["state"]["timestamp_ns"]) >= int(
                current["state"]["timestamp_ns"]
            ):
                latest_tracks[track_id] = track
            if track_passes_policy(track, resolved_policy):
                qualified_ids.append(track_id)
                first_qualified.setdefault(track_id, int(track["state"]["timestamp_ns"]))
        metrics = candidate_batch["metrics"]
        reservation = metrics.get("track_guided_reservation", {})
        reservation_count = int(reservation.get("retained_reservation_count", 0))
        candidate_count = len(candidates)
        total_candidates += candidate_count
        total_reservations += reservation_count
        windows_at_candidate_cap += int(bool(metrics.get("output_truncated")))
        per_window.append(
            {
                "frame_indices": [int(value) for value in candidate_batch["frame_indices"]],
                "reference_timestamp_ns": int(candidate_batch["reference_timestamp_ns"]),
                "candidate_count": candidate_count,
                "confirmed_or_coasted_track_count": len(tracks),
                "diagnostic_qualified_track_count": len(qualified_ids),
                "diagnostic_qualified_track_ids": sorted(qualified_ids),
                "reservation_count": reservation_count,
                "output_truncated": bool(metrics.get("output_truncated")),
            }
        )

    rows = []
    for track_id, track in sorted(latest_tracks.items()):
        qualified_latest = track_passes_policy(track, resolved_policy)
        rows.append(
            _compact_track(
                track,
                ever_qualified=track_id in first_qualified,
                qualified_at_latest=qualified_latest,
                first_qualification_timestamp_ns=first_qualified.get(track_id),
            )
        )
    qualified_latest_ids = [
        row["track_id"]
        for row in rows
        if row["passes_frozen_diagnostic_policy_at_latest_observation"]
    ]
    summary = report["summary"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(REPOSITORY)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
        "input": {
            "motion_report": {"path": str(path), "sha256": digest},
            "source": report["source"],
        },
        "operating_point": {
            "cfar_threshold_sigma": float(threshold_key),
            "diagnostic_policy": resolved_policy.to_dict(),
            "diagnostic_policy_used_by_live_tracker": False,
        },
        "pipeline_health": {
            "frames_read": int(summary["frames_read"]),
            "pairs_attempted": int(summary["pairs_attempted"]),
            "accepted_global_transforms": int(summary["accepted_global_transforms"]),
            "rejected_global_transforms": int(summary["rejected_global_transforms"]),
            "detection_ready_frames": int(summary["matched_filter"]["detection_ready_frames"]),
            "synthetic_tracking_windows": len(batches),
        },
        "screening_counts": {
            "total_candidates_across_windows": total_candidates,
            "unique_confirmed_or_coasted_track_count": len(rows),
            "ever_diagnostic_qualified_track_count": len(first_qualified),
            "qualified_at_latest_observation_track_count": len(qualified_latest_ids),
            "qualified_at_latest_observation_track_ids": qualified_latest_ids,
            "track_guided_reservation_count": total_reservations,
            "windows_at_candidate_cap": windows_at_candidate_cap,
        },
        "per_window": per_window,
        "tracks": rows,
        "interpretation": {
            "ground_truth_available": False,
            "real_object_count": None,
            "false_alarm_count": None,
            "precision": None,
            "recall": None,
            "warning": (
                "Every track is unreviewed. Counts are screening workload, not real-object "
                "detections or false alarms. The motion/persistence rule is diagnostic only."
            ),
        },
    }


def write_track_csv(path: str | Path, tracks: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite track CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(tracks[0]) if tracks else ["track_id", "review_label"]
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracks)
    return destination


def _propagated_reference_position(
    state: Sequence[float], state_timestamp_ns: int, frame_timestamp_ns: int
) -> np.ndarray:
    values = np.asarray(state, np.float64)
    dt = (frame_timestamp_ns - state_timestamp_ns) / 1e9
    return values[:2] + values[2:] * dt


def render(
    *,
    video_path: str | Path,
    timestamps_path: str | Path,
    motion_report_path: str | Path,
    output_path: str | Path,
    threshold_sigma: float = 8.0,
    policy: FieldReviewPolicy | None = None,
    max_frames: int | None = None,
    output_fps: float = 3.0,
    max_candidates_shown: int = 24,
    max_tracks_shown: int = 24,
) -> Path:
    from PIL import Image, ImageDraw

    if output_fps <= 0:
        raise ValueError("output_fps must be positive")
    if max_candidates_shown < 0 or max_tracks_shown < 0:
        raise ValueError("display limits cannot be negative")
    resolved_policy = policy or FieldReviewPolicy()
    _, report, _ = _read_report(motion_report_path)
    threshold_key, batches = _threshold_batches(report, threshold_sigma)
    report_frame_count = int(report["summary"]["frames_read"])
    frame_limit = min(max_frames or report_frame_count, report_frame_count)
    stabilization = {
        int(item["frame_index"]): item for item in report["stabilized_frames"]
    }
    source = FfmpegVideoSource(
        video_path,
        timestamp_csv=timestamps_path,
        timestamp_policy="require_sidecar",
        bit_depth=int(report["config"]["effective_resolved"]["input"]["bit_depth"]),
        max_frames=frame_limit,
        timestamp_gap_factor=float(
            report["config"]["effective_resolved"]["input"]["timestamp_gap_factor"]
        ),
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite visualization: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    title_font = _font(28, bold=True)
    heading_font = _font(22, bold=True)
    body_font = _font(18)
    small_font = _font(15)
    with tempfile.TemporaryDirectory(prefix="seaqr_field_review_") as directory:
        frame_directory = Path(directory)
        display_limits: tuple[float, float] | None = None
        rendered_count = 0
        for frame in source:
            original = frame.image
            if display_limits is None:
                sample = original[::8, ::8]
                display_limits = tuple(
                    float(value) for value in np.percentile(sample, (0.5, 99.5))
                )
            gray = Image.fromarray(_tone_map(original, *display_limits), mode="L").convert("RGB")
            main, main_box = _fit_main(gray)
            canvas = Image.new("RGB", CANVAS_SIZE, "#0a0f18")
            canvas.paste(main, main_box[:2])
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 18), "SEAQR Phase 15 — unlabeled field screening", fill="white", font=title_font)
            draw.rectangle(main_box, outline="#496078", width=2)
            state = stabilization[frame.frame_index]
            matrix = np.asarray(state["source_to_reference_matrix"], np.float64)
            batch = latest_completed_batch(frame.frame_index, batches)
            candidate_count = 0
            track_count = 0
            qualified: list[Mapping[str, Any]] = []
            displayed_tracks: list[tuple[Mapping[str, Any], np.ndarray]] = []
            reservation_count = 0
            batch_frames: Sequence[int] | None = None
            if batch is not None:
                candidate_batch = batch["candidate_batch"]
                batch_frames = candidate_batch["frame_indices"]
                if int(candidate_batch["segment_index"]) == int(state["segment_index"]):
                    candidates = candidate_batch["candidates"]
                    candidate_count = len(candidates)
                    dt = (
                        frame.timestamp_ns - int(candidate_batch["reference_timestamp_ns"])
                    ) / 1e9
                    candidate_points = np.asarray(
                        [
                            np.asarray(item["discrete_position_xy_px"], np.float64)
                            + np.asarray(item["discrete_velocity_xy_px_s"], np.float64) * dt
                            for item in candidates[:max_candidates_shown]
                        ],
                        np.float64,
                    )
                    if candidate_points.size:
                        for item, point in zip(
                            candidates[:max_candidates_shown],
                            map_reference_points_to_source(candidate_points, matrix),
                            strict=True,
                        ):
                            x, y = _display_point(point, frame.shape, main_box)
                            is_reserved = item.get("selection", {}).get("reservation_track_id") is not None
                            _draw_cross(draw, x, y, "#b899ff" if is_reserved else "#ff4058")
                    tracks = batch["confirmed_or_coasted_tracks"]
                    track_count = len(tracks)
                    qualified = [
                        track for track in tracks if track_passes_policy(track, resolved_policy)
                    ]
                    qualified.sort(
                        key=lambda item: (
                            -int(item["quality_evidence"]["observation_count"]),
                            -float(item["quality_evidence"]["measurement_speed_px_s"]["mean"]),
                            int(item["track_id"]),
                        )
                    )
                    for track in qualified[:max_tracks_shown]:
                        reference_point = _propagated_reference_position(
                            track["state"]["mean"],
                            int(track["state"]["timestamp_ns"]),
                            frame.timestamp_ns,
                        )[None, :]
                        point = map_reference_points_to_source(reference_point, matrix)[0]
                        x, y = _display_point(point, frame.shape, main_box)
                        if main_box[0] <= x < main_box[2] and main_box[1] <= y < main_box[3]:
                            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#00f0ff", width=3)
                            draw.text((x + 11, y - 11), f"T{track['track_id']}", fill="#00f0ff", font=small_font)
                            displayed_tracks.append((track, point))
                    reservation_count = int(
                        candidate_batch["metrics"]
                        .get("track_guided_reservation", {})
                        .get("retained_reservation_count", 0)
                    )

            panel_x = 1420
            draw.text((panel_x, 80), f"Frame {frame.frame_index:02d}/{frame_limit - 1:02d}", fill="white", font=heading_font)
            draw.text((panel_x, 116), f"CFAR {threshold_key} · no injection", fill="#a9bad0", font=body_font)
            y = 160
            if batch_frames is None:
                draw.text((panel_x, y), "Waiting for first 4-frame window", fill="#ffd84d", font=body_font)
            else:
                draw.text((panel_x, y), f"Window {list(batch_frames)}", fill="white", font=body_font)
                y += 34
                draw.text((panel_x, y), f"Candidates: {candidate_count}", fill="#ff6b79", font=heading_font)
                y += 34
                draw.text((panel_x, y), f"Confirmed/coasted: {track_count}", fill="#ffb45e", font=body_font)
                y += 30
                draw.text((panel_x, y), f"Motion-qualified: {len(qualified)}", fill="#00f0ff", font=heading_font)
                y += 34
                draw.text((panel_x, y), f"Reservations: {reservation_count}", fill="#b899ff", font=body_font)
                y += 44
                draw.text((panel_x, y), "Top diagnostic tracks", fill="white", font=heading_font)
                y += 32
                for track in qualified[:6]:
                    quality = track["quality_evidence"]
                    draw.text(
                        (panel_x, y),
                        (
                            f"T{int(track['track_id']):04d}  "
                            f"v={float(quality['measurement_speed_px_s']['mean']):.2f} px/s  "
                            f"n={int(quality['observation_count'])}"
                        ),
                        fill="#dce7f5",
                        font=small_font,
                    )
                    y += 23
            draw.text((panel_x, 510), "Track-centered pixel crops", fill="white", font=heading_font)
            for crop_index, (track, point) in enumerate(displayed_tracks[:4]):
                crop_array = _local_crop(original, float(point[0]), float(point[1]), radius=24)
                crop = Image.fromarray(crop_array, mode="L").convert("RGB").resize((92, 92))
                crop_x = panel_x + (crop_index % 2) * 215
                crop_y = 548 + (crop_index // 2) * 138
                canvas.paste(crop, (crop_x, crop_y))
                draw.rectangle((crop_x, crop_y, crop_x + 92, crop_y + 92), outline="#00f0ff", width=2)
                draw.text(
                    (crop_x, crop_y + 98),
                    f"T{track['track_id']} · {track['quality_evidence']['measurement_speed_px_s']['mean']:.2f} px/s",
                    fill="#00f0ff",
                    font=small_font,
                )
            if not displayed_tracks:
                draw.text((panel_x, 550), "No qualified track in this frame", fill="#71849a", font=small_font)
            draw.text((panel_x, 838), "Red ×  top CFAR candidates", fill="#ff4058", font=small_font)
            draw.text((panel_x, 862), "Purple ×  reserved candidate", fill="#b899ff", font=small_font)
            draw.text((panel_x, 886), "Cyan ○  diagnostic moving track", fill="#00f0ff", font=small_font)
            draw.text((panel_x, 925), "UNLABELED SCREENING", fill="#ffd84d", font=heading_font)
            draw.multiline_text(
                (panel_x, 962),
                "Not verified objects or false alarms.\nPSF and tracker noise are provisional.",
                fill="#ffd84d",
                font=body_font,
                spacing=8,
            )
            canvas.save(frame_directory / f"frame_{frame.frame_index:04d}.png")
            rendered_count += 1
        if rendered_count == 0:
            raise ValueError("video produced no frames")
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-framerate",
            f"{output_fps:g}",
            "-i",
            str(frame_directory / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--motion-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--tracks-csv", type=Path)
    parser.add_argument("--cfar-threshold", type=float, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-fps", type=float, default=3)
    parser.add_argument("--max-candidates-shown", type=int, default=24)
    parser.add_argument("--max-tracks-shown", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = FieldReviewPolicy()
    summary = analyze(
        args.motion_report,
        threshold_sigma=args.cfar_threshold,
        policy=policy,
    )
    output = render(
        video_path=args.video,
        timestamps_path=args.timestamps,
        motion_report_path=args.motion_report,
        output_path=args.output,
        threshold_sigma=args.cfar_threshold,
        policy=policy,
        max_frames=args.max_frames,
        output_fps=args.output_fps,
        max_candidates_shown=args.max_candidates_shown,
        max_tracks_shown=args.max_tracks_shown,
    )
    print(f"Wrote {output}")
    if args.summary_output is not None:
        summary_output = write_json_exclusive(args.summary_output, summary)
        print(f"Wrote {summary_output}")
    if args.tracks_csv is not None:
        csv_output = write_track_csv(args.tracks_csv, summary["tracks"])
        print(f"Wrote {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
