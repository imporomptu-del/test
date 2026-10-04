"""Render a human-viewable diagnostic video from an injected RAW16 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import SyntheticInjector, load_injection_spec
from .frame_source import FfmpegVideoSource


CANVAS_SIZE = (1920, 1080)
MAIN_BOX = (24, 72, 1390, 1008)


def map_reference_points_to_source(
    points_xy: np.ndarray, source_to_reference: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_xy, np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [point, 2]")
    matrix = np.asarray(source_to_reference, np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("source_to_reference must be 3x3")
    inverse = np.linalg.inv(matrix)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = (inverse @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def latest_completed_batch(
    frame_index: int, batches: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    completed = [
        batch
        for batch in batches
        if int(batch["candidate_batch"]["frame_indices"][-1]) <= frame_index
    ]
    return completed[-1] if completed else None


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _tone_map(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    scaled = (image.astype(np.float32) - lower) * (255.0 / max(upper - lower, 1))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _fit_main(image: Any) -> tuple[Any, tuple[int, int, int, int]]:
    left, top, right, bottom = MAIN_BOX
    available_width = right - left
    available_height = bottom - top
    scale = min(available_width / image.width, available_height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(size)
    x = left + (available_width - size[0]) // 2
    y = top + (available_height - size[1]) // 2
    return resized, (x, y, x + size[0], y + size[1])


def _display_point(
    point_xy: Sequence[float], source_shape: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[float, float]:
    height, width = source_shape
    left, top, right, bottom = box
    return (
        left + float(point_xy[0]) * (right - left) / width,
        top + float(point_xy[1]) * (bottom - top) / height,
    )


def _draw_cross(draw: Any, x: float, y: float, color: str, radius: int = 5) -> None:
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)


def _local_crop(image: np.ndarray, x: float, y: float, radius: int = 16) -> np.ndarray:
    center_x = int(round(x))
    center_y = int(round(y))
    x0 = max(0, center_x - radius)
    x1 = min(image.shape[1], center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(image.shape[0], center_y + radius + 1)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    low, high = np.percentile(crop, (2, 99.5))
    return _tone_map(crop, float(low), float(high))


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {resolved}")
    return value


def render(
    *,
    video_path: str | Path,
    timestamps_path: str | Path,
    motion_report_path: str | Path,
    accuracy_report_path: str | Path,
    injection_spec_path: str | Path,
    output_path: str | Path,
    threshold_snr: float = 8,
    max_frames: int = 16,
    output_fps: float = 2,
) -> Path:
    from PIL import Image, ImageDraw

    motion = _load_json(motion_report_path)
    accuracy = _load_json(accuracy_report_path)
    if motion.get("schema_version") != "seaqr.tiny-target.motion.v10":
        raise ValueError("visualizer requires a motion.v10 report")
    if accuracy.get("schema_version") != "seaqr.tiny-target.injected-raw-evaluation.v1":
        raise ValueError("visualizer requires an injected RAW evaluation v1 report")
    threshold_key = f"{float(threshold_snr):g}"
    threshold_parameter = motion["evaluation"].get(
        "candidate_threshold_sweep_parameter",
        "score_threshold_snr",
    )
    phase_title = (
        "SEAQR Phase 12 — local-CFAR candidate selection"
        if threshold_parameter == "cfar_threshold_sigma"
        else "SEAQR Phase 11 — raw-SNR candidate selection"
    )
    try:
        batches = motion["evaluation"]["candidate_threshold_sweep"][threshold_key]
        probes = motion["evaluation"]["injected_truth_score_probes"]
        windows = accuracy["window_evidence"][threshold_key]
    except KeyError as exc:
        raise ValueError(f"threshold {threshold_key} is absent from the reports") from exc
    probe_by_frames = {tuple(item["frame_indices"]): item for item in probes}
    evidence_by_frames = {tuple(item["frame_indices"]): item for item in windows}
    stabilization = {
        int(item["frame_index"]): item for item in motion["stabilized_frames"]
    }
    spec, _ = load_injection_spec(injection_spec_path)
    injector = SyntheticInjector(spec)
    source = FfmpegVideoSource(
        video_path,
        timestamp_csv=timestamps_path,
        timestamp_policy="require_sidecar",
        bit_depth=16,
        max_frames=max_frames,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing visualization: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    title_font = _font(30, bold=True)
    heading_font = _font(24, bold=True)
    body_font = _font(20)
    small_font = _font(17)
    target_colors = ("#00f0ff", "#5cff76", "#ffd84d", "#ff8b3d")
    with tempfile.TemporaryDirectory(prefix="seaqr_evaluation_visual_") as directory:
        frame_directory = Path(directory)
        display_limits: tuple[float, float] | None = None
        rendered_count = 0
        for frame in source:
            original = frame.image
            injected = injector.inject(frame).image
            if display_limits is None:
                sample = original[::8, ::8]
                display_limits = tuple(float(value) for value in np.percentile(sample, (0.5, 99.5)))
            gray = Image.fromarray(_tone_map(injected, *display_limits), mode="L").convert("RGB")
            main, main_box = _fit_main(gray)
            canvas = Image.new("RGB", CANVAS_SIZE, "#0a0f18")
            canvas.paste(main, main_box[:2])
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 18), phase_title, fill="white", font=title_font)
            draw.rectangle(main_box, outline="#496078", width=2)
            state = stabilization[frame.frame_index]
            matrix = np.asarray(state["source_to_reference_matrix"], np.float64)
            batch = latest_completed_batch(frame.frame_index, batches)
            active_events = injector.records[-1]["events"]
            for target_index, event in enumerate(active_events):
                px, py = _display_point(event["position_xy_px"], frame.shape, main_box)
                color = target_colors[target_index % len(target_colors)]
                draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline=color, width=3)
                draw.text((px + 12, py - 12), event["target_id"], fill=color, font=small_font)
            shown_candidates = 0
            batch_frames: tuple[int, ...] | None = None
            candidate_count = 0
            matches: list[Mapping[str, Any]] = []
            confirmed_truth_matches: list[Mapping[str, Any]] = []
            output_truncated = False
            if batch is not None:
                candidate_batch = batch["candidate_batch"]
                batch_frames = tuple(int(value) for value in candidate_batch["frame_indices"])
                if int(candidate_batch["segment_index"]) == int(state["segment_index"]):
                    candidates = candidate_batch["candidates"]
                    candidate_count = len(candidates)
                    output_truncated = bool(
                        candidate_batch["metrics"]["output_truncated"]
                    )
                    points = np.asarray(
                        [item["discrete_position_xy_px"] for item in candidates[:30]],
                        np.float64,
                    )
                    source_points = map_reference_points_to_source(points, matrix)
                    for point in source_points:
                        x, y = _display_point(point, frame.shape, main_box)
                        if main_box[0] <= x < main_box[2] and main_box[1] <= y < main_box[3]:
                            _draw_cross(draw, x, y, "#ff4058")
                            shown_candidates += 1
                    window_evidence = evidence_by_frames[batch_frames]
                    matches = window_evidence["matching"]["matches"]
                    confirmed_truth_matches = window_evidence.get(
                        "confirmed_track_matching", {}
                    ).get("matches", [])
                    matched_points = np.asarray(
                        [
                            candidates[int(item["candidate_index"])][
                                "discrete_position_xy_px"
                            ]
                            for item in matches
                        ],
                        np.float64,
                    )
                    if matched_points.size:
                        matched_source = map_reference_points_to_source(
                            matched_points,
                            matrix,
                        )
                        for point in matched_source:
                            x, y = _display_point(point, frame.shape, main_box)
                            _draw_cross(draw, x, y, "#ffffff", radius=10)
                            draw.ellipse(
                                (x - 6, y - 6, x + 6, y + 6),
                                outline="#5cff76",
                                width=3,
                            )
            panel_x = 1420
            draw.text((panel_x, 76), f"Frame {frame.frame_index:02d} / {max_frames - 1:02d}", fill="white", font=heading_font)
            draw.text(
                (panel_x, 112),
                f"segment {state['segment_index']}  stabilization Δ=({matrix[0,2]:+.2f}, {matrix[1,2]:+.2f}) px",
                fill="#a9bad0",
                font=small_font,
            )
            y_text = 154
            if batch_frames is None:
                draw.text((panel_x, y_text), "No completed 4-frame detection window yet", fill="#ffd84d", font=body_font)
            else:
                draw.text((panel_x, y_text), f"Window: {list(batch_frames)}", fill="white", font=body_font)
                y_text += 30
                cap_state = "cap active" if output_truncated else "below cap"
                draw.text(
                    (panel_x, y_text),
                    f"Candidate output: {candidate_count}/256 ({cap_state})",
                    fill="#ff9b54",
                    font=body_font,
                )
                y_text += 30
                probe = probe_by_frames[batch_frames]
                valid_target_count = sum(
                    bool(target.get("valid_score_available"))
                    for target in probe["targets"]
                )
                draw.text(
                    (panel_x, y_text),
                    (
                        f"Injected detections: {len(matches)}/4 "
                        f"({len(matches)}/{valid_target_count} valid)"
                    ),
                    fill="#5cff76",
                    font=heading_font,
                )
                y_text += 34
                draw.text(
                    (panel_x, y_text),
                    f"Truth-matched confirmed tracks: {len(confirmed_truth_matches)}",
                    fill="#b899ff" if confirmed_truth_matches else "#a9bad0",
                    font=small_font,
                )
                y_text += 34
                draw.text((panel_x, y_text), "Dense truth-location evidence", fill="white", font=heading_font)
                y_text += 35
                for target_index, target in enumerate(probe["targets"]):
                    color = target_colors[target_index % len(target_colors)]
                    flux = float(target["flux_dn"])
                    if target.get("valid_score_available"):
                        score_text = (
                            f"raw {target['local_peak_score_snr']:.1f} "
                            f"rank ≥{target['surface_rank_lower_bound']:,}"
                        )
                    else:
                        score_text = "invalid integration support"
                    draw.text((panel_x, y_text), f"● {flux:,.0f} DN", fill=color, font=body_font)
                    draw.text((panel_x + 138, y_text), score_text, fill="#dce7f5", font=small_font)
                    y_text += 22
                    if target.get("selection_score_available"):
                        draw.text(
                            (panel_x + 138, y_text),
                            (
                                f"local {target['local_peak_selection_score']:.1f}σ "
                                f"rank ≥{target['selection_surface_rank_lower_bound']:,}"
                            ),
                            fill="#8ac7ff",
                            font=small_font,
                        )
                        y_text += 24
                    else:
                        y_text += 8
            draw.text((panel_x, 512), "Red ×: top 30 ranked candidates", fill="#ff4058", font=small_font)
            draw.text((panel_x, 538), f"{shown_candidates} visible", fill="#dce7f5", font=small_font)
            draw.text((panel_x, 566), "White × + green ring: matched detection", fill="#5cff76", font=small_font)
            draw.text((panel_x, 604), "Locally normalized target crops", fill="white", font=heading_font)
            crop_y = 642
            for target_index, event in enumerate(active_events):
                crop_array = _local_crop(
                    injected,
                    float(event["position_xy_px"][0]),
                    float(event["position_xy_px"][1]),
                )
                crop = Image.fromarray(crop_array, mode="L").convert("RGB").resize((112, 112))
                crop_x = panel_x + (target_index % 2) * 230
                local_y = crop_y + (target_index // 2) * 166
                canvas.paste(crop, (crop_x, local_y))
                draw.rectangle((crop_x, local_y, crop_x + 112, local_y + 112), outline=target_colors[target_index], width=3)
                draw.text(
                    (crop_x, local_y + 118),
                    f"{float(event['requested_flux_dn']):,.0f} DN",
                    fill=target_colors[target_index],
                    font=small_font,
                )
            if not active_events:
                draw.text((panel_x, crop_y), "Targets enter at frame 7", fill="#a9bad0", font=body_font)
            draw.text(
                (24, 1024),
                (
                    "Diagnostic rendering: injected source pixels + report overlays. "
                    "Colored circles are truth; white/green marks are matched detections."
                ),
                fill="#a9bad0",
                font=small_font,
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
    parser.add_argument("--accuracy-report", required=True, type=Path)
    parser.add_argument("--injection-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-snr", type=float, default=8)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--output-fps", type=float, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = render(
        video_path=args.video,
        timestamps_path=args.timestamps,
        motion_report_path=args.motion_report,
        accuracy_report_path=args.accuracy_report,
        injection_spec_path=args.injection_spec,
        output_path=args.output,
        threshold_snr=args.threshold_snr,
        max_frames=args.max_frames,
        output_fps=args.output_fps,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
