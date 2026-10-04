"""Extract the frozen V39 source-only grid; creates no truth labels or scores.

Only the four explicit local 8-bit source paths below are opened. There is no
directory discovery, seeking, detector import, journal access or remote access.
The output directory must not exist. Run only after this implementation and its
synthetic tests have been reviewed; run() freezes their hashes before decoding.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/tiny_target/accuracy_v39_20260925/validation_coverage_plan_v1.json"
MANIFEST_SHA256 = "cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc"
TEST = ROOT / "tests/unit/test_accuracy_v40_coverage.py"
PROTOCOL = ROOT / "docs/phase20_airborne_accuracy_protocol.md"
INVENTORY = ROOT / "docs/accuracy_v39_validation_inventory.md"
PLAN = ROOT / "docs/accuracy_v40_plan.md"
CLIPS = ("0029", "0126", "0055", "0082")
COUNTS = {"0029": 687, "0126": 674, "0055": 689, "0082": 691}
SOURCES = {
    "0029": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0029.avi",
    "0126": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0126.avi",
    "0055": ROOT.parent / "outputs/v7_frozen_evaluation_20260913/sources/chunk_0055.avi",
    "0082": ROOT.parent / "outputs/v7_frozen_evaluation_20260913/sources/chunk_0082.avi",
}
WIDTH, HEIGHT = 4784, 3190
CROP_W, CROP_H, WINDOW_FRAMES = 256, 192, 20
SHEET_COLS, SHEET_ROWS, GAP, HEADER, LABEL = 5, 4, 8, 68, 24


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def array_sha256(array):
    """Hash C-order native bytes; dtype/shape/channel order are bound separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)


def validate_window(window, *, width, height, frame_count):
    for field in ("frame_start", "frame_end_inclusive", "frame_count"):
        if type(window.get(field)) is not int:
            raise ValueError(f"Invalid integer window field: {field}")
    start, end = window["frame_start"], window["frame_end_inclusive"]
    crop = window.get("crop_xywh")
    if (start < 0 or end >= frame_count or end - start + 1 != WINDOW_FRAMES
            or window["frame_count"] != WINDOW_FRAMES):
        raise ValueError("Window must contain exactly 20 in-range consecutive frames")
    if (not isinstance(crop, list) or len(crop) != 4
            or any(type(value) is not int for value in crop)):
        raise ValueError("Invalid crop coordinates")
    x, y, w, h = crop
    if (w, h) != (CROP_W, CROP_H) or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Native crop is outside the source image")


def validate_manifest(manifest):
    if (manifest.get("schema") != "seaqr.accuracy-v39-validation-coverage-plan.v1"
            or manifest.get("allowlisted_clips") != list(CLIPS)
            or set(manifest.get("source_claims", {})) != set(CLIPS)):
        raise ValueError("Invalid frozen coverage manifest or allowlist")
    expected = []
    for clip in CLIPS:
        claim = manifest["source_claims"][clip]
        if (claim.get("width"), claim.get("height"), claim.get("declared_frame_count"),
                claim.get("declared_input_bit_depth"), claim.get("nominal_container_fps")) != (
                    WIDTH, HEIGHT, COUNTS[clip], 8, 10):
            raise ValueError("Source declaration differs from the frozen native grid")
        for stratum, numerator in (("early", 1), ("middle", 5), ("late", 9)):
            start = (COUNTS[clip] - WINDOW_FRAMES) * numerator // 10
            for row, y in enumerate((0, 1499, 2998)):
                for col, x in enumerate((0, 2264, 4528)):
                    expected.append(dict(window_id=f"grid_{clip}_{stratum}_r{row}c{col}",
                        clip=clip, temporal_stratum=stratum, frame_start=start,
                        frame_end_inclusive=start + 19, frame_count=20,
                        spatial_row=row, spatial_column=col, crop_xywh=[x, y, 256, 192]))
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != 108:
        raise ValueError("All 108 frozen windows are required")
    for window, reference in zip(windows, expected):
        validate_window(window, width=WIDTH, height=HEIGHT, frame_count=COUNTS[reference["clip"]])
        if any(window.get(key) != value for key, value in reference.items()):
            raise ValueError("Window differs from frozen enumeration or geometry")
        if window.get("labels") is not None:
            raise ValueError("Coverage extraction cannot import labels")
    if (manifest.get("coverage_totals", {}).get("windows") != 108
            or manifest.get("coverage_totals", {}).get("native_crop_frames") != 2160):
        raise ValueError("Frozen coverage totals differ")
    return windows


def load_manifest(path=MANIFEST):
    # Hash these exact bytes before parsing, avoiding a check/read race.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen manifest SHA-256 mismatch")
    manifest = json.loads(data)
    validate_manifest(manifest)
    return manifest


def extract_clip(capture, windows, claim):
    """Sequential grabs from frame zero through EOF; retrieve selected frames only.

    A grab advances exactly one frame, with its reported next-frame position
    checked. Retrieval never seeks. All selected crops are copied before the
    next grab, so no full-frame image is retained in the output.
    """
    width, height, count = claim["width"], claim["height"], claim["declared_frame_count"]
    if not capture.isOpened():
        raise ValueError("Source decoder did not open")
    metadata = dict(width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    fps=capture.get(cv2.CAP_PROP_FPS),
                    reported_frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    backend=capture.getBackendName())
    if ((metadata["width"], metadata["height"], metadata["reported_frame_count"])
            != (width, height, count) or metadata["fps"] != claim["nominal_container_fps"]):
        raise ValueError("Source decoder dimensions/fps/frame count mismatch")
    if capture.get(cv2.CAP_PROP_POS_FRAMES) != 0:
        raise ValueError("Sequential extraction must begin at frame zero")
    selected = {}
    arrays = {}
    for window in windows:
        validate_window(window, width=width, height=height, frame_count=count)
        key = window["window_id"]
        if key in arrays:
            raise ValueError("Duplicate window identity")
        arrays[key] = np.empty((20, CROP_H, CROP_W, 3), dtype=np.uint8)
        for frame in range(window["frame_start"], window["frame_end_inclusive"] + 1):
            selected.setdefault(frame, []).append(window)
    retrieved = []
    for frame in range(count):
        if not capture.grab() or capture.get(cv2.CAP_PROP_POS_FRAMES) != frame + 1:
            raise ValueError(f"Sequential source indexing failed at frame {frame}")
        if frame not in selected:
            continue
        ok, bgr = capture.retrieve()
        if not ok or bgr is None or bgr.shape != (height, width, 3) or bgr.dtype != np.uint8:
            raise ValueError(f"Native BGR decode failed at frame {frame}")
        for window in selected[frame]:
            x, y, w, h = window["crop_xywh"]
            arrays[window["window_id"]][frame - window["frame_start"]] = bgr[y:y + h, x:x + w]
        retrieved.append(frame)
        del bgr
    if capture.grab():
        raise ValueError("Actual source contains more frames than the frozen count")
    metadata.update(actual_sequential_frame_count=count, eof_verified=True,
                    retrieved_frame_indices=retrieved, retained_full_frames=0,
                    decoded_dtype="uint8", decoded_channel_order="BGR")
    return arrays, metadata


def sheet_origins():
    return [(col * (CROP_W + GAP), HEADER + row * (CROP_H + LABEL + GAP))
            for row in range(SHEET_ROWS) for col in range(SHEET_COLS)]


def contact_sheet(frames, window):
    if frames.shape != (20, CROP_H, CROP_W, 3) or frames.dtype != np.uint8:
        raise ValueError("Contact sheet requires all 20 native uint8 BGR crops")
    canvas = np.full((HEADER + 4 * (CROP_H + LABEL) + 3 * GAP,
                      5 * CROP_W + 4 * GAP, 3), 24, dtype=np.uint8)
    lines = [window["window_id"] + " | UNMARKED SOURCE | native 1:1",
             f'ROI {window["crop_xywh"]} | all 20 frames | no contrast adjustment | no truth labels']
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 23 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, (240, 240, 240), 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(sheet_origins()):
        # Copy only; all text is outside the source pixel rectangles.
        canvas[y:y + CROP_H, x:x + CROP_W] = frames[index]
        cv2.putText(canvas, f'frame {window["frame_start"] + index}', (x + 4, y + CROP_H + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def export_window(output, window, frames):
    key = window["window_id"]
    archive = output / "native" / (key + ".npz")
    sheet = output / "sheets" / (key + ".png")
    indices = np.arange(window["frame_start"], window["frame_end_inclusive"] + 1, dtype=np.int64)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, native_bgr=frames, frame_indices=indices)
    canvas = contact_sheet(frames, window)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Lossless contact sheet PNG encoding failed")
    write_bytes(sheet, encoded.tobytes())
    return dict(window_id=key, clip=window["clip"], temporal_stratum=window["temporal_stratum"],
                spatial_row=window["spatial_row"], spatial_column=window["spatial_column"],
                frame_start=window["frame_start"], frame_end_inclusive=window["frame_end_inclusive"],
                frame_indices=indices.tolist(), crop_xywh=window["crop_xywh"],
                native_array=dict(key="native_bgr", dtype="uint8", shape=list(frames.shape),
                    channel_order="BGR", byte_order="C", sha256=array_sha256(frames),
                    per_frame_sha256=[array_sha256(frame) for frame in frames]),
                frame_index_array=dict(key="frame_indices", dtype=str(indices.dtype),
                    shape=list(indices.shape), sha256=array_sha256(indices)),
                native_archive=dict(path=str(archive.relative_to(output)), sha256=sha256_file(archive)),
                contact_sheet=dict(path=str(sheet.relative_to(output)), sha256=sha256_file(sheet),
                    shape=list(canvas.shape), columns=5, rows=4, native_pixel_scale=1,
                    crop_origins_xy=sheet_origins(), labels_outside_source_pixels=True),
                source_review_status="pending", class_adjudication_status="pending",
                annotation_status="pending", score_status="not_run")


def write_index(output, records):
    items = []
    for record in records:
        key = html.escape(record["window_id"])
        items.append(f'<li><a href="{record["contact_sheet"]["path"]}">{key}</a> '
                     f'frames {record["frame_start"]}–{record["frame_end_inclusive"]}; '
                     f'ROI {record["crop_xywh"]}; '
                     f'<a href="{record["native_archive"]["path"]}">native BGR array</a></li>')
    body = """<!doctype html><meta charset="utf-8"><title>V40 source-only coverage</title>
<h1>V40 source-only coverage: 108 windows / 2,160 crop-frames</h1>
<p>Every sheet contains all 20 frames, left-to-right then top-to-bottom, at native
256×192 sampling. Open the PNG at actual size (100%, not fit-to-window) for review.
Frame and ROI labels are outside the source pixels; no contrast adjustment or overlays.
NPZ members are native_bgr (20,192,256,3 uint8 BGR) and frame_indices (20 int64).
All source review and class adjudication remain pending. This is development
coverage, not airborne truth, verified negative exposure, or a generalization test.
Do not replace dark, obstructed, uncertain or uninformative windows.</p>
<p>Review each source frame independently before any candidate scores. Record
unknown visibility/class separately; no interpolation or absence inferred from darkness.
Additional context requires separately recorded scope.</p><ol>"""
    write_bytes(output / "index.html", (body + "\n".join(items) + "</ol>\n").encode("utf-8"))


def run(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Exclusive fresh output directory required")
    manifest = load_manifest()
    windows = manifest["windows"]
    input_hashes = {str(MANIFEST): MANIFEST_SHA256}
    for path in (Path(__file__).resolve(), TEST, PROTOCOL, INVENTORY, PLAN):
        expected = manifest["inputs_sha256"].get(str(path))
        input_hashes[str(path)] = verify_hash(path, expected) if expected else sha256_file(path)
    sources = {}
    for clip in CLIPS:
        source = SOURCES[clip]
        digest = verify_hash(source, manifest["source_claims"][clip]["declared_source_sha256"])
        input_hashes[str(source)] = digest
        sources[clip] = dict(local_path=str(source), sha256=digest,
                             manifest_claim=manifest["source_claims"][clip])
    # Exclusive creation is also the race-safe overwrite guard.
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ("implementation", "native", "sheets"):
        (output / subdir).mkdir()
    for path in (Path(__file__).resolve(), TEST, MANIFEST, PROTOCOL, INVENTORY, PLAN):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != input_hashes[str(path)]:
            raise ValueError(f"Input changed while freezing: {path}")
        write_bytes(output / "implementation" / path.name, data)
    frozen_at = utc_now()
    write_json(output / "freeze.json", dict(schema="seaqr.accuracy-v40-source-coverage-freeze.v1",
        frozen_at_utc=frozen_at, inputs_sha256=input_hashes, sources=sources, windows=windows,
        manifest_sha256=MANIFEST_SHA256, native_crop_frames=2160, expected_windows=108,
        render=dict(columns=5, rows=4, native_dimensions_wh=[256, 192],
                    native_scale=1, channel_order="BGR", labels_outside_pixels=True,
                    contrast_adjustment=False, lossless_png=True),
        procedure="Sequential grab from zero through EOF; retrieve only the 60 selected frames per clip; retain only declared crops.",
        runtime=dict(python=sys.version, platform=platform.platform(), opencv=cv2.__version__, numpy=np.__version__),
        no_detector_or_journal_inputs=True, no_new_labels=True, no_scores=True,
        source_review_status="pending", development_only=True, authoritative_negative_exposure=False))
    print(f"Frozen before decode: {output / 'freeze.json'}", flush=True)
    try:
        records, decoded = [], {}
        decode_started = utc_now()
        for clip in CLIPS:
            active = [window for window in windows if window["clip"] == clip]
            capture = cv2.VideoCapture(str(SOURCES[clip]))
            try:
                arrays, decoded[clip] = extract_clip(capture, active, manifest["source_claims"][clip])
            finally:
                capture.release()
            for window in active:
                records.append(export_window(output, window, arrays.pop(window["window_id"])))
            print(f"Source-only coverage complete: {clip}, 27 windows / 540 crop-frames", flush=True)
        if len(records) != 108 or sum(len(record["frame_indices"]) for record in records) != 2160:
            raise ValueError("Incomplete coverage; no completed packet may be published")
        write_index(output, records)
        end_hashes = {path: verify_hash(Path(path), digest) for path, digest in input_hashes.items()}
        write_json(output / "packet.json", dict(schema="seaqr.accuracy-v40-source-coverage-packet.v1",
            frozen_at_utc=frozen_at, decode_started_at_utc=decode_started, finished_at_utc=utc_now(),
            freeze_sha256=sha256_file(output / "freeze.json"), manifest_sha256=MANIFEST_SHA256,
            sources=sources, actual_decoder_metadata=decoded, windows=records,
            windows_count=108, native_crop_frames=2160, source_review_status="pending",
            labels_created=False, scores_computed=False, development_only=True,
            verified_negative_roi_seconds=0, end_inputs_sha256=end_hashes))
        # The completion receipt binds every generated file; its own digest is
        # printed as the external final hash, avoiding a self-hash construction.
        artifacts = {str(path.relative_to(output)): sha256_file(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
        write_json(output / "completion.json", dict(schema="seaqr.accuracy-v40-source-coverage-completion.v1",
            completed=True, finished_at_utc=utc_now(), artifacts_sha256=artifacts,
            end_inputs_sha256=end_hashes, windows=108, native_crop_frames=2160))
        print(json.dumps(dict(output=str(output), completion_sha256=sha256_file(output / "completion.json"))), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", dict(completed=False, failed_at_utc=utc_now(),
                                                error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output)
