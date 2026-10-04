#!/usr/bin/env python3
"""Bounded public AOT intake; never runs or configures a detector.

Phases are explicit so exact image size is reported before image access.
No AWS credentials, requester-pays option, whole-bucket scan, or CPU fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com/"
META_KEY = "part1/ImageSets/groundtruth.json"
META_ETAG = '"574928912ee2c1aa7821386330872d94"'
META_BYTES = 455074222
RANGE_BYTES = 8 * 1024 * 1024
FRAME_CAP = 300
IMAGE_CAP = 1024 ** 3
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    """Idempotent only for identical bytes; never overwrite valid artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def save_json(path, obj):
    save(path, (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def fetch(url, limit, *, byte_range=None, etag=None):
    if not url.startswith(BASE):
        raise ValueError("source outside public AOT bucket")
    headers = {"User-Agent": "SEAQR-bounded-AOT-intake/1"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    if etag:
        headers["If-Match"] = etag
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45, context=ssl.create_default_context()) as response:
        if response.geturl() != url:
            raise ValueError("unexpected redirect")
        expected_status = 206 if byte_range else 200
        if response.status != expected_status:
            raise ValueError(f"unexpected HTTP status {response.status}")
        length = int(response.headers.get("Content-Length", "-1"))
        chunked_listing = length == -1 and url.startswith(BASE + "?list-type=2&")
        if (length < 0 and not chunked_listing) or length > limit:
            raise ValueError("missing/oversize Content-Length")
        if etag and response.headers.get("ETag") != etag:
            raise ValueError("source ETag changed")
        if byte_range:
            expected = f"bytes {byte_range[0]}-{byte_range[1]}/{META_BYTES}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("incorrect Content-Range")
        data = response.read(limit + 1)
        if (length >= 0 and len(data) != length) or len(data) > limit:
            raise ValueError("transfer length mismatch")
        return data, dict(response.headers)


def parse_complete_samples(data):
    """Parse complete top-level sample objects in a deliberately truncated prefix."""
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    metadata_match = re.match(r'\s*\{\s*"metadata"\s*:\s*', text)
    if not metadata_match:
        raise ValueError("unexpected AOT document root")
    metadata, pos = decoder.raw_decode(text, metadata_match.end())
    match = re.match(r'\s*,\s*"samples"\s*:\s*\{\s*', text[pos:])
    if not match:
        raise ValueError("missing samples object")
    pos += match.end()
    samples = []
    seen = set()
    while pos < len(text):
        pos += len(text[pos:]) - len(text[pos:].lstrip())
        if text[pos:pos + 1] == "}":
            break
        try:
            flight, key_end = decoder.raw_decode(text, pos)
            colon = re.match(r"\s*:\s*", text[key_end:])
            if not colon:
                if not text[key_end:].strip():
                    break
                raise ValueError("missing sample colon")
            sample, end = decoder.raw_decode(text, key_end + colon.end())
        except json.JSONDecodeError:
            break  # only complete samples are retained
        if not re.fullmatch(r"[a-f0-9]{32}", flight) or flight in seen:
            raise ValueError("invalid/duplicate sequence ID")
        seen.add(flight)
        samples.append((flight, sample))
        pos = end
        tail = re.match(r"\s*,\s*", text[pos:])
        if tail:
            pos += tail.end()
        elif text[pos:].lstrip().startswith("}"):
            break
        elif text[pos:].strip():
            raise ValueError("unexpected sample separator")
    if not samples:
        raise ValueError("no complete samples in bounded metadata")
    return metadata, samples


def frames_for(flight, sample):
    meta = sample["metadata"]
    if meta["resolution"] != {"height": 2048, "width": 2448} or meta["fps"] != 10.0:
        raise ValueError("unexpected image size/cadence")
    frames = {}
    for entity in sample["entities"]:
        index = entity["blob"]["frame"]
        timestamp = entity["time"]
        if type(index) is not int or type(timestamp) is not int:
            raise ValueError("frame/time must be exact integers")
        name = f"{timestamp}{flight}.png"
        if entity["flight_id"] != flight or entity["img_name"] != name:
            raise ValueError("inconsistent source identity")
        row = frames.setdefault(index, {
            "source_frame": index, "timestamp_ns": str(timestamp),
            "img_name": name, "entities": [],
        })
        if row["timestamp_ns"] != str(timestamp) or row["img_name"] != name:
            raise ValueError("multiple images/timestamps for frame")
        has_bb = "bb" in entity
        if has_bb != ("id" in entity):
            raise ValueError("incomplete object label")
        if has_bb:
            bb = entity["bb"]
            if not isinstance(entity["id"], str) or not entity["id"].strip():
                raise ValueError("invalid object ID")
            if (not isinstance(bb, list) or len(bb) != 4
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in bb)
                    or min(bb[2:]) <= 0):
                raise ValueError("invalid bounding box")
            horizon = entity.get("labels", {}).get("is_above_horizon")
            if horizon is not None and (type(horizon) is not int or horizon not in (-1, 0, 1)):
                raise ValueError("invalid horizon label")
            if any(e.get("id") == entity["id"] for e in row["entities"]):
                raise ValueError("duplicate per-frame object ID")
        row["entities"].append(entity)
    ordered = [frames[i] for i in sorted(frames)]
    if len(ordered) != meta["number_of_frames"]:
        raise ValueError("annotation frame census disagrees with publisher metadata")
    times = [int(row["timestamp_ns"]) for row in ordered]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("non-increasing source timestamps")
    for row in ordered:
        row["airborne_label_count"] = sum("bb" in e for e in row["entities"])
        if row["airborne_label_count"] and any("bb" not in e for e in row["entities"]):
            raise ValueError("mixed empty and labeled frame records")
    return ordered


def choose_candidates(samples):
    candidates, census = [], []
    for flight, sample in samples:
        try:
            full = frames_for(flight, sample)
            rows = full[:FRAME_CAP]
            positive = sum(r["airborne_label_count"] > 0 for r in rows)
            contiguous = len(rows) == FRAME_CAP and all(
                b["source_frame"] == a["source_frame"] + 1 for a, b in zip(rows, rows[1:]))
            qualifies = contiguous and positive >= 30 and len(rows) - positive >= 20
            census.append({"flight_id": flight, "published_frames": len(full),
                           "prefix_labeled_frames": positive, "prefix_empty_frames": len(rows) - positive,
                           "prefix_contiguous": contiguous, "qualifies": qualifies})
            if qualifies:
                candidates.append((flight, sample, rows))
        except (KeyError, TypeError, ValueError) as exc:
            census.append({"flight_id": flight, "qualifies": False, "error": str(exc)})
    return candidates, census


def metadata_phase(out):
    raw_path = out / "metadata/groundtruth.prefix8MiB.bin"
    if raw_path.exists():
        data = raw_path.read_bytes()
        receipt = json.loads((out / "metadata/source_receipt.json").read_text())
        if len(data) != RANGE_BYTES or sha(data) != receipt["sha256"]:
            raise ValueError("cached metadata does not match receipt")
    else:
        data, headers = fetch(BASE + META_KEY, RANGE_BYTES,
                              byte_range=(0, RANGE_BYTES - 1), etag=META_ETAG)
        save(raw_path, data)
        save_json(out / "metadata/source_receipt.json", {
            "url": BASE + META_KEY, "range": [0, RANGE_BYTES - 1],
            "source_object_bytes": META_BYTES, "etag": META_ETAG, "sha256": sha(data),
            "access_utc": datetime.now(timezone.utc).isoformat(), "headers": headers,
        })
    publisher_meta, samples = parse_complete_samples(data)
    candidates, census = choose_candidates(samples)
    save_json(out / "metadata/census.json", {
        "publisher_metadata": publisher_meta, "metadata_prefix_sha256": sha(data),
        "complete_sequence_objects": len(samples), "sequence_prefixes": census,
        "selection_rule": "first publisher-order eligible sequence, first 300 frames; >=30 labeled and >=20 empty; all contiguous",
        "eligible_sequences": len(candidates), "representative_dataset_sample": False,
    })
    if not candidates:
        raise ValueError("no qualifying metadata-only candidate; criteria unchanged")
    flight, sample, rows = candidates[0]
    sample = copy.deepcopy(sample)
    normalized_ranges = 0
    for entity in sample["entities"]:
        value = entity["blob"].get("range_distance_m")
        if isinstance(value, float) and math.isnan(value):
            entity["blob"]["range_distance_m"] = None
            normalized_ranges += 1
    rows = frames_for(flight, sample)[:FRAME_CAP]
    sample["seaqr_derived_data_notice"] = {
        "changed": True, "changes": "Selected complete sequence extracted and pretty-printed; optional NaN ranges mapped to null, never zero.",
        "normalized_nan_ranges": normalized_ranges,
        "unmodified_source_prefix_sha256": sha(data),
        "provider": "Amazon Airborne Object Tracking",
        "license": "https://cdla.dev/permissive-1-0/",
    }
    save_json(out / "metadata/selected_sequence_annotations.json", sample)
    save_json(out / "preselection.json", {
        "part": "part1", "flight_id": flight, "frames": rows,
        "metadata_prefix_sha256": sha(data), "max_source_png_bytes": IMAGE_CAP,
        "bb_convention": "left,top,width,height", "pilot_type": "development convenience sample",
        "physical_class_source": "publisher airborne annotations, not SEAQR predictions",
        "derived_data_notice": sample["seaqr_derived_data_notice"],
    })
    print(json.dumps({"complete_sequences": len(samples), "eligible": len(candidates),
                      "selected": flight, "frames": len(rows), "labeled": sum(r["airborne_label_count"] > 0 for r in rows)}))


def inventory_phase(out):
    selection = json.loads((out / "preselection.json").read_text())
    flight = selection["flight_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", flight) or selection["part"] != "part1":
        raise ValueError("invalid selection identity")
    prefix = f"part1/Images/{flight}/"
    objects, token = {}, None
    for page in range(3):
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BASE + "?" + urllib.parse.urlencode(query)
        cache = out / f"metadata/selected_images_page{page}.xml"
        data = cache.read_bytes() if cache.exists() else fetch(url, 1024 * 1024)[0]
        save(cache, data)
        root = ET.fromstring(data)
        if root.findtext("s:Prefix", namespaces=NS) != prefix:
            raise ValueError("listing returned wrong prefix")
        for item in root.findall("s:Contents", NS):
            key = item.findtext("s:Key", namespaces=NS)
            if not key.startswith(prefix) or key in objects:
                raise ValueError("wrong-prefix/duplicate object")
            objects[key] = {"key": key, "bytes": int(item.findtext("s:Size", namespaces=NS)),
                            "etag": item.findtext("s:ETag", namespaces=NS)}
        if root.findtext("s:IsTruncated", namespaces=NS) == "false":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            raise ValueError("missing pagination token")
    else:
        raise ValueError("listing page cap exceeded")
    annotation_path = out / "metadata/selected_sequence_annotations.json"
    full_rows = frames_for(flight, json.loads(annotation_path.read_text()))
    if set(objects) != {prefix + row["img_name"] for row in full_rows}:
        raise ValueError("sequence image inventory differs from complete annotation census")
    for row in selection["frames"]:
        key = prefix + row["img_name"]
        row["source_object"] = objects[key]
    if any(type(r["source_object"]["bytes"]) is not int or not 0 < r["source_object"]["bytes"] <= IMAGE_CAP for r in selection["frames"]):
        raise ValueError("invalid per-image byte count")
    total = sum(row["source_object"]["bytes"] for row in selection["frames"])
    if len(selection["frames"]) != FRAME_CAP or total > IMAGE_CAP:
        raise ValueError(f"image cap exceeded: {len(selection['frames'])} frames, {total} bytes")
    selection["source_png_bytes"] = total
    selection["listed_sequence_objects"] = len(objects)
    selection["preselection_sha256"] = sha((out / "preselection.json").read_bytes())
    save_json(out / "frozen_image_manifest.json", selection)
    print(json.dumps({"frames": len(selection["frames"]), "source_png_bytes": total,
                      "source_MiB": total / 1024 ** 2, "flight_id": flight}))


def download_phase(out):
    import cv2
    import numpy as np
    manifest = json.loads((out / "frozen_image_manifest.json").read_text())
    if sha((out / "preselection.json").read_bytes()) != manifest["preselection_sha256"]:
        raise ValueError("preselection changed")
    preselection = json.loads((out / "preselection.json").read_text())
    stripped = copy.deepcopy(manifest)
    for field in ("source_png_bytes", "listed_sequence_objects", "preselection_sha256"):
        stripped.pop(field)
    for row in stripped["frames"]:
        row.pop("source_object")
    if stripped != preselection:
        raise ValueError("manifest differs from frozen preselection")
    rows = manifest["frames"]
    if any(type(r["source_object"]["bytes"]) is not int or not 0 < r["source_object"]["bytes"] <= IMAGE_CAP for r in rows):
        raise ValueError("invalid per-image byte count")
    total = sum(r["source_object"]["bytes"] for r in rows)
    if len(rows) != FRAME_CAP or total != manifest["source_png_bytes"] or total > IMAGE_CAP:
        raise ValueError("image cap/manifest mismatch")
    for row in rows:
        obj = row["source_object"]
        expected = f"part1/Images/{manifest['flight_id']}/{row['img_name']}"
        if obj["key"] != expected or not re.fullmatch(r"[0-9]{19}[a-f0-9]{32}\.png", row["img_name"]):
            raise ValueError("source outside frozen sequence")
        if not re.fullmatch(r'"[a-f0-9]{32}"', obj["etag"]):
            raise ValueError("unsupported source checksum convention")
    records = []
    for i, row in enumerate(rows):
        obj = row["source_object"]
        path = out / "source_png" / row["img_name"]
        data = path.read_bytes() if path.exists() else fetch(BASE + obj["key"], obj["bytes"], etag=obj["etag"])[0]
        if len(data) != obj["bytes"]:
            raise ValueError("image length differs from frozen listing")
        # Publisher's non-multipart object ETags are also checked as an integrity guard.
        if not re.fullmatch(r'"[a-f0-9]{32}"', obj["etag"]):
            raise ValueError("unsupported source checksum convention")
        if hashlib.md5(data).hexdigest() != obj["etag"].strip('"'):
            raise ValueError("PNG checksum differs from source ETag")
        pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.shape != (2048, 2448) or pixels.dtype != np.uint8:
            raise ValueError("source is not native 2448x2048 gray8")
        save(path, data)
        records.append({"source_frame": row["source_frame"], "timestamp_ns": row["timestamp_ns"],
                        "img_name": row["img_name"], "png_sha256": sha(data),
                        "pixel_sha256": sha(pixels.tobytes()), "bytes": len(data)})
        if (i + 1) % 25 == 0:
            print(f"validated {i + 1}/{len(rows)} native gray8 frames", flush=True)
    times = [int(r["timestamp_ns"]) for r in rows]
    deltas = np.diff(np.asarray(times, dtype=np.int64)).astype(np.float64) / 1e6
    nominal_error_ms = [(t - times[0] - i * 100000000) / 1e6 for i, t in enumerate(times)]
    objects = Counter(e["id"] for r in rows for e in r["entities"] if "id" in e)
    receipt = {
        "manifest_sha256": sha((out / "frozen_image_manifest.json").read_bytes()),
        "frames": len(records), "source_png_bytes": total, "resolution": [2448, 2048],
        "dtype": "uint8", "grayscale": True, "source_frame_range": [rows[0]["source_frame"], rows[-1]["source_frame"]],
        "labeled_frames": sum(r["airborne_label_count"] > 0 for r in rows),
        "publisher_empty_label_frames": sum(r["airborne_label_count"] == 0 for r in rows),
        "object_annotation_counts": objects, "source_timestamp_span_seconds": (times[-1] - times[0]) / 1e9,
        "timestamp_delta_ms": {"min": float(deltas.min()), "max": float(deltas.max()), "mean": float(deltas.mean())},
        "max_abs_nominal10Hz_timestamp_error_ms": max(abs(x) for x in nominal_error_ms),
        "detector_run": False, "independent_validation": False, "images": records,
    }
    save_json(out / "download_validation.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "images"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["metadata", "inventory", "download"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"metadata": metadata_phase, "inventory": inventory_phase, "download": download_phase}[args.phase](args.output)


if __name__ == "__main__":
    main()
