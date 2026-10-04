"""Bounded, frozen U8 nuisance-origin selection and source-only review packet.

No detector is executed here. Source patches are native nearest-pixel crops,
not the actual stabilized detector image. Candidate-free is not negative truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT.parent / "outputs/seaqr_discovery_pair_20260928/sources/chunk_0240.avi"
JOURNAL_PATH = ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930/evidence/0240/run/frames.jsonl"
MANIFEST_PATH = ROOT / "configs/evaluation/discovery_pair_regression_20260929.json"
SOURCE_SHA = "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"
JOURNAL_SHA = "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92"
MANIFEST_SHA = "10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"
SCHEMA = "seaqr.nuisance-origin.packet.v1"
HEIGHT, WIDTH, FRAMES, PATCH = 3190, 4784, 673, 17
RADIUS = PATCH // 2
CONTEXT = 129
OFFSETS = ((48, 0), (-48, 0), (0, 48), (0, -48))
BURST_ANCHORS, TARGET_ANCHORS = (60, 75, 90), (435, 450)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def read_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "output exists; no overwrite")
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_file(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), "regular absolute unlinked input required")
    require(sha(path) == expected, "input SHA256 differs: " + str(path))
    return path


def point(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value), "invalid point")
    return [float(v) for v in value]


def peak(candidate):
    """Scores belong to the original peak, not the consolidated centroid."""
    shape = candidate.get("shape")
    xy = shape["peak_reference_xy"] if shape else [candidate["x"], candidate["y"]]
    xy = point(xy)
    require(all(v == int(v) for v in xy), "score peak is not native integer")
    return [int(v) for v in xy]


def source_xy(row, reference_xy):
    matrix = np.asarray(row["source_to_reference"], dtype=np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "invalid source transform")
    try:
        transformed = np.linalg.solve(matrix, np.array([*point(reference_xy), 1.0]))
    except np.linalg.LinAlgError as error:
        raise ValueError("singular source transform") from error
    require(np.isfinite(transformed).all() and abs(transformed[2]) > 1e-12, "invalid homogeneous transform")
    return (transformed[:2] / transformed[2]).tolist()


def full_patch(xy, size=PATCH):
    require(type(size) is int and 1 <= size <= CONTEXT and size % 2 == 1, "bounded odd crop size required")
    x, y = point(xy)
    radius = size // 2
    # Continuous support is checked as well as nearest-pixel crop support.
    return radius <= x <= WIDTH - 1 - radius and radius <= y <= HEIGHT - 1 - radius


def available(rows, frames, xy):
    return full_patch(xy) and all(full_patch(source_xy(rows[f], xy)) for f in frames)


def comparator(rows, frames, xy):
    for dx, dy in OFFSETS:
        trial = [xy[0] + dx, xy[1] + dy]
        if available(rows, frames, trial) and all(
                math.dist(trial, peak(c)) > PATCH for f in frames for c in rows[f]["candidates"]):
            return trial
    return None


def select_episodes(rows, references):
    """Pure deterministic selection; references contain source measurement points."""
    require(len(rows) == FRAMES and all(r.get("frame_index") == i for i, r in enumerate(rows)),
            "complete ordered 673-frame journal required")
    reference = {r["frame_index"]: point(r["measurement_source_xy"]) for r in references}
    require(len(reference) == len(references), "duplicate target reference")
    requests = [(f, p, "burst") for f in BURST_ANCHORS for p in ("bright", "dark")]
    requests += [(f, "dark", "known_target") for f in TARGET_ANCHORS]
    episodes, grouped = [], {}
    for anchor, polarity, kind in requests:
        name = f"{kind}_{anchor:03d}_{polarity}"
        frames = list(range(anchor - 2, anchor + 3))
        require(len({rows[f]["segment"] for f in frames}) == 1, "sample crosses coordinate reset")
        candidates = [(i, c) for i, c in enumerate(rows[anchor]["candidates"]) if c["polarity"] == polarity]
        for _, c in candidates:
            point(c["source_xy"])
            require(type(c["score"]) in (int, float) and math.isfinite(c["score"]), "nonfinite score")
            peak(c)
        if kind == "known_target":
            require(anchor in reference, "missing declared target anchor")
            candidates = [(i, c) for i, c in candidates if math.dist(c["source_xy"], reference[anchor]) <= 8]
            candidates.sort(key=lambda item: (math.dist(item[1]["source_xy"], reference[anchor]),
                            -item[1]["score"], peak(item[1])[1], peak(item[1])[0], item[0]))
        else:
            candidates.sort(key=lambda item: (-item[1]["score"], peak(item[1])[1], peak(item[1])[0], item[0]))
        episode = dict(episode_id=name, kind=kind, anchor_frame=anchor, frames=frames, polarity=polarity,
            sample_status="missing_candidate", comparator_status="sample_unavailable",
            label="uncertain_no_class_truth" if kind == "burst" else "user_confirmed_pass_baseline_derived_anchor",
            eligible_anchor_candidates=len(candidates),
            independent_per_frame_ground_truth=False, target_reference_source_xy=reference.get(anchor) if kind == "known_target" else None)
        if candidates:
            index, c = candidates[0]
            xy = peak(c)
            episode.update(candidate_index=index, candidate_score=c["score"],
                candidate_response_dn=c["response_dn"], candidate_noise_sigma_dn=c["noise_sigma_dn"],
                candidate_centroid_reference_xy=[c["x"], c["y"]], candidate_source_xy=c["source_xy"],
                sample_reference_xy=xy, score_coordinate_basis="original accepted peak, not centroid",
                sample_status="available" if available(rows, frames, xy) else "insufficient_patch_support")
            if episode["sample_status"] == "available":
                control = comparator(rows, frames, xy)
                episode.update(comparator_reference_xy=control,
                    comparator_status="available" if control else "no_candidate_free_offset_with_full_support",
                    comparator_label="uncertain_candidate_free_not_negative_truth")
                for f in frames:
                    for role, location in (("sample", xy), ("comparator", control)):
                        if location is not None:
                            grouped.setdefault(str(f), []).append(dict(point_id=name + "/" + role,
                                episode_id=name, role=role, reference_xy=location, source_xy=source_xy(rows[f], location)))
        episodes.append(episode)
    return episodes, {k: grouped[k] for k in sorted(grouped, key=int)}


def validate_plan(plan):
    """File-I/O-free plan validation for the remote passive instrumentation."""
    require(plan.get("schema") == SCHEMA and plan.get("clip") == "0240"
            and plan.get("patch_size") == PATCH and plan.get("source_shape_hw") == [HEIGHT, WIDTH]
            and plan.get("source_frames") == FRAMES, "plan schema/scope differs")
    require(plan.get("input_sha256") == dict(journal=JOURNAL_SHA, source=SOURCE_SHA, manifest=MANIFEST_SHA),
            "plan input hashes differ")
    episodes = plan.get("episodes", [])
    expected = [(f"burst_{f:03d}_{p}", f) for f in BURST_ANCHORS for p in ("bright", "dark")]
    expected += [(f"known_target_{f:03d}_dark", f) for f in TARGET_ANCHORS]
    require([(e["episode_id"], e["anchor_frame"]) for e in episodes] == expected, "episode inventory differs")
    by_id, expected_groups = {}, {}
    for e in episodes:
        require(e["frames"] == list(range(e["anchor_frame"] - 2, e["anchor_frame"] + 3)), "episode frame bounds differ")
        by_id[e["episode_id"]] = e
        require(e["sample_status"] in {"available", "missing_candidate", "insufficient_patch_support"}, "bad sample status")
        require(e.get("independent_per_frame_ground_truth") is False, "ground-truth scope changed")
        require(e.get("label") == ("uncertain_no_class_truth" if e["episode_id"].startswith("burst_")
                else "user_confirmed_pass_baseline_derived_anchor"), "sample labels changed")
        if e["sample_status"] == "available":
            roles = [("sample", e["sample_reference_xy"])]
            if e["comparator_status"] == "available":
                require(tuple(b - a for a, b in zip(e["sample_reference_xy"], e["comparator_reference_xy"]))
                        in OFFSETS, "comparator offset changed")
                roles.append(("comparator", e["comparator_reference_xy"]))
            else:
                require(e["comparator_status"] == "no_candidate_free_offset_with_full_support", "bad missing comparator")
            for role, xy in roles:
                require(full_patch(xy) and all(type(v) is int for v in xy), "invalid reference patch center")
                for f in e["frames"]:
                    expected_groups.setdefault(str(f), []).append((e["episode_id"] + "/" + role, role, xy))
        else:
            require(e["comparator_status"] == "sample_unavailable", "unexpected comparator for unavailable sample")
    groups = plan.get("frame_points")
    require(isinstance(groups, dict) and set(groups) == set(expected_groups), "capture frame inventory differs")
    for key, records in groups.items():
        require(len(records) == len(expected_groups[key]), "capture point count differs")
        actual = [(p["point_id"], p["role"], p["reference_xy"]) for p in records]
        require(actual == expected_groups[key], "capture point inventory differs")
        for p in records:
            require(p["episode_id"] in by_id and p["point_id"] == p["episode_id"] + "/" + p["role"]
                    and full_patch(p["source_xy"]), "invalid source point or identity")
    return plan


def build_plan(journal_path=JOURNAL_PATH, manifest_path=MANIFEST_PATH, source_path=SOURCE_PATH):
    require(Path(journal_path) == JOURNAL_PATH and Path(manifest_path) == MANIFEST_PATH
            and Path(source_path) == SOURCE_PATH, "only the exact approved local0240 input paths are allowed")
    paths = dict(journal=verify_file(journal_path, JOURNAL_SHA), manifest=verify_file(manifest_path, MANIFEST_SHA),
                 source=verify_file(source_path, SOURCE_SHA))
    with paths["journal"].open() as stream:
        rows = [read_json(line) for line in stream]
    manifest = read_json(paths["manifest"].read_text())
    require(manifest["positive_pass"]["clip_id"] == "0240", "target manifest clip differs")
    episodes, groups = select_episodes(rows, manifest["positive_pass"]["baseline_measurements"])
    result = dict(schema=SCHEMA, clip="0240", patch_size=PATCH, source_shape_hw=[HEIGHT, WIDTH],
        source_frames=FRAMES, nominal_fps=10, input_paths={k: str(v) for k, v in paths.items()},
        input_sha256=dict(journal=JOURNAL_SHA, source=SOURCE_SHA, manifest=MANIFEST_SHA),
        episodes=episodes, frame_points=groups, raw16_accessed=False, sealed_holdouts_accessed=False,
        selection="Burst: highest score per polarity; ties peak y,x then candidate index. Known: nearest dark source candidate within8px; ties score descending, peak y,x,index.",
        comparator_policy="First full-support offset in [(48,0),(-48,0),(0,48),(0,-48)] with no archived candidate peak within17px inclusive across all five frames.",
        coordinate_policy="Fixed anchor peak reference center over five frames; source coordinates inverse archived source_to_reference. This is not a moving target-following crop.",
        limitations=["Detector-selected exploratory packet, not accuracy/recall/false-positive estimates.",
            "Candidate-free comparators and burst samples remain uncertain; absence is not verified.",
            "User-confirmed target pass uses baseline-derived coordinates, not independent per-frame truth.",
            "Source crops show nearest native source pixels, not the actual warped/filtered detector input."])
    for key, path in paths.items():
        verify_file(path, result["input_sha256"][key])
    return validate_plan(result)


def native_crop(gray, xy, size=PATCH):
    require(gray.shape == (HEIGHT, WIDTH) and gray.dtype == np.uint8, "native U8 source required")
    require(full_patch(xy, size), "insufficient source support; no clamping")
    x, y = [math.floor(float(v) + .5) for v in xy]
    radius = size // 2
    return gray[y - radius:y + radius + 1, x - radius:x + radius + 1].copy(), [x, y]


def render(plan_path, output):
    """Decode only hash-verified0240 sequentially; write new lossless native crops."""
    import cv2
    plan_path, output = Path(plan_path), Path(output)
    plan = validate_plan(read_json(plan_path.read_text()))
    expected = build_plan(**{k + "_path": v for k, v in plan["input_paths"].items()})
    require(plan == expected, "plan differs from deterministic frozen selection")
    require(not output.exists() and not output.is_symlink(), "render destination exists; no overwrite")
    plan_digest = sha(plan_path)
    output.mkdir()
    cap = cv2.VideoCapture(plan["input_paths"]["source"])
    require(cap.isOpened(), "could not open verified0240")
    crops, records, contexts, count = {}, [], [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            require(count < FRAMES and bgr.shape == (HEIGHT, WIDTH, 3) and bgr.dtype == np.uint8,
                    "source decoded shape/count/type differs")
            if str(count) in plan["frame_points"]:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                for p in plan["frame_points"][str(count)]:
                    patch, center = native_crop(gray, p["source_xy"])
                    name = p["point_id"].replace("/", "_") + f"_f{count:03d}_native.png"
                    path = output / name
                    require(not path.exists() and cv2.imwrite(str(path), patch), "crop write failed")
                    require(np.array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), patch), "saved native crop differs")
                    crops[(p["point_id"], count)] = patch
                    records.append(dict(point_id=p["point_id"], frame_index=count, source_crop_center_xy=center,
                        source_xy=p["source_xy"], reference_xy=p["reference_xy"], path=name, sha256=sha(path)))
                for episode in plan["episodes"]:
                    if episode["anchor_frame"] != count or episode["sample_status"] != "available":
                        continue
                    p = next(p for p in plan["frame_points"][str(count)]
                             if p["point_id"] == episode["episode_id"] + "/sample")
                    record = dict(episode_id=episode["episode_id"], frame_index=count, size=CONTEXT,
                        source_xy=p["source_xy"], status="insufficient_source_support_no_clamping", files=[])
                    if full_patch(p["source_xy"], CONTEXT):
                        native, center = native_crop(gray, p["source_xy"], CONTEXT)
                        marked = cv2.cvtColor(native, cv2.COLOR_GRAY2BGR)
                        middle = CONTEXT // 2
                        cv2.circle(marked, (middle, middle), 7, (0, 200, 255), 1, cv2.LINE_8)
                        for delta in (-1, 1):
                            cv2.line(marked, (middle + delta * 10, middle), (middle + delta * 16, middle), (0, 200, 255), 1)
                            cv2.line(marked, (middle, middle + delta * 10), (middle, middle + delta * 16), (0, 200, 255), 1)
                        for kind, pixels in (("unmarked", native), ("marked", marked)):
                            name = episode["episode_id"] + f"_context_{kind}_native.png"
                            path = output / name
                            require(not path.exists() and cv2.imwrite(str(path), pixels), "context write failed")
                            if kind == "unmarked":
                                require(np.array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), native), "context source pixels changed")
                            record["files"].append(dict(kind=kind, path=name, sha256=sha(path)))
                        record.update(status="available", source_crop_center_xy=center,
                            marker="Orientation only: nearest source pixel to the selected anchor peak; not a new detection")
                    contexts.append(record)
            count += 1
    finally:
        cap.release()
    require(count == FRAMES, "incomplete source decoding")
    sheets = []
    for e in plan["episodes"]:
        if e["sample_status"] != "available":
            continue
        canvas = np.full((386, 5 * 144, 3), 24, np.uint8)
        cv2.putText(canvas, e["episode_id"] + " | native source / not detector input", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
        cv2.putText(canvas, "U8 identity contrast 0..255 shared | nearest 8x | uncertain comparator", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
        for ri, role in enumerate(("sample", "comparator")):
            for ci, f in enumerate(e["frames"]):
                p = crops.get((e["episode_id"] + "/" + role, f))
                if p is None:
                    continue
                y, x = 68 + ri * 157, ci * 144 + 4
                zoom = cv2.resize(p, (136, 136), interpolation=cv2.INTER_NEAREST)
                canvas[y:y + 136, x:x + 136] = zoom[..., None]
                cv2.putText(canvas, f"{role} f{f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .38, (240, 240, 240), 1)
        name = e["episode_id"] + "_contact.png"
        require(cv2.imwrite(str(output / name), canvas), "contact sheet write failed")
        sheets.append(dict(path=name, sha256=sha(output / name)))
    require(sha(plan_path) == plan_digest, "plan changed during rendering")
    for key, path in plan["input_paths"].items():
        verify_file(path, plan["input_sha256"][key])
    receipt = dict(schema=SCHEMA + ".source-render", passed=True, plan_sha256=plan_digest,
        source_sha256=SOURCE_SHA, decoded_frames=count, crops=records, contact_sheets=sheets, anchor_contexts=contexts,
        contrast="identity U8 mapping0..255 shared across every crop; no per-patch stretch",
        detector_run=False, source_appearance_only=True, raw16_accessed=False, sealed_holdouts_accessed=False)
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-output", type=Path)
    action.add_argument("--render-plan", type=Path)
    parser.add_argument("--render-output", type=Path)
    args = parser.parse_args()
    if args.plan_output:
        require(args.render_output is None, "render output only with render plan")
        write_json(args.plan_output, build_plan())
    else:
        require(args.render_output is not None, "render output required")
        render(args.render_plan, args.render_output)


if __name__ == "__main__":
    main()
