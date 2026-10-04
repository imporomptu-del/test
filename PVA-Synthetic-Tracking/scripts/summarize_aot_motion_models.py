#!/usr/bin/env python3
"""Compact, exclusive postprocessing of the completed local motion-model result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO.parent / "outputs/seaqr_aot_pilot_20260927/motion_models_01"
MODELS = ("translation", "similarity", "affine")
PREVIOUS = (0, 42, 85, 127, 170, 212, 255, 298)
SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick(source, keys):
    return {key: source[key] for key in keys.split() if key in source}


def compact_fit(fit):
    if fit is None:
        return None
    return pick(fit, "valid reason error model iterations_requested iterations_completed solver_calls "
        "huber_delta_px rank singular_values condition_number parameters native_matrix native_linear_determinant "
        "final_solver_weight_note")


def compact_fold(fold):
    result = pick(fold, "fold_id model test_rectangle excluded_rectangle train_count test_count guard_count "
        "train_cell_counts train_occupied_cells valid unavailable_reasons test_error_summary "
        "known_transform_prediction_error_summary")
    result["fit"] = compact_fit(fold["fit"])
    return result


def summarize(source, source_hash, script_hash):
    require(source.get("schema") == "seaqr.aot.motion-models.v1" and source.get("passed") is True,
            "only a completed passed model comparison may be summarized")
    require(source.get("hashes_before") == source.get("hashes_after") and bool(source.get("hashes_before")),
            "comparison input/artifact integrity differs")
    expected = ([f"aot_{index + 1:03d}_adjacent" for index in PREVIOUS]
        + [f"aot_prev{index:03d}_dx{shift[0]:+d}_dy{shift[1]:+d}" for index in PREVIOUS for shift in SHIFTS])
    require([row["case_id"] for row in source["rows"]] == expected, "fixed 64-case inventory differs")
    rows = []
    for ordinal, row in enumerate(source["rows"]):
        require(row["ordinal"] == ordinal and set(row["models"]) == set(MODELS), "case/model order differs")
        result = pick(row, "ordinal case_id source_kind previous_index current_index arm selected_count "
            "accepted_count lost_count original_fixed_support expected_shift_xy original_scientific_gates "
            "original_candidate_translation descriptive_differences")
        original = row.get("original_fit")
        result["original_fit"] = None if original is None else pick(original,
            "model mapping parameters previous_to_current_matrix quality_status rejection_reasons metrics")
        result["models"] = {}
        for model in MODELS:
            item = row["models"][model]
            out = item["out_of_fold"]
            require(out["expected_count"] == row["accepted_count"]
                and out["scored_count"] + out["unscored_count"] == out["expected_count"]
                and out["complete"] == (out["unscored_count"] == 0), "evaluation denominator differs")
            require([fold["fold_id"] for fold in item["folds"]] == list(range(4)), "fold inventory differs")
            require(len(out["cell_errors"]) == 48 and sum(cell["count"] for cell in out["cell_errors"]) == out["scored_count"]
                and sum(cell["expected_count"] for cell in out["cell_errors"]) == out["expected_count"]
                and sum(cell["unscored_count"] for cell in out["cell_errors"]) == out["unscored_count"], "cell support differs")
            result["models"][model] = dict(folds=[compact_fold(fold) for fold in item["folds"]],
                out_of_fold=pick(out, "complete expected_count scored_count unscored_count valid_fold_count "
                    "unavailable_fold_count error_summary summary_scope cell_errors supported_cell_count "
                    "worst_supported_cell_median worst_supported_cell_p90 known_transform_prediction_error_summary "
                    "known_transform_interpretation"))
        rows.append(result)
    return dict(schema="seaqr.aot.motion-models-summary.v1", passed=True, source_result_sha256=source_hash,
        summarizer_sha256=script_hash, source_artifact_hashes=source["hashes_after"], design=source["design"],
        limitations=source["limitations"], automatic_model_selection=False, production_promotion=False,
        interpretation="Saved-match agreement is not physical motion truth. Known-shift mapping error is separate from raw LK error and lost-track recovery.",
        omitted="Per-point arrays, indices and training weights remain unchanged in the full source result.", rows=rows)


def figure(summary):
    """One shared-scale view; incomplete model/case results are not plotted."""
    from PIL import Image, ImageDraw, ImageFont
    def font(size):
        for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()
    image = Image.new("RGB", (1800, 1130), "white")
    draw = ImageDraw.Draw(image)
    title, normal, small = font(32), font(21), font(17)
    colors = dict(translation="#2166ac", similarity="#1b7837", affine="#b35806")
    actual = summary["rows"][:8]
    values = [row["models"][model]["out_of_fold"]["error_summary"]["quantiles"][metric]
        for row in actual for model in MODELS for metric in ("p50", "p90")
        if row["models"][model]["out_of_fold"]["complete"] and row["models"][model]["out_of_fold"]["scored_count"]]
    maximum = max(values, default=1.0)
    require(all(math.isfinite(value) and value >= 0 for value in values), "invalid plot errors")
    upper = max(1.0, math.ceil(maximum * 1.1))
    draw.text((45, 28), "Spatial holdout: actual adjacent AOT pairs", font=title, fill="black")
    draw.text((45, 77), "Out-of-fold prediction error against saved accepted matches — not physical motion truth", font=normal, fill="#333333")
    for index, model in enumerate(MODELS):
        x = 50 + index * 280
        draw.ellipse((x, 117, x + 14, 131), fill=colors[model])
        draw.text((x + 23, 111), model, font=normal, fill=colors[model])
    lefts, panel_width = (260, 1040), 610
    for panel, metric in enumerate(("p50", "p90")):
        left = lefts[panel]
        draw.text((left, 155), "Median error" if metric == "p50" else "90th-percentile error", font=normal, fill="black")
        for tick in range(6):
            x = left + panel_width * tick / 5
            draw.line((x, 200, x, 990), fill="#dddddd", width=1)
            draw.text((x - 16, 1000), f"{upper * tick / 5:g}", font=small, fill="#444444")
        for ordinal, row in enumerate(actual):
            y = 245 + ordinal * 100
            if panel == 0:
                label = f"{row['previous_index']} → {row['current_index']}"
                draw.text((45, y - 15), label, font=normal, fill="black")
                draw.text((45, y + 12), f"accepted n={row['accepted_count']}", font=small, fill="#555555")
            for index, model in enumerate(MODELS):
                out = row["models"][model]["out_of_fold"]
                yy = y + (index - 1) * 24
                if not out["complete"] or not out["scored_count"]:
                    draw.text((left + 8, yy - 10), f"{model}: incomplete/unavailable", font=small, fill=colors[model])
                    continue
                value = out["error_summary"]["quantiles"][metric]
                x = left + panel_width * value / upper
                draw.line((left, yy, x, yy), fill=colors[model], width=2)
                draw.ellipse((x - 5, yy - 5, x + 5, yy + 5), fill=colors[model])
                draw.text((x + 9, yy - 11), f"{value:.2f}", font=small, fill=colors[model])
        draw.text((left, 1040), "Native pixels — same axis limits in both panels", font=small, fill="#444444")
    draw.text((45, 1090), "Every original accepted point is tested once. No original-inlier filtering; no automatic model promotion.", font=small, fill="#333333")
    return image, dict(filename="actual_holdout_errors.png", plotted_cases=8, metrics=["p50", "p90"],
        shared_axis_limits_px=[0, upper], display_rule="upper=max(1,ceil(1.1*largest complete-case plotted value)); incomplete cases not plotted",
        comparison_target="saved accepted correspondences, not physical motion truth")


def run(output, make_figure=False):
    output = Path(output)
    require(output == OUTPUT_DIR and output.resolve() == OUTPUT_DIR and output.is_dir() and not output.is_symlink(), "outside fixed summary scope")
    targets = [output / "summary.json"] + ([output / "actual_holdout_errors.png"] if make_figure else [])
    require(all(not path.exists() and not path.is_symlink() for path in targets), "existing summary/figure; refusing overwrite")
    source_path = output / "result.json"
    require(source_path.is_file() and not source_path.is_symlink(), "missing/linked full result")
    source_hash, script_hash = sha(source_path), sha(Path(__file__))
    with source_path.open() as stream:
        summary = summarize(json.load(stream), source_hash, script_hash)
    picture = None
    if make_figure:
        picture, summary["figure"] = figure(summary)
    require(sha(source_path) == source_hash and sha(Path(__file__)) == script_hash, "result/postprocessor changed while summarizing")
    if picture is not None:
        with (output / "actual_holdout_errors.png").open("xb") as stream:
            picture.save(stream, format="PNG")
        summary["figure"]["sha256"] = sha(output / "actual_holdout_errors.png")
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(dict(summary=str(output / "summary.json"), sha256=sha(output / "summary.json"), source_result_sha256=source_hash)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    run(args.output, args.figure)


if __name__ == "__main__":
    main()
