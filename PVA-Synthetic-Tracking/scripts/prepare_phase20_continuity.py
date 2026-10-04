"""Freeze the single coast-appearance trial before fresh closed-loop execution."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    previous = ROOT / "results/tiny_target/phase20/efficiency_v4_20260914/pipeline"
    old = json.loads((previous / "freeze.json").read_text())
    sources = old["sources"]
    if set(sources) != {"0126", "0029", "0055", "0082"}:
        raise ValueError("Unexpected source scope")
    base = json.loads((previous / "resident_config.json").read_text())
    config = asdict(VisibleConfig(**dict(base,
        tracking_association_appearance="log_response_coast",
        cuda_median_library=args.workspace + "/libseaqr_integrated.so")))
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    with config_path.open("x") as handle:
        json.dump(config, handle, indent=2)
    paths = sorted((ROOT / "tiny_target").rglob("*.py"))
    paths += [ROOT / "scripts/run_phase20_closed_loop_batch.py", ROOT / "configs/evaluation/phase20_motion_v8.json"]
    frozen = dict(schema="seaqr.frozen-closed-loop-development.v1",
        sources=sources,
        files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        compiled_library_sha256=sha256(previous / "libseaqr_integrated.so"),
        parent_freeze_sha256=sha256(previous / "freeze.json"),
        config_sha256=sha256(config_path),
        algorithm_change="Retain discounted measured log-response evidence only within the existing coast budget.",
        settings_not_changed="Thresholds, position gates, independent confirmation, shape model, caps, native sampling and cadence.",
        labels_used_during_processing=False,
        jobs=[dict(name="pva_" + cid, clip_id=cid, max_frames=None) for cid in ("0029", "0126", "0055", "0082")])
    freeze_path = args.output / "freeze.json"
    with freeze_path.open("x") as handle:
        json.dump(frozen, handle, indent=2)
    with tarfile.open(args.output / "runtime.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        for path in (config_path, freeze_path):
            archive.add(path, arcname=path.name)
    print(json.dumps(dict(workspace=args.workspace, archive=str(args.output / "runtime.tar.gz"),
                          runtime_sha256=sha256(args.output / "runtime.tar.gz")), indent=2))


if __name__ == "__main__":
    main()
