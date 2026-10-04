"""Package only the frozen experimental runtime and its PVA configuration."""
import argparse
import json
from pathlib import Path
import tarfile
from run_phase20_maturity import write
from score_phase20_accuracy import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    frozen = json.loads((a.root / "freeze.json").read_text())
    if digest(a.root / "config.json") != frozen["config_sha256"]:
        raise ValueError("Configuration changed")
    for name, sha in frozen["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Implementation changed")
    cfg = json.loads((a.root / "config.json").read_text())
    cfg["motion_backend"] = "pva"
    cp = a.root / "pva_config.json"
    write(cp, cfg)
    archive = a.root / "pva_runtime.tar.gz"
    with archive.open("xb") as f, tarfile.open(fileobj=f, mode="w:gz") as tar:
        for name in sorted(frozen["code_sha256"]):
            tar.add(ROOT / name, arcname=name, recursive=False)
        motion = "configs/evaluation/phase20_motion_v8.json"
        tar.add(ROOT / motion, arcname=motion, recursive=False)
        tar.add(cp, arcname="pva_config.json", recursive=False)
    write(a.root / "pva_transfer_manifest.json", dict(
        archive_sha256=digest(archive), config_sha256=digest(cp),
        motion_config_sha256=digest(ROOT / motion),
        implementation_sha256=frozen["code_sha256"],
        source_clip="chunk_0126.avi", maximum_frames=230,
        config_change_from_cpu={"motion_backend": ["cpu_translation", "pva"]}))
    print(archive.resolve())


if __name__ == "__main__":
    main()
