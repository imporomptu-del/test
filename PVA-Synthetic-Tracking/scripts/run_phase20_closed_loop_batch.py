"""One-worker frozen development execution; no tuning, labels or media search."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256


def main():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    if set(frozen["sources"]) != {"0126", "0029", "0055", "0082"}:
        raise ValueError("Unexpected source scope")
    for path, expected in frozen["files_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError("Frozen runtime changed: " + path)
    if sha256(ROOT / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    if sha256(ROOT / "libseaqr_integrated.so") != frozen["compiled_library_sha256"]:
        raise ValueError("Compiled library changed")
    if 'gpu_transition' in frozen:
        build_path=ROOT/'libseaqr_integrated.so.build.json'
        build=json.loads(build_path.read_text())
        transition=frozen['gpu_transition']
        if (sha256(build_path)!=transition['candidate_build_sha256']
                or build['library_sha256']!=transition['after_library_sha256']
                or build['reference_library_sha256']!=transition['before_library_sha256']
                or transition['after_library_sha256']!=frozen['compiled_library_sha256']
                or build['diagnostic_only'] is not False):
            raise ValueError('GPU build/transition provenance changed')
        for name,expected in build['sources_sha256'].items():
            if sha256(ROOT/'scripts'/name)!=expected:
                raise ValueError('GPU build source changed: '+name)
    if 'native_shape_build' in frozen:
        if (sha256(ROOT/'libseaqr_shapes.so') != frozen['native_shape_build']['library_sha256']
                or sha256(ROOT/'libseaqr_shapes.so.build.json') != frozen['native_shape_build_sha256']):
            raise ValueError('Native shape library/build changed')
    names = set()
    for job in frozen["jobs"]:
        cid = job["clip_id"]
        if cid not in frozen["sources"] or job["name"] != "pva_" + cid or job["name"] in names:
            raise ValueError("Unexpected or duplicate job")
        if job["max_frames"] is not None:
            raise ValueError("This batch requires full clips")
        names.add(job["name"])
    lock = (ROOT / "batch.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(running=True, current=None, completed=[], error=None, started_unix=time.time())

    def save():
        temporary = ROOT / "status.pending.json"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(ROOT / "status.json")

    try:
        for job in frozen["jobs"]:
            name, cid = job["name"], job["clip_id"]
            source = frozen["sources"][cid]
            if source["path"] != "/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_" + cid + ".avi":
                raise ValueError("Source path outside the explicit allowlist")
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError("Source changed")
            output = ROOT / name
            if output.exists():
                raise ValueError("Never overwrite an output")
            status.update(current=name, current_started_unix=time.time())
            save()
            print(json.dumps(dict(start=name)), flush=True)
            command = [sys.executable, "-m", "tiny_target.visible_baseline", "--source", source["path"],
                "--config", str(ROOT / "config.json"), "--motion-config", str(ROOT / "configs/evaluation/phase20_motion_v8.json"),
                "--output", str(output)]
            with (ROOT / (name + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10800)
            report = json.loads((output / "report.json").read_text())
            if not report["completed"] or not report["full_clip"] or report["frames"] != source["frames"]:
                raise ValueError("Incomplete result")
            if "execution_only_reference" in frozen:
                from compare_phase20_exact_runs import compare
                reference = frozen["execution_only_reference"]
                before = Path(reference["workspace"]) / name
                for filename, expected in reference["artifacts_sha256"][name].items():
                    if sha256(before / filename) != expected:
                        raise ValueError("Execution-only reference changed")
                compare(before, output, output / "comparison.json", gpu_transition=frozen.get('gpu_transition'))
            entry = dict(name=name, clip_id=cid, frames=report["frames"], fps=report["processed_fps"],
                         qualified_proposal_workload=report["qualified_track_count"])
            status["completed"].append(entry)
            save()
            print(json.dumps(entry), flush=True)
    except BaseException as exc:
        status["error"] = repr(exc)
        raise
    finally:
        status.update(running=False, finished_unix=time.time())
        save()
        lock.close()


if __name__ == "__main__":
    main()
