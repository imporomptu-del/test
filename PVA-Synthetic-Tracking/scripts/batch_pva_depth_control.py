#!/usr/bin/env python3
"""Four fresh two-level probes using the pinned, unchanged safety supervisor.

Only the scoped bundle/command/child-schema callbacks and output schema are
rebound inside this process. The original process ownership, stop, thermal,
deadline, sequential execution, no-retry and parity implementation is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

SCHEMA="seaqr.pva-depth-control.batch.v1"
REFERENCE_WORKSPACE=Path("/tmp/seaqr_static_pva_texture_20261001_SKYCZ8")
REFERENCE_BATCH_SHA="2f74bac6432a93d62e558eb59798baf033892f81f4c3d114f741397b0c7471c7"
PHASES=(("bridge","base"),("bridge","trace"),("texture","base"),("texture","trace"))


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def imported(path,digest,name):
    require(path.is_file() and not path.is_symlink() and sha(path)==digest,"Pinned supervisor/helper differs")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(workspace,freeze_path,digest):
    workspace,freeze_path=Path(workspace),Path(freeze_path)
    require(re.fullmatch(r"/tmp/seaqr_pva_depth_control_20261001_[A-Za-z0-9]{6}",str(workspace))
        and workspace.is_dir() and not workspace.is_symlink() and os.geteuid()!=0,"Unprivileged scoped workspace required")
    require(freeze_path==workspace/"freeze.json" and freeze_path.is_file() and not freeze_path.is_symlink()
        and re.fullmatch(r"[a-f0-9]{64}",digest) and sha(freeze_path)==digest,"Caller-bound freeze differs")
    supervisor=imported(REFERENCE_WORKSPACE/"batch_static_pva_texture.py",REFERENCE_BATCH_SHA,"depth_original_supervisor")
    frozen=supervisor.read(freeze_path)
    require(isinstance(frozen.get("files"),dict),"Missing frozen file identities")
    require(sha(Path(__file__))==frozen["files"].get("batch_pva_depth_control.py"),"Depth supervisor identity differs")
    probe=imported(workspace/"probe_pva_depth_control.py",frozen["files"].get("probe_pva_depth_control.py"),"depth_worker")
    old=probe.old_helper()
    require(probe.bundle(workspace,freeze_path,digest,old)==frozen,"Depth bundle differs")
    require(supervisor.PHASES==PHASES and supervisor.EXECUTION==probe.EXECUTION,"Original supervision policy differs")
    return supervisor,probe,old,frozen


def command(workspace,digest,case,mode):
    require((case,mode) in PHASES,"Undeclared case/mode")
    args=[sys.executable,"-I",str(Path(workspace)/"probe_pva_depth_control.py"),"--workspace",str(workspace),
        "--freeze",str(Path(workspace)/"freeze.json"),"--freeze-sha256",digest,"--case",case]
    if mode=="trace":args.append("--trace")
    return args


def validate_result(row,case,mode,digest,old):
    require(row.get("schema")=="seaqr.pva-depth-control.v1" and row.get("completed") is True
        and row.get("passed_integrity") is True and row.get("case")==case and row.get("mode")==mode
        and row.get("input_sha256",{}).get("freeze_sha256")==digest
        and row.get("pyramid_depth_changed") is True and row.get("other_motion_settings_changed") is False,
        "Depth child contract differs")
    canonical=row.get("canonical_nontiming")
    require(isinstance(canonical,dict) and canonical.get("effective_motion_configuration",{}).get("pyramid_levels")==2
        and old.canonical_sha(canonical)==row.get("canonical_nontiming_sha256"),"Depth canonical output differs")


def run(workspace,freeze_path,digest):
    workspace=Path(workspace)
    supervisor,probe,old,frozen=load(workspace,freeze_path,digest)
    comparison_path=workspace/"depth_comparison.json"
    failure_path=workspace/"depth_comparison_failure.json"
    require(all(not path.exists() and not path.is_symlink() for path in (comparison_path,failure_path)),
            "Existing comparison; no overwrite")
    original_bundle=lambda workspace,freeze_path,digest:probe.bundle(workspace,freeze_path,digest,old)
    child_check=lambda row,case,mode,digest:validate_result(row,case,mode,digest,old)
    with patch.object(supervisor,"bundle",original_bundle),patch.object(supervisor,"command",command), \
         patch.object(supervisor,"validate_result",child_check),patch.object(supervisor,"SCHEMA",SCHEMA):
        status=supervisor.run(workspace,freeze_path,digest)
    try:
        require(probe.bundle(workspace,freeze_path,digest,old)==frozen,"Bundle changed before comparison")
        require([(p["case"],p["mode"]) for p in status["phases"]]==list(PHASES),"Completed phase inventory differs")
        hashes={"batch_status.json":sha(workspace/"batch_status.json"),"parity.json":status["parity_sha256"]}
        require(old.read(workspace/"batch_status.json")==status,"Saved supervisor receipt differs")
        old.pinned(workspace/"parity.json",status["parity_sha256"])
        rows={}
        for phase in status["phases"]:
            name=phase["case"]+"_"+phase["mode"]
            require(phase["name"]==name and phase["returncode"]==0,"Completed phase role/status differs")
            path=workspace/(name+".json")
            old.pinned(path,phase["result_sha256"])
            rows[name]=old.read(path)
            validate_result(rows[name],phase["case"],phase["mode"],digest,old)
            hashes[name+".json"]=phase["result_sha256"]
        comparison=probe.scientific_comparison(rows,probe.references(old),status["parity_passed"])
        comparison.update(schema=SCHEMA+".comparison",freeze_sha256=digest,protocol=probe.PROTOCOL,
            input_sha256=hashes,reference_results=probe.REFERENCE_RESULTS,
            reference_freeze_sha256=probe.REFERENCE_FREEZE_SHA,original_supervisor_sha256=REFERENCE_BATCH_SHA)
        require(probe.bundle(workspace,freeze_path,digest,old)==frozen,"Bundle changed after comparison")
        probe.references(old)
        for name,expected in hashes.items():old.pinned(workspace/name,expected)
        with comparison_path.open("x") as stream:json.dump(comparison,stream,indent=2,allow_nan=False);stream.write("\n")
    except BaseException as exc:
        with failure_path.open("x") as stream:
            json.dump(dict(schema=SCHEMA+".comparison-failure",hardware_phases_complete=status["complete"],
                scientific_comparison_complete=False,error=repr(exc),freeze_sha256=digest),stream,indent=2,allow_nan=False)
        raise
    return dict(complete=status["complete"],execution_passed=status["execution_passed"],parity_passed=status["parity_passed"],
        interpretable=comparison["interpretable"],depth_comparison_sha256=sha(comparison_path))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True);args=parser.parse_args()
    print(json.dumps(run(args.workspace,args.freeze,args.freeze_sha256)))
