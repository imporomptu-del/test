"""Bounded single-worker candidate experiment; no clocks, installs or retries."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HELPER_SHA = 'a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf'


def main(workspace):
    workspace = Path(workspace)
    if not re.fullmatch(r'/tmp/seaqr_feature_supply_20260929_[A-Za-z0-9]{6}', str(workspace)):
        raise ValueError('wrong experiment workspace')
    if not workspace.is_dir() or workspace.is_symlink() or os.geteuid() == 0:
        raise ValueError('real unprivileged workspace required')
    helper_path = workspace/'batch_discovery_pair.py'
    if helper_path.is_symlink() or hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('supervisor helper identity differs')
    spec = importlib.util.spec_from_file_location('frozen_discovery_safety', helper_path)
    safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety)
    require, sha = safety.require, safety.sha
    freeze_path = workspace/'freeze.json'
    require(not freeze_path.is_symlink(), 'linked freeze')
    freeze = json.loads(freeze_path.read_text())
    require(freeze['schema'] == 'feature_supply.v1' and set(freeze['sources']) == {'0170', '0240'}, 'wrong scope')
    require({'batch_feature_supply.py', 'batch_discovery_pair.py', 'run_discovery_feature_supply.py',
             'feature_supply_plan.json'} <= set(freeze['files']), 'incomplete bundle')
    freeze_sha = sha(freeze_path)

    def check_bundle():
        require(sha(freeze_path) == freeze_sha, 'freeze changed')
        for name, digest in freeze['files'].items():
            require(Path(name).name == name and name not in ('.', '..', 'freeze.json')
                    and not (workspace/name).is_symlink(), 'unsafe bundle member')
            require(sha(workspace/name) == digest, 'bundle changed: '+name)

    check_bundle()
    for name in ('batch_status.json', 'batch_status.tmp', 'batch.lock', 'telemetry.jsonl'):
        require(not (workspace/name).exists() and not (workspace/name).is_symlink(), 'existing batch evidence')
    began = time.monotonic()
    status = dict(schema='seaqr.feature-supply.batch.v1', complete=False, passed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=freeze_sha,
                  supervisor_pid=os.getpid(), phases=[], current=None, error=None,
                  clock_writes_performed=False, production_changed=False)
    child = None

    def save():
        status['elapsed_s'] = time.monotonic()-began
        temp = workspace/'batch_status.tmp'
        with temp.open('x') as stream:
            json.dump(status, stream, indent=2, allow_nan=False)
        os.replace(temp, workspace/'batch_status.json')

    def interrupted(sig, frame):
        raise RuntimeError('supervisor signal '+str(sig))

    with (workspace/'batch.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            save()
            with (workspace/'telemetry.jsonl').open('x') as telemetry:
                for mode in ('preflight', 'run'):
                    for clip in ('0170', '0240'):
                        check_bundle()
                        require(max(safety.temperatures().values()) < 65, 'too warm to start')
                        name = mode+'_'+clip
                        cmd = [sys.executable, '-I', '-u', str(workspace/'run_discovery_feature_supply.py'),
                               '--workspace', str(workspace), '--clip', clip, '--'+mode]
                        phase = dict(name=name, command=cmd, elapsed_s=None, returncode=None)
                        status['phases'].append(phase)
                        status['current'] = name
                        start = time.monotonic()
                        with (workspace/(name+'.log')).open('x') as log:
                            child = subprocess.Popen(cmd, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                                     stdin=subprocess.DEVNULL, start_new_session=True)
                            phase['pid'] = child.pid
                            save()
                            print(json.dumps(dict(phase=name, status='starting')), flush=True)
                            while child.poll() is None:
                                temps = safety.temperatures()
                                telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                    phase=name, temperatures_c=temps))+'\n')
                                telemetry.flush()
                                require(max(temps.values()) < 75, 'temperature limit reached')
                                require(time.monotonic()-start < 900, 'phase deadline reached')
                                require(time.monotonic()-began < 3600, 'batch deadline reached')
                                time.sleep(2)
                            phase.update(returncode=child.returncode, elapsed_s=time.monotonic()-start)
                            require(child.returncode == 0, 'failed phase '+name)
                        child = None
                        save()
                        print(json.dumps(dict(phase=name, status='complete', elapsed_s=phase['elapsed_s'])), flush=True)
            check_bundle()
            status.update(complete=True, passed=True, current=None)
        except BaseException as exc:
            status['error'] = repr(exc)
            if child is not None:
                safety.stop_owned(child)
                status['owned_child_stopped'] = True
            raise
        finally:
            save()
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    main(parser.parse_args().workspace)
