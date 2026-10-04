"""Serial, fresh-process clean/probed replay; stop on any failed gate."""
import subprocess
import sys
from pathlib import Path

from run_accuracy_v56_diagnostic import verify, write

HERE = Path(__file__).resolve().parent


def main():
    verify()
    if any((HERE/name).exists() for name in ('batch_status.json', 'clean', 'probe', 'clean.v56.json', 'probe.v56.json')):
        raise FileExistsError('Fresh batch outputs required')
    rows = []
    passed, error = False, None
    try:
        for arm in ('clean', 'probe', 'audit'):
            command = ([sys.executable, str(HERE/'run_accuracy_v56_diagnostic.py'), arm]
                       if arm != 'audit' else [sys.executable, str(HERE/'audit_accuracy_v56_replay.py'),
                            '--run', str(HERE), '--output', str(HERE/'independent_audit.json')])
            print('Starting', arm, flush=True)
            with (HERE/(arm+'.log')).open('x') as log:
                done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            rows.append(dict(arm=arm, command=command, returncode=done.returncode))
            if done.returncode:
                raise RuntimeError(arm+' failed; existing evidence retained')
            print('Completed', arm, flush=True)
        passed = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(HERE/'batch_status.json', dict(passed=passed, error=error, runs=rows,
                                            concurrent_workers=1, settings_changed=False))


if __name__ == '__main__':
    main()
