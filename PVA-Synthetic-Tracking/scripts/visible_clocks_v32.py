"""One-shot privileged clock guard; immutable video children run as serg."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import select
import signal
import stat
import subprocess
import sys
import time

V31 = Path('/tmp/seaqr_visible_threads_v31_QvwNyq')
HERE = Path(__file__).resolve().parent
SOURCES = ('visible_clocks_v32.py', 'test_visible_clocks_v32.py', 'visible_clocks_v32_plan.md')
MODES = ('v26_default', 'combined_default', 'v26', 'combined')
ORDERS = (MODES, tuple(reversed(MODES)), ('combined_default', 'v26', 'combined', 'v26_default'))
POLICIES = {
    **{f'cpu{i}': dict(path=f'/sys/devices/system/cpu/cpufreq/policy{i}',
        minimum='scaling_min_freq', maximum='scaling_max_freq', current='scaling_cur_freq',
        governor='scaling_governor', available='scaling_available_frequencies',
        expected_min=729600, expected_max=2201600, expected_governor='schedutil') for i in (0, 4, 8)},
    'gpu': dict(path='/sys/class/devfreq/17000000.gpu', minimum='min_freq', maximum='max_freq',
        current='cur_freq', governor='governor', available='available_frequencies',
        expected_min=306000000, expected_max=1300500000, expected_governor='nvhost_podgov'),
}
MIN_PATHS = {p['path']+'/'+p['minimum'] for p in POLICIES.values()}
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GOTO_NUM_THREADS')


def read_node(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        value = os.read(fd, 65536)
        if len(value) == 65536:
            raise ValueError('Oversized sysfs value')
        return value.decode().strip()
    finally:
        os.close(fd)


def write_min(path, value):
    if str(path) not in MIN_PATHS or type(value) is not int:
        raise ValueError('Write outside four integer frequency floors')
    # Only existing kernel controls, never create/truncate an arbitrary file.
    fd = os.open(str(path), os.O_WRONLY)
    try:
        data = (str(value)+'\n').encode()
        if os.write(fd, data) != len(data):
            raise OSError('Short sysfs write')
    finally:
        os.close(fd)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def snapshot(reader=read_node):
    return {name: {key: (reader(p['path']+'/'+p[key]) if key == 'governor'
        else int(reader(p['path']+'/'+p[key]))) for key in ('minimum', 'maximum', 'governor')}
        for name, p in POLICIES.items()}


def validate_original(saved, reader=read_node):
    if set(saved) != set(POLICIES):
        raise ValueError('Unexpected frequency policies')
    for name, p in POLICIES.items():
        wanted = dict(minimum=p['expected_min'], maximum=p['expected_max'], governor=p['expected_governor'])
        if saved[name] != wanted:
            raise ValueError('Changed original clock policy '+name)
        available = [int(x) for x in reader(p['path']+'/'+p['available']).split()]
        if wanted['minimum'] not in available or wanted['maximum'] not in available:
            raise ValueError('Unsupported frequency '+name)
    if reader('/sys/devices/system/cpu/online') != '0-11':
        raise ValueError('Unexpected online CPU topology')


def expected_policy(saved, fixed):
    if type(fixed) is not bool:
        raise ValueError('Boolean clock mode required')
    return {n: dict(v, minimum=v['maximum'] if fixed else v['minimum']) for n, v in saved.items()}


def check_policy(saved, fixed, reader=read_node):
    actual = snapshot(reader)
    if actual != expected_policy(saved, fixed):
        raise ValueError('Clock policy no longer matches declared arm')
    return actual


def set_policy(saved, fixed, reader=read_node, writer=write_min):
    # Max/governor changes by another actor invalidate rather than expand scope.
    actual = snapshot(reader)
    for n in POLICIES:
        if any(actual[n][k] != saved[n][k] for k in ('maximum', 'governor')):
            raise ValueError('External max/governor change '+n)
    target = expected_policy(saved, fixed)
    for n, p in POLICIES.items():
        if actual[n]['minimum'] != target[n]['minimum']:
            writer(p['path']+'/'+p['minimum'], target[n]['minimum'])
    return check_policy(saved, fixed, reader)


def restore(saved, reader=read_node, writer=write_min):
    # Continue across individual failed writes so one unavailable node cannot
    # prevent the other independently controlled policies being restored.
    errors = []
    for n, p in POLICIES.items():
        try:
            path = p['path']+'/'+p['minimum']
            if int(reader(path)) != saved[n]['minimum']:
                writer(path, saved[n]['minimum'])
        except BaseException as exc:
            errors.append(dict(policy=n, error=repr(exc)))
    try:
        actual = snapshot(reader)
    except BaseException as exc:
        actual = None
        errors.append(dict(error=repr(exc)))
    return dict(restored=not errors and actual == saved, errors=errors, actual=actual, saved=saved)


def sensors(reader=read_node):
    result = dict(monotonic_ns=time.monotonic_ns(), realtime_ns=time.time_ns(), temperatures={}, clocks={})
    for path in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        name = reader(path/'type')
        try:
            result['temperatures'][name] = int(reader(path/'temp'))
        except (OSError, ValueError) as exc:
            result['temperatures'][name] = dict(error=repr(exc))
    for n, p in POLICIES.items():
        try:
            result['clocks'][n] = int(reader(p['path']+'/'+p['current']))
        except (OSError, ValueError) as exc:
            result['clocks'][n] = dict(error=repr(exc))
    return result


def thermal_check(row, limit=75000):
    temps = row['temperatures']
    for name in ('cpu-thermal', 'tj-thermal'):
        if type(temps.get(name)) is not int or not 0 < temps[name] < limit:
            raise RuntimeError('Mandatory thermal sensor absent/unsafe: '+name)
    if any(type(v) is int and (v <= 0 or v >= limit) for v in temps.values()):
        raise RuntimeError('Thermal experiment cutoff')


def schedule():
    result = [dict(name='smoke_'+c, clip=c, mode='combined', fixed=True, audit=True)
              for c in ('0126', '0082')]
    for repeat, order in enumerate(ORDERS):
        clips = ('0126', '0082') if repeat != 1 else ('0082', '0126')
        for c in clips:
            fixed_order = (False, True) if (repeat+(c == '0082')) % 2 == 0 else (True, False)
            for fixed in fixed_order:
                for mode in order:
                    result.append(dict(name=f'{c}_repeat{repeat}_{"fixed" if fixed else "auto"}_{mode}',
                        clip=c, mode=mode, fixed=fixed, audit=False))
    return result


def command(spec, directory):
    if spec not in schedule():
        raise ValueError('Unscheduled or out-of-scope video request')
    cmd = ['/usr/bin/python3', str(V31/'run_visible_threads_v31.py'), '--clip', spec['clip'],
        '--mode', spec['mode'], '--output', str(directory/spec['name']), '--frames', '128']
    if spec['audit']:
        cmd += ['--audit']
    return cmd


def child_environment(mode, account):
    if mode not in MODES:
        raise ValueError('Unknown numerical policy')
    env = dict(PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name,
        LANG='C.UTF-8', PYTHONUNBUFFERED='1')
    if not mode.endswith('_default'):
        env['OPENBLAS_NUM_THREADS'] = '1'
    return env


def proc_identity(pid):
    value = Path(f'/proc/{pid}/stat').read_text()
    fields = value[value.rindex(')')+2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgrp=int(fields[2]))


def stop_owned(identity):
    if identity is None:
        return
    pid = identity['pid']
    try:
        if proc_identity(pid) != identity or identity['pgrp'] != pid:
            return
        os.killpg(pid, signal.SIGTERM)
        time.sleep(1)
        # The unreaped process-group leader remains our child (or was owned by
        # the dead controller). Do not signal a reused, unrelated PID/group.
        if proc_identity(pid) == identity:
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
        pass


def guard_reason(now, started, heartbeat, eof, lease=60, hard=3600):
    if eof:
        return 'controller_pipe_closed'
    if now-started >= hard:
        return 'hard_deadline'
    if now-heartbeat >= lease:
        return 'heartbeat_lost'
    return None


def watchdog(read_fd, ready_fd, saved, directory):
    # Independent process survives handled terminal/controller signals. No
    # imports from the user's video runtime execute with elevated privileges.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    started = heartbeat = time.monotonic()
    active, pending, reason, done = None, b'', None, False
    os.write(ready_fd, b'R')
    os.close(ready_fd)
    try:
        while not done:
            readable, _, _ = select.select([read_fd], [], [], .5)
            eof = False
            if readable:
                data = os.read(read_fd, 65536)
                eof = not data
                pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    message = json.loads(line)
                    heartbeat = time.monotonic()
                    if message['kind'] == 'active':
                        active = message['identity']
                    elif message['kind'] == 'done':
                        done = True
                    elif message['kind'] != 'heartbeat':
                        raise ValueError('Unexpected watchdog message')
            reason = guard_reason(time.monotonic(), started, heartbeat, eof)
            if reason:
                break
        if not done:
            stop_owned(active)
        result = restore(saved)
        result.update(reason='normal_exit' if done else reason, watchdog_pid=os.getpid())
        write(directory/'watchdog_restoration.json', result)
        os._exit(0 if result['restored'] else 2)
    except BaseException as exc:
        stop_owned(active)
        result = restore(saved)
        result.update(reason='watchdog_error', error=repr(exc))
        try:
            write(directory/'watchdog_failure.json', result)
        finally:
            os._exit(3)


class Guard:
    def __init__(self, saved, directory):
        r, w = os.pipe()
        rr, rw = os.pipe()
        self.pid = os.fork()
        if not self.pid:
            os.close(w)
            os.close(rr)
            watchdog(r, rw, saved, directory)
            os._exit(4)
        os.close(r)
        os.close(rw)
        self.fd = w
        try:
            ready, _, _ = select.select([rr], [], [], 5)
            if not ready or os.read(rr, 1) != b'R':
                raise RuntimeError('Restoration watchdog not ready')
        except BaseException:
            os.close(w)
            raise
        finally:
            os.close(rr)

    def send(self, kind='heartbeat', **value):
        data = (json.dumps(dict(kind=kind, **value))+'\n').encode()
        if os.write(self.fd, data) != len(data):
            raise RuntimeError('Lost watchdog heartbeat')

    def finish(self):
        self.send('done')
        os.close(self.fd)
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                if not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                    raise RuntimeError('Watchdog restoration verification failed')
                return
            time.sleep(.1)
        raise RuntimeError('Watchdog did not confirm restoration')


def verify_freeze(directory):
    f = read(directory/'freeze.json')
    if not f['pre_run'] or set(f['sources']) != set(SOURCES):
        raise ValueError('Incomplete source freeze')
    for n, digest in f['sources'].items():
        if sha(directory/n) != digest:
            raise ValueError('Source changed '+n)
    if f['v31_freeze_sha256'] != sha(V31/'freeze.json') or f['v31_batch_sha256'] != sha(V31/'batch.json'):
        raise ValueError('Frozen dependency changed')
    if f['unit_log_sha256'] != sha(directory/'unit.log') or '\nOK\n' not in (directory/'unit.log').read_text():
        raise ValueError('Unit test evidence invalid')
    if f['dependency_preflight_sha256'] != sha(directory/'dependency_preflight.json') or read(directory/'dependency_preflight.json')['returncode']:
        raise ValueError('Dependency preflight evidence invalid')
    return f


def competing_experiments():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if not args or int(path.parent.name) == os.getpid():
            continue
        executable = Path(args[0]).name
        video = (len(args) > 1 and executable.startswith('python')
            and args[1].startswith('/tmp/seaqr_')
            and Path(args[1]).name.startswith(('run_', 'batch_', 'profile_', 'replay_', 'check_')))
        if video or executable in ('nsys', 'tegrastats'):
            found.append(dict(pid=int(path.parent.name), args=args[:4]))
    return found


def exclusive_lock():
    # Held by controller AND its forked watchdog, never by video children.
    fd = os.open('/run/lock/seaqr-visible-clocks-v32.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if info.st_uid != 0 or not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError('Unsafe benchmark lock file')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def prepare(directory):
    if os.geteuid() == 0:
        raise ValueError('Prepare as normal user, not root')
    if any((directory/n).exists() for n in ('freeze.json', 'unit.log', 'run')):
        raise FileExistsError('Fresh prepared directory required')
    saved = snapshot()
    validate_original(saved)
    probe = sensors()
    thermal_check(probe, 65000)
    competing = competing_experiments()
    if competing:
        raise RuntimeError('Other experiment active: '+repr(competing))
    batch = read(V31/'batch.json')
    if not batch['passed'] or batch['error'] is not None or len(batch['rows']) != 42:
        raise ValueError('Completed v31 prerequisite required')
    with (directory/'unit.log').open('x') as log:
        subprocess.run(['/usr/bin/python3', '-m', 'unittest', '-v', 'test_visible_clocks_v32'],
            cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Imported/verified only in an unprivileged process, without decoding media.
    done = subprocess.run(['/usr/bin/python3', '-c',
        f'import sys; sys.path.insert(0, {str(V31)!r}); from run_visible_threads_v31 import verify_freeze; verify_freeze(); '
        'from run_visible_combined_v29 import verify_freeze as old; old(); print("dependencies verified")'],
        env=dict(os.environ, PYTHONPATH=str(Path('/tmp/seaqr_visible_combined_v29_s8XhL1'))),
        capture_output=True, text=True)
    write(directory/'dependency_preflight.json', dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr))
    if done.returncode:
        raise RuntimeError('Dependency preflight failed')
    write(directory/'freeze.json', dict(pre_run=True, prepared_ns=time.time_ns(),
        sources={n:sha(directory/n) for n in SOURCES}, unit_log_sha256=sha(directory/'unit.log'),
        v31_freeze_sha256=sha(V31/'freeze.json'), v31_batch_sha256=sha(V31/'batch.json'),
        dependency_preflight_sha256=sha(directory/'dependency_preflight.json'),
        policy=saved, telemetry=probe, schedule=schedule(), settings_changed=False, media_accessed=False))
    print('Prepared only: no clocks changed; no video decoded.', flush=True)


def run(directory):
    if os.geteuid() != 0 or os.environ.get('SUDO_USER') != 'serg':
        raise PermissionError('Run once via interactive sudo as serg; never send a password to chat')
    if not re.fullmatch(r'/tmp/seaqr_visible_clocks_v32_[A-Za-z0-9]+', str(directory)):
        raise ValueError('Fresh isolated /tmp v32 directory required')
    account = pwd.getpwnam('serg')
    if int(os.environ.get('SUDO_UID', '-1')) != account.pw_uid or directory.stat().st_uid != account.pw_uid:
        raise ValueError('Unexpected experiment owner')
    lock_fd = exclusive_lock()
    competing = competing_experiments()
    if competing:
        raise RuntimeError('Other experiment active: '+repr(competing))
    freeze = verify_freeze(directory)
    saved = snapshot()
    validate_original(saved)
    if saved != freeze['policy'] or freeze['schedule'] != schedule():
        raise ValueError('Original settings/schedule changed since preparation')
    thermal_check(sensors(), 65000)
    # An exclusive per-workspace run directory prevents accidental reruns.
    output = directory/'run'
    output.mkdir(mode=0o755)
    os.chown(output, account.pw_uid, account.pw_gid)
    write(output/'original_policy.json', saved)
    write(output/'run_identity.json', dict(controller_pid=os.getpid(), uid=os.getuid(),
        child_uid=account.pw_uid, child_groups=os.getgrouplist('serg', account.pw_gid),
        freeze_sha256=sha(directory/'freeze.json'), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    record = dict(completed=False, error=None, rows=[], schedule=schedule(), settings_restored=False,
        diagnostic_only=True, raw16_accessed=False, full_regression_run=False, defaults_changed=False)
    begin = time.monotonic()
    guard, child, active, monitor = None, None, None, None
    def abort(signum, frame):
        raise InterruptedError('Stop requested by signal '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, abort)
    try:
        guard = Guard(saved, output)
        with (output/'telemetry.jsonl').open('x', buffering=1) as monitor:
            for spec in schedule():
                guard.send()
                if time.monotonic()-begin >= 3300:
                    raise TimeoutError('55 minute batch deadline')
                set_policy(saved, spec['fixed'])
                # Same settle interval and external monitoring in every cell.
                for _ in range(10):
                    guard.send()
                    sample = sensors(); thermal_check(sample)
                    check_policy(saved, spec['fixed'])
                    monitor.write(json.dumps(dict(**sample, trial=spec['name'], phase='settle'))+'\n')
                    time.sleep(.5)
                cmd = command(spec, output)
                row = dict(**spec, command=cmd, started_ns=time.time_ns(), returncode=None)
                record['rows'].append(row)
                print('Starting '+spec['name']+f' ({len(record["rows"])}/50)', flush=True)
                status = output/'status.json'
                temp = output/'status.next'
                write(temp, dict(running=True, current=spec, completed=len(record['rows'])-1, scheduled=50))
                os.replace(temp, status)
                tick = time.monotonic()
                with (output/(spec['name']+'.log')).open('x') as log:
                    child = subprocess.Popen(cmd, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                        env=child_environment(spec['mode'], account), user=account.pw_uid, group=account.pw_gid,
                        extra_groups=os.getgrouplist('serg', account.pw_gid), start_new_session=True, close_fds=True)
                    active = proc_identity(child.pid)
                    guard.send('active', identity=active)
                    while child.poll() is None:
                        guard.send()
                        sample = sensors(); thermal_check(sample)
                        check_policy(saved, spec['fixed'])
                        monitor.write(json.dumps(dict(**sample, trial=spec['name'], phase='video'))+'\n')
                        if time.monotonic()-tick >= 180 or time.monotonic()-begin >= 3300:
                            raise TimeoutError('Child/batch deadline')
                        time.sleep(.5)
                    row.update(returncode=child.returncode, process_wall_s=time.monotonic()-tick)
                    guard.send('active', identity=None)
                    active, child = None, None
                if row['returncode']:
                    raise RuntimeError('Frozen child failed '+spec['name'])
                r = read(output/(spec['name']+'.v31.json'))
                if not r['passed'] or r['error'] is not None or r['count'] != 128 or r['traced']:
                    raise ValueError('Child exact-output gate failed')
                row.update(fps=r['fps'], wall_s=r['wall_s'], v31_sha256=sha(output/(spec['name']+'.v31.json')),
                    receipt_sha256=sha(output/(spec['name']+'.v29.json')), log_sha256=sha(output/(spec['name']+'.log')))
                print(f'Passed {spec["name"]}: {r["fps"]:.3f} FPS', flush=True)
        record['completed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        print('Stopping: '+repr(exc), flush=True)
    finally:
        # Do not let a second terminal signal interrupt restoration midway.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        stop_owned(active)
        if child is not None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                record['error'] = str(record['error'])+'; owned child did not exit'
        restored = restore(saved)
        write(output/'controller_restoration.json', restored)
        if guard is not None:
            try:
                guard.finish()
            except BaseException as exc:
                record['error'] = str(record['error'])+'; '+repr(exc)
        watchdog_ok = (output/'watchdog_restoration.json').exists() and read(output/'watchdog_restoration.json')['restored']
        record['settings_restored'] = restored['restored'] and watchdog_ok
        record['finished_ns'] = time.time_ns()
        write(output/'batch.json', record)
        temp = output/'status.next'
        if temp.exists():
            # Preserve interrupted status publication, rather than overwrite it.
            temp = output/'status.final'
        write(temp, dict(running=False, completed=sum(r.get('returncode') == 0 and 'fps' in r for r in record['rows']),
            scheduled=50, error=record['error'], settings_restored=record['settings_restored']))
        os.replace(temp, output/'status.json')
        print('Settings restored and checked: '+str(record['settings_restored']), flush=True)
    if record['error'] or not record['completed'] or not record['settings_restored']:
        os.close(lock_fd)
        raise SystemExit(1)
    os.close(lock_fd)
    print('All 50 trials finished. Ready for independent evidence audit; no defaults promoted.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'preflight', 'run'))
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(HERE)
    elif a.action == 'preflight':
        saved = snapshot(); validate_original(saved)
        probe = sensors(); thermal_check(probe, 65000)
        print(json.dumps(dict(settings=saved, telemetry=probe, changed=False), indent=2))
    else:
        run(HERE)
