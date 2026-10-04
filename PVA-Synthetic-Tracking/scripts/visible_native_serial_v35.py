"""v35: native scoring alone, with frozen serial execution and guarded clocks."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
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
import traceback

V31 = Path('/tmp/seaqr_visible_threads_v31_QvwNyq')
HERE = Path(__file__).resolve().parent
V34 = Path('/tmp/seaqr_visible_validation_v34_aqGC5d')
V25 = Path('/tmp/seaqr_visible_native_v25_qpJZ6C')
SOURCES = ('visible_native_serial_v35.py', 'run_visible_native_serial_v35.py',
    'test_visible_native_serial_v35.py', 'test_run_visible_native_serial_v35.py',
    'visible_native_serial_v35_plan.md', 'start_visible_native_serial_v35_tmux.sh',
    'v34_audit.json', 'audit_visible_validation_v34.py')
COUNTS = {'0029':687, '0126':674, '0055':689, '0082':691}
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
TRANSITION_TIMEOUT = 3.0
POLL_INTERVAL = .05
STABLE_SAMPLES = 3


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
        raise ValueError('Clock policy no longer matches declared arm: '+repr(dict(
            actual=actual, expected=expected_policy(saved, fixed))))
    return actual


class PolicyTransitionError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def unchanged_limits(actual, saved):
    for n in POLICIES:
        if any(actual[n][k] != saved[n][k] for k in ('maximum', 'governor')):
            raise ValueError('External max/governor change '+n)


def wait_policy(saved, fixed, reader=read_node, *, safety=None,
                clock=time.monotonic, sleeper=time.sleep):
    """Wait for three consecutive snapshots, never retry/suppress forever.

    Safety runs before each observation. A real max/governor change fails
    immediately. Temporarily unreadable policy data resets stability and is
    retained in the receipt; it cannot create a successful readback.
    """
    target = expected_policy(saved, fixed)
    start = clock()
    receipt = dict(verified=False, target=target, observations=[], error=None,
        timeout_s=TRANSITION_TIMEOUT, poll_interval_s=POLL_INTERVAL, required_stable_samples=STABLE_SAMPLES)
    stable = 0
    try:
        while True:
            observation = dict(elapsed_s=clock()-start, actual=None, read_error=None)
            receipt['observations'].append(observation)
            if safety is not None:
                observation['safety'] = safety()
            try:
                actual = snapshot(reader)
            except (OSError, ValueError) as exc:
                observation['read_error'] = repr(exc)
                stable = 0
            else:
                observation['actual'] = actual
                unchanged_limits(actual, saved)
                stable = stable+1 if actual == target else 0
            elapsed = clock()-start
            observation.update(elapsed_s=elapsed, consecutive_matches=stable)
            if elapsed > TRANSITION_TIMEOUT:
                raise TimeoutError('Clock policy did not stabilize within three seconds')
            if stable >= STABLE_SAMPLES:
                receipt.update(verified=True, elapsed_s=elapsed)
                return receipt
            if elapsed >= TRANSITION_TIMEOUT:
                raise TimeoutError('Clock policy did not stabilize within three seconds')
            sleeper(min(POLL_INTERVAL, TRANSITION_TIMEOUT-elapsed))
    except BaseException as exc:
        receipt.update(error=repr(exc), elapsed_s=clock()-start, traceback=traceback.format_exc())
        raise PolicyTransitionError('Clock policy convergence failed: '+repr(exc), receipt) from exc


def set_policy(saved, fixed, reader=read_node, writer=write_min, *, safety=None,
               clock=time.monotonic, sleeper=time.sleep):
    receipt = dict(verified=False, target=expected_policy(saved, fixed), before=None, writes=[], error=None)
    start = clock()
    try:
        if safety is not None:
            receipt['safety_before'] = safety()
        receipt['before'] = snapshot(reader)
        unchanged_limits(receipt['before'], saved)
        # Submit every floor request, including apparently unchanged values:
        # a readable policy may lag an earlier queued QoS update.
        for n, p in POLICIES.items():
            row = dict(policy=n, path=p['path']+'/'+p['minimum'], value=receipt['target'][n]['minimum'])
            receipt['writes'].append(row)
            writer(row['path'], row['value'])
            row['written'] = True
        receipt['verification'] = wait_policy(saved, fixed, reader, safety=safety, clock=clock, sleeper=sleeper)
        receipt.update(verified=True, elapsed_s=clock()-start)
        return receipt
    except BaseException as exc:
        if isinstance(exc, PolicyTransitionError):
            receipt['verification'] = exc.receipt
        receipt.update(error=repr(exc), elapsed_s=clock()-start, traceback=traceback.format_exc())
        raise PolicyTransitionError('Clock transition failed: '+repr(exc), receipt) from exc


def restore(saved, reader=read_node, writer=write_min, *, heartbeat=None,
            clock=time.monotonic, sleeper=time.sleep):
    errors, writes = [], []
    # ALWAYS resubmit the saved floors. Merely seeing the old value is not
    # sufficient: a preceding interrupted transition may still be queued.
    # An individual failure must not prevent attempts on all other policies.
    for n, p in POLICIES.items():
        row = dict(policy=n, path=p['path']+'/'+p['minimum'], value=saved[n]['minimum'])
        writes.append(row)
        try:
            writer(row['path'], row['value'])
            row['written'] = True
        except BaseException as exc:
            row['error'] = repr(exc)
            errors.append(dict(policy=n, error=repr(exc)))
    # Restoration must still lower clocks during high temperature or a sensor
    # failure. Heartbeat failure is recorded but cannot prevent verification.
    def keep_alive():
        if heartbeat is not None:
            try:
                heartbeat()
            except BaseException as exc:
                if not any(e.get('heartbeat') for e in errors):
                    errors.append(dict(heartbeat=True, error=repr(exc)))
    try:
        verification = wait_policy(saved, False, reader, safety=keep_alive, clock=clock, sleeper=sleeper)
    except PolicyTransitionError as exc:
        verification = exc.receipt
        errors.append(dict(error=repr(exc)))
    actual = verification['observations'][-1]['actual'] if verification['observations'] else None
    return dict(restored=not errors and verification['verified'] and actual == saved,
        errors=errors, actual=actual, saved=saved, writes=writes, verification=verification)


def transition(output, label, saved, fixed, safety):
    path = output/'transitions'/(label+'.json')
    if path.exists():
        raise FileExistsError('Existing transition evidence '+str(path))
    try:
        result = set_policy(saved, fixed, safety=safety)
    except PolicyTransitionError as exc:
        write(path, dict(label=label, fixed=fixed, **exc.receipt))
        raise
    write(path, dict(label=label, fixed=fixed, **result))
    return dict(label=label, fixed=fixed, sha256=sha(path), elapsed_s=result['elapsed_s'])


def hardware_preflight(output, saved, safety):
    result = dict(passed=False, media_accessed=False, transitions=[], error=None)
    start = time.monotonic()
    try:
        for i in range(3):
            for fixed in (True, False):
                if time.monotonic()-start > 30:
                    raise TimeoutError('Hardware transition preflight exceeded 30 seconds')
                label = f'preflight_{i}_{"fixed" if fixed else "auto"}'
                result['transitions'].append(transition(output, label, saved, fixed, safety))
        if time.monotonic()-start > 30:
            raise TimeoutError('Hardware transition preflight exceeded 30 seconds')
        result['passed'] = True
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_s'] = time.monotonic()-start
        write(output/'transition_preflight.json', result)
    return result


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
    def row(name, clip, kind, arm, repeat=None):
        return dict(name=name, clip=clip, kind=kind, arm=arm, repeat=repeat,
            mode='combined_default', fixed=True, audit=kind=='smoke',
            traced=False, frames=None if kind=='full' else 128)
    result=[row('smoke_'+c,c,'smoke','native') for c in ('0126','0082')]
    # Pair order alternates within and between workloads/repeats.
    for i in range(3):
        for c in (('0126','0082') if i%2==0 else ('0082','0126')):
            arms=('reference','native') if (i+(c=='0082'))%2==0 else ('native','reference')
            result += [row(f'prefix_repeat{i}_{c}_{arm}',c,'prefix',arm,i) for arm in arms]
    result += [row('full_'+c,c,'full','native') for c in COUNTS]
    return result


def expected_count(spec):
    return COUNTS[spec['clip']] if spec['frames'] is None else 128


def command(spec, directory):
    if spec not in schedule():
        raise ValueError('Unscheduled or out-of-scope video request')
    cmd=['/usr/bin/python3', str(directory.parent/'run_visible_native_serial_v35.py'),
         '--trial', spec['name'], '--output', str(directory/spec['name'])]
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
    check_v34_audit(directory)
    check_generated(directory)
    if f['generated_sha256']!=sha(directory/'generated_01.json') or f['v25_freeze_sha256']!=sha(V25/'freeze.json'):
        raise ValueError('Native dependency/gate changed')
    if f['v34_batch_sha256']!=sha(V34/'run/batch.json'):
        raise ValueError('v34 prerequisite changed')
    return f


def check_v34_audit(directory):
    a=read(directory/'v34_audit.json')
    if not (a['verified'] and a['completed'] and a['completed_trials']==a['scheduled_trials']==16
            and a['recorded_settings_restored'] and not a['raw16_accessed']
            and not a['defaults_changed'] and a['freeze_sha256']==sha(V34/'freeze.json')
            and a['verifier_sha256']==sha(directory/'audit_visible_validation_v34.py')):
        raise ValueError('Completed independent v34 audit required')
    batch=read(V34/'run/batch.json')
    if not (batch['completed'] and batch['error'] is None and batch['settings_restored']
            and len(batch['rows'])==16):
        raise ValueError('Completed v34 prerequisite required')
    return a


def check_generated(directory):
    g=read(directory/'generated_01.json')
    original=read(V25/'generated_01.json')
    if not (g['passed'] and not g['real_media_read'] and len(g['cases'])==47 and len(g['primitive'])==11
            and all(c['exact'] for c in g['cases']+g['primitive']) and g['native_calls']>0 and g['fallbacks']>0
            and g['passthroughs']==2 and g['independent_outputs'] and g['reentrant_calls']==12
            and g['gil_probe']['passed'] and g['gil_probe']['other_thread_progress_during_call']
            and g['transformed_sha256']==original['transformed_sha256']
            and g['reference_sha256']==original['reference_sha256']
            and g['library_sha256']==original['library_sha256']==sha(V25/'build/libnative_motion_v25.so')
            and g['source_sha256']==original['source_sha256']
            and all(sha(V25/n)==h for n,h in g['source_sha256'].items())
            and [r['clip'] for r in g['replays']]==['0126','0082']):
        raise ValueError('Fresh native generated gate failed')
    for r in g['replays']:
        p=V25/f"attribute_{r['clip']}_reference.fits.json"
        if not (r['exact'] and [c['frame'] for c in r['cases']]==list(range(1,128))
                and all(c['exact'] for c in r['cases']) and r['native_calls']==127 and r['fallbacks']==0
                and r['json_sha256']==sha(p) and r['npz_sha256']==sha(p.with_suffix('.npz'))):
            raise ValueError('Fresh native correspondence replay failed')
    return g


def percentile95(values):
    if not values or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('Invalid latency samples')
    v=sorted(values); p=(len(v)-1)*.95; lo=math.floor(p); hi=math.ceil(p)
    return v[lo]+(v[hi]-v[lo])*(p-lo)


def performance_gate(directory):
    """Read only the twelve clean prefixes. Never include smokes or full clips."""
    rows={}; evidence={}
    for s in (x for x in schedule() if x['kind']=='prefix'):
        path=directory/s['name']; r=read(path.with_suffix('.v35.json')); base=read(path.with_suffix('.v35base.json'))
        if not (all(r[k]==v for k,v in s.items()) and r['passed'] and r['error'] is None
                and r['count']==128 and r['bindings_restored']
                and base['schema']=='seaqr.visible-native-serial-base-v35.v1'
                and base['native_motion_v25_enabled']==(s['arm']=='native')
                and base['execution_policy']=='serial_reference' and not base['staged_v24_enabled']
                and r['receipt_sha256']==sha(path.with_suffix('.v35base.json'))
                and base['passed'] and base['error'] is None and base['processed_frames']==128
                and r['fps']==base['fps'] and r['wall_s']==base['wall_s']
                and math.isfinite(r['wall_s']) and r['wall_s']>0
                and math.isclose(r['fps'],128/r['wall_s'],rel_tol=1e-12)):
            raise ValueError('Invalid exact prefix evidence '+s['name'])
        frames=base['execution']['frames']
        samples=dict(consumer_cadence=base['consumer_frame_ms'],
            request_to_complete=[(f['consumer_complete_ns']-f['request_ns'])/1e6 for f in frames])
        if any(len(v)!=128 for v in samples.values()):
            raise ValueError('Missing latency samples')
        p95={k:percentile95(v) for k,v in samples.items()}
        rows[(s['clip'],s['repeat'],s['arm'])]=dict(wall_s=r['wall_s'],samples=samples,p95=p95)
        evidence[s['name']]=sha(path.with_suffix('.v35.json'))
    clips={}
    for c in ('0126','0082'):
        refs=[rows[c,i,'reference'] for i in range(3)]
        natives=[rows[c,i,'native'] for i in range(3)]
        ratios=[a['wall_s']/b['wall_s'] for a,b in zip(refs,natives)]
        speedup=sum(x['wall_s'] for x in refs)/sum(x['wall_s'] for x in natives)
        latency={}
        for label in ('consumer_cadence','request_to_complete'):
            baseline=percentile95([x for r in refs for x in r['samples'][label]])
            candidate=percentile95([x for r in natives for x in r['samples'][label]])
            worse=sum(b['p95'][label]>a['p95'][label] for a,b in zip(refs,natives))
            latency[label]=dict(reference_p95_ms=baseline,native_p95_ms=candidate,
                worse_pairs=worse,consistent_regression=candidate>baseline and worse>=2)
        clips[c]=dict(paired_speedups=ratios,pooled_speedup=speedup,
            reference_fps=384/sum(x['wall_s'] for x in refs),
            native_fps=384/sum(x['wall_s'] for x in natives),latency=latency)
    passed=(all(x['pooled_speedup']>=1.05 and all(r>1 for r in x['paired_speedups'])
                and not any(m['consistent_regression'] for m in x['latency'].values()) for x in clips.values())
            and max(x['pooled_speedup'] for x in clips.values())>=1.10)
    return dict(schema='seaqr.native-serial-v35-performance.v1',passed=passed,clips=clips,
        evidence_sha256=evidence,minimum_both_speedup=1.05,minimum_one_speedup=1.10,
        all_pairs_must_improve=True,full_regression_permitted=passed,
        new_accuracy_validated=False,production_approved=False)


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
    check_v34_audit(directory)
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
        subprocess.run(['/usr/bin/python3', '-m', 'unittest', '-v', 'test_visible_native_serial_v35', 'test_run_visible_native_serial_v35'],
            cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Imported/verified only in an unprivileged process, without decoding media.
    done = subprocess.run(['/usr/bin/python3', str(directory/'run_visible_native_serial_v35.py'), '--preflight'],
        env=child_environment('combined_default', pwd.getpwnam('serg')),
        capture_output=True, text=True)
    write(directory/'dependency_preflight.json', dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr))
    if done.returncode:
        raise RuntimeError('Dependency preflight failed')
    check_generated(directory)
    write(directory/'freeze.json', dict(pre_run=True, prepared_ns=time.time_ns(),
        sources={n:sha(directory/n) for n in SOURCES}, unit_log_sha256=sha(directory/'unit.log'),
        v31_freeze_sha256=sha(V31/'freeze.json'), v31_batch_sha256=sha(V31/'batch.json'),
        dependency_preflight_sha256=sha(directory/'dependency_preflight.json'),
        generated_sha256=sha(directory/'generated_01.json'), v25_freeze_sha256=sha(V25/'freeze.json'),
        v34_batch_sha256=sha(V34/'run/batch.json'),
        policy=saved, telemetry=probe, schedule=schedule(), settings_changed=False, media_accessed=False,
        runtime_reference=read(V34/'run/full_repeat0_0126.v34.json')['runtime_before']))
    print('Prepared only: no clocks changed; no video decoded.', flush=True)


def run(directory):
    if os.geteuid() != 0 or os.environ.get('SUDO_USER') != 'serg':
        raise PermissionError('Run once via interactive sudo as serg; never send a password to chat')
    if not re.fullmatch(r'/tmp/seaqr_visible_native_serial_v35_[A-Za-z0-9]+', str(directory)):
        raise ValueError('Fresh isolated /tmp v35 directory required')
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
    (output/'transitions').mkdir(mode=0o755)
    write(output/'original_policy.json', saved)
    write(output/'run_identity.json', dict(controller_pid=os.getpid(), uid=os.getuid(),
        child_uid=account.pw_uid, child_groups=os.getgrouplist('serg', account.pw_gid),
        freeze_sha256=sha(directory/'freeze.json'), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    record = dict(completed=False, error=None, rows=[], schedule=schedule(), settings_restored=False,
        phase='initializing', current_trial=None, transitions=[],
        diagnostic_only=True, raw16_accessed=False, full_regression_run=False, defaults_changed=False)
    begin = time.monotonic()
    guard, child, active, monitor = None, None, None, None
    def transition_safety():
        guard.send()
        if time.monotonic()-begin >= 3300:
            raise TimeoutError('55 minute batch deadline')
        sample = sensors()
        thermal_check(sample)
        return sample

    def abort(signum, frame):
        raise InterruptedError('Stop requested by signal '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, abort)
    try:
        guard = Guard(saved, output)
        record['phase'] = 'hardware_transition_preflight'
        print('Checking six clock transitions before any video processing...', flush=True)
        hardware_preflight(output, saved, transition_safety)
        record['transition_preflight_sha256'] = sha(output/'transition_preflight.json')
        print('Hardware clock-transition preflight passed.', flush=True)
        with (output/'telemetry.jsonl').open('x', buffering=1) as monitor:
            for spec in schedule():
                if spec['kind']=='full' and 'performance_gate' not in record:
                    gate=performance_gate(output)
                    write(output/'performance_gate.json',gate)
                    record['performance_gate']=gate
                    if not gate['passed']:
                        record['skipped_full_trials']=[x['name'] for x in schedule() if x['kind']=='full']
                        print('Performance gate not met; full regressions skipped. No promotion.',flush=True)
                        break
                guard.send()
                if time.monotonic()-begin >= 3300:
                    raise TimeoutError('55 minute batch deadline')
                record.update(phase='clock_transition', current_trial=spec['name'])
                record['transitions'].append(transition(output, spec['name'], saved, spec['fixed'], transition_safety))
                # Same settle interval and external monitoring in every cell.
                record['phase'] = 'settle'
                for _ in range(10):
                    guard.send()
                    sample = sensors(); thermal_check(sample)
                    check_policy(saved, spec['fixed'])
                    monitor.write(json.dumps(dict(**sample, trial=spec['name'], phase='settle'))+'\n')
                    time.sleep(.5)
                cmd = command(spec, output)
                record['phase'] = 'video'
                row = dict(**spec, command=cmd, started_ns=time.time_ns(), returncode=None)
                record['rows'].append(row)
                print('Starting '+spec['name']+f' ({len(record["rows"])}/{len(schedule())})', flush=True)
                status = output/'status.json'
                temp = output/'status.next'
                write(temp, dict(running=True, current=spec, completed=len(record['rows'])-1, scheduled=len(schedule())))
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
                        if time.monotonic()-tick >= 600 or time.monotonic()-begin >= 3300:
                            raise TimeoutError('Child/batch deadline')
                        time.sleep(.5)
                    row.update(returncode=child.returncode, process_wall_s=time.monotonic()-tick)
                    guard.send('active', identity=None)
                    active, child = None, None
                if row['returncode']:
                    raise RuntimeError('Frozen child failed '+spec['name'])
                r = read(output/(spec['name']+'.v35.json'))
                if not r['passed'] or r['error'] is not None or r['count'] != expected_count(spec) or any(r[k]!=v for k,v in spec.items()):
                    raise ValueError('Child exact-output gate failed')
                row.update(fps=r['fps'], wall_s=r['wall_s'], v35_sha256=sha(output/(spec['name']+'.v35.json')),
                    receipt_sha256=sha(output/(spec['name']+'.v35base.json')), log_sha256=sha(output/(spec['name']+'.log')))
                print(f'Passed {spec["name"]}: {r["fps"]:.3f} FPS', flush=True)
                record['full_regression_run'] = sum(x.get('kind')=='full' and 'fps' in x for x in record['rows'])==4
        record.update(completed=True, phase='trials_complete')
    except BaseException as exc:
        record['error'] = repr(exc)
        record['traceback'] = traceback.format_exc()
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
        restored = restore(saved, heartbeat=guard.send if guard is not None else None)
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
            scheduled=len(schedule()), error=record['error'], settings_restored=record['settings_restored']))
        os.replace(temp, output/'status.json')
        print('Settings restored and checked: '+str(record['settings_restored']), flush=True)
    if record['error'] or not record['completed'] or not record['settings_restored']:
        os.close(lock_fd)
        raise SystemExit(1)
    os.close(lock_fd)
    print('Scheduled gate sequence finished. Ready for independent evidence audit; no defaults promoted.', flush=True)


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


