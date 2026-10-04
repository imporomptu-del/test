"""Clock guard tests: fake hardware only; never touch real sysfs or video."""
import copy
from contextlib import ExitStack
import errno
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/visible_clocks_v33.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parent/'visible_clocks_v33.py'
spec = importlib.util.spec_from_file_location('clocks_under_test', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class Hardware:
    def __init__(self):
        self.values = {'/sys/devices/system/cpu/online': '0-11'}
        self.writes = []
        for p in v.POLICIES.values():
            for key, value in ((p['minimum'], p['expected_min']), (p['maximum'], p['expected_max']),
                               (p['governor'], p['expected_governor']),
                               (p['available'], f'{p["expected_min"]} {p["expected_max"]}')):
                self.values[p['path']+'/'+key] = str(value)

    def read(self, path):
        return self.values[str(path)]

    def write(self, path, value):
        if path not in v.MIN_PATHS:
            raise AssertionError('Unexpected hardware write')
        self.writes.append((path, value))
        self.values[path] = str(value)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.after_sleep = lambda: None

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.after_sleep()

    @property
    def kwargs(self):
        return dict(clock=self.clock, sleeper=self.sleep)


class DeferredHardware(Hardware):
    def __init__(self, timer, delay=.12):
        super().__init__()
        self.timer, self.delay, self.pending = timer, delay, {}
        timer.after_sleep = self.commit_due

    def write(self, path, value):
        if path not in v.MIN_PATHS:
            raise AssertionError('Unexpected hardware write')
        self.writes.append((path, value))
        # New QoS request replaces an earlier request for the same floor.
        self.pending[path] = (self.timer.clock()+self.delay, str(value))

    def commit_due(self):
        for path, (due, value) in list(self.pending.items()):
            if self.timer.clock() >= due:
                self.values[path] = value
                del self.pending[path]


class TestClocks(unittest.TestCase):
    def setUp(self):
        self.t = FakeTime()
        self.h = Hardware()
        self.saved = v.snapshot(self.h.read)

    def test_round_trip_only_four_floors(self):
        v.validate_original(self.saved, self.h.read)
        v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        self.assertEqual(len(self.h.writes), 4)
        v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        self.assertEqual(len(self.h.writes), 8)
        result = v.restore(self.saved, self.h.read, self.h.write, **self.t.kwargs)
        self.assertTrue(result['restored'])
        self.assertEqual(len(self.h.writes), 12)
        self.assertEqual(result['actual'], self.saved)

    def test_partial_write_failure_is_restorable(self):
        def failure(path, value):
            if len(self.h.writes) == 2:
                raise OSError('simulated third policy write failure')
            self.h.write(path, value)
        with self.assertRaises(v.PolicyTransitionError):
            v.set_policy(self.saved, True, self.h.read, failure, **self.t.kwargs)
        self.assertNotEqual(v.snapshot(self.h.read), self.saved)
        self.assertTrue(v.restore(self.saved, self.h.read, self.h.write, **self.t.kwargs)['restored'])

    def test_restore_continues_after_failure_and_reports_it(self):
        v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        first = next(iter(v.POLICIES.values()))
        bad = first['path']+'/'+first['minimum']
        def failure(path, value):
            if path == bad:
                raise OSError('simulated read-only policy')
            self.h.write(path, value)
        result = v.restore(self.saved, self.h.read, failure, **self.t.kwargs)
        self.assertFalse(result['restored'])
        self.assertEqual(len(result['errors']), 2)
        self.assertEqual(result['actual']['gpu'], self.saved['gpu'])

    def test_delayed_application_converges_in_both_directions(self):
        h = DeferredHardware(self.t)
        calls = []
        fixed = v.set_policy(self.saved, True, h.read, h.write,
            safety=lambda: calls.append(self.t.clock()), **self.t.kwargs)
        self.assertTrue(fixed['verified'])
        history = fixed['verification']['observations']
        self.assertEqual(history[0]['consecutive_matches'], 0)
        self.assertEqual(history[-1]['consecutive_matches'], 3)
        self.assertGreaterEqual(fixed['elapsed_s'], .2)
        self.assertEqual(len(calls), len(history)+1)
        result = v.restore(self.saved, h.read, h.write, **self.t.kwargs)
        self.assertTrue(result['restored'])
        self.assertEqual(result['actual'], self.saved)
        self.assertFalse(h.pending)

    def test_restore_cancels_pending_fixed_request_even_if_readback_original(self):
        h = DeferredHardware(self.t)
        for n, p in v.POLICIES.items():
            h.write(p['path']+'/'+p['minimum'], self.saved[n]['maximum'])
        self.assertEqual(v.snapshot(h.read), self.saved)
        result = v.restore(self.saved, h.read, h.write, **self.t.kwargs)
        self.assertTrue(result['restored'])
        self.assertEqual(len(h.writes), 8)
        self.t.sleep(1)
        self.assertEqual(v.snapshot(h.read), self.saved)

    def test_permanent_mismatch_times_out_with_history(self):
        with self.assertRaises(v.PolicyTransitionError) as caught:
            v.set_policy(self.saved, True, self.h.read, lambda p, x: None, **self.t.kwargs)
        evidence = caught.exception.receipt
        self.assertFalse(evidence['verified'])
        self.assertAlmostEqual(evidence['elapsed_s'], 3)
        self.assertGreater(len(evidence['verification']['observations']), 3)
        self.assertIn('TimeoutError', evidence['verification']['error'])
        self.assertIn('traceback', evidence)

    def test_a_single_matching_snapshot_cannot_pass(self):
        # Target matches at t=0 and .10 but not .05: need a fresh stable streak.
        def noisy(path):
            if str(path) in v.MIN_PATHS and .04 < self.t.now < .09:
                return '123'
            return self.h.read(path)
        r = v.wait_policy(self.saved, False, noisy, **self.t.kwargs)
        self.assertEqual([x['consecutive_matches'] for x in r['observations']], [1, 0, 1, 2, 3])
        self.assertAlmostEqual(r['elapsed_s'], .2)

    def test_transient_read_failure_resets_stability(self):
        def failing(path):
            if .04 < self.t.now < .09:
                raise OSError(errno.EAGAIN, 'temporarily unavailable')
            return self.h.read(path)
        r = v.wait_policy(self.saved, False, failing, **self.t.kwargs)
        self.assertTrue(r['verified'])
        self.assertIn('BlockingIOError', r['observations'][1]['read_error'])
        self.assertEqual(r['observations'][1]['consecutive_matches'], 0)
        self.assertAlmostEqual(r['elapsed_s'], .2)

    def test_persistent_read_failure_is_bounded(self):
        with self.assertRaises(v.PolicyTransitionError) as caught:
            v.wait_policy(self.saved, False, lambda p: 'unreadable', **self.t.kwargs)
        self.assertAlmostEqual(self.t.now, 3)
        self.assertTrue(all(x['read_error'] for x in caught.exception.receipt['observations']))

    def test_thermal_failure_during_wait_aborts_without_more_retries(self):
        def safety():
            if self.t.now >= .05:
                raise RuntimeError('thermal cutoff')
        with self.assertRaises(v.PolicyTransitionError) as caught:
            v.set_policy(self.saved, True, self.h.read, self.h.write,
                safety=safety, **self.t.kwargs)
        self.assertIn('thermal cutoff', str(caught.exception))
        self.assertAlmostEqual(self.t.now, .05)
        self.assertTrue(v.restore(self.saved, self.h.read, self.h.write, **self.t.kwargs)['restored'])

    def test_interrupt_is_recorded_and_does_not_become_a_success(self):
        def interrupt():
            raise InterruptedError('simulated signal')
        with self.assertRaises(v.PolicyTransitionError) as caught:
            v.set_policy(self.saved, True, self.h.read, self.h.write,
                safety=interrupt, **self.t.kwargs)
        self.assertFalse(self.h.writes)
        self.assertFalse(caught.exception.receipt['verified'])

    def test_restore_heartbeat_failure_does_not_block_floor_recovery(self):
        v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        def failure():
            raise BrokenPipeError('lost watchdog')
        r = v.restore(self.saved, self.h.read, self.h.write, heartbeat=failure, **self.t.kwargs)
        self.assertEqual(r['actual'], self.saved)
        self.assertTrue(r['verification']['verified'])
        self.assertFalse(r['restored'])
        self.assertEqual(len(r['errors']), 1)

    def test_during_trial_policy_mismatch_still_fails_immediately(self):
        v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        before = self.t.now
        with self.assertRaisesRegex(ValueError, 'actual.*expected'):
            v.check_policy(self.saved, False, self.h.read)
        self.assertEqual(self.t.now, before)

    def test_real_maximum_change_during_wait_aborts_immediately(self):
        self.h.values[v.POLICIES['gpu']['path']+'/max_freq'] = '999'
        with self.assertRaises(v.PolicyTransitionError) as caught:
            v.wait_policy(self.saved, False, self.h.read, **self.t.kwargs)
        self.assertEqual(self.t.now, 0)
        self.assertIn('External max/governor change', str(caught.exception))

    def test_transition_failure_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/'transitions').mkdir()
            error = v.PolicyTransitionError('test', dict(verified=False, error='test', observations=[]))
            with patch.object(v, 'set_policy', side_effect=error) as setter:
                with self.assertRaises(v.PolicyTransitionError):
                    v.transition(output, 'trial', self.saved, True, lambda: None)
                result = v.read(output/'transitions/trial.json')
                self.assertFalse(result['verified'])
                with self.assertRaises(FileExistsError):
                    v.transition(output, 'trial', self.saved, True, lambda: None)
                self.assertEqual(setter.call_count, 1)

    def test_hardware_preflight_is_six_transitions_and_no_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            def change(output, label, saved, fixed, safety):
                return dict(label=label, fixed=fixed)
            with patch.object(v, 'transition', side_effect=change) as t, patch.object(v, 'command') as cmd:
                r = v.hardware_preflight(Path(tmp), self.saved, lambda: None)
            self.assertTrue(r['passed'])
            self.assertFalse(r['media_accessed'])
            self.assertEqual(t.call_count, 6)
            self.assertEqual([x['fixed'] for x in r['transitions']], [True, False]*3)
            cmd.assert_not_called()

    def test_failed_preflight_is_persisted_and_stops_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(v, 'transition', side_effect=RuntimeError('test')) as t:
                with self.assertRaises(RuntimeError):
                    v.hardware_preflight(Path(tmp), self.saved, lambda: None)
            self.assertEqual(t.call_count, 1)
            self.assertFalse(v.read(Path(tmp)/'transition_preflight.json')['passed'])

    def test_run_preflight_failure_restores_without_launching_video(self):
        # Exercise the controller's actual failure/finally path; all privileged
        # operations and hardware are mocked, only temporary evidence is real.
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            d = Path(tmp)
            (d/'freeze.json').write_text('{}')
            account = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(),
                pw_dir='/home/serg', pw_name='serg')
            stack.enter_context(patch.dict(v.os.environ, SUDO_USER='serg', SUDO_UID=str(account.pw_uid)))
            for target, name, value in (
                (v.os, 'geteuid', 0), (v.re, 'fullmatch', True),
                (v.pwd, 'getpwnam', account), (v.os, 'chown', None),
                (v.os, 'getgrouplist', [account.pw_gid]), (v.signal, 'signal', None),
                (v, 'exclusive_lock', os.open('/dev/null', os.O_RDONLY)),
                (v, 'competing_experiments', []),
                (v, 'verify_freeze', dict(policy=self.saved, schedule=v.schedule())),
                (v, 'snapshot', self.saved), (v, 'validate_original', None),
                (v, 'sensors', dict(temperatures={'cpu-thermal': 50000, 'tj-thermal': 51000})),
            ):
                stack.enter_context(patch.object(target, name, return_value=value))
            guard = stack.enter_context(patch.object(v, 'Guard')).return_value
            guard.finish.side_effect = lambda: v.write(d/'run/watchdog_restoration.json', dict(restored=True))
            restoration = stack.enter_context(patch.object(v, 'restore', return_value=dict(restored=True)))
            stack.enter_context(patch.object(v, 'hardware_preflight', side_effect=TimeoutError('test preflight')))
            popen = stack.enter_context(patch.object(v.subprocess, 'Popen'))
            with self.assertRaises(SystemExit) as caught:
                v.run(d)
            self.assertEqual(caught.exception.code, 1)
            popen.assert_not_called()
            restoration.assert_called_once_with(self.saved, heartbeat=guard.send)
            guard.finish.assert_called_once_with()
            result = v.read(d/'run/batch.json')
            self.assertFalse(result['completed'])
            self.assertTrue(result['settings_restored'])
            self.assertEqual(result['phase'], 'hardware_transition_preflight')
            self.assertEqual(result['rows'], [])
            self.assertIn('test preflight', result['traceback'])

    def test_unknown_original_and_topology_rejected(self):
        wrong = copy.deepcopy(self.saved)
        wrong['cpu0']['minimum'] = 1
        with self.assertRaises(ValueError):
            v.validate_original(wrong, self.h.read)
        self.h.values['/sys/devices/system/cpu/online'] = '0-7'
        with self.assertRaises(ValueError):
            v.validate_original(self.saved, self.h.read)

    def test_unsupported_max_rejected(self):
        p = v.POLICIES['gpu']
        self.h.values[p['path']+'/'+p['available']] = str(p['expected_min'])
        with self.assertRaises(ValueError):
            v.validate_original(self.saved, self.h.read)

    def test_external_max_not_overwritten(self):
        p = v.POLICIES['gpu']
        self.h.values[p['path']+'/'+p['maximum']] = '999'
        with self.assertRaises(v.PolicyTransitionError):
            v.set_policy(self.saved, True, self.h.read, self.h.write, **self.t.kwargs)
        self.assertFalse(self.h.writes)
        self.assertFalse(v.restore(self.saved, self.h.read, self.h.write, **self.t.kwargs)['restored'])
        self.assertEqual(self.h.values[p['path']+'/'+p['maximum']], '999')

    def test_actual_write_allowlist_before_os_open(self):
        with patch.object(v.os, 'open', side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):
                v.write_min('/tmp/arbitrary', 1)
            with self.assertRaises(ValueError):
                v.write_min(next(iter(v.MIN_PATHS)), '1')

    def test_thermal_guards_and_optional_unavailable(self):
        normal = dict(temperatures={'cpu-thermal': 50000, 'tj-thermal': 51000,
                                   'gpu-thermal': dict(error='EAGAIN')})
        v.thermal_check(normal)
        for name, value in (('cpu-thermal', None), ('tj-thermal', 75000), ('gpu-thermal', 78000)):
            altered = copy.deepcopy(normal)
            altered['temperatures'][name] = value
            with self.assertRaises(RuntimeError):
                v.thermal_check(altered)
        normal['temperatures']['cpu-thermal'] = 65000
        with self.assertRaises(RuntimeError):
            v.thermal_check(normal, 65000)

    def test_complete_balanced_schedule(self):
        s = v.schedule()
        self.assertEqual(len(s), 50)
        self.assertEqual(len({x['name'] for x in s}), 50)
        self.assertTrue(all(x['audit'] and x['fixed'] and x['mode'] == 'combined' for x in s[:2]))
        for clip in ('0126', '0082'):
            for mode in v.MODES:
                for fixed in (False, True):
                    rows = [x for x in s if not x['audit'] and (x['clip'], x['mode'], x['fixed']) == (clip, mode, fixed)]
                    self.assertEqual(len(rows), 3)
        self.assertEqual(sum(128 for x in s), 6400)

    def test_commands_refuse_arbitrary_clip_or_mode(self):
        s = v.schedule()[2]
        cmd = v.command(s, Path('/tmp/output'))
        self.assertEqual(cmd[0], '/usr/bin/python3')
        self.assertEqual(cmd[-2:], ['--frames', '128'])
        for update in (dict(clip='arbitrary'), dict(mode='v20'), dict(name='../escape'), dict(audit=True)):
            with self.assertRaises(ValueError):
                v.command(dict(s, **update), Path('/tmp/output'))

    def test_thread_environment_clean_and_explicit(self):
        account = SimpleNamespace(pw_dir='/home/serg', pw_name='serg')
        for mode in v.MODES:
            env = v.child_environment(mode, account)
            expected = None if mode.endswith('_default') else '1'
            self.assertEqual(env.get('OPENBLAS_NUM_THREADS'), expected)
            self.assertNotIn('PYTHONPATH', env)
            self.assertTrue(all(k not in env for k in v.THREAD_KEYS[1:]))

    def test_watchdog_decision(self):
        self.assertIsNone(v.guard_reason(20, 0, 15, False))
        self.assertEqual(v.guard_reason(20, 0, 15, True), 'controller_pipe_closed')
        self.assertEqual(v.guard_reason(80, 0, 20, False), 'heartbeat_lost')
        self.assertEqual(v.guard_reason(3600, 0, 3599, False), 'hard_deadline')

    def test_stop_does_not_signal_reused_identity(self):
        identity = dict(pid=123, start_ticks=9, pgrp=123)
        with patch.object(v, 'proc_identity', return_value=dict(identity, start_ticks=10)), patch.object(v.os, 'killpg') as kill:
            v.stop_owned(identity)
            kill.assert_not_called()

    @unittest.skipUnless(hasattr(os, 'fork'), 'POSIX watchdog')
    def test_real_watchdog_pipe_loss_restores_mock_hardware(self):
        # Fork a real independent watchdog but replace all hardware access with
        # fake functions. Pipe EOF simulates a SIGKILLed controller. This is not
        # a hardware restoration test and never invokes privileged controls.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            r, w = os.pipe(); rr, rw = os.pipe()
            with patch.object(v, 'restore', return_value=dict(restored=True, saved=self.saved, actual=self.saved, errors=[])):
                pid = os.fork()
                if not pid:
                    os.close(w); os.close(rr)
                    v.watchdog(r, rw, self.saved, directory)
                    os._exit(8)
                os.close(r); os.close(rw)
                try:
                    ready, _, _ = select.select([rr], [], [], 3)
                    self.assertTrue(ready)
                    self.assertEqual(os.read(rr, 1), b'R')
                    os.close(w); w = None
                    deadline = time.monotonic()+5
                    status = None
                    while time.monotonic() < deadline:
                        found, result = os.waitpid(pid, os.WNOHANG)
                        if found:
                            status = result; break
                        time.sleep(.02)
                    self.assertIsNotNone(status)
                    self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                    receipt = json.loads((directory/'watchdog_restoration.json').read_text())
                    self.assertTrue(receipt['restored'])
                    self.assertEqual(receipt['reason'], 'controller_pipe_closed')
                finally:
                    if w is not None:
                        os.close(w)
                    os.close(rr)
                    try:
                        found, _ = os.waitpid(pid, os.WNOHANG)
                        if not found:
                            os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass


if __name__ == '__main__':
    unittest.main()
