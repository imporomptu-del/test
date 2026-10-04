"""Clock guard tests: fake hardware only; never touch real sysfs or video."""
import copy
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
SCRIPT = ROOT/'scripts/visible_clocks_v32.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parent/'visible_clocks_v32.py'
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


class TestClocks(unittest.TestCase):
    def setUp(self):
        self.h = Hardware()
        self.saved = v.snapshot(self.h.read)

    def test_round_trip_only_four_floors(self):
        v.validate_original(self.saved, self.h.read)
        v.set_policy(self.saved, True, self.h.read, self.h.write)
        self.assertEqual(len(self.h.writes), 4)
        v.set_policy(self.saved, True, self.h.read, self.h.write)
        self.assertEqual(len(self.h.writes), 4)
        result = v.restore(self.saved, self.h.read, self.h.write)
        self.assertTrue(result['restored'])
        self.assertEqual(len(self.h.writes), 8)
        self.assertEqual(result['actual'], self.saved)

    def test_partial_write_failure_is_restorable(self):
        def failure(path, value):
            if len(self.h.writes) == 2:
                raise OSError('simulated third policy write failure')
            self.h.write(path, value)
        with self.assertRaises(OSError):
            v.set_policy(self.saved, True, self.h.read, failure)
        self.assertNotEqual(v.snapshot(self.h.read), self.saved)
        self.assertTrue(v.restore(self.saved, self.h.read, self.h.write)['restored'])

    def test_restore_continues_after_failure_and_reports_it(self):
        v.set_policy(self.saved, True, self.h.read, self.h.write)
        first = next(iter(v.POLICIES.values()))
        bad = first['path']+'/'+first['minimum']
        def failure(path, value):
            if path == bad:
                raise OSError('simulated read-only policy')
            self.h.write(path, value)
        result = v.restore(self.saved, self.h.read, failure)
        self.assertFalse(result['restored'])
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['actual']['gpu'], self.saved['gpu'])

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
        with self.assertRaises(ValueError):
            v.set_policy(self.saved, True, self.h.read, self.h.write)
        self.assertFalse(self.h.writes)
        self.assertFalse(v.restore(self.saved, self.h.read, self.h.write)['restored'])
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
