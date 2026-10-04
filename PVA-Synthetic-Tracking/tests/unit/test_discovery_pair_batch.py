import importlib.util
from pathlib import Path
import signal
import subprocess
import unittest
from unittest.mock import MagicMock, patch


spec = importlib.util.spec_from_file_location('discovery_batch', Path(__file__).parents[2]/'scripts/batch_discovery_pair.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchSafetyTests(unittest.TestCase):
    def sensor(self, name, value):
        directory = MagicMock()
        kind, temp = MagicMock(), MagicMock()
        kind.read_text.return_value = name
        if isinstance(value, Exception):
            temp.read_text.side_effect = value
        else:
            temp.read_text.return_value = value
        directory.__truediv__.side_effect = lambda p: kind if p == 'type' else temp
        return directory

    def test_sleeping_gpu_does_not_hide_cpu_and_junction(self):
        sensors = [self.sensor('cpu-thermal', '45000'),
                   self.sensor('tj-thermal', '46000'),
                   self.sensor('gpu-thermal', BlockingIOError('idle'))]
        with patch.object(batch.Path, 'glob', return_value=sensors):
            self.assertEqual(batch.temperatures(), {'cpu-thermal': 45., 'tj-thermal': 46.})

    def test_mandatory_missing_or_invalid_is_failure(self):
        for sensors in ([self.sensor('cpu-thermal', '45000')],
                        [self.sensor('tj-thermal', '45000')],
                        [self.sensor('cpu-thermal', '140000'), self.sensor('tj-thermal', '45000')],
                        [self.sensor('cpu-thermal', 'bad'), self.sensor('tj-thermal', '45000')]):
            with self.subTest(sensors=sensors), patch.object(batch.Path, 'glob', return_value=sensors):
                with self.assertRaises(ValueError):
                    batch.temperatures()

    def test_python310_idle_sysfs_read_is_missing_not_zero(self):
        error = TypeError("can't concat NoneType to bytes")
        with patch.object(batch.Path, 'glob', return_value=[self.sensor('cpu-thermal', '44000'),
                    self.sensor('tj-thermal', '45000'), self.sensor('gpu-thermal', error)]):
            self.assertEqual(batch.temperatures(), {'cpu-thermal': 44., 'tj-thermal': 45.})
        with patch.object(batch.Path, 'glob', return_value=[self.sensor('cpu-thermal', error),
                    self.sensor('tj-thermal', '45000')]):
            with self.assertRaisesRegex(ValueError, 'mandatory CPU'):
                batch.temperatures()

    def test_finished_child_not_signaled(self):
        child = MagicMock()
        child.poll.return_value = 0
        with patch.object(batch.os, 'killpg') as kill:
            batch.stop_owned(child)
        kill.assert_not_called()

    def test_only_owned_process_group_signaled(self):
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        with patch.object(batch.os, 'killpg') as kill:
            batch.stop_owned(child)
        kill.assert_called_once_with(12345, signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=10)

    def test_escalation_bounded_to_owned_group(self):
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('owned', 10), 0]
        with patch.object(batch.os, 'killpg') as kill:
            batch.stop_owned(child)
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (12345, signal.SIGKILL))

    def test_wrong_workspace_rejected_before_access(self):
        for path in ('/tmp', '/tmp/seaqr_discovery_pair_20260928_abcdef/..', '/tmp/unrelated_abcdef'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'wrong workspace'):
                batch.main(path)


if __name__ == '__main__':
    unittest.main()
