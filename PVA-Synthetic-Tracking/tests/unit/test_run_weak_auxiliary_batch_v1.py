from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts' if (ROOT/'scripts/run_weak_auxiliary_batch_v1.py').is_file() else HERE.parent))
import run_weak_auxiliary_batch_v1 as m


class BatchTests(unittest.TestCase):
    def exercise(self,code):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();source=root/Path(m.__file__).name
            source.write_bytes(Path(m.__file__).read_bytes())
            frozen=root/'freeze.json'
            frozen.write_text(json.dumps(dict(files_sha256={source.name:m.sha(source)},original_root='/tmp/generated-only',
                original_audits_sha256={c:'a'*64 for c in ('0029','0126','0055')})))
            with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=code)) as calls:
                if code:
                    with self.assertRaisesRegex(RuntimeError,'failed stage replay_0029'):m.run(frozen,m.sha(frozen))
                else:m.run(frozen,m.sha(frozen))
                self.assertEqual(calls.call_count,1 if code else 7)
            status=json.loads((root/'results_01/batch_status.json').read_text())
            self.assertIs(status['passed'],not bool(code))
            with self.assertRaises(FileExistsError):m.run(frozen,m.sha(frozen))

    def test_stop_at_first_failure_without_overwrite(self):self.exercise(1)
    def test_serial_complete_stage_order(self):self.exercise(0)
    def test_changed_freeze_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'freeze.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'differs'):m.run(p,'0'*64)
            self.assertFalse((p.parent/'results_01').exists())


if __name__=='__main__':unittest.main()
