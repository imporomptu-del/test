import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_accuracy_v48_math as audit


class V48MathTests(unittest.TestCase):
    def test_unchanged_math_and_original_globals_with_fresh_evidence(self):
        before=dict(audit.original.run.__globals__)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            output=root/"math.json"
            with mock.patch.object(audit,"OUTPUT_ROOT",root):
                with self.assertRaisesRegex(ValueError,"execute"):
                    audit.run(output)
                self.assertFalse(output.exists())
                result=audit.run(output,execute=True)
                self.assertTrue(result["completed"]);self.assertTrue(result["passed"])
                self.assertEqual(result["issues"],[])
                self.assertEqual(result["matrix"]["realizations"],768)
                self.assertEqual(result["matrix"]["direct_partial_regressions"],2304)
                self.assertEqual(result["centering"]["cases"],38)
                self.assertEqual(len(result["structural_unknowns"]["records"]),4)
                freeze=json.loads(output.with_name("math_freeze.json").read_text())
                self.assertEqual(freeze["manifest"],audit.original.matrix_manifest())
                self.assertEqual(freeze["dependency_sha256"],result["audit_files_sha256"])
                self.assertIn(str(Path(audit.__file__).resolve()),result["audit_files_sha256"])
                for name,digest in result["audit_files_sha256"].items():
                    self.assertEqual(audit.original._sha(name),digest)
                with self.assertRaises(FileExistsError):audit.run(output,execute=True)
        self.assertEqual(before,audit.original.run.__globals__)

    def test_wrong_output_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"dedicated"):
                audit.run(Path(tmp)/"outside.json",execute=True)


if __name__=="__main__":unittest.main()
