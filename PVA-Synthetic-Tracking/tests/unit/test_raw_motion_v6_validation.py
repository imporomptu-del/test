from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import run_raw16_cpu_v6 as validation


class CpuV6ValidationTests(unittest.TestCase):
    def fixture(self):
        value = dict(eligible={'shape':[100], 'sha256':'a'*64},
                     selected={'shape':[30], 'sha256':'b'*64}, reasons={}, coverage={})
        return dict(passed=True, baseline_sha256=validation.BASELINE_SHA, cases=[
            dict(name=name, points=100, exact=True,
                 outputs={mode:deepcopy(value) for mode in ('frozen_v5','current_reference','batched')})
            for name in ('native_raw16','native_raw16_masked','odd_u8',
                         'odd_raw12_radius8','large_radius_fallback')])

    def test_frozen_execution_only_config_and_gate(self):
        validation.verify_config()
        validation.verify_cpu_report(self.fixture())

    def test_exact_label_does_not_hide_different_indices(self):
        data=self.fixture()
        data['cases'][0]['outputs']['batched']['selected']['sha256']='c'*64
        with self.assertRaises(ValueError):
            validation.verify_cpu_report(data)

    def test_missing_case_or_wrong_oracle_rejected(self):
        for change in ('case','oracle'):
            data=self.fixture()
            if change=='case': data['cases'].pop()
            else: data['baseline_sha256']='0'*64
            with self.assertRaises(ValueError): validation.verify_cpu_report(data)

    def test_empty_identity_not_a_parity_pass(self):
        data=self.fixture()
        data['cases'][0]['outputs']={k:{} for k in ('frozen_v5','current_reference','batched')}
        with self.assertRaises(ValueError): validation.verify_cpu_report(data)


if __name__ == '__main__':
    unittest.main()
