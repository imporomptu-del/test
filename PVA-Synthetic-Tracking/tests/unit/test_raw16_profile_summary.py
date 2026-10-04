from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from summarize_raw16_profile_v6 import validate_accounting


class ProfileSummaryTests(unittest.TestCase):
    def fixture(self):
        return dict(wall_s=10., exclusive_sum_s=10., accounting_error_ns=0,
            groups={'other':dict(exclusive_s=4.,fraction=.4),'compute':dict(exclusive_s=6.,fraction=.6)},
            spans=[dict(path='root',group='other',calls=1,inclusive_s=10.,exclusive_s=4.),
                   dict(path='root/child',group='compute',calls=2,inclusive_s=6.,exclusive_s=6.)])

    def test_consistent_nested_totals(self): validate_accounting(self.fixture())

    def test_inclusive_double_counting_rejected(self):
        data=self.fixture(); data['groups']['other']['exclusive_s']=10.
        with self.assertRaises(ValueError): validate_accounting(data)

    def test_wrong_fraction_or_call_count_or_tree_rejected(self):
        for field in ('fraction','calls','path','exclusive_s'):
            data=deepcopy(self.fixture())
            if field=='fraction': data['groups']['other']['fraction']=.6
            else: data['spans'][1][field]={'calls':0,'path':'root/missing/child','exclusive_s':5.}[field]
            with self.assertRaises(ValueError): validate_accounting(data)


if __name__ == '__main__': unittest.main()
