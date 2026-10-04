"""The experimental build changes one predicate's location, never policy."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_phase20_peak_gate import BEFORE, AFTER, transform


class PeakGateTransformTests(unittest.TestCase):
    def test_exactly_one_reversible_block_change(self):
        text=(ROOT/'scripts/phase20_cuda_resident.cu').read_text()
        changed=transform(text)
        self.assertEqual(changed.replace(AFTER,BEFORE,1),text)
        self.assertLess(changed.index('if(signed_r<'),changed.index('for(int dy=-2;'))

    def test_missing_and_duplicate_block_fail_closed(self):
        for value in ('',BEFORE+BEFORE,BEFORE.replace('threshold,noise','threshold*2,noise')):
            with self.assertRaises(ValueError):transform(value)


if __name__=='__main__':unittest.main()
