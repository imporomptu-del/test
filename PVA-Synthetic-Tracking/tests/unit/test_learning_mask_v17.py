from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from learning_mask_v17 import LearningMaskV17, reference
from build_learning_mask_v17 import build
from run_visible_v17 import scope
from batch_visible_v17 import schedule


class ExactLearningMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='seaqr-v17-unit-')
        cls.candidate = LearningMaskV17(build(Path(cls.temp.name)/'build'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rounding_edges_overlap_strides_and_noncanonical_bool(self):
        s = np.full((35, 82), 255, np.uint8).view(bool)[:, ::2]
        regions = [dict(support_reference_xy=[[0,0],[40,34],[.5,1.5],[2.5,3.5],[-.5,0],
                                             [20,20],[20,20],[1e100,-1e100]])]
        before = s.tobytes()
        for margin in (.5, 1, 2, 3.5, 8, 16):
            a,b = reference(s, regions, margin), self.candidate(s, regions, margin)
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(before, s.tobytes())

    def test_dense_branch_empty_and_non_bool_preserved(self):
        for shape in ((1,1), (19,23)):
            for dtype in (bool, np.uint8):
                s = np.ones(shape, dtype)
                for regions in ([], [dict(support_reference_xy=[[0,0]]*1024)]):
                    a,b=reference(s,regions,16), self.candidate(s,regions,16)
                    self.assertEqual(a.dtype,b.dtype)
                    self.assertEqual(a.tobytes(),b.tobytes())

    def test_invalid_input_rejected(self):
        for margin in (0,-1,17,float('nan')):
            with self.assertRaises(ValueError):
                self.candidate(np.ones((4,4),bool), [], margin)
        for region in ([],[[float('nan'),0]],[[0,0]]*1025):
            with self.assertRaises(ValueError):
                self.candidate(np.ones((4,4),bool), [dict(support_reference_xy=region)], 2)

    def test_native_failure_does_not_fallback(self):
        with self.assertRaises(RuntimeError):
            self.candidate.sparse(np.ones((4,4),bool), [np.array([[100,100]])], np.array([[0,0]]))

    def test_8bit_scope_and_schedule(self):
        for args in [('0040','candidate',128), ('0029','candidate',128), ('0126','candidate',64), ('0126','bad',128)]:
            with self.assertRaises(ValueError):
                scope(*args)
        rows=schedule()
        self.assertEqual(len(rows),12)
        self.assertEqual([r['mode'] for r in rows[:8]], ['reference','candidate']*2+['candidate','reference']*2)
        self.assertEqual([r['clip'] for r in rows[8:]], ['0029','0126','0055','0082'])
        for r in rows:
            scope(r['clip'],r['mode'],r['frames'])
