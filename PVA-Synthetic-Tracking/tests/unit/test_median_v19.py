"""Median network generation, exhaustive rank proof and strict transformation."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from median_v19 import network,transform,OLD
from build_median_v19 import proof


class MedianV19Tests(unittest.TestCase):
    def test_exhaustive_zero_one_rank(self):
        with tempfile.TemporaryDirectory(prefix='seaqr_median_v19_') as d:
            result=proof(Path(d)/'proof')
        self.assertTrue(result['passed']);self.assertEqual(result['cases'],1<<25)
        self.assertEqual(result['minmax_nodes'],202)

    def test_random_finite_rank_and_topological_order(self):
        rng=np.random.default_rng(619)
        nodes,final=network()
        a=rng.normal(size=(25,10000)).astype(np.float32)
        values={k:a[k] for k in range(25)}
        for k,op,i,j in nodes:
            self.assertLess(i,k);self.assertLess(j,k)
            values[k]=(np.minimum if op=='min' else np.maximum)(values[i],values[j])
        np.testing.assert_array_equal(values[final],np.median(a,axis=0))

    def test_transform_changes_only_network_and_keeps_guarded_reference(self):
        source=(ROOT/'scripts/phase20_cuda_median.cu').read_text()
        result=transform(source)
        before,after=source.split(OLD)
        self.assertTrue(result.startswith(before));self.assertTrue(result.endswith(after))
        self.assertEqual(result.count(OLD),1)
        self.assertIn('bits==0x80000000U',result)
        self.assertIn('mag<0x00800000U',result)
        self.assertIn('mag>=0x7f800000U',result)
        for bad in (source.replace(OLD,''),source+OLD):
            with self.assertRaises(ValueError):transform(bad)


if __name__=='__main__':unittest.main()
