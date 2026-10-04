"""Ensure kernel profiling changes host launch sites only and fails closed."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('kernel_builder',ROOT/'scripts/build_phase20_kernel_probe.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class KernelProbeTests(unittest.TestCase):
    def test_roundtrip_host_instrumentation(self):
        for name, sites in builder.SITES.items():
            source=(ROOT/'scripts'/name).read_text()
            result=builder.instrument(name,source)
            self.assertEqual(len(list(builder.LAUNCH.finditer(result))),len(sites))
            for match, (_, site, _) in zip(builder.LAUNCH.finditer(source),sites):
                replacement=('do { int probe_error=seaqr_kernel_probe::begin('+str(site)+'); '
                    'if(probe_error)return probe_error; '+match.group(0)+
                    ' probe_error=seaqr_kernel_probe::end(); if(probe_error)return probe_error; } while(0);')
                self.assertIn(replacement,result)
                result=result.replace(replacement,match.group(0),1)
            self.assertEqual(result,source)

    def test_missing_launch_rejected(self):
        with self.assertRaises(ValueError):
            builder.instrument('phase20_cuda_integrated.cu','')

    def test_extra_launch_rejected(self):
        name='phase20_cuda_integrated.cu'
        with self.assertRaises(ValueError):
            builder.instrument(name,(ROOT/'scripts'/name).read_text()+'\nfoo<<<1,1>>>(0);')

    def test_single_statement_if_preserved(self):
        name='phase20_cuda_warp_exact.cu'
        result=builder.instrument(name,(ROOT/'scripts'/name).read_text())
        self.assertIn('if(kind)do {',result)
        self.assertIn('else do {',result)


if __name__=='__main__':
    unittest.main()
