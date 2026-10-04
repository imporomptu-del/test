"""End-to-end gate must reject numerical/output changes, not just changed counts."""
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tiny_target.visible_baseline import VisibleConfig

spec=importlib.util.spec_from_file_location('integrated_gate',Path(__file__).resolve().parents[2]/'scripts/run_phase20_integrated_batch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class IntegratedParityGateTests(unittest.TestCase):
    def check(self,mutator=None):
        cfg=VisibleConfig(motion_backend='pva',state_update_backend='cuda_resident',
            spatial_background='median5',spatial_filter_backend='cuda_median5',cuda_median_library='/old.so',
            pixel_noise_enabled=True,pixel_noise_model='background_residual')
        with tempfile.TemporaryDirectory() as temp:
            before=Path(temp)/'before';after=Path(temp)/'after';before.mkdir();after.mkdir()
            for path,config in [(before,cfg),(after,replace(cfg,cuda_median_library='/new.so',stabilization_execution='cuda_cubic_resident'))]:
                (path/'launch.json').write_text(json.dumps(dict(source_sha256='unchanged',configuration=asdict(config))))
                (path/'report.json').write_text(json.dumps(dict(frames=230,processed_fps=1.5 if path==after else 1.,timings_ms={})))
                with (path/'frames.jsonl').open('w') as f:
                    for i in range(230):
                        row=dict(frame_index=i,timestamp_ns=i*100_000_000,segment=0,source_to_reference=[[1,0,0],[0,1,0],[0,0,1]],
                            candidates=[],tracks=[],tracking_metrics={},coverage=dict(detection_ms=8 if path==after else 10,searchable_pixels=100),
                            motion=dict(pva_failure=False,reset=False,motion_backends=dict(cpu_fallback=False,gaussian_pyramid='PVA',harris='PVA',optical_flow_pyrlk='PVA')))
                        if path==after and i==100 and mutator:mutator(row)
                        f.write(json.dumps(row)+'\n')
            return module.compare_prefix(before,after)

    def test_timing_only_changes_pass(self):
        self.assertTrue(self.check()['exact'])

    def test_changed_candidate_fails(self):
        with self.assertRaises(AssertionError):self.check(lambda row:row['candidates'].append({'x':1}))

    def test_changed_coverage_fails(self):
        with self.assertRaises(AssertionError):self.check(lambda row:row['coverage'].update(searchable_pixels=99))

    def test_pva_fallback_fails(self):
        with self.assertRaises(ValueError):self.check(lambda row:row['motion']['motion_backends'].update(cpu_fallback=True))
