from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from summarize_raw16_cpu_v6 import semantic, V5_SHA, CONFIG_SHA


class CpuSummaryTests(unittest.TestCase):
    def report(self):
        return dict(source={'pva_stabilization':{'configuration':{'sha256':V5_SHA},
            'metrics':{'median_pva_total_ms':900.,'accepted_transforms_applied':63}}},
            screening={'synthetic_tracking':{'windows':[{
                'synthetic_tracking_metrics':{'cuda':{'library_path':'/tmp/old/build/cuda/libtiny_target_cuda.so'}},
                'window_timing_ms':20.,'score':7.25}],
                'track_pool':[{'hits':6,'x':100.25,'score':9.1}]}},
            injection=None,configuration={'effective':{'threshold':3.}})

    def test_execution_identity_and_known_timings_only(self):
        a=self.report(); b=deepcopy(a)
        b['source']['pva_stabilization']['configuration']['sha256']=CONFIG_SHA
        b['source']['pva_stabilization']['metrics']['median_pva_total_ms']=100.
        w=b['screening']['synthetic_tracking']['windows'][0]
        w['synthetic_tracking_metrics']['cuda']['library_path']='/tmp/new/build/cuda/libtiny_target_cuda.so'
        w['window_timing_ms']=1.
        self.assertEqual(semantic(a),semantic(b))

    def test_subpixel_score_hit_and_threshold_changes_are_not_hidden(self):
        for key in ('hits','x','score'):
            a=self.report();b=deepcopy(a)
            b['screening']['synthetic_tracking']['track_pool'][0][key] += 1e-5
            self.assertNotEqual(semantic(a),semantic(b))
        a=self.report();b=deepcopy(a)
        b['configuration']['effective']['threshold'] += 1e-5
        self.assertNotEqual(semantic(a),semantic(b))

    def test_unknown_motion_config_or_library_path_fails(self):
        a=self.report()
        a['source']['pva_stabilization']['configuration']['sha256']='0'*64
        with self.assertRaises(ValueError): semantic(a)
        a=self.report()
        a['screening']['synthetic_tracking']['windows'][0]['synthetic_tracking_metrics']['cuda']['library_path']='/unverified.so'
        with self.assertRaises(ValueError): semantic(a)

    def test_only_known_injection_path_is_normalized(self):
        a=self.report();b=deepcopy(a)
        a['injection']={'identity':{'path':'/tmp/old/configs/evaluation/raw16_full_frame_controls_v2.json'},'errors':[.25]}
        b['injection']={'identity':{'path':'/tmp/new/configs/evaluation/raw16_full_frame_controls_v2.json'},'errors':[.25]}
        self.assertEqual(semantic(a),semantic(b))
        b['injection']['errors'][0]=.26
        self.assertNotEqual(semantic(a),semantic(b))


if __name__ == '__main__':
    unittest.main()
