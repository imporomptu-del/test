"""Only the declared binary transition is admissible; no policy/numeric tolerances."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import video_checks_v19 as module


class VideoChecksV19Tests(unittest.TestCase):
    def setUp(self):
        self.transition=dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256='d'*64,
            after_library_sha256='e'*64,candidate_build_sha256='f'*64)
        cfg=dict(cuda_median_library='reference.so',threshold=4)
        left=dict(configuration=cfg,source_sha256='source',fps=10,motion_config_sha256='motion',
            package_sha256={'module':'digest'},config_sha256='b'*64,
            exact_cuda_stabilization=dict(library_sha256='d'*64,conformance={'exact':True}),
            external_accelerators={'median':dict(library_path='reference.so',library_sha256='d'*64)})
        right=deepcopy(left);right['configuration']['cuda_median_library']='candidate.so'
        right['config_sha256']='c'*64;right['exact_cuda_stabilization']['library_sha256']='e'*64
        right['external_accelerators']['median'].update(library_path='candidate.so',library_sha256='e'*64)
        report=dict(configuration=right['configuration'],completed=True,frames=4,full_clip=True,
            faint_target_synthetic_branch_enabled=False,counts={},qualified_tracks=[],qualified_track_count=0,
            availability='available_unlabeled',detection_status='available_unlabeled')
        old=deepcopy(report);old['configuration']=cfg
        motion=dict(passed=True,error=None,closed=True,branch='visible',mode='reuse',processed_frames=4,
            reuse_hits=2,reuse_misses=1,adapter_sha256='adapter',method_sha256='method',wrapper_sha256='wrapper',
            runtime_sha256={'module':'digest'},motion=[dict(frame=i,identity={'value':i}) for i in range(1,4)])
        self.values={'r/launch.json':left,'o/launch.json':right,'r/report.json':old,'o/report.json':report,
                     'r.execution.json':deepcopy(motion),'o.execution.json':motion}
        self.config=deepcopy(right['configuration'])

    def call(self,journal_error=None):
        with patch.object(module,'read',side_effect=lambda p:self.values[str(p)]), \
             patch.object(module,'sha',return_value='a'*64), \
             patch.object(module,'validate_decode'),patch.object(module,'shape_accelerator'), \
             patch.object(module,'check_prefix',side_effect=journal_error):
            return module.check(Path('r'),Path('o'),4,True,'candidate',self.config,'c'*64,self.transition)

    def test_declared_transition_passes(self):
        self.assertTrue(self.call()['exact'])

    def test_threshold_change_rejected(self):
        self.config['threshold']=3
        self.values['o/launch.json']['configuration']['threshold']=3
        with self.assertRaisesRegex(AssertionError,'Only the explicit'):self.call()

    def test_reverse_binary_transition_rejected(self):
        self.transition['before_library_sha256']='e'*64
        self.transition['after_library_sha256']='d'*64
        with self.assertRaises(ValueError):self.call()

    def test_non_timing_journal_difference_rejected(self):
        with self.assertRaisesRegex(AssertionError,'numeric journal difference'):
            self.call(AssertionError('numeric journal difference'))

    def test_motion_and_aggregate_differences_rejected(self):
        original=deepcopy(self.values)
        self.values['o.execution.json']['motion'][0]['identity']['value']=99
        with self.assertRaisesRegex(AssertionError,'Complete motion'):self.call()
        self.values=original;self.values['o/report.json']['qualified_track_count']=1
        with self.assertRaisesRegex(AssertionError,'Aggregate'):self.call()


if __name__=='__main__':unittest.main()
