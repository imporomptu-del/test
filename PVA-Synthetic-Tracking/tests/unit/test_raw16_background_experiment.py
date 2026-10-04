from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import run_raw16_background_v7 as experiment


class BackgroundExperimentTests(unittest.TestCase):
    def report(self, gpu=False):
        return dict(configuration=dict(identity=dict(sha256=experiment.CONFIG_SHA if gpu else experiment.CPU_CONFIG_SHA),
            effective=dict(background_execution='cuda_temporal_exact_v1' if gpu else 'masked_ufunc', threshold=3.)),
            source=dict(pva_stabilization=dict(configuration=dict(sha256=experiment.v6.CONFIG_SHA))),
            screening=dict(synthetic_tracking=dict(windows=[]), tracks=[dict(x=2., timestamp_ns=123)]), injection=None)

    def test_only_verified_execution_enum_is_normalized(self):
        a, b = self.report(), self.report(True)
        self.assertEqual(experiment.normalized_report(a), experiment.normalized_report(b))
        b['configuration']['effective']['threshold'] = 3.00001
        self.assertNotEqual(experiment.normalized_report(a), experiment.normalized_report(b))
        b = self.report(True)
        b['screening']['tracks'][0]['timestamp_ns'] += 1
        self.assertNotEqual(experiment.normalized_report(a), experiment.normalized_report(b))

    def test_wrong_configuration_identity_rejected(self):
        for mutation in ('unknown_mode', 'wrong_identity', 'old_motion'):
            r = self.report(True)
            if mutation == 'unknown_mode': r['configuration']['effective']['background_execution'] = 'anything'
            if mutation == 'wrong_identity': r['configuration']['identity']['sha256'] = experiment.CPU_CONFIG_SHA
            if mutation == 'old_motion': r['source']['pva_stabilization']['configuration']['sha256'] = 'wrong'
            with self.assertRaises(ValueError): experiment.normalized_report(r)

    def test_cpu_oracle_rejects_any_arithmetic_change(self):
        old = 'class DensePointScreener:\n def _events_for_frame(self, current):\n  value = current + 1\n  return value\n'
        dispatch = '  if self.config.background_execution == "cuda_temporal_exact_v1":\n   return self._events_for_frame_cuda(current)\n'
        candidate = old.replace('  value =', dispatch + '  value =')
        experiment.cpu_oracle(old, candidate)
        with self.assertRaises(ValueError): experiment.cpu_oracle(old, candidate.replace('current + 1', 'current + 2'))
        with self.assertRaises(ValueError): experiment.cpu_oracle(old, old)

    def test_audit_requires_completeness_and_exact_arrays(self):
        counts = {'source_frame': 64, 'pva_motion': 63, 'background_and_filter': 64,
            'background_whitened': 63, 'cuda_shift_stack': 6, 'candidate_ranking': 6,
            'candidate_extract': 6, 'synthetic_association': 6, 'finalize': 1}
        rows = [dict(stage=stage, value=dict(sha256='a'*64)) for stage, n in counts.items() for _ in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'left', Path(directory)/'right'
            def save(path, values): path.write_text(''.join(json.dumps(r)+'\n' for r in values))
            save(left, rows); save(right, rows)
            self.assertTrue(experiment.compare_audits(left, right)['passed'])
            mutated = deepcopy(rows); mutated[6]['value']['sha256'] = 'b'*64
            save(right, mutated)
            self.assertFalse(experiment.compare_audits(left, right)['passed'])
            save(left, rows[:-1]); save(right, rows[:-1])
            self.assertFalse(experiment.compare_audits(left, right)['passed'])

    def test_allowlist_precedes_runtime_and_media(self):
        from argparse import Namespace
        with patch.object(experiment, 'verify_runtime') as verify:
            with self.assertRaises(ValueError): experiment.run(Namespace(clip='unknown'))
        verify.assert_not_called()

    def test_timing_order_is_predeclared_balanced_and_bounded(self):
        from batch_raw16_background_v7 import schedule
        rows = schedule()
        self.assertEqual(len(rows), 9)
        self.assertEqual(len({r['name'] for r in rows}), 9)
        self.assertEqual({r['clip'] for r in rows}, {'0029','0040'})
        self.assertEqual(sum(r['injected'] for r in rows), 1)
        for clip in ('0029','0040'):
            sequence = [r for r in rows if r['clip']==clip and not r['injected']]
            self.assertEqual([r['gpu'] for r in sequence], [False,True,True,False])
            self.assertEqual([r['pair'] for r in sequence], [1,1,2,2])


if __name__ == '__main__':
    unittest.main()
