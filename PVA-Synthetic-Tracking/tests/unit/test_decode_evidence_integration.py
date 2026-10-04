"""Synthetic receipt fixtures exercise the complete supplemental verifier.

The full-summary receipt is a fixture for the separately tested full finalizer,
not a claim that these tiny synthetic journals are the real four-clip cohort.
"""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tiny_target.visible_baseline import VisibleConfig, sha256
from tiny_target.visible_decode import decode_contract
from test_speed_finalization import make_run, write

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('decode_evidence_integration', ROOT/'scripts/verify_phase20_decode_evidence.py')
module = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec.loader.exec_module(module)


class DecodeEvidenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.before = make_run(self.root/'before', 2.)
        self.after = make_run(self.root/'after', 4., self.before)
        candidate_cfg = json.loads((self.after/'config.json').read_text())
        candidate_cfg['frame_decode_execution'] = 'prefetch_one'
        write(self.after/'config.json', candidate_cfg)
        frozen = json.loads((self.after/'freeze.json').read_text())
        frozen['config_sha256'] = sha256(self.after/'config.json')
        frozen['files_sha256']['scripts/repeat_phase20_kernel_speed.py'] = 'repeat_script_fixture'
        write(self.after/'freeze.json', frozen)
        rows = ''.join(json.dumps(dict(frame_index=i, candidates=[], timings_ms={}))+'\n' for i in range(128))
        for base in (self.before, self.after):
            for cid in ('0029', '0126', '0055', '0082'):
                path = base/('pva_' + cid)
                (path/'frames.jsonl').write_text(rows)
                if base == self.after:
                    launch = json.loads((path/'launch.json').read_text())
                    launch.update(configuration=candidate_cfg, config_sha256=frozen['config_sha256'],
                        frame_decode=decode_contract('prefetch_one'))
                    write(path/'launch.json', launch)
                    report = json.loads((path/'report.json').read_text())
                    report['frame_decode'] = self.stats('prefetch_one', 1)
                    write(path/'report.json', report)
        self.full = self.root/'full_summary.json'
        write(self.full, dict(freeze_sha256=sha256(self.after/'freeze.json'),
            passed_execution_equivalence=True, full_clip_frames=2741, production_ready=False,
            timing_reference=dict(freeze_sha256=sha256(self.before/'freeze.json'))))
        repeats = self.after/'repeats'
        repeats.mkdir()
        (repeats/'tegrastats.log').write_text('Synthetic telemetry receipt fixture\n')
        summary = dict(passed=True, prefix_frames=128, alternating_pairs_per_clip=3,
            clip_ids=['0126', '0082'], candidate_freeze_sha256=sha256(self.after/'freeze.json'),
            reference_freeze_sha256=sha256(self.before/'freeze.json'),
            script_sha256='repeat_script_fixture', comparisons=[], summary={})
        for cid in ('0126','0082'):
            for pair in range(3):
                values = {}
                for label, base, fps, mode in (('before',self.before,2.,'sequential'),
                                              ('after',self.after,4.,'prefetch_one')):
                    path = repeats/f'{cid}_pair{pair}_{label}'
                    (path/'implementation').mkdir(parents=True)
                    original = base/('pva_' + cid)
                    (path/'implementation/example.py').write_bytes((original/'implementation/example.py').read_bytes())
                    launch = json.loads((original/'launch.json').read_text())
                    launch.update(max_frames=128, frame_decode=decode_contract(mode))
                    write(path/'launch.json', launch)
                    report = dict(completed=True, full_clip=False, frames=128, configuration=launch['configuration'],
                        source_sha256=launch['source_sha256'], processed_fps=fps, elapsed_seconds=128/fps,
                        frame_decode=self.stats(mode,128), timings_ms={})
                    write(path/'report.json', report)
                    (path/'frames.jsonl').write_text(rows)
                    values[label] = dict(fps=fps, timings_ms={}, frame_decode=report['frame_decode'],
                        journal_sha256=sha256(path/'frames.jsonl'))
                summary['comparisons'].append(dict(clip_id=cid, pair=pair,
                    order=['before','after'] if pair % 2 == 0 else ['after','before'], speedup=2., **values))
            summary['summary'][cid] = dict(before_fps_median=2., after_fps_median=4., paired_speedups=[2.,2.,2.])
        self.receipt = repeats/'repeat_summary.json'
        write(self.receipt, summary)
        self.sample = repeats/'0126_pair0_after'

    @staticmethod
    def stats(mode, count):
        return dict(contract=decode_contract(mode), decoded_frames=count, consumed_frames=count,
            read_calls=count, maximum_observed_frames_ahead=int(mode=='prefetch_one'),
            worker_joined=True, capture_released=True, dropped_frames=0)

    def verify(self):
        args = ['verify', '--root',str(self.after),'--reference',str(self.before),
            '--full-summary',str(self.full),'--output',str(self.root/'verified.json')]
        with patch.object(sys, 'argv', args), patch('builtins.print'):
            module.main()

    def test_complete_receipt_passes(self):
        self.verify()
        result = json.loads((self.root/'verified.json').read_text())
        self.assertTrue(result['passed'])
        self.assertEqual(result['repeated_frames_compared'], 1536)
        self.assertFalse(result['production_ready'])

    def test_one_changed_non_timing_pixel_location_fails(self):
        path = self.sample/'frames.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[65]['candidates'] = [dict(x=1,y=2)]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(AssertionError,'non-timing'):
            self.verify()

    def test_unjoined_worker_fails(self):
        path = self.sample/'report.json'
        report = json.loads(path.read_text()); report['frame_decode']['worker_joined'] = False
        write(path, report)
        with self.assertRaisesRegex(ValueError,'drain'):
            self.verify()

    def test_duplicate_repeat_pair_fails(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['comparisons'][1] = receipt['comparisons'][0]
        write(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.verify()

    def test_changed_runtime_snapshot_fails(self):
        (self.sample/'implementation/example.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'snapshot'):
            self.verify()

    def test_optimistic_timing_receipt_fails(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['comparisons'][0]['after']['fps'] *= 2
        write(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError,'timing'):
            self.verify()
