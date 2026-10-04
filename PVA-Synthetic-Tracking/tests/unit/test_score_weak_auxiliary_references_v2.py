"""Generated file-boundary fixtures; no camera media or experimental outcomes."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[2] / 'scripts'
if not (SCRIPTS / 'score_weak_auxiliary_references_v2.py').is_file():
    SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))
import score_weak_auxiliary_references_v2 as m
import score_weak_auxiliary_references_v1 as v1

WRAPPER = v1.helper()
OLD = WRAPPER.historical()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / 'clean.jsonl.gz'
        self.raw = b'{"one":1}\n{"two":2}\n'
        self.path.write_bytes(gzip.compress(self.raw, mtime=0))
        self.approved = {str(self.path): dict(compressed_sha256=m.file_sha(self.path),
            uncompressed=dict(sha256=hashlib.sha256(self.raw).hexdigest(), bytes=len(self.raw), frames=2))}
        self.adapter = m.AuditedMetadata(OLD, self.approved)

    def read(self, adapter=None):
        verified = {}
        with m.VerifiedTextJournal(adapter or self.adapter, self.path, verified) as stream:
            result = [OLD.decode(row) for row in stream]
        return result, verified

    def test_valid_gzip_binding_decode_and_final_rehash(self):
        bindings = {}
        self.adapter.bind(bindings, self.path, m.file_sha(self.path))
        result, verified = self.read()
        self.assertEqual(result, [{'one': 1}, {'two': 2}])
        self.assertEqual(verified[str(self.path)], self.approved[str(self.path)]['uncompressed'])
        WRAPPER.unchanged(self.adapter, bindings)
        # The original historical helper is intentionally still restrictive.
        with self.assertRaisesRegex(ValueError, 'Only metadata'):
            OLD.bind({}, self.path, m.file_sha(self.path))

    def test_compressed_mutation_before_read_fails(self):
        self.path.write_bytes(gzip.compress(b'{}\n', mtime=0))
        with self.assertRaisesRegex(ValueError, 'Changed bound'):
            self.read()

    def test_mutation_after_read_fails_final_rehash(self):
        bindings = {}
        self.adapter.bind(bindings, self.path, m.file_sha(self.path))
        self.read()
        self.path.write_bytes(gzip.compress(self.raw, mtime=1))
        with self.assertRaisesRegex(ValueError, 'Changed metadata'):
            WRAPPER.unchanged(self.adapter, bindings)

    def test_wrong_hash_or_conflicting_binding_fails(self):
        with self.assertRaisesRegex(ValueError, 'differs from pinned'):
            self.adapter.bind({}, self.path, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.adapter.bind({str(self.path): '0'*64}, self.path, m.file_sha(self.path))

    def test_raw_hash_size_and_count_bindings_enforced(self):
        for key, value in [('sha256', '0'*64), ('bytes', len(self.raw)+1), ('frames', 3), ('frames', 1)]:
            approved = deepcopy(self.approved)
            approved[str(self.path)]['uncompressed'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.read(m.AuditedMetadata(OLD, approved))

    def test_unapproved_other_gzip_media_and_relative_paths_fail(self):
        for name in ('other.jsonl.gz', 'video.avi.gz', 'image.npz', 'clean.gz'):
            path = self.root / name
            path.write_bytes(self.path.read_bytes())
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.adapter.bind({}, path, m.file_sha(path))
        with self.assertRaises(ValueError):
            self.adapter.bind({}, Path('clean.jsonl.gz'), m.file_sha(self.path))

    def test_symlink_and_parent_alias_fail(self):
        linked = self.root / 'sub'
        linked.symlink_to(self.root, target_is_directory=True)
        alias = linked / self.path.name
        approved = {str(alias): self.approved[str(self.path)]}
        with self.assertRaises(ValueError):
            m.AuditedMetadata(OLD, approved).bind({}, alias, m.file_sha(self.path))

    def test_truncated_gzip_rejected_even_if_new_compressed_digest_pinned(self):
        self.path.write_bytes(self.path.read_bytes()[:-6])
        approved = deepcopy(self.approved)
        approved[str(self.path)]['compressed_sha256'] = m.file_sha(self.path)
        with self.assertRaises((EOFError, OSError, ValueError)):
            self.read(m.AuditedMetadata(OLD, approved))

    def test_invalid_json_or_text_rows_fail_closed(self):
        for raw in (b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{}', b'\n', b'\xff\n'):
            self.path.write_bytes(gzip.compress(raw))
            approved = {str(self.path): dict(compressed_sha256=m.file_sha(self.path),
                uncompressed=dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), frames=1))}
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                self.read(m.AuditedMetadata(OLD, approved))

    def test_row_size_bound_and_partial_consumption(self):
        with patch.object(m, 'MAX_LINE', 4), self.assertRaisesRegex(ValueError, 'oversized'):
            self.read()
        with self.assertRaisesRegex(ValueError, 'entire journal'):
            with m.VerifiedTextJournal(self.adapter, self.path, {}) as stream:
                next(stream)

    def test_historical_plain_json_guard_unchanged(self):
        path = self.root / 'metadata.json'
        write_json(path, {'ok': True})
        self.assertEqual(self.adapter.read(self.adapter.bind({}, path, OLD.sha(path))), {'ok': True})


class FullPolicyIntegrationTests(unittest.TestCase):
    """Real v1 score(), score_rows(), matching, EOF and output paths; synthetic data."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.freeze = self.root / 'freeze.json'
        shutil.copy2(SCRIPTS / 'score_weak_auxiliary_references_v1.py', self.root)
        write_json(self.freeze, dict(schema='seaqr.weak-auxiliary-replay.freeze.v1', pre_run=True,
            files_sha256={'score_weak_auxiliary_references_v1.py': m.V1_SHA}))
        self.freeze_sha = m.file_sha(self.freeze)
        self.output = self.root / 'score.json'
        self.samples = []
        for i in range(356):
            actual = '0/bright:1' if i < 352 else ('0/bright:2' if i < 355 else None)
            qualified = actual if i < 352 else None
            saved = {stage: dict(hit=value is not None, assigned_id=value,
                all_gated_ids=[value] if value else []) for stage, value in zip(OLD.STAGES, (actual, qualified))}
            self.samples.append(dict(panel='synthetic', clip_id='0029', window_id=str(i), frame_index=0,
                source_xy=[10.,20.] if i < 352 else ([50.,60.] if i < 355 else [80.,90.]),
                position_uncertainty_px=1., polarity='bright', saved=saved))
        self.audits = {}
        for clip in m.CLIPS:
            base = self.root / clip
            base.mkdir()
            tracks = [dict(segment=0, track_id='bright:'+str(i), measured=True,
                qualified_moving=i == 1, measurement_source_xy=xy, source_xy=xy)
                for i, xy in [(1,[10.,20.]), (2,[50.,60.])]]
            clean = dict(frame_index=0, timestamp_ns=0, segment=0, tracks=tracks, motion={},
                source_to_reference=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                coverage=dict(full_shape_hw=[3190,4784], configured_crop=None, native_pixel_sampling=True,
                    detection_ready=True, warmup=False, searchable_pixels=123))
            primary = dict(frame_index=0, timestamp_ns=0, segment=0, records=tracks)
            event = dict(identity='0/bright:3', primary_track_id='bright:3', record_type='auxiliary_gap_support',
                ordinary_measurement=False, qualified_detection=False, physical_identity_verified=False,
                current_weak_observation=True, prediction_from_weak=False, evidence_type='current_weak_observation',
                source_xy=[81.,90.], origin_measurement_source_xy=[80.,90.], frame_index=0, origin_frame_index=0,
                current_weak_measurement_reference_xy=[80.,90.], strong_anchor_timestamp_ns=0)
            auxiliary = dict(frame_index=0, timestamp_ns=0, segment=0, capture_scheduled=True,
                auxiliary_records=[event], capture_calls=[dict(status='geometrically_complete_capture_supplied')],
                auxiliary_metrics=dict(prior_count=1, prior_strong_eligible_count=1, prepared_query_count=1,
                    actual_provider_calls=1, decisions=[dict(status='weak_auxiliary_correction')], dropped_auxiliary=[]))
            pins, uncompressed = {}, {}
            for name, row in zip(m.NAMES, (clean, primary, auxiliary)):
                raw = (json.dumps(row)+'\n').encode()
                (base/name).write_bytes(gzip.compress(raw, mtime=0))
                pins[str(base/name)] = m.file_sha(base/name)
                uncompressed[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), frames=1)
            receipt = dict(schema='seaqr.weak-auxiliary-replay.run.v1', passed=True, error=None,
                clip=clip, expected_frames=1, processed_frames=1, freeze_sha256=self.freeze_sha,
                source_sha256=OLD.SOURCES[clip], production_changed=False, media_accessed=False,
                detector_rerun=False, weak_learning_enabled=False,
                artifacts_sha256={Path(p).name:d for p,d in pins.items()}, artifact_uncompressed=uncompressed)
            write_json(base/'receipt.json', receipt)
            pins[str(base/'receipt.json')] = m.file_sha(base/'receipt.json')
            audit = dict(schema='seaqr.weak-auxiliary-replay.audit.v1', passed=True, clip=clip, frames=1,
                freeze_sha256=self.freeze_sha, source_sha256=OLD.SOURCES[clip], input_files_sha256=pins,
                production_changed=False, source_media_accessed=False,
                **{field: True for field in ('primary_records_metrics_exact',
                    'historical_private_state_output_learning_digests_exact',
                    'current_primary_state_output_learning_isolation_exact', 'all_primary_query_priors_verified',
                    'auxiliary_numerical_and_lifecycle_checks_passed')})
            write_json(base/'independent_audit.json', audit)
            self.audits[clip] = m.file_sha(base/'independent_audit.json')

    def run_policy(self, mutate=None):
        spec = importlib.util.spec_from_file_location('_fixture_v1', self.root/'score_weak_auxiliary_references_v1.py')
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        wrapper = SimpleNamespace(**{name:getattr(WRAPPER,name) for name in ('assigned','unchanged','write_fresh')})
        wrapper.load_references = lambda *args: dict(samples=self.samples, counts=WRAPPER.sample_counts(self.samples))
        with patch.dict(OLD.COUNTS, {clip:1 for clip in m.CLIPS}):
            bindings = {}
            approved = m.audited_journals(OLD, self.root, self.audits, self.freeze_sha, bindings)
            if mutate:
                mutate(policy)
            return m.execute_policy(policy, wrapper, OLD, approved, {'test_fixture':True}, bindings,
                dict(directory=self.root, references=self.root/'unused_fixture_reference.json', freeze=self.freeze,
                    freeze_sha256=self.freeze_sha, output=self.output))

    def test_complete_gzip_score_reuses_policy_and_does_not_promote_weak(self):
        result = self.run_policy()
        self.assertEqual(result['primary_hits'], dict(actual_measurement=355, qualified_measurement=352))
        self.assertEqual(result['auxiliary_reference_hits']['current_weak'], 1)
        self.assertEqual(len(result['decompressed_journals_verified']), 9)
        self.assertEqual(len(result['reference_records']), 356)
        self.assertFalse(result['production_promoted'])
        self.assertFalse(result['independent_airborne_accuracy_established'])
        self.assertEqual(json.loads(self.output.read_text()), result)
        with self.assertRaises(ValueError):
            OLD.regular(self.root/'0029/clean.jsonl.gz')

    def test_existing_report_never_overwritten(self):
        self.output.write_text('preserved')
        with self.assertRaisesRegex(ValueError, 'Fresh absolute'):
            self.run_policy()
        self.assertEqual(self.output.read_text(), 'preserved')

    def test_pinned_audit_or_receipt_mutation_refused(self):
        for name in ('independent_audit.json', 'receipt.json'):
            path = self.root/'0029'/name
            raw = path.read_bytes()
            path.write_bytes(raw+b' ')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Changed bound'):
                self.run_policy()
            self.assertFalse(self.output.exists())
            path.write_bytes(raw)

    def test_post_scoring_journal_tamper_blocks_publication(self):
        def mutate(policy):
            original = policy.score_rows
            def score_rows(*args, **kwargs):
                result = original(*args, **kwargs)
                path = self.root/'0029/clean.jsonl.gz'
                path.write_bytes(path.read_bytes()+b'changed')
                return result
            policy.score_rows = score_rows
        with self.assertRaisesRegex(ValueError, 'Changed metadata'):
            self.run_policy(mutate)
        self.assertFalse(self.output.exists())

    def test_failed_audit_flag_and_cross_bound_journal_refused(self):
        path = self.root/'0029/independent_audit.json'
        original = json.loads(path.read_text())
        for field in ('passed', 'primary_records_metrics_exact',
                      'auxiliary_numerical_and_lifecycle_checks_passed'):
            audit = deepcopy(original)
            audit[field] = False
            write_json(path, audit)
            self.audits['0029'] = m.file_sha(path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Audit failed'):
                self.run_policy()
        audit = deepcopy(original)
        audit['input_files_sha256'][str(self.root/'0029/primary.jsonl.gz')] = '0'*64
        write_json(path, audit)
        self.audits['0029'] = m.file_sha(path)
        with self.assertRaisesRegex(ValueError, 'Receipt/audit journal mismatch'):
            self.run_policy()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
