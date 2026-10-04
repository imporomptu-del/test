"""Report-only transport repair; execute the immutable v1 scoring policy.

Only the nine explicitly audited JSONL gzip journals gain access. Historical
file guards and all matching/denominator/scoring code remain unchanged on disk.
No media, capture arrays, detector, tracker or native libraries are opened.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

V1_SHA = '78ea505d348951fd73f9b3c3fd5aa0aaeb13c9ba87debbb314d13daa41024952'
CLIPS = ('0029', '0126', '0055')
NAMES = ('clean.jsonl.gz', 'primary.jsonl.gz', 'auxiliary.jsonl.gz')
MAX_LINE = 64 * 1024 * 1024  # Same row bound as the completed independent audit.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def literal_file(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'Literal absolute regular file required')
    return path


def file_sha(path):
    result = hashlib.sha256()
    with literal_file(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


class AuditedMetadata:
    """Proxy only bind/sha; leave the historical JSON and scoring helpers intact."""
    def __init__(self, historical, approved):
        self.historical = historical
        self.approved = deepcopy(approved)

    def __getattr__(self, name):
        return getattr(self.historical, name)

    def compressed_path(self, path):
        path = literal_file(path)
        require(str(path) in self.approved and path.name in NAMES,
                'Compressed journal was not explicitly audited')
        return path

    def sha(self, path):
        if str(path) in self.approved:
            return file_sha(self.compressed_path(path))
        return self.historical.sha(path)

    def bind(self, bindings, path, expected):
        if str(path) not in self.approved:
            return self.historical.bind(bindings, path, expected)
        path = self.compressed_path(path)
        require(digest_ok(expected) and expected == self.approved[str(path)]['compressed_sha256'],
                'Journal digest differs from pinned audit')
        require(self.sha(path) == expected, 'Changed bound compressed journal')
        require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting binding')
        bindings[str(path)] = expected
        return path


class VerifiedTextJournal:
    """Bounded streaming UTF-8 reader; verify exact decompressed bytes at EOF."""
    def __init__(self, adapter, path, verified):
        self.path = adapter.compressed_path(path)
        self.expected = adapter.approved[str(self.path)]['uncompressed']
        adapter.bind({}, self.path, adapter.approved[str(self.path)]['compressed_sha256'])
        self.verified = verified
        self.digest = hashlib.sha256()
        self.size = self.frames = 0
        self.finished = False

    def __enter__(self):
        self.stream = gzip.open(self.path, 'rb')
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        raw = self.stream.readline(MAX_LINE + 1)
        if not raw:
            observed = dict(sha256=self.digest.hexdigest(), bytes=self.size, frames=self.frames)
            require(observed == self.expected, 'Decompressed journal binding differs')
            self.verified[str(self.path)] = observed
            self.finished = True
            raise StopIteration
        require(len(raw) <= MAX_LINE and raw.strip() and raw.endswith(b'\n'),
                'Missing, oversized or unterminated journal row')
        self.size += len(raw)
        self.frames += 1
        require(self.size <= self.expected['bytes'] and self.frames <= self.expected['frames'],
                'Excess decompressed journal extent')
        self.digest.update(raw)
        return raw.decode('utf-8', errors='strict')

    def __exit__(self, kind, value, trace):
        self.stream.close()
        if kind is None:
            require(self.finished, 'Scoring did not consume the entire journal')


def audited_journals(old, directory, audits_sha256, freeze_sha256, bindings):
    require(set(audits_sha256) == set(CLIPS), 'Exactly three audit pins required')
    approved = {}
    for clip in CLIPS:
        base = directory / clip
        audit_path = base / 'independent_audit.json'
        audit = old.read(old.bind(bindings, audit_path, audits_sha256[clip]))
        require(audit.get('schema') == 'seaqr.weak-auxiliary-replay.audit.v1'
                and audit.get('clip') == clip and audit.get('frames') == old.COUNTS[clip]
                and audit.get('freeze_sha256') == freeze_sha256
                and audit.get('source_sha256') == old.SOURCES[clip], 'Audit provenance differs')
        for field in ('passed', 'primary_records_metrics_exact',
                      'historical_private_state_output_learning_digests_exact',
                      'current_primary_state_output_learning_isolation_exact',
                      'all_primary_query_priors_verified', 'auxiliary_numerical_and_lifecycle_checks_passed'):
            require(audit.get(field) is True, 'Audit failed: ' + field)
        require(audit.get('production_changed') is False and audit.get('source_media_accessed') is False,
                'Audit scope differs')
        expected = audit['input_files_sha256']
        receipt_path = base / 'receipt.json'
        receipt = old.read(old.bind(bindings, receipt_path, expected[str(receipt_path)]))
        require(receipt.get('schema') == 'seaqr.weak-auxiliary-replay.run.v1'
                and receipt.get('passed') is True and receipt.get('error') is None
                and receipt.get('clip') == clip and receipt.get('freeze_sha256') == freeze_sha256
                and receipt.get('source_sha256') == old.SOURCES[clip]
                and receipt.get('processed_frames') == receipt.get('expected_frames') == old.COUNTS[clip],
                'Receipt provenance/completeness differs')
        for field in ('production_changed', 'media_accessed', 'detector_rerun', 'weak_learning_enabled'):
            require(receipt.get(field) is False, 'Receipt scope differs: ' + field)
        require(set(receipt['artifacts_sha256']) == set(receipt['artifact_uncompressed']) == set(NAMES),
                'Changed compressed journal inventory')
        for name in NAMES:
            path = base / name
            digest = receipt['artifacts_sha256'][name]
            require(digest_ok(digest) and expected[str(path)] == digest, 'Receipt/audit journal mismatch')
            raw = receipt['artifact_uncompressed'][name]
            require(set(raw) == {'sha256', 'bytes', 'frames'} and digest_ok(raw['sha256'])
                    and type(raw['bytes']) is int and raw['bytes'] > 0
                    and type(raw['frames']) is int and raw['frames'] == old.COUNTS[clip],
                    'Invalid decompressed journal binding')
            approved[str(path)] = dict(compressed_sha256=digest, uncompressed=raw)
    return approved


def execute_policy(v1, wrapper, old, approved, provenance, bindings, arguments):
    """Inject scoped I/O adapters into a fresh imported v1 module, not its policy."""
    adapter = AuditedMetadata(old, approved)
    verified = {}

    def open_journal(path, mode):
        require(mode == 'rt', 'Report journals are read-only text')
        return VerifiedTextJournal(adapter, path, verified)

    def unchanged(active, scoring_bindings):
        require(set(verified) == set(approved), 'Not all nine journals were consumed')
        wrapper.unchanged(active, scoring_bindings)
        wrapper.unchanged(old, bindings)

    def write_fresh(active, output, value):
        result = dict(value, schema='seaqr.weak-auxiliary-references.v2',
                      original_scoring_schema=value['schema'], reporting_repair=provenance,
                      decompressed_journals_verified=verified)
        result['inputs_sha256'] = dict(bindings, **value['inputs_sha256'])
        return wrapper.write_fresh(active, output, result)

    original_helper, original_gzip = v1.helper, v1.gzip
    try:
        v1.helper = lambda: SimpleNamespace(historical=lambda: adapter,
            load_references=wrapper.load_references, assigned=wrapper.assigned,
            unchanged=unchanged, write_fresh=write_fresh)
        v1.gzip = SimpleNamespace(open=open_journal)
        return v1.score(**arguments)
    finally:
        v1.helper, v1.gzip = original_helper, original_gzip


def score(manifest, manifest_sha256, output):
    # This bootstrap imports only the known metadata scorer, then uses its strict
    # decoder and guards for every metadata input (including the repair manifest).
    bootstrap_path = Path(__file__).resolve().with_name('score_feature_selection_references.py')
    require(file_sha(bootstrap_path) ==
            'b8698675b8296c66a4923e625219a035a5c3082a0b9b711543830a7072febd20',
            'Bootstrap historical helper changed')
    spec = importlib.util.spec_from_file_location('_report_repair_bootstrap', bootstrap_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    bindings = {}
    doc = bootstrap.read(bootstrap.bind(bindings, manifest, manifest_sha256))
    require(doc.get('schema') == 'seaqr.weak-auxiliary-report-repair.freeze.v2'
            and doc.get('report_only') is True, 'Pinned report-only repair manifest required')
    root = Path(manifest).parent
    require(set(doc['files_sha256']) == {Path(__file__).name, 'test_score_weak_auxiliary_references_v2.py'},
            'Repair source/test inventory differs')
    for name, digest in doc['files_sha256'].items():
        bootstrap.bind(bindings, root / name, digest)
    require(Path(__file__).resolve() == root / Path(__file__).name, 'Run the pinned repair copy')
    freeze, freeze_sha = Path(doc['original_freeze']), doc['original_freeze_sha256']
    frozen = bootstrap.read(bootstrap.bind(bindings, freeze, freeze_sha))
    require(frozen.get('schema') == 'seaqr.weak-auxiliary-replay.freeze.v1'
            and frozen.get('pre_run') is True, 'Original pre-run freeze required')
    for name, digest in frozen['files_sha256'].items():
        require(Path(name).name == name, 'Unsafe frozen filename')
        bootstrap.bind(bindings, freeze.parent / name, digest)
    path = freeze.parent / 'score_weak_auxiliary_references_v1.py'
    bootstrap.bind(bindings, path, V1_SHA)
    spec = importlib.util.spec_from_file_location('_immutable_auxiliary_scorer_v1', path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    wrapper = v1.helper()
    old = wrapper.historical()
    require(file_sha(bootstrap.__file__) == wrapper.HISTORICAL_SHA, 'Bootstrap changed during import')
    bootstrap.bind(bindings, Path(bootstrap.__file__).resolve(), wrapper.HISTORICAL_SHA)
    directory = Path(doc['directory'])
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir()
            and directory == freeze.parent / 'results_01', 'Original replay directory required')
    approved = audited_journals(old, directory, doc['audits_sha256'], freeze_sha, bindings)
    provenance = dict(manifest_sha256=manifest_sha256, original_scorer_sha256=V1_SHA,
        scope='Compressed metadata I/O only; immutable v1 scoring and matching policy reused.',
        production_changed=False, detector_rerun=False, settings_changed=False)
    return execute_policy(v1, wrapper, old, approved, provenance, bindings,
        dict(directory=directory, references=freeze.parent / 'references.json', freeze=freeze,
             freeze_sha256=freeze_sha, output=output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps({key: result[key] for key in ('passed_integrity', 'primary_hits', 'auxiliary_reference_hits')}))
