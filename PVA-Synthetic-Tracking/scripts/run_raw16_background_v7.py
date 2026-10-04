"""Opt-in exact GPU background experiment on two frozen 64-frame RAW16 prefixes.

Reject unknown sources/configurations before media access. Audit downloads and
hashes all intermediate arrays; timing mode omits that heavy instrumentation.
Every run must also match the archived v6 report and source/motion identities.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import run_raw16_cpu_v6 as v6
from profile_raw16_efficiency import Instrumentation, compact, sha, source_paths, write_json
from profile_raw16_v6 import BASELINE_HASHES, RUNTIME_SHA, Spans
from summarize_raw16_cpu_v6 import compare_source_motion, digest, difference, semantic
from tiny_target import dense_screen as dense
from tiny_target.raw_background_cuda import DEFAULT_LIBRARY, RawBackgroundCuda

CONFIG = ROOT/'configs/evaluation/raw16_background_v7.json'
CONFIG_SHA = 'c9f5980a80a78655562f9552f7b4b89036d85bb43a8377d59144be818568a816'
CPU_CONFIG_SHA = 'ff0af1af9da5320f46566bb65fbe85fcc6c5112d250a4b08f91526dfc80517d9'
LIBRARY_SHA = '0264c2d422f99c726b3f7826fb5c8c56e325711fd30e2958dd26b9989f8127e6'
INJECTED_HASHES = {
    'report.json': 'ea5ee39afb95971ce1029be0f0650a20a3821adb44106daa7c093dea53ca600f',
    'source_frames.json': 'b1c4670167eb3fa6880ea28c4ea4091a48989cb2aa724cb767de29aa88f5e577',
    'motion_profile.json': '41890ac7c7be265e08a2bb313ab4b2470934dd64ba283d985fc3f8bf38ea379b',
    'checks.json': 'e8e2c34fecdd8bf5c78344444cc56ddad51279c82e61c3f566235be3cb53ddc2',
}


def read(path):
    return json.loads(Path(path).read_text())


def normalized_report(report):
    """Normalize exactly one verified execution enum, never detector policy."""
    expected = {'masked_ufunc': CPU_CONFIG_SHA, 'cuda_temporal_exact_v1': CONFIG_SHA}
    mode = report['configuration']['effective']['background_execution']
    if mode not in expected or report['configuration']['identity']['sha256'] != expected[mode]:
        raise ValueError('Unknown detector configuration/execution pair')
    if report['source']['pva_stabilization']['configuration']['sha256'] != v6.CONFIG_SHA:
        raise ValueError('Both arms must use the same frozen v6 motion configuration')
    result = semantic(report)
    result['detector']['background_execution'] = '<verified_exact_execution>'
    return result


def compare(left, right):
    a, b = normalized_report(read(left/'report.json')), normalized_report(read(right/'report.json'))
    return dict(exact_semantics=a == b, first_difference=difference(a, b),
                left_semantic_sha256=digest(a), right_semantic_sha256=digest(b),
                **compare_source_motion(left, right))


def cpu_oracle(source, candidate):
    def method(text):
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                   and n.name == 'DensePointScreener')
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_events_for_frame')
    original, current = method(source), method(candidate)
    dispatch = ast.parse('if self.config.background_execution == "cuda_temporal_exact_v1":\n'
                         '    return self._events_for_frame_cuda(current)\n').body[0]
    if ast.dump(current.body[0]) != ast.dump(dispatch):
        raise ValueError('Unexpected CPU/GPU dispatch')
    current.body = current.body[1:]
    if ast.dump(original) != ast.dump(current):
        raise ValueError('The frozen CPU background implementation changed')


def verify_runtime(archive, component):
    if sha(archive) != RUNTIME_SHA:
        raise ValueError('Frozen v6 source archive changed')
    frozen = {}
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile():
                continue
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member')
            content = source.extractfile(member).read()
            expected = hashlib.sha256(content).hexdigest()
            if str(path) == 'tiny_target/dense_screen.py':
                cpu_oracle(content.decode(), (ROOT/path).read_text())
            elif sha(ROOT/path) != expected:
                raise ValueError(f'Undeclared change to frozen v6: {path}')
            frozen[str(path)] = expected
    package = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'tiny_target').rglob('*.py')
               if not p.name.startswith('._')}
    expected_package = {n for n in frozen if n.startswith('tiny_target/') and n.endswith('.py')}
    if set(package) != expected_package | {'tiny_target/raw_background_cuda.py'}:
        raise ValueError('Unexpected runtime module inventory')
    report = read(component)
    if not (report['passed'] is True and report['native_enabled'] is True
            and report['frame_comparisons'] == 1128 and len(report['cases']) == 28
            and all(r['exact'] is True for r in report['cases'])
            and any(r.get('stationary') and r['frames']==512 for r in report['cases'])
            and any(r.get('seed_underflow') and r['frames']==8 for r in report['cases'])
            and report['real_media_read'] is False
            and report['point_filter_backend'] == 'unchanged_cpu_opencv'
            and report['direct_gpu_point_filter_probe']['production_enabled'] is False):
        raise ValueError('Complete exact component gate required before media')
    if report['package_sha256'] != package:
        raise ValueError('Runtime changed after generated component gate')
    if report['script_sha256'] != sha(ROOT/'scripts/check_raw16_background_cuda.py'):
        raise ValueError('Component test harness changed')
    if report['cuda_source_sha256'] != sha(ROOT/'tiny_target/detection/cuda/raw_background.cu'):
        raise ValueError('CUDA source changed')
    if report['library_sha256'] != LIBRARY_SHA or sha(DEFAULT_LIBRARY) != LIBRARY_SHA:
        raise ValueError('CUDA binary differs from tested component')
    cpu = read(v6.full.CONFIG)
    gpu = read(CONFIG)
    if sha(CONFIG) != CONFIG_SHA or sha(v6.full.CONFIG) != CPU_CONFIG_SHA:
        raise ValueError('Frozen detector configuration changed')
    if gpu.pop('background_execution') != 'cuda_temporal_exact_v1':
        raise ValueError('Unknown GPU path')
    if cpu.pop('background_execution') != 'masked_ufunc' or cpu != gpu:
        raise ValueError('Only background execution may change')
    return dict(archive_sha256=RUNTIME_SHA, component_sha256=sha(component),
                package_sha256=package, library_sha256=LIBRARY_SHA,
                cpu_oracle_ast_exact=True)


class Audit(Instrumentation):
    """Add resident GPU snapshots to the existing full intermediate audit."""

    def __init__(self, events):
        super().__init__('audit', events)
        self.screener = None
        self.in_background = False

    def record(self, name, value):
        if name == 'background_and_filter' and self.screener._background_cuda is not None:
            state = self.screener._background_cuda.debug_state()
            value.update(location=state['location'], variance=state['variance'], support=state['history'])
        super().record(name, value)

    def install(self, context):
        original = dense.DensePointScreener._events_for_frame
        cv2 = dense._load_cv2()
        original_filter = cv2.filter2D

        def background(screener, frame):
            self.screener = screener
            self.in_background = True
            try:
                return original(screener, frame)
            finally:
                self.in_background = False

        def filtered(image, *args, **kwargs):
            if self.in_background:
                self.record('background_whitened', image)
            return original_filter(image, *args, **kwargs)

        context.enter_context(patch.object(dense.DensePointScreener, '_events_for_frame', background))
        context.enter_context(patch.object(cv2, 'filter2D', filtered))
        super().install(context)


def compare_audits(left, right):
    a, b = [json.loads(s) for s in Path(left).read_text().splitlines()], [json.loads(s) for s in Path(right).read_text().splitlines()]
    counts = {name: sum(r['stage'] == name for r in a) for name in {r['stage'] for r in a}}
    expected = {'source_frame': 64, 'pva_motion': 63, 'background_and_filter': 64,
                'background_whitened': 63, 'cuda_shift_stack': 6, 'candidate_ranking': 6,
                'candidate_extract': 6, 'synthetic_association': 6, 'finalize': 1}
    complete = all(counts.get(k) == n for k, n in expected.items())
    return dict(passed=complete and a == b, complete=complete, exact=a == b,
                first_difference=difference(a, b), event_count=len(a), counts=counts,
                left_sha256=sha(left), right_sha256=sha(right))


def run(args):
    source_paths(args.clip)  # String allowlist only; no media access here.
    if args.injected and args.clip != '0040':
        raise ValueError('Frozen controls are limited to 0040')
    if args.output.exists():
        raise FileExistsError(args.output)
    profiling = getattr(args, 'profile', False)
    if profiling and args.audit:
        raise ValueError('Do not mix array audit downloads with profiling')
    runtime = verify_runtime(args.runtime_archive, args.component)
    suffix = '_injected' if args.injected else ''
    baseline = args.evidence/f'full_frame_{args.clip}{suffix}_v6'
    hashes = INJECTED_HASHES if args.injected else BASELINE_HASHES[args.clip]
    for name, expected in hashes.items():
        if sha(baseline/name) != expected:
            raise ValueError(f'Archived v6 reference changed: {name}')
    audit_path = args.output.parent/(args.output.name + '.audit.jsonl')
    instances = []
    original_init = dense.DensePointScreener.__init__

    def remember(screener, *a, **kw):
        original_init(screener, *a, **kw)
        instances.append(screener)

    error = None
    status = None
    spans = Spans() if profiling else None
    try:
        with ExitStack() as context:
            context.enter_context(patch.object(dense.DensePointScreener, '__init__', remember))
            if args.audit:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                events = context.enter_context(audit_path.open('x'))
                Audit(events).install(context)
            if spans is not None:
                spans.install(context)
                spans.wrap(context, RawBackgroundCuda, 'step', label='cuda_temporal_support_host')
            if args.gpu:
                context.enter_context(patch.object(v6.full, 'CONFIG', CONFIG))
                context.enter_context(patch.object(v6.full, 'FROZEN_HASHES', {
                    **v6.full.FROZEN_HASHES, CONFIG: CONFIG_SHA, DEFAULT_LIBRARY: LIBRARY_SHA}))
            call_args = argparse.Namespace(action='full', output=args.output, evidence=args.evidence,
                clip=args.clip, injected=args.injected, reference=False)
            status = (v6.run(call_args) if spans is None else
                      spans.call('validation_run', 'validation_and_orchestration', v6.run, call_args))
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        for screener in instances:
            screener.close()
        if args.output.is_dir():
            write_json(args.output/'background_experiment.json', dict(
                schema_version='seaqr.raw16-gpu-background-experiment.v7', gpu=args.gpu,
                audit=args.audit, profile=profiling, runtime=runtime, error=error, wrapper_sha256=sha(__file__),
                audit_path=str(audit_path) if args.audit else None,
                audit_sha256=sha(audit_path) if args.audit else None,
                warning='Same bounded development inputs, no real-airborne accuracy claim. '
                        'GPU temporal/support only; CPU OpenCV point filter retained. Defaults unchanged.'))
            if spans is not None and spans.nodes:
                write_json(args.output/'stage_profile.json', dict(timing=spans.summary(), error=error))
    comparison = compare(baseline, args.output)
    checks = read(args.output/'checks.json')['checks']
    passed = (status == 0 and comparison['exact_semantics'] and comparison['source_frames_exact']
              and comparison['motion_points_exact'] and checks['processing_integrity_passed']
              and checks['detection_availability_passed'])
    write_json(args.output/'background_parity.json', dict(passed=passed, comparison=comparison,
        baseline_hashes=hashes, real_airborne_accuracy_validated=False))
    print(json.dumps(dict(passed=passed, clip=args.clip, gpu=args.gpu, audit=args.audit,
                         comparison=comparison)), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--injected', action='store_true')
    raise SystemExit(run(parser.parse_args()))
