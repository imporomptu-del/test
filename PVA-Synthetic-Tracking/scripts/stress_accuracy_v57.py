"""Deterministic generated-patch stress test of the nonpromoted V57 shadow.

No camera media, journals, labels or caches are opened. A known analytic point
on a stronger edge is deliberately retained as a counterexample: persistence
does not turn edge preference into proof that no point is present.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from accuracy_v36_context import PointEdgeDiagnostic
from accuracy_v57_persistence import CausalEdgePersistence, PersistenceConfig


ROOT = Path(__file__).resolve().parents[1]
V36_SOURCE = ROOT/'scripts/accuracy_v36_context.py'
V36_TEST = ROOT/'tests/unit/test_accuracy_v36_context.py'
V36_SHA = '3776898144c354343a8074d4188bc976781dc5164a2dc8ff8a652c321b6df866'
V36_TEST_SHA = '9dea356486e76f7df645328abd984c5e94c0c8602e48f3d45ecb83a8bb218738'
CONFIG = dict(required_consecutive_edges=2, maximum_edge_gap_ns=200_000_000,
              maximum_coast_frames=7, maximum_coast_ns=700_000_000)
SEQUENCE_FRAMES = 4
FRAME_INTERVAL_NS = 100_000_000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def generated_cases():
    """Fixed noiseless float64 patches; no fitted or outcome-selected settings."""
    y, x = np.mgrid[-12:13, -12:13].astype(np.float64)
    background = 93.0 + .7*x - 1.3*y + .12*x*x - .05*x*y + .09*y*y
    point = np.exp(-(x*x+y*y)/(2.0*2.0**2))
    edge = np.tanh(x/2.0)
    extended = np.exp(-.5*((x/6.0)**2+(y/1.5)**2))
    lobes = [np.exp(-((x-dx)**2+y*y)/2.0) for dx in (-4.0, 4.0)]
    common = dict(background='93 + .7*x - 1.3*y + .12*x^2 - .05*x*y + .09*y^2',
                  coordinates='integer x,y in [-12,12]; y-major 25x25',
                  noise='none', quantization='none', clipping='none',
                  physical_sensor_calibration=False)
    definitions = [
        ('pure_point_bright', 'bright', 12*point, [dict(kind='isotropic_gaussian', signed_amplitude_dn=12., sigma_px=2., center_xy=[0.,0.])], None),
        ('pure_point_dark', 'dark', -12*point, [dict(kind='isotropic_gaussian', signed_amplitude_dn=-12., sigma_px=2., center_xy=[0.,0.])], None),
        ('pure_edge_bright', 'bright', 20*edge, [], dict(kind='tanh_x', signed_amplitude_dn=20., width_px=2.)),
        ('pure_edge_dark', 'dark', -20*edge, [], dict(kind='tanh_x', signed_amplitude_dn=-20., width_px=2.)),
        ('point_on_strong_edge_bright', 'bright', 6*point+20*edge,
         [dict(kind='isotropic_gaussian', signed_amplitude_dn=6., sigma_px=2., center_xy=[0.,0.])],
         dict(kind='tanh_x', signed_amplitude_dn=20., width_px=2.)),
        ('point_on_strong_edge_dark', 'dark', -6*point+20*edge,
         [dict(kind='isotropic_gaussian', signed_amplitude_dn=-6., sigma_px=2., center_xy=[0.,0.])],
         dict(kind='tanh_x', signed_amplitude_dn=20., width_px=2.)),
        ('extended_gaussian_bright', 'bright', 12*extended,
         [dict(kind='anisotropic_gaussian', signed_amplitude_dn=12., sigma_xy_px=[6.,1.5], center_xy=[0.,0.])], None),
        ('two_lobes_bright', 'bright', 6*lobes[0]+6*lobes[1],
         [dict(kind='isotropic_gaussian', signed_amplitude_dn=6., sigma_px=1., center_xy=[dx,0.]) for dx in (-4.,4.)], None),
        ('quadratic_only', 'bright', np.zeros_like(x), [], None),
    ]
    output = []
    for name, polarity, signal, components, background_edge in definitions:
        # Preserve the exact left-associated expression in the existing V36
        # counterexample, rather than changing rounding via background+(p+e).
        value = background+6.0*point+20.0*edge if name == 'point_on_strong_edge_bright' else background+signal
        patch = np.ascontiguousarray(value, dtype=np.float64)
        patch.flags.writeable = False
        output.append(dict(case_id=name, polarity=polarity, patch=patch,
            generation=dict(common, source_components=components, edge=background_edge),
            analytic_localized_source_present=bool(components),
            known_v36_counterexample=name == 'point_on_strong_edge_bright',
            physical_airborne_class=None))
    return output


def sequence(features, polarity):
    """Four repeated object-centered patches; baseline qualification is assumed.

    Generated coordinates move one pixel per observation for schema mechanics.
    They do not result from detection/tracking or certify physical motion.
    """
    require(polarity in ('bright', 'dark'), 'Known polarity required')
    require(asdict(PersistenceConfig()) == CONFIG, 'Frozen V57 default budgets changed')
    adapter = CausalEdgePersistence()
    output = []
    identity = polarity+':generated'
    for frame in range(SEQUENCE_FRAMES):
        track = dict(track_id=identity, segment=0, measured=True, qualified_moving=True,
                     measurement_source_xy=[128.+frame,128.])
        row = dict(frame_index=frame, timestamp_ns=frame*FRAME_INTERVAL_NS, segment=0,
                   motion=dict(reset=False), tracks=[track])
        reason = ('unknown_uninformative_patch' if not features['informative'] else
                  'point_preferred' if features['point_minus_edge_fraction'] > 0 else 'edge_preferred_or_tie')
        record = dict(features=features, reason=reason)
        before = _canonical([row, record])
        verdicts = adapter.update(row, {(0,identity): record})
        require(_canonical([row,record]) == before, 'Shadow mutated generated inputs')
        require(len(verdicts) == 1 and verdicts[0]['measured'] is True
                and verdicts[0]['baseline_qualified'] is True, 'Unexpected sequence output scope')
        output.append(dict(frame_index=frame, timestamp_ns=frame*FRAME_INTERVAL_NS,
                           assumed_baseline_track=track, verdict=verdicts[0]))
    return output


def run_stress():
    """Return deterministic analytic evidence, never a calibrated accuracy score."""
    require(sha(V36_SOURCE) == V36_SHA and sha(V36_TEST) == V36_TEST_SHA,
            'Unchanged V36 source/counterexample test hash required')
    require(asdict(PersistenceConfig()) == CONFIG, 'Frozen V57 default budgets changed')
    diagnostic = PointEdgeDiagnostic()
    results = []
    for case in generated_cases():
        patch = case['patch']
        require(patch.shape == (25,25) and patch.dtype == np.float64 and np.isfinite(patch).all(),
                'Invalid generated patch')
        before = hashlib.sha256(patch.astype('<f8',copy=False).tobytes(order='C')).hexdigest()
        features = diagnostic.measure(patch, case['polarity'])
        require(hashlib.sha256(patch.astype('<f8',copy=False).tobytes(order='C')).hexdigest() == before,
                'Diagnostic mutated patch')
        decisions = sequence(features, case['polarity'])
        rejected = [item['frame_index'] for item in decisions if not item['verdict']['accepted']]
        results.append({key:value for key,value in case.items() if key != 'patch'} | dict(
            patch_sha256_float64_le_y_major=before, generated_features=features,
            sequence=decisions, rejected_frame_indices=rejected,
            generated_source_present_veto=case['analytic_localized_source_present'] and bool(rejected),
            repeated_patches_are_independent_trials=False))
    by_id = {item['case_id']:item for item in results}
    counterexample = by_id['point_on_strong_edge_bright']
    feature = counterexample['generated_features']
    verdicts = [item['verdict'] for item in counterexample['sequence']]
    require(feature['informative'] is True and feature['point_minus_edge_fraction'] < 0
            and feature['conditional_informative'] is True
            and abs(feature['point_gain_after_edge_fraction']-1.) < 1e-10
            and abs(feature['point_after_edge_amplitude_dn']-6.) < 1e-8,
            'Known point-on-edge analytical counterexample not reproduced')
    require(verdicts[0]['accepted'] is True and verdicts[0]['edge_streak_count'] == 1
            and verdicts[1]['accepted'] is False and verdicts[1]['edge_streak_count'] == 2
            and verdicts[1]['reason'] == 'edge_consecutive_rejected',
            'Expected persistent known-source veto not exercised')
    sources = [V36_SOURCE, V36_TEST, ROOT/'scripts/accuracy_v57_persistence.py', Path(__file__)]
    return dict(schema='seaqr.accuracy-v57-generated-stress.v1', completed=True,
        mechanics_checks_passed=True, case_count=len(results),
        policy_configuration=CONFIG, observations_per_sequence=SEQUENCE_FRAMES,
        timestamp_interval_ns=FRAME_INTERVAL_NS, cases=results,
        known_point_on_strong_edge_counterexample_reproduced=True,
        known_counterexample_rejected=True,
        known_source_present_veto_observed=True, promotion_allowed=False,
        universal_source_preservation_demonstrated=False,
        conclusion='Two adjacent informative edge-preferred measurements veto a known present point on a stronger edge; persistence alone is not a safe source-preservation rule.',
        limitations=[
            'No real media, journals, camera caches or labels were read.',
            'Analytic noiseless float64 patches are not calibrated optical, sensor, quantized 8-bit or RAW16 simulations.',
            'Point/edge template banks differ in size and their gain difference is not a probability or physical-class label.',
            'Generated localized source components do not establish airborne identity or real-world detectability.',
            'Repeated object-centered patches are correlated stress inputs, not independent trials.',
            'Baseline-qualified actual measurements and their positions are stipulated for adapter testing; the detector and tracker were not run.',
            'Extended and multilobed profiles only expose template-shape sensitivity; they are not labeled object categories.',
            'Unknown quadratic-only patches are retained as unknown, not asserted empty real scenes.',
            'No precision, recall, false-positive rate or generalization estimate is reported.',
            'Mechanics passing does not authorize production promotion; the known-source veto is an explicit counterexample.',
        ], source_media_accessed=False, real_journals_scored=False,
        raw16_accessed=False, sealed_holdout_accessed=False,
        production_changed=False, source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), 'New exclusive output path required')
    result = run_stress()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
