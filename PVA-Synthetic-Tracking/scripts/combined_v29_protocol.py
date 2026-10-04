"""Pure schedule, scope and timing rules for four fixed algorithm arms."""
import math
import numpy as np

WORKLOADS = ('0126', '0082')
CLIPS = ('0029', '0126', '0055', '0082')
ARMS = ('v20', 'v26', 'v28', 'combined')
ORDERS = (ARMS, tuple(reversed(ARMS)), ('v28', 'v20', 'combined', 'v26'))
BOUNDARIES = ('consumer_cadence', 'queue_aware', 'request_to_complete')


def arm_flags(arm):
    if arm not in ARMS:
        raise ValueError('Unknown combined experiment arm')
    return arm in ('v26', 'combined'), arm in ('v28', 'combined')


def schedule():
    return [dict(name='smoke_' + c, clip=c, arm='combined', frames=128, state_audit=True)
            for c in WORKLOADS] + [
        dict(name=f'{c}_repeat{i}_{arm}', clip=c, arm=arm, frames=128, state_audit=False)
        for c in WORKLOADS for i, order in enumerate(ORDERS) for arm in order]


def full_schedule():
    return [dict(name='full_' + c, clip=c, arm='combined', frames=None, state_audit=False) for c in CLIPS]


def validate_scope(clip, arm, frames, state_audit):
    arm_flags(arm)
    if type(state_audit) is not bool or (frames is not None and (type(frames) is not int or frames != 128)):
        raise ValueError('Invalid experiment extent')
    if clip not in CLIPS or (frames == 128 and clip not in WORKLOADS):
        raise ValueError('Outside frozen development scope')
    if frames is None and arm != 'combined':
        raise ValueError('Only the combined full regression is in scope')
    if state_audit and (arm != 'combined' or frames != 128):
        raise ValueError('State audit restricted to combined smokes')


def samples(receipt):
    rows = receipt['execution']['frames']
    return dict(consumer_cadence=receipt['consumer_frame_ms'],
        queue_aware=[(r['consumer_complete_ns']-r['ready_ns'])/1e6 for r in rows],
        request_to_complete=[(r['consumer_complete_ns']-r['request_ns'])/1e6 for r in rows])


def speed_gate(receipts):
    expected = {s['name'] for s in schedule() if not s['state_audit']}
    if set(receipts) != expected:
        raise ValueError('Complete timing schedule required')
    result = {}
    for clip in WORKLOADS:
        arms = {a: [receipts[f'{clip}_repeat{i}_{a}'] for i in range(3)] for a in ARMS}
        for arm, rows in arms.items():
            for r in rows:
                if (not r['passed'] or r['error'] is not None or r['clip'] != clip or r['arm'] != arm
                        or r['frames'] != 128 or r['processed_frames'] != 128 or r['state_audit']):
                    raise ValueError('Invalid timing receipt')
                if type(r['wall_s']) not in (int, float) or not math.isfinite(r['wall_s']) or r['wall_s'] <= 0:
                    raise ValueError('Invalid pipeline duration')
                if not math.isclose(r['fps'], 128/r['wall_s'], rel_tol=1e-12):
                    raise ValueError('FPS denominator differs')
                for values in samples(r).values():
                    if len(values) != 128 or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
                        raise ValueError('Missing/invalid latency samples')
        fps = {a: 384/sum(r['wall_s'] for r in rows) for a, rows in arms.items()}
        comparisons = {}
        for arm in ARMS[1:]:
            paired = [arms[arm][i]['fps']/arms['v20'][i]['fps'] for i in range(3)]
            latency = {}
            for label in BOUNDARIES:
                distributions = {a: [samples(r)[label] for r in arms[a]] for a in ('v20', arm)}
                p95 = {a: float(np.percentile([v for row in values for v in row], 95))
                       for a, values in distributions.items()}
                ratios = [float(np.percentile(distributions[arm][i], 95)/np.percentile(distributions['v20'][i], 95)) for i in range(3)]
                latency[label] = dict(pooled_p95_ms=p95, paired_p95_ratios=ratios,
                    consistent_regression=p95[arm] > p95['v20'] and sum(x > 1 for x in ratios) >= 2)
            comparisons[arm] = dict(speedup=fps[arm]/fps['v20'], paired_speedups=paired, latency=latency)
        combined = comparisons['combined']
        not_worse_than_singles = fps['combined'] >= max(fps['v26'], fps['v28'])
        passed = (combined['speedup'] >= 1.2 and all(x > 1 for x in combined['paired_speedups'])
            and not any(x['consistent_regression'] for x in combined['latency'].values()) and not_worse_than_singles)
        result[clip] = dict(pooled_fps=fps, vs_v20=comparisons,
            combined_not_worse_than_singles=not_worse_than_singles, passed=passed)
    return dict(passed=all(v['passed'] for v in result.values()), clips=result)
