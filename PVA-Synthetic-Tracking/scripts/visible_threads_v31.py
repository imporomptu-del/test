"""Pure scope/environment/schedule for one bounded numerical-thread intervention."""
import math

MODES = ('v20', 'v26_default', 'v26', 'v28', 'combined_default', 'combined')
ORDERS = (MODES, tuple(reversed(MODES)), ('v28','combined','v20','combined_default','v26','v26_default'))
KEYS = ('OPENBLAS_NUM_THREADS','GOTO_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')


def policy(mode):
    if mode not in MODES:
        raise ValueError('Unknown numerical-thread arm')
    return mode.removesuffix('_default'), ('inherited' if mode == 'v20' or mode.endswith('_default') else 'one')


def environment(original, mode):
    result = dict(original)
    # Freeze a genuinely unset original policy; never silently rewrite an
    # existing user setting when naming an arm default/inherited.
    if any(original.get(k) is not None for k in KEYS):
        raise ValueError('Unexpected pre-existing numerical thread setting')
    if policy(mode)[1] == 'one':
        result['OPENBLAS_NUM_THREADS'] = '1'
    return result


def scope(clip, mode, frames, audit, traced):
    arm, setting = policy(mode)
    if type(audit) is not bool or type(traced) is not bool or audit and traced:
        raise ValueError('Invalid instrumentation combination')
    if frames is None:
        if clip not in ('0029','0126','0055','0082') or mode != 'combined' or audit or traced:
            raise ValueError('Full runs restricted to gated candidate')
    elif type(frames) is not int or frames != 128 or clip not in ('0126','0082'):
        raise ValueError('Only two development prefixes')
    if audit and mode != 'combined':
        raise ValueError('Only candidate state smokes')
    if traced and mode not in ('combined','combined_default'):
        raise ValueError('Only matched causal traces')


def schedule():
    result = [dict(name='smoke_'+c, clip=c, mode='combined', frames=128, audit=True, traced=False) for c in ('0126','0082')]
    result += [dict(name=f'{c}_repeat{i}_{mode}', clip=c, mode=mode, frames=128, audit=False, traced=False)
        for c in ('0126','0082') for i, order in enumerate(ORDERS) for mode in order]
    return result


def traces():
    return [dict(name=f'trace_{c}_{mode}', clip=c, mode=mode, frames=128, audit=False, traced=True)
        for c, modes in (('0126',('combined_default','combined')), ('0082',('combined','combined_default'))) for mode in modes]


def full():
    return [dict(name='full_'+c, clip=c, mode='combined', frames=None, audit=False, traced=False) for c in ('0029','0126','0055','0082')]


def gate(receipts):
    from combined_v29_protocol import speed_gate, samples
    expected = {s['name']: s for s in schedule() if not s['audit']}
    if set(receipts) != set(expected):
        raise ValueError('Complete six-arm timing schedule required')
    for name, r in receipts.items():
        s = expected[name]
        if (not r['passed'] or r['error'] is not None or r['clip'] != s['clip']
                or r['arm'] != policy(s['mode'])[0] or r['frames'] != 128
                or r['processed_frames'] != 128 or r['state_audit']):
            raise ValueError('Wrong additional control scope')
        if (type(r['wall_s']) not in (int,float) or not math.isfinite(r['wall_s']) or r['wall_s'] <= 0
                or not math.isclose(r['fps'],128/r['wall_s'],rel_tol=1e-12)):
            raise ValueError('Invalid control timing')
        if any(len(v) != 128 or not all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in v)
               for v in samples(r).values()):
            raise ValueError('Invalid control latency samples')
    selected = {s['name']: receipts[s['name']] for s in schedule() if not s['audit'] and s['mode'] in ('v20','v26','v28','combined')}
    original = speed_gate(selected)
    extra = {}
    for clip in ('0126','0082'):
        fps = {m: 384/sum(receipts[f'{clip}_repeat{i}_{m}']['wall_s'] for i in range(3)) for m in MODES}
        extra[clip] = dict(pooled_fps=fps, candidate_not_worse_than_default_controls=fps['combined'] >= max(fps['v26_default'],fps['combined_default']),
            thread_only_combined_speedup=fps['combined']/fps['combined_default'],
            thread_only_gpu_speedup=fps['v26']/fps['v26_default'])
    return dict(passed=original['passed'] and all(r['candidate_not_worse_than_default_controls'] for r in extra.values()),
                original_gate=original, additional_controls=extra)
