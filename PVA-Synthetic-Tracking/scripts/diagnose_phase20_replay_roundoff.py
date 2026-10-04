"""Separate numerical diagnostics; never replace or relabel the exact comparison."""
import argparse
import itertools
import json
import math
from pathlib import Path

from score_phase20_accuracy import digest
from run_phase20_maturity import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    exact = json.loads((a.root / 'pva_exact_replay_match.json').read_text())
    paths = [a.root / 'current_replay/pva_0126/frames.jsonl', a.root / 'pva_0126/frames.jsonl']
    numeric, other = {}, []

    def compare(x, y, path):
        if x == y:
            return
        if isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for k in x:
                compare(x[k], y[k], path + '.' + k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for u, v in zip(x, y):
                compare(u, v, path + '[]')
        elif isinstance(x, float) and isinstance(y, float) and math.isfinite(x) and math.isfinite(y):
            s = numeric.setdefault(path, dict(count=0, max_absolute_difference=0.0))
            s['count'] += 1
            s['max_absolute_difference'] = max(s['max_absolute_difference'], abs(x-y))
        else:
            other.append(dict(path=path, before=x, after=y))

    count = 0
    with paths[0].open() as left, paths[1].open() as right:
        for l, r in itertools.zip_longest(left, right):
            if l is None or r is None:
                raise ValueError('Unequal lengths')
            x, y = json.loads(l), json.loads(r)
            for key in ('frame_index', 'timestamp_ns', 'segment', 'source_to_reference',
                        'candidates', 'tracks', 'tracking_metrics'):
                compare(x[key], y[key], key)
            cx, cy = dict(x['coverage']), dict(y['coverage'])
            cx.pop('detection_ms', None); cy.pop('detection_ms', None)
            compare(cx, cy, 'coverage')
            count += 1
    if count != exact['frames']:
        raise ValueError('Wrong comparison length')
    write(a.root / 'pva_roundoff_diagnostic.json', dict(
        frames=count, exact_comparison_passed=exact['exact_candidate_tracking_coverage_match'],
        exact_candidates_mapping_and_coverage=not any(
            k.split('.')[0] in ('frame_index', 'timestamp_ns', 'segment', 'source_to_reference', 'candidates', 'coverage')
            for k in numeric) and not other,
        all_discrete_fields_identical=not other,
        numeric_differences=numeric, non_float_differences=other,
        maximum_absolute_numeric_difference=max((v['max_absolute_difference'] for v in numeric.values()), default=0),
        interpretation='Exact equality failed. Tiny cross-platform floating-point differences are consistent with roundoff; this does not alter the exact result or any reference/localization tolerance.',
        journals_sha256={str(p.resolve()): digest(p) for p in paths}))


if __name__ == '__main__':
    main()
