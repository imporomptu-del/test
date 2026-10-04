"""Export separate observation alerts and track context from a bound journal.

No media decoding, tracker execution, threshold change, or qualification changes.
Completion is published only after full inventory and before/after hash checks.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_output import ObservationOutput
import tiny_target.visible_output as implementation


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: '+key)
        result[key] = value
    return result


def _invalid(value):
    raise ValueError('Nonfinite JSON value: '+value)


def decode(line):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON exponent: '+value)
        return number
    return json.loads(line, object_pairs_hook=_unique, parse_constant=_invalid,
                      parse_float=finite_float)


def replay(journal, expected_sha256, expected_frames, stream_id, output):
    journal, output = Path(journal).resolve(), Path(output).resolve()
    if (type(expected_frames) is not int or expected_frames <= 0
            or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Positive full frame count and lowercase SHA256 required')
    policy = ObservationOutput(stream_id)
    if sha(journal) != expected_sha256:
        raise ValueError('Journal hash mismatch before export')
    code = {str(Path(p).resolve()): sha(p) for p in (__file__, implementation.__file__)}
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    identities = {'observation_alerts': set(), 'track_context': set()}
    digest = hashlib.sha256()
    count = 0
    with journal.open('rb') as source, (output/'channels.jsonl').open('x') as destination:
        for line in source:
            digest.update(line)
            row = decode(line)
            if (not isinstance(row, dict) or type(row.get('frame_index')) is not int
                    or row['frame_index'] != count or count >= expected_frames):
                raise ValueError('Full journal must be ordered, complete, and start at zero')
            channels = policy.update(row)
            # Direct baseline-subset oracle, independent of the projection's lists.
            baseline = {f"{stream_id}/{t['segment']}/{t['track_id']}": t
                        for t in row['tracks'] if t['qualified_moving']}
            alert_oracle = {k: t for k, t in baseline.items() if t['measured']}
            for channel, expected in [('track_context', baseline), ('observation_alerts', alert_oracle)]:
                actual = {t['identity']: t for t in channels[channel]}
                if len(actual) != len(channels[channel]) or set(actual) != set(expected):
                    raise AssertionError('Channel identity inventory differs from baseline subset')
                for key, track in expected.items():
                    point = track['measurement_source_xy'] if track['measured'] else track['source_xy']
                    if actual[key]['source_xy'] != point or actual[key]['measured'] != track['measured']:
                        raise AssertionError('Channel changed source coordinate or measurement status')
                identities[channel].update(actual)
                counts[channel+'_states'] += len(actual)
                counts[channel+'_frames'] += bool(actual)
            counts['retained_prediction_states'] += sum(not t['measured'] for t in baseline.values())
            counts['alerts_without_current_observation'] += sum(not t['measured'] for t in channels['observation_alerts'])
            counts['prediction_age_unknown_states'] += sum(not t['measured'] and t['last_measurement_age_ns'] is None
                                                         for t in channels['track_context'])
            destination.write(json.dumps(channels, allow_nan=False)+'\n')
            count += 1
    if count != expected_frames or digest.hexdigest() != expected_sha256 or sha(journal) != expected_sha256:
        raise ValueError('Incomplete or changed input; output is not a completed export')
    for path, expected in code.items():
        if sha(path) != expected:
            raise ValueError('Projection implementation changed during replay')
    summary = dict(schema='seaqr.visible-output-replay.v1', completed=True, frames=count,
                   stream_id=stream_id, input_journal=str(journal), input_sha256=expected_sha256,
                   channels_sha256=sha(output/'channels.jsonl'), code_sha256=code,
                   counts=dict(counts), unique_identity_counts={k:len(v) for k,v in identities.items()},
                   detector_or_tracker_rerun=False, qualification_changed=False,
                   all_qualified_measured_observations_preserved=True,
                   all_qualified_prediction_context_preserved=True,
                   predictions_in_observation_alerts=0,
                   interpretation='Observation output volume, not verified object or false-alarm counts; physical class unknown')
    with (output/'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--journal-sha256', required=True)
    parser.add_argument('--expected-frames', type=int, required=True)
    parser.add_argument('--stream-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.journal, args.journal_sha256, args.expected_frames, args.stream_id, args.output)
    print(json.dumps(dict(frames=result['frames'], counts=result['counts']), indent=2))


if __name__ == '__main__':
    main()
