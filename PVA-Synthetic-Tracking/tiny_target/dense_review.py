"""Separate bounded retained hypotheses from a smaller human-review preview.

This module reads reports, never video. It does not rerank tracks, expand
detector budgets, or label unlabeled hypotheses as airborne objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _ids(tracks: Sequence[Mapping[str, Any]]) -> list[int]:
    ids = [track['track_id'] for track in tracks]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError('Track IDs must be integers')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate track IDs make the result set ambiguous')
    return ids


def describe_track_outputs(
    retained: Sequence[Mapping[str, Any]], preview: Sequence[Mapping[str, Any]],
    *, capacity: int, preview_limit: int,
) -> dict[str, Any]:
    for value in (capacity, preview_limit):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Result budgets must be positive integers')
    retained_ids, preview_ids = _ids(retained), _ids(preview)
    if len(retained) > capacity or len(preview) > preview_limit:
        raise ValueError('Reported result set exceeds its configured budget')
    index = dict(zip(retained_ids, retained))
    if any(track_id not in index for track_id in preview_ids):
        raise ValueError('Preview contains a track missing from the retained pool')
    if any(track != index[track['track_id']] for track in preview):
        raise ValueError('Preview track differs from its retained counterpart')
    return {
        'retained_tracks': {'field': 'track_pool', 'count': len(retained), 'capacity': capacity,
                            'complete_for_retained_pool': True, 'unbounded_all_tracks': False},
        'review_preview': {'field': 'shortlist', 'count': len(preview), 'limit': preview_limit,
                           'omitted_retained_count': len(retained) - len(preview),
                           'is_complete_retained_set': set(retained_ids) == set(preview_ids),
                           'purpose': 'human_review_only'},
        'warning': 'Retained tracks are bounded, unlabeled hypotheses. The preview can omit retained tracks; neither set confirms airborne objects.',
    }


def analyze_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get('schema_version') != 'seaqr.tiny-target.dense-screen.v1':
        raise ValueError('Expected a dense-screen report')
    synthetic = report['screening']['synthetic_tracking']
    if synthetic is None:
        raise ValueError('This report has no synthetic-tracking output')
    retained, preview = synthetic['track_pool'], synthetic['shortlist']
    if (_count(synthetic['qualified_track_pool_count'], 'Retained count') != len(retained)
            or _count(synthetic['shortlist_count'], 'Preview count') != len(preview)):
        raise ValueError('Reported track counts disagree with the saved arrays')
    config = report['configuration']['effective']
    contract = describe_track_outputs(retained, preview,
        capacity=config['synthetic_retained_track_pool_size'],
        preview_limit=config['max_shortlist_tracks_per_clip'])
    availability = report['screening'].get('availability')
    observed_windows = _count(synthetic['window_count'], 'Window count')
    valid_windows = (availability.get('synthetic_windows_with_valid_ranking')
                     if availability is not None else None)
    if valid_windows is not None:
        _count(valid_windows, 'Valid-ranking window count')
        if valid_windows > observed_windows or (valid_windows == 0 and retained):
            raise ValueError('Availability contradicts the recorded output')
    available = valid_windows > 0 if valid_windows is not None else None
    return dict(output_contract=contract, retained_tracks=retained, review_preview=preview,
        observed_window_count=observed_windows, detection_available=available,
        availability=availability,
        interpretation='unlabeled_review_workload' if available else
                       'detection_unavailable' if available is False else 'availability_not_recorded')


def _md(value: Any) -> str:
    return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;')


def export_review(report_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    path = Path(report_path).resolve()
    raw = path.read_bytes()
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    report = json.loads(raw, parse_constant=invalid_constant)
    result = analyze_report(report)
    identity = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    output = Path(output_directory).resolve()
    contract = result['output_contract']
    preview_ids = set(_ids(result['review_preview']))
    control_matches = ((report.get('injection') or {}).get('synthetic_track_pool_evaluation') or {}).get('matched_targets', [])
    controls = {item['track_id']: item['target_id'] for item in control_matches}
    manifest = dict(schema_version='seaqr.dense-review-bundle.v1', source_report=identity,
        output_contract=contract, detection_available=result['detection_available'],
        interpretation=result['interpretation'], observed_window_count=result['observed_window_count'],
        files={'retained_tracks': 'retained_tracks.json', 'review_preview': 'review_preview.json', 'index': 'results.md'})
    lines = ['# Retained results and review preview', '', contract['warning'], '',
        f"Retained tracks: **{len(result['retained_tracks'])}** (bounded pool capacity {contract['retained_tracks']['capacity']}).",
        f"Preview: **{len(result['review_preview'])}**; **{contract['review_preview']['omitted_retained_count']} retained tracks are not shown in that preview**.", '',
        f"Search availability: **{_md(result['interpretation'])}**; emitted windows: {result['observed_window_count']}.",
        'Zero tracks from an unavailable search are not evidence of an empty scene.', '',
        f"[All retained track data]({output / 'retained_tracks.json'}) · [Preview data only]({output / 'review_preview.json'})", '',
        '## All retained tracks', '',
        'Original pool order is preserved. Synthetic-control labels, when present, refer only to injected test signals.', '',
        '| Pool rank | Track ID | Reference frames | Window hits | In preview? | Injected control |',
        '|---:|---:|---|---:|---|---|']
    for rank, track in enumerate(result['retained_tracks'], 1):
        lines.append(f"| {rank} | {track['track_id']} | {_md(track['first_reference_frame_index'])}–{_md(track['last_reference_frame_index'])} | {_md(track['hit_count'])} | {'Yes' if track['track_id'] in preview_ids else 'No'} | {_md(controls.get(track['track_id'], '—'))} |")
    if not result['retained_tracks']:
        lines.append('| — | — | — | — | — | No retained tracks |')
    # Validate/serialize everything before creating an exclusive new bundle.
    # Malformed input must not leave a partially successful-looking review.
    documents = {'manifest.json': json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)}
    for name, tracks in (('retained_tracks', result['retained_tracks']), ('review_preview', result['review_preview'])):
        documents[f'{name}.json'] = json.dumps(dict(source_report=identity, output_role=name,
            output_contract=contract, interpretation=result['interpretation'], tracks=tracks),
            indent=2, sort_keys=True, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in documents.items():
        with (output / filename).open('x', encoding='utf-8') as handle:
            handle.write(content + '\n')
    with (output / 'results.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_review(args.report, args.output_directory), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
