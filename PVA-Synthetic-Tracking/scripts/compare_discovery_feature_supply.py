"""Compare full-clip candidate availability and a frozen baseline-derived pass.

No classifier, independent ground truth, threshold search, or promotion decision.
The output-state totals measure review workload, not false-positive counts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def validate_receipt(receipt, arm, clip, source_hash):
    require(receipt['clip'] == clip and receipt['source']['sha256'] == source_hash,
            'receipt source identity differs')
    if arm == 'baseline':
        require(receipt['schema'] == 'seaqr.discovery-pair.baseline.v1'
                and receipt['algorithm_changed'] is False, 'wrong baseline arm')
    else:
        require(receipt['schema'] == 'seaqr.discovery-feature-supply.v1'
                and receipt['algorithm_changed'] is True and receipt['feature_algorithm_changed'] is True,
                'wrong candidate arm')
        require(receipt['candidate'] == dict(harris_gain=16, harris_capacity_policy='complete_grid', feature_image_scale=.5),
                'candidate declaration differs')
        for key in ('detector_configuration_changed', 'tracker_configuration_changed',
                    'global_motion_gates_changed', 'production_promotion', 'annotations_supplied_to_detector',
                    'raw16_accessed', 'sealed_holdouts_accessed'):
            require(receipt[key] is False, 'candidate scope changed: '+key)
        audit = receipt['feature_adapter']
        original, effective = audit['original_motion_configuration'], audit['effective_motion_configuration']
        require(original['harris_capacity_policy'] == 'legacy_default' and effective['harris_capacity_policy'] == 'complete_grid',
                'wrong capacity override')
        require(dict(original, harris_capacity_policy='complete_grid') == effective
                and effective['feature_image_scale'] == .5, 'unexpected motion configuration change')


def retention(rows, references, radius=8.0, polarity='dark'):
    require(math.isfinite(radius) and radius > 0, 'invalid matching radius')
    reference = {r['frame_index']: r['measurement_source_xy'] for r in references}
    require(reference and len(reference) == len(references), 'empty/duplicate reference frames')
    require(all(len(xy) == 2 and all(math.isfinite(x) for x in xy) for xy in reference.values()),
            'nonfinite reference coordinates')
    identities = defaultdict(list)
    seen, details = set(), []
    for row in rows:
        frame = row['frame_index']
        if frame not in reference:
            continue
        require(frame not in seen, 'duplicate evaluation frame')
        seen.add(frame)
        point = reference[frame]
        matches = []
        if not row['detection_ready']:
            details.append(dict(frame_index=frame, detection_ready=False, matched_identities=[]))
            continue
        row_ids = set()
        for track in row['tracks']:
            identity = f"{row['segment']}/{track['track_id']}"
            require(identity not in row_ids, 'duplicate track identity in frame')
            row_ids.add(identity)
            if (not track['qualified_moving'] or not track['measured']
                    or track['track_id'].split(':')[0] != polarity):
                continue
            xy = track['measurement_source_xy']
            require(xy is not None and len(xy) == 2 and all(math.isfinite(x) for x in xy),
                    'qualified actual measurement lacks finite source coordinates')
            distance = math.hypot(xy[0] - point[0], xy[1] - point[1])
            if distance <= radius:
                identities[identity].append(frame)
                matches.append(dict(identity=identity, distance_native_px=distance,
                                    measurement_source_xy=xy))
        details.append(dict(frame_index=frame, detection_ready=True, matched_identities=matches))
    require(seen == set(reference), 'missing reference frames')
    coherent = sorted(identity for identity, frames in identities.items() if len(frames) == len(reference))
    return dict(baseline_derived_not_independent_truth=True, radius_native_px=radius,
                reference_frames=len(reference), any_identity_matched_frames=sum(bool(d['matched_identities']) for d in details),
                ambiguous_frames=sum(len(d['matched_identities']) > 1 for d in details),
                best_coherent_identity_frames=max(map(len, identities.values()), default=0),
                complete_coherent_identities=coherent, preservation_guard_passed=bool(coherent),
                matched_frames_by_identity=dict(sorted(identities.items())), details=details)


def collect(path, first=0, last=672):
    counts, reasons = Counter(), Counter()
    ids, needed_rows = set(), []
    longest = current = 0
    with Path(path).open() as stream:
        total = 0
        for index, line in enumerate(stream):
            row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            require(row['frame_index'] == index and row['timestamp_ns'] == index * 100000000,
                    'noncontiguous index or changed nominal timeline')
            coverage = row['coverage']
            require(coverage['full_shape_hw'] == [3190, 4784] and coverage['native_pixel_sampling']
                    and coverage['configured_crop'] is None, 'non-native or cropped detector input')
            total += 1
            ready = not coverage['warmup'] and coverage['searchable_pixels'] > 0
            require(coverage['detection_ready'] == ready, 'readiness field disagrees with inputs')
            if 430 <= index <= 464:
                needed_rows.append(dict(frame_index=index, segment=row['segment'], tracks=row['tracks'], detection_ready=ready))
            if not first <= index <= last:
                continue
            counts['frames'] += 1
            counts['ready_frames'] += ready
            counts['warmup_frames'] += coverage['warmup']
            counts['motion_resets'] += row['motion']['reset']
            counts['pva_runtime_errors'] += row['motion'].get('pva_failure', False)
            counts['candidate_records'] += len(row['candidates'])
            counts['dropped_at_tile_cap'] += coverage['dropped_at_tile_cap']
            counts['dropped_at_frame_cap'] += coverage['dropped_at_frame_cap']
            current = 0 if ready else current + 1
            longest = max(longest, current)
            reasons.update(row['motion'].get('rejection_reasons', []))
            for track in row['tracks']:
                if track['qualified_moving']:
                    counts['qualified_measured_states' if track['measured'] else 'qualified_predicted_states'] += 1
                    ids.add((row['segment'], track['track_id']))
            active = 0
            for polarity in ('bright', 'dark'):
                metrics = row['tracking_metrics'].get(polarity, {})
                active += metrics.get('active_track_count', 0)
                counts['dropped_track_births'] += metrics.get('dropped_birth_count_at_active_track_cap', 0)
            counts['maximum_active_tracks'] = max(counts['maximum_active_tracks'], active)
    require(total == 673, 'incomplete or extra frames')
    for key in ('qualified_measured_states', 'qualified_predicted_states', 'pva_runtime_errors'):
        counts[key] += 0
    ready = counts['ready_frames']
    return dict(counts=dict(counts), ready_fraction=ready/counts['frames'],
                qualified_identities=len(ids), longest_unavailable_streak=longest,
                motion_rejection_reasons=dict(reasons),
                measured_states_per_ready_frame=counts['qualified_measured_states']/ready if ready else None,
                counts_are_object_or_false_positive_counts=False), needed_rows


def run(baseline_root, candidate_root, regression, output):
    baseline_root, candidate_root, regression, output = map(Path, (baseline_root, candidate_root, regression, output))
    require(not output.exists(), 'fresh comparison output required')
    intake = read(regression)
    positive = intake['positive_pass']
    require(positive['clip_id'] == '0240' and positive['first_frame'] == 430
            and positive['last_frame_inclusive'] == 464, 'wrong positive interval')
    require(positive['retention_contract']['radius_native_px'] == 8, 'changed frozen matching radius')
    source_hashes = {'0170':'12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc',
                     '0240':'2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585'}
    bound, results = {}, {}
    recorded = {item['clip_id']: item for item in intake['recordings']}
    for clip in ('0170', '0240'):
        results[clip] = {}
        for arm, root in (('baseline', baseline_root), ('candidate', candidate_root)):
            base = root/clip
            receipt, report = read(base/'execution_receipt.json'), read(base/'run/report.json')
            validate_receipt(receipt, arm, clip, source_hashes[clip])
            bound[str(base/'execution_receipt.json')] = sha(base/'execution_receipt.json')
            if arm == 'baseline':
                require(bound[str(base/'execution_receipt.json')] == recorded[clip]['artifacts']['execution_receipt']['sha256'],
                        'baseline receipt differs from frozen regression intake')
            require(sha(base/'preflight.json') == receipt['preflight_sha256'], 'preflight differs from inference receipt')
            bound[str(base/'preflight.json')] = receipt['preflight_sha256']
            require(receipt['passed'] is True and report['completed'] is True and report['full_clip'] is True
                    and report['frames'] == 673 and report['source_sha256'] == source_hashes[clip],
                    'incomplete, failed, or wrong-source inference')
            for key, name in (('journal_sha256','frames.jsonl'), ('report_sha256','report.json'), ('launch_sha256','launch.json')):
                file = base/'run'/name
                require(sha(file) == receipt[key], 'run input differs from inference receipt')
                if arm == 'baseline':
                    require(receipt[key] == recorded[clip]['artifacts'][key.removesuffix('_sha256')]['sha256'],
                            'baseline artifacts differ from frozen regression intake')
                bound[str(file)] = receipt[key]
            full, selected = collect(base/'run/frames.jsonl')
            require(full['counts']['ready_frames'] == report['availability']['counts']['detection_ready_frames'],
                    'independent ready count differs')
            full.update(processed_fps=report['processed_fps'], elapsed_seconds=report['elapsed_seconds'],
                        stage_timings_ms=report['timings_ms'],
                        timing_is_uncontrolled_cross_run_diagnostic=True)
            if clip == '0240':
                full['positive_pass'] = retention(selected, positive['baseline_measurements'])
                full['burst_50_105'], _ = collect(base/'run/frames.jsonl', 50, 105)
            results[clip][arm] = full
    bound[str(regression)] = sha(regression)
    bound[str(Path(__file__).resolve())] = sha(Path(__file__))
    assessment = dict(
        both_candidate_clips_at_least_95pct_ready=all(results[c]['candidate']['ready_fraction'] >= .95 for c in results),
        no_new_pva_runtime_errors=all(results[c]['candidate']['counts']['pva_runtime_errors'] == 0 for c in results),
        candidate_preserves_coherent_positive=results['0240']['candidate']['positive_pass']['preservation_guard_passed'],
        promotion_allowed=False,
        remaining_requirements=['Earlier four-clip reference regression validation',
            'Independent source annotations; baseline-derived matches are not recall',
            'Label-specific clutter assessment and controlled runtime comparison'])
    result = dict(schema='seaqr.feature-supply.comparison.v1', input_sha256=bound, clips=results,
                  assessment=assessment, physical_airborne_class_from_user_review_not_algorithm=True,
                  nuisance_scopes_are_verified_negative=False, production_changed=False)
    require(all(sha(p) == digest for p,digest in bound.items()), 'comparison inputs changed')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'candidate-root', 'regression', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result['assessment']))
