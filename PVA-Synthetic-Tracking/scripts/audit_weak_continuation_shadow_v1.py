"""Independent artifact audit for the isolated weak-continuation shadow.

No producer, detector, tracker, media decoder or native library is imported.
Native captures are initially treated as hash-bound opaque arrays; their saved
metadata and the baseline/private-state/learning parity are audited separately.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


SCHEMA = 'seaqr.weak-continuation-shadow.audit.v1'
RUN_SCHEMA = 'seaqr.weak-continuation-shadow.run.v1'
SOURCES = {
    '0029': (687, '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359'),
    '0126': (674, 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'),
    '0055': (689, 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f'),
}
TIMING_PATHS = frozenset({('timings_ms',), ('motion', 'pva_timings_ms'),
    ('motion', 'warp_timings_ms'), ('motion', 'motion_fit', 'timing_ms'), ('coverage', 'detection_ms')})
NATIVE_STATE_KEYS = frozenset(('background', 'variance', 'support', 'eligible', 'stats',
                              'sigmas', 'peaks', 'counts', 'searchable', 'host_eligible'))
IDENTITY = re.compile(r'^\d+/(bright|dark):\d+$')
CONFIG_SHA = '7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f'
MOTION_SHA = 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1'
ARCHIVE_SHA = {'0029':'680ca10a85875bad4e5aa1636438d3808b765b4231cd6a7da890f31a632222be',
    '0126':'665411689d5a5a27dacf64c667ac2d72a43773dc4f560ecbef398d801afca826',
    '0055':'b151e8888d39c4de494a8944d66aa8ff6a08217ac323b3e6b05b329d54bfd95c'}
WINDOWS = {'0029':((70,81),(270,281),(344,355)),
    '0126':((16,28),(200,218),(258,270),(289,301),(324,336),(449,461),(478,490),(522,534)),
    '0055':((100,111),(300,311),(500,511))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def loads(text):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key: '+key)
            value[key] = item
        return value
    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), 'Nonfinite JSON number')
        return parsed
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exact(left, right, description):
    require(encoded(left) == encoded(right), 'Changed '+description)


def without_timing(value, path=()):
    if isinstance(value, dict):
        return {key: without_timing(item, path+(key,)) for key, item in value.items()
                if path+(key,) not in TIMING_PATHS}
    if isinstance(value, list):
        return [without_timing(item, path+('[]',)) for item in value]
    return value


class Evidence:
    def __init__(self, directory):
        self.root = Path(directory)
        require(self.root.is_absolute() and self.root.resolve() == self.root and
                self.root.is_dir() and not self.root.is_symlink(), 'Canonical evidence directory required')
        self.hashes = {}

    def path(self, relative, expected=None):
        part = Path(relative)
        require(type(relative) is str and relative and not part.is_absolute() and
                '..' not in part.parts and '.' not in part.parts and str(part) == relative,
                'Unsafe evidence path')
        target = self.root
        for name in part.parts:
            target /= name
            require(not target.is_symlink(), 'Symlink evidence refused')
        require(target.is_file(), 'Missing evidence: '+relative)
        actual = file_sha(target)
        require(expected is None or (digest_ok(expected) and actual == expected), 'Changed bound '+relative)
        require(relative not in self.hashes or self.hashes[relative] == actual, 'Evidence changed during audit: '+relative)
        self.hashes[relative] = actual
        return target

    def read(self, relative, expected=None):
        path = self.path(relative, expected)
        value = loads(path.read_text())
        self.path(relative, self.hashes[relative])
        require(type(value) is dict, 'JSON object required: '+relative)
        return value

    def unchanged(self):
        for relative, digest in list(self.hashes.items()):
            self.path(relative, digest)


def hash_map(value, description, keys=None):
    require(type(value) is dict and bool(value), 'Missing '+description)
    require(all(type(k) is str and digest_ok(v) for k, v in value.items()), 'Invalid '+description)
    require(keys is None or set(value) == keys, 'Changed '+description+' keys')


def validate_receipt(receipt, arm, clip, count, source_sha, freeze_sha, plan_sha):
    require(receipt.get('schema') == RUN_SCHEMA and receipt.get('passed') is True
            and receipt.get('error') is None, 'Failed/missing arm receipt')
    require(receipt.get('arm') == arm and receipt.get('clip') == clip, 'Wrong receipt arm/clip')
    require(type(receipt.get('processed_frames')) is int and receipt['processed_frames'] == count
            and type(receipt.get('expected_frames')) is int and receipt['expected_frames'] == count,
            'Incomplete arm receipt')
    require(receipt.get('production_changed') is False and receipt.get('weak_learning_enabled') is False,
            'Production/learning isolation is missing')
    for key, value in (('freeze_sha256', freeze_sha), ('plan_sha256', plan_sha), ('source_sha256', source_sha)):
        require(digest_ok(value) and receipt.get(key) == value, 'Wrong '+key)
    require(type(receipt.get('capture_count')) is int and receipt['capture_count'] >= 0, 'Invalid capture count')
    require((arm == 'clean' and receipt.get('trace_sha256') is None and receipt['capture_count'] == 0)
            or (arm == 'shadow' and digest_ok(receipt.get('trace_sha256'))), 'Wrong trace/capture arm binding')
    rows = receipt.get('baseline_digests')
    require(type(rows) is list and len(rows) == count, 'Incomplete baseline digest sequence')
    for i, row in enumerate(rows):
        require(type(row) is dict and set(row) == {'frame', 'output', 'state', 'learning'}
                and type(row['frame']) is int and row['frame'] == i, 'Invalid baseline digest row')
        require(all(digest_ok(row[k]) for k in ('output', 'state', 'learning')), 'Invalid private-state digest')
    runtime = receipt.get('runtime')
    require(type(runtime) is dict and set(runtime) == {'before', 'after'}, 'Runtime before/after required')
    for side in ('before', 'after'):
        require(type(runtime[side]) is dict and bool(runtime[side]), 'Empty runtime identity')
    exact({k:v for k,v in runtime['before'].items() if k != 'opencv_threads'},
          {k:v for k,v in runtime['after'].items() if k != 'opencv_threads'}, 'numerical runtime')
    return rows


def frame_row(row, index, description):
    require(type(row) is dict and type(row.get('frame_index')) is int and row['frame_index'] == index,
            'Noncontiguous '+description)
    require(type(row.get('timestamp_ns')) is int and row['timestamp_ns'] == index*100000000,
            'Changed nominal frame cadence')
    require(type(row.get('segment')) is int and row['segment'] >= 0, 'Invalid segment')


def validate_trace(row, baseline, index, evidence):
    frame_row(row, index, 'shadow trace')
    require(row['segment'] == baseline['segment'], 'Trace/baseline segment mismatch')
    require(type(row.get('capture_scheduled')) is bool, 'Capture schedule flag required')
    for key in ('prior_forecasts', 'strong_proposals', 'records', 'capture_files'):
        require(type(row.get(key)) is list, 'Missing trace '+key)
    require(type(row.get('metrics')) is dict, 'Missing shadow metrics')
    before, after = row.get('native_state_before'), row.get('native_state_after')
    if before is None or after is None:
        require(before is None and after is None and not row['capture_files'], 'Missing paired native state guard')
    else:
        hash_map(before, 'pre-capture native state', NATIVE_STATE_KEYS)
        hash_map(after, 'post-capture native state', NATIVE_STATE_KEYS)
        exact(before, after, 'mutable native detector state')
        require(row['capture_scheduled'], 'Native capture outside frozen schedule')
    seen = set()
    for f in row['prior_forecasts']:
        require(type(f) is dict and type(f.get('identity')) is str and IDENTITY.fullmatch(f['identity']), 'Invalid forecast identity')
        require(f['identity'] not in seen, 'Duplicate forecast identity')
        seen.add(f['identity'])
    captures = set()
    for capture in row['capture_files']:
        require(type(capture) is dict and set(capture) == {'identity','path','sha256','metadata_path','metadata_sha256'},
                'Invalid capture descriptor')
        identity = capture['identity']
        require(identity in seen and identity not in captures, 'Unexpected/duplicate captured forecast')
        captures.add(identity)
        for name, suffix, key in (('path','.npz','sha256'), ('metadata_path','.json','metadata_sha256')):
            relative = capture[name]
            require(type(relative) is str and relative.startswith('captures/') and Path(relative).suffix == suffix,
                    'Capture path outside bounded artifact subtree')
            evidence.path('shadow/'+relative, capture[key])
        meta = evidence.read('shadow/'+capture['metadata_path'], capture['metadata_sha256'])
        require(meta.get('frame') == index and type(meta.get('frame')) is int and
                meta.get('segment') == row['segment'] and meta.get('prelearning') is True,
                'Wrong pre-learning capture provenance')
        require(meta.get('read_only_capture') is True and meta.get('production_selection_unchanged') is True,
                'Missing passive-capture guarantees')
    require(row['capture_scheduled'] or not row['capture_files'], 'Stored captures outside schedule')
    return {capture['path'] for capture in row['capture_files']}


def validate_original_artifacts(evidence, arm, clip, count, source_sha):
    launch = evidence.read(arm+'/launch.json')
    require(launch.get('source') == '/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_'+clip+'.avi'
            and launch.get('source_sha256') == source_sha, 'Wrong source launch')
    require(launch.get('config_sha256') == CONFIG_SHA and launch.get('motion_config_sha256') == MOTION_SHA,
            'Wrong configuration launch')
    require(launch.get('expected_frames') == count and launch.get('max_frames') is None
            and launch.get('fps') == 10 and launch.get('annotations_supplied_to_detector') is False,
            'Incomplete/changed baseline launch')
    hash_map(launch.get('package_sha256'), 'production package hashes')
    hash_map(launch.get('code_sha256'), 'production code hashes')
    require(type(launch.get('configuration')) is dict, 'Missing original configuration')
    report = evidence.read(arm+'/report.json')
    require(report.get('completed') is True and report.get('full_clip') is True and
            report.get('frames') == count and report.get('source_sha256') == source_sha, 'Incomplete baseline report')
    exact(report.get('configuration'), launch['configuration'], 'report configuration')
    decode = report.get('frame_decode', {})
    require(decode.get('decoded_frames') == count and decode.get('consumed_frames') == count and
            decode.get('dropped_frames') == 0 and decode.get('worker_joined') is True and
            decode.get('capture_released') is True, 'Decode completeness/cleanup missing')
    v29 = evidence.read(arm+'.v29.json')
    require(v29.get('schema') == 'seaqr.visible-combined-v29.v1' and v29.get('passed') is True
            and v29.get('error') is None and v29.get('clip') == clip and v29.get('arm') == 'combined'
            and v29.get('frames') is None and v29.get('processed_frames') == count,
            'Original v29 replay failed')
    require(v29.get('execution_policy') == 'serial_reference' and v29.get('config_sha256') == CONFIG_SHA
            and v29.get('freeze_sha256') == '60b79d450672d131b517e9ed5a33fdeb40a6a2a9b29c6c0f584a9a8e93cc0dbc',
            'Wrong v29 execution identity')
    for key in ('raw16_accessed','defaults_changed','production_approved','new_accuracy_validated',
                'staged_v24_enabled','native_motion_v25_enabled'):
        require(v29.get(key) is False, 'Changed v29 scope '+key)
    comparison = v29.get('comparison', {})
    require(comparison.get('exact') is True and comparison.get('frames') == count and
            comparison.get('reference_journal_sha256') == ARCHIVE_SHA[clip], 'Missing unchanged archive comparison')
    evidence.path(arm+'/frames.jsonl', comparison.get('journal_sha256'))
    evidence.path(arm+'.execution.json', comparison.get('execution_sha256'))
    return launch


class ShadowStateAudit:
    """Independent covariance/kinematics and strong-only lifecycle checks."""
    def __init__(self, nms_radius):
        self.previous = {}
        self.used = {}
        self.radius = max(2., float(nms_radius))
        self.applied = self.forecasts = self.records = 0
        self.max_error = 0.
        self.status_counts = {}

    def close(self, left, right, description):
        a, b = np.asarray(left, float), np.asarray(right, float)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid '+description)
        err = float(np.max(np.abs(a-b)))
        self.max_error = max(err, self.max_error)
        require(err <= 1e-7, 'Incorrect '+description)

    @staticmethod
    def covariance(value, size):
        array = np.asarray(value, float)
        require(array.shape == (size,size) and np.isfinite(array).all() and np.array_equal(array,array.T),
                'Invalid symmetric covariance')
        require(np.linalg.eigvalsh(array).min() > 0, 'Nonpositive covariance')
        return array

    @staticmethod
    def predict(mean, p, dt):
        f = np.eye(4); f[0,2] = f[1,3] = dt
        load = np.vstack((np.eye(2)*(dt*dt/2),np.eye(2)*dt))
        prior = f@p@f.T+3600*(load@load.T)
        return f@mean, (prior+prior.T)*.5

    @staticmethod
    def correct(mean, p, point, variance):
        h = np.eye(4)[:2]; r = np.eye(2)*variance
        gain = p@h.T@np.linalg.inv(h@p@h.T+r)
        after = mean+gain@(np.asarray(point)-h@mean)
        residual = np.eye(4)-gain@h
        covariance = residual@p@residual.T+gain@r@gain.T
        return after, (covariance+covariance.T)*.5

    @staticmethod
    def gate(f, point):
        d = np.asarray(point)-np.asarray(f['reference_xy'])
        return float(d@d) <= 2025 and float(d@np.linalg.inv(np.asarray(f['innovation_covariance_2x2']))@d) <= 25

    def overlap(self, point, strong):
        for p in strong:
            if math.hypot(point[0]-p['x'],point[1]-p['y']) <= self.radius:
                return True
            if list(point) in p.get('shape',{}).get('support_reference_xy',[]):
                return True
        return False

    def check_observations(self, observation, prior, forecasts):
        require(observation.get('schema') == 'seaqr.weak-continuation-information.v1'
                and observation.get('focal_identity') == prior['identity'], 'Wrong weak observation owner')
        require(observation.get('position_gate_px') == 45 and observation.get('mahalanobis_squared_gate') == 25,
                'Changed inherited weak geometry')
        peaks = observation.get('observed_peaks')
        require(type(peaks) is list and observation.get('observed_peak_count') == len(peaks), 'Incorrect peak count')
        require(type(observation.get('coverage_known')) is bool and
                observation['coverage_known'] == (observation.get('coverage_unknown_reasons') == []), 'Invalid coverage declaration')
        weak = sum(p.get('evidence_partition') == 'weak_temporal' for p in peaks)
        strong = sum(p.get('evidence_partition') == 'original_threshold_pass' for p in peaks)
        require(weak+strong == len(peaks) and observation.get('weak_peak_count') == weak and
                observation.get('original_threshold_peak_count') == strong, 'Incorrect weak/strong partition')
        for i, peak in enumerate(peaks):
            require(peak.get('descriptive_rank') == i+1 and peak.get('polarity') == prior['polarity']
                    and self.gate(prior, peak['reference_xy']), 'Peak outside original owner geometry')
            require(peak.get('identity_assignment') is None and peak.get('acceptance_decision') is None,
                    'Enumerator fabricated identity')
            require((peak['signed_centered_temporal_dn'] >= peak['temporal_threshold_dn']) ==
                    peak['original_temporal_threshold_pass'] and peak['signed_centered_temporal_dn'] > 0
                    and peak['score'] > 0,
                    'Peak score/partition inconsistent')
            expected = sorted(f['identity'] for f in forecasts.values() if f['identity'] != prior['identity']
                              and f['polarity'] == prior['polarity'] and self.gate(f,peak['reference_xy']))
            require(expected == [p['identity'] for p in peak['competing_prior_identity_gates']], 'Competing owner gates omitted/changed')
        return peaks

    def step(self, row, baseline):
        ts, segment = row['timestamp_ns'], row['segment']
        forecasts = {f['identity']:f for f in row['prior_forecasts']}
        expected = {key:old for key,old in self.previous.items() if old['record']['segment']==segment and
                    0 < (ts-old['timestamp'])/1e9 <= 1.0}
        require(set(forecasts) == set(expected), 'Missing/noncausal prior forecast')
        predictions = {}
        for key, old in expected.items():
            f = forecasts[key]; record = old['record']
            mean, p = self.predict(old['mean'],old['covariance'],(ts-old['timestamp'])/1e9)
            predictions[key] = (mean,p)
            self.close(f['predicted_mean'],mean,'past-only forecast mean')
            self.close(f['predicted_covariance'],p,'past-only forecast covariance')
            self.covariance(f['predicted_covariance'],4)
            self.close(f['reference_xy'],mean[:2],'forecast center')
            self.close(f['innovation_covariance_2x2'],p[:2,:2]+np.eye(2)*4,'strong innovation covariance')
            self.close(f['strong_measurement_covariance'],np.eye(2)*4,'strong measurement covariance')
            require(f.get('frame_index') == row['frame_index'] and f.get('timestamp_ns') == ts
                    and f.get('segment') == segment, 'Forecast time/segment differs')
            anchor = old['last_strong']; unused = self.used.get(key) != anchor
            age = (ts-anchor)/1e9
            require(f.get('prior_last_strong_timestamp_ns') == anchor and f.get('strong_age_seconds') == age
                    and f.get('weak_budget_available') is unused, 'Strong anchor/budget changed')
            require(f.get('prior_qualified') is record['qualified_moving'] and
                    f.get('prior_confirmation_timestamp_ns') == record['confirmation_timestamp_ns'] and
                    f.get('prior_associated_update_count') == record['hits'] and
                    f.get('prior_independent_confirmation_hits') == record['independent_hits'], 'Strong qualification evidence changed')
            eligible = record['qualified_moving'] and record['confirmation_timestamp_ns'] is not None and unused and 0 < age <= .7
            require(f.get('query_eligible') is eligible, 'Changed past-only query eligibility')
            self.forecasts += 1
        # Journal serialization adds source_xy to the original proposals later.
        exact([{k:v for k,v in p.items() if k!='source_xy'} for p in row['strong_proposals']],
              [{k:v for k,v in p.items() if k!='source_xy'} for p in baseline['candidates']], 'original strong proposal input')
        wc = row['metrics'].get('weak_continuation', {})
        require(wc.get('schema') == 'seaqr.weak-continuation-shadow.v1' and wc.get('frame_index') == row['frame_index'], 'Missing weak step metrics')
        for key,value in (('ordinary_measured_is_strong_only',True),('detector_learning_feedback',False),
                          ('strong_associations_recomputed_by_shadow',True),('frozen_detector_stream',True)):
            require(wc.get(key) is value, 'Shadow isolation missing '+key)
        require(wc.get('prior_track_count') == len(forecasts) and wc.get('prior_query_eligible_count') ==
                sum(f['query_eligible'] for f in forecasts.values()), 'Wrong prior metrics counts')
        require(len(row['records']) <= 512, 'Shadow active-track capacity exceeded')
        decisions, updated, applied = [], {}, 0
        captures = {c['identity'] for c in row['capture_files']}
        queried = set()
        matrix = np.asarray(baseline['source_to_reference'],float)
        for record in row['records']:
            key = str(record['segment'])+'/'+record['track_id']
            require(record['segment'] == segment and key not in updated, 'Invalid/duplicate shadow output identity')
            note = record.get('weak_evidence', {}); decisions.append(note)
            require(note.get('identity') == key and type(note.get('applied')) is bool and
                    note.get('is_ordinary_measurement') is False and note.get('physical_identity_verified') is False,
                    'Missing weak evidence classification')
            status = note.get('status')
            require(status != 'invalid_capture_or_weak_covariance', 'Invalid capture/covariance is a scientific integrity failure')
            self.status_counts[status] = self.status_counts.get(status,0)+1
            old, prior = self.previous.get(key), forecasts.get(key)
            mean = np.asarray(record['reference_xy']+record['velocity_reference_xy_px_s'],float)
            if record['measured']:
                require(status=='strong_measurement_priority' and not note['applied'], 'Weak replaced a strong measurement')
                require(record['measurement_source_xy'] is not None and record['measurement_score'] is not None, 'Missing actual strong measurement')
                self.used.pop(key,None)
                anchor = ts
                if prior is None:
                    require(record['hits']==1 and record['independent_hits']==1, 'Shadow weak birth or incomplete history')
                    self.close(mean[2:],[0,0],'strong birth velocity')
                    p = np.diag([4.,4.,22500.,22500.])
                else:
                    point = matrix@np.array(record['measurement_source_xy']+[1.])
                    predicted,p = predictions[key]
                    calculated,p = self.correct(predicted,p,point[:2]/point[2],4.)
                    self.close(mean,calculated,'strong posterior state')
            else:
                require(prior is not None, 'Unmeasured output without causal prior')
                anchor = prior['prior_last_strong_timestamp_ns']; predicted,p = predictions[key]
                require(record['measurement_source_xy'] is None and record['measurement_score'] is None
                        and record.get('learning_shape_reference_xy') is None, 'Weak entered strong measurement/learning fields')
                for field, value in (('hits',prior['prior_associated_update_count']),
                    ('independent_hits',prior['prior_independent_confirmation_hits']),
                    ('confirmation_timestamp_ns',prior['prior_confirmation_timestamp_ns'])):
                    require(record[field] == value, 'Weak changed strong-only '+field)
                for field in ('excursion_px','motion_quality','qualified_moving'):
                    exact(record[field],old['record'][field],'unmeasured strong-only '+field)
                wanted_lifecycle = 'coasted' if prior['prior_confirmation_timestamp_ns'] is not None else 'tentative'
                require(record['lifecycle'] == wanted_lifecycle, 'Weak changed coast/tentative lifecycle')
                if not prior['prior_qualified'] or prior['prior_confirmation_timestamp_ns'] is None:
                    wanted='not_prior_strong_qualified'
                elif (ts-anchor)/1e9 > .7:
                    wanted='strong_age_expired'
                elif not prior['weak_budget_available']:
                    wanted='weak_budget_used_for_strong_gap'
                elif 'observations' not in note:
                    wanted='missing_capture'
                    require(key not in captures, 'Stored capture was silently discarded')
                else:
                    require(row['capture_scheduled'] and key in captures, 'Observation without scheduled bound capture')
                    queried.add(key)
                    observation=note['observations'];peaks=self.check_observations(observation,prior,forecasts)
                    if not observation['coverage_known']:wanted='capture_coverage_unknown'
                    elif observation['original_threshold_peak_count']:wanted='original_threshold_peak_in_gate'
                    elif len(peaks)!=1:wanted='no_unique_weak_peak'
                    elif peaks[0]['competing_prior_identity_gates']:wanted='competing_prior_identity_gate'
                    elif self.overlap(peaks[0]['reference_xy'],row['strong_proposals']):wanted='overlaps_current_strong_evidence'
                    else:wanted='weak_kinematic_correction'
                require(status == wanted and note['applied'] is (wanted=='weak_kinematic_correction'), 'Weak decision disagrees with frozen rule')
                if note['applied']:
                    peak = note['observations']['observed_peaks'][0]
                    require(note['measurement_reference_xy']==peak['reference_xy'] and note['score']==peak['score']
                            and note['strong_anchor_timestamp_ns']==anchor, 'Weak correction changed observation/anchor')
                    self.close(note['mean_before'],predicted,'weak prior mean')
                    self.close(note['covariance_before'],p,'weak prior covariance')
                    calculated,p = self.correct(predicted,p,peak['reference_xy'],8.)
                    self.close(note['mean_after'],calculated,'weak Joseph mean')
                    self.close(note['covariance_after'],p,'weak Joseph covariance')
                    self.close(mean,calculated,'weak output kinematics')
                    self.used[key]=anchor;applied+=1
                else:
                    self.close(mean,predicted,'coast kinematics')
            updated[key]=dict(mean=mean,covariance=p,last_strong=anchor,timestamp=ts,record=record)
            self.records+=1
        exact(decisions,wc.get('decisions'),'per-record weak decisions')
        require(wc.get('applied_count')==applied and captures==queried, 'Weak applied/capture accounting differs')
        self.applied+=applied
        self.previous=updated;self.used={k:v for k,v in self.used.items() if k in updated}


def audit_run(directory, clip, freeze_sha256, plan_sha256):
    require(clip in SOURCES, 'Clip outside frozen development cohort')
    require(digest_ok(freeze_sha256) and digest_ok(plan_sha256), 'Caller-bound freeze and plan SHA required')
    count, source_sha = SOURCES[clip]
    evidence = Evidence(directory)
    receipts = {arm:evidence.read(arm+'.shadow.json') for arm in ('clean','shadow')}
    private = {arm:validate_receipt(receipts[arm], arm, clip, count, source_sha,
                                   freeze_sha256, plan_sha256) for arm in receipts}
    launches = {arm:validate_original_artifacts(evidence,arm,clip,count,source_sha) for arm in receipts}
    exact(launches['clean'],launches['shadow'],'baseline launches')
    exact(private['clean'], private['shadow'], 'baseline output/state/learning digests')
    exact(receipts['clean']['runtime'], receipts['shadow']['runtime'], 'between-arm numerical runtime')
    paths = [evidence.path(arm+'/frames.jsonl') for arm in ('clean','shadow')]
    trace = evidence.path('shadow/shadow_trace.jsonl', receipts['shadow']['trace_sha256'])
    captures = set()
    state_audit = ShadowStateAudit(launches['clean']['configuration']['tracking_peak_nms_radius_px'])
    with paths[0].open() as clean, paths[1].open() as shadow, trace.open() as tracing:
        actual = 0
        for i, lines in enumerate(itertools.zip_longest(clean, shadow, tracing)):
            require(all(type(line) is str and line.strip() for line in lines), 'Missing/blank/unequal artifact row')
            left, right, diagnostic = map(loads, lines)
            frame_row(left, i, 'clean journal'); frame_row(right, i, 'shadow baseline journal')
            exact(without_timing(left), without_timing(right), 'non-timing baseline journal frame '+str(i))
            require(diagnostic.get('capture_scheduled') is any(lo<=i<=hi for lo,hi in WINDOWS[clip]), 'Frozen capture window differs')
            captures.update(validate_trace(diagnostic, left, i, evidence))
            state_audit.step(diagnostic,left)
            actual += 1
        require(actual == count, 'Incomplete/full-clip journal extent')
    require(len(captures) == receipts['shadow']['capture_count'], 'Capture inventory/count mismatch')
    evidence.unchanged()
    return dict(schema=SCHEMA, passed=True, clip=clip, frames=count, source_sha256=source_sha,
        freeze_sha256=freeze_sha256, plan_sha256=plan_sha256, files_sha256=evidence.hashes,
        baseline_journal_non_timing_exact=True, baseline_output_state_learning_digests_exact=True,
        native_state_guards_unchanged=True, capture_files_verified=len(captures),
        shadow_forecasts_verified=state_audit.forecasts,shadow_records_verified=state_audit.records,
        weak_kinematic_corrections=state_audit.applied,weak_status_counts=state_audit.status_counts,
        maximum_state_arithmetic_absolute_error=state_audit.max_error,
        state_arithmetic_audit_tolerance=1e-7,association_membership_tolerance=0,
        timing_exclusions=[list(p) for p in sorted(TIMING_PATHS)], production_changed=False,
        weak_learning_enabled=False, native_arrays_interpreted=False, source_media_accessed=False,
        producer_imported=False, limitations=['Baseline parity and passive provenance do not establish object identity or accuracy.',
            'Native capture arrays are hash-bound but not numerically interpreted by this artifact audit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--clip', choices=tuple(SOURCES), required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'Fresh audit output required')
    result = audit_run(args.directory, args.clip, args.freeze_sha256, args.plan_sha256)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
