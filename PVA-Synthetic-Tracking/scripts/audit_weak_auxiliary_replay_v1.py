"""Independent metadata/array audit of the isolated auxiliary weak replay.

No tracker, producer, detector, native library, or source-media decoder imports.
Archived capture coverage is conditional on the old shadow's query schedule.
"""
from __future__ import annotations

import hashlib
import gzip
import itertools
import importlib.util
import json
import math
from pathlib import Path
import re
import zipfile
from copy import deepcopy

import numpy as np

SCHEMA = 'seaqr.weak-auxiliary-replay.audit.v1'
RUN_SCHEMA = 'seaqr.weak-auxiliary-replay.run.v1'
FREEZE_SCHEMA = 'seaqr.weak-auxiliary-replay.freeze.v1'
ATOL = 1e-7
ORIGINAL_ROOT = Path('/tmp/seaqr_weak_shadow_v1_LveJSx')
ORIGINAL_FREEZE = 'e93849b1e29b8ef6a18fe23688ece8f8c959b72191081a1abb2f59af11eca8b2'
ORIGINAL_PLAN = '3156b6f47383cbdf68375728cd3eed5e5fdd13dfec4d2578f538eb2df6e91395'
ENUM_SHA = '7fab347bc47d51df96c73cae3efb6a89266cde77938d3c0b5ddbb23a8f26ff15'
WINDOWS = {'0029':((70,81),(270,281),(344,355)),
    '0126':((16,28),(200,218),(258,270),(289,301),(324,336),(449,461),(478,490),(522,534)),
    '0055':((100,111),(300,311),(500,511))}
SOURCES = {
    '0029': (687, '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359'),
    '0126': (674, 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'),
    '0055': (689, 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def loads(text):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, 'Duplicate JSON key')
            output[key] = value
        return output
    def number(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON number')
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def exact(left, right, name):
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(encode(left) == encode(right), 'Changed '+name)


def normalized_sha(value):
    """Digest the already tagged old normalized() payload, without double-tagging."""
    return hashlib.sha256(json.dumps(value, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def decoded_normalized(value):
    """Interpret the pinned old digest format without importing any runtime class."""
    if value is None or type(value) in (int, str, bool):
        return value
    require(type(value) is list and value and type(value[0]) is str, 'Invalid normalized state')
    tag = value[0]
    if tag == 'float':
        require(len(value) == 2 and type(value[1]) is str, 'Invalid tagged float')
        result = float.fromhex(value[1])
        require(math.isfinite(result),'Nonfinite tagged float outside empty running statistics')
        return result
    if tag in ('list', 'tuple', 'set'):
        require(len(value) == 2 and type(value[1]) is list, 'Invalid tagged sequence')
        return [decoded_normalized(item) for item in value[1]]
    if tag == 'deque':
        require(len(value) == 3 and (value[1] is None or type(value[1]) is int), 'Invalid tagged deque')
        return [decoded_normalized(item) for item in value[2]]
    if tag in ('dict', 'dataclass'):
        require(len(value) == (2 if tag == 'dict' else 3), 'Invalid tagged mapping')
        pairs = value[-1]
        require(type(pairs) is list, 'Invalid tagged mapping entries')
        if tag=='dataclass' and value[1]=='_RunningMoments':
            require(all(type(p) in (list,tuple) and len(p)==2 and type(p[0]) is str for p in pairs)
                and len({p[0] for p in pairs})==len(pairs),'Invalid running-statistic fields')
            fields=dict(pairs)
            if fields.get('count')==0:
                require(type(fields['count']) is int and set(fields)=={'count','total','total_squared','minimum','maximum'}
                    and decoded_normalized(fields['total'])==0 and decoded_normalized(fields['total_squared'])==0
                    and fields['minimum']==['float','inf'] and fields['maximum']==['float','-inf'],
                    'Malformed empty running-statistic sentinels')
                return dict(count=0,total=0.,total_squared=0.,
                    minimum={'opaque_empty_running_minimum':'inf'},maximum={'opaque_empty_running_maximum':'-inf'})
        output = {}
        for pair in pairs:
            require(type(pair) in (list, tuple) and len(pair) == 2, 'Invalid tagged mapping pair')
            key = decoded_normalized(pair[0])
            require(type(key) in (str, int, bool) and key not in output, 'Duplicate/invalid tagged key')
            output[key] = decoded_normalized(pair[1])
        return output
    if tag == 'array':
        require(len(value) == 4 and type(value[1]) is str and type(value[2]) is list
            and type(value[3]) is str, 'Invalid tagged array')
        dtype = np.dtype(value[1])
        require(dtype.kind in 'biuf' and not dtype.hasobject and len(value[2]) <= 3
            and all(type(n) is int and 0 <= n <= 1000000 for n in value[2]), 'Unsafe tagged array')
        count = math.prod(value[2])
        require(count <= 4000000 and len(value[3]) == count*dtype.itemsize*2, 'Tagged array byte length differs')
        raw = bytes.fromhex(value[3])
        array = np.frombuffer(raw, dtype=dtype).reshape(value[2])
        require(np.isfinite(array).all(), 'Nonfinite tagged state array')
        return array.tolist()
    raise ValueError('Unknown normalized state tag: '+tag)


def vector(value, size, name):
    require(type(value) is list and len(value) == size and
        all(type(v) in (int, float) and math.isfinite(v) for v in value), 'Invalid '+name)
    return np.asarray(value, np.float64)


def covariance(value, size, name):
    require(type(value) is list and len(value) == size, 'Invalid '+name)
    array = np.stack([vector(row, size, name) for row in value])
    require(np.array_equal(array, array.T) and np.linalg.eigvalsh(array).min() > 0,
        'Nonpositive or asymmetric '+name)
    return array


def close(left, right, name):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),
        'Invalid '+name)
    error = float(np.max(np.abs(a-b)))
    require(error <= ATOL, 'Incorrect '+name)
    return error


def predict(mean, p, dt, acceleration_sigma=60.):
    require(type(dt) in (int, float) and math.isfinite(dt) and dt >= 0, 'Invalid prediction interval')
    f = np.eye(4); f[0, 2] = f[1, 3] = dt
    load = np.vstack((np.eye(2)*(dt*dt/2), np.eye(2)*dt))
    result = f@p@f.T + acceleration_sigma**2*(load@load.T)
    return f@mean, (result+result.T)*.5


def correct(mean, p, point, strong_r):
    """Independent Joseph form, Rweak exactly two times the strong covariance."""
    h = np.eye(4)[:2]
    r = 2.*strong_r
    k = p@h.T@np.linalg.inv(h@p@h.T+r)
    mean_after = mean+k@(point-h@mean)
    residual = np.eye(4)-k@h
    p_after = residual@p@residual.T+k@r@k.T
    return mean_after, (p_after+p_after.T)*.5


def full_disk_in_bounds(center, bounds, radius=45.):
    x, y = vector(center, 2, 'primary query center')
    require(type(bounds) is list and len(bounds) == 4 and all(type(v) is int for v in bounds)
        and 0 <= bounds[0] < bounds[2] and 0 <= bounds[1] < bounds[3], 'Invalid capture bounds')
    return bool(x-radius >= bounds[0] and y-radius >= bounds[1]
        and x+radius <= bounds[2]-1 and y+radius <= bounds[3]-1)


class Bindings:
    """Recheck exact metadata/NPZ/code inputs; never hash or open source media."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected):
        path = Path(path)
        require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink() and path.suffix in ('.json','.jsonl','.gz','.npz','.py','.so'),
            'Canonical metadata/array/code input required')
        require(digest_ok(expected), 'Caller-bound input SHA required')
        actual = sha(path)
        require(actual == expected and self.hashes.get(str(path),actual) == actual, 'Changed bound '+str(path))
        self.hashes[str(path)] = actual
        return path

    def read(self, path, expected):
        path = self.bind(path,expected)
        require(path.suffix == '.json' and path.stat().st_size <= 128*1024*1024, 'Bounded JSON metadata required')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'JSON changed while reading')
        value = loads(raw)
        require(type(value) is dict, 'Metadata object required')
        self.bind(path,expected)
        return value

    def unchanged(self):
        for path,digest in list(self.hashes.items()):
            self.bind(path,digest)


def scoped_path(root, relative):
    require(type(relative) is str and bool(relative), 'Relative artifact path required')
    part = Path(relative)
    require(not part.is_absolute() and '..' not in part.parts and '.' not in part.parts
        and str(part) == relative, 'Unsafe relative artifact path')
    path = Path(root)
    for name in part.parts:
        path /= name
        require(not path.is_symlink(), 'Symlink artifact refused')
    return path


def json_lines(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path,'rb') as stream:
        while True:
            line = stream.readline(64*1024*1024+1)
            if not line:
                return
            require(len(line) <= 64*1024*1024 and line.endswith(b'\n') and line.strip(), 'Malformed/oversized JSONL row')
            value = loads(line)
            require(type(value) is dict, 'JSONL object required')
            yield value


def frame_row(row, index):
    require(type(row.get('frame_index')) is int and row['frame_index'] == index
        and type(row.get('timestamp_ns')) is int and row['timestamp_ns'] == index*100000000
        and type(row.get('segment')) is int and row['segment'] >= 0, 'Incomplete/changed causal sequence')


def tagged_guard(before, after, before_sha, after_sha, name):
    require(digest_ok(before_sha) and digest_ok(after_sha), 'Missing '+name+' digest')
    require(normalized_sha(before) == before_sha and normalized_sha(after) == after_sha,
        'Unreproducible '+name+' digest')
    exact(before,after,name)
    require(before_sha == after_sha, 'Changed '+name+' digest')
    return decoded_normalized(before)


def learning_centers(records, previous_timestamp, timestamp, segment, config):
    if not (bool(config['learning_exclusion_radius_px']) or config['learning_protection_geometry']=='observed_shape') or previous_timestamp is None:
        return []
    dt = (timestamp-previous_timestamp)/1e9
    if not 0 < dt <= config['coast_seconds']:
        return []
    selected = [r for r in records if r['segment']==segment and r['measured'] and r['qualified_moving']]
    if config['learning_protection_geometry'] == 'observed_shape':
        return [dict(support_reference_xy=[[p[j]+dt*r['velocity_reference_xy_px_s'][j] for j in (0,1)]
            for p in r['learning_shape_reference_xy']]) for r in selected if r.get('learning_shape_reference_xy')]
    return [[r['reference_xy'][j]+dt*r['velocity_reference_xy_px_s'][j] for j in (0,1)] for r in selected]


class PrimaryAudit:
    def __init__(self, config):
        self.config = config
        self.previous = None
        self.previous_digest = None
        self.frames = 0
        self.max_error = 0.

    def step(self, row, auxiliary, clean, expected_digest):
        i = self.frames
        for r in (row,auxiliary,clean): frame_row(r,i)
        require(row['segment']==auxiliary['segment']==clean['segment'],'Primary segment differs')
        exact(row['records'],clean['tracks'],'original primary records')
        exact(row['metrics'],clean['tracking_metrics'],'original primary metrics')
        exact(row['original_digest_reference'],expected_digest,'original digest reference')
        state=tagged_guard(row['primary_state_before_aux'],row['primary_state_after_aux'],
            row['primary_state_sha256_before_aux'],row['primary_state_sha256_after_aux'],'primary state')
        require(row['primary_state_sha256_before_aux']==expected_digest['state'],'Historical private-state digest differs')
        output=tagged_guard(row['primary_output_normalized_before_aux'],row['primary_output_normalized_after_aux'],
            row['primary_output_sha256_before_aux'],row['primary_output_sha256_after_aux'],'primary output')
        exact(output,[row['records'],row['metrics']],'primary output payload')
        require(row['primary_output_sha256_before_aux']==expected_digest['output'],'Historical output digest differs')
        actual=row['actual_learning_normalized']
        require(normalized_sha(actual)==row['actual_learning_sha256']==expected_digest['learning'],
            'Historical actual learning digest differs')
        exact(decoded_normalized(actual),row['learning_centers'],'actual learning payload')
        learning=tagged_guard(row['next_learning_normalized_before_aux'],row['next_learning_normalized_after_aux'],
            row['learning_sha256_before_aux'],row['learning_sha256_after_aux'],'next learning')
        exact(learning,row['next_learning_centers_before_aux'],'next learning payload')
        exact(learning,row['next_learning_centers_after_aux'],'next learning after payload')
        exact(learning,learning_centers(row['records'],row['timestamp_ns'],row['timestamp_ns']+100000000,
            row['segment'],self.config),'strong-only next learning geometry')
        require(digest_ok(row['primary_prepare_state_sha256_before']) and
            row['primary_prepare_state_sha256_before']==row['primary_prepare_state_sha256_after'],
            'Prepare changed primary state')
        if self.previous is not None:
            require(row['primary_prepare_state_sha256_before']==self.previous_digest,'Primary state changed between frames')
            exact(row['learning_centers'],learning_centers(self.previous['previous_records'],
                self.previous['previous_timestamp_ns'],row['timestamp_ns'],row['segment'],self.config),
                'past-only actual learning geometry')
        else:
            exact(row['learning_centers'],[],'cold-start learning')
        exact(auxiliary['primary_records_after_strong'],row['records'],'detached primary records')
        expected_priors={}
        if self.previous is not None:
            for polarity,manager in self.previous['managers'].items():
                if manager['_segment_index']!=row['segment'] or manager['_last_timestamp_ns'] is None:
                    continue
                if (row['timestamp_ns']-manager['_last_timestamp_ns'])/1e9>manager['config']['maximum_timestamp_gap_s']:
                    continue
                for tid,track in manager['_tracks'].items():
                    key=f"{row['segment']}/{polarity}:{tid}"
                    expected_priors[key]=(polarity,tid,track)
        priors=auxiliary['primary_forecasts']
        require(len(priors)==len(expected_priors) and len({p['identity'] for p in priors})==len(priors),
            'Missing/duplicate primary prior gates')
        for p in priors:
            require(p['identity'] in expected_priors,'Nonprimary query owner')
            polarity,tid,track=expected_priors[p['identity']]
            mean,cov=predict(np.asarray(track['mean']),np.asarray(track['covariance']),
                (row['timestamp_ns']-track['state_timestamp_ns'])/1e9,self.config['acceleration_sigma_px_s2'])
            anchor=track['last_measurement_timestamp_ns']; age=(row['timestamp_ns']-anchor)/1e9
            qualified=f'{polarity}:{tid}' in self.previous['qualified']
            require(type(p['track_id']) is int and p['track_id']==tid and p['polarity']==polarity
                and p['frame_index']==i and p['timestamp_ns']==row['timestamp_ns'] and p['segment']==row['segment'],
                'Wrong primary prior provenance')
            for key,wanted in dict(prior_last_strong_timestamp_ns=anchor,strong_age_seconds=age,
                prior_qualified=qualified,prior_associated_update_count=track['associated_update_count'],
                prior_independent_confirmation_hits=track['independent_confirmation_hits'],
                prior_confirmation_timestamp_ns=track['confirmation_timestamp_ns'],weak_budget_available=True,
                query_eligible=qualified and track['confirmation_timestamp_ns'] is not None and 0<age<=self.config['coast_seconds']).items():
                exact(p[key],wanted,'strong-only forecast '+key)
            r=np.eye(2)*self.config['position_sigma_px']**2
            for key,wanted in (('predicted_mean',mean),('predicted_covariance',cov),('reference_xy',mean[:2]),
                ('strong_measurement_covariance',r),('innovation_covariance_2x2',cov[:2,:2]+r)):
                self.max_error=max(self.max_error,close(p[key],wanted,'primary forecast '+key))
        post=auxiliary['primary_post_strong']
        expected_post={f"{row['segment']}/{polarity}:{tid}":(polarity,tid,track)
            for polarity,manager in state['managers'].items() for tid,track in manager['_tracks'].items()}
        require(len(post)==len(expected_post) and len({p['identity'] for p in post})==len(post),'Incomplete poststrong snapshot')
        for p in post:
            require(p['identity'] in expected_post,'Wrong poststrong snapshot identity')
            polarity,tid,track=expected_post[p['identity']]
            for key,value in p.items():
                if key in ('identity','polarity','track_id'):continue
                if key=='predicted_mean':wanted=track['mean']
                elif key=='predicted_covariance':wanted=track['covariance']
                else:
                    require(key in track,'Unexpected readable poststrong field')
                    wanted=track[key]
                exact(value,wanted,'poststrong '+key)
            require(p['polarity']==polarity and p['track_id']==tid,'Poststrong owner mismatch')
        self.previous=state;self.previous_digest=row['primary_state_sha256_after_aux'];self.frames+=1


def in_frame(point,shape):
    return bool(0<=point[0]<shape[1] and 0<=point[1]<shape[0])


def project(matrix,point):
    value=matrix@np.array([*point,1.],float)
    require(np.isfinite(value).all() and value[2]!=0,'Invalid source projection')
    return value[:2]/value[2]


def overlap(point,proposals,radius):
    return any(math.hypot(point[0]-p['x'],point[1]-p['y'])<=radius or
        list(point) in p.get('shape',{}).get('support_reference_xy',[]) for p in proposals)


class AuxiliaryAudit:
    def __init__(self,config):
        self.config=config;self.previous=None;self.max_error=0.;self.counts={};self.statuses={}

    def step(self,row,clean,capture_check):
        before,after=row['aux_state_before'],row['aux_state_after']
        index,ts,segment=row['frame_index'],row['timestamp_ns'],row['segment']
        for state in (before,after):
            require(state['schema']=='seaqr.weak-auxiliary-state.v1' and state['fps']==10
                and state['poisoned'] is False and state['pending'] is None and state['maximum_timestamp_gap_s']==1.,
                'Incomplete/poisoned auxiliary state')
            exact(state['config'],{k:self.config[k] for k in state['config']},'auxiliary configuration')
            exact(state['used_strong_gaps'],{k:s['strong_anchor_timestamp_ns'] for k,s in state['states'].items()},'owned used-gap budget')
        if self.previous is not None:exact(before,self.previous,'auxiliary temporal state')
        else:
            require(not before['states'] and all(before[k] is None for k in
                ('last_frame_index','last_timestamp_ns','last_segment')),'Auxiliary cold start differs')
        require(after['last_frame_index']==index and after['last_timestamp_ns']==ts and after['last_segment']==segment,
            'Auxiliary state timestamp differs')
        priors={p['identity']:p for p in row['primary_forecasts']}
        records={f"{segment}/{r['track_id']}":r for r in row['primary_records_after_strong']}
        require(len(records)==len(row['primary_records_after_strong']),'Duplicate primary identity')
        reset=('segment_reset' if before['last_segment'] is not None and before['last_segment']!=segment else
            'timestamp_gap_reset' if before['last_timestamp_ns'] is not None and (ts-before['last_timestamp_ns'])/1e9>1 else None)
        states=deepcopy(before['states'])
        eligible_states={} if reset else states
        queries=[p for p in row['primary_forecasts'] if p['query_eligible'] and
            (p['identity'] not in eligible_states or eligible_states[p['identity']]['strong_anchor_timestamp_ns']!=p['prior_last_strong_timestamp_ns'])]
        exact(row['auxiliary_queries'],queries,'primary-only budget-filtered query set')
        drops=[]
        if reset:
            drops=[dict(identity=k,reason=reset) for k in states];states={}
        for identity in list(states):
            if identity not in records:
                drops.append(dict(identity=identity,reason='primary_deletion'));del states[identity]
        metrics=row['auxiliary_metrics'];decisions=metrics['decisions'];outputs=row['auxiliary_records']
        require([d['identity'] for d in decisions]==list(records),'Missing/reordered auxiliary decisions')
        output_map={r['identity']:r for r in outputs}
        require(len(output_map)==len(outputs),'Duplicate auxiliary output')
        expected_outputs=[];call_index=0;weak_count=0
        inverse=np.linalg.inv(np.asarray(clean['source_to_reference'],float));shape=clean['coverage']['full_shape_hw']
        for note in decisions:
            identity=note['identity'];record=records[identity];prior=priors.get(identity)
            reason=('strong_measurement_priority' if record['measured'] else
                'no_prior_same_segment_track' if prior is None else
                'strong_age_expired' if prior['strong_age_seconds']>self.config['coast_seconds'] else
                'not_prior_strong_qualified' if not prior['query_eligible'] or not record['qualified_moving'] else None)
            if reason:
                if identity in states:
                    drops.append(dict(identity=identity,reason=reason));del states[identity]
                require(note['status']==reason and note['applied'] is False and note['observations'] is None,
                    'Weak bypassed strong lifecycle')
                continue
            if identity in states and states[identity]['strong_anchor_timestamp_ns']!=prior['prior_last_strong_timestamp_ns']:
                drops.append(dict(identity=identity,reason='primary_strong_anchor_changed'));del states[identity]
            old=states.get(identity);is_weak=old is None
            if old is not None:
                mean_before=vector(old['mean'],4,'auxiliary mean');cov_before=covariance(old['covariance'],4,'auxiliary covariance')
                mean,p=predict(mean_before,cov_before,(ts-old['state_timestamp_ns'])/1e9,self.config['acceleration_sigma_px_s2'])
                state=deepcopy(old);state.update(mean=mean.tolist(),covariance=p.tolist(),state_timestamp_ns=ts)
                # Existing out-of-bounds predictions still retain their consumed budget.
                states[identity]=state
                wanted='auxiliary_prediction_from_weak'
                require(note['observations'] is None and note['applied'] is False,'Propagated weak point became a fresh measurement')
            else:
                if not in_frame(prior['reference_xy'],shape):
                    require(note['status']=='query_prior_out_of_bounds' and not note['applied'],'Out-of-bounds query used')
                    continue
                require(call_index<len(row['capture_calls']),'Missing required capture attempt')
                call=row['capture_calls'][call_index];call_index+=1
                observation=capture_check(call,prior,row['primary_forecasts'],shape)
                exact(note['observations'],observation,'captured weak observations')
                if observation is None:wanted='missing_capture'
                elif not observation['coverage_known']:wanted='capture_coverage_unknown'
                elif observation['original_threshold_peak_count']:wanted='original_threshold_peak_in_gate'
                elif observation['observed_peak_count']!=1:wanted='no_unique_weak_peak'
                elif observation['observed_peaks'][0]['competing_prior_identity_gates']:wanted='competing_prior_identity_gate'
                elif overlap(observation['observed_peaks'][0]['reference_xy'],clean['candidates'],max(2.,self.config['tracking_peak_nms_radius_px'])):
                    wanted='overlaps_current_strong_evidence'
                else:wanted='weak_auxiliary_correction'
                exact(note['coverage_known'],None if observation is None else observation['coverage_known'],'coverage decision')
                if wanted!='weak_auxiliary_correction':
                    require(note['status']==wanted and note['applied'] is False,'Changed frozen weak selection rule')
                    continue
                peak=observation['observed_peaks'][0]
                exact(note['measurement_reference_xy'],peak['reference_xy'],'weak observation position')
                exact(note['score'],peak['score'],'weak observation score')
                mean_before=vector(prior['predicted_mean'],4,'primary seed mean')
                cov_before=covariance(prior['predicted_covariance'],4,'primary seed covariance')
                mean,p=correct(mean_before,cov_before,np.asarray(peak['reference_xy']),
                    covariance(prior['strong_measurement_covariance'],2,'strong R'))
                state=dict(mean=mean.tolist(),covariance=p.tolist(),state_timestamp_ns=ts,
                    strong_anchor_timestamp_ns=prior['prior_last_strong_timestamp_ns'],origin_frame_index=index,
                    origin_timestamp_ns=ts,origin_measurement_reference_xy=peak['reference_xy'],
                    origin_measurement_source_xy=project(inverse,peak['reference_xy']).tolist(),primary_prior_at_weak=prior)
            source=project(inverse,mean[:2])
            if (is_weak and not in_frame(state['origin_measurement_source_xy'],shape)) or not in_frame(mean[:2],shape) or not in_frame(source,shape):
                require(note['status']=='auxiliary_coordinate_out_of_bounds' and note['applied'] is False,
                    'Out-of-bounds auxiliary presented')
                continue
            require(note['status']==wanted and note['applied'] is is_weak,'Auxiliary evidence type differs')
            require(identity in output_map,'Missing supported auxiliary record')
            out=output_map[identity];expected_outputs.append(identity);states[identity]=state;weak_count+=int(is_weak)
            for key,wanted_value in dict(record_type='auxiliary_gap_support',primary_track_id=record['track_id'],
                segment=segment,frame_index=index,timestamp_ns=ts,current_weak_observation=is_weak,prediction_from_weak=not is_weak,
                evidence_type='current_weak_observation' if is_weak else 'prediction_from_weak',ordinary_measurement=False,
                qualified_detection=False,physical_identity_verified=False,
                current_weak_measurement_reference_xy=state['origin_measurement_reference_xy'] if is_weak else None).items():
                exact(out[key],wanted_value,'auxiliary record '+key)
            for key,wanted_value in (('mean_before',mean_before),('covariance_before',cov_before),('mean_after',mean),
                ('covariance_after',p),('reference_xy',mean[:2]),('velocity_reference_xy_px_s',mean[2:]),('source_xy',source)):
                self.max_error=max(self.max_error,close(out[key],wanted_value,'auxiliary '+key))
            for key in ('strong_anchor_timestamp_ns','origin_frame_index','origin_timestamp_ns',
                'origin_measurement_reference_xy','origin_measurement_source_xy','primary_prior_at_weak'):
                exact(out[key],state[key],'auxiliary origin '+key)
        require(call_index==len(row['capture_calls']),'Unrequested capture query')
        exact([r['identity'] for r in outputs],expected_outputs,'auxiliary output inventory')
        exact(metrics['dropped_auxiliary'],drops,'auxiliary deletions')
        require(set(after['states'])==set(states),'Missing/additional auxiliary state')
        for identity,state in states.items():
            actual=after['states'][identity]
            require(set(actual)==set(state),'Auxiliary state fields differ')
            for key,value in state.items():
                if key in ('mean','covariance'):
                    self.max_error=max(self.max_error,close(actual[key],value,'owned auxiliary '+key))
                else:exact(actual[key],value,'owned auxiliary '+key)
        wanted_counts=dict(prior_count=len(priors),prior_strong_eligible_count=sum(p['query_eligible'] for p in priors.values()),
            prepared_query_count=len(queries),actual_provider_calls=call_index,primary_live_count=len(records),
            auxiliary_state_count=len(states),auxiliary_record_count=len(outputs),current_weak_observation_count=weak_count,
            prediction_from_weak_count=len(outputs)-weak_count)
        for key,value in wanted_counts.items():
            exact(metrics[key],value,'auxiliary count '+key);self.counts[key]=self.counts.get(key,0)+value
        require(metrics['schema']=='seaqr.weak-auxiliary.v1' and metrics['frame_index']==index,'Wrong auxiliary metric schema')
        exact(metrics['reset_reason'],reset,'auxiliary reset')
        for key in ('primary_feedback','ordinary_measurement_credit','detector_learning_feedback','physical_identity_established'):
            require(metrics[key] is False,'Auxiliary authority leak')
        for d in decisions:self.statuses[d['status']]=self.statuses.get(d['status'],0)+1
        self.previous=deepcopy(after)


def covers(rect,prior,shape):
    exact(rect['shape_hw'],shape,'capture native dimensions')
    if not full_disk_in_bounds(prior['reference_xy'],rect['tile_bounds_exclusive_xyxy']):
        return False
    x,y=prior['reference_xy'];h,w=shape;b=rect['capture_bounds_exclusive_xyxy']
    full_disk_in_bounds(prior['reference_xy'],b)  # validates the rectangle, independently of the halo test
    return (b[0]<=max(0,math.ceil(x-45)-2) and b[1]<=max(0,math.ceil(y-45)-2)
        and b[2]>min(w-1,math.floor(x+45)+2) and b[3]>min(h-1,math.floor(y+45)+2))


class CaptureAudit:
    """Independent provenance/geometry; replay the frozen pure pixel enumerator."""
    def __init__(self,bindings,root,pins,enumerator):
        self.bindings,self.root,self.pins,self.enumerator=bindings,root,pins,enumerator
        self.counts={};self.files=set()

    def begin(self,old_trace,shape,enabled):
        self.old=old_trace;self.shape=shape;self.enabled=enabled;self.inventory={};self.cache={}
        require(old_trace['capture_scheduled'] is enabled,'Old capture schedule differs')
        for item in old_trace['capture_files']:
            desc={k:item[k] for k in ('path','sha256','metadata_path','metadata_sha256')}
            for key,digest_key in (('path','sha256'),('metadata_path','metadata_sha256')):
                name=desc[key]
                require(type(name) is str and name.startswith('captures/') and
                    self.pins.get('shadow/'+name)==desc[digest_key],'Capture lacks original audit binding')
            key=desc['path']
            if key in self.inventory:
                exact(self.inventory[key][0],desc,'duplicate capture descriptor');continue
            meta=self.bindings.read(scoped_path(self.root,'shadow/'+desc['metadata_path']),desc['metadata_sha256'])
            require(type(meta.get('frame')) is int and meta['frame']==old_trace['frame_index']
                and type(meta.get('segment')) is int and meta['segment']==old_trace['segment'],
                'Capture time/segment mismatch')
            for flag in ('prelearning','baseline_learning_unchanged','read_only_capture','production_selection_unchanged'):
                require(meta.get(flag) is True,'Capture provenance missing '+flag)
            self.inventory[key]=(desc,meta)

    def __call__(self,call,prior,priors,shape):
        exact(call['forecast'],prior,'detached capture query')
        require(call['query_identity']==prior['identity'],'Capture query owner differs')
        self.counts['required_queries']=self.counts.get('required_queries',0)+1
        chosen=next(((d,m) for _,(d,m) in sorted(self.inventory.items()) if covers(m['rectangle'],prior,shape)),None) if self.enabled else None
        status=('outside_frozen_window' if not self.enabled else
            'no_geometrically_complete_cached_capture' if chosen is None else 'geometrically_complete_capture_supplied')
        require(call['status']==status,'Changed deterministic capture selection')
        self.counts[status]=self.counts.get(status,0)+1
        if chosen is None:
            require(call['descriptor'] is None,'Missing capture fabricated');return None
        desc,meta=chosen;exact(call['descriptor'],desc,'selected capture binding')
        key=desc['path'];self.files.add(key)
        if key not in self.cache:
            path=self.bindings.bind(scoped_path(self.root,'shadow/'+key),desc['sha256'])
            with zipfile.ZipFile(path) as archive:
                entries=archive.infolist()
                require(len({e.filename for e in entries})==len(entries)
                    and {'values.npy','flags.npy'}<={e.filename for e in entries}
                    and sum(e.file_size for e in entries)<=128*1024*1024,'Unsafe capture archive')
            with np.load(path,allow_pickle=False) as data:
                values,flags=data['values'].copy(),data['flags'].copy()
            self.cache[key]=values,flags
        values,flags=self.cache[key]
        observed=self.enumerator(values,flags,deepcopy(meta),deepcopy(prior),deepcopy(priors))
        key='full_usable_coverage' if observed['coverage_known'] else 'censored_or_incomplete_coverage'
        self.counts[key]=self.counts.get(key,0)+1
        return observed


def load_contract(directory,freeze,freeze_sha256):
    directory,freeze=Path(directory),Path(freeze)
    require(directory.is_absolute() and directory.resolve()==directory and directory.is_dir()
        and not directory.is_symlink(),'Canonical replay output directory required')
    bound=Bindings();frozen=bound.read(freeze,freeze_sha256)
    require(frozen.get('schema')==FREEZE_SCHEMA and frozen.get('pre_run') is True
        and frozen.get('original_root')==str(ORIGINAL_ROOT) and frozen.get('original_freeze_sha256')==ORIGINAL_FREEZE
        and frozen.get('original_plan_sha256')==ORIGINAL_PLAN,'Wrong frozen auxiliary protocol')
    require(set(frozen['original_audits_sha256'])==set(SOURCES),'Incomplete original cohort pins')
    bundle=frozen['files_sha256']
    require(type(bundle) is dict and all(Path(k).name==k and digest_ok(v) for k,v in bundle.items()),'Invalid source bundle')
    for name,digest in bundle.items():bound.bind(scoped_path(freeze.parent,name),digest)
    for name in ('audit_weak_auxiliary_replay_v1.py','test_audit_weak_auxiliary_replay_v1.py',
        'run_weak_auxiliary_replay_v1.py','weak_continuation_auxiliary_v1.py','weak_continuation_information_v1.py'):
        require(name in bundle,'Missing frozen source '+name)
    require(sha(__file__)==bundle['audit_weak_auxiliary_replay_v1.py'] and
        bundle['weak_continuation_information_v1.py']==ENUM_SHA,'Auditor/enumerator source differs')
    for key in ('geometry','batch_library'):
        require(set(frozen[key])=={'path','sha256'},'Missing native dependency binding')
        bound.bind(frozen[key]['path'],frozen[key]['sha256'])
    receipt_path=directory/'receipt.json';receipt=bound.read(receipt_path,sha(receipt_path))
    clip=receipt.get('clip');require(clip in SOURCES,'Wrong development clip')
    count,source=SOURCES[clip]
    for key,value in dict(schema=RUN_SCHEMA,passed=True,error=None,clip=clip,expected_frames=count,processed_frames=count,
        source_sha256=source,freeze_sha256=freeze_sha256,original_freeze_sha256=ORIGINAL_FREEZE,
        original_plan_sha256=ORIGINAL_PLAN,original_directory=str(ORIGINAL_ROOT/clip),
        original_audit_sha256=frozen['original_audits_sha256'][clip],production_changed=False,weak_learning_enabled=False,
        media_accessed=False,detector_rerun=False,old_shadow_forecasts_or_accepted_weak_events_used=False).items():
        exact(receipt.get(key),value,'run receipt '+key)
    require(set(receipt['artifacts_sha256'])=={'clean.jsonl.gz','primary.jsonl.gz','auxiliary.jsonl.gz'},'Unexpected output artifact inventory')
    for name,digest in receipt['artifacts_sha256'].items():bound.bind(scoped_path(directory,name),digest)
    for key in ('scalar_fallbacks','optimized_fallbacks','innovation_fallbacks'):
        require(type(receipt['native_counters'][key]) is int and receipt['native_counters'][key]==0,'Native fallback recorded')
    oldfreeze=bound.read(ORIGINAL_ROOT/'freeze.json',ORIGINAL_FREEZE)
    oldplan=bound.read(ORIGINAL_ROOT/'plan.json',ORIGINAL_PLAN)
    require(oldfreeze['schema']=='seaqr.weak-continuation-shadow.freeze.v1' and oldfreeze['pre_run'] is True
        and oldfreeze['plan_sha256']==ORIGINAL_PLAN and oldplan['schema']=='seaqr.weak-continuation-shadow.plan.v1',
        'Original freeze/plan differs')
    audit=bound.read(ORIGINAL_ROOT/clip/'independent_audit.json',frozen['original_audits_sha256'][clip])
    for key,value in dict(schema='seaqr.weak-continuation-shadow.audit.v1',passed=True,clip=clip,frames=count,
        source_sha256=source,freeze_sha256=ORIGINAL_FREEZE,plan_sha256=ORIGINAL_PLAN,
        baseline_journal_non_timing_exact=True,baseline_output_state_learning_digests_exact=True,
        native_state_guards_unchanged=True,production_changed=False,weak_learning_enabled=False).items():
        exact(audit.get(key),value,'original audit '+key)
    pins=audit['files_sha256'];oldroot=ORIGINAL_ROOT/clip
    for name,digest in pins.items():
        if not name.startswith('shadow/captures/'):bound.bind(scoped_path(oldroot,name),digest)
    reference=bound.read(oldroot/'clean/launch.json',pins['clean/launch.json'])
    require(reference['source_sha256']==source and reference['expected_frames']==count and reference['fps']==10
        and reference['max_frames'] is None and reference['annotations_supplied_to_detector'] is False
        and reference['config_sha256']=='7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f'
        and reference['motion_config_sha256']=='fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1',
        'Historical launch differs')
    oldreceipt=bound.read(oldroot/'clean.shadow.json',pins['clean.shadow.json'])
    require(oldreceipt['passed'] is True and oldreceipt['error'] is None and oldreceipt['clip']==clip
        and len(oldreceipt['baseline_digests'])==count,'Missing complete original private-state oracle')
    for i,row in enumerate(oldreceipt['baseline_digests']):
        require(row['frame']==i and all(digest_ok(row[k]) for k in ('state','output','learning')),'Invalid original digest row')
    exact(receipt['runtime_hashes']['package'],reference['package_sha256'],'frozen runtime package identity')
    for item in receipt['runtime_hashes']['adapters'].values():bound.bind(item['path'],item['sha256'])
    expected_bundle={str(freeze):freeze_sha256,**{str(freeze.parent/name):digest for name,digest in bundle.items()}}
    exact(receipt['bundle_files_sha256'],expected_bundle,'runner verified bundle')
    expected_root={str(ORIGINAL_ROOT/'freeze.json'):ORIGINAL_FREEZE,str(ORIGINAL_ROOT/'plan.json'):ORIGINAL_PLAN,
        str(oldroot/'independent_audit.json'):frozen['original_audits_sha256'][clip]}
    exact(receipt['original_root_files_sha256'],expected_root,'runner original root bindings')
    for path,digest in receipt['input_files_sha256'].items():
        relative=str(Path(path).relative_to(oldroot))
        require(pins.get(relative)==digest,'Input is not in passed original audit')
        bound.bind(path,digest)
    enum_path=freeze.parent/'weak_continuation_information_v1.py'
    spec=importlib.util.spec_from_file_location('aux_audit_frozen_enumerator',enum_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return bound,frozen,receipt,reference,oldreceipt,pins,module.enumerate_peaks


def audit_run(directory,freeze,freeze_sha256):
    bound,frozen,receipt,reference,oldreceipt,pins,enumerator=load_contract(directory,freeze,freeze_sha256)
    clip=receipt['clip'];count,source=SOURCES[clip];directory=Path(directory);oldroot=ORIGINAL_ROOT/clip
    primary=PrimaryAudit(reference['configuration']);auxiliary=AuxiliaryAudit(reference['configuration'])
    capture=CaptureAudit(bound,oldroot,pins,enumerator)
    # Stream complete inputs, and independently hash every decompressed output byte.
    paths=[directory/name for name in ('clean.jsonl.gz','primary.jsonl.gz','auxiliary.jsonl.gz')]
    paths += [oldroot/'clean/frames.jsonl',oldroot/'shadow/shadow_trace.jsonl']
    from contextlib import ExitStack
    raw_hashes=[hashlib.sha256() for _ in range(3)];raw_sizes=[0,0,0];frames=0
    with ExitStack() as stack:
        streams=[stack.enter_context((gzip.open if path.suffix=='.gz' else open)(path,'rb')) for path in paths]
        for i,lines in enumerate(itertools.zip_longest(*streams)):
            require(i<count and all(type(line) is bytes and line.strip() and len(line)<=64*1024*1024
                and line.endswith(b'\n') for line in lines),'Missing/excessive/oversized causal row')
            require(lines[0]==lines[3],'Clean source line was rewritten or filtered')
            for j in range(3):raw_hashes[j].update(lines[j]);raw_sizes[j]+=len(lines[j])
            clean,p,a,_,old=map(loads,lines)
            frame_row(old,i);require(old['segment']==clean['segment'],'Old capture coordinate segment differs')
            exact([{k:v for k,v in p.items() if k!='source_xy'} for p in clean['candidates']],
                [{k:v for k,v in p.items() if k!='source_xy'} for p in old['strong_proposals']],'original strong proposal stream')
            enabled=any(lo<=i<=hi for lo,hi in WINDOWS[clip])
            require(a['capture_scheduled'] is enabled,'Auxiliary capture schedule differs')
            capture.begin(old,clean['coverage']['full_shape_hw'],enabled)
            primary.step(p,a,clean,oldreceipt['baseline_digests'][i])
            auxiliary.step(a,clean,capture)
            frames+=1
    require(frames==count,'Incomplete clip')
    for j,path in enumerate(paths[:3]):
        exact(receipt['artifact_uncompressed'][path.name],dict(sha256=raw_hashes[j].hexdigest(),bytes=raw_sizes[j],frames=count),
            'uncompressed artifact '+path.name)
    require(raw_hashes[0].hexdigest()==pins['clean/frames.jsonl'],'Clean raw bytes differ from original hash')
    bound.unchanged()
    return dict(schema=SCHEMA,passed=True,clip=clip,frames=count,freeze_sha256=freeze_sha256,source_sha256=source,
        input_files_sha256=bound.hashes,primary_records_metrics_exact=True,historical_private_state_output_learning_digests_exact=True,
        current_primary_state_output_learning_isolation_exact=True,all_primary_query_priors_verified=True,
        auxiliary_numerical_and_lifecycle_checks_passed=True,auxiliary_counts=auxiliary.counts,
        auxiliary_status_counts=auxiliary.statuses,capture_counts=capture.counts,unique_npz_files_used=len(capture.files),
        max_primary_forecast_absolute_error=primary.max_error,max_auxiliary_absolute_error=auxiliary.max_error,
        numerical_formula_tolerance=ATOL,primary_parity_tolerance=0,source_media_accessed=False,producer_imported=False,
        capture_arrays_reenumerated_with_pinned_original_function=True,production_changed=False,
        limitations=['Historical captures were selected by the former shadow. Missing/censored coverage is not a negative observation or blind accuracy evidence.',
            'Auxiliary equations and lifecycle are independently reconstructed; pixel enumeration reuses the pinned original pure enumerator, not an independent algorithm.',
            'State/output/learning isolation does not establish physical identity, target recall, false-positive rate, or deployment readiness.'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','freeze','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(),'Fresh audit output required')
    result=audit_run(args.directory,args.freeze,freeze_sha256=args.freeze_sha256)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
